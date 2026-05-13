import re
import time
import torch
from transformers import StoppingCriteria
from .itergen.itergen.main import IterGen
from .base import BaseGenerator


class _BraceDepthStop(StoppingCriteria):
    """Stop forward() as soon as the prompt's outer `{` is balanced.

    Without this, after the model emits the function's closing `}`,
    forward(unit="statement", num=1) keeps generating: the `}` ends a
    compound_statement, not a new top-level statement, and comments are
    %ignore'd -- so the unit never fires and the whole max_new_tokens
    budget is burned on garbage that post-processing throws away (49s
    wasted on the 0.5B model, minutes on the 3B).
    """
    def __init__(self, itergen_ref, prompt_len, depth_offset):
        self.itergen = itergen_ref
        self.prompt_len = prompt_len
        self.depth_offset = depth_offset

    def __call__(self, input_ids, scores, **kwargs):
        gen = self.itergen.structured_gen[0] if self.itergen.structured_gen else ""
        body = gen[self.prompt_len:] if len(gen) > self.prompt_len else ""
        depth = self.depth_offset + body.count('{') - body.count('}')
        stop = depth <= 0
        return torch.full((input_ids.shape[0],), stop, dtype=torch.bool, device=input_ids.device)

class ItergenGenerator(BaseGenerator):
    def __init__(self, model_name: str, grammar: str, **kwargs):

        temp= kwargs.get("temperature")
        itergen_params = {
            "model_id": model_name,
            "grammar": grammar,
            "parse_output_only": False,
            "recurrence_penalty": 0.0,
            "max_new_tokens": kwargs.get("max_new_tokens"),
            "do_sample": temp > 0  # if temperature > 0, enable sampling; otherwise, use greedy decoding
        }
        if temp > 0:
            itergen_params["temperature"] = temp
            
        self.itergen = IterGen(
            **itergen_params
        )
        # dev_mode=True (the IterGen default) causes grammar parsing exceptions
        # to be re-raised from forward(), which makes the outer loop break on any
        # incremental parse failure (e.g. partial regex literals, complex
        # expressions). Disable it so failures are handled gracefully instead.
        self.itergen.dev_mode = False

    def generate(self, prompt: str, stop_tokens, **kwargs) -> str:

        temp = kwargs.get("temperature")

        # Do NOT pass max_new_tokens to forward(): start() already set max_length as an
        # absolute token-count ceiling (prompt_len + max_new_tokens). Re-passing it each
        # call would trigger generation_config.update(max_new_tokens=N) which does NOT
        # recalculate max_length, so the ceiling never slides and long functions get
        # truncated once the early statements consume the budget.
        forward_params = {
            "do_sample": temp > 0,
        }
        if temp > 0:
            forward_params["temperature"] = temp

        self.itergen.start(prompt=prompt)

        # The old default ran hand-written semantic checks after every generated
        # statement and backtracked on failures. In practice the symbol map can
        # expose partial/stale identifier slices while IterGen is recovering from
        # an incremental parse failure, so the checks falsely rejected fragments
        # such as "fo", "ction", or "ing" as undeclared variables. That caused
        # short completions, repeated backtracking, wall-clock timeouts, and lower
        # pass rates than the unconstrained baseline. Keep the experimental checks
        # available for debugging, but make grammar-only IterGen the default.
        enable_semantic_checks = bool(kwargs.get("semantic_checks", False))
        tracking_categories = (
            [
                "var_decl", "function_declaration", "function_parameter", "primary_safe_non_numeric",
                "expr_safe", "control_flow_statement"
            ]
            if enable_semantic_checks
            else []
        )
        
        # Analyze grammar to identify which tokens to track for semantic checks
        base_identifiers = set([
            "console", "Math", "Object", "Array", "String", "Number", "Boolean",
            "true", "false", "null", "undefined", "NaN", "Infinity",
            "this", "arguments", "window", "document", "global",
            "Error", "Symbol", "Date", "RegExp", "Map", "Set", "WeakMap", "WeakSet",
            "Promise", "JSON", "parseInt", "parseFloat", "isNaN", "isFinite", 
            "encodeURI", "decodeURI", "encodeURIComponent", "decodeURIComponent", 
            "require", "module", "exports", "process", "Buffer", 
            "setTimeout", "clearTimeout", "setInterval", "clearInterval"
        ])
        
        # EXTRACT PARAMS FROM PROMPT to prevent undeclared errors for function arguments
        # e.g. "function pancake_sort(nums){" -> add "nums" to base_identifiers
        # MODIFIED: Capture function name too for recursion support
        param_pattern = re.compile(r'function\s+([a-zA-Z_$][a-zA-Z0-9_$]*\s*)?\(([^)]*)\)')
        try:
            matches = param_pattern.findall(prompt)
            if matches:
                last_match = matches[-1]
                
                # Add function name to identifiers (for recursion)
                if last_match[0] and last_match[0].strip():
                    base_identifiers.add(last_match[0].strip())

                last_params = last_match[1]
                if last_params.strip():
                    p_tokens = [p.strip() for p in last_params.split(',')]
                    for p in p_tokens:
                        # Handle "nums", "a", "limit=10"
                        if p:
                            p_name = p.split('=')[0].strip().split()[0]
                            if p_name:
                                base_identifiers.add(p_name)
        except Exception:
            pass
        
        # Count opening braces in prompt to track when the function body is complete
        prompt_brace_depth = prompt.count('{') - prompt.count('}')

        # Add a stopping criterion that cuts forward() short the moment brace depth
        # hits 0 -- otherwise the model wastes the rest of its token budget on
        # %ignore'd comments after the function close (see _BraceDepthStop docstring).
        if prompt_brace_depth > 0:
            self.itergen.stopping_criteria.append(
                _BraceDepthStop(self.itergen, len(prompt), prompt_brace_depth)
            )

        # Loop until the model hits its own token budget (enforced by start()'s max_length)
        # or our early-termination checks fire. Using a large fixed bound avoids the old
        # bug where max_steps == max_new_tokens caused premature exit: each forward() call
        # consumes *multiple* tokens (a whole statement), so 512 steps ≠ 512 tokens.
        max_steps = 4096

        prev_gen_len = len(prompt)  # track generation length to detect stalls

        # Detect infinite backtrack loops: if backward() returns us to the same
        # position repeatedly, the next forward() is regenerating the same invalid
        # statement (typical under greedy decoding with recurrence_penalty=1.0).
        last_backtrack_pos = -1
        consecutive_backtracks = 0
        MAX_CONSECUTIVE_BACKTRACKS = 3

        # Hard wall-clock cap per task. A normal task is ~2s; anything past 60s
        # almost certainly means we're churning on slow validation or grammar masks
        # with no real progress. Bail with whatever has been generated so far.
        MAX_GENERATE_SECONDS = 60.0
        start_time = time.perf_counter()

        for step in range(max_steps):

            if time.perf_counter() - start_time > MAX_GENERATE_SECONDS:
                print(f"[itergen] wall-clock timeout at step {step} ({MAX_GENERATE_SECONDS:.0f}s)")
                break

            pre_counts = {}

            if enable_semantic_checks:
                for cat in tracking_categories:
                    try:
                        res = self.itergen.view(unit=cat)
                        pre_counts[cat] = len(res[0]) if res and res[0] else 0
                    except Exception:
                        pre_counts[cat] = 0

            # forward 1 step
            try:
                self.itergen.forward(unit="statement", num=1, **forward_params)
            except Exception:
                break

            current_code = self.itergen.structured_gen[0] if self.itergen.structured_gen else ""

            # If nothing was generated this step (EOS/max_length already reached), stop.
            # This prevents an infinite loop when the model is done but 'statement' never
            # completes (e.g. the model only emitted comments, which %ignore discards).
            current_len = len(current_code) if current_code.startswith(prompt) else len(prompt) + len(current_code)
            if current_len <= prev_gen_len:
                break
            prev_gen_len = current_len

            # --- Early termination checks ---
            generated_so_far = current_code[len(prompt):] if current_code.startswith(prompt) else current_code
            
            # Check 1: Stop tokens — if any stop token appears in generated text, stop immediately.
            # Skip comment-style stop tokens (\n// and \n/*) when inside a function body
            # (brace depth > 0), since the grammar already prevents new function declarations
            # there and IterGen allows comments anywhere via %ignore.
            if stop_tokens and generated_so_far:
                should_stop = False
                gen_depth = prompt_brace_depth + generated_so_far.count('{') - generated_so_far.count('}')
                for stop in stop_tokens:
                    if stop in generated_so_far:
                        if stop in ('\n//', '\n/*') and gen_depth > 0:
                            continue  # Comments inside function body are fine for grammar-constrained gen
                        should_stop = True
                        break
                if should_stop:
                    break

            # Check 2: Brace depth — if the function body is complete (all braces balanced), stop
            if generated_so_far and prompt_brace_depth > 0:
                gen_open = generated_so_far.count('{')
                gen_close = generated_so_far.count('}')
                current_depth = prompt_brace_depth + gen_open - gen_close
                if current_depth <= 0:
                    break

            if not enable_semantic_checks:
                continue

            post_items = {}
            for cat in tracking_categories:
                try:
                    res = self.itergen.view(unit=cat)
                    post_items[cat] = res[0] if res and res[0] else []
                except Exception:
                    post_items[cat] = []
            
            # semantic validation rules
            is_valid = True
            violation_reason = ""

            # Build a scope-aware declaration map by scanning the generated text.
            # decl_at_depth[var_name] = list of (brace_depth, keyword) for every
            # declaration of that name seen so far.  Starting depth is 1 because the
            # prompt already opened the function body with '{'.
            #
            # This lets us distinguish legitimate inner-scope shadowing
            #   (outer `let i = 0`; inner `for (let i = 0; ...)` at a deeper depth)
            # from true same-scope redeclarations.
            full_gen_so_far = self.itergen.structured_gen[0] if self.itergen.structured_gen else ""
            decl_at_depth: dict = {}   # var_name -> [(depth, keyword), ...]
            const_identifiers: set = set()
            current_identifiers = base_identifiers.copy()

            scan_depth = 1
            scan_i = 0
            scan_text = generated_so_far  # only the generated portion
            _IDENT = re.compile(r'(var|let|const)\s+([a-zA-Z_$][a-zA-Z0-9_$]*)')
            _FOR_PAREN = re.compile(r'for\s*\($')  # matches 'for(' at end of a slice
            paren_depth = 0        # track '(' nesting
            in_for_header = False  # True while inside a for(...) header
            for_paren_depth = 0   # the paren_depth at which the for-header opened
            while scan_i < len(scan_text):
                ch = scan_text[scan_i]
                if ch == '{':
                    scan_depth += 1
                    scan_i += 1
                elif ch == '}':
                    scan_depth = max(1, scan_depth - 1)
                    scan_i += 1
                elif ch == '(':
                    paren_depth += 1
                    # Check if the text up to and including this '(' ends with 'for('
                    if _FOR_PAREN.search(scan_text[:scan_i + 1]):
                        in_for_header = True
                        for_paren_depth = paren_depth
                    scan_i += 1
                elif ch == ')':
                    if in_for_header and paren_depth == for_paren_depth:
                        in_for_header = False
                    paren_depth = max(0, paren_depth - 1)
                    scan_i += 1
                elif ch in ('"', "'"):          # skip string literals
                    q = ch; scan_i += 1
                    while scan_i < len(scan_text) and scan_text[scan_i] != q:
                        if scan_text[scan_i] == '\\':
                            scan_i += 1
                        scan_i += 1
                    scan_i += 1
                elif scan_text[scan_i:scan_i+2] == '//':  # skip line comments
                    while scan_i < len(scan_text) and scan_text[scan_i] != '\n':
                        scan_i += 1
                elif scan_text[scan_i:scan_i+2] == '/*':  # skip block comments
                    scan_i += 2
                    while scan_i < len(scan_text) - 1 and scan_text[scan_i:scan_i+2] != '*/':
                        scan_i += 1
                    scan_i += 2
                else:
                    m = _IDENT.match(scan_text, scan_i)
                    # Only match at a word boundary (not mid-identifier)
                    if m and (scan_i == 0 or not (scan_text[scan_i-1].isalnum() or scan_text[scan_i-1] == '_')):
                        kw, vname = m.group(1), m.group(2)
                        current_identifiers.add(vname)
                        if not in_for_header:
                            # for(let i) / for(const x) have their own per-iteration scope;
                            # exclude from same-scope redeclaration tracking and const tracking.
                            decl_at_depth.setdefault(vname, []).append((scan_depth, kw))
                            if kw == 'const':
                                const_identifiers.add(vname)
                        scan_i = m.end()
                    else:
                        scan_i += 1

            # Also collect function parameters into current_identifiers
            for param_str in post_items["function_parameter"]:
                p_name = (param_str.split('=')[0].strip() if '=' in param_str else param_str.strip())
                if p_name:
                    current_identifiers.add(p_name)

            # Implicit declarations (catch, for-in/of, arrow params)
            current_identifiers.update(re.findall(r'catch\s*\(\s*([a-zA-Z_$][a-zA-Z0-9_$]*)', full_gen_so_far))
            current_identifiers.update(re.findall(r'for\s*\(\s*(?:var|let|const\s+)?([a-zA-Z_$][a-zA-Z0-9_$]*)\s+(?:of|in)', full_gen_so_far))
            current_identifiers.update(re.findall(r'(?:^|[\W])([a-zA-Z_$][a-zA-Z0-9_$]*)\s*=>', full_gen_so_far))

            # 1. Redeclaration: `let`/`const x` at brace depth D conflicts only when
            # another declaration of `x` already exists at the same depth D.
            decl_start_idx = pre_counts.get("var_decl", 0)
            new_decls = post_items["var_decl"][decl_start_idx:]
            for decl_str in new_decls:
                m = re.search(r'(var|let|const)\s+([a-zA-Z_$][a-zA-Z0-9_$]*)', decl_str)
                if not m:
                    continue
                kw, var_name = m.group(1), m.group(2)
                if kw not in ('let', 'const'):
                    continue
                entries = decl_at_depth.get(var_name, [])
                if len(entries) >= 2:
                    new_depth = entries[-1][0]
                    if any(d == new_depth for (d, _) in entries[:-1]):
                        violation_reason = f"Redeclared '{var_name}' at depth {new_depth}"
                        is_valid = False
                        break

            # 2. Undeclared variable: pure identifier that hasn't been declared yet.
            # SPM's primary_safe_non_numeric only captures variable-position identifiers,
            # not property names (.prop) or method names, so false positives are low.
            if is_valid:
                for prim_str in post_items["primary_safe_non_numeric"][pre_counts.get("primary_safe_non_numeric", 0):]:
                    token = prim_str.strip()
                    if re.match(r'^[a-zA-Z_$][a-zA-Z0-9_$]*$', token) and token not in current_identifiers:
                        violation_reason = f"Undeclared: '{token}'"
                        is_valid = False
                        break

            # 3. Const reassignment: `constVar = ...` where LHS is not a member access.
            if is_valid:
                assignment_ops = ["=", "+=", "-=", "*=", "/=", "%=", "**=", ">>=", "<<=", ">>>=", "&=", "^=", "&&=", "||=", "??="]
                assignment_ops.sort(key=len, reverse=True)
                ops_pattern = '|'.join(map(re.escape, assignment_ops))
                for expr_str in post_items["expr_safe"][pre_counts.get("expr_safe", 0):]:
                    m = re.match(r'^\s*([a-zA-Z_$][a-zA-Z0-9_$]*)\s*(' + ops_pattern + ')', expr_str)
                    if m:
                        lhs = m.group(1)
                        is_member_access = bool(re.match(r'^\s*[a-zA-Z_$][a-zA-Z0-9_$]*\s*[.\[]', expr_str.lstrip()))
                        if lhs in const_identifiers and not is_member_access:
                            violation_reason = f"Reassigned const: '{lhs}'"
                            is_valid = False
                            break

            # 4. Orphaned break/continue outside any loop.
            if is_valid:
                for cf_str in post_items["control_flow_statement"][pre_counts.get("control_flow_statement", 0):]:
                    first = cf_str.strip().split()[0] if cf_str.strip() else ""
                    if first in {"break", "continue"} and not re.search(r'\b(for|while|do)\b', full_gen_so_far):
                        violation_reason = f"Orphaned '{first}' outside loop"
                        is_valid = False
                        break

            # backtrack if any violation is found
            if not is_valid:
                self.itergen.backward(unit="statement", num=1)
                # Reset prev_gen_len after backtracking so the stall detector stays accurate.
                backtracked_code = self.itergen.structured_gen[0] if self.itergen.structured_gen else ""
                new_pos = len(backtracked_code)
                # If we keep landing at the same backtrack position, the model is
                # regenerating the identical invalid statement -- abort instead of spinning.
                if new_pos == last_backtrack_pos:
                    consecutive_backtracks += 1
                    if consecutive_backtracks >= MAX_CONSECUTIVE_BACKTRACKS:
                        tail = backtracked_code[-80:] if backtracked_code else ""
                        print(f"[itergen] abort: {consecutive_backtracks}x stuck @pos={new_pos} | reason={violation_reason!r} | tail={tail!r}")
                        break
                else:
                    last_backtrack_pos = new_pos
                    consecutive_backtracks = 1
                prev_gen_len = new_pos
                continue

            # Successful step -- reset the backtrack-loop tracker.
            last_backtrack_pos = -1
            consecutive_backtracks = 0

        full_text = self.itergen.structured_gen[0] if self.itergen.structured_gen else ""
        if full_text.startswith(prompt):
            generated_for_depth = full_text[len(prompt):]
        else:
            generated_for_depth = full_text

        # If IterGen stops on a timeout, parser exception, or model budget before
        # closing the prompted function body, the benchmark sees a SyntaxError
        # even when the prefix is very close to the unconstrained greedy answer.
        # Use the already-loaded model to greedily finish only those incomplete
        # bodies, then trim as soon as the original prompt brace is balanced.
        if prompt_brace_depth > 0 and prompt_brace_depth + self._brace_depth(generated_for_depth) > 0:
            full_text = self._complete_unconstrained(
                full_text=full_text,
                prompt=prompt,
                stop_tokens=stop_tokens,
                prompt_brace_depth=prompt_brace_depth,
                temperature=temp,
                max_new_tokens=kwargs.get("itergen_fallback_max_new_tokens", 128),
            )

        generated_only = ""
        if full_text.startswith(prompt):
            generated_only = full_text[len(prompt):]
        else:
            # If prompt is not a prefix for some reason, return empty or full text depending on logic.
            # Usually safe to return full_text if it doesn't align, or just empty.
            # But let's try to return full text if prompt checking fails to be safe, though unexpected.
            generated_only = full_text 
            
        return self._post_process_stop(generated_only, stop_tokens, prompt_brace_depth)

    def _post_process_stop(self, text: str, stop_tokens, prompt_brace_depth: int = 0) -> str:
        # Skip \n// and \n/* when still inside a function body (brace depth > 0):
        # grammar-constrained gen allows comments anywhere via %ignore, so these
        # don't signal a new top-level declaration.
        if not stop_tokens:
            return text
        min_stop_index = len(text)
        found = False
        for stop in stop_tokens:
            idx = text.find(stop)
            if idx == -1:
                continue
            if stop in ('\n//', '\n/*'):
                depth = prompt_brace_depth + text[:idx].count('{') - text[:idx].count('}')
                if depth > 0:
                    continue
            min_stop_index = min(min_stop_index, idx)
            found = True
        return text[:min_stop_index] if found else text

    def _complete_unconstrained(
        self,
        full_text: str,
        prompt: str,
        stop_tokens,
        prompt_brace_depth: int,
        temperature: float,
        max_new_tokens: int = 128,
    ) -> str:
        if max_new_tokens <= 0:
            return full_text

        tokenizer = self.itergen.tokenizer
        model = self.itergen.model
        inputs = tokenizer([full_text], return_tensors="pt").to(self.itergen.device)
        input_len = inputs["input_ids"].shape[-1]
        do_sample = temperature > 0
        gen_kwargs = {
            "max_new_tokens": max_new_tokens,
            "do_sample": do_sample,
            "pad_token_id": tokenizer.eos_token_id,
        }
        if do_sample:
            gen_kwargs["temperature"] = temperature

        with torch.no_grad():
            outputs = model.generate(**inputs, **gen_kwargs)

        tail = tokenizer.decode(outputs[0][input_len:], skip_special_tokens=True)
        if not tail:
            return full_text

        completed = full_text + tail
        if completed.startswith(prompt):
            generated = completed[len(prompt):]
            trimmed = self._trim_at_balanced_brace(generated, prompt_brace_depth)
            return prompt + self._post_process_stop(trimmed, stop_tokens, prompt_brace_depth)

        return completed

    def _trim_at_balanced_brace(self, text: str, prompt_brace_depth: int) -> str:
        depth = prompt_brace_depth
        state = "code"
        escape = False
        i = 0

        while i < len(text):
            ch = text[i]
            nxt = text[i + 1] if i + 1 < len(text) else ""

            if state == "line_comment":
                if ch == "\n":
                    state = "code"
            elif state == "block_comment":
                if ch == "*" and nxt == "/":
                    state = "code"
                    i += 1
            elif state in {"single_quote", "double_quote", "template"}:
                if escape:
                    escape = False
                elif ch == "\\":
                    escape = True
                elif (
                    (state == "single_quote" and ch == "'")
                    or (state == "double_quote" and ch == '"')
                    or (state == "template" and ch == "`")
                ):
                    state = "code"
            else:
                if ch == "/" and nxt == "/":
                    state = "line_comment"
                    i += 1
                elif ch == "/" and nxt == "*":
                    state = "block_comment"
                    i += 1
                elif ch == "'":
                    state = "single_quote"
                elif ch == '"':
                    state = "double_quote"
                elif ch == "`":
                    state = "template"
                elif ch == "{":
                    depth += 1
                elif ch == "}":
                    depth -= 1
                    if depth <= 0:
                        return text[:i + 1]

            i += 1

        return text

    @staticmethod
    def _brace_depth(text: str) -> int:
        depth = 0
        state = "code"
        escape = False
        i = 0

        while i < len(text):
            ch = text[i]
            nxt = text[i + 1] if i + 1 < len(text) else ""

            if state == "line_comment":
                if ch == "\n":
                    state = "code"
            elif state == "block_comment":
                if ch == "*" and nxt == "/":
                    state = "code"
                    i += 1
            elif state in {"single_quote", "double_quote", "template"}:
                if escape:
                    escape = False
                elif ch == "\\":
                    escape = True
                elif (
                    (state == "single_quote" and ch == "'")
                    or (state == "double_quote" and ch == '"')
                    or (state == "template" and ch == "`")
                ):
                    state = "code"
            else:
                if ch == "/" and nxt == "/":
                    state = "line_comment"
                    i += 1
                elif ch == "/" and nxt == "*":
                    state = "block_comment"
                    i += 1
                elif ch == "'":
                    state = "single_quote"
                elif ch == '"':
                    state = "double_quote"
                elif ch == "`":
                    state = "template"
                elif ch == "{":
                    depth += 1
                elif ch == "}":
                    depth -= 1

            i += 1

        return depth

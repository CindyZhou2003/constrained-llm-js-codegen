import re
from .itergen.itergen.main import IterGen
from .base import BaseGenerator

class ItergenGenerator(BaseGenerator):
    def __init__(self, model_name: str, grammar: str, **kwargs):

        temp= kwargs.get("temperature")
        itergen_params = {
            "model_id": model_name,
            "grammar": grammar,
            "parse_output_only": False,
            "recurrence_penalty": 1.0,
            "max_new_tokens": kwargs.get("max_new_tokens"),
            "do_sample": temp > 0  # if temperature > 0, enable sampling; otherwise, use greedy decoding
        }
        if temp > 0:
            itergen_params["temperature"] = temp
            
        self.itergen = IterGen(
            **itergen_params
        )
        
    def generate(self, prompt: str, stop_tokens, **kwargs) -> str:
        
        temp= kwargs.get("temperature")
        itergen_params = {
            "model_id": self.itergen.model_id,
            "grammar": self.itergen.grammar,
            "parse_output_only": True,            
            "recurrence_penalty": 0.0,
            "max_new_tokens": kwargs.get("max_new_tokens"),
            "do_sample": temp > 0  # if temperature > 0, enable sampling; otherwise, use greedy decoding
        }
        if temp > 0:
            itergen_params["temperature"] = temp
        
        self.itergen.start(prompt=prompt)
        
        tracking_categories = [
            "var_decl", "function_declaration", "function_parameter", "primary_safe_non_numeric",
            "expr_safe", "control_flow_statement"
        ]
        
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
        
        # print(f"DEBUG: Starting generation. Max tokens: {kwargs.get('max_new_tokens')}")
        
        # Count opening braces in prompt to track when the function body is complete
        prompt_brace_depth = prompt.count('{') - prompt.count('}')
        
        # Increase loop limit significantly. Assuming 1 step ~= 1 token roughly, use max_new_tokens + buffer
        max_steps = kwargs.get("max_new_tokens", 512)
        # Ensure at least 500 steps if max_new_tokens is small or None, as steps are granular
        if max_steps < 500:
            max_steps = 500

        for step in range(max_steps):
            
            # self.itergen.forward(unit="statement", num=1, **intergen_params)
            # print(f"Step {step+1}:{self.itergen.structured_gen[0] if self.itergen.structured_gen else ''}\n")
            # semantic checks
            pre_counts = {}

            for cat in tracking_categories:
                try:
                    res = self.itergen.view(unit=cat)
                    pre_counts[cat] = len(res[0]) if res and res[0] else 0
                except Exception:
                    pre_counts[cat] = 0


            # forward 1 step
            try:
                self.itergen.forward(unit="statement", num=1, **itergen_params)
            except Exception as e:
                # Simply log and break on any error - don't try complex recovery
                print(f"DEBUG: Forward step failed: {e}")
                break

            current_code = self.itergen.structured_gen[0] if self.itergen.structured_gen else ""
            # print(f"DEBUG: Step {step+1} code len: {len(current_code)}")
            print(f"DEBUG: Step {step+1} current code:\n{current_code}\n---")

            # --- Early termination checks ---
            generated_so_far = current_code[len(prompt):] if current_code.startswith(prompt) else current_code
            
            # Check 1: Stop tokens — if any stop token appears in generated text, stop immediately
            if stop_tokens and generated_so_far:
                should_stop = False
                for stop in stop_tokens:
                    if stop in generated_so_far:
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
            current_identifiers = base_identifiers.copy()
            const_identifiers = set() # Track consts separately
            
            # 1. Track Declarations and Const Immutability
            decl_start_idx = pre_counts.get("var_decl", 0)
            for idx, decl_str in enumerate(post_items["var_decl"]):
                match = re.search(r'\s*(?:var|let|const)\s+([a-zA-Z_$][a-zA-Z0-9_$]*)', decl_str)
                if match:
                    var_name = match.group(1)
                    decl_keyword = decl_str.strip().split()[0]  # 'var', 'let', or 'const'
                    
                    # Rule 1a: Check for Redeclaration (Only for let/const, var allows redeclaration)
                    if idx >= decl_start_idx and decl_keyword in ('let', 'const'):
                        if var_name in current_identifiers:
                            violation_reason = f"Redeclared variable: '{var_name}'"
                            is_valid = False
                            break

                    current_identifiers.add(var_name)
                    # Identify if this is a 'const' declaration
                    if decl_str.strip().startswith("const"):
                        const_identifiers.add(var_name)
            

            for param_str in post_items["function_parameter"]:
                # simple split for now
                if "=" in param_str:
                     p_name = param_str.split('=')[0].strip()
                else:
                     p_name = param_str.strip()
                if p_name:
                    current_identifiers.add(p_name)

            # Heuristic: Scan for implicit declarations (catch, for-in/of, arrow functions) that might be missed by simple views
            # Since we re-enable strict checking, we must ensure these valid declaring contexts are captured.
            full_gen_so_far = self.itergen.structured_gen[0] if self.itergen.structured_gen else ""
            
            # 1. catch(e)
            current_identifiers.update(re.findall(r'catch\s*\(\s*([a-zA-Z_$][a-zA-Z0-9_$]*)', full_gen_so_far))
            # 2. for (x of y) / for (x in y)
            current_identifiers.update(re.findall(r'for\s*\(\s*(?:var|let|const\s+)?([a-zA-Z_$][a-zA-Z0-9_$]*)\s+(?:of|in)', full_gen_so_far))
            # 3. Arrow function params (simple Identifier => ...)
            current_identifiers.update(re.findall(r'(?:^|[\W])([a-zA-Z_$][a-zA-Z0-9_$]*)\s*=>', full_gen_so_far))

            # DISABLED: Undeclared variable check causes too many false positives
            # The parser extracts identifiers from comments, property accesses, etc.
            # which falsely triggers this check (e.g., 'ray' from 'array' in comments).
            # Grammar constraints ensure syntactic validity; semantic checks are over-engineering.
            # for prim_str in post_items["primary_safe_non_numeric"][pre_counts.get("primary_safe_non_numeric", 0):]:
            #     token = prim_str.strip()
            #     if re.match(r'^[a-zA-Z_$][a-zA-Z0-9_$]*$', token):
            #             if token not in current_identifiers:
            #                violation_reason = f"Undeclared var: '{token}'"
            #                is_valid = False
            #                break

            # 2. Check Assignments (Const reassignment & Literal assignment)
            if is_valid:
                assignment_ops = ["=", "+=", "-=", "*=", "/=", "%=", "**=", ">>=", "<<=", ">>>=", "&=", "^=", "&&=", "||=", "??="]
                assignment_ops.sort(key=len, reverse=True) # Longest first
                ops_pattern = '|'.join(map(re.escape, assignment_ops))
                
                for expr_str in post_items["expr_safe"][pre_counts.get("expr_safe", 0):]:
                    match_assign = re.match(r'^\s*([a-zA-Z_$][a-zA-Z0-9_$]*)\s*(' + ops_pattern + ')', expr_str)
                    
                    if match_assign:
                        left_side_token = match_assign.group(1)
                            
                        # Rule 2a: Prevent reassigning a const variable
                        if left_side_token in const_identifiers:
                            violation_reason = f"Reassigned const: '{left_side_token}'"
                            is_valid = False
                            break
                                
                        # Rule 2b: Prevent assigning to literals (e.g., 5 = x, true = y)
                        is_literal = (
                            left_side_token in {"true", "false", "null", "undefined"} or 
                            re.match(r'^\d', left_side_token) or 
                            left_side_token.startswith('"') or 
                            left_side_token.startswith("'")
                        )
                        if is_literal:
                            violation_reason = f"Invalid assignment to literal: '{left_side_token}'"
                            is_valid = False
                            break

            # 3. Check for Orphaned Loop Controls
            if is_valid:
                for cf_str in post_items["control_flow_statement"][pre_counts.get("control_flow_statement", 0):]:
                    token_first = cf_str.strip().split()[0] if cf_str.strip() else ""
                    if token_first in {"break", "continue"}:
                        # Heuristic: Check if a loop keyword exists anywhere in the text generated so far
                        full_gen_so_far = self.itergen.structured_gen[0] if self.itergen.structured_gen else ""
                        if not any(loop_kw in full_gen_so_far for loop_kw in ["for", "while"]):
                            violation_reason = f"Orphaned '{token_first}' outside of loop"
                            is_valid = False
                            break

            # backtrack if any violation is found
            if not is_valid:
                # print(f"DEBUG: Violation found: {violation_reason}. Backtracking...")
                current_state = self.itergen.structured_gen[0] if self.itergen.structured_gen else ""
                print(f"DEBUG: Step {step} current code before backtrack:\n{current_state}\n---")
                self.itergen.backward(unit="statement", num=1)
                continue

        full_text = self.itergen.structured_gen[0] if self.itergen.structured_gen else ""
        generated_only = ""
        if full_text.startswith(prompt):
            generated_only = full_text[len(prompt):]
        else:
            # If prompt is not a prefix for some reason, return empty or full text depending on logic.
            # Usually safe to return full_text if it doesn't align, or just empty.
            # But let's try to return full text if prompt checking fails to be safe, though unexpected.
            generated_only = full_text 
            
        return self._post_process_stop(generated_only, stop_tokens)
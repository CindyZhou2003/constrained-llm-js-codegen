import sys
import os
itergen_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'benchmark', 'itergen'))
syncode_outer = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'syncode'))
syncode_inner=os.path.join(syncode_outer, 'syncode')

if syncode_inner not in sys.path:
    sys.path.insert(0, syncode_inner)
if syncode_outer not in sys.path:
    sys.path.insert(0, syncode_outer)
if itergen_root not in sys.path:
    sys.path.insert(0, itergen_root)

from benchmark.itergen.itergen.main import IterGen
from .base import BaseGenerator
import re
import traceback

class ItergenGenerator(BaseGenerator):
    def __init__(self, model_name: str, grammar: str):
        if grammar and os.path.exists(grammar):
            with open(grammar, 'r', encoding='utf-8') as f:
                grammar_content = f.read()
        else:
            raise ValueError("Structured generation requires a valid .lark grammar file.")

        self.itergen = IterGen(
            model_id=model_name,
            grammar=grammar_content,
            parse_output_only=True,
            recurrence_penalty=0.0
        )

    def _flatten_tokens(self, nested_ast):
        """
        Recursively flattens a nested AST structure into a list of tokens.
        """
        if isinstance(nested_ast, str):
            return [nested_ast]
        res = []
        for item in nested_ast:
            res.extend(self._flatten_tokens(item))
        return res

    def generate(self, prompt: str, **gen_args) -> str:
        filtered_args = gen_args.copy()
        
        filtered_args.pop("stop_tokens", None)
        filtered_args.pop("temperature", None)
        
        self.itergen.start(prompt=prompt)
        # print(f">>> Starting Step-by-Step Generation...")
        
        # Step 1: Analyze grammar to identify which tokens to track for semantic checks
        base_identifiers = set([
            "console", "Math", "Object", "Array", "String", "Number", 
            "true", "false", "null", "undefined", "NaN", "Infinity",
            "this", "arguments", "window", "document"
        ])
        
        func_match = re.search(r'function\s+\w+\s*\(([^)]*)\)', prompt)
        if func_match:
            args_str = func_match.group(1)
            for arg in args_str.split(','):
                arg = arg.strip()
                if arg:
                    base_identifiers.add(arg)

        tracking_categories = [
            "var_decl", "function_declaration", "function_parameter", "primary_safe"
        ]

        previous_statement = ""
        repeat_count = 0
        
        for _ in range(50):
            try:
                pre_counts = {}
                for cat in tracking_categories:
                    try:
                        res = self.itergen.view(unit=cat)
                        pre_counts[cat] = len(res[0]) if res and res[0] else 0
                    except Exception:
                        pre_counts[cat] = 0

                # forward 1 step
                self.itergen.forward(unit="statement", num=1, **filtered_args)
                
                # catch statement
                try:
                    statements = self.itergen.view(unit="statement")
                except Exception as e:
                    # print(f"DEBUG: view('statement') failed or rule not tracked: {e}")
                    break
                
                if not statements or not statements[0]:
                    break
                    
                latest_stmt = "".join(self._flatten_tokens(statements[0][-1])).strip()
                # print(f"Step {step}: {latest_stmt}")
                
                post_items = {}
                for cat in tracking_categories:
                    try:
                        res = self.itergen.view(unit=cat)
                        post_items[cat] = res[0] if res and res[0] else []
                    except Exception:
                        post_items[cat] = []

                # semantic checks
                is_valid = True
                current_identifiers = base_identifiers.copy()
                
                for decl_ast in post_items["var_decl"]:
                    tokens = self._flatten_tokens(decl_ast)
                    if len(tokens) >= 2:
                        current_identifiers.add(tokens[1])

                for param_ast in post_items["function_parameter"]:
                    tokens = self._flatten_tokens(param_ast)
                    if tokens:
                        current_identifiers.add(tokens[0])

                for prim_ast in post_items["primary_safe"][pre_counts.get("primary_safe", 0):]:
                    tokens = self._flatten_tokens(prim_ast)
                    if len(tokens) == 1:
                        token = tokens[0]
                        if re.match(r'^[a-zA-Z_$][a-zA-Z0-9_$]*$', token):
                            if token not in current_identifiers:
                                # print(f"--> [Semantic Violation] Undeclared var: '{token}'. Backtracking...")
                                is_valid = False
                                break

                # logic-based anti-pattern checks
                try:
                    all_stmts_ast = self.itergen.view(unit="statement")
                    current_code_str = "\n".join(["".join(self._flatten_tokens(stmt)) for stmt in all_stmts_ast[0]])
                except:
                    current_code_str = latest_stmt
                
                # no unconditional early return inside loops (common anti-pattern)
                if re.search(r'for\s*\([^)]+\)\s*\{[\s\S]*?if\s*\([^)]+\)\s*\{[^}]*return[^}]*\}[\s\S]*?else\s*\{[^}]*return', current_code_str):
                    # print("--> [Semantic Violation] Anti-Pattern: Unconditional early return inside loop. Backtracking...")
                    is_valid = False

                # no "return true" when prompt implies checking all elements
                if "each element" in prompt.lower() or "all elements" in prompt.lower():
                    if "return true" in latest_stmt and "for" in current_code_str:
                        open_braces = current_code_str.count('{')
                        close_braces = current_code_str.count('}')
                        if open_braces > close_braces:
                            # print("--> [Semantic Violation] Anti-Pattern: Should not 'return true' inside the loop. Backtracking...")
                            is_valid = False
                            
                # no lazy comments or empty blocks that suggest the model is stalling
                lazy_regex = r'\b(do something|code here|your code|implement)\b'
                is_lazy_text = bool(re.search(lazy_regex, latest_stmt.lower()))
                is_empty_block = bool(re.search(r'\{\s*(//[^\n]*)?\s*\}', latest_stmt))
                
                if is_lazy_text or is_empty_block:
                    # print(f"--> [Semantic Violation] Anti-Pattern: Lazy comment or empty block detected in '{latest_stmt.strip()[:30]}...'. Backtracking...")
                    is_valid = False
                
                # backtracking if any violation detected
                if not is_valid:
                    self.itergen.backward(unit="statement", num=1)
                    continue
                
                if latest_stmt == previous_statement:
                    repeat_count += 1
                    if repeat_count >= 3:
                        break
                else:
                    previous_statement = latest_stmt
                    repeat_count = 0
                    
            except Exception as e:
                # print(f"\n!!! FATAL ERROR IN LOOP !!!\n{e}")
                # traceback.print_exc()
                break

        # get final generated code
        full_text = self.itergen.structured_gen[0] if self.itergen.structured_gen else ""
        return self._post_process_stop(full_text, stop_tokens=gen_args.get("stop_tokens"))
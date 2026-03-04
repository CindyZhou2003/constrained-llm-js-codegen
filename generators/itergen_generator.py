import sys
import os
import re
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

    def generate(self, prompt: str, **kwargs) -> str:
        filtered_args = kwargs.copy()
        filtered_args.pop("stop_tokens", None)
        filtered_args.pop("temperature", None)
        
        self.itergen.start(prompt=prompt)
        
        for step in range(30):
            
            self.itergen.forward(unit="statement", num=1, **filtered_args)
            print(f"Step {step+1}:{self.itergen.structured_gen[0] if self.itergen.structured_gen else ''}\n")
        

        full_text = self.itergen.structured_gen[0] if self.itergen.structured_gen else ""
        
        # combined_text = prompt + full_text
        # if combined_text.count('{') > combined_text.count('}'):
        #     full_text += "\n}"
            
        return self._post_process_stop(full_text, stop_tokens=kwargs.get("stop_tokens"))
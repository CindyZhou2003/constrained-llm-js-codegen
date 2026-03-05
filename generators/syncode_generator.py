from syncode import Syncode
from .base import BaseGenerator

class SyncodeGenerator(BaseGenerator):
    def __init__(self, model_name: str, grammar: str, **kwargs):
        
        temp = kwargs.get("temperature")
        
        syncode_params = {
            "model": model_name, 
            "mode": 'grammar_mask', 
            "grammar": grammar, 
            "parse_output_only": False,
            # "max_new_tokens": kwargs.get("max_new_tokens"),
            "do_sample": temp > 0  # if temperature > 0, enable sampling; otherwise, use greedy decoding
        }
        
        if temp > 0:
            syncode_params["temperature"] = temp
            
        self.sc = Syncode(**syncode_params)
    

    def generate(self, prompt: str, stop_tokens, **kwargs) -> str:
        completions = self.sc.infer(prompt, stop_words=stop_tokens)
        text = completions[0] if completions else ""
        return self._post_process_stop(text, stop_tokens)
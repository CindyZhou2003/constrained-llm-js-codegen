from syncode import Syncode
from .base import BaseGenerator

class SyncodeGenerator(BaseGenerator):
    def __init__(self, model_name: str, grammar: str):
        self.sc = Syncode(
            model=model_name, 
            mode='grammar_mask', 
            grammar=grammar, 
            parse_output_only=True
        )

    def generate(self, prompt: str, stop_tokens=None, **kwargs) -> str:
        completions = self.sc.infer(prompt, stop_words=stop_tokens)
        text = completions[0] if completions else ""
        return self._post_process_stop(text, stop_tokens)
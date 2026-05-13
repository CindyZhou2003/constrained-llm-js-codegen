import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from .base import BaseGenerator

class HFGenerator(BaseGenerator):
    def __init__(self, model_name: str, **kwargs):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=dtype, device_map="auto", trust_remote_code=True
        ).eval()

    def generate(self, prompt: str, stop_tokens=None, **kwargs) -> str:
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        input_len = inputs.input_ids.shape[1]
        
        temp = kwargs.get("temperature")
        do_sample = temp > 0
        model_params = {
            "max_new_tokens": kwargs.get("max_new_tokens"),
            "do_sample": do_sample,
            "temperature": temp if do_sample else None,
            "top_p": None,
            "top_k": None,
            "repetition_penalty": None,
        }
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                **model_params,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        text = self.tokenizer.decode(outputs[0][input_len:], skip_special_tokens=True)
        return self._post_process_stop_js(text, stop_tokens, prompt)

    def _post_process_stop_js(self, text: str, stop_tokens, prompt: str) -> str:
        if not stop_tokens:
            return text

        prompt_brace_depth = self._brace_depth(prompt)
        min_stop_index = len(text)
        found = False

        for stop in stop_tokens:
            start = 0
            while True:
                idx = text.find(stop, start)
                if idx == -1:
                    break

                depth = prompt_brace_depth + self._brace_depth(text[:idx])
                if depth <= 0:
                    min_stop_index = min(min_stop_index, idx)
                    found = True
                    break

                start = idx + len(stop)

        return text[:min_stop_index] if found else text

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

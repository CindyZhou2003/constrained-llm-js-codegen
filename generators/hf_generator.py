import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from .base import BaseGenerator

class HFGenerator(BaseGenerator):
    def __init__(self, model_name: str):
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
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=kwargs.get("max_new_tokens", 512),
                temperature=kwargs.get("temperature", 0.2),
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        text = self.tokenizer.decode(outputs[0][input_len:], skip_special_tokens=True)
        return self._post_process_stop(text, stop_tokens)
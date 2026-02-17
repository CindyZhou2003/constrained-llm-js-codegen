import os
import torch
import json
import gzip
from typing import List, Dict, Any, Optional
from transformers import AutoTokenizer, AutoModelForCausalLM, LogitsProcessorList
from syncode import Syncode


class UnifiedCodeGenerator:
    def __init__(self, model_name: str, device: str = None, model_kwargs: Optional[Dict[str, Any]] = None):
        """
        One-time initialization of the model and tokenizer.
        """
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        print(f"--- Loading Model: {model_name} to {self.device} ---")
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            padding_side="left",
            trust_remote_code=True,
        )
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=dtype,
            device_map="auto" if self.device == "cuda" else None,
            trust_remote_code=True     
        ).cuda()
        self.model.eval()
        
        # Cache Syncode processors by language to avoid re-compiling grammars
        self._syncode_cache = {}

    def _get_syncode(self, grammar: str):
        """Internal helper to setup Syncode LogitsProcessor."""
        if Syncode is None:
            raise ImportError("Syncode is not installed. Run 'pip install syncode'.")
        
        if grammar not in self._syncode_cache:
            print(f"Initializing Syncode Grammar for: {grammar}")
            sc = Syncode(
                model=self.model_name, 
                mode='grammar_mask', 
                grammar=grammar, # e.g., 'javascript.lark'
                parse_output_only=True
            )
            self._syncode_cache[grammar] = sc
        return self._syncode_cache[grammar]

    def _post_process_stop(self, text: str, stop_tokens: List[str]) -> str:
        """Implements the MultiPL-E stop token slicing logic."""
        if not stop_tokens:
            return text
        
        min_stop_index = len(text)
        found = False
        for stop in stop_tokens:
            idx = text.find(stop)
            if idx != -1:
                min_stop_index = min(min_stop_index, idx)
                found = True
        return text[:min_stop_index] if found else text

    def generate(
        self, 
        prompt: str, 
        mode: str = "unconstrained", # "unconstrained" or "syncode"
        grammar: str = "javascript", # Can also be "javascript.lark" for Syncode grammar
        stop_tokens: Optional[List[str]] = None,
        max_new_tokens: int = 512,
        temperature: float = 0.2,
        top_p: float = 0.95,
        **extra_params
    ) -> str:
        """
        The main API entry point. Returns the generated string.
        
        :param mode: 'unconstrained' or 'syncode'
        :param grammar: The grammar for Syncode constraints (e.g. 'javascript')
        """
        self.model.eval()
        
        if mode == "syncode":
            syn_llm = self._get_syncode(grammar)
            completions= syn_llm.infer(
                prompt,
                stop_words=stop_tokens
            )
            generated_text = completions[0] if completions else ""
        
        # default generation logic for unconstrained mode
        else:
            inputs = self.tokenizer(
                prompt,
                padding=True,
                return_tensors="pt",
                return_token_type_ids=False,
                truncation=True,
                max_length=max_new_tokens-1,  # Ensure total length fits model context
            ).to(self.device)
            input_len = inputs.input_ids.shape[1]

            # Base parameters
            gen_config = {
                "max_new_tokens": max_new_tokens,
                "do_sample": True if temperature > 0 else False,
                "temperature": temperature if temperature > 0 else 1.0,
                "top_p": 0.95,
                "pad_token_id": self.tokenizer.eos_token_id,
                "logits_processor": LogitsProcessorList()
            }
            gen_config.update(extra_params)

            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    use_cache=True,
                    **gen_config
                )
            
            generated_text = self.tokenizer.decode(outputs[0][input_len:], skip_special_tokens=True)
            
            # Apply MultiPL-E clean-up
        return self._post_process_stop(generated_text, stop_tokens)

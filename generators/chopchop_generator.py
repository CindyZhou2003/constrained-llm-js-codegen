"""
Generic ChopChop generation framework.

Language-specific components (AST nodes, constructors, pruners, prompt helpers)
are provided by language plugins, e.g. generators/javascript_chopchop.py.

Typical usage:

    from generators.javascript_chopchop import (
        CONSTRUCTORS, JS_START_RULE, JS_CONTEXT,
        make_js_pruner, extract_js_prefix, build_js_prompt,
    )
    gen = ChopchopGenerator(
        model_name="microsoft/phi-2",
        grammar="generators/grammars/javascript_chopchop.lark",
        constructors=CONSTRUCTORS,
        start_rule=JS_START_RULE,
        pruner_fn=make_js_pruner("basic"),
        extract_prefix_fn=extract_js_prefix,
        build_prompt_fn=build_js_prompt,
        context=JS_CONTEXT,
    )
    output = gen.generate(prompt, stop_tokens=[...], temperature=0.2)
"""

import gc
import sys
import time
from pathlib import Path
from typing import Callable, List, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.cache_utils import DynamicCache
from transformers.utils import logging as hf_logging

from .base import BaseGenerator


# Make ChopChop package imports available when running from repository root.
CHOPCHOP_ROOT = Path(__file__).resolve().parent / "chopchop"
if str(CHOPCHOP_ROOT) not in sys.path:
    sys.path.insert(0, str(CHOPCHOP_ROOT))

from core.rewrite import rewriter  # type: ignore  # noqa: E402
from llm.realizability import RealizabilityChecker  # type: ignore  # noqa: E402
from llm.run_llm import Config, LanguageModelRunner, ModelConfig  # type: ignore  # noqa: E402
from core.lark.from_lark import parse_attribute_grammar  # type: ignore  # noqa: E402


class _SafeRealizabilityChecker(RealizabilityChecker):
    """
    Wraps the upstream RealizabilityChecker with safety guards:

      - RecursionError / NetworkXError guard (treat as realizable on error)
      - Cumulative budget: after `cumulative_budget` seconds have been spent
        in realizable() calls total, all further calls return True so that
        generation can finish within the overall timeout.

    The rewriter's global state is cleared before each call to prevent stale
    graph nodes from a previous (possibly timed-out) run causing KeyErrors
    in the networkx dependency graph.
    """

    def __init__(self, *args, per_call_timeout: float = 30.0,
                 cumulative_budget: float = 120.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.per_call_timeout = per_call_timeout
        self.cumulative_budget = cumulative_budget
        self._total_time = 0.0
        self._bypass = False

    def reset_budget(self):
        """Reset the cumulative time budget (call before each generation run)."""
        self._total_time = 0.0
        self._bypass = False

    def realizable(self, prefix: str, final: bool = False) -> bool:
        if self._bypass:
            return True

        t0 = time.monotonic()
        try:
            result = super().realizable(prefix, final=final)
        except RecursionError:
            rewriter.clear()
            result = True
        except Exception:
            rewriter.clear()
            result = True

        elapsed = time.monotonic() - t0
        self._total_time += elapsed

        if self._total_time >= self.cumulative_budget:
            self._bypass = True

        return result


class _LocalLanguageModelRunner(LanguageModelRunner):
    """
    Local subclass of the upstream LanguageModelRunner that overrides the
    model-loading, tokenization, and token-generation methods.

    Supports both chat-template models (e.g. Qwen Instruct) and plain
    causal-LM models (e.g. Phi-2) without modifying upstream code.
    """

    def _load_model_and_tokenizer(self):
        hf_logging.set_verbosity_error()
        model_id = self.model_config.model_id.strip()
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            device_map="auto",
            torch_dtype=self.model_config.dtype,
        )

        vocab_size = model.get_input_embeddings().num_embeddings
        if len(tokenizer) > vocab_size:
            model.resize_token_embeddings(len(tokenizer))

        return model, tokenizer

    def _tokenize_prompt(
        self, prompt: str, context: str, fixed_prefix: str = ""
    ) -> torch.Tensor:
        """
        Convert the task prompt into input token IDs.

        Chat-template models (e.g. Qwen2.5-Coder-Instruct): formats prompt as
        system+user message pair via apply_chat_template.

        Plain causal-LM models (e.g. Phi-2): concatenates context and prompt
        as a single string.

        If fixed_prefix is provided (used during retry after a partial timeout),
        its tokens are appended so that generation continues from that point.
        """
        if getattr(self.tokenizer, "chat_template", None):
            messages = [
                {"role": "system", "content": context},
                {"role": "user", "content": prompt},
            ]
            input_ids = self.tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                add_special_tokens=False,
                return_tensors="pt",
                padding=True,
            )
        else:
            text = f"{context}\n\n{prompt}\n"
            input_ids = self.tokenizer(
                text,
                add_special_tokens=False,
                return_tensors="pt",
            )["input_ids"]

        if fixed_prefix:
            prefix_tokens = self.tokenizer(
                fixed_prefix,
                add_special_tokens=False,
                return_tensors="pt",
            )["input_ids"]
            input_ids = torch.cat([input_ids, prefix_tokens], dim=-1)

        return input_ids.to(self.model.device)

    def _generate_next_token(
        self,
        input_ids: torch.Tensor,
        config: Config,
        generated_tokens: List[int],
        forbidden_tokens: set[int],
        cache: DynamicCache,
    ):
        """
        Generate the next single token, enforcing grammar-driven token masking
        via bad_words_ids (tokens that would make the prefix unrealizable are
        set to -inf before sampling).
        """
        bad_words = [[tok] for tok in forbidden_tokens] if forbidden_tokens else None

        inp = torch.tensor([list(input_ids[0]) + generated_tokens])
        inp = inp.to(self.model_config.device)
        attention_mask = torch.ones_like(inp, dtype=torch.long)

        if self.tokenizer.eos_token_id in forbidden_tokens:
            eos_token_id = None
        else:
            eos_token_id = self.tokenizer.eos_token_id

        do_sample = config.temperature > 0
        gen_kwargs = dict(
            attention_mask=attention_mask,
            do_sample=do_sample,
            pad_token_id=self.tokenizer.eos_token_id,
            eos_token_id=eos_token_id,
            max_new_tokens=1,
            bad_words_ids=bad_words,
            repetition_penalty=config.repetition_penalty,
            num_return_sequences=1,
            output_scores=True,
            return_dict_in_generate=True,
            past_key_values=cache,
        )
        if do_sample:
            gen_kwargs.update(
                temperature=config.temperature,
                top_p=config.top_p,
                top_k=config.top_k,
            )

        return self.model.generate(inp, **gen_kwargs)

    def __del__(self):
        try:
            if hasattr(self, "model"):
                del self.model
            if hasattr(self, "tokenizer"):
                del self.tokenizer
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass


class ChopchopGenerator(BaseGenerator):
    """
    Grammar-constrained code generator using the ChopChop framework.

    Parameters
    ----------
    model_name : str
        HuggingFace model ID.
    grammar : str
        Path to a .lark attribute grammar file.
    constructors : list
        Language-specific AST node classes for parse_attribute_grammar().
    start_rule : str
        Grammar entry point (e.g. "function_declaration").
    pruner_fn : callable, optional
        Callable(asts) -> asts applied inside the realizability checker.
        Defaults to the identity function (no extra pruning).
    extract_prefix_fn : callable, optional
        Callable(prompt: str) -> str that extracts a fixed_prefix from the raw
        prompt (e.g. the function signature without the opening brace).
        Defaults to returning an empty string.
    build_prompt_fn : callable, optional
        Callable(prompt: str) -> str that builds the instruction prompt for the
        model from the raw prompt text.
        Defaults to the identity function.
    context : str, optional
        System context string for the model. Defaults to a generic assistant
        prompt.
    """

    def __init__(
        self,
        model_name: str,
        grammar: Optional[str],
        *,
        constructors: list,
        start_rule: str,
        pruner_fn: Optional[Callable] = None,
        extract_prefix_fn: Optional[Callable[[str], str]] = None,
        build_prompt_fn: Optional[Callable[[str], str]] = None,
        context: str = "You are a helpful coding assistant.",
        **kwargs,
    ):
        self.model_name = model_name
        self.context = context
        self.fixed_prefix = kwargs.get("fixed_prefix", "")
        self.extract_prefix_fn = extract_prefix_fn or (lambda p: "")
        self.build_prompt_fn = build_prompt_fn or (lambda p: p)

        grammar_source = self._load_grammar_source(grammar)

        effective_pruner = pruner_fn if pruner_fn is not None else (lambda asts: asts)

        lexer_spec, parser = parse_attribute_grammar(
            constructors, grammar_source, start_rule
        ).build_parser()
        self.checker = _SafeRealizabilityChecker(effective_pruner, parser, lexer_spec)

        self.runner = _LocalLanguageModelRunner(ModelConfig(model_id=model_name))

    @staticmethod
    def _load_grammar_source(grammar: Optional[str]) -> str:
        if not grammar:
            raise FileNotFoundError(
                "ChopChop requires an explicit grammar file path via --grammar"
            )
        grammar_path = Path(grammar)
        if grammar_path.exists():
            return grammar_path.read_text(encoding="utf-8")
        raise FileNotFoundError(f"Grammar file not found: {grammar}")

    def generate(self, prompt: str, stop_tokens=None, **kwargs) -> str:
        """
        Generate a single completion for the given prompt.

        Flow:
        1. Build the instruction prompt via build_prompt_fn.
        2. Extract the fixed_prefix via extract_prefix_fn (unless overridden).
        3. Run ChopChop's grammar-constrained generation loop.
        4. Strip the fixed_prefix and opening brace from the output.
        5. Apply stop-token truncation.
        """
        temperature = float(kwargs.get("temperature"))
        max_new_tokens = kwargs.get("max_new_tokens")
        context = kwargs.get("context", self.context)
        fixed_prefix = kwargs.get("fixed_prefix", self.fixed_prefix)
        safety_timeout = int(kwargs.get("safety_timeout", 300))
        stop_tokens = kwargs.get("stop_tokens", stop_tokens) or []

        if not fixed_prefix:
            fixed_prefix = self.extract_prefix_fn(prompt)

        task_prompt = kwargs.get("task_prompt")
        if not task_prompt:
            task_prompt = self.build_prompt_fn(prompt)

        self.checker.reset_budget()

        try:
            run_info = self.runner.run(
                Config(
                    temperature=temperature,
                    max_new_tokens=max_new_tokens,
                    timeout=safety_timeout,
                ),
                prompt=task_prompt,
                context=context,
                fixed_prefix=fixed_prefix,
                realizability_checker=self.checker,
            )
        except RecursionError:
            self.last_run_info = None
            return ""

        self.last_run_info = run_info
        output = run_info.output

        # The grammar start rule produces "function foo(…) { body }".
        # Strip fixed_prefix and the opening "{" since the caller's prompt
        # already contains them, leaving only "body }" to append.
        if fixed_prefix and output.startswith(fixed_prefix):
            output = output[len(fixed_prefix):].lstrip()
            if output.startswith("{"):
                output = output[1:]

        output = self._post_process_stop(output, stop_tokens)

        return output

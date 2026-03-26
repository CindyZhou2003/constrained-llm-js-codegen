import gc
import sys
from pathlib import Path
from typing import List, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.cache_utils import DynamicCache
from transformers.utils import logging as hf_logging

from .base import BaseGenerator


# Make ChopChop package imports available when running from repository root.
CHOPCHOP_ROOT = Path(__file__).resolve().parent / "chopchop"
if str(CHOPCHOP_ROOT) not in sys.path:
    sys.path.insert(0, str(CHOPCHOP_ROOT))

from core.grammar import Application, Binary, Ternary, Unary, Zeroary  # type: ignore  # noqa: E402
from core.grammar import ASTLeaf, EmptySet, TreeGrammar, Union, as_tree  # type: ignore  # noqa: E402
from core.lark.from_lark import parse_attribute_grammar  # type: ignore  # noqa: E402
from core.rewrite import rewrite  # type: ignore  # noqa: E402
from llm.realizability import RealizabilityChecker  # type: ignore  # noqa: E402
from llm.run_llm import Config, LanguageModelRunner, ModelConfig  # type: ignore  # noqa: E402


class _LocalLanguageModelRunner(LanguageModelRunner):
    """Script-local compatibility layer; avoids editing upstream ChopChop files."""

    def _load_model_and_tokenizer(self):
        # Suppress non-actionable generation warnings (e.g., ignored sampling args in greedy mode).
        hf_logging.set_verbosity_error()
        model_id = self.model_config.model_id.strip()
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            device_map="auto",
            torch_dtype=self.model_config.dtype,
        )

        # Only expand embeddings if tokenizer is larger than checkpoint vocab.
        vocab_size = model.get_input_embeddings().num_embeddings
        if len(tokenizer) > vocab_size:
            model.resize_token_embeddings(len(tokenizer))

        return model, tokenizer

    def _has_closed_brace(self, generated_tokens: List[int]) -> bool:
        if not generated_tokens:
            return False
        decoded = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
        return "}" in decoded

    def _tokenize_prompt(
        self, prompt: str, context: str, fixed_prefix: str = ""
    ) -> torch.Tensor:
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
        bad_words = [[tok] for tok in forbidden_tokens] if forbidden_tokens else None

        inp = torch.tensor([list(input_ids[0]) + generated_tokens])
        inp = inp.to(self.model_config.device)
        attention_mask = torch.ones_like(inp, dtype=torch.long)

        if self.tokenizer.eos_token_id in forbidden_tokens:
            eos_token_id = None
        else:
            eos_token_id = self.tokenizer.eos_token_id

        # In function completion mode, delay EOS briefly so the model can emit the closing brace.
        if getattr(self, "require_trailing_brace", False):
            max_wait = int(getattr(self, "brace_wait_tokens", 64))
            if len(generated_tokens) < max_wait and not self._has_closed_brace(generated_tokens):
                eos_token_id = None

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


# Constructors matching generators/chopchop/grammars/javascript_chopchop_enhanced.lark
class Var(Unary): ...
class Num(Unary): ...
class Str(Unary): ...
class TemplateStr(Unary): ...
class TrueLit(Zeroary): ...
class FalseLit(Zeroary): ...
class NullLit(Zeroary): ...
class EmptyArray(Zeroary): ...
class Group(Unary): ...
class ReturnStmt(Unary): ...
class ReturnVoidStmt(Zeroary): ...
class ExprStmt(Unary): ...
class Neg(Unary): ...
class Not(Unary): ...
class UnaryPlus(Unary): ...
class TypeofExpr(Unary): ...
class AwaitExpr(Unary): ...
class Call0(Unary): ...
class ArrayLit(Unary): ...
class ThisLit(Zeroary): ...
class NewExpr(Unary): ...
class EmptyObject(Zeroary): ...
class ObjectLit(Unary): ...
class PropertyPair(Binary): ...
class PropertySeq(Binary): ...
class SpreadProperty(Unary): ...
class FunctionBodyClose(Unary): ...
class TopLevelSeq(Binary): ...
class ImportStmt(Unary): ...
class ImportFromStmt(Binary): ...
class ImportModuleStmt(Unary): ...
class ExportDefaultExpr(Unary): ...
class ExportDecl(Unary): ...
class ExportNamed(Unary): ...
class ExportNamedFrom(Binary): ...
class ImportName(Unary): ...
class ImportNamedSet(Unary): ...
class ImportSpec(Unary): ...
class ImportSpecAs(Binary): ...
class ImportSpecSeq(Binary): ...
class FunctionExpr(Binary): ...
class AsyncFunctionExpr(Binary): ...
class ClassExpr(Unary): ...
class FunctionDecl(Ternary): ...
class AsyncFunctionDecl(Ternary): ...
class ClassDecl(Binary): ...
class ClassExtDecl(Ternary): ...
class EmptyClassBody(Zeroary): ...
class NonemptyClassBody(Unary): ...
class ClassMemberSeq(Binary): ...
class ClassProperty(Unary): ...
class ClassPropertyInit(Binary): ...
class ClassMethod(Ternary): ...
class EmptyBlock(Zeroary): ...
class NonemptyBlock(Unary): ...
class IfThen(Binary): ...
class IfThenElse(Ternary): ...
class WhileStmt(Binary): ...
class ForHeader(Ternary): ...
class ForInitDecl(Binary): ...
class ForInitAssign(Binary): ...
class ForInitExpr(Unary): ...
class ForStmt(Binary): ...
class ForInStmt(Ternary): ...
class ForOfStmt(Ternary): ...
class DoWhileStmt(Binary): ...
class WithStmt(Binary): ...
class BreakStmt(Zeroary): ...
class ContinueStmt(Zeroary): ...
class ThrowStmt(Unary): ...
class TryCatch(Ternary): ...
class TryFinally(Binary): ...
class TryCatchFinally(Binary): ...
class VarDecl(Binary): ...
class LetDecl(Binary): ...
class AssignStmt(Binary): ...
class AssignAddStmt(Binary): ...
class AssignSubStmt(Binary): ...
class AssignMulStmt(Binary): ...
class AssignDivStmt(Binary): ...
class AssignModStmt(Binary): ...
class AssignBitOrStmt(Binary): ...
class AssignBitXorStmt(Binary): ...
class AssignBitAndStmt(Binary): ...
class AssignLShiftStmt(Binary): ...
class AssignRShiftStmt(Binary): ...
class AssignURShiftStmt(Binary): ...
class AssignPowerStmt(Binary): ...
class AssignLogicOrStmt(Binary): ...
class AssignLogicAndStmt(Binary): ...
class AssignNullishStmt(Binary): ...
class StmtSeq(Binary): ...
class Add(Binary): ...
class Sub(Binary): ...
class Mul(Binary): ...
class Div(Binary): ...
class Mod(Binary): ...
class Power(Binary): ...
class BitOr(Binary): ...
class BitXor(Binary): ...
class BitAnd(Binary): ...
class LShift(Binary): ...
class RShift(Binary): ...
class URShift(Binary): ...
class Eq(Binary): ...
class Neq(Binary): ...
class StrictEq(Binary): ...
class StrictNeq(Binary): ...
class Lt(Binary): ...
class Lte(Binary): ...
class Gt(Binary): ...
class Gte(Binary): ...
class InstanceOf(Binary): ...
class In(Binary): ...
class LogicOr(Binary): ...
class LogicAnd(Binary): ...
class NullishCoalesce(Binary): ...
class TernaryExpr(Ternary): ...
class SwitchStmt(Binary): ...
class CaseClause(Binary): ...
class DefaultCaseClause(Unary): ...
class CaseClauseSeq(Binary): ...
class MemberAccess(Binary): ...
class CallN(Binary): ...
class Args(Binary): ...
class IndexAccess(Binary): ...
class RegexLit(Unary): ...
class BitNot(Unary): ...
class VoidExpr(Unary): ...
class DeleteExpr(Unary): ...
class PreInc(Unary): ...
class PreDec(Unary): ...
class PostInc(Unary): ...
class PostDec(Unary): ...
class ArrowFunc1(Binary): ...
class ArrowFunc0(Unary): ...
class ArrowFuncN(Binary): ...
class ArrowBodyBlock(Unary): ...
class ArrowBodyExpr(Unary): ...
class NoParams(Zeroary): ...
class Params(Unary): ...
class Param(Unary): ...
class ParamSeq(Binary): ...

# Pruner
@rewrite
def _basic_js_pruner(t: TreeGrammar) -> TreeGrammar:
    """Conservative semantic pruning for JS codegen.

    Removes clearly undesirable branches while keeping broad solution space.
    """
    match t:
        case Union(children):
            return Union.of(_basic_js_pruner(c) for c in children)
        case ReturnVoidStmt():
            return EmptySet()
        case ThrowStmt(_):
            return EmptySet()
        case WithStmt(_, _):
            return EmptySet()
        case CallN(callee, args):
            # Avoid dangerous dynamic code execution patterns.
            callee_tree = as_tree(callee)
            match callee_tree:
                case Var(v):
                    tok = as_tree(v)
                    match tok:
                        case ASTLeaf(is_complete=True, prefix=prefix) if prefix in {"eval", "Function"}:
                            return EmptySet()
                        case _:
                            pass
                case _:
                    pass
            return CallN.of(
                _basic_js_pruner(callee),
                _basic_js_pruner(args),
                is_tree=t.is_tree,
            )
        case Application(children):
            return t.__class__.of(
                *(_basic_js_pruner(c) for c in children),
                is_tree=t.is_tree,
            )
        case _:
            return t


CONSTRUCTORS: list[type[Application]] = [
    Var,
    Num,
    Str,
    TemplateStr,
    TrueLit,
    FalseLit,
    NullLit,
    EmptyArray,
    Group,
    ReturnStmt,
    ReturnVoidStmt,
    ExprStmt,
    Neg,
    Not,
    UnaryPlus,
    TypeofExpr,
    AwaitExpr,
    Call0,
    ArrayLit,
    ThisLit,
    NewExpr,
    EmptyObject,
    ObjectLit,
    PropertyPair,
    PropertySeq,
    SpreadProperty,
    FunctionBodyClose,
    TopLevelSeq,
    ImportStmt,
    ImportFromStmt,
    ImportModuleStmt,
    ExportDefaultExpr,
    ExportDecl,
    ExportNamed,
    ExportNamedFrom,
    ImportName,
    ImportNamedSet,
    ImportSpec,
    ImportSpecAs,
    ImportSpecSeq,
    FunctionExpr,
    AsyncFunctionExpr,
    ClassExpr,
    FunctionDecl,
    AsyncFunctionDecl,
    ClassDecl,
    ClassExtDecl,
    EmptyClassBody,
    NonemptyClassBody,
    ClassMemberSeq,
    ClassProperty,
    ClassPropertyInit,
    ClassMethod,
    EmptyBlock,
    NonemptyBlock,
    IfThen,
    IfThenElse,
    WhileStmt,
    ForHeader,
    ForInitDecl,
    ForInitAssign,
    ForInitExpr,
    ForStmt,
    ForInStmt,
    ForOfStmt,
    DoWhileStmt,
    WithStmt,
    BreakStmt,
    ContinueStmt,
    ThrowStmt,
    TryCatch,
    TryFinally,
    TryCatchFinally,
    VarDecl,
    AssignStmt,
    AssignAddStmt,
    AssignSubStmt,
    AssignMulStmt,
    AssignDivStmt,
    AssignModStmt,
    AssignBitOrStmt,
    AssignBitXorStmt,
    AssignBitAndStmt,
    AssignLShiftStmt,
    AssignRShiftStmt,
    AssignURShiftStmt,
    AssignPowerStmt,
    AssignLogicOrStmt,
    AssignLogicAndStmt,
    AssignNullishStmt,
    StmtSeq,
    Add,
    Sub,
    Mul,
    Div,
    Mod,
    Power,
    BitOr,
    BitXor,
    BitAnd,
    LShift,
    RShift,
    URShift,
    Eq,
    Neq,
    StrictEq,
    StrictNeq,
    Lt,
    Lte,
    Gt,
    Gte,
    InstanceOf,
    In,
    LogicOr,
    LogicAnd,
    NullishCoalesce,
    TernaryExpr,
    SwitchStmt,
    CaseClause,
    DefaultCaseClause,
    CaseClauseSeq,
    MemberAccess,
    IndexAccess,
    CallN,
    Args,
    RegexLit,
    BitNot,
    VoidExpr,
    DeleteExpr,
    PreInc,
    PreDec,
    PostInc,
    PostDec,
    ArrowFunc1,
    ArrowFunc0,
    ArrowFuncN,
    ArrowBodyBlock,
    ArrowBodyExpr,
    NoParams,
    Params,
    Param,
    ParamSeq,
    LetDecl,
]


class ChopchopGenerator(BaseGenerator):
    def __init__(self, model_name: str, grammar: Optional[str], **kwargs):
        self.model_name = model_name
        self.context = kwargs.get("context", "You are a JavaScript coding assistant.")
        self.fixed_prefix = kwargs.get("fixed_prefix", "")
        self.pruner = kwargs.get("pruner", "basic")
        self.task_mode = kwargs.get("task_mode", "auto")
        self.brace_wait_tokens = int(kwargs.get("brace_wait_tokens", 64))

        grammar_source = self._load_grammar_source(grammar)
        lexer_spec, parser = parse_attribute_grammar(
            CONSTRUCTORS, grammar_source, "start"
        ).build_parser()

        if self.pruner in (None, "none", "identity"):
            pruner_fn = lambda asts: asts
        elif self.pruner == "basic":
            pruner_fn = _basic_js_pruner
        else:
            raise ValueError(f"Unsupported pruner mode: {self.pruner}")

        self.checker = RealizabilityChecker(pruner_fn, parser, lexer_spec)
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

    @staticmethod
    def _build_task_prompt(raw_prompt: str) -> str:
        lines = raw_prompt.splitlines()
        comments = [ln.strip()[2:].strip() for ln in lines if ln.strip().startswith("//")]
        signature = ""
        for ln in lines:
            stripped = ln.strip()
            if stripped.startswith("function "):
                signature = stripped
                break

        parts: list[str] = []
        if comments:
            parts.append("Task: " + " ".join(comments))
        if signature:
            parts.append("Function signature: " + signature)
        parts.append(
            "Output only JavaScript code that continues the given snippet."
        )
        parts.append("Do not write explanations, Markdown, bullets, or placeholders.")
        parts.append("Do not repeat comments or the function declaration.")
        return "\n".join(parts)

    @staticmethod
    def _finalize_completion_output(text: str) -> str:
        def _find_unclosed_string_start(s: str) -> int:
            state = None
            start_idx = -1
            escaped = False
            for i, ch in enumerate(s):
                if state is None:
                    if ch in ("'", '"', "`"):
                        state = ch
                        start_idx = i
                else:
                    if escaped:
                        escaped = False
                    elif ch == "\\":
                        escaped = True
                    elif ch == state:
                        state = None
                        start_idx = -1
            return start_idx

        def _trim_to_last_complete_statement(s: str) -> str:
            last_semi = s.rfind(";")
            if last_semi != -1:
                return s[: last_semi + 1].rstrip()

            # No complete statement yet: drop an incomplete trailing line.
            last_newline = s.rstrip().rfind("\n")
            if last_newline != -1:
                return s[:last_newline].rstrip()

            # Single unfinished line.
            return ""

        # 1. Drop duplicated function declarations (model sometimes repeats the prompt header).
        stripped = text.lstrip()
        if stripped.startswith("function "):
            # Remove the first line only; keep the rest as the function body.
            stripped_lines = stripped.splitlines()
            if stripped_lines:
                stripped = "\n".join(stripped_lines[1:])

        # 1.5 If generation ended inside a string/template literal, discard the incomplete tail.
        unclosed_at = _find_unclosed_string_start(stripped)
        if unclosed_at != -1:
            stripped = _trim_to_last_complete_statement(stripped[:unclosed_at])

        # 2. Heuristic formatting to fix "one-line" output issues.
        #    Insert newlines before common statement keywords if they follow a semicolon or brace.
        formatted = stripped
        keywords = ["if", "return", "let", "const", "var", "while", "for", "do", "switch", "try", "throw", "function", "class", "import", "export"]
        for kw in keywords:
            formatted = formatted.replace(f"; {kw}", f";\n{kw}")
            formatted = formatted.replace(f"}} {kw}", f"}}\n{kw}")
        
        # Also split after opening braces to avoid "{ if..."
        formatted = formatted.replace("{ ", "{\n")

        # 3. Brace counting to ensure function closure.
        #    The task prompt ends with "{" (opening the function body), so we need
        #    net_balance = (opens_in_gen + 1) - closes_in_gen <= 0.
        #    i.e., we need close_count >= open_count + 1.
        open_count = formatted.count("{")
        close_count = formatted.count("}")
        
        # Calculate how many closing braces are missing.
        # We start with 1 open brace (from prompt).
        needed_closes = (open_count + 1) - close_count
        
        if needed_closes > 0:
            formatted = formatted.rstrip()
            formatted += ("\n}" * needed_closes)
            
        return formatted

    def generate(self, prompt: str, stop_tokens=None, **kwargs) -> str:
        temperature = float(kwargs.get("temperature"))
        context = kwargs.get("context", self.context)
        fixed_prefix = kwargs.get("fixed_prefix", self.fixed_prefix)
        task_mode = kwargs.get("task_mode", self.task_mode)
        stop_tokens = kwargs.get("stop_tokens", stop_tokens) or []
        task_prompt = kwargs.get("task_prompt")
        if not task_prompt:
            task_prompt = self._build_task_prompt(prompt)

        stripped_prompt = prompt.rstrip()
        checker = self.checker

        use_close_checker = False
        if task_mode == "function_completion":
            use_close_checker = True
        elif task_mode == "auto" and stripped_prompt.endswith("{"):
            use_close_checker = True

        self.runner.require_trailing_brace = use_close_checker
        self.runner.brace_wait_tokens = self.brace_wait_tokens

        run_info = self.runner.run(
            Config(temperature=temperature),
            prompt=task_prompt,
            context=context,
            fixed_prefix=fixed_prefix,
            realizability_checker=checker,
        )

        output = run_info.output

        output = self._post_process_stop(output, stop_tokens)

        if use_close_checker:
            output = self._finalize_completion_output(output)

        return output

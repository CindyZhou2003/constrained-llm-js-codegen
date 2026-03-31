import gc
import sys
import time
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
from core.grammar import ASTLeaf, EmptySet, TreeGrammar, as_tree  # type: ignore  # noqa: E402
from core.lark.from_lark import parse_attribute_grammar  # type: ignore  # noqa: E402
from core.rewrite import rewrite, rewriter  # type: ignore  # noqa: E402
from llm.realizability import RealizabilityChecker  # type: ignore  # noqa: E402
from llm.run_llm import Config, LanguageModelRunner, ModelConfig  # type: ignore  # noqa: E402


class _SafeRealizabilityChecker(RealizabilityChecker):
    """
    Wraps the upstream RealizabilityChecker with safety guards:

      - RecursionError / NetworkXError guard (treat as realizable on error)
      - Per-call wall-clock timeout: if a single call exceeds `per_call_timeout`
        we cannot interrupt it (the rewriter uses global mutable state that is
        not thread-safe), but we track the time spent and switch to "bypass
        mode" for subsequent calls.
      - Cumulative budget: after `cumulative_budget` seconds have been spent
        in realizable() calls total, all further calls return True (skip
        checking) so that generation can finish within the overall timeout.

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

        # Clear the rewriter's global mutable state before each call to
        # prevent stale nodes from a previous (possibly slow) call causing
        # KeyError / NetworkXError in the dependency graph traversal.
        rewriter.clear()

        t0 = time.monotonic()
        try:
            result = super().realizable(prefix, final=final)
        except RecursionError:
            result = True
        except Exception:
            # NetworkXError or any other internal error — allow the token
            # rather than crashing the entire generation run.
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

    ChopChop's upstream runner assumes a specific HuggingFace model interface.
    This subclass re-implements the relevant hooks so we can:
      - Load models with our own device-map and dtype settings.
      - Support both chat-template models (e.g. Qwen Instruct) and plain
        causal-LM models (e.g. Phi-2) without modifying upstream code.
      - Suppress EOS too early in function-completion mode to give the model
        a chance to emit the closing brace before stopping.
    """

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
        """
        Check whether the tokens generated so far have closed the function body.

        Used by the EOS-suppression logic: when completing a function body we
        delay the end-of-sequence signal until the model has emitted enough
        closing braces to balance the function.  The prompt already supplies
        the opening '{', so the generated text must contain one more '}' than
        '{' for the function to be properly closed.

        Counting braces in the raw decoded text is an approximation (it
        does not account for braces inside string literals), but it is
        cheap and sufficient in practice: false negatives merely keep
        EOS suppression active a little longer, giving the model more
        runway to complete the function body.
        """
        if not generated_tokens:
            return False
        decoded = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
        open_count = decoded.count("{")
        close_count = decoded.count("}")
        # The prompt's opening '{' is not in decoded; we need one extra close.
        return close_count > open_count

    def _tokenize_prompt(
        self, prompt: str, context: str, fixed_prefix: str = ""
    ) -> torch.Tensor:
        """
        Convert the task prompt into input token IDs, handling two model families:

        - Chat-template models (e.g. Qwen2.5-Coder-Instruct): the tokenizer
          ships with a Jinja2 chat template, so we format the prompt as a
          system+user message pair and call apply_chat_template.  This inserts
          the correct special tokens (<|im_start|> etc.) automatically.

        - Plain causal-LM models (e.g. Phi-2): no chat template is present, so
          we concatenate the system context and user prompt as a single string.

        If a fixed_prefix is provided (used during the retry pass after a
        partial timeout), its tokens are appended directly to the prompt IDs so
        that generation continues from where it left off.
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
            # Append already-generated tokens so the model continues from a checkpoint.
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
        Generate the next single token, enforcing two real-time constraints:

        1. Grammar-driven token masking (forbidden_tokens / bad_words_ids)
           ChopChop's realizability checker returns a set of token IDs that
           would make the current prefix unrealizable according to the grammar.
           These are passed as bad_words_ids, causing the model's logits for
           those tokens to be set to -inf before sampling.  The result is that
           every token the model emits is guaranteed to keep the prefix within
           the grammar's language.

        2. EOS suppression in function-completion mode (require_trailing_brace)
           Small models tend to emit EOS before closing the function body.
           When require_trailing_brace is True, we suppress EOS for the first
           brace_wait_tokens steps *and* until at least one '}' has appeared.
           Once a closing brace is seen the model is free to stop normally.
        """
        bad_words = [[tok] for tok in forbidden_tokens] if forbidden_tokens else None

        # Concatenate the original prompt tokens with all tokens generated so far.
        inp = torch.tensor([list(input_ids[0]) + generated_tokens])
        inp = inp.to(self.model_config.device)
        attention_mask = torch.ones_like(inp, dtype=torch.long)

        # Respect ChopChop's grammar constraint: if EOS is forbidden by the
        # realizability checker at this position, mask it out entirely.
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


# Constructors matching generators/grammars/javascript_chopchop.lark
class Var(Unary): ...
class Num(Unary): ...
class Str(Unary): ...
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

# ---------------------------------------------------------------------------
# Grammar-level pruner
# ---------------------------------------------------------------------------
# The pruner runs *inside* ChopChop's realizability checker: before the
# checker evaluates whether a partial token prefix is extendable, it applies
# this function to prune away grammar branches that are technically reachable
# but undesirable.  Returning EmptySet() for a branch makes the checker treat
# it as unrealizable, so the model's logits for tokens that would lead into
# that branch are masked to -inf.
#
# Every rule below trades a small amount of generation flexibility for a
# meaningful reduction in checker cost or output quality improvement.
# ---------------------------------------------------------------------------
@rewrite
def _basic_js_pruner(t: TreeGrammar) -> TreeGrammar:
    """
    Basic pruner for JavaScript code generation:
    - Removes calls to eval and Function constructors, which can lead to unbounded recursion in realizability checking.
    - Removes trivial expression statements that don't contribute to meaningful code (e.g., "true;", "'hello';").
    """

    # ------------------------------------------------------------------
    # Rule 0 — Aggressive feature pruning
    # ------------------------------------------------------------------
    # These constructs almost never appear in simple function-body
    # completions (e.g. MBPP tasks) and their grammar branches add
    # significant cost to the realizability checker.  Removing them
    # reduces the branching factor at stmt (~17→10), primary_expr
    # (~15→12), and unary_expr (~11→8), which compounds at each
    # nesting level for a major speedup on deeply nested code.
    if isinstance(t, (
        # From stmt: class declarations, async function declarations,
        # try-catch blocks, with statements, switch, do-while
        ClassDecl, ClassExtDecl, AsyncFunctionDecl,
        TryCatch, TryFinally, TryCatchFinally, WithStmt,
        SwitchStmt, DoWhileStmt,
        # From primary_expr: class/function expressions, regex
        ClassExpr, FunctionExpr, AsyncFunctionExpr, RegexLit,
    )):
        return EmptySet()

    # ------------------------------------------------------------------
    # Rule 1 — Dangerous callee pruning: eval() and Function()
    # ------------------------------------------------------------------
    if isinstance(t, (Call0, CallN)):
        callee_tree = as_tree(t.children[0])
        if isinstance(callee_tree, Var):
            tok = as_tree(callee_tree.children[0])
            if isinstance(tok, ASTLeaf) and tok.is_complete:
                if tok.prefix in {"eval", "Function"}:
                    return EmptySet()

    # ------------------------------------------------------------------
    # Rule 2 — Unreachable-code pruning (terminator statements)
    # ------------------------------------------------------------------
    if isinstance(t, StmtSeq):
        current_stmt = t.children[0]
        if _is_terminator(current_stmt):
            return EmptySet()

    # ------------------------------------------------------------------
    # Rule 3 — Trivial expression-statement pruning
    # ------------------------------------------------------------------
    if isinstance(t, ExprStmt):
        expr_tree = as_tree(t.children[0])
        if isinstance(expr_tree, (TrueLit, FalseLit, NullLit,
                                  ThisLit, EmptyArray, Num)):
            return EmptySet()

    # ------------------------------------------------------------------
    # Rule 4 — Recursive propagation through compound nodes
    # ------------------------------------------------------------------
    # For any compound grammar node (binary, ternary, etc.) we recursively
    # prune each child subtree.  If *any* child becomes EmptySet (i.e. has
    # no valid completions after pruning) the entire parent node is also
    # pruned — there is no point keeping a node whose required sub-tree is
    # empty.  This propagates the effects of rules 1–4 upward through the
    # grammar tree automatically.
    if isinstance(t, Application):
        new_children = [_basic_js_pruner(c) for c in t.children]
        if any(isinstance(c, EmptySet) for c in new_children):
            return EmptySet()
        return t.__class__.of(
            *new_children,
            is_tree=t.is_tree,
        )

    return t

def _is_terminator(stmt_node) -> bool:
    """check if a statement is a terminator (return, throw, break, continue)"""
    n = as_tree(stmt_node)
    return isinstance(n, (ReturnStmt, ReturnVoidStmt, ThrowStmt, BreakStmt, ContinueStmt))

CONSTRUCTORS: list[type[Application]] = [
    Var,
    Num,
    Str,
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
]


class ChopchopGenerator(BaseGenerator):
    def __init__(self, model_name: str, grammar: Optional[str], **kwargs):
        self.model_name = model_name
        self.context = kwargs.get("context", "You are a JavaScript coding assistant.")
        self.fixed_prefix = kwargs.get("fixed_prefix", "")
        self.pruner = kwargs.get("pruner", "basic")
        self.task_mode = kwargs.get("task_mode", "auto")
        self.brace_wait_tokens = int(kwargs.get("brace_wait_tokens", 256))

        grammar_source = self._load_grammar_source(grammar)

        if self.pruner in (None, "none", "identity"):
            pruner_fn = lambda asts: asts
        elif self.pruner == "basic":
            pruner_fn = _basic_js_pruner
        else:
            raise ValueError(f"Unsupported pruner mode: {self.pruner}")

        # General-purpose parser (all start alternatives).
        lexer_spec, parser = parse_attribute_grammar(
            CONSTRUCTORS, grammar_source, "start"
        ).build_parser()
        self.checker = _SafeRealizabilityChecker(pruner_fn, parser, lexer_spec)

        # Function-body-only parser: start rule restricted to
        # function_body_with_close so the checker doesn't waste budget
        # exploring top_level_seq / stmt_seq alternatives that are
        # irrelevant when completing a function body.
        func_grammar = self._make_func_completion_grammar(grammar_source)
        func_lexer, func_parser = parse_attribute_grammar(
            CONSTRUCTORS, func_grammar, "start"
        ).build_parser()
        self.checker_func = _SafeRealizabilityChecker(
            pruner_fn, func_parser, func_lexer
        )

        self.runner = _LocalLanguageModelRunner(ModelConfig(model_id=model_name))

    @staticmethod
    def _make_func_completion_grammar(grammar_source: str) -> str:
        """Restrict the start rule to function_body_with_close only.

        This eliminates the top_level_seq and stmt_seq alternatives from the
        start symbol, so the realizability checker only explores grammar
        paths relevant to a function body.  The result is dramatically
        lower per-token checking cost (roughly 3x fewer top-level branches)
        and the grammar itself enforces that the closing '}' must be
        generated before EOS becomes realizable.
        """
        return grammar_source.replace(
            "start: top_level_seq\n        | stmt_seq\n        | function_body_with_close",
            "start: function_body_with_close",
        )

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
        """
        Construct a clean instruction prompt from the raw code snippet.

        The raw prompt is typically a JS comment describing the task followed
        by a partial function declaration, e.g.:

            // Write a function to count substrings ...
            function count_Substrings(s){

        We extract only the task description and function signature, then surround
        them with explicit output-format constraints.  This is necessary because:

        - Without the constraints, models tend to emit filler text like
          'Here in JavaScript' or repeat the comment verbatim.
        - Repeating the function signature wastes tokens and confuses the
          post-processor (which expects to receive only the body).

        Prompt-level constraints applied here
        (post-processing enforcement is in _finalize_completion_output):
          C1. Task + signature only — no full raw prompt leakage.
          C2. "Output only JavaScript code" — suppresses markdown fences,
              bullet points, and explanation paragraphs.
          C3. "Never output prose" — explicit ban on common placeholder
              phrases ("Here in JavaScript", "Solution:", "Explanation:").
          C4. "Do not repeat comments or the function declaration" — prevents
              the model from re-emitting the prompt header inside the body.
        """
        lines = raw_prompt.splitlines()
        # Extract every leading // comment as the task description.
        comments = [ln.strip()[2:].strip() for ln in lines if ln.strip().startswith("//")]
        # Extract the first `function` line as the signature (without the opening brace body).
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
        # C2: enforce code-only output.
        parts.append(
            "Output only JavaScript code that continues the given snippet."
        )
        # C3: ban common prose-placeholder phrases that small models tend to emit.
        parts.append("Do not write explanations, Markdown, bullets, or placeholders.")
        parts.append(
            "Never output prose like 'Here in JavaScript', 'Solution:', or 'Explanation:'."
        )
        # C4: prevent re-emission of the prompt header inside the body.
        parts.append("Do not repeat comments or the function declaration.")
        return "\n".join(parts)

    @staticmethod
    def _finalize_completion_output(text: str) -> str:
        """
        Post-process the raw model output into a well-formed JS function body.

        ChopChop guarantees that every emitted token is individually valid
        according to the grammar, but it cannot guarantee global properties
        like balanced braces or the absence of runaway loops (because those
        require unbounded lookahead).  This method applies a cascade of
        heuristic repair steps, ordered from most structural to most cosmetic:

          Step 1   — Remove a duplicated function declaration header.
          Step 2 — Trim text that ends inside an unclosed string literal.
          Step 3 — Trim text that ends with an unmatched '(' or '['.
          Step 4 — Truncate self-repetition loops.
          Step 5   — Insert newlines to un-collapse one-liner output.
          Step 6 — Drop natural-language placeholder lines.
          Step 7   — Balance braces so the function is syntactically closed.
        """

        # ------------------------------------------------------------------
        # Helper: detect natural-language placeholder lines (Step 6)
        # ------------------------------------------------------------------
        # Heuristic: a line is "natural language" if it contains no code
        # punctuation and consists of 3+ purely alphabetic whitespace-separated
        # words.  This catches phrases like "Here in JavaScript" or
        # "Solution goes here" that small models occasionally emit as filler.
        # False-positive risk is low because real JS always contains at least
        # one of: ; { } ( ) = . [ ] ' "
        def _looks_like_natural_language_line(line: str) -> bool:
            stripped_line = line.strip()
            if not stripped_line:
                return False
            if stripped_line.startswith("//"):
                return False

            # Keep obvious code-like lines.
            code_markers = [
                ";","{","}","(",")","=","=>",".","[","]","'",'"',
            ]
            if any(marker in stripped_line for marker in code_markers):
                return False

            tokens = stripped_line.split()
            if len(tokens) < 3:
                return False

            # Treat multi-word alphabetic lines as likely prose placeholders.
            return all(tok.replace("_", "").isalpha() for tok in tokens)

        # ------------------------------------------------------------------
        # Helper: find where generation ended inside an unclosed string (Step 2)
        # ------------------------------------------------------------------
        # Returns the character index of the opening quote of the last
        # unclosed string/template literal, or -1 if all strings are closed.
        # Generation can end mid-string when the safety timeout fires while
        # the model is in the middle of emitting a string argument.
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

        # ------------------------------------------------------------------
        # Helper: find the earliest unmatched '(' or '[' (Step 3)
        # ------------------------------------------------------------------
        # Returns the index of the leftmost unmatched opening paren/bracket
        # (ignoring characters inside string literals), or -1 if balanced.
        # Unmatched parens appear when the timeout fires mid-expression, e.g.
        # `map.set(sum, (` — the outer call's '(' and the inner '(' are both
        # unmatched.  We trim to just before the earliest one.
        def _find_unmatched_open_paren(s: str) -> int:
            str_state = None
            escaped = False
            paren_stack: list[int] = []
            bracket_stack: list[int] = []
            for i, ch in enumerate(s):
                if str_state is not None:
                    if escaped:
                        escaped = False
                    elif ch == "\\":
                        escaped = True
                    elif ch == str_state:
                        str_state = None
                    continue
                if ch in ("'", '"', "`"):
                    str_state = ch
                elif ch == "(":
                    paren_stack.append(i)
                elif ch == ")":
                    if paren_stack:
                        paren_stack.pop()
                elif ch == "[":
                    bracket_stack.append(i)
                elif ch == "]":
                    if bracket_stack:
                        bracket_stack.pop()
            unmatched = paren_stack + bracket_stack
            return min(unmatched) if unmatched else -1

        # ------------------------------------------------------------------
        # Helper: trim back to the last complete statement
        # ------------------------------------------------------------------
        # After detecting an unclosed string or unmatched paren we discard
        # everything from that point onward.  Then this helper finds the last
        # semicolon in what remains, so we keep only whole statements rather
        # than a trailing half-statement.
        def _trim_to_last_complete_statement(s: str) -> str:
            last_semi = s.rfind(";")
            if last_semi != -1:
                return s[: last_semi + 1].rstrip()

            # No complete statement yet: drop an incomplete trailing line.
            last_newline = s.rstrip().rfind("\n")
            if last_newline != -1:
                return s[:last_newline].rstrip()

            # Single unfinished line — nothing to salvage.
            return ""

        # ------------------------------------------------------------------
        # Step 1: Remove a duplicated function declaration header
        # ------------------------------------------------------------------
        # The model sometimes echoes the prompt's `function foo(...) {` line
        # at the start of its output.  Since the prompt already ends with that
        # line, including it again would produce a doubly-nested declaration.
        # We strip only the very first line when it starts with `function `.
        stripped = text.lstrip()
        if stripped.startswith("function "):
            stripped_lines = stripped.splitlines()
            if stripped_lines:
                stripped = "\n".join(stripped_lines[1:])

        # ------------------------------------------------------------------
        # Step 2: Trim text that ends inside an unclosed string literal
        # ------------------------------------------------------------------
        # If the safety timeout fired while the model was emitting a string
        # argument (e.g. `console.log('hell`) the tail is syntactically
        # invalid.  We locate the opening quote of the unclosed literal and
        # trim back to the last complete statement before it.
        unclosed_at = _find_unclosed_string_start(stripped)
        if unclosed_at != -1:
            stripped = _trim_to_last_complete_statement(stripped[:unclosed_at])

        # ------------------------------------------------------------------
        # Step 3: Trim text that ends with an unmatched '(' or '['
        # ------------------------------------------------------------------
        # Timeout mid-expression leaves dangling open parens/brackets, e.g.
        # `map.set(sum, (`.  These make the output unparseable.  We find the
        # earliest unmatched opener and trim back to the last ';' before it,
        # so the surrounding statements are preserved intact.
        unmatched_at = _find_unmatched_open_paren(stripped)
        if unmatched_at != -1:
            stripped = _trim_to_last_complete_statement(stripped[:unmatched_at])

        # Step 4: Detect and truncate self-repetition loops.
        #      Models sometimes get stuck regenerating the same lines inside a nested expression
        #      (e.g. emitting `function(name){ let x = 0; let y = 0; ...` that mirrors the outer
        #      body).  When two consecutive non-empty lines are *both* repeats of content that
        #      already appeared earlier in the output, we treat the earlier occurrence as the end
        #      of the valid generation and drop everything from the repetition point onwards.
        def _truncate_at_repetition(s: str) -> str:
            # Detect repetition loops: when two *consecutive* non-empty lines both
            # reproduce content that already appeared earlier in the output, the model
            # has most likely looped back into regenerating the function body.
            # We require TWO consecutive repeats before acting so that legitimate
            # nested-scope re-declarations (e.g. `let count = 0;` inside an inner
            # function) don't trigger a false truncation — in that case the repeated
            # line is followed by new content, which resets the counter.
            # When we do confirm a loop, we cut back to just before the FIRST repeated
            # line (not the second), so the looped content is fully removed.
            lines = s.splitlines(keepends=True)
            seen: set[str] = set()
            first_repeat_idx: int = -1
            prev_was_repeat = False
            for i, line in enumerate(lines):
                content = line.strip()
                if not content:
                    prev_was_repeat = False
                    first_repeat_idx = -1
                    continue
                if content in seen:
                    if prev_was_repeat:
                        # Confirmed loop: truncate before the first repeated line.
                        return "".join(lines[:first_repeat_idx]).rstrip()
                    first_repeat_idx = i
                    prev_was_repeat = True
                else:
                    seen.add(content)
                    prev_was_repeat = False
                    first_repeat_idx = -1
            return s

        stripped = _truncate_at_repetition(stripped)

        # ------------------------------------------------------------------
        # Step 5: Truncate at function body close
        # ------------------------------------------------------------------
        # When the realizability checker exhausts its time budget and switches
        # to bypass mode, the model may continue generating after the closing
        # '}' that completes the function body (e.g. chat-style text like
        # "Human\nExcellent! Can you explain...").  We detect the first
        # position where brace balance reaches -1 (one more '}' than '{',
        # accounting for the prompt's opening '{') and discard everything
        # after that closing brace.  Braces inside string literals are
        # ignored to avoid false positives.
        def _truncate_at_body_close(s: str) -> str:
            str_state = None
            escaped = False
            balance = 0  # net open braces in the generated text
            for i, ch in enumerate(s):
                if str_state is not None:
                    if escaped:
                        escaped = False
                    elif ch == "\\":
                        escaped = True
                    elif ch == str_state:
                        str_state = None
                    continue
                if ch in ("'", '"', "`"):
                    str_state = ch
                elif ch == "{":
                    balance += 1
                elif ch == "}":
                    balance -= 1
                    if balance < 0:
                        # This '}' closes the prompt's opening '{'.
                        return s[: i + 1]
            return s

        stripped = _truncate_at_body_close(stripped)

        # ------------------------------------------------------------------
        # Step 6: Drop natural-language placeholder lines
        # ------------------------------------------------------------------
        # Even after the prompt-level constraints (C2/C3 in _build_task_prompt)
        # some models leak prose lines like "Here in JavaScript" into the
        # output.  The _looks_like_natural_language_line heuristic catches
        # these: a line that contains no code-punctuation characters and is
        # made up of 3+ purely alphabetic words is treated as prose and removed.
        formatted = stripped
        filtered_lines = [
            ln for ln in formatted.splitlines() if not _looks_like_natural_language_line(ln)
        ]
        formatted = "\n".join(filtered_lines)

        # ------------------------------------------------------------------
        # Step 7: Balance braces to close the function body
        # ------------------------------------------------------------------
        # The original prompt ends with `{` (opening the function body), so
        # the generated text must supply at least one more `}` than it opens.
        # We count all `{` and `}` in the generated portion, then append the
        # missing closing braces.  Each missing brace is placed on its own
        # line for readability.
        #
        # Invariant: needed_closes = (open_in_gen + 1) - close_in_gen
        #   +1 accounts for the prompt's opening brace.
        open_count = formatted.count("{")
        close_count = formatted.count("}")
        needed_closes = (open_count + 1) - close_count

        if needed_closes > 0:
            formatted = formatted.rstrip()
            formatted += ("\n}" * needed_closes)

        return formatted

    def generate(self, prompt: str, stop_tokens=None, **kwargs) -> str:
        """
        Generate a single JavaScript completion for the given prompt.

        Overall flow
        ------------
        1. Build a task-focused instruction prompt from the raw code snippet.
        2. Determine whether we are in function-completion mode (i.e. the
           prompt ends with '{'), which enables EOS suppression and
           post-processing.
        3. Run ChopChop's grammar-constrained generation loop:
             - Each step, the realizability checker masks tokens that would
               make the prefix grammatically unrealizable.
             - The pruner (applied inside the checker) additionally masks
               tokens leading to pruned branches (eval, dead code, etc.).
             - Generation stops when EOS is emitted or safety_timeout expires.
        4. If the run timed out with partial output, retry once with a larger
           time budget, continuing from the partial output as a fixed prefix.
           This handles slow samples without blocking the whole batch.
        5. Apply _finalize_completion_output to repair common structural
           issues in the raw output (truncated expressions, brace imbalance,
           repetition loops, prose leakage).
        """
        temperature = float(kwargs.get("temperature"))
        context = kwargs.get("context", self.context)
        fixed_prefix = kwargs.get("fixed_prefix", self.fixed_prefix)
        task_mode = kwargs.get("task_mode", self.task_mode)
        # safety_timeout: per-sample wall-clock limit for the first generation pass.
        # retry_timeout: budget for the single retry pass after a partial timeout.
        safety_timeout = int(kwargs.get("safety_timeout", 300))
        retry_timeout = int(kwargs.get("retry_timeout", max(600, safety_timeout * 2)))
        stop_tokens = kwargs.get("stop_tokens", stop_tokens) or []
        task_prompt = kwargs.get("task_prompt")
        if not task_prompt:
            task_prompt = self._build_task_prompt(prompt)

        stripped_prompt = prompt.rstrip()

        # Detect function-completion mode: the prompt ends with '{' meaning the
        # model must close the function body.  This enables EOS suppression
        # (so the model doesn't stop before emitting '}') and post-processing.
        use_close_checker = False
        if task_mode == "function_completion":
            use_close_checker = True
        elif task_mode == "auto" and stripped_prompt.endswith("{"):
            use_close_checker = True

        # Use the function-body-only checker when completing a function body.
        # This restricts the grammar to function_body_with_close, which:
        #  - Forces the model to emit '}' before EOS is realizable.
        #  - Eliminates top_level_seq / stmt_seq exploration overhead.
        checker = self.checker_func if use_close_checker else self.checker
        checker.reset_budget()

        self.runner.require_trailing_brace = use_close_checker
        self.runner.brace_wait_tokens = self.brace_wait_tokens

        try:
            run_info = self.runner.run(
                Config(temperature=temperature, timeout=safety_timeout),
                prompt=task_prompt,
                context=context,
                fixed_prefix=fixed_prefix,
                realizability_checker=checker,
            )

            # Timeout-continuation: if function-completion timed out but produced
            # some output, try to extend it once with a larger budget.  We pass
            # the partial output as fixed_prefix so generation resumes exactly
            # where it left off (tokens are already realized, no re-checking).
            if use_close_checker and run_info.timed_out and run_info.output:
                retry_info = self.runner.run(
                    Config(temperature=temperature, timeout=retry_timeout),
                    prompt=task_prompt,
                    context=context,
                    fixed_prefix=run_info.output,
                    realizability_checker=checker,
                )
                if len(retry_info.output) > len(run_info.output):
                    run_info = retry_info
        except RecursionError:
            # Last-resort fallback: a RecursionError that escaped
            # _SafeRealizabilityChecker means the grammar hit a truly
            # pathological branch.  Return empty rather than crashing the batch.
            return ""

        output = run_info.output

        # Apply any caller-specified stop-token truncation (mirrors the
        # behavior of unconstrained/syncode generators for consistency).
        output = self._post_process_stop(output, stop_tokens)

        # Apply structural repair in function-completion mode.
        if use_close_checker:
            output = self._finalize_completion_output(output)

        return output
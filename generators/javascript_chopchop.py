"""
JavaScript language plugin for ChopchopGenerator.

Provides:
  - AST node classes matching generators/grammars/javascript_chopchop.lark
  - CONSTRUCTORS  — list passed to parse_attribute_grammar()
  - JS_START_RULE — grammar entry point
  - JSEnv, environment helpers, and pruner functions
  - make_js_pruner(mode)    — pruner factory
  - extract_js_prefix(prompt)  — extract function signature for fixed_prefix
  - build_js_prompt(prompt)    — build instruction prompt for the model
  - JS_CONTEXT               — default system context string
"""

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

# Make ChopChop package imports available.
CHOPCHOP_ROOT = Path(__file__).resolve().parent / "chopchop"
if str(CHOPCHOP_ROOT) not in sys.path:
    sys.path.insert(0, str(CHOPCHOP_ROOT))

from core.grammar import Application, Binary, Ternary, Unary, Zeroary  # type: ignore
from core.grammar import ASTLeaf, EmptySet, TreeGrammar, Union, as_tree  # type: ignore
from core.rewrite import rewrite  # type: ignore


# ---------------------------------------------------------------------------
# AST node classes — must match generators/grammars/javascript_chopchop.lark
# ---------------------------------------------------------------------------

class Var(Unary): ...
class Num(Unary): ...
class Str(Unary): ...
class TrueLit(Zeroary): ...
class FalseLit(Zeroary): ...
class NullLit(Zeroary): ...
class ThisLit(Zeroary): ...
class EmptyArray(Zeroary): ...
class ArrayLit(Unary): ...
class EmptyObject(Zeroary): ...
class ObjectLit(Unary): ...
class PropertyPair(Binary): ...
class PropertyStrPair(Binary): ...
class PropertySeq(Binary): ...
class Group(Unary): ...
class NewExpr(Unary): ...
class FunctionBodyClose(Unary): ...
class StmtSeq(Binary): ...
class ExprStmt(Unary): ...
class FunctionDecl(Ternary): ...
class FunctionExpr(Binary): ...
class EmptyBlock(Zeroary): ...
class NonemptyBlock(Unary): ...
class IfThen(Binary): ...
class IfThenElse(Ternary): ...
class WhileStmt(Binary): ...
class DoWhileStmt(Binary): ...
class ForStmt(Binary): ...
class ForInitLet(Binary): ...
class ForInitConst(Binary): ...
class ForInitVar(Binary): ...
class ForInitAssign(Binary): ...
class ForInitExpr(Unary): ...
class BreakStmt(Zeroary): ...
class ContinueStmt(Zeroary): ...
class ReturnStmt(Unary): ...
class ReturnVoidStmt(Zeroary): ...
class ThrowStmt(Unary): ...
class TryCatch(Ternary): ...
class LetDecl(Binary): ...
class ConstDecl(Binary): ...
class VarDecl(Binary): ...
class AssignExpr(Ternary): ...
class TernaryExpr(Ternary): ...
class LogicOr(Binary): ...
class LogicAnd(Binary): ...
class Eq(Binary): ...
class Neq(Binary): ...
class StrictEq(Binary): ...
class StrictNeq(Binary): ...
class Lt(Binary): ...
class Lte(Binary): ...
class Gt(Binary): ...
class Gte(Binary): ...
class InstanceOf(Binary): ...
class LShift(Binary): ...
class RShift(Binary): ...
class URShift(Binary): ...
class Add(Binary): ...
class Sub(Binary): ...
class Mul(Binary): ...
class Div(Binary): ...
class Mod(Binary): ...
class Power(Binary): ...
class Neg(Unary): ...
class UnaryPlus(Unary): ...
class Not(Unary): ...
class TypeofExpr(Unary): ...
class VoidExpr(Unary): ...
class PreInc(Unary): ...
class PreDec(Unary): ...
class MemberAccess(Binary): ...
class IndexAccess(Binary): ...
class Call0(Unary): ...
class CallN(Binary): ...
class PostInc(Unary): ...
class PostDec(Unary): ...
class Args(Binary): ...
class NoParams(Zeroary): ...
class Params(Unary): ...
class Param(Unary): ...
class ParamSeq(Binary): ...
class BitwiseAnd(Binary): ...
class BitwiseOr(Binary): ...
class BitwiseXor(Binary): ...
class BitwiseNot(Unary): ...
class EmptySwitch(Unary): ...
class SwitchStmt(Binary): ...
class SwitchStmtDefault(Ternary): ...
class SwitchStmtDefaultOnly(Binary): ...
class CaseClauses(Binary): ...
class EmptyCase(Unary): ...
class CaseClause(Binary): ...
class EmptyDefault(Zeroary): ...
class DefaultClause(Unary): ...
class ForInLetStmt(Ternary): ...
class ForInConstStmt(Ternary): ...
class ForInVarStmt(Ternary): ...
class ForOfLetStmt(Ternary): ...
class ForOfConstStmt(Ternary): ...
class ForOfVarStmt(Ternary): ...


CONSTRUCTORS: list[type[Application]] = [
    Var, Num, Str, TrueLit, FalseLit, NullLit, ThisLit,
    EmptyArray, ArrayLit, EmptyObject, ObjectLit,
    PropertyPair, PropertyStrPair, PropertySeq,
    Group, NewExpr,
    StmtSeq, ExprStmt,
    FunctionDecl, FunctionExpr,
    EmptyBlock, NonemptyBlock,
    IfThen, IfThenElse,
    WhileStmt, DoWhileStmt,
    ForStmt, ForInitLet, ForInitConst, ForInitVar, ForInitAssign, ForInitExpr,
    BreakStmt, ContinueStmt,
    ReturnStmt, ReturnVoidStmt, ThrowStmt, TryCatch,
    LetDecl, ConstDecl, VarDecl,
    AssignExpr, TernaryExpr,
    LogicOr, LogicAnd,
    Eq, Neq, StrictEq, StrictNeq,
    Lt, Lte, Gt, Gte, InstanceOf,
    LShift, RShift, URShift,
    Add, Sub, Mul, Div, Mod, Power,
    Neg, UnaryPlus, Not, TypeofExpr, VoidExpr,
    PreInc, PreDec,
    MemberAccess, IndexAccess,
    Call0, CallN,
    PostInc, PostDec,
    Args, NoParams, Params, Param, ParamSeq,
    BitwiseAnd, BitwiseOr, BitwiseXor, BitwiseNot,
    EmptySwitch, SwitchStmt, SwitchStmtDefault, SwitchStmtDefaultOnly,
    CaseClauses, EmptyCase, CaseClause, EmptyDefault, DefaultClause,
    ForInLetStmt, ForInConstStmt, ForInVarStmt,
    ForOfLetStmt, ForOfConstStmt, ForOfVarStmt,
]

JS_START_RULE = "function_declaration"

JS_CONTEXT = "You are a JavaScript coding assistant."


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

_GLOBAL_JS_NAMES: frozenset[str] = frozenset({
    "Math", "Array", "Object", "String", "Number", "Boolean",
    "console", "JSON", "parseInt", "parseFloat", "isNaN", "isFinite",
    "undefined", "Infinity", "NaN", "arguments",
})


@dataclass(frozen=True)
class JSEnv:
    """Tracks declared variable/function names and loop-nesting context."""
    names: frozenset[str]
    in_loop: bool

    def add(self, *new_names: str) -> "JSEnv":
        return JSEnv(self.names | frozenset(new_names), self.in_loop)

    def enter_loop(self) -> "JSEnv":
        return JSEnv(self.names, in_loop=True)


_default_js_env = JSEnv(_GLOBAL_JS_NAMES, in_loop=False)


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

def _is_terminator(stmt_node) -> bool:
    n = as_tree(stmt_node)
    return isinstance(n, (ReturnStmt, ReturnVoidStmt, ThrowStmt, BreakStmt, ContinueStmt))


def _var_name(tok_child) -> str | None:
    t = as_tree(tok_child)
    if isinstance(t, ASTLeaf) and t.is_complete:
        return t.prefix
    return None


def _lvalue_name(lvalue_tree) -> str | None:
    match lvalue_tree:
        case Var(tok):
            return _var_name(tok)
    return None


def _decl_name(stmt_tree) -> str | None:
    match stmt_tree:
        case LetDecl(lhs, _) | ConstDecl(lhs, _) | VarDecl(lhs, _):
            return _lvalue_name(as_tree(lhs))
        case ForInitLet(lhs, _) | ForInitConst(lhs, _) | ForInitVar(lhs, _):
            return _lvalue_name(as_tree(lhs))
        case FunctionDecl(name, _, _):
            return _var_name(name)
    return None


def _param_names(params_tree) -> frozenset[str]:
    match params_tree:
        case NoParams():
            return frozenset()
        case Params(param_list):
            return _collect_params(as_tree(param_list))
    return frozenset()


def _collect_params(node) -> frozenset[str]:
    match node:
        case Param(tok):
            name = _var_name(tok)
            return frozenset({name}) if name else frozenset()
        case ParamSeq(first_id, rest):
            first_name = _var_name(first_id)
            others = _collect_params(as_tree(rest))
            return (frozenset({first_name}) if first_name else frozenset()) | others
    return frozenset()


# ---------------------------------------------------------------------------
# Pruner functions
# ---------------------------------------------------------------------------

@rewrite
def js_prune_case_clauses(env: JSEnv, clauses: TreeGrammar) -> TreeGrammar:
    match clauses:
        case Union(children):
            return Union.of(js_prune_case_clauses(env, c) for c in children)
        case CaseClauses(head, tail):
            pruned_head = js_prune_case_clause(env, head)
            if isinstance(pruned_head, EmptySet):
                return EmptySet()
            return CaseClauses.of(pruned_head, js_prune_case_clauses(env, tail))
        case _:
            return js_prune_case_clause(env, clauses)


@rewrite
def js_prune_case_clause(env: JSEnv, clause: TreeGrammar) -> TreeGrammar:
    match clause:
        case Union(children):
            return Union.of(js_prune_case_clause(env, c) for c in children)
        case EmptyCase(expr):
            return EmptyCase.of(js_prune_expr(env, expr))
        case CaseClause(expr, stmts):
            pruned_expr = js_prune_expr(env, expr)
            if isinstance(pruned_expr, EmptySet):
                return EmptySet()
            return CaseClause.of(pruned_expr, js_prune_stmts(env, stmts))
        case _:
            return clause


@rewrite
def js_prune_default_clause(env: JSEnv, clause: TreeGrammar) -> TreeGrammar:
    match clause:
        case Union(children):
            return Union.of(js_prune_default_clause(env, c) for c in children)
        case EmptyDefault():
            return clause
        case DefaultClause(stmts):
            return DefaultClause.of(js_prune_stmts(env, stmts))
        case _:
            return clause


@rewrite
def js_prune_stmts(env: JSEnv, stmts: TreeGrammar) -> TreeGrammar:
    match stmts:
        case Union(children):
            return Union.of(js_prune_stmts(env, child) for child in children)
        case StmtSeq(head, tail):
            head_tree = as_tree(head)
            if head_tree is not None and _is_terminator(head_tree):
                return EmptySet()
            pruned_head = js_prune_stmt(env, head)
            if isinstance(pruned_head, EmptySet):
                return EmptySet()
            new_env = env
            if head_tree is not None:
                name = _decl_name(head_tree)
                if name:
                    new_env = env.add(name)
            pruned_tail = js_prune_stmts(new_env, tail)
            return StmtSeq.of(pruned_head, pruned_tail)
        case _:
            return js_prune_stmt(env, stmts)


@rewrite
def js_prune_stmt(env: JSEnv, stmt: TreeGrammar) -> TreeGrammar:
    match stmt:
        case Union(children):
            return Union.of(js_prune_stmt(env, child) for child in children)
        case EmptySet():
            return EmptySet()
        case DoWhileStmt() | TryCatch() | FunctionExpr():
            return EmptySet()
        case BreakStmt() | ContinueStmt():
            return stmt if env.in_loop else EmptySet()
        case LetDecl(lhs, rhs) | ConstDecl(lhs, rhs) | VarDecl(lhs, rhs):
            return stmt.__class__.of(lhs, js_prune_expr(env, rhs))
        case ExprStmt(expr):
            expr_tree = as_tree(expr)
            if isinstance(expr_tree, (TrueLit, FalseLit, NullLit,
                                      ThisLit, EmptyArray, Num, Str, Var)):
                return EmptySet()
            return ExprStmt.of(js_prune_expr(env, expr))
        case ReturnStmt(expr):
            return ReturnStmt.of(js_prune_expr(env, expr))
        case ReturnVoidStmt():
            return stmt
        case ThrowStmt(expr):
            return ThrowStmt.of(js_prune_expr(env, expr))
        case IfThen(cond, body):
            return IfThen.of(js_prune_expr(env, cond), js_prune_stmt(env, body))
        case IfThenElse(cond, then_b, else_b):
            return IfThenElse.of(
                js_prune_expr(env, cond),
                js_prune_stmt(env, then_b),
                js_prune_stmt(env, else_b),
            )
        case WhileStmt(cond, body):
            return WhileStmt.of(
                js_prune_expr(env, cond),
                js_prune_stmt(env.enter_loop(), body),
            )
        case ForStmt(for_init, body):
            loop_env = env.enter_loop()
            init_tree = as_tree(for_init)
            if init_tree is not None:
                name = _decl_name(init_tree)
                if name:
                    loop_env = loop_env.add(name)
            pruned_init = js_prune_stmt(env, for_init)
            if isinstance(pruned_init, EmptySet):
                return EmptySet()
            return ForStmt.of(pruned_init, js_prune_stmt(loop_env, body))
        case (ForInLetStmt(ident, iterable, body)
              | ForInConstStmt(ident, iterable, body)
              | ForInVarStmt(ident, iterable, body)
              | ForOfLetStmt(ident, iterable, body)
              | ForOfConstStmt(ident, iterable, body)
              | ForOfVarStmt(ident, iterable, body)):
            loop_env = env.enter_loop()
            id_name = _var_name(as_tree(ident))
            if id_name:
                loop_env = loop_env.add(id_name)
            return stmt.__class__.of(
                ident,
                js_prune_expr(env, iterable),
                js_prune_stmt(loop_env, body),
            )
        case EmptySwitch(expr):
            return EmptySwitch.of(js_prune_expr(env, expr))
        case SwitchStmt(expr, cases):
            pruned_cases = js_prune_case_clauses(env, cases)
            if isinstance(pruned_cases, EmptySet):
                return EmptySet()
            return SwitchStmt.of(js_prune_expr(env, expr), pruned_cases)
        case SwitchStmtDefault(expr, cases, default):
            pruned_cases = js_prune_case_clauses(env, cases)
            pruned_default = js_prune_default_clause(env, default)
            if isinstance(pruned_cases, EmptySet) or isinstance(pruned_default, EmptySet):
                return EmptySet()
            return SwitchStmtDefault.of(
                js_prune_expr(env, expr), pruned_cases, pruned_default
            )
        case SwitchStmtDefaultOnly(expr, default):
            pruned_default = js_prune_default_clause(env, default)
            if isinstance(pruned_default, EmptySet):
                return EmptySet()
            return SwitchStmtDefaultOnly.of(js_prune_expr(env, expr), pruned_default)
        case EmptyBlock():
            return stmt
        case NonemptyBlock(stmts):
            return NonemptyBlock.of(js_prune_stmts(env, stmts))
        case FunctionDecl(name, params, body):
            name_tree = as_tree(name)
            params_tree = as_tree(params)
            inner_env = env
            if name_tree is not None:
                func_name = _var_name(name_tree)
                if func_name:
                    inner_env = inner_env.add(func_name)
            if params_tree is not None:
                inner_env = inner_env.add(*_param_names(params_tree))
            return FunctionDecl.of(name, params, js_prune_stmt(inner_env, body))
        case _:
            if isinstance(stmt, Application):
                new_children = [js_prune_expr(env, c) for c in stmt.children]
                if any(isinstance(c, EmptySet) for c in new_children):
                    return EmptySet()
                return stmt.__class__.of(*new_children, is_tree=stmt.is_tree)
            return stmt


@rewrite
def js_prune_expr(env: JSEnv, expr: TreeGrammar) -> TreeGrammar:
    match expr:
        case Union(children):
            return Union.of(js_prune_expr(env, child) for child in children)
        case EmptySet():
            return EmptySet()
        case Var(tok):
            name = _var_name(tok)
            if name is not None and name not in env.names:
                return EmptySet()
            return expr
        case FunctionExpr():
            return EmptySet()
        case Call0(callee) | CallN(callee, _):
            callee_tree = as_tree(callee)
            if isinstance(callee_tree, Var):
                name = _var_name(callee_tree.children[0])
                if name in {"eval", "Function"}:
                    return EmptySet()
            new_children = [js_prune_expr(env, c) for c in expr.children]
            if any(isinstance(c, EmptySet) for c in new_children):
                return EmptySet()
            return expr.__class__.of(*new_children, is_tree=expr.is_tree)
        case MemberAccess(obj, prop):
            pruned_obj = js_prune_expr(env, obj)
            if isinstance(pruned_obj, EmptySet):
                return EmptySet()
            return MemberAccess.of(pruned_obj, prop)
        case (TrueLit() | FalseLit() | NullLit() | ThisLit()
              | EmptyArray() | Num(_) | Str(_)):
            return expr
        case _:
            if isinstance(expr, Application):
                new_children = [js_prune_expr(env, c) for c in expr.children]
                if any(isinstance(c, EmptySet) for c in new_children):
                    return EmptySet()
                return expr.__class__.of(*new_children, is_tree=expr.is_tree)
            return expr


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def make_js_pruner(mode: Optional[str] = "none"):
    """
    Return a pruner callable for use with ChopchopGenerator.

    Modes:
      "none" / "identity" / None — identity (no pruning beyond grammar)
      "basic"                    — env-aware scope and feature pruning
    """
    if mode in (None, "none", "identity"):
        return lambda asts: asts
    elif mode == "basic":
        return lambda asts: js_prune_stmt(_default_js_env, asts)
    else:
        raise ValueError(f"Unknown JS pruner mode: {mode!r}. Use 'none' or 'basic'.")


def extract_js_prefix(raw_prompt: str) -> str:
    """
    Return the function signature line with the trailing '{' stripped,
    to be used as fixed_prefix for ChopchopGenerator.

    Example input line:  "function count_Substrings(s) {"
    Example return value: "function count_Substrings(s)"
    """
    for ln in raw_prompt.splitlines():
        stripped = ln.strip()
        if stripped.startswith("function "):
            return stripped.rstrip("{").rstrip()
    return ""


def build_js_prompt(raw_prompt: str) -> str:
    """
    Construct a clean instruction prompt from the raw code snippet.

    The signature is pre-seeded via fixed_prefix so it is omitted here
    to prevent the model from repeating it at the start of its response.
    """
    lines = raw_prompt.splitlines()
    comments = [ln.strip()[2:].strip() for ln in lines if ln.strip().startswith("//")]

    parts: list[str] = []
    if comments:
        parts.append("Task: " + " ".join(comments))
    parts.append("Output the complete function body enclosed in curly braces, e.g. { ... }.")
    parts.append("Do not write explanations, Markdown, bullets, or placeholders.")
    parts.append(
        "Never output prose like 'Here in JavaScript', 'Solution:', or 'Explanation:'."
    )
    parts.append("Do not repeat comments or the function declaration.")
    return "\n".join(parts)

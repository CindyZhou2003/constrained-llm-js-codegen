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

# --- Primary / Literals ------------------------------------------------
class Var(Unary): ...
class Num(Unary): ...
class Str(Unary): ...
class TemplateLit(Unary): ...
class RegexLit(Unary): ...
class TrueLit(Zeroary): ...
class FalseLit(Zeroary): ...
class NullLit(Zeroary): ...
class UndefinedLit(Zeroary): ...
class ThisLit(Zeroary): ...
class SuperExpr(Zeroary): ...

# --- Arrays / Objects --------------------------------------------------
class EmptyArray(Zeroary): ...
class ArrayLit(Unary): ...
class ArrayItemSeq(Binary): ...
class SpreadElement(Unary): ...
class EmptyObject(Zeroary): ...
class ObjectLit(Unary): ...
class PropertyPair(Binary): ...
class PropertyStrPair(Binary): ...
class ComputedPropertyPair(Binary): ...
class ShorthandProperty(Unary): ...
class SpreadProperty(Unary): ...
class MethodProperty(Ternary): ...
class AsyncMethodProperty(Ternary): ...
class PropertySeq(Binary): ...

# --- Grouping / New ----------------------------------------------------
class Group(Unary): ...
class NewExpr(Unary): ...

# --- Statements --------------------------------------------------------
class StmtSeq(Binary): ...
class ExprStmt(Unary): ...

# --- Functions (sync, async, generator) --------------------------------
class FunctionDecl(Ternary): ...
class AsyncFunctionDecl(Ternary): ...
class GeneratorDecl(Ternary): ...
class AsyncGeneratorDecl(Ternary): ...
class FunctionExpr(Binary): ...
class AsyncFunctionExpr(Binary): ...
class GeneratorExpr(Binary): ...
class AsyncGeneratorExpr(Binary): ...

# --- Arrow Expressions -------------------------------------------------
class ArrowExprIdent(Binary): ...
class ArrowExprNoParams(Unary): ...
class ArrowExprParams(Binary): ...
class AsyncArrowIdent(Binary): ...
class AsyncArrowNoParams(Unary): ...
class AsyncArrowParams(Binary): ...

# --- Classes -----------------------------------------------------------
class ClassDecl(Binary): ...
class ClassDeclExtends(Ternary): ...
class ClassExpr(Unary): ...
class ClassExprExtends(Binary): ...
class EmptyClassBody(Zeroary): ...
class ClassBody(Unary): ...
class ClassMembers(Binary): ...
class ClassMethod(Ternary): ...
class StaticMethod(Ternary): ...
class AsyncClassMethod(Ternary): ...
class ClassField(Binary): ...
class StaticField(Binary): ...

# --- Import / Export ---------------------------------------------------
class ImportBare(Unary): ...
class ImportDefault(Binary): ...
class ImportNamed(Binary): ...
class ImportDefaultNamed(Ternary): ...
class ImportSpecSeq(Binary): ...
class ImportSpec(Unary): ...
class ImportAlias(Binary): ...
class ExportDefaultExpr(Unary): ...
class ExportDecl(Unary): ...
class ExportFuncDecl(Unary): ...
class ExportClassDecl(Unary): ...
class ExportNamed(Unary): ...
class ExportNamedFrom(Binary): ...
class ExportSpecSeq(Binary): ...
class ExportSpec(Unary): ...
class ExportAlias(Binary): ...

# --- Blocks / Control --------------------------------------------------
class EmptyBlock(Zeroary): ...
class NonemptyBlock(Unary): ...
class IfThen(Binary): ...
class IfThenElse(Ternary): ...
class WhileStmt(Binary): ...
class DoWhileStmt(Binary): ...
class WithStmt(Binary): ...
@dataclass(frozen=True)
class ForStmt(Application):
    init: TreeGrammar
    condition: TreeGrammar
    update: TreeGrammar
    body: TreeGrammar
class ForInitLet(Binary): ...
class ForInitConst(Binary): ...
class ForInitVar(Binary): ...
class ForInitAssign(Binary): ...
class ForInitExpr(Unary): ...
class ForInitLetNoInit(Unary): ...
class ForInitConstNoInit(Unary): ...
class ForInitVarNoInit(Unary): ...
class BreakStmt(Zeroary): ...
class ContinueStmt(Zeroary): ...
class ReturnStmt(Unary): ...
class ReturnVoidStmt(Zeroary): ...
class ThrowStmt(Unary): ...
class TryCatch(Ternary): ...
class TryCatchNoParam(Binary): ...
@dataclass(frozen=True)
class TryCatchFinally(Application):
    try_body: TreeGrammar
    catch_param: TreeGrammar
    catch_body: TreeGrammar
    finally_body: TreeGrammar
class TryCatchNoParamFinally(Ternary): ...
class TryFinally(Binary): ...

# --- Declarations ------------------------------------------------------
class LetDecl(Binary): ...
class ConstDecl(Binary): ...
class VarDecl(Binary): ...
class LetDeclNoInit(Unary): ...
class ConstDeclNoInit(Unary): ...
class VarDeclNoInit(Unary): ...

# --- Expressions: assignment / ternary ---------------------------------
class AssignExpr(Ternary): ...
class TernaryExpr(Ternary): ...

# --- Logical / Bitwise -------------------------------------------------
class LogicOr(Binary): ...
class LogicAnd(Binary): ...
class NullishCoalesce(Binary): ...
class BitwiseOr(Binary): ...
class BitwiseXor(Binary): ...
class BitwiseAnd(Binary): ...

# --- Comparison --------------------------------------------------------
class Eq(Binary): ...
class Neq(Binary): ...
class StrictEq(Binary): ...
class StrictNeq(Binary): ...
class Lt(Binary): ...
class Lte(Binary): ...
class Gt(Binary): ...
class Gte(Binary): ...
class InstanceOf(Binary): ...
class InExpr(Binary): ...

# --- Shift -------------------------------------------------------------
class LShift(Binary): ...
class RShift(Binary): ...
class URShift(Binary): ...

# --- Arithmetic --------------------------------------------------------
class Add(Binary): ...
class Sub(Binary): ...
class Mul(Binary): ...
class Div(Binary): ...
class Mod(Binary): ...
class Power(Binary): ...

# --- Unary -------------------------------------------------------------
class Neg(Unary): ...
class UnaryPlus(Unary): ...
class Not(Unary): ...
class BitwiseNot(Unary): ...
class TypeofExpr(Unary): ...
class VoidExpr(Unary): ...
class DeleteExpr(Unary): ...
class AwaitExpr(Unary): ...
class YieldExpr(Unary): ...
class YieldStarExpr(Unary): ...
class PreInc(Unary): ...
class PreDec(Unary): ...

# --- Member / Call / Postfix -------------------------------------------
class MemberAccess(Binary): ...
class OptionalChain(Binary): ...
class IndexAccess(Binary): ...
class OptionalIndexAccess(Binary): ...
class Call0(Unary): ...
class CallN(Binary): ...
class OptionalCall0(Unary): ...
class OptionalCallN(Binary): ...
class PostInc(Unary): ...
class PostDec(Unary): ...
class Args(Binary): ...
class SpreadArg(Unary): ...

# --- Parameters --------------------------------------------------------
class NoParams(Zeroary): ...
class Params(Unary): ...
class Param(Unary): ...
class ParamSeq(Binary): ...
class DefaultParam(Binary): ...
class RestParam(Unary): ...

# --- Switch ------------------------------------------------------------
class EmptySwitch(Unary): ...
class SwitchStmt(Binary): ...
class SwitchStmtDefault(Ternary): ...
class SwitchStmtDefaultOnly(Binary): ...
class CaseClauses(Binary): ...
class EmptyCase(Unary): ...
class CaseClause(Binary): ...
class EmptyDefault(Zeroary): ...
class DefaultClause(Unary): ...

# --- For-in / For-of ---------------------------------------------------
class ForInLetStmt(Ternary): ...
class ForInConstStmt(Ternary): ...
class ForInVarStmt(Ternary): ...
class ForOfLetStmt(Ternary): ...
class ForOfConstStmt(Ternary): ...
class ForOfVarStmt(Ternary): ...


CONSTRUCTORS: list[type[Application]] = [
    # Primary / Literals
    Var, Num, Str, TemplateLit, RegexLit,
    TrueLit, FalseLit, NullLit, UndefinedLit, ThisLit, SuperExpr,
    # Arrays / Objects
    EmptyArray, ArrayLit, ArrayItemSeq, SpreadElement,
    EmptyObject, ObjectLit,
    PropertyPair, PropertyStrPair, ComputedPropertyPair,
    ShorthandProperty, SpreadProperty, MethodProperty, AsyncMethodProperty,
    PropertySeq,
    # Grouping / New
    Group, NewExpr,
    # Statements
    StmtSeq, ExprStmt,
    # Functions (sync, async, generator)
    FunctionDecl, AsyncFunctionDecl, GeneratorDecl, AsyncGeneratorDecl,
    FunctionExpr, AsyncFunctionExpr, GeneratorExpr, AsyncGeneratorExpr,
    # Arrow expressions
    ArrowExprIdent, ArrowExprNoParams, ArrowExprParams,
    AsyncArrowIdent, AsyncArrowNoParams, AsyncArrowParams,
    # Classes
    ClassDecl, ClassDeclExtends, ClassExpr, ClassExprExtends,
    EmptyClassBody, ClassBody, ClassMembers,
    ClassMethod, StaticMethod, AsyncClassMethod, ClassField, StaticField,
    # Import / Export
    ImportBare, ImportDefault, ImportNamed, ImportDefaultNamed,
    ImportSpecSeq, ImportSpec, ImportAlias,
    ExportDefaultExpr, ExportDecl, ExportFuncDecl, ExportClassDecl,
    ExportNamed, ExportNamedFrom, ExportSpecSeq, ExportSpec, ExportAlias,
    # Blocks / Control
    EmptyBlock, NonemptyBlock,
    IfThen, IfThenElse,
    WhileStmt, DoWhileStmt, WithStmt,
    ForStmt, ForInitLet, ForInitConst, ForInitVar, ForInitAssign, ForInitExpr,
    ForInitLetNoInit, ForInitConstNoInit, ForInitVarNoInit,
    BreakStmt, ContinueStmt,
    ReturnStmt, ReturnVoidStmt, ThrowStmt,
    TryCatch, TryCatchNoParam, TryCatchFinally, TryCatchNoParamFinally, TryFinally,
    LetDecl, ConstDecl, VarDecl, LetDeclNoInit, ConstDeclNoInit, VarDeclNoInit,
    # Expressions
    AssignExpr, TernaryExpr,
    LogicOr, LogicAnd, NullishCoalesce,
    BitwiseOr, BitwiseXor, BitwiseAnd,
    Eq, Neq, StrictEq, StrictNeq,
    Lt, Lte, Gt, Gte, InstanceOf, InExpr,
    LShift, RShift, URShift,
    Add, Sub, Mul, Div, Mod, Power,
    Neg, UnaryPlus, Not, BitwiseNot,
    TypeofExpr, VoidExpr, DeleteExpr, AwaitExpr,
    YieldExpr, YieldStarExpr,
    PreInc, PreDec,
    MemberAccess, OptionalChain, IndexAccess, OptionalIndexAccess,
    Call0, CallN, OptionalCall0, OptionalCallN,
    PostInc, PostDec,
    Args, SpreadArg,
    NoParams, Params, Param, ParamSeq, DefaultParam, RestParam,
    # Switch
    EmptySwitch, SwitchStmt, SwitchStmtDefault, SwitchStmtDefaultOnly,
    CaseClauses, EmptyCase, CaseClause, EmptyDefault, DefaultClause,
    # For-in / For-of
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
    "Date", "RegExp", "Error", "TypeError", "RangeError", "SyntaxError",
    "Map", "Set", "WeakMap", "WeakSet", "Promise", "Symbol", "Proxy",
    "console", "JSON", "parseInt", "parseFloat", "isNaN", "isFinite",
    "undefined", "Infinity", "NaN", "arguments",
    "encodeURIComponent", "decodeURIComponent", "encodeURI", "decodeURI",
    "setTimeout", "setInterval", "clearTimeout", "clearInterval",
    "globalThis", "require", "module", "exports",
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
        case LetDeclNoInit(lhs) | ConstDeclNoInit(lhs) | VarDeclNoInit(lhs):
            return _lvalue_name(as_tree(lhs))
        case ForInitLet(lhs, _) | ForInitConst(lhs, _) | ForInitVar(lhs, _):
            return _lvalue_name(as_tree(lhs))
        case ForInitLetNoInit(lhs) | ForInitConstNoInit(lhs) | ForInitVarNoInit(lhs):
            return _lvalue_name(as_tree(lhs))
        case (FunctionDecl(name, _, _)
              | AsyncFunctionDecl(name, _, _)
              | GeneratorDecl(name, _, _)
              | AsyncGeneratorDecl(name, _, _)):
            return _var_name(name)
        case ClassDecl(name, _) | ClassDeclExtends(name, _, _):
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
            inner = as_tree(tok)
            return _extract_single_param(inner) if inner is not None else _extract_single_param(tok)
        case ParamSeq(first_id, rest):
            inner = as_tree(first_id)
            first = _extract_single_param(inner) if inner is not None else _extract_single_param(first_id)
            return first | _collect_params(as_tree(rest))
        case DefaultParam(tok, _):
            name = _var_name(tok)
            return frozenset({name}) if name else frozenset()
        case RestParam(tok):
            name = _var_name(tok)
            return frozenset({name}) if name else frozenset()
    return frozenset()


def _extract_single_param(node) -> frozenset[str]:
    """Extract the bound name from a single parameter (plain, default, or rest)."""
    match node:
        case DefaultParam(tok, _):
            name = _var_name(tok)
            return frozenset({name}) if name else frozenset()
        case RestParam(tok):
            name = _var_name(tok)
            return frozenset({name}) if name else frozenset()
        case _:
            name = _var_name(node)
            return frozenset({name}) if name else frozenset()


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
        case BreakStmt() | ContinueStmt():
            return stmt if env.in_loop else EmptySet()
        case LetDecl(lhs, rhs) | ConstDecl(lhs, rhs) | VarDecl(lhs, rhs):
            return stmt.__class__.of(lhs, js_prune_expr(env, rhs))
        case LetDeclNoInit(_) | ConstDeclNoInit(_) | VarDeclNoInit(_):
            return stmt
        case ExprStmt(expr):
            return ExprStmt.of(js_prune_expr(env, expr))
        case ReturnStmt(expr):
            return ReturnStmt.of(js_prune_expr(env, expr))
        case ReturnVoidStmt():
            return stmt
        case ThrowStmt(expr):
            return ThrowStmt.of(js_prune_expr(env, expr))
        case TryCatch(try_body, catch_param, catch_body):
            catch_env = env
            catch_name = _var_name(catch_param)
            if catch_name:
                catch_env = env.add(catch_name)
            return TryCatch.of(
                js_prune_stmt(env, try_body),
                catch_param,
                js_prune_stmt(catch_env, catch_body),
            )
        case TryCatchNoParam(try_body, catch_body):
            return TryCatchNoParam.of(
                js_prune_stmt(env, try_body),
                js_prune_stmt(env, catch_body),
            )
        case TryCatchFinally(try_body, catch_param, catch_body, finally_body):
            catch_env = env
            catch_name = _var_name(catch_param)
            if catch_name:
                catch_env = env.add(catch_name)
            return TryCatchFinally.of(
                js_prune_stmt(env, try_body),
                catch_param,
                js_prune_stmt(catch_env, catch_body),
                js_prune_stmt(env, finally_body),
            )
        case TryCatchNoParamFinally(try_body, catch_body, finally_body):
            return TryCatchNoParamFinally.of(
                js_prune_stmt(env, try_body),
                js_prune_stmt(env, catch_body),
                js_prune_stmt(env, finally_body),
            )
        case TryFinally(try_body, finally_body):
            return TryFinally.of(
                js_prune_stmt(env, try_body),
                js_prune_stmt(env, finally_body),
            )
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
        case DoWhileStmt(body, cond):
            return DoWhileStmt.of(
                js_prune_stmt(env.enter_loop(), body),
                js_prune_expr(env, cond),
            )
        case WithStmt(expr, body):
            return WithStmt.of(js_prune_expr(env, expr), js_prune_stmt(env, body))
        case ForStmt(for_init, cond, update, body):
            loop_env = env.enter_loop()
            init_tree = as_tree(for_init)
            if init_tree is not None:
                name = _decl_name(init_tree)
                if name:
                    loop_env = loop_env.add(name)
            pruned_init = js_prune_stmt(env, for_init)
            if isinstance(pruned_init, EmptySet):
                return EmptySet()
            return ForStmt.of(
                pruned_init,
                js_prune_expr(loop_env, cond),
                js_prune_expr(loop_env, update),
                js_prune_stmt(loop_env, body),
            )
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
        case (FunctionDecl(name, params, body)
              | AsyncFunctionDecl(name, params, body)
              | GeneratorDecl(name, params, body)
              | AsyncGeneratorDecl(name, params, body)):
            name_tree = as_tree(name)
            params_tree = as_tree(params)
            inner_env = env
            if name_tree is not None:
                func_name = _var_name(name_tree)
                if func_name:
                    inner_env = inner_env.add(func_name)
            if params_tree is not None:
                inner_env = inner_env.add(*_param_names(params_tree))
            return stmt.__class__.of(name, params, js_prune_stmt(inner_env, body))
        case ClassDecl(name, body):
            return ClassDecl.of(name, js_prune_stmt(env, body))
        case ClassDeclExtends(name, superclass, body):
            return ClassDeclExtends.of(name, js_prune_expr(env, superclass), js_prune_stmt(env, body))
        case _:
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
        case (FunctionExpr(params, body)
              | AsyncFunctionExpr(params, body)
              | GeneratorExpr(params, body)
              | AsyncGeneratorExpr(params, body)):
            params_tree = as_tree(params)
            inner_env = env
            if params_tree is not None:
                inner_env = inner_env.add(*_param_names(params_tree))
            return expr.__class__.of(params, js_prune_stmt(inner_env, body))
        case ArrowExprIdent(ident, body) | AsyncArrowIdent(ident, body):
            inner_env = env
            id_name = _var_name(ident)
            if id_name:
                inner_env = inner_env.add(id_name)
            return expr.__class__.of(ident, js_prune_expr(inner_env, body))
        case ArrowExprNoParams(body) | AsyncArrowNoParams(body):
            return expr.__class__.of(js_prune_expr(env, body))
        case ArrowExprParams(params, body) | AsyncArrowParams(params, body):
            params_tree = as_tree(params)
            inner_env = env
            if params_tree is not None:
                inner_env = inner_env.add(*_param_names(params_tree))
            return expr.__class__.of(params, js_prune_expr(inner_env, body))
        case ClassExpr(body):
            return ClassExpr.of(js_prune_stmt(env, body))
        case ClassExprExtends(superclass, body):
            return ClassExprExtends.of(js_prune_expr(env, superclass), js_prune_stmt(env, body))
        case Call0(callee) | CallN(callee, _) | OptionalCall0(callee) | OptionalCallN(callee, _):
            callee_tree = as_tree(callee)
            if isinstance(callee_tree, Var):
                name = _var_name(callee_tree.children[0])
                if name in {"eval", "Function"}:
                    return EmptySet()
            new_children = [js_prune_expr(env, c) for c in expr.children]
            if any(isinstance(c, EmptySet) for c in new_children):
                return EmptySet()
            return expr.__class__.of(*new_children, is_tree=expr.is_tree)
        case (MemberAccess(obj, prop)
              | OptionalChain(obj, prop)
              | IndexAccess(obj, prop)
              | OptionalIndexAccess(obj, prop)):
            pruned_obj = js_prune_expr(env, obj)
            if isinstance(pruned_obj, EmptySet):
                return EmptySet()
            return expr.__class__.of(pruned_obj, prop)
        case (TrueLit() | FalseLit() | NullLit() | UndefinedLit()
              | ThisLit() | SuperExpr()
              | EmptyArray() | EmptyObject()
              | Num(_) | Str(_) | TemplateLit(_) | RegexLit(_)):
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

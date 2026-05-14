"""
Runs the README example from generators/chopchop/README.md exactly.

Two runs are performed with the same model and prompt:
  1. Unconstrained  — no realizability checker.
  2. Constrained    — sum_of_evens pruner (only programs whose numbers are all even).

Usage:
    python tools/readme_example.py
"""

import sys
from pathlib import Path

# Make local ChopChop package importable from repo root.
REPO_ROOT = Path(__file__).resolve().parents[1]
CHOPCHOP_ROOT = REPO_ROOT / "generators" / "chopchop"
if str(CHOPCHOP_ROOT) not in sys.path:
    sys.path.insert(0, str(CHOPCHOP_ROOT))

from core.grammar import Application, Binary, Unary, TreeGrammar, Union, EmptySet, ASTLeaf, as_tree  # type: ignore
from core.lark.from_lark import parse_attribute_grammar  # type: ignore
from core.rewrite import rewrite, rewriter  # type: ignore
from llm.realizability import RealizabilityChecker  # type: ignore
from llm.run_llm import Config, LanguageModelRunner, ModelConfig  # type: ignore


# ---------------------------------------------------------------------------
# Grammar (from README)
# ---------------------------------------------------------------------------
GRAMMAR_SOURCE = r"""
NUM: /[0-9]+/
WS: /\s+/

start: expr ";"
expr: expr "+" num {Add}
    | num
num: NUM {Num}

%ignore WS
"""


# ---------------------------------------------------------------------------
# Abstract syntax (from README)
# ---------------------------------------------------------------------------
class Add(Binary):
    ...

class Num(Unary):
    ...


# ---------------------------------------------------------------------------
# Pruner (from README) — "sum_of_evens": removes programs with odd integers.
# Note: the docstring in the README says "Remove ASTs that contain even
# integers" but the code removes *odd* ones (int(prefix) % 2 == 1 → EmptySet).
# The implementation is correct; the docstring is wrong.
# ---------------------------------------------------------------------------
@rewrite
def sum_of_evens(t: TreeGrammar) -> TreeGrammar:
    """Remove ASTs that contain odd integers (keeping only even-number programs)."""
    match t:
        case Union(children):
            return Union.of(sum_of_evens(c) for c in children)
        case Num(arg):
            token = as_tree(arg)
            match token:
                case ASTLeaf(is_complete=True, prefix=prefix) if int(prefix) % 2 == 1:
                    return EmptySet()
                case _:
                    return t
        case Add(left, right):
            return Add(sum_of_evens(left), sum_of_evens(right))
        case _:
            return EmptySet()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    # Build parser (shared across both runs).
    ast_constructors: list[type[Application]] = [Add, Num]
    lexer_spec, grammar = parse_attribute_grammar(
        ast_constructors, GRAMMAR_SOURCE, "start"
    ).build_parser()

    checker = RealizabilityChecker(sum_of_evens, grammar, lexer_spec)

    # Model from README.
    model_config = ModelConfig(model_id="codellama/CodeLlama-7b-Instruct-hf")
    model_runner = LanguageModelRunner(model_config=model_config)

    prompt = "Generate an arithmetic expression using numbers and addition, ending with a semicolon."
    context = "You are a helpful assistant. Output only the expression, nothing else. The expression must end with a semicolon."
    gen_config = Config(temperature=0.5, timeout=120)

    
    # --- Run 1: unconstrained ---
    print("=" * 60)
    print("Run 1: UNCONSTRAINED")
    rewriter.clear()
    out_unconstrained = model_runner.run(
        gen_config, prompt, context, realizability_checker=None
    )
    print(f"Output : {out_unconstrained.output!r}")
    print(f"Tokens generated : {out_unconstrained.num_tokens_generated}")
    print(f"Timed out        : {out_unconstrained.timed_out}")

    # --- Run 2: constrained (sum_of_evens) ---
    print()
    print("=" * 60)
    print("Run 2: CONSTRAINED (sum_of_evens — only even numbers)")
    rewriter.clear()
    out_constrained = model_runner.run(
        gen_config, prompt, context, realizability_checker=checker
    )
    print(f"Output : {out_constrained.output!r}")
    print(f"Tokens generated       : {out_constrained.num_tokens_generated}")
    print(f"Tokens guessed (total) : {out_constrained.num_tokens_guessed}")
    print(f"Realizability time (s) : {out_constrained.total_realizability_time:.3f}")
    print(f"Timed out              : {out_constrained.timed_out}")

    # --- Comparison ---
    print()
    print("=" * 60)
    print("Comparison")
    print("=" * 60)
    print(f"The prompt is: {prompt}")
    print(f"Unconstrained : {out_unconstrained.output!r}")
    print(f"Constrained   : {out_constrained.output!r}")

    def all_even(text: str) -> bool:
        import re
        nums = re.findall(r"\d+", text)
        return all(int(n) % 2 == 0 for n in nums) if nums else False

    print()
    print(f"Unconstrained has only even nums : {all_even(out_unconstrained.output)}")
    print(f"Constrained   has only even nums : {all_even(out_constrained.output)}")


if __name__ == "__main__":
    main()

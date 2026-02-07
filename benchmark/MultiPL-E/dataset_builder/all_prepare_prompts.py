"""
Run this script to call prepare_prompts_for_hfhub.py for every experiment combination
in libexperiments.
"""
import subprocess
from libexperiments import LANGS, VARIATIONS

def config_lang(lang: str) -> str:
    """Normalize language name for dataset config (e.g., go_test.go -> go)."""
    if lang == "go_test.go":
        return "go"
    return lang

def prompt_terminology(variation):
    match variation:
        case "keep":
            return "verbatim"
        case "remove":
            return "verbatim"
        case "transform":
            return "verbatim"
        case "reworded":
            return "reworded"

def doctests(variation):
    match variation:
        case "keep":
            return "keep"
        case "remove":
            return "remove"
        case "transform":
            return "transform"
        case "reworded":
            return "transform"

def originals(variation, dataset):
    if dataset == "mbpp":
        return "../datasets/mbpp-typed"
    if dataset == "humaneval_plus":
        return "../datasets/humaneval_plus"
    match variation:
        case "keep":
            return "../datasets/originals"
        case "remove":
            return "../datasets/originals-with-cleaned-doctests"
        case "transform":
            return "../datasets/originals-with-cleaned-doctests"
        case "reworded":
            return f"../datasets/originals-with-cleaned-doctests"
    
def prepare(lang: str, variation: str, dataset: str):
    if dataset == "mbpp":
        if variation == "remove" or variation == "transform":
            return

    d = doctests(variation)
    o = originals(variation, dataset)
    p = prompt_terminology(variation)
    target_dir = "../prompts"
    cfg_lang = config_lang(lang)
    output_spec = f"jsonl:{target_dir}/{dataset}-{cfg_lang}-{variation}.jsonl"

    cmd = f"python3 prepare_prompts_for_hfhub.py --lang humaneval_to_{lang}.py" + \
         f" --prompt-terminology {p} --doctests {d} --originals {o}" + \
         f" --original-dataset {dataset} --output {output_spec}"

    print(cmd)

    result = subprocess.run(cmd, shell=True, encoding="utf-8")
    if  result.returncode != 0:
        exit(1)

if __name__ == "__main__":
    for lang in LANGS:
        for variation in VARIATIONS:
            for dataset in [ "mbpp", "humaneval"]:
                prepare(lang, variation, dataset)

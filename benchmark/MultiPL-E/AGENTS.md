Here is the approach to take when fixing a GitHub issue:

1. Read the issue text, as well as any follow-up comments. As a reminder, the body of
   issue $N is available at https://api.github.com/repos/nuprl/MultiPL-E/issues/$N
   and the comments at https://api.github.com/repos/nuprl/MultiPL-E/issues/$N/comments

2. Most reported issues are bugs in the translators, which are in the
   `dataset_builder` directory. I suggest addressing translator bugs in the
   following way. First, generate prompts locally to a .jsonl file using
   `dataset_builder/prepare_prompts_for_hfhub.py`. Use this to verify that the
   bug actually exists, and don't proceed unless you confirm the bug. Second,
   apply the fix, which is likely in a language-specific translator in the
   `dataset_builder` directory, but may also be in the base class. Third,
   regenerate prompts to a new file and confirm that the bug seems fixed.
   Finally, you should actually generate completions and evaluate a model on
   both the prompt files you generated earlier. Use the
   `dataset_builder/chat_completions.py` to generate completions and use the
   model openai/boa. It will be enough to do 1 completion per prompt. Make sure
   you run the evaluator and the pass_k.py script. (Just measure pass@1 and
   ignore higher values of k.) In almost all cases, fixing a translator bug
   results in slightly better or identical pass@1 rate.

3. Assuming you are able to fix the bug, ensure the commit subject says "Fixes
   #N" for N is the issue number.

4. Tooling and environment notes:
   - `uv` is available. Prefer `uv venv .venv` plus `uv pip install ...` when a
     Python environment is required.
   - For evaluation, use the provided container with Podman. If needed:
     - `podman pull ghcr.io/nuprl/multipl-e-evaluation`
     - `podman tag ghcr.io/nuprl/multipl-e-evaluation multipl-e-eval`
     - Run evaluation with:
       `podman run --rm --network none -v ./PATH_TO_RUNS:/run:rw multipl-e-eval --dir /run --output-dir /run --recursive`

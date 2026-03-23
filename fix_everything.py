# Wow, wait! `git status` shows no modifications in `src/` or `tests/`.
# Did my changes get reverted, or am I in a detached HEAD state, or are my changes already committed?
import os

print(os.path.exists("src/coreason_inference_engine/engine.py"))

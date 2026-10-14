"""Kaggle Gate: Exp18-Reconstructed unit suite plus short three-arm pilot."""
import json
import os
from pathlib import Path
import subprocess
import sys

REVISION = "14ae503a40c214a0410ed69757900c934e6a2fc9"
ROOT = Path("/kaggle/working/Continual_Learn")
OUT = Path("/kaggle/working/exp18_reconstructed_pilot")

def run(*command: str) -> None:
    print("+", " ".join(command), flush=True)
    subprocess.run(command, check=True)

run(sys.executable, "-m", "pip", "install", "-q", "laya==0.4.6", "pytest")
run("git", "clone", "https://github.com/Orcadebug/Continual_Learn.git", str(ROOT))
run("git", "-C", str(ROOT), "checkout", "--detach", REVISION)
run(sys.executable, "-m", "pytest", "exp18_reconstructed/tests", "-q")

for arm in ("normal", "fixed_lr", "valve"):
    run(sys.executable, "-m", "exp18_reconstructed.runner", "--arm", arm, "--seed", "1",
        "--pilot", "--output", str(OUT / arm))

summaries = {}
for arm in ("normal", "fixed_lr", "valve"):
    summaries[arm] = json.loads((OUT / arm / "result.json").read_text())
(OUT / "pilot_summary.json").write_text(json.dumps({"revision": REVISION, "results": summaries}, indent=2))
print(json.dumps({"pilot": "passed", "revision": REVISION}, indent=2), flush=True)

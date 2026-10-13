# Source & Reproducibility Note

Everything in `src/` is the actual implementation the results in this repo
were produced with (written Oct 9–12, 2026, run in Google Colab on a T4).

## The model is a dependency, not a missing file

Laya itself is **not** vendored here because it was never local code:

- Install: `pip install laya`
- Checkpoints load from Hugging Face repo `convaiinnovations/laya` through
  the package's `load_checkpoint` (see the scripts' imports from
  `laya.train`). The architecture under test is the package's
  `DecisionModel` — nothing needs to be re-implemented or guessed.

## Layout of src/

- `lab_machinery.py` — shared helpers for Experiments 1–5 era scripts.
- `exp1_interference.py` … `exp5_replay.py` — Experiments 1–5 (standalone).
- `exp6_autonomous.py`, `exp6_autonomous_b.py`, `exp6c_*.py` — Experiment 6
  family (autonomous RL lifetime + policy-extraction tests).
- `exp7_sharpening_b.py` — **the gridworld machinery**: the `World` class
  (8×8, wall + door, keydoor and switch rules), the egocentric observation
  builder, `encode_state`, `forward_full`. `exp7b.py` is a byte-identical
  copy under the filename the Colab notebook expects in `/content`.
  `exp7_sharpening.py` is the earlier revision.
- `exp8_*_cell.py` … `exp16_*_cell.py` — Experiments 8–16 as Colab cell
  scripts. They expect the machinery loaded first (the notebook executes
  `exp7b.py` split at `def main` — everything above `main` is machinery).
- `exp17_stage1_cell.py` — Experiment 17 Stage 1 (world-model test),
  registered in `RESEARCH_LOG.md`. **It has never run; no Exp 17 results
  exist.** It defines its own corrected `ego_text`/`cell_passable` and
  expects the Exp 7B machinery in scope. Run order in Colab: T4 runtime →
  `pip install laya` → put `exp7b.py` in `/content` → run the machinery
  cell → `%run /content/exp17_stage1_cell.py`.
- `sandbox_harness.py` + `SANDBOX_DESIGN.md` — the monitored long-run
  harness (checkpoints with optimizer state, crash recovery + drill,
  episode-level logging, regression rollback, held-out comparisons).
  Built and compile-verified; not yet run.
- `EXP17_LAUNCH_PACKET.md`, `EXP17_RELAUNCH_CHECKLIST_2026-10-12.md`,
  `EXP17_DESIGN_SCAFFOLD.md` — launch route and design history for Exp 17
  (the scaffold is superseded by the registration in `RESEARCH_LOG.md`).

## Honest gaps (do not paper over these)

- **No trained checkpoints were saved from Experiments 13–16** — weights
  lived only inside the Colab runtime. This is the instrumentation gap
  documented in the Exp 16 audit in `RESEARCH_LOG.md`. Recipes, seeds, and
  hyperparameters are fully specified in the scripts, so runs are
  reproducible from the shipped checkpoint, but weight-level continuation
  from those specific lifetimes is impossible.
- The Experiment 6 policy used in the extraction tests was likewise
  runtime-only.
- **No Experiment 18 exists in this record** — no design, registration, or
  script. Exp 17 is the next run in the program. Any Exp 18 must be
  designed against this source and Exp 17's registered gates, not invented
  from the results summaries.

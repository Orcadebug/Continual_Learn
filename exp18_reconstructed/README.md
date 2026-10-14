# Experiment 18 — Reconstructed: Adaptive Learning Valve

## Status

This is a **reconstruction**, not a recovered original Experiment 18
implementation.  `src/SOURCE.md` records that the original source and trained
checkpoints do not exist in the repository.  The reconstruction is based on
the registered request, the surviving Exp 16 recipe, and the verified
Experiment 17 sensor correction.  Results produced here must never be
presented as a continuation of a previously executed Exp 18.

## Provenance retained from the record

- Gridworld and Laya encoder interface: `src/exp7_sharpening_b.py`.
- Correct switch-door observation/passability semantics: `src/exp17_stage1_cell.py`.
- Self-imitation admission, replay capacity, protected replay, novelty policy,
  phase schedule, and optimizer family: `src/exp16_completion_cell.py`.
- Full-state Kaggle resume drill: `kaggle_recovery_producer/` and
  `kaggle_recovery_consumer/`.

## Explicit reconstruction decisions

1. **No phase labels enter the controller.** Experience IDs and recency are
   the only eligibility signals for its reference batch.
2. Every primary arm has the same admission rule, replay budget, current
   update objective, initialization/seed protocol, and interaction budget.
3. `normal` uses AdamW at the base learning rate. `fixed_lr` uses the same
   AdamW setup with learning rates multiplied by 0.30. `valve` uses the base
   rates and transforms only its proposed gradient.
4. A valve update compares the ordinary current weighted-replay objective
   (`g_new`) to a disjoint retained-reference minibatch (`g_ref`). Exact
   experience-ID overlap is forbidden and logged.
5. The reference batch is observation-only for diagnostics: its before/after
   loss does not decide admission, retirement, phase transitions, or
   evaluation.

The frozen protocol is in `configs/frozen_protocol.json`.  Modify neither its
training values nor its decision thresholds after validation data is observed.

## Gates before the five-seed study

1. Unit tests: projection math, finite-gradient guard, exact-ID exclusion,
   frozen evaluation, and true 0.30x-LR control.
2. One short, separately labelled pilot per arm with checkpoint/resume.
3. Record code revision and environment manifest, then run the five paired
   seeds without changing the protocol.


# Continual_Learn — Laya Continual-Learning Research

Results and records from the Laya continual-learning program (Oct 9–12, 2026):
can Laya — a 421M-parameter ModernBERT-based decision model that answers typed
questions in a single forward pass — learn autonomously in an unfamiliar
environment, keep what it learns, adapt to unwarned rule changes, and stay
competent at old and new skills at once?

**Governing criterion (Sai's):** not peak score — become good, stay good,
adapt to a change it wasn't warned about, and remain competent at both the
new and the still-valid old situations.

## Contents

- `ALL-EXPERIMENT-RESULTS.md` — every experiment's results and verdict in one
  document, with the both-at-once battery scoreboard.
- `RESEARCH_LOG.md` — the full program record: architecture findings,
  experiment designs, predictions registered before each run, results,
  syntheses, and the Experiment 16 consistency audit.
- `results/` — raw result files per experiment (`laya_exp<N>_results.json`;
  `_partial` files are interrupted runs whose partial data was recovered).

## Headline findings

1. Decision knowledge lives in the **encoder**, distributed across depth;
   forgetting is representational overwrite, not readout drift (Exps 1–4).
2. Supervised continual learning is solved in the scoped setting: full
   fine-tuning + small verified replay + retiring superseded data gives zero
   forgetting and clean rule replacement (Exp 5).
3. Autonomous learning is **not** solved: RL barely sharpens (a training-signal
   problem — supervised training on the same states reaches 95–97%, Exps 6–7);
   self-imitation learns but plateaus at ~0.44–0.50 simultaneous competence
   across three worlds (random ≈ 0.33–0.37) and fails the unwarned rule
   change reliably across seeds (Exps 8–16, replicated in Exp 14).
4. The plateau is not optimization strength (Exp 15) and not trajectory
   structure (Exp 16). Current diagnosis: successful trajectories mix causal,
   useless, and counterproductive actions, and unconditional imitation
   reinforces all three.

## Status

Experiment 17 (world model: action-conditioned transition prediction, then
gated planning) is registered in the research log and scripted, but has not
run yet. A monitored long-run sandbox harness (checkpoints, crash recovery,
episode-level logs, regression rollback, held-out checkpoint comparisons)
is built and waiting behind Experiment 17's registered gate.

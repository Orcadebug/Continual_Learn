# Experiment 17 — World-model pathway: DESIGN SCAFFOLD (prepared 2026-10-10, pre-registered trigger)

Status: NOT launched. This scaffold exists so the pivot is immediate if its trigger fires.
Exp 16 is still running (launched ~11:53 EDT 2026-10-10); no Exp 16 result is inferred here.

## Trigger (registered in RESEARCH_LOG.md, Sai's plan 2026-10-10)
Launch Exp 17 only if Exp 16 also fails on Sai's verdict criterion — judged ONLY on the
held-out end-to-end battery across lifetimes vs the Exp 14 distribution (overall ~0.500;
empty 0.541 / keydoor 0.542 / switch 0.417):
- Exp 16 decision rule (registered): if all three arms are within noise on the battery,
  trajectory structure doesn't matter either → pivot here.
- If an Exp 16 arm clearly beats the Exp 14 battery distribution, Exp 17 is NOT next;
  follow the winning arm instead. This scaffold stays on the shelf.

## Hypothesis (Sai's, registered 2026-10-10)
Self-imitation trains only on successful trajectories, whose action labels are unreliable
(causal, unnecessary, and counterproductive-but-unpunished actions are all reinforced).
A transition predictor P(o_{t+1}, r_t | h_t, a_t) trained on ALL interactions has dense,
never-wrong labels: every step — including blocked moves and failures — correctly teaches
the world's structure. Action selection by predicted consequences should therefore learn
from the ~majority of experience that success-only imitation discards.

## Design (grounded in existing lab machinery — lab_machinery.py / exp16_segments_cell.py)
- Reuse unchanged: `World` (reset_layout / observe / step), the egocentric observation
  text (`ego_text`, including the repaired switch mention), `encode_state`, the encoder +
  `forward_full` stack, phases, admission-independent behavior, and the unified eval
  instrument (n=24, seed0=707000) + both-at-once battery from Exps 15–16.
- New component: a transition head on the shared encoder, trained on every step of every
  episode (success or failure) to predict, given (observation text, action):
  1. the next observation's structured features (relative positions of key/door/goal/
     switch as they appear in the ego text, has_key / switch_on flags), and
  2. the step reward r_t.
  Predicting structured features rather than raw next-token text keeps the target small,
  exactly labeled, and checkable against `World` state.
- Action selection (inference, no oracle): at each step, score every action by its
  predicted next-state — prefer predicted states that reduce distance-to-goal
  (reuse `dist_to_goal_map` geometry only as the scoring definition, never as input),
  break ties by predicted reward; act greedily on that score. Horizon: 1-step first
  (registered simplification); multi-step rollout is a follow-up, not part of Exp 17.
- Decision learner: frozen out. Exp 17 tests the world pathway alone against the same
  phases/battery; combining pathways is a later experiment, so attribution stays clean.
- Matched controls: same episode budgets, same seeds (707000 eval family; lifetime seed
  offsets matching Exps 14–16), 2 lifetimes minimum, results flushed per lifetime to
  /content/laya_exp17_results.json, Colab T4 only, ~2 credits/hr cap, disconnect/delete
  runtime between runs.

## Metrics (mandatory, same as Exp 16 so numbers stay comparable)
- Decomposed greedy metrics at every eval and in the battery: mile_rate =
  P(reach milestone), cond_succ = P(goal | milestone), end-to-end success, per world.
- World-model quality, reported separately and NEVER used as the verdict: next-feature
  prediction accuracy and reward-prediction error on held-out steps (Exp 15's lesson:
  confidence/internal metrics do not count — only the battery does).

## Registered predictions (written before any Exp 17 run)
1. Battery overall ≥ 0.60 (vs Exp 14's 0.500 and Exp 15's 0.452) — the dense-signal
   advantage should show up end-to-end, not just in prediction accuracy.
2. Battery switch ≥ 0.50 — rule-change adaptation is where success-only data is scarcest
   and all-interactions data is richest.
3. mile_rate improves before cond_succ does: knowing what actions *do* should first fix
   reaching the mechanic (key/switch), with finishing limited by 1-step horizon.
4. Falsification: prediction accuracy high but battery ≤ ~0.50 → the bottleneck is not
   signal density either; it is action selection / horizon, and the next move is
   multi-step planning on the same world model, not more training.

## Verdict criterion (Sai's, unchanged)
Judge ONLY on the held-out battery across lifetimes vs the Exp 14 distribution.
No phase-end instrument substitutions, no internal-metric substitutions.

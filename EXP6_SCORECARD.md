# Experiment 6 — Scorecard against Sai's criterion (2026-10-09 19:47 EDT)

Sai's test, in his words in substance: not reaching 100% once — whether Laya can
**become good at something, stay good at it, encounter a change it wasn't warned
about, adapt, and then remain competent at both the new and the still-valid old
situations.** Experiment 5's supervised replay success does NOT guarantee the
replay arm here — trajectory replay under a moving policy is a different mechanism.

## Baselines (v2, 8×8 world, measured)
- RANDOM: 33–37% success, ~46 steps
- FROZEN Laya: 30% success, ~47–49 steps — indistinguishable from random

## Five-part judgment (fill from v2 EVAL/PROBE lines when the run lands)

| Part | Where it is measured | RL | RL+REPLAY |
|---|---|---|---|
| 1. Become good | End-of-P1 / end-of-P2 eval vs 33%/30% floors | RL interim: peaked 0.75 @ 34 steps early | pending |
| 2. Stay good | Eval *variance* across a phase, not the max | **Already failed once: 0.75 → 0.0 by ep 50**, then 0.33 → 0.42–0.58 in key-door phase | pending — this is the arm where memory should show as *stability* |
| 3. Unwarned change | P3 (switch rule; key is a decoy) adaptation speed | in progress at last interim | pending |
| 4. Old returns | P4 (empty worlds return) first + final eval vs end-of-P1 | pending | pending |
| 5. Both at once | **NOT measured by v2** — no switch re-probe after P4, no saved policies | — | — |

Part 5 is closed by `exp6c_coexistence.py` (prepared 2026-10-09 evening, compiles,
core training code identical to v2): saves each arm's policy after every phase,
adds no-training interleaved probes after P2/P3/P4 (20 layouts/era, greedy),
and headlines **worst-era success** at the final probe.
Pre-registered 6c bars (working bars, not Sai's ≥95%/≥90% program targets):
final worst-era ≥ 0.50 and every era ≥ its random baseline + 0.20;
the comparative claim that matters is RL+REPLAY worst-era > RL worst-era.

## How to report the v2 finish (per the 19:48 EDT commitment in chat)
- Report against the five parts above, not against ever touching 1.00.
- If RL+REPLAY is stable where RL oscillated, name the mechanism plainly:
  memory as stabilizer. If it oscillates too, the bottleneck is not memory —
  it is credit assignment / exploration, and that becomes the next target
  (this fallback was pre-registered for v1/v2 in the research log).
- Do not claim part 5 from v2 data under any outcome; it requires 6c.

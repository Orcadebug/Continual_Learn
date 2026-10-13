# Laya Continual-Learning Program — All Experiment Results

Compiled 2026-10-12 from the raw result files and the research log
(`~/workspace/laya_continual/`). Sai's governing criterion throughout:
become good → stay good → adapt to an unwarned change → remain competent
at the new AND the still-valid old. Never peak score.

---

## Phase I — Where does Laya's knowledge live? (supervised probes, Oct 9)

### Experiment 1 — Interference on semantic tasks
Sequential fine-tuning on 6 semantic decision tasks, FULL vs FROZEN encoder.
- FULL: every task ends at 1.00 accuracy — zero forgetting, positive transfer (later tasks start higher).
- FROZEN: nothing changes at all (probe accuracy flat at 0.8828).
- **Verdict: Laya's learning is encoder-bound. The encoder is the learning organ.**

### Experiment 2 — Arbitrary mappings + an unwarned rule change
Harder tasks (arbitrary mappings, no semantic similarity to lean on).
- FULL: catastrophic forgetting confirmed — Task 1 goes 1.00 → 0.50 after all six tasks, with a recency gradient (recent tasks intact, old ones overwritten).
- Rule change: the new rule is re-learned to 1.00 and the stale rule drops to 0.00 — but at collateral cost to every other task (0.47–0.98).
- FROZEN: learns nothing (new rule 0.289).
- **Verdict: every update is a global overwrite. Plasticity does not decay, but nothing is protected.**

### Experiment 3 — Locality test (train only the top layers)
Testing Sai's hypothesis that knowledge could reorganize *locally*.
- LOCAL (top layers only): task accuracies 0.33–0.63 vs FULL's 1.00; probe accuracy peaks ~0.35 vs ~0.93–1.00.
- **Verdict: locality by depth is dead. Decision knowledge is distributed across the encoder's depth.**

### Experiment 4 — Transplant test (storage vs retrieval)
Testing Sai's second hypothesis: maybe forgetting is only a *readout* failure and the information survives.
- Early encoder + early head (E1H1): 1.00. Final encoder + final head (E6H6): 0.50.
- Final encoder + early head (E6H1): 0.50 — **no recovery**.
- Early encoder + final head (E1H6): 1.00 — perfect.
- **Verdict: the retrieval-failure hypothesis is refuted. Forgetting is storage-side: the encoder's representations themselves are overwritten. The reader is innocent.**

### Experiment 5 — Replay + retirement (the supervised winner)
Full fine-tuning + a small verified replay memory (96 items/stage) + retiring superseded data when a rule changes.
- After all 6 tasks: [1.00, 1.00, 1.00, 1.00, 1.00, 1.00].
- After the rule change: new rule 1.00, stale rule 0.00, **all other tasks still 1.00** — zero forgetting, clean replacement, zero collateral.
- **Verdict: the scoped continual-learning question is answered. Winner: full FT + curated replay + retirement — the three-part controller sketch (stream → fast memory → slow model, with a controller that curates, rehearses, retires).**

## Phase II — Can it learn autonomously? (gridworld era, Oct 9–10)

Setting: 8×8 text gridworld, one continuous lifetime — P1 empty → P2 key+door →
P3 hidden rule change (a switch opens the door; the key is a decoy) → P4 empty returns.
Baselines: random ≈ 0.33–0.37 success; frozen Laya ≈ 0.30 (≈ random).

### Experiment 6 — Autonomous Mastery Test (RL, actor-critic)
- RL arm final evals: P1 0.333, P2 0.417, P3 0.583 — but the curve oscillates violently (P1 eval went 0.75 → 0.00 → 0.33).
- RL + trajectory replay: P3 0.167 — replay was **net negative** on adaptation.
- **Verdict: the RL loop learns weakly and unstably. Exp 5's supervised-replay win does NOT transfer to replaying trajectories under a moving policy.**

### Experiment 6B/6C — Policy Extraction Test
Six action-selection strategies on the frozen trained policy, 40 paired episodes each:
- Greedy, greedy+rotation-averaging, greedy+history: **0.075 on both worlds** — the "stall": a blocked move repeats the identical state text, and the deterministic memoryless policy loops forever.
- Sampling: 0.33–0.45. Mean top-2 margin ≈ 0.009, entropy ≈ 1.386 ≈ ln 4 (maximum).
- **Verdict: after 180 RL episodes the policy is nearly flat. Extraction tricks can't rescue a policy that never sharpened.**

### Experiment 7 — Why doesn't the policy sharpen?
- Supervised ceiling, ASCII states: 0.954 held-out accuracy, 0.675 env success.
- Supervised, egocentric text: 0.975 accuracy, 0.775 success.
- Shaped-reward RL: entropy never moves (1.34–1.38 across 80 episodes), greedy 0.458.
- **Verdict: the sharpening failure is specific to the RL training signal. Supervised gradients on the same states reach 95%+. It is a signal problem, not capacity or format.**

### Experiment 8 — Self-imitation (imitate your own successful episodes)
- P1: **first sharpening ever** — entropy 1.095 → 0.782, margin → 0.578, training success 0.84–0.92, greedy ≈ 0.58–0.67.
- P2: collapse (greedy 0.25 → 0.083 → 0.167); memory becomes 59–84% stale P1 pairs — an echo chamber. Reachable successes teach "ignore the key," so the key-fetch skill is never demonstrated, hence never imitated.
- P3: no adaptation (0.167–0.25). P4 retention: 0.50.
- **Verdict: become-good ✓, stay-good ~, adapt ✗. Imitation ceiling = currently reachable successes.**

### Experiment 9 — Curated self-imitation (efficiency admission, phase weighting, collapse purge)
- P1 greedy 0.75 ✓. P2 peaked 0.583 → collapsed 0.167 ✗. P3 ≤ 0.25 ✗.
- P4 retention 0.75 — best of the era ✓.
- The purge never fired: its trigger (sampling success) is luck-inflated and masks greedy collapse.
- **Verdict: curation improves retention, not adaptation. Detector lesson: watch mechanism-specific success, never raw sampling success.**

### Experiment 10 — + novelty-directed exploration
- **Crashed mid-P3** (NaN probabilities → CUDA assert; partial data only).
- P1 greedy 0.833 / 0.75 / 0.833 / **0.917 — best single-world result of the program**.
- P2: novelty DID generate door successes (0.20 / 0.08 / 0.16) but greedy still collapsed (0.333 → 0.167).
- **Verdict: exploration solved success *generation*. The bottleneck moved to consolidation.**

### Experiment 11 — Consolidation fixes (per-phase novelty reset, protected door successes)
- **First genuine adaptation:** P2 door successes sustained (0.40 / 0.48 / 0.36 / 0.28); greedy 0.333 → 0.333 → 0.833 → 0.583, ending ≥ 0.50 with greedy steps down to 24.4 — real key-fetching, not wandering. The purge fired for the first time (3,262 pairs dropped).
- P3 still failed (0.167–0.333); P4 0.50.
- **Critical harness finding:** the observation text in Exps 8–11 *never mentioned the switch* and falsely said "The key opens it" in switch worlds — **all P3/switch numbers in Exps 8–11 are invalid as learning evidence.** P1/P2 stand.
- **Verdict: keydoor adaptation is real. A generation of switch results was measured on a blind agent.**

### Experiment 12 — Sensor repair + percentile admission + longer P3 (140 eps)
- Admission fix worked (P1: 54 episodes admitted, greedy 0.75 flat).
- But P2 peaked 0.667 → collapsed 0.167; P3 no adaptation (≤ 0.333); the 6,000-pair memory cap churned out consolidated skills; **P4 retention 0.167 — worst of the era.**
- **Verdict: net regression. Stability — the absence of any anchor against overwrite — is the binding constraint.**

### Experiment 13 — Protected core memory
Each phase's last ~300 admitted pairs become protected (never evicted/purged); 8/32 of every batch drawn from the core.
- First both-at-once battery (final weights, all three worlds, n=24): **empty 0.417, keydoor 0.458, switch 0.333** (overall 0.403).
- P4 0.50 (vs Exp 12's 0.167) — the core prevented the churn disaster but created no stability. P2 ended 0.333.
- **Verdict: passable everywhere, good nowhere. And a methodological warning: single-seed lifetimes with n=12 evals are too noisy for one-patch-per-failure attribution.**

### Experiment 14 — Replication (3 lifetimes of the frozen Exp 13 recipe)
- P2_end [0.167, 0.083, 0.50] mean 0.25. P3_end [0.083, 0.167, 0.167] mean 0.139 — **switch adaptation at phase end fails consistently across seeds: systematic, not noise.**
- P4_end mean 0.472.
- Battery means: **empty 0.541, keydoor 0.542, switch 0.417 → overall 0.500.**
- **Verdict: the recipe family reliably produces a generalist mediocrity (~0.50) and fails the rule-change moment. Exp 11's peak was partly luck.**

### Experiment 15 — Optimization density (training rounds 2→5, encoder lr 1e-5)
Judged ONLY on the held-out battery vs Exp 14 (Sai's criterion), 2 lifetimes.
- Battery: empty 0.438, keydoor 0.438, switch 0.480 → **overall 0.452 vs Exp 14's 0.500.**
- **Verdict: FAILED. Optimization strength is not the bottleneck.** (Supports Sai's reframing: the problem is what experience becomes, not optimizer effort.)

### Experiment 16 — Milestone segmentation (Sai's design; 3 arms × 2 lifetimes)
Same recipe; arms differ only in what an admitted success contributes:
A = full trajectories · B = post-milestone suffix only · C = full + segment stored again as its own example.
- Battery overall: **A 0.444** [0.444, 0.445] · **B 0.403** [0.347, 0.458] · **C 0.361** [0.222, 0.500].
- Per-world means — A: empty 0.438 / keydoor 0.438 / switch 0.459. B: 0.334 / 0.438 / 0.438. C: 0.479 / 0.292 / 0.313.
- C is a variance amplifier: the worst lifetime of the experiment and the best, including the program's best single battery cell (C-R2 empty 0.75 at 19.5 steps) next to a same-lifetime keydoor collapse.
- Decomposed metrics refuted the clean stories in both directions: suffix-only training does NOT sacrifice milestone-reaching (keydoor reach B 0.625 ≥ A 0.605); conversion differences are small and flip sign across worlds.
- **Audit (2026-10-11):** the milestone is *optional* in these worlds (the goal can spawn on the agent's side), so milestone-reach/conversion metrics mix two layout populations — roughly half of all battery successes come from episodes where the milestone never happened. One cell (A-R2 switch cond_succ 0.143) is arithmetically impossible given its own reach rate and is labeled **unverified**. Exps 13–16 saved no episode-level data or checkpoints — an instrumentation gap, fixed in Exp 17's spec.
- **Verdict: no reliable winner. Neither optimization density (15) nor trajectory structure (16) beats plain full-trajectory self-imitation, which itself plateaus at ~0.44. The registered pivot condition fired.**

### Experiment 17 — World model (registered 2026-10-11, NOT YET RUN)
Stage 1: action-conditioned transition prediction vs persistence / action-blind /
history-conditioned baselines on one fixed dataset, with a shuffled-action control.
Stage 2 (only if Stage 1 passes its registered gate): 1-step lookahead planning vs
the Exp 16 arm-A baseline. Status: script built and compile-verified; both launch
attempts on 2026-10-11 failed before any code executed (file-upload failure, then
the Google sign-in expired). Blocked on Sai's own Colab sign-in.

---

## The battery scoreboard (both-at-once test, overall mean of 3 worlds)

| Experiment | Empty | Keydoor | Switch | Overall |
|---|---|---|---|---|
| 13 (1 lifetime) | 0.417 | 0.458 | 0.333 | 0.403 |
| 14 (3 lifetimes) | 0.541 | 0.542 | 0.417 | 0.500 |
| 15 (2 lifetimes) | 0.438 | 0.438 | 0.480 | 0.452 |
| 16-A (2 lifetimes) | 0.438 | 0.438 | 0.459 | 0.444 |
| 16-B (2 lifetimes) | 0.334 | 0.438 | 0.438 | 0.403 |
| 16-C (2 lifetimes) | 0.479 | 0.292 | 0.313 | 0.361 |

## What the program has established
1. Knowledge lives in the encoder, distributed across depth; forgetting is representational overwrite, not readout drift. (Exps 1–4)
2. Supervised continual learning is SOLVED in this scoped setting: full FT + curated replay + retirement gives zero forgetting and clean rule replacement. (Exp 5)
3. Autonomous learning is NOT solved: RL barely learns (signal problem, Exp 7); self-imitation learns but plateaus at ~0.44–0.50 simultaneous competence and fails the unwarned rule change reliably. (Exps 6–16)
4. The plateau is not optimization strength (Exp 15) and not trajectory structure (Exp 16). Current diagnosis: successful trajectories mix causal, useless, and counterproductive actions, and unconditional imitation reinforces all three, compounded by distribution shift — Experience → Reusable knowledge → Reliable behavior breaks at the middle arrow.
5. Measurement itself was a failure source twice (blind switch sensor, Exps 8–11; optional-milestone confound, Exp 16 audit) — which is why Exp 17 ships with episode-level logging and automated consistency checks.

# Stage 1 Monitored Sandbox — Design

Built 2026-10-12. Script: `sandbox_harness.py` (38,654 chars, `py_compile` verified).
Engineering only — it wraps the frozen Exp 16 arm-A recipe and changes **no
experiment science**: same World, same egocentric observation, same admission,
same protected core, same training loop, same seeds (torch 6000, rng 5000),
same eval instrument (n=24, seed0=707000). Arm is fixed to "A" (full-trajectory
self-imitation, rounds=5, encoder lr 1e-5, novelty 30%).

## What the harness adds around the recipe

1. **Checkpoint / resume** — the whole lifetime is one progress dict, saved at
   every 25-episode eval and at every phase end. A single lifetime may span
   several Colab runtimes.
2. **Episode-level eval logging** — every eval episode (in-phase checkpoints and
   batteries) appends a record
   `{run, eval_tag, world, episode_seed, door_required, agent_start, goal_pos,
   milestone_reached, success, steps, success_via_milestone}` with
   `door_required := (agent_x < 4) != (goal_x < 4)` (false in empty worlds).
   This implements the Exp 16 audit's required instrumentation fixes
   (RESEARCH_LOG.md, Directive §§4/17).
3. **Consistency assert before any number prints** — for every eval cell:
   `successes == milestone_successes + no_milestone_successes` in raw counts,
   and every printed rate is accompanied by its raw counts, plus the
   door-required-conditioned success count/rate.
4. **Per-phase battery + regression detection with rollback** — a 3-world
   battery runs after each phase (Exp 16 ran it only at lifetime end; evals are
   no-grad argmax rollouts on their own seeded RNGs, so adding them does not
   perturb the training trajectory). Best overall battery is tracked; a
   battery more than **0.15 below the best** prints a `REGRESSION` line and
   triggers an automatic **rollback** to the best checkpoint (details below).
   The best checkpoint is only ever replaced by a *better* battery, never
   overwritten by a worse state.
5. **Held-out checkpoint comparison** — a separate eval set (seed base
   **909000**, n=24/world) that is never used for training, admission, or model
   selection. It runs at every phase-end checkpoint purely so frozen-policy
   learning can be compared across checkpoints independently of the battery
   that drives regression/rollback decisions.
6. **Recovery drill mode** — a runnable kill-and-resume integrity drill with a
   printed PASS/FAIL summary (procedure below).
7. **Incremental results flush** — `/content/laya_sandbox_results.json` is
   rewritten after each phase (now including batteries, held-out results,
   rollback diagnostics, and history), so the notebook-output recovery pattern
   used in Exps 15/16 still works even if every file is lost.

## Checkpoint / resume protocol

- **Save points:** after every 25 episodes (aligned with the eval cadence) and
  after every phase (post-battery).
- **Contents** (`laya_sandbox_state.pt`, via `torch.save`): model + value-head
  state_dicts, optimizer state, working memory, protected core, per-phase
  success-length history (admission state), visit counts, phase/episode
  counters, within-phase loop state (recent-success windows, phase pairs,
  purge count), full history, all battery results, best-battery value, the
  complete episode log, and all RNG states (torch CPU, CUDA, and the behavior
  `random.Random`).
- **Locations:** written to `/content/laya_sandbox_state.pt` and mirrored to
  Google Drive folder `laya_sandbox` (`MyDrive/laya_sandbox`) when Drive can be
  mounted; the mount is guarded — if it fails, the run continues on /content
  only and says so.
- **Resume:** on start, the script looks for a state file **Drive first, then
  /content**. If found it rebuilds model/optimizer exactly as a fresh run,
  loads all state, restores RNG states, prints a `RESUMED` line, and continues
  at the next un-run episode. If none is found, it starts the lifetime fresh.
- **Episode log integrity:** the JSONL is *rewritten from checkpointed state*
  at every checkpoint rather than appended to, so a reclaim landing between an
  eval and its checkpoint cannot duplicate records on resume.

## What survives a runtime reclaim — and what doesn't

| Survives (via Drive mirror) | Lost on reclaim |
|---|---|
| State checkpoint (weights, optimizer, memory, core, RNG, counters) | Anything since the last checkpoint — at most 25 episodes of behavior/training |
| Best checkpoint (`laya_sandbox_best.pt`) | The in-flight episode and transient CUDA state |
| Episode JSONL, results JSON | The runtime itself; wall-clock continuity |
| The notebook's printed output (HEARTBEAT / EVAL / BATTERY / RESULTS_JSON lines) | Exact bit-for-bit trajectory — CUDA nondeterminism means a resumed run is a faithful continuation, not an identical one (same caveat as Exp 16's 11-of-12 replication) |

If Drive cannot be mounted, **nothing survives a reclaim** except what is
visible in the notebook's saved cell outputs — the script logs that condition
at startup so it is never a surprise.

## Regression + rollback rule

After each phase battery: `overall = mean(empty, keydoor, switch)` success
rates. If `overall > best`, best updates and the state is also copied to
`laya_sandbox_best.pt`. If `overall < best − 0.15`:

1. a `REGRESSION` line is printed naming both numbers;
2. diagnostics are snapshotted into the results JSON (`rollbacks` list: tag,
   phase, live battery, best battery, drop, memory size, protected-core size,
   stale fraction, admitted total, best-checkpoint path);
3. the live model is **automatically rolled back**: weights, value head,
   optimizer state, and RNG states are restored from the best checkpoint, and
   a `ROLLBACK event` line is logged;
4. the run **continues** from the rolled-back policy. Memory, protected core,
   and progress counters stay live — experience is not discarded; only the
   policy/optimizer state rewinds. The best checkpoint file is never modified
   by a rollback, and never overwritten by a worse battery.

If no best checkpoint can be found when a rollback triggers, a
`ROLLBACK-FAILED` line is logged, the failure is recorded in the results JSON,
and the run continues unmodified rather than crashing.

## Held-out checkpoint comparison

- Set: same three worlds, same greedy instrument, **seed0 = 909000**, n=24 —
  layout-disjoint from the standard battery (707000) and from all behavior,
  admission, and training paths (behavior uses the lifetime RNG; training uses
  admitted episodes only).
- Runs at every phase-end checkpoint, immediately after the standard battery,
  and is logged per checkpoint in the results JSON under `heldout` (per-world
  raw counts beside rates, same consistency assert as the battery).
- It is **comparison-only**: held-out numbers never update the best battery,
  never trigger or suppress regression/rollback, and never influence
  admission or training. Held-out episodes are not written to the episode
  JSONL, keeping that log exactly the audit-specified battery/checkpoint
  record.

## Acceptance criteria for trusting an unattended run

A long sandbox run counts as **trusted** only when all four hold:

1. **Exp 17 gates passed (science gate — no harness change).** The world-model
   hypothesis the program is building on must clear its registered gates
   first: Stage 1 — action-conditioned prediction beats action-blind by
   ≥ 0.15 on held-out displacement (and persistence by ≥ 0.25), shuffled-action
   control within 0.02 of action-blind, event recall blocked ≥ 0.80 and
   key/switch gain ≥ 0.50; Stage 2 — the planner beats the Exp 16 arm-A
   battery baseline per its registered criteria. **Until Exp 17 Stage 1+2
   pass, a sandbox long run is an engineering exercise, not a trusted
   unattended run**, regardless of how well the harness behaves.
2. **Recovery drill passed once** (procedure below).
3. **Regression rollback active** — the rule above is in the shipped harness
   (it is; verify the `ROLLBACK` code path is present in the launched copy).
4. **Held-out comparison logging active** — `heldout` entries appear in the
   results JSON at every phase checkpoint of the run being trusted.

### Recovery drill procedure (criterion 2)

The drill runs on **separate drill paths and a separate Drive folder**
(`*_drill_*`, tag `SANDBOX-A-R1-DRILL`) — it can never contaminate a real
run's checkpoints.

1. In a cell before the harness cell:
   `import os; os.environ["SANDBOX_MODE"] = "drill"` (or pass `--drill`).
2. Run the harness cell. Stage 1 starts a fresh drill lifetime and stops at
   the first checkpoint (episode 25 of P1). The script then simulates the
   kill: the invocation returns, all in-memory state is dropped, and stage 2
   restarts **from the checkpoint file only**, verifies integrity, and trains
   3 more episodes. (For the strongest version, do this once across a *real*
   runtime reclaim: stop after stage 1's checkpoint line, delete the runtime,
   reconnect, and re-run — resume takes the same path.)
3. Read the printed summary. The drill checks, each printed as
   `DRILL CHECK <name>: PASS/FAIL`:
   - `resume_found` — a checkpoint was located and loaded;
   - `counters_identical` — phase index, next-episode counter, and admitted
     count match the pre-kill checkpoint exactly;
   - `memory_core_identical` — memory and protected-core sizes match;
   - `optimizer_state_loaded` — optimizer state entries in the checkpoint
     equal what the resumed optimizer actually holds (and are non-empty if
     any optimizer steps had run before the kill);
   - `rng_states_present` — torch and Python RNG states were saved and
     restored;
   - `progress_continuous` — after resume, training advances the episode
     counter by exactly the 3 drill episodes within the same phase.
4. Final line: `DRILL RESULT: PASS (6/6 checks)` or FAIL with the failing
   checks named. **A long run does not count as trusted until this drill has
   passed once.** Drill artifacts stay under the drill paths; delete them
   whenever — the real run never reads them.

## Configuration note (read before launching)

`PHASES` in the script is P1 80 / P2 100 / **P3 100** / P4 40 — byte-exact
against exp16_segments_cell.py. (An earlier draft carried P3 = 140 from the
Exp 12 variant; corrected 2026-10-12 after checking the Exp 16 source.)

## Launch steps (typed kernel-string route, per EXP17_LAUNCH_PACKET.md)

Do **not** use the file-upload picker (it failed for Exp 17). The route that
carried Exps 13–16:

1. Sai signs into Google himself if the browser is signed out (his standing
   rule — CAPTCHA/sign-in is his).
2. Open the Laya Colab notebook, connect **T4 only** (~2 credits/hr cap; never
   larger GPUs). If the runtime was reclaimed, re-run the setup cells first so
   the `laya` package is importable (the harness defines everything else
   itself — it does **not** depend on the notebook's machinery cells).
3. **First time only — run the recovery drill** (criterion 2 in the acceptance
   section above): in a prior cell set
   `import os; os.environ["SANDBOX_MODE"] = "drill"`, then load and run the
   harness as in step 4. Confirm `DRILL RESULT: PASS` before any long run.
   For the real run, use a fresh runtime (or clear the env var) so drill mode
   is off.
4. In a fresh cell, set `EXPSRC = r'''<full text of sandbox_harness.py>'''`,
   write it to `/content/sandbox_harness.py`, run `py_compile` on it, then
   `%load` it into a fresh cell and run. (38,654 chars — about 2.6× the
   Exp 16/17 cells; same transport, larger payload.)
5. Watch the first lines: `DRIVE mounted` (or the DRIVE-unavailable warning),
   then either `SANDBOX start` (fresh) or `RESUMED from ...` (continuation).
   First `CHECKPOINT saved` line should appear at episode 25 of P1.
6. During the run, dismiss Colab idle / "runtime will be deleted" warnings;
   the cell prints HEARTBEAT lines at every eval so progress is visible.
7. If the runtime is reclaimed: reconnect T4, repeat step 4 — the script
   resumes from the Drive checkpoint automatically. Verify the `RESUMED` line
   shows the expected phase/episode before leaving it.
8. When it prints `SANDBOX done`, read the final `RESULTS_JSON:` line and
   `/content/laya_sandbox_results.json`, then disconnect and delete the
   runtime — never leave it idling on credits.

## Known reuse limits (from building this)

- `exp16_segments_cell.py` does not define its own machinery (World,
  cell_passable, encode_state, forward_full lived in notebook cells). The
  harness takes World/encode/forward **verbatim from `lab_machinery.py`**,
  `cell_passable` **verbatim from `exp17_stage1_cell.py`** (kind-aware;
  lab_machinery's older version misclassifies the switch-world door once the
  decoy key is held), and `ego_text` verbatim from the Exp 16 cell itself
  (lab_machinery's ego_text is the stale pre-Exp-12 version — do not use it).
- Exp 16's `greedy_eval` computed milestone metrics in the same pass but kept
  no per-episode data (the audit's instrumentation gap), so the eval had to be
  extended with record-keeping; its success/milestone arithmetic is unchanged.
- Checkpointing at episode granularity required moving Exp 16's loop-local
  variables (memory deque, phase pairs, recent windows) into an explicit
  progress dict. The science functions are byte-identical; only their state
  plumbing changed, and within-phase transient state is checkpointed too, so a
  resume does not re-run the phase-start purge check or lose phase pairs owed
  to the protected core.

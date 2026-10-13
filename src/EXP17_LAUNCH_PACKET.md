# Experiment 17 Stage 1 — Relaunch Packet (prepared 2026-10-11 ~09:05 EDT)

## Status at preparation
- Exp 17 Stage 1 is REGISTERED in RESEARCH_LOG.md (2026-10-11, Directive §§5/9) and NOT running.
- Launch attempt 2026-10-11 ~06:14 EDT failed at the script-UPLOAD step only: notebook opened, T4 connected, setup/machinery cells ran, file picker stayed at "No file chosen". No experiment line ran; no results exist.
- Script verified locally at preparation: `exp17_stage1_cell.py`, 14,933 chars, `py_compile` OK.

## Fallback route (the one that carried Exp 13–16)
Do NOT use the file-upload picker. Transfer the script as a typed string into a fresh code cell, the route used for every Exp 13–16 launch:
1. Reopen the Laya Colab notebook; if Google signed the browser out, park for Sai's own sign-in (he does CAPTCHAs/sign-in himself) — do not work around it.
2. Connect T4 only. GPU cap: ~2 Colab credits/hr, T4 only, never larger GPUs. Disconnect/delete the runtime between runs so it never idles on credits.
3. If the runtime was reclaimed, re-run the setup + machinery cells first (the script depends on them: TOK, DEVICE, ACTIONS, World, collate_items, forward_full, question_for — all defined by the earlier cells, not by the script itself). Dismiss idle-warning dialogs to keep the session alive during the run; keep cells printing progress.
4. Paste/type the full text of `exp17_stage1_cell.py` as one cell (14,933 chars — same size class as the Exp 16 cells that launched this way) and run it.
5. Watch for the script's own sanity checks: tokenization skips == 0, finite eval logits per arm, event-count consistency printed before metrics. Partial results flush after each arm — a reclaim mid-run loses little.

## What Stage 1 tests (registered, do not re-scope)
One fixed transition dataset (~400 episodes/world, layout-level held-out split), six arms: persistence, action-blind, action-conditioned, history-conditioned, shuffled-action control, frozen-encoder diagnostic.
Gate for Stage 2 (planner) — ALL must hold, per registration:
- Action-conditioning benefit: action-conditioned beats action-blind by >= 0.15 on held-out displacement prediction (registered P1 also: >= persistence + 0.25)
- Shuffled-action control within 0.02 of the action-blind arm on displacement (P2)
- Event recall: blocked >= 0.80; key_gain / switch_gain >= 0.50 (P3)
If C − B < 0.05 on displacement, or accuracy is high but event recall ~0 (persistence shortcut): Stage 2 does NOT run; next step is a representation/input-format diagnostic, not planning.

## Reporting (Sai's Directive format)
Report in: current objective / completed work / evidence (raw counts beside rates) / confidence / interpretation / remaining uncertainty / next experiment / research status. Never report predictions as results; label anything unrecovered UNVERIFIED. When the run finishes or blocks, disconnect the runtime.

## Scope note
Sai's 2026-10-11 line "Please don't stop until you got it and don't ask for permission" is treated as the bounded mandate already recorded: keep the Laya loop running inside the T4 / ~2 credits-per-hour cap, reporting verdicts and genuine blockers (§20 pause conditions) — it does not authorize larger GPUs, above-cap spend, or new experiment scope beyond the registered Stage 1 → gated Stage 2 sequence.

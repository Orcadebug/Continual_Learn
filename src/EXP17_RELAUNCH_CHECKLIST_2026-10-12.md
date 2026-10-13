# Exp 17 relaunch checklist — prepared 2026-10-12 ~19:00 EDT

## Verified status as of 2026-10-12 18:23 UTC (14:23 EDT)
- Stage 1 sandbox built and verified earlier today (15:31 UTC); Sai's four trust gates folded in (15:26 UTC).
- Sai said go; launch dispatched into signed-in Colab on T4 (15:33 UTC).
- Programmatic typing into Colab editors was refused; two byte-verified pastes requested:
  machinery cell (expected LEN 19629) and second cell (expected LEN 14931) (16:14 UTC).
- Sai replied "Keep trying" instead of pasting (16:15 UTC).
- Later finding (18:21 UTC): an earlier paste had landed then vanished — the notebook was open
  in TWO windows overwriting each other's saved state. Files were then uploaded directly
  to session storage instead.
- Browser session then reset while idle and signed out of Google (18:23 UTC).
  Current state: parked at the Google password page. Nothing is running, no Stage 1 line
  has executed in this attempt, no credits are burning.

## Superseded plan — do NOT present as current
- The pending update created 2026-10-12 17:08 UTC ("clipboard-paste injection before
  re-asking him to paste") predates the 18:21/18:23 findings. The paste route is no longer
  the plan; session-storage upload is the route that worked, and sign-in is now the blocker.

## Relaunch sequence (when Sai is at the browser)
1. Sai signs into Google/Colab himself (his standing preference: he handles sign-in/CAPTCHAs).
2. Close any duplicate copy of the notebook — exactly ONE window open, so saved state
   cannot be overwritten by a second copy (root cause of the vanished paste).
3. Sai says "go".
4. Agent uploads the Exp 17 files directly to session storage (route verified 18:21 UTC),
   local sources in ~/workspace/laya_continual/ (exp17_stage1_cell.py; machinery paste
   source paste1_mach_cell.txt / paste2_exp17_cell.txt, byte sizes on disk).
5. Do not call it "running" until the first collection lines actually print.
6. Bounded mandate only: T4, ~2 Colab credits/hr cap, disconnect between runs;
   surface only at verdicts, genuine blockers, or completion.

# Equivalence audit — Exp18-Reconstructed

## Reproduced from surviving implementation

| Component | Source | Reconstruction disposition |
|---|---|---|
| 8×8 empty/keydoor/switch gridworld | `src/exp7_sharpening_b.py` | Reused through the original class |
| Switch sensor and passability repair | `src/exp17_stage1_cell.py` | Reapplied verbatim in `corrected_cell_passable` and `corrected_ego_text` |
| Phase sequence | Exp16 cells | 80 / 100 / 100 / 40 episodes |
| Successful-trajectory admission | Exp16 cells | Same percentile rule and door-use oversampling |
| Working replay capacity / protected core | Exp16 cells | 4,000 / last 300 phase items |
| Replay objective | Exp16 cells | Cross entropy on self-generated admitted actions |
| Base optimizer | Exp16 cells | AdamW, encoder `1e-5`, non-encoder `1e-3`, no weight decay |

## New, registered Exp18 intervention

The surviving record contains no Exp18 source, registration, or checkpoint.
The following are therefore **new reconstruction choices**, not claims of
historical equivalence:

- three primary arms: normal, true 0.30× AdamW learning-rate control, valve;
- reference batch selected by persistent experience ID and update age only;
- A-GEM first-order projection; and
- per-update interference, optimizer-effect, and reference-loss diagnostics.

## Irrecoverable differences

1. No original Exp18 code or model checkpoint exists, so bitwise equivalence
   cannot be established.
2. The historical 0.660 Normal R1 battery is a reported single-seed number,
   not a compatible regression fixture. It will not be used to tune or accept
   this reconstruction.
3. Exp16 did not save per-update gradients, reference batches, or AdamW
   moments. Those diagnostic histories cannot be recreated.

## Parity assertions enforced by the reconstruction

- Identical base objective and replay/admission recipe across arms.
- Fixed arm changes optimizer learning rates, not gradient tensors.
- Valve reference IDs are excluded from the current update IDs.
- The valve is the only arm allowed to alter the proposed gradient.
- Evaluation hashes trainable weights before/after and fails if it changes
  them.

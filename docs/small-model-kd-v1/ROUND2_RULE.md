# Round 2 rule for the small-model study

Frozen 2026-11-07, before any Round 1 result for Qwen3, Gemma 3, or Kanana existed.
Round 1 is `default_trials()`, the same grid all three generic-kd-v1 families used.

L = learning rate of the Round 1 CE trial with the lowest validation CE.

| L | Round 2 learning rates (both CE and KD) | Precedent |
|---|---|---|
| 2e-5 | 1e-5, 1.5e-5, 3e-5, 4e-5 | Same shape, lower edge |
| 5e-5 | 3e-5, 4e-5, 6e-5, 8e-5 | Qwen3.5 mid-range refinement |
| 1e-4 | 7.5e-5, 1.25e-4, 1.5e-4, 2.5e-4 | LFM2.5 |
| 2e-4 | 1.5e-4, 2.5e-4, 3e-4, 4e-4 | Gemma 4, upper edge |

- CE trials: alpha 1, temperature 1, forward KL stored as the contract value.
- KD trials: temperature and divergence copied from the best Round 1 KD trial.
- KD alpha: 0.7, or 0.9 if the Teacher's best validation PPL is not lower than the best Round 1 CE validation PPL (LFM2.5 precedent).
- Stop after Round 2 for every family, whatever the sign of the result.
- Select per method the lowest validation CE across both rounds. Ties go to Round 1, then the lower trial ID.
- Seeds 43 and 44 repeat only the selected CE and KD settings.
- The test split plays no part in any of these choices.

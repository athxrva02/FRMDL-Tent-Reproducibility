# Deviation report — reproduced vs README example (severity 5, seed 1)

Flags: ⚠️ |Δ| > 1pp (note in write-up) · ❗ |Δ| > 2pp (investigate cuDNN/torch version; upstream pinned torch 1.8.1).


## WRN-28-10 (Standard) · source

- **mean**: reproduced 43.5 vs README 43.5 → Δ +0.0pp ✅
- all per-corruption cells within 1pp ✅

## WRN-28-10 (Standard) · norm

- **mean**: reproduced 20.4 vs README 20.4 → Δ +0.0pp ✅
- all per-corruption cells within 1pp ✅

## WRN-28-10 (Standard) · tent

- **mean**: reproduced 18.6 vs README 18.6 → Δ -0.0pp ✅
- all per-corruption cells within 1pp ✅

---
**Summary:** 0 cell(s) flagged ⚠️ (>1pp), 0 cell(s) flagged ❗ (>2pp).


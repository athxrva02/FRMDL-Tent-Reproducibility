# Reproduction Plan — Tent (FRMDL Group Project)

## Context

This is the group project for TU Delft's FRMDL course (60% of the grade). The team of 3 will reproduce **"Tent: Fully Test-Time Adaptation by Entropy Minimization"** (Wang et al., ICLR 2021) using the authors' upstream code already cloned into `tent/`. Because we are using existing code, the assignment requires each student to own ≥1 reproducibility criterion. The deliverable is a blog post (graded 20% motivation, 50% content, 30% exposition).

**Scope (decided with the user):**
- Compute: personal GPUs (laptop/desktop) → CIFAR-10-C only, both upstream WRN architectures.
- Criteria split: **Reproduced (A)** + **Ablation (B)** + **Hyperparams check (C)**.
- Team identification: generic Student A/B/C.

The pitch already names the headline ablation: **BN-stat adaptation (`norm`) vs entropy minimization (`tent`)**. Reproduction targets are the two README tables (WRN-28-10 `Standard` and WRN-40-2 `Hendrycks2020AugMix_WRN`) at severity 5 across the 15 CIFAR-10-C corruption types.

## Existing code we will reuse (do not re-implement)

All in `tent/`:
- `tent/cifar10c.py::evaluate` — top-level evaluation loop iterating (severity × corruption type) and calling `model.reset()` between combinations.
- `tent/tent.py::Tent`, `configure_model`, `collect_params`, `forward_and_adapt`, `softmax_entropy` — the method.
- `tent/norm.py::Norm`, `configure_model` (BN-stat baseline; also exposes `reset_stats` and `no_stats` flags that are **not** currently wired through `cifar10c.py`).
- `tent/conf.py` — YACS config; all knobs (`OPTIM.LR`, `OPTIM.STEPS`, `OPTIM.METHOD`, `TEST.BATCH_SIZE`, `MODEL.EPISODIC`, `MODEL.ARCH`, `CORRUPTION.*`) can be overridden from the CLI with trailing `KEY VALUE` pairs — no code changes needed for any of the planned sweeps.
- `tent/cfgs/{source,norm,tent}.yaml` — base configs; clone and override rather than editing in place.

No new utilities are needed for tasks A and C. Task B requires a small wrapper to pass `Norm`'s `reset_stats` / `no_stats` flags through `setup_norm` (one ~5-line patch in `cifar10c.py`).

## Task split

### Student A — *Reproduced*
**Goal:** Re-run the upstream three configs end-to-end on both architectures and reproduce the README tables.
**Runs (6 total):**
1. `python cifar10c.py --cfg cfgs/source.yaml`
2. `python cifar10c.py --cfg cfgs/norm.yaml`
3. `python cifar10c.py --cfg cfgs/tent.yaml`
4. (1–3) repeated with `MODEL.ARCH Hendrycks2020AugMix_WRN`.
**Deliverables:** two tables matching the README format (mean error + per-corruption at severity 5); a third table or plot showing severity 1–5 trend per method; a short note on any number that deviates >1pp from upstream and the suspected cause (cuDNN/seed/lib versions).

### Student B — *Ablation: BN-stat vs entropy*
**Goal:** Decompose Tent's improvement into the BN-stat-adaptation component (already given by `norm`) and the entropy-minimization component (the gradient updates Tent adds on top). Run on WRN-28-10 `Standard` at severity 5 only to keep run count tractable.
**Ablation grid (one row per config, all share base `cfgs/tent.yaml` unless noted):**

| Variant | Forward stats | Affine params updated? | Loss | Notes |
|---|---|---|---|---|
| (a) source | running (eval) | none | — | baseline |
| (b) norm | batch | none | — | BN-stat only |
| (c) tent (full) | batch | BN γ, β | entropy | upstream Tent |
| (d) entropy-only | running (eval) | BN γ, β | entropy | needs small patch: keep `track_running_stats=True` in `tent.configure_model` |
| (e) tent w/ no affine update | batch | none | — | identical to (b) — sanity check |
| (f) tent, episodic | batch | BN γ, β | entropy | `MODEL.EPISODIC True` — isolates online accumulation effect |
| (g) reset_stats norm | reset running, then batch updates | none | — | uses `Norm(reset_stats=True)` |

Variants (d) and (g) require the small wrappers below; (a)–(c) and (f) are pure CLI overrides.

**Code changes (Student B owns):**
1. In `tent/tent.py`, add an `adapt_stats: bool = True` flag to `configure_model` so variant (d) can keep running stats while still training BN affine params.
2. In `tent/cifar10c.py::setup_norm`, read `cfg.BN.RESET_STATS` / `cfg.BN.NO_STATS` and forward them to `Norm(...)`; add the two new fields to `conf.py::_C.BN`.
3. Add `cfgs/ablation_*.yaml` for the four configs not expressible as a one-line override.

**Deliverable:** a single bar/line chart of mean error across the seven variants + a 2-paragraph interpretation: how much of Tent's gain comes from BN stats vs. from gradient updates on BN affine params.

### Student C — *Hyperparams check*
**Goal:** Map Tent's sensitivity to the choices `conf.py` exposes. WRN-28-10 `Standard`, severity 5, all 15 corruption types.

**Sweeps (each is one axis varied with others at upstream defaults):**
1. **Learning rate:** `OPTIM.LR ∈ {1e-4, 3e-4, 1e-3 (default), 3e-3, 1e-2}` — 5 runs.
2. **Update steps per batch:** `OPTIM.STEPS ∈ {1 (default), 2, 4, 8}` — 4 runs.
3. **Batch size:** `TEST.BATCH_SIZE ∈ {32, 64, 100, 200 (default), 400}` — 5 runs. (Smallest batches stress the BN-stat estimator and are the most diagnostic.)
4. **Optimizer:** `OPTIM.METHOD ∈ {Adam (default), SGD}` with SGD using upstream momentum=0.9; 1 extra run.
5. **Episodic vs online:** `MODEL.EPISODIC ∈ {False (default), True}` — 1 run; isolates the effect of continual adaptation.

Total ≈ 16 runs, each ~15–30 min on a personal GPU (WRN-28-10 forward + one BN-affine step per batch).

**Deliverable:** five small plots (one per sweep axis) of mean CIFAR-10-C-5 error vs. the swept value, plus a paragraph identifying which axis matters most. Flag any setting that breaks Tent (loss diverges, worse than source).

## Shared infrastructure (light)

- All three students keep results in CSV under `tent/output/<student-initial>/` (one row per `(method, arch, severity, corruption, seed)`). Logs from `cifar10c.py` already record per-corruption error — a tiny `parse_logs.py` (Student C writes, A and B reuse) is enough; do not over-engineer.
- Seeds: keep `RNG_SEED=1` for the main numbers; Student A runs *one* additional seed (`RNG_SEED=2`) per config to quantify variance.
- Environment: `pip install -r tent/requirements.txt` (torch 1.8.1, robustbench v0.1, yacs, iopath). RobustBench will auto-download checkpoints to `CKPT_DIR` and CIFAR-10-C to `DATA_DIR` on first run — Student A does this first and shares the populated dirs (or each student downloads independently).

## Critical files to modify

- `tent/tent.py` (Student B): add `adapt_stats` flag to `configure_model`.
- `tent/cifar10c.py` (Student B): wire BN flags into `setup_norm`.
- `tent/conf.py` (Student B): add `_C.BN.RESET_STATS`, `_C.BN.NO_STATS`.
- `tent/cfgs/ablation_*.yaml` (Student B): 4 new configs.
- `tent/parse_logs.py` (Student C, optional): tiny log → CSV helper.

No changes to `tent/norm.py` or the upstream `Tent` class itself are required.

## Verification

- **Smoke test (each student, day 1):** `python cifar10c.py --cfg cfgs/source.yaml CORRUPTION.SEVERITY [5] CORRUPTION.TYPE [gaussian_noise] CORRUPTION.NUM_EX 1000` — completes in <2 min, confirms CUDA + RobustBench checkpoint + CIFAR-10-C download.
- **Reproduction sanity (Student A):** WRN-28-10 `tent` mean error at severity 5 should be **18.6 ± ~1pp**. Deviation >2pp → investigate cuDNN/torch version (upstream pinned 1.8.1).
- **Ablation sanity (Student B):** variant (e) ("tent, no affine update") must match variant (b) ("norm") to within numerical noise — if not, the patch is wrong.
- **Hyperparam sanity (Student C):** default-LR run in the sweep must match Student A's `tent` number.

## Out of scope

- ImageNet-C (no upstream reference code, compute prohibitive on personal GPUs).
- Non-BN architectures (Tent's `check_model` enforces BN presence — would be a separate "New algorithm variant" criterion, not chosen).
- Dent / adversarial-perturbation extension mentioned in upstream README.

## Blog post outline (collective)

1. Why test-time adaptation matters (motivation).
2. Tent in one figure.
3. Reproduction tables (Student A).
4. Ablation: BN-stat vs entropy (Student B) — the core narrative.
5. Hyperparameter sensitivity (Student C) — practical takeaways.
6. Do our results uphold the paper's main conclusions? (required by grading rubric).
7. Per-student contribution paragraphs (required by assignment).

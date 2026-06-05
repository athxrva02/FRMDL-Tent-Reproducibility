# Reproduction Plan — Tent (FRMDL Group Project)

## Context

This is the group project for TU Delft's FRMDL course (60% of the grade). A team of 3
reproduces **"Tent: Fully Test-Time Adaptation by Entropy Minimization"** (Wang et al.,
ICLR 2021) using the authors' upstream code already cloned into `tent/`. Because we use
existing code, the assignment requires each student to own ≥1 reproducibility criterion. The
deliverable is a blog post (graded 20% motivation, 50% content, 30% exposition).

**Compute reality (this revision).** We no longer have personal GPUs; all runs happen on
**Google Colab free tier (single T4)**, which has session time limits and disconnect risk.
We are also time-boxed. The plan is therefore **cut to the headline result plus exactly three
ablations**. Every *experiment* (Students A and B) is a **pure CLI override** of the upstream
config — no code changes. The only upstream-code edit is Student C's small, self-contained
**device-agnostic port of `cifar10c.py`** (the "New code variant" criterion); `tent.py`,
`norm.py`, and the method logic are untouched. Total GPU work is ≈3 reproduction runs + ≈11
short sweep runs.

The three ablations are the ones the Tent paper itself foregrounds (arXiv 2006.10726): the
**source → norm(BN) → tent** decomposition, **batch-size dependence** (scale LR with batch
size; Tent collapses at tiny batches), and **number of update steps**. The paper's own
which-params ablation is dropped because it would require patching `tent.py`.

## Scope (reduced)

- **Architecture:** WRN-28-10 `Standard` **only** (drop `Hendrycks2020AugMix_WRN`).
- **Severity:** **5 only** (drop the severity 1–5 sweep).
- **Corruptions:** all 15 for the headline reproduction table; a representative 5-corruption
  subset for the ablation sweeps (see Student B).
- **Reproduction target** (from `tent/README.md`, severity-5 mean error %):

  | arch | source | norm | tent |
  |---|---:|---:|---:|
  | WRN-28-10 `Standard` | 43.5 | 20.4 | 18.6 |

  The README notes its example is *"for explanation, not reproduction,"* so these are the
  README *example* numbers, not the paper's own tables — call this out in the blog.

## Existing code we reuse (do not re-implement)

All in `tent/`:
- `tent/cifar10c.py::evaluate` — evaluation loop over (severity × corruption type), calling
  `model.reset()` between combinations.
- `tent/tent.py`, `tent/norm.py` — the Tent method and the BN-stat baseline.
- `tent/conf.py` — YACS config. Every knob is overridable from the CLI with trailing
  `KEY VALUE` pairs; **no code changes are needed for any run in this plan.** Defaults:
  `TEST.BATCH_SIZE 128`, `OPTIM.LR 1e-3`, `OPTIM.STEPS 1`. Note `cfgs/tent.yaml` overrides
  `TEST.BATCH_SIZE` to **200** (and keeps `OPTIM.LR 1e-3`, `OPTIM.STEPS 1`).
- `tent/cfgs/{source,norm,tent}.yaml` — base configs; override on the CLI rather than editing.
- `tent/repro_a/` — reproducibility + analysis tooling already in the repo:
  `run_all.sh` (runner), `parse_logs.py` (log → tidy CSV), `make_tables.py` (CSV → tables/plots),
  `Tent_Colab.ipynb` + `COLAB.md` (Colab Py3.8 env, Drive persistence, disconnect-resume).

> **YACS gotcha (applies to every command below).** List overrides are coerced via
> `literal_eval`, so string-valued lists need quoted elements:
> `CORRUPTION.TYPE "['gaussian_noise']"`, **not** `[gaussian_noise]`. Numeric lists like
> `CORRUPTION.SEVERITY [5]` need no inner quotes.

## Task split (3 students, distinct criteria)

| Student | Criterion | Owns |
|---|---|---|
| **A** | **Reproduced** | the reduced headline table (source/norm/tent, sev5, 15 corruptions) |
| **B** | **Ablation study** | all three ablations below |
| **C** | **New code variant** | a device-agnostic **port of upstream `cifar10c.py`** + the `repro_a/` runner/parser/plotter generalized to drive A *and* B |

> The batch-size and update-steps sweeps are framed here as **ablations** (ablating a design
> choice of the method), not as a separate "hyperparameter check" — this is how we keep three
> distinct criteria after dropping the old Hyperparams task.

### Student A — *Reproduced*

Re-run the three upstream configs on `Standard` at severity 5 over all 15 corruptions. Adding
`CORRUPTION.SEVERITY [5]` cuts runtime ≈5× versus the cfg default (which sweeps 5 severities).
One extra `tent` run at `RNG_SEED 2` gives a variance number.

```bash
# from tent/
python cifar10c.py --cfg cfgs/source.yaml MODEL.ARCH Standard CORRUPTION.SEVERITY [5] SAVE_DIR output/A/source
python cifar10c.py --cfg cfgs/norm.yaml   MODEL.ARCH Standard CORRUPTION.SEVERITY [5] SAVE_DIR output/A/norm
python cifar10c.py --cfg cfgs/tent.yaml   MODEL.ARCH Standard CORRUPTION.SEVERITY [5] SAVE_DIR output/A/tent
python cifar10c.py --cfg cfgs/tent.yaml   MODEL.ARCH Standard CORRUPTION.SEVERITY [5] RNG_SEED 2 SAVE_DIR output/A/tent_seed2
```

**Deliverable:** one README-format severity-5 table (mean + per-corruption) for the three
methods, plus a short note on any number deviating >1pp from the README and the suspected cause
(cuDNN/seed/lib versions — the full env is logged at the top of every log by `conf.py`).

### Student B — *Ablation study* (three ablations)

All on `Standard`, severity 5, **online tent**. Sweeps run on a **representative 5-corruption
subset** spanning the four corruption groups (noise / blur / weather / digital):

```
CORRUPTION.TYPE "['gaussian_noise','motion_blur','fog','contrast','jpeg_compression']" CORRUPTION.SEVERITY [5]
```

**Ablation 1 — BN-stat vs entropy decomposition.** *Reuses Student A's full-15 source/norm/tent
severity-5 runs — zero extra GPU runs.* Attribute Tent's improvement to its two components:

- `source − norm` → the **BN-stat-adaptation** gain (batch statistics replacing running stats).
- `norm − tent` → the **entropy-gradient** gain (the BN-affine updates Tent adds on top).

**Deliverable:** a grouped/stacked bar chart of the two components + a 2-paragraph reading of
how much of Tent's gain is BN stats vs. gradient updates.

**Ablation 2 — Batch size (LR scaled proportionally).** Following the paper's protocol (lower
the batch size, lower the LR by the same factor), scale from the `tent.yaml` reference
(BS 200, LR 1e-3): **`LR = 1e-3 × BS / 200`**. Sweep `BS ∈ {8, 16, 32, 64, 128, 200}` → 6 short
runs on the 5-corruption subset.

```bash
# BS=8  -> LR=4e-5     BS=16 -> LR=8e-5     BS=32 -> LR=1.6e-4
# BS=64 -> LR=3.2e-4   BS=128-> LR=6.4e-4   BS=200-> LR=1e-3 (matches A's tent)
python cifar10c.py --cfg cfgs/tent.yaml MODEL.ARCH Standard \
  CORRUPTION.SEVERITY [5] \
  CORRUPTION.TYPE "['gaussian_noise','motion_blur','fog','contrast','jpeg_compression']" \
  TEST.BATCH_SIZE 32 OPTIM.LR 1.6e-4 SAVE_DIR output/B/bs/bs032
```

**Deliverable:** mean-error vs. batch-size line plot; show the small-batch collapse and explain
it via the degraded batch-statistic estimate Tent relies on.

**Ablation 3 — Number of update steps per batch.** Sweep `OPTIM.STEPS ∈ {1, 2, 4, 8, 16}` → 5
short runs on the 5-corruption subset (BS/LR at `tent.yaml` defaults).

```bash
python cifar10c.py --cfg cfgs/tent.yaml MODEL.ARCH Standard \
  CORRUPTION.SEVERITY [5] \
  CORRUPTION.TYPE "['gaussian_noise','motion_blur','fog','contrast','jpeg_compression']" \
  OPTIM.STEPS 4 SAVE_DIR output/B/steps/steps04
```

**Deliverable:** mean-error vs. steps line plot; show diminishing returns / over-adaptation as
steps grow (online updates accumulate across batches).

### Student C — *New code variant*

The criterion (`assignment.md`): *"Rewrote or ported existing code to be more efficient/
readable."* C owns **two** things — a genuine **port of the upstream paper code**, plus the
reproducibility/analysis pipeline that drives Students A and B.

**C.1 — Device-agnostic port of `tent/cifar10c.py` (the actual "code variant").** Upstream
hard-codes the GPU: `cifar10c.py` calls `.cuda()` unconditionally (model at lines 23–24, test
tensors at line 47), and `requirements.txt` pins `torch==1.8.1`, so the code cannot run on CPU
or Apple-Silicon (MPS) at all. Port it to be **device-agnostic** so the *same* script runs on
the T4, on a CPU box, and on a Mac:

- Add a one-line device selector — `device = "cuda" if torch.cuda.is_available() else ("mps" if
  torch.backends.mps.is_available() else "cpu")` — and replace the two `.cuda()` sites with
  `.to(device)`. `tent.py` and `norm.py` need **no** changes (verified: they carry no hardcoded
  device — they operate on whatever device the model/tensors already live on).
- Surface the chosen device in the log (next to the existing torch/cuda/cudnn version line in
  `conf.py`) so every run records where it executed.
- **Why this is a real code variant, not just infra:** it changes upstream behaviour (CUDA-only
  → portable), is readable/minimal, and is *practically* valuable — students can smoke-test and
  debug locally on a laptop instead of burning scarce free-tier T4 minutes, and CPU runs give a
  third independent environment for the reproducibility cross-check.
- **Keep the port traceable for authorship:** do it on its own commits touching only
  `cifar10c.py` (and the one log line in `conf.py`), so the diff is unambiguously C's work and
  does not collide with B's experiments (B changes nothing in upstream code).

**C.2 — Reproducibility + analysis pipeline.** Generalize the existing `tent/repro_a/` tooling
so **one** config-driven runner + parser + plotter drives **both** A's reproduction and B's
sweeps:

- **`run_all.sh`** — reduce the default matrix to single arch / severity 5; add a small driver
  (or sibling script) that enumerates B's BS and STEPS sweep points and writes to
  `output/B/{bs,steps}/<value>/`.
- **`parse_logs.py`** — it already recovers `arch`/`method`/`seed` from the logged YACS config
  dump; extend the same content-parsing to also capture `TEST.BATCH_SIZE` and `OPTIM.STEPS`, so
  sweep points are identified from the log content (robust to the directory scheme). Relax the
  hard-coded "expected 900 rows" check.
- **`make_tables.py`** — add the two ablation line plots (BS, STEPS) and the decomposition bar
  chart alongside the existing severity-5 table.

**Deliverable:** the ported `cifar10c.py` (with a short before/after note on the CUDA-only →
portable change and a CPU-vs-T4 sanity comparison on the smoke test), plus the unified pipeline
that regenerates every table/plot in the blog from the raw logs.

## Shared infrastructure (light)

- Results in CSV under `tent/output/<student-initial>/`, one row per
  `(method, arch, severity, corruption, seed[, batch_size, steps])`. `parse_logs.py` already
  emits this from the upstream logs.
- Seeds: `RNG_SEED=1` for headline numbers; Student A does one `RNG_SEED=2` tent run for variance.
- RobustBench auto-downloads the `Standard` checkpoint to `./ckpt` and CIFAR-10-C to `./data` on
  first run. Whoever runs first shares the populated dirs (or each downloads independently).
- **Parallelize across Colab accounts.** Three students = three free T4 sessions. B's BS and
  STEPS sweep points are independent, so they can be split across the A/C accounts to stay under
  the per-session time limit.

## Effort & time estimate

GPU figures are **T4 free-tier wall-clock** for WRN-28-10 on CIFAR-10-C (10k images/corruption
at severity 5); treat them as ±50% — T4 throughput and Colab I/O vary run to run.

| Student | GPU runs | T4 wall-clock | Human effort | Dominated by |
|---|---|---|---|---|
| **A — Reproduced** | 4 (source, norm, tent, tent seed-2) × 15 corruptions | **~45–60 min** | **~1–1.5 days** | running + building the table & deviation note (tooling already exists) |
| **B — Ablation study** | ~11 (6 batch-size + 5 steps) on the 5-corruption subset; ablation 1 reuses A's runs | **~1.5–2 h** | **~1.5 days** | runs + 3 figures + interpretation paragraphs |
| **C — New code variant** | ~0 (smoke tests only) | **~10–20 min** | **~1.5–2 days** | Python: the `cifar10c.py` port + pipeline coding |

Plus a **one-time ~15–20 min** first-run download (RobustBench checkpoint + CIFAR-10-C, shared
via Drive) and a **~2 min smoke test** per student on day 1.

**Where the GPU time goes.** A is cheap (forward-only, or one BN-affine step/batch). B's cost is
front-loaded by two heavy corners — **small batch sizes** (BS=8 → ~1,250 batches/corruption) and
**STEPS=16** (16× backward per batch); those two points alone are ~half of B's GPU time. C burns
almost no GPU — its cost is engineering hours, not compute.

**Project-level.**
- **Total GPU ≈ 3–4 T4-hours** across the team — comfortably within free tier, and lower in
  wall-clock once B's independent sweep points are split across the three Colab accounts.
- **Critical path: A → B**, with **C in parallel throughout.** A must finish first (its
  source/norm/tent runs feed B's ablation 1, and everyone needs the shared checkpoint/data).
  C's port and pipeline don't block on results, so C runs alongside.
- **Calendar:** with TA coordination and collective blog writing, plan **~1 week part-time per
  student** — the experiments are a small slice; the blog (30% of the grade) absorbs the rest.

**If B is time-pressed:** STEPS=16 is the single biggest variable (30+ min on a slow T4 day).
Cap the steps sweep at 8, or subsample `CORRUPTION.NUM_EX 2000` for the steps sweep only — both
cut that corner cheaply without hurting the trend.

## Verification

- **Smoke test (day 1, each student):**
  ```bash
  python cifar10c.py --cfg cfgs/source.yaml MODEL.ARCH Standard \
    CORRUPTION.SEVERITY [5] CORRUPTION.TYPE "['gaussian_noise']" CORRUPTION.NUM_EX 1000 \
    SAVE_DIR output/_smoke
  ```
  Completes in <2 min; confirms CUDA + RobustBench checkpoint + CIFAR-10-C download.
- **Reproduction sanity (A):** severity-5 mean error ≈ `source 43.5 / norm 20.4 / tent 18.6`,
  each within ~1pp. A gap >2pp ⇒ investigate torch/cuDNN versions (upstream pins `torch==1.8.1`).
- **Ablation consistency (B):** the `BS=200, LR=1e-3` and `STEPS=1` sweep points (the
  `tent.yaml` defaults) must match Student A's `tent` number on the same 5 corruptions — if not,
  an override is wrong.
- **Ablation signal (B):** `BS=8` should be dramatically worse than `BS=200` (collapse);
  `STEPS=16` should not meaningfully beat `STEPS=1` (often worse).
- **Port sanity (C):** after the device-agnostic port, the smoke test runs on **CPU** (and on
  the T4) with no code changes, and the CPU vs. T4 `source` error on `gaussian_noise` sev5
  agrees to within rounding — confirming the port changed only the device, not the math.

## Out of scope

- Second architecture (`Hendrycks2020AugMix_WRN`) and the severity 1–4 sweep.
- ImageNet-C (no upstream reference code; compute prohibitive on free T4).
- Non-BN architectures (Tent's `check_model` enforces `BatchNorm2d`).
- Code-patch ablations from the earlier plan (entropy-only with running stats, `reset_stats`
  norm, which-params-to-adapt) and the old ~16-run hyperparameter grid.

## Blog post outline (collective)

1. Why test-time adaptation matters (motivation).
2. Tent in one figure.
3. Reduced reproduction table — WRN-28-10, severity 5 (Student A).
4. Three ablations (Student B): BN-stat vs entropy · batch size · update steps — the core narrative.
5. Code variant + reproducibility tooling (Student C) — device-agnostic port (CUDA-only →
   CPU/MPS/CUDA) and the one-command pipeline that regenerates every table/plot on free T4.
6. Do our results uphold the paper's main conclusions? (required by grading rubric).
7. Per-student contribution paragraphs (required by assignment).

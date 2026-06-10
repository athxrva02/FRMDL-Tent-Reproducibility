# Tent — FRMDL Reproducibility Project

Reproduction of **"Tent: Fully Test-Time Adaptation by Entropy Minimization"**
(Wang et al., ICLR 2021).

| Criterion | Scope |
|---|---|
| **Reproduced** | Headline table: source / norm / tent on WRN-28-10, severity 5, 15 corruptions |
| **Ablation study** | Three ablations: BN-stat vs entropy · batch-size · update steps |
| **New code variant** | Device-agnostic port of `tent/cifar10c.py` (CUDA-only → CPU/MPS/CUDA) |

See [`docs/reproduction-plan.md`](docs/reproduction-plan.md) for the full task
split and compute budget. See [`docs/student-b-ablation-plan.md`](docs/student-b-ablation-plan.md)
for the detailed ablation study plan.

---

## Reproduction target

From the upstream [`tent/README.md`](tent/README.md) (severity-5 mean error %):

| arch | source | norm | tent |
|---|---:|---:|---:|
| WRN-28-10 `Standard` | 43.5 | 20.4 | 18.6 |

> The upstream README notes these numbers are "for explanation, not reproduction"
> — they are example outputs, not the paper's own tables. Call this out in the blog.

---

## Quickstart — Google Colab (recommended)

Open **[`Tent_Colab.ipynb`](Tent_Colab.ipynb)** and run top to bottom. The notebook
handles the Python-3.8 environment, Drive persistence, disconnect-resume, and all
three students' experiments in one place:

| Notebook section | Cells | Student | What runs |
|---|---|---|---|
| Setup (GPU, Drive, repo, data) | 0–10 | shared | env + downloads |
| Reproduction | 11–26 (approx) | A | 4 headline runs, tables, deviation report |
| Ablation 1 — BN-stat vs entropy | 27–31 | B | analysis only (reuses A's logs) |
| Ablation 2 — Batch-size sweep | 32–38 | B | 6 runs: BS ∈ {8,16,32,64,128,200} |
| Ablation 3 — Update-steps sweep | 39–45 | B | 5 runs: STEPS ∈ {1,2,4,8,16} |

> For Google Colab setup details (Python 3.8 venv, Drive symlinks, checkpoint
> download workaround) see [`tent/repro_a/COLAB.md`](tent/repro_a/COLAB.md).

---

## Two execution environments

| Step | Where | Why |
|---|---|---|
| GPU runs (`run_all.sh`, ablation sweeps) | **Colab T4 / CUDA box** | `cifar10c.py` calls `.cuda()` unconditionally (until Student C's port lands); `torch==1.8.1` pinned — no Apple Silicon |
| Analysis (`parse_logs.py`, `make_tables.py`) | **anywhere — laptop OK** | Pure stdlib + pandas + matplotlib; no torch |

---

## Reproduction

### Run (from project root, on GPU)

```bash
cd tent
pip install -r requirements.txt          # torch 1.8.1, robustbench v0.1, yacs, iopath

bash ../run_all.sh --smoke               # day-1 sanity (~2 min, downloads ckpt + data)
bash ../run_all.sh                       # full 4-run set: source/norm/tent @seed1, tent @seed2
```

Each run evaluates **all 15 corruptions at severity 5**. Logs land in
`output/A/Standard/<method>/seed<seed>/`. The RobustBench checkpoint and CIFAR-10-C
download to `./ckpt` and `./data` on first run — Student A runs first and shares
these with B and C.

### Analyze (laptop)

```bash
pip install pandas matplotlib
python parse_logs.py                     # output/A/**/*.txt → output/A/results.csv
python make_tables.py                    # results.csv → table, variance, deviation report
```

Generated in `output/A/`:

| File | Contents |
|---|---|
| `results.csv` | Tidy CSV: `arch, method, seed, severity, corruption, error` (60 rows for the full 4-run set) |
| `table_sev5_Standard.md` | README-format severity-5 table (headline deliverable) |
| `variance.md` | Seed-1 vs seed-2 severity-5 mean and gap |
| `deviation_report.md` | Reproduced vs README numbers — ⚠️ >1 pp, ❗ >2 pp |

`make_tables.py --seed N` selects the headline seed (default `1`).

---

## Ablation Study

All ablations run on **WRN-28-10 Standard, severity 5**, using the 5-corruption
subset that spans all four corruption groups:

```
gaussian_noise  (noise)  ·  motion_blur  (blur)
fog  (weather)  ·  contrast  (digital)  ·  jpeg_compression  (digital)
```

### Ablation 1 — BN-stat vs Entropy Decomposition

**Zero extra GPU runs.** Decomposes Tent's total improvement into two components
using Student A's existing `source`/`norm`/`tent` outputs:

```
BN-stat gain  =  source_err − norm_err    (batch statistics replace training running stats)
Entropy gain  =  norm_err   − tent_err    (BN affine γ, β updated by entropy gradient)
```

Expected split from the README reference numbers: ~93% BN-stat, ~7% entropy.

**Output:** stacked bar chart saved to `output/B/ablation1_decomposition.png`.

### Ablation 2 — Batch-Size Dependence

6 runs sweeping `TEST.BATCH_SIZE` with `OPTIM.LR` scaled proportionally
(`LR = 1e-3 × BS / 200`):

```bash
# from tent/, using the venv python (or run via Tent_Colab.ipynb cells 33-37)
python cifar10c.py --cfg cfgs/tent.yaml MODEL.ARCH Standard \
  CORRUPTION.SEVERITY [5] \
  CORRUPTION.TYPE "['gaussian_noise','motion_blur','fog','contrast','jpeg_compression']" \
  TEST.BATCH_SIZE 32 OPTIM.LR 1.6e-4 SAVE_DIR ../output/B/bs/bs032
```

| BS | LR | Expected |
|---:|---|---|
| 200 | 1e-3 | Reference — must match Student A tent on 5 corruptions |
| 128 | 6.4e-4 | Near-reference |
| 64 | 3.2e-4 | Slight degradation |
| 32 | 1.6e-4 | Noticeable degradation |
| 16 | 8e-5 | Heavy degradation |
| 8 | 4e-5 | Collapse (may exceed `norm` baseline) |

**Output:** `output/B/bs/bs{NNN}/` logs + `output/B/ablation2_batch_size.png`.

### Ablation 3 — Number of Update Steps

5 runs sweeping `OPTIM.STEPS` at `tent.yaml` defaults (BS=200, LR=1e-3):

```bash
python cifar10c.py --cfg cfgs/tent.yaml MODEL.ARCH Standard \
  CORRUPTION.SEVERITY [5] \
  CORRUPTION.TYPE "['gaussian_noise','motion_blur','fog','contrast','jpeg_compression']" \
  OPTIM.STEPS 4 SAVE_DIR ../output/B/steps/steps04
```

| STEPS | Backward passes / batch | Expected |
|---:|---:|---|
| 1 | 1 | Reference — must match Student A tent on 5 corruptions |
| 2 | 2 | Comparable |
| 4 | 4 | Degradation starts |
| 8 | 8 | Clear degradation |
| 16 | 16 | Max drift; may exceed `norm` baseline |

**Output:** `output/B/steps/steps{NN}/` logs + `output/B/ablation3_steps.png`.

---

## Student C — New Code Variant

Device-agnostic port of `tent/cifar10c.py`: replaces the two unconditional
`.cuda()` calls with `.to(device)`, adds a one-line device selector, and surfaces
the chosen device in the log. `tent.py` and `norm.py` require no changes.

After the port, the smoke test runs identically on a T4, a CPU box, and Apple
Silicon — enabling local debugging without burning Colab GPU time.

---

## Output directory layout

```
output/
  A/
    Standard/
      source/seed1/   norm/seed1/   tent/seed1/   tent/seed2/
    results.csv
    table_sev5_Standard.md   variance.md   deviation_report.md
  B/
    bs/
      bs008/  bs016/  bs032/  bs064/  bs128/  bs200/
    steps/
      steps01/  steps02/  steps04/  steps08/  steps16/
    ablation1_decomposition.png
    ablation2_batch_size.png
    ablation3_steps.png
```

---

## Sanity checks

### Student A
- Smoke test completes in ~2 min and writes a log to `output/A/_smoke/`.
- `tent` severity-5 mean ≈ **18.6 ± 1 pp** (`norm` ≈ 20.4, `source` ≈ 43.5).
  Gap >2 pp → investigate cuDNN/torch version (logged at top of every run by `conf.py`).
- `results.csv` has **60 rows** for the full 4-run set (4 runs × 15 corruptions).
- `deviation_report.md` flags nothing >2 pp on a clean reproduction.

### Student B — three-way consistency check

The following three points all use the same effective config and must agree to
within **0.5 pp** on the 5-corruption-subset mean error at severity 5:

| Point | Location |
|---|---|
| Student A `tent` (5-corruption mean) | `output/A/results.csv` filtered to subset |
| Ablation 2 BS=200 | `output/B/bs/bs200/` |
| Ablation 3 STEPS=1 | `output/B/steps/steps01/` |

If any pair disagrees by >0.5 pp, a CLI override is wrong — re-check
`CORRUPTION.TYPE` quoting and `CORRUPTION.SEVERITY`.

Additional ablation signals:
- **Ablation 1:** BN-stat gain >> entropy gain in every corruption; entropy gain > 0.
- **Ablation 2:** BS=8 error > BS=200 error; ideally BS=8 > `norm` baseline (collapse).
- **Ablation 3:** STEPS=16 error > STEPS=1 error; ideally STEPS=16 ≥ `norm` baseline.

---

## Notes / coordination

- **YACS gotcha:** string-valued list overrides need quoted elements —
  `CORRUPTION.TYPE "['gaussian_noise']"`, not `[gaussian_noise]`. Numeric lists
  like `CORRUPTION.SEVERITY [5]` need no inner quotes.
- **`parse_logs.py`** recovers `arch`/`method`/`seed` from the directory layout
  and from the YACS config dump in each log. Student C's extended version also
  captures `TEST.BATCH_SIZE` and `OPTIM.STEPS` for the ablation sweeps.
- **No upstream code changes for A or B.** `cifar10c.py`, `tent.py`, `norm.py`,
  and `conf.py` are untouched by A's runs and B's sweeps. Student C's port is
  the only upstream edit, isolated on its own commits.
- **Critical path:** A → B (B's Ablation 1 reuses A's logs; B's sweeps need the
  shared checkpoint and data). C runs in parallel throughout.
- **Colab time-boxing:** BS=8 (Ablation 2) and STEPS=16 (Ablation 3) are each
  ~25–40 min on a T4 — run them last or split across multiple Colab accounts.
  If pressed for time, cap the steps sweep at STEPS=8 or reduce
  `CORRUPTION.NUM_EX 2000` for the STEPS=16 point only.

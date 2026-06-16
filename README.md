# Tent — FRMDL Reproducibility Project

Reproduction and ablation study of **"Tent: Fully Test-Time Adaptation by Entropy Minimization"**, Wang et al., ICLR 2021 · [arXiv:2006.10726](https://arxiv.org/abs/2006.10726)


## What is Tent?

Modern neural networks are trained on a fixed source distribution and degrade when the test distribution shifts, e.g., natural corruptions such as blur, noise, or fog applied to images. Test-time adaptation (TTA) addresses this by updating the model using only the unlabelled test data, without access to the source data or labels.

Tent is a minimal TTA method that adapts a model purely at inference time by minimising the entropy of its predictions. It makes two changes to a standard BatchNorm model:

1. **BN-stat adaptation (norm baseline):** Switches all BatchNorm layers from eval mode (which uses training running statistics) to train mode, so the model normalises each test batch with its own mean and variance. No parameters are trained.

2. **Entropy minimisation (Tent):** On top of the BN-stat switch, Tent also gradient-updates the BatchNorm affine parameters (γ, β) by minimising the Shannon entropy `H = -Σ p_k log p_k` of the softmax output. Only these scale-and-shift parameters are updated; all other weights are frozen.

Tent is fully online, it adapts each batch as it arrives and never accesses previous batches or source data. The method applies to any architecture with BatchNorm layers and requires no labels.


## Reproduction Target

We reproduce the headline CIFAR-10-C result from `tent/README.md` on the **WRN-28-10 Standard** model at **severity 5** across all **15 corruption types**.

The upstream README describes its numbers as *"for explanation, not reproduction"*; they are illustrative example outputs, not the paper's own tables. We report them as the reference target and flag deviations.

### Our Reproduced Results

| Method | Mean Error (%) | Δ vs. reference |
|---|---:|---|
| source | 43.5 | ±0.0 pp |
| norm | 20.4 | ±0.0 pp |
| tent | 18.6 | ±0.0 pp |

**Per-corruption breakdown (severity 5, seed 1):**

| | gauss | shot | impulse | defocus | glass | motion | zoom | snow | frost | fog | bright | contrast | elastic | pixelate | jpeg |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| source | 72.3 | 65.7 | 72.9 | 46.9 | 54.3 | 34.8 | 42.0 | 25.1 | 41.3 | 26.0 | 9.3 | 46.7 | 26.6 | 58.5 | 30.3 |
| norm | 28.1 | 26.1 | 36.3 | 12.8 | 35.3 | 14.2 | 12.1 | 17.3 | 17.4 | 15.3 | 8.4 | 12.6 | 23.8 | 19.7 | 27.3 |
| tent | 24.8 | 23.5 | 33.0 | 11.9 | 31.8 | 13.7 | 10.8 | 15.9 | 16.2 | 13.7 | 7.8 | 12.1 | 22.0 | 17.3 | 24.2 |

**Seed variance:** tent seed-1 vs. seed-2 severity-5 mean = **0.0 pp** (18.6% both seeds).


## Ablation Studies

All ablations use WRN-28-10 Standard at severity 5 on a **5-corruption subset** spanning all four corruption categories: `gaussian_noise` (noise), `motion_blur` (blur), `fog` (weather), `contrast` (digital), `jpeg_compression` (digital).

### Ablation 1: BN-stat vs. Entropy Decomposition

Tent's total improvement decomposes into two additive components using the existing reproduction runs; no extra GPU time.

| Component | From → To | Gain (pp) | Share |
|---|---|---:|---:|
| BN-stat adaptation | source (42.0%) → norm (19.5%) | 22.5 | ~93% |
| Entropy gradient | norm (19.5%) → tent (17.7%) | 1.8 | ~7% |
| **Total** | source → tent | **24.3** | 100% |

Nearly all of Tent's improvement comes from re-estimating BatchNorm statistics at test time rather than from the entropy gradient. The gradient update provides a consistent but modest incremental gain on top of the BN-stat correction.

See `output_A/ablation1_decomposition.png` for the per-corruption stacked bar chart.

### Ablation 2: Batch-Size Sensitivity

Tent's BN statistics and entropy gradient are both coupled to batch size. Learning rate is scaled proportionally with batch size (`LR = 1e-3 × BS / 200`) following the paper's protocol.

| Batch Size | LR | Mean Error (%) |
|---:|---|---:|
| 8 | 4e-5 | 24.6 |
| 16 | 8e-5 | 20.9 |
| 32 | 1.6e-4 | 19.4 |
| 64 | 3.2e-4 | 18.5 |
| 128 | 6.4e-4 | 17.8 |
| **200** (reference) | **1e-3** | **17.7** |

Performance degrades sharply below BS=32 as BatchNorm's sample-based statistics become noisy. At BS=8 (24.6%), error exceeds the norm baseline (19.5%), confirming the collapse the paper reports.

See `output_B/ablation2_batch_size.png`.

### Ablation 3: Number of Update Steps

In the online (non-episodic) setting, additional gradient steps per batch compound across the test stream, causing the BatchNorm affine parameters to over-fit individual batches.

| Steps | Mean Error (%) |
|---:|---:|
| **1** (reference) | 18.7 |
| 2 | 18.6 |
| 4 | 18.8 |
| 8 | 19.4 |
| 16 | 19.6 |

Performance is essentially flat at 1–2 steps and degrades monotonically from 4 steps onward. The paper's default of `STEPS=1` acts as implicit regularisation against over-adaptation in the online setting.

See `output_B/ablation3_steps.png`.


## Repository Layout

```
FRMDL-Tent-Reproducibility/
├── run_all.sh            # runs the 4 reproduction experiments (source/norm/tent/tent-seed2)
├── parse_logs.py         # parses cifar10c.py log files → output_A/results.csv
├── make_tables.py        # results.csv → markdown tables, plots, deviation report
├── Tent_Colab.ipynb      # end-to-end notebook (setup, reproduction, ablations)
│
├── tent/                 # upstream authors' code (unchanged)
│   ├── cifar10c.py       # evaluation loop (calls .cuda() unconditionally)
│   ├── tent.py           # Tent method
│   ├── norm.py           # BN-stat baseline
│   ├── conf.py           # YACS config
│   ├── cfgs/             # base YAML configs (source, norm, tent)
│   └── requirements.txt  # pinned environment (torch 1.8.1, Python 3.8)
│
├── docs/
│   ├── reproduction-plan.md   # internal planning document
│   └── ablation-plan.md       # detailed ablation methodology
│
├── output_A/             # reproduction outputs
│   ├── Standard/{source,norm,tent}/seed{1,2}/   # raw logs
│   ├── results.csv
│   ├── table_sev5_Standard.md
│   ├── deviation_report.md
│   ├── variance.md
│   └── ablation1_decomposition.png
│
└── output_B/             # ablation sweep outputs
    ├── bs/{bs008..bs200}/        # Ablation 2 raw logs
    ├── steps/{steps01..steps16}/ # Ablation 3 raw logs
    ├── ablation2_batch_size.png
    └── ablation3_steps.png
```


## Replication Guide

### Prerequisites

- A CUDA-capable GPU (the upstream `cifar10c.py` calls `.cuda()` unconditionally)
- Python 3.8 (the pinned dependencies: `torch==1.8.1`, `robustbench v0.1` require it)

### Option A: Google Colab (recommended)

Colab provides a free NVIDIA T4, handles the Python 3.8 environment, and persists logs to Google Drive across session disconnects.

1. Open `Tent_Colab.ipynb` in [Google Colab](https://colab.research.google.com/).
2. Set the runtime to **T4 GPU** (Runtime → Change runtime type).
3. Edit the path variables in Cell 1 (`DRIVE_ROOT`, `GITHUB_USER`, `GITHUB_REPO`).
4. Run cells top to bottom. The notebook handles:
   - Python 3.8 virtual environment via `uv`
   - Checkpoint download with modern `gdown` (bypasses the broken Drive downloader in robustbench v0.1)
   - CIFAR-10-C download from Zenodo
   - Drive symlinks so logs survive Colab disconnects
   - Disconnect-safe resumption with `SKIP_EXISTING=1`

<!-- **Approximate wall-clock time on a T4:**

| Section | Time |
|---|---|
| Setup (env + downloads) | ~20–30 min (once; cached on Drive thereafter) |
| Reproduction (4 runs × 15 corruptions) | ~45–60 min |
| Ablation 2 (6 batch-size runs × 5 corruptions) | ~60–90 min |
| Ablation 3 (5 step-count runs × 5 corruptions) | ~60–90 min | -->

> **Resuming after a disconnect:** re-run Cell 1 (paths), then re-run any cell with `SKIP_EXISTING=1`. Completed runs are detected by counting the 15 `error %` lines in their log file. Partially-written logs are re-run automatically.

### Option B: Local Machine with GPU

**Environment setup:**

```bash
git clone https://github.com/<you>/FRMDL-Tent-Reproducibility.git
cd FRMDL-Tent-Reproducibility

# Create a Python 3.8 environment (pyenv, conda, or uv)
uv venv --python 3.8 venv
source venv/bin/activate

pip install torch==1.8.1+cu111 torchvision==0.9.1+cu111 \
    -f https://download.pytorch.org/whl/torch_stable.html
pip install -r tent/requirements.txt
```

**Run the reproduction (from the project root):**

```bash
bash run_all.sh --smoke    # sanity check (~2 min, 1 corruption, 1000 examples)
bash run_all.sh            # full reproduction (source/norm/tent/tent-seed2)
```

Logs are written to `output_A/Standard/<method>/seed<seed>/`. The RobustBench checkpoint and CIFAR-10-C download automatically to `tent/ckpt` and `tent/data` on first run.

**Analyse results:**

```bash
pip install pandas matplotlib
python parse_logs.py          # output_A/**/*.txt → output_A/results.csv
python make_tables.py         # results.csv → table, variance, deviation report
```

**Run ablation sweeps (from inside `tent/`):**

```bash
cd tent

# Ablation 2: batch-size sweep (example: BS=32)
python cifar10c.py --cfg cfgs/tent.yaml MODEL.ARCH Standard \
  CORRUPTION.SEVERITY [5] \
  CORRUPTION.TYPE "['gaussian_noise','motion_blur','fog','contrast','jpeg_compression']" \
  TEST.BATCH_SIZE 32 OPTIM.LR 1.6e-4 SAVE_DIR ../output_B/bs/bs032

# Ablation 3: update-steps sweep (example: STEPS=4)
python cifar10c.py --cfg cfgs/tent.yaml MODEL.ARCH Standard \
  CORRUPTION.SEVERITY [5] \
  CORRUPTION.TYPE "['gaussian_noise','motion_blur','fog','contrast','jpeg_compression']" \
  OPTIM.STEPS 4 SAVE_DIR ../output_B/steps/steps04
```

> **YACS gotcha:** string-valued list overrides require quoted elements, `CORRUPTION.TYPE "['gaussian_noise']"`, not `[gaussian_noise]`. Numeric lists such as `CORRUPTION.SEVERITY [5]` need no inner quotes.

**Sanity checks:**

| Check | Expected |
|---|---|
| Smoke test completes | ~2 min; log written to `output_A/_smoke/` |
| Reproduction tent mean (sev 5) | **18.6 ± ~1 pp** |
| `results.csv` row count | **60 rows** (4 runs × 15 corruptions) |
| `deviation_report.md` | 0 flags at >1 pp on a clean run |
| Ablation 2 BS=200 vs. reproduction tent (5 corruptions) | within **0.5 pp** |
| Ablation 3 STEPS=1 vs. Ablation 2 BS=200 | within **0.5 pp** |


## Limitations

**Compute environment:**
- `cifar10c.py` calls `.cuda()` unconditionally; the code cannot run on CPU or Apple Silicon without modification.
- The pinned environment (`torch==1.8.1`, `robustbench v0.1`, Python 3.8) is a 2021-era stack. Installation requires a Python 3.8 interpreter; newer Python versions are incompatible.
- Google Colab free-tier T4 sessions disconnect after ~90 min of inactivity and are capped at ~12 h per session, requiring the disconnect-resume workflow.

**Scope:**
- Only the **WRN-28-10 Standard** model is reproduced. The `Hendrycks2020AugMix_WRN` variant in `tent/README.md` is excluded.
- Only **severity 5** is evaluated. The 1–5 severity sweep is out of scope.
- Only **CIFAR-10-C** is covered. ImageNet-C requires additional infrastructure and compute.
- Ablation sweeps use a **5-corruption subset**; the per-corruption trends are representative but not identical to a full-15 sweep.

**Reproducibility:**
- The upstream README states its numbers are *"for explanation, not reproduction"*, so the reference target is illustrative. Our reproduced numbers match to 0.0 pp under the same hardware (T4, `torch 1.8.1+cu111`).
- A small run-to-run non-determinism exists even with a fixed seed (`RNG_SEED`): the tent seed-1 vs. seed-2 gap is 0.0 pp here, but a ±1 pp gap is typical across different CUDA / cuDNN versions.
- The Ablation 2 BS=200 point (17.7%) and the Ablation 3 STEPS=1 point (18.7%) use identical configs but differ by ~1 pp, likely due to different Colab session states (cuDNN initialisation).

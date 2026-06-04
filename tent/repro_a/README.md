# Student A — Reproduced

Re-run the three upstream Tent configs (`source`, `norm`, `tent`) end-to-end on
WRN-28-10 (`Standard`) at **severity 5**, plus one extra `tent` run at seed 2 for
a variance number — **4 runs total** — and check the numbers against
[`../README.md`](../README.md)'s CIFAR-10-C example table.

**Reproduction target** (severity-5 mean error %, from the upstream README):

| arch | source | norm | tent |
|---|---:|---:|---:|
| WRN-28-10 `Standard` | 43.5 | 20.4 | 18.6 |

The README notes its example is "for explanation, not reproduction," so these are
the *example* numbers, not the paper's own tables — call this out in the blog.

> **Scope (reduced for Colab free-tier T4).** Single architecture, severity 5
> only, all 15 corruptions. See [`../../docs/reproduction-plan.md`](../../docs/reproduction-plan.md)
> for the full task split (A reproduced · B ablations · C code variant).

## Two environments

| Step | Where | Why |
|---|---|---|
| `run_all.sh` (the 4 runs) | **CUDA GPU box / Colab T4** | `cifar10c.py` calls `.cuda()` unconditionally; `requirements.txt` pins `torch==1.8.1` (no Apple-Silicon/MPS). Will not run on macOS. |
| `parse_logs.py`, `make_tables.py` | **anywhere (laptop OK)** | Pure text/CSV + plotting; no torch. |

> **On Google Colab?** Open [`Tent_Colab.ipynb`](Tent_Colab.ipynb) (runnable
> cells) or read [`COLAB.md`](COLAB.md) (the same flow explained) — they handle the
> Python-3.8 environment the upstream pins need, Drive persistence, and
> disconnect-resume.

## 1. Run the experiments (on the GPU box)

From the `tent/` directory:

```bash
pip install -r requirements.txt          # torch 1.8.1, robustbench v0.1, yacs, iopath

bash repro_a/run_all.sh --smoke          # day-1 sanity: ~2 min, downloads ckpt + data
bash repro_a/run_all.sh                  # full 4-run set (source/norm/tent @seed1 + tent @seed2)
```

Each run evaluates **all 15 corruptions at severity 5** (the script passes
`CORRUPTION.SEVERITY [5]` explicitly). Logs land in
`output/A/Standard/<method>/seed<seed>/`. The RobustBench checkpoint downloads to
`./ckpt` and CIFAR-10-C to `./data` on first run — Student A runs first and shares
these populated dirs with B and C.

Bring the `output/A/` tree back to wherever you do the analysis.

## 2. Analyze (laptop)

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install pandas matplotlib

python repro_a/parse_logs.py             # output/A/**.txt -> output/A/results.csv
python repro_a/make_tables.py            # results.csv -> table, variance, deviation report
```

Generated in `output/A/`:

- `results.csv` — tidy `arch, method, seed, severity, corruption, error` (60 rows on a full 4-run set).
- `table_sev5_Standard.md` — README-format severity-5 table (the headline deliverable).
- `variance.md` — seed-1 vs seed-2 severity-5 mean and the gap (tent only has both seeds).
- `deviation_report.md` — reproduced vs README, auto-flagging |Δ|>1pp (⚠️) and >2pp (❗).

`make_tables.py --seed N` selects which seed drives the headline table/report
(default `1`). `variance.md` always compares all seeds present.

## Sanity checks

- **Smoke** finishes in ~2 min and writes a log under `output/A/_smoke`.
- **Reproduction:** WRN-28-10 `tent` sev-5 mean = **18.6 ± ~1pp** (`norm` ≈ 20.4,
  `source` ≈ 43.5). A gap >2pp ⇒ investigate cuDNN/torch versions (logged at the top
  of every log via `conf.py`); upstream pinned `torch==1.8.1`.
- **Parser:** `results.csv` has **60** rows for the full 4-run set (4 × 15 corruptions
  at severity 5); fewer is fine for partial/smoke runs.
- **Tables:** `deviation_report.md` flags nothing >2pp on a clean reproduction.

## Notes / coordination

- `parse_logs.py` is the minimal log→CSV helper; it recovers `arch`/`method`/`seed`
  from the directory layout **and** from the YACS config dump in each log (so the
  seed-2 run is identified by its logged `RNG_SEED 2`, not just the path). Student C
  generalizes it to also capture `TEST.BATCH_SIZE`/`OPTIM.STEPS` for the ablation sweeps.
- This task touches **no upstream code** (`cifar10c.py`, `tent.py`, `norm.py`,
  `conf.py`) — pure re-evaluation, so authorship stays clean and there are no merge
  conflicts with Student B's experiments or Student C's `cifar10c.py` port.
- YACS gotcha: string-valued list overrides need quoted elements on the CLI —
  `CORRUPTION.TYPE "['gaussian_noise']"`, not `[gaussian_noise]` (the latter throws
  a type-mismatch error). Numeric lists like `CORRUPTION.SEVERITY [5]` are fine.

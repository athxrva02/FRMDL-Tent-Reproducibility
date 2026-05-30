# Student A — Reproduced

Re-run the three upstream Tent configs (`source`, `norm`, `tent`) end-to-end on
both WRN architectures and two seeds, and check the numbers against
[`../README.md`](../README.md)'s CIFAR-10-C example tables.

**Reproduction targets** (severity-5 mean error %, from the upstream README):

| arch | source | norm | tent |
|---|---:|---:|---:|
| WRN-28-10 `Standard` | 43.5 | 20.4 | 18.6 |
| WRN-40-2 `Hendrycks2020AugMix_WRN` | 18.3 | 14.5 | 12.1 |

The README notes its example is "for explanation, not reproduction," so these are
the *example* numbers, not the paper's own tables — call this out in the blog.

## Two environments

| Step | Where | Why |
|---|---|---|
| `run_all.sh` (the 12 runs) | **CUDA GPU box** | `cifar10c.py` calls `.cuda()` unconditionally; `requirements.txt` pins `torch==1.8.1` (no Apple-Silicon/MPS). Will not run on macOS. |
| `parse_logs.py`, `make_tables.py` | **anywhere (laptop OK)** | Pure text/CSV + plotting; no torch. |

## 1. Run the experiments (on the GPU box)

From the `tent/` directory:

```bash
pip install -r requirements.txt          # torch 1.8.1, robustbench v0.1, yacs, iopath

bash repro_a/run_all.sh --smoke          # day-1 sanity: ~2 min, downloads ckpt + data
bash repro_a/run_all.sh                  # full 2 archs x 3 methods x 2 seeds = 12 runs
```

Each run already sweeps **all 5 severities x 15 corruptions** (the cfg defaults), so
the severity-trend deliverable needs no extra runs. Logs land in
`output/A/<arch>/<method>/seed<seed>/`. The RobustBench checkpoint downloads to
`./ckpt` and CIFAR-10-C to `./data` on first run — Student A runs first and shares
these populated dirs with B and C.

Bring the `output/A/` tree back to wherever you do the analysis.

## 2. Analyze (laptop)

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install pandas matplotlib

python repro_a/parse_logs.py             # output/A/**.txt -> output/A/results.csv
python repro_a/make_tables.py            # results.csv -> tables, plot, reports
```

Generated in `output/A/`:

- `results.csv` — tidy `arch, method, seed, severity, corruption, error` (900 rows on a full run).
- `table_sev5_<arch>.md` — README-format severity-5 tables (the headline deliverable).
- `severity_trend.md` + `.png` — mean error vs severity 1–5, per method, per arch.
- `variance.md` — seed-1 vs seed-2 severity-5 mean and the gap.
- `deviation_report.md` — reproduced vs README, auto-flagging |Δ|>1pp (⚠️) and >2pp (❗).

`make_tables.py --seed N` selects which seed drives the headline tables/report
(default `1`).

## Sanity checks

- **Smoke** finishes in ~2 min and writes a log under `output/A/_smoke`.
- **Reproduction:** WRN-28-10 `tent` sev-5 mean = **18.6 ± ~1pp**. A gap >2pp ⇒
  investigate cuDNN/torch versions (logged at the top of every log via `conf.py`);
  upstream pinned `torch==1.8.1`.
- **Parser:** `results.csv` has **900** rows for the full 12-run matrix
  (2 × 3 × 2 × 5 × 15).
- **Tables:** `deviation_report.md` flags nothing >2pp on a clean reproduction.

## Notes / coordination

- `parse_logs.py` is the minimal log→CSV helper the plan nominally assigns to
  Student C. Student A needs it first (A runs first), so it lives here; C can reuse
  or supersede it.
- This task touches **no upstream code** (`cifar10c.py`, `tent.py`, `norm.py`,
  `conf.py`) — pure re-evaluation, so authorship stays clean and there are no merge
  conflicts with Student B's edits.
- YACS gotcha: string-valued list overrides need quoted elements on the CLI —
  `CORRUPTION.TYPE "['gaussian_noise']"`, not `[gaussian_noise]` (the latter throws
  a type-mismatch error). Numeric lists like `CORRUPTION.SEVERITY [5]` are fine.

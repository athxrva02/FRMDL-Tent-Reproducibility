# Running the Student A reproduction on Google Colab

> **Just run the notebook:** open [`Tent_Colab.ipynb`](Tent_Colab.ipynb) in Colab
> and run cells top to bottom. This file is the prose explanation of what those
> cells do and why; the notebook is the thing you actually execute.

Colab gives a free NVIDIA T4 GPU, which is what the unconditional `.cuda()` in
`cifar10c.py` needs. Two things make a naive run fail, and the notebook handles
both:

1. **Python version.** The upstream pins are a **Python 3.8 / 2020-era** set
   (`torch==1.8.1`; robustbench v0.1 drags in `numpy~=1.19.4`), which won't build
   on Colab's stock Python 3.11+. We make a throwaway **Python 3.8** env with `uv`
   and install `requirements.txt` *unchanged* inside it, so the reproduction stays
   faithful. The one unavoidable deviation is `torch 1.8.1+cu111` (vs the authors'
   cu102) — `deviation_report.md` flags any resulting wobble.
2. **Broken Google-Drive downloads.** RobustBench v0.1 fetches its checkpoints
   *and* the CIFAR-10-C arrays from Google Drive with a downloader that predates
   Google's confirmation page, so it silently saves HTML instead of the real files
   (`UnpicklingError: invalid load key, '<'` for checkpoints; `Cannot load file
   containing pickled data` for the `.npy`s). We route around it: **modern `gdown`
   for the two checkpoints** and the **official Zenodo tarball for CIFAR-10-C**,
   pre-fetched before the runs so they don't stall mid-matrix.

> A T4 runs the full 12-run matrix in roughly **2–4 h**. Free Colab disconnects on
> idle (~90 min) and caps sessions (~12 h), so logs stream straight to Drive and
> `run_all.sh` supports `SKIP_EXISTING=1` to resume. For the headline 18.6 number
> first, use the WRN-28-10 / seed-1 fast path at the end.

All paths below assume `DRIVE_ROOT=/content/drive/MyDrive/frmdl_tent`; change it in
the notebook's path cell if yours differs.

## 0. GPU runtime

New notebook → **Runtime ▸ Change runtime type ▸ T4 GPU**, then `!nvidia-smi -L`.

## 1. Mount Drive + define paths

```python
from google.colab import drive
drive.mount('/content/drive')
```

The notebook then defines `REPO_DIR`, `TENT_DIR`, `VENV`, `OUT_ROOT`, `DATA_DIR`,
`CKPT_DIR`, `REPO_URL` and `os.makedirs`-es the Drive dirs. Every later shell cell
interpolates these with `{...}` (more robust than `$VAR`), so **run this cell
first.**

## 2. Clone the repo (via `subprocess`)

```python
if not os.path.isdir(REPO_DIR):
    subprocess.run(['git', 'clone', '-b', BRANCH, REPO_URL, REPO_DIR], check=True)
else:
    subprocess.run(['git', '-C', REPO_DIR, 'pull', '--ff-only'], check=True)
print(sorted(os.listdir(f'{TENT_DIR}/repro_a')))
```

Passing args as a **list** (no shell) avoids the URL/destination getting glued
together. The repo is public, so no token is needed; for a private repo set
`REPO_URL = f'https://{GITHUB_USER}:<PAT>@github.com/{GITHUB_USER}/{GITHUB_REPO}.git'`.

## 3. Symlink data + checkpoints onto Drive

```python
subprocess.run(['ln', '-sfn', DATA_DIR, f'{TENT_DIR}/data'], check=True)
subprocess.run(['ln', '-sfn', CKPT_DIR, f'{TENT_DIR}/ckpt'], check=True)
```

`cifar10c.py` uses the default `./data` / `./ckpt`, which now resolve to Drive — so
the big downloads happen once and survive restarts.

## 4. Fetch both checkpoints with modern `gdown`

```python
!pip install -q -U gdown
CKPT_CORR = f'{CKPT_DIR}/cifar10/corruptions'
os.makedirs(CKPT_CORR, exist_ok=True)
CHECKPOINTS = {
    'Standard':                '1t98aEuzeTL8P7Kpd5DIrCoCL21BNZUhC',
    'Hendrycks2020AugMix_WRN': '1wy7gSRsUZzCzj8QhmTbcnwmES_2kkNph',
}
for name, gid in CHECKPOINTS.items():
    dst = f'{CKPT_CORR}/{name}.pt'
    if (not os.path.exists(dst)) or os.path.getsize(dst) < 1_000_000:
        subprocess.run(['gdown', gid, '-O', dst], check=True)
    print(name, os.path.getsize(dst) // (1024*1024), 'MB')
```

RobustBench only downloads when the `.pt` is missing, so these pre-fetched files
are used as-is. The size check re-pulls anything < 1 MB (a stale HTML page).

## 5. Fetch CIFAR-10-C from Zenodo

```python
CIFAR_C = f'{DATA_DIR}/cifar10c'
os.makedirs(CIFAR_C, exist_ok=True)
if not os.path.exists(f'{CIFAR_C}/labels.npy'):
    subprocess.run(['wget', '-q', '--show-progress',
        'https://zenodo.org/records/2535967/files/CIFAR-10-C.tar?download=1',
        '-O', '/content/CIFAR-10-C.tar'], check=True)
    subprocess.run(['tar', '-xf', '/content/CIFAR-10-C.tar',
        '-C', CIFAR_C, '--strip-components=1'], check=True)
```

`--strip-components=1` drops the tarball's top `CIFAR-10-C/` folder so the arrays
land directly in `<DATA_DIR>/cifar10c/*.npy`, exactly where robustbench looks.
Expect ~19–20 `.npy` files (15 needed + a few extras + `labels.npy`).

> The notebook version of this cell **re-downloads if the existing files are
> corrupt**, not just if they're missing — it `np.load`s `labels.npy` and
> `gaussian_noise.npy` and only trusts them if they actually open. This matters
> because robustbench's broken Drive downloader leaves HTML files that *exist* but
> won't load; a plain "skip if present" guard would keep using them.

## 6. Build the Python 3.8 environment

```python
!pip install -q uv
!uv venv --python 3.8 {VENV}
!uv pip install --python {VENV}/bin/python \
    torch==1.8.1+cu111 torchvision==0.9.1+cu111 \
    -f https://download.pytorch.org/whl/torch_stable.html
!uv pip install --python {VENV}/bin/python -r {TENT_DIR}/requirements.txt
```

Then verify the env sees the GPU and can load the checkpoints:

```python
!{VENV}/bin/python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
# expect: 1.8.1+cu111 True Tesla T4
```

## 7. Smoke test (~2 min)

```python
!PY={VENV}/bin/python OUT_ROOT={OUT_ROOT} bash {TENT_DIR}/repro_a/run_all.sh --smoke
```

`run_all.sh` `cd`s into `tent/` itself, so calling it by absolute path is fine.

## 8. Run the full matrix

```python
!PY={VENV}/bin/python OUT_ROOT={OUT_ROOT} SKIP_EXISTING=1 bash {TENT_DIR}/repro_a/run_all.sh
```

- `OUT_ROOT` on Drive ⇒ each run's log is persisted as it finishes.
- `SKIP_EXISTING=1` ⇒ re-running the cell after a disconnect skips only runs that
  **completed** — `run_all.sh` counts the 75 `error %` lines a full sweep logs
  (5 severities × 15 corruptions), so a crashed/header-only log is *not* skipped;
  it re-runs. No manual cleanup of partial runs needed.

## 9. Analyze (stock Colab Python — no venv)

`parse_logs.py` is stdlib-only and `make_tables.py` needs only pandas + matplotlib
(preinstalled in Colab's base interpreter):

```python
!python {TENT_DIR}/repro_a/parse_logs.py  --root {OUT_ROOT} --out {OUT_ROOT}/results.csv
!python {TENT_DIR}/repro_a/make_tables.py --csv  {OUT_ROOT}/results.csv --out {OUT_ROOT}
```

Then display the tables, `severity_trend.png`, `variance.md`, and
`deviation_report.md` (all on Drive under `OUT_ROOT`).

**Reproduction sanity:** WRN-28-10 `tent` severity-5 mean should be **18.6 ± ~1pp**.
A gap >2pp ⇒ note it — most likely cu111 / a slightly different cuDNN than the
authors', which `deviation_report.md` flags for you.

## Fast path: headline WRN-28-10 numbers first

```python
os.chdir(TENT_DIR)
for m in ['source', 'norm', 'tent']:
    !{VENV}/bin/python cifar10c.py --cfg cfgs/{m}.yaml RNG_SEED 1 SAVE_DIR {OUT_ROOT}/Standard/{m}/seed1
```

Then run the analysis in step 9 — it works fine on a partial `output_A/` tree.

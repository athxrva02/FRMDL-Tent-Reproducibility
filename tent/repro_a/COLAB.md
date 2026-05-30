# Running the Student A reproduction on Google Colab

> **Fastest start:** open [`Tent_Colab.ipynb`](Tent_Colab.ipynb) directly in Colab
> (it has every step below as runnable cells). This file is the prose explanation
> of what those cells do.

Colab gives a free NVIDIA T4 GPU, which is what the unconditional `.cuda()` in
`cifar10c.py` needs. The catch: the upstream stack is a **Python 3.8 / 2020-era**
pin set (`torch==1.8.1`, `numpy~=1.19.4` via robustbench v0.1), and Colab's stock
interpreter is Python 3.11+. Installing the pins there fails to build.

**Strategy:** create a throwaway **Python 3.8** environment with `uv`, install the
upstream `requirements.txt` *unchanged* inside it (so the reproduction stays
faithful), and run the experiments through that interpreter. Logs go to Google
Drive so they survive Colab disconnects.

> A T4 runs the full 12-run matrix in roughly **2–4 h**. Free Colab disconnects on
> idle (~90 min) and caps sessions (~12 h) with daily GPU limits, so the steps
> below write logs straight to Drive and `run_all.sh` supports `SKIP_EXISTING=1`
> to resume. If you just want the headline 18.6 number first, run the WRN-28-10 /
> seed-1 subset (see the last section).

## 0. Before you touch Colab (do this locally)

`repro_a/` is untracked right now. Commit and push it (and the `CLAUDE.md` fix) to
the branch Colab will pull:

```bash
git add tent/repro_a CLAUDE.md
git commit -m "Add Student A reproduction tooling"
git push origin reproducibility
```

If the GitHub repo is **private**, either make it public for the duration, or
create a fine-grained Personal Access Token (PAT) with read access — you'll paste
it into the clone URL in step 2.

## 1. New notebook + GPU runtime

New Colab notebook → **Runtime ▸ Change runtime type ▸ Hardware accelerator: T4
GPU**. Verify (a notebook cell):

```python
!nvidia-smi -L
```

## 2. Mount Drive and clone the repo

```python
from google.colab import drive
drive.mount('/content/drive')
```

```bash
%cd /content
# public repo:
!git clone -b reproducibility https://github.com/athxrva02/FRMDL-Tent-Reproducibility.git
# private repo instead (PAT = your token):
# !git clone -b reproducibility https://<PAT>@github.com/athxrva02/FRMDL-Tent-Reproducibility.git
```

## 3. Build the Python 3.8 environment

```bash
!pip install -q uv
!uv venv --python 3.8 /content/venv
# torch/torchvision for CUDA 11.1 (works on the T4; satisfies the ==1.8.1 pin):
!uv pip install --python /content/venv/bin/python \
    torch==1.8.1+cu111 torchvision==0.9.1+cu111 \
    -f https://download.pytorch.org/whl/torch_stable.html
# the rest of the upstream pins, unchanged:
!uv pip install --python /content/venv/bin/python \
    -r /content/FRMDL-Tent-Reproducibility/tent/requirements.txt
```

Confirm the GPU is visible *inside the 3.8 env*:

```bash
!/content/venv/bin/python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
# expect: 1.8.1+cu111 True Tesla T4
```

## 4. Smoke test (~2 min)

```bash
%cd /content/FRMDL-Tent-Reproducibility/tent
!PY=/content/venv/bin/python \
  OUT_ROOT=/content/drive/MyDrive/frmdl_tent/output_A \
  bash repro_a/run_all.sh --smoke
```

This downloads the RobustBench checkpoint (`./ckpt`) and CIFAR-10-C (`./data`) and
writes one log under `…/output_A/_smoke`. If it finishes clean, the environment is
good.

## 5. Run the full matrix

```bash
%cd /content/FRMDL-Tent-Reproducibility/tent
!PY=/content/venv/bin/python \
  OUT_ROOT=/content/drive/MyDrive/frmdl_tent/output_A \
  SKIP_EXISTING=1 \
  bash repro_a/run_all.sh
```

- `OUT_ROOT` on Drive ⇒ each run's log is persisted as it finishes.
- `SKIP_EXISTING=1` ⇒ if Colab disconnects, just re-run this exact cell; completed
  runs (those with a `.txt` already in their dir) are skipped and it continues.

> CIFAR-10-C is ~2.9 GB. Downloading it to `./data` (ephemeral `/content`) is fast
> but repeats after a fresh session. To avoid re-downloading, point the data/ckpt
> dirs at Drive too by appending `DATA_DIR /content/drive/MyDrive/frmdl_tent/data
> CKPT_DIR /content/drive/MyDrive/frmdl_tent/ckpt` — but note Drive FUSE reads are
> slower, so re-downloading to `/content` each session is often the better trade.

## 6. Analyze (stock Colab Python — no venv needed)

`parse_logs.py` is stdlib-only and `make_tables.py` needs only pandas + matplotlib,
both preinstalled in Colab's base interpreter:

```bash
%cd /content/FRMDL-Tent-Reproducibility/tent
!python repro_a/parse_logs.py  --root /content/drive/MyDrive/frmdl_tent/output_A \
                               --out  /content/drive/MyDrive/frmdl_tent/output_A/results.csv
!python repro_a/make_tables.py --csv  /content/drive/MyDrive/frmdl_tent/output_A/results.csv \
                               --out  /content/drive/MyDrive/frmdl_tent/output_A
```

Then inspect the deliverables (they're on Drive):

```python
from IPython.display import Markdown, Image
display(Markdown(open('/content/drive/MyDrive/frmdl_tent/output_A/table_sev5_Standard.md').read()))
display(Markdown(open('/content/drive/MyDrive/frmdl_tent/output_A/deviation_report.md').read()))
Image('/content/drive/MyDrive/frmdl_tent/output_A/severity_trend.png')
```

**Reproduction sanity:** WRN-28-10 `tent` severity-5 mean should be **18.6 ± ~1pp**.
A gap >2pp ⇒ note it — the most likely cause is exactly this: cu111 / a slightly
different cuDNN than the authors', which `deviation_report.md` flags for you.

## Fast path: just the headline number first

To get the WRN-28-10 figures (source/norm/tent, seed 1) before committing hours to
the full matrix, run the three configs directly instead of `run_all.sh`:

```bash
%cd /content/FRMDL-Tent-Reproducibility/tent
!for m in source norm tent; do \
  /content/venv/bin/python cifar10c.py --cfg cfgs/$m.yaml \
    RNG_SEED 1 SAVE_DIR /content/drive/MyDrive/frmdl_tent/output_A/Standard/$m/seed1; \
done
```

Then run the analysis in step 6 — it works fine on a partial `output_A/` tree.

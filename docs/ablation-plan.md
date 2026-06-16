# Ablation Study Plan
## Tent: Fully Test-Time Adaptation by Entropy Minimization

**Role:** Ablation study criterion  
**Depends on:** reproduction runs in `output_A/` (Ablation 1 reuses them zero-cost; reproduction must finish first)  
**Compute:** ~11 short runs on a 5-corruption subset; ~1.5–2 h T4 wall-clock total

---

## Setup: Shared CLI Context

All runs are executed from `tent/`. Every command is a pure CLI override of
`cfgs/tent.yaml` — no code changes. The 5-corruption subset spans all four
corruption groups from the paper (noise / blur / weather / digital):

```
gaussian_noise  (noise)
motion_blur     (blur)
fog             (weather)
contrast        (digital)
jpeg_compression (digital)
```

Override on every command:
```
CORRUPTION.TYPE "['gaussian_noise','motion_blur','fog','contrast','jpeg_compression']"
CORRUPTION.SEVERITY [5]
```

Reference config (`cfgs/tent.yaml` defaults after yaml load):  
`MODEL.ARCH Standard` · `TEST.BATCH_SIZE 200` · `OPTIM.LR 1e-3` · `OPTIM.STEPS 1` · `RNG_SEED 1`

**YACS gotcha:** list values must use quoted elements for strings:
`CORRUPTION.TYPE "['gaussian_noise']"` — not `[gaussian_noise]`. Numeric lists
like `CORRUPTION.SEVERITY [5]` need no inner quotes.

---

## Ablation 1 — BN-Stat vs. Entropy Decomposition

### Theoretical grounding

Tent is best understood as composing two distinct adaptation mechanisms:

1. **BN-stat adaptation** (`source → norm`): The `norm` baseline switches all
   `BatchNorm2d` layers from eval mode (uses training running-stats μ_train, σ_train)
   to train mode (uses current mini-batch statistics μ_batch, σ_batch). No
   parameters are updated — only the normalisation statistics change.  
   In code: `norm.configure_model()` sets each BN to `m.train()` and optionally
   disables running-stats entirely (`track_running_stats=False`).  
   Paper claim: this alone accounts for a large share of the domain-shift correction,
   because the training running-stats are a poor estimate of the test distribution.

2. **Entropy-gradient update** (`norm → tent`): On top of BN-stat adaptation,
   Tent also optimises the BN **affine** parameters (γ, β — `weight` and `bias`)
   via gradient descent on the mean Shannon entropy of the softmax prediction.  
   Loss: `H(f(x)) = -Σ p_k log p_k` where p = softmax(logits).  
   In code: `tent.collect_params()` walks `named_modules()` and collects only
   `BatchNorm2d.weight` and `BatchNorm2d.bias`; `configure_model()` additionally
   sets `track_running_stats=False` (same as norm's no-stats mode) and clears
   running buffers to force batch-statistic use.

The decomposition budget from the README reference numbers (severity 5, mean error %):

| segment | from | to | gain (pp) | share of total |
|---|---|---|---|---|
| BN-stat adaptation | 43.5 (source) | 20.4 (norm) | 23.1 | ~93 % |
| Entropy gradient | 20.4 (norm) | 18.6 (tent) | 1.8 | ~7 % |
| **Total** | 43.5 | 18.6 | **24.9** | 100 % |

The key narrative for the blog: almost all of Tent's gain over the frozen source
model comes from re-estimating BN statistics at test time, not from the entropy
gradient. The entropy gradient is a real but modest incremental improvement.

### Data requirement

Zero extra GPU runs. The three reproduction outputs at severity 5 over 15 corruptions
are all that is needed:

```
output_A/Standard/source/seed1/
output_A/Standard/norm/seed1/
output_A/Standard/tent/seed1/
```

Extract the 5-corruption subset by filtering `parse_logs.py`'s CSV on
`corruption ∈ {gaussian_noise, motion_blur, fog, contrast, jpeg_compression}`
and `severity == 5`.

### Deliverable: grouped/stacked bar chart

**Data to plot (per corruption + mean column):**

| corruption | source error | norm error | BN-stat gain | entropy gain |
|---|---|---|---|---|
| gaussian_noise | … | … | source−norm | norm−tent |
| motion_blur | … | … | … | … |
| fog | … | … | … | … |
| contrast | … | … | … | … |
| jpeg_compression | … | … | … | … |
| **mean** | … | … | … | … |

Stacked bar: each corruption gets two bars stacked from baseline (tent error):
- bottom segment = entropy gain = norm_err − tent_err
- top segment = BN-stat gain = source_err − norm_err
- absolute bar height = total gain over source

Optionally add a grouped bar variant showing raw error levels for the three
methods side by side.

**Interpretation:**

Quantitative: Report the per-corruption split. Note which corruptions
show a larger entropy-gradient contribution (typically those with lower signal,
e.g., noise types) and which are dominated by the BN-stat correction. Quote
the mean breakdown (~93 % BN-stat, ~7 % entropy-gradient at the README reference
numbers — update with actual reproduced numbers).

Mechanistic: Explain why BN-stat adaptation dominates. At test time,
CIFAR-10-C corruptions shift the pixel statistics significantly (fog reduces
contrast; Gaussian noise inflates variance); these shifts propagate into BN
activations. Replacing training running-stats with current-batch estimates
corrects this shift without any optimisation. The entropy gradient then fine-tunes
the affine scale/shift in a task-discriminative direction, but the first-order
correction (statistics re-estimation) does the heavy lifting. This also explains
why Tent cannot work without BN layers (`check_model` asserts `has_bn`).

---

## Ablation 2 — Batch-Size Dependence (LR Scaled Proportionally)

### Theoretical grounding

Tent's BN-stat adaptation and entropy gradient are both coupled to batch size:

**BN statistics quality**: With `track_running_stats=False`, every BN layer
computes μ and σ from the current mini-batch alone. The sample variance of a
batch-size-B sample from a distribution has variance proportional to 1/B. At
B=8 on a 512-dimensional feature map, each BN unit's estimate is extremely noisy.
Noisy normalisation corrupts the feature space and the entropy signal.

**Gradient estimate quality**: Adam with noisy gradients from a very small batch
can take harmful update steps, especially in high-curvature directions of the
affine-parameter loss landscape.

**LR scaling protocol**: The paper and upstream README advise scaling LR linearly
with batch size. This is a standard rule-of-thumb (linear scaling rule from Goyal
et al. 2017) that keeps the expected update magnitude per sample roughly constant.
Reference: `tent.yaml` sets `TEST.BATCH_SIZE 200` and `OPTIM.LR 1e-3`.  
Formula: `LR = 1e-3 × BS / 200`.

### Sweep matrix

| BS | LR | SAVE_DIR |
|---|---|---|
| 8 | 4e-5 | output_B/bs/bs008 |
| 16 | 8e-5 | output_B/bs/bs016 |
| 32 | 1.6e-4 | output_B/bs/bs032 |
| 64 | 3.2e-4 | output_B/bs/bs064 |
| 128 | 6.4e-4 | output_B/bs/bs128 |
| 200 | 1e-3 | output_B/bs/bs200 |

The `BS=200, LR=1e-3` point matches the reproduction tent run on the same 5 corruptions
and serves as the consistency check (both must give the same number).

### CLI commands (6 runs)

```bash
cd tent/

# BS=8, LR=4e-5
python cifar10c.py --cfg cfgs/tent.yaml MODEL.ARCH Standard \
  CORRUPTION.SEVERITY [5] \
  CORRUPTION.TYPE "['gaussian_noise','motion_blur','fog','contrast','jpeg_compression']" \
  TEST.BATCH_SIZE 8 OPTIM.LR 4e-5 SAVE_DIR ../output_B/bs/bs008

# BS=16, LR=8e-5
python cifar10c.py --cfg cfgs/tent.yaml MODEL.ARCH Standard \
  CORRUPTION.SEVERITY [5] \
  CORRUPTION.TYPE "['gaussian_noise','motion_blur','fog','contrast','jpeg_compression']" \
  TEST.BATCH_SIZE 16 OPTIM.LR 8e-5 SAVE_DIR ../output_B/bs/bs016

# BS=32, LR=1.6e-4
python cifar10c.py --cfg cfgs/tent.yaml MODEL.ARCH Standard \
  CORRUPTION.SEVERITY [5] \
  CORRUPTION.TYPE "['gaussian_noise','motion_blur','fog','contrast','jpeg_compression']" \
  TEST.BATCH_SIZE 32 OPTIM.LR 1.6e-4 SAVE_DIR ../output_B/bs/bs032

# BS=64, LR=3.2e-4
python cifar10c.py --cfg cfgs/tent.yaml MODEL.ARCH Standard \
  CORRUPTION.SEVERITY [5] \
  CORRUPTION.TYPE "['gaussian_noise','motion_blur','fog','contrast','jpeg_compression']" \
  TEST.BATCH_SIZE 64 OPTIM.LR 3.2e-4 SAVE_DIR ../output_B/bs/bs064

# BS=128, LR=6.4e-4
python cifar10c.py --cfg cfgs/tent.yaml MODEL.ARCH Standard \
  CORRUPTION.SEVERITY [5] \
  CORRUPTION.TYPE "['gaussian_noise','motion_blur','fog','contrast','jpeg_compression']" \
  TEST.BATCH_SIZE 128 OPTIM.LR 6.4e-4 SAVE_DIR ../output_B/bs/bs128

# BS=200, LR=1e-3  (reference — must match A's tent on these 5 corruptions)
python cifar10c.py --cfg cfgs/tent.yaml MODEL.ARCH Standard \
  CORRUPTION.SEVERITY [5] \
  CORRUPTION.TYPE "['gaussian_noise','motion_blur','fog','contrast','jpeg_compression']" \
  TEST.BATCH_SIZE 200 OPTIM.LR 1e-3 SAVE_DIR ../output_B/bs/bs200
```

**Runtime note:** BS=8 is the most expensive run in this sweep. With 10,000
images per corruption × 5 corruptions = 50,000 images, BS=8 yields 6,250
batches per corruption (31,250 total). Each batch does a forward + backward pass.
Expect ~25–40 min on T4. BS=200 yields 250 batches/corruption (~5 min).

### Consistency check

Before writing up, verify:
```
mean error at output_B/bs/bs200 (5 corruptions)
  ≈ mean of (gaussian_noise, motion_blur, fog, contrast, jpeg) rows
    from output_A/Standard/tent/seed1
```
If these differ by > 0.5pp, an override is wrong — re-check the command.

### Deliverable: mean-error vs. batch-size line plot

- X axis: batch size (log scale: 8, 16, 32, 64, 128, 200)
- Y axis: mean error (%) over the 5 corruptions
- One line for tent (the sweep), one horizontal reference line for norm at BS=200
  (no batch-size dependence baseline, since norm has no optimizer)
- Mark the BS=200 tent point as the reference

Expected shape: steep error increase as BS drops below ~64, with a sharp "cliff"
at BS=8. The norm reference line is flat and lower than small-batch tent,
illustrating that at very small batches tent is worse than norm alone.

**Explanation:** At small batch sizes, BN's sample-based statistics
become unreliable (high variance estimator of the true distribution mean/variance).
Since Tent disables running statistics entirely (`track_running_stats=False`),
there is no fallback to the training prior — every BN layer normalises with noisy
batch estimates. This corrupts the logit distribution, and the entropy gradient
then minimises entropy of a noisy signal rather than a clean one. The proportional
LR reduction partially mitigates the gradient noise but cannot fix the BN-stat
noise. The practical implication: Tent requires a minimum batch size of around
32–64 to be competitive with the simpler norm baseline.

---

## Ablation 3 — Number of Update Steps per Batch

### Theoretical grounding

The upstream `Tent` class in `tent.py` performs `OPTIM.STEPS` gradient updates
per batch:

```python
# tent.py:30-32
for _ in range(self.steps):
    outputs = forward_and_adapt(x, self.model, self.optimizer)
```

`forward_and_adapt` does one full forward pass + backward pass + Adam step on
the entropy loss, each time. When `episodic=False` (the default), the model's
BN affine parameters accumulate changes across *all* batches during the test run.

**Why more steps can hurt (online, non-episodic setting):**

Within a single batch, additional steps over-fit the affine parameters to the
entropy of that batch's specific samples. The affine parameters drift away from
the generalised direction that benefits the whole test stream. Subsequent batches
then start from a worse initialisation, and the drift compounds. This is distinct
from the episodic setting (where the model would reset) — in the online setting,
each step leaves a residual on top of all previous steps from all previous batches.

**Why the first step dominates:** After one gradient step, the entropy of the
current batch has already decreased substantially (the logit distribution is
sharpened). Additional steps yield diminishing marginal entropy reduction on the
same batch while pushing the affine parameters into a region that may not
generalise.

### Sweep matrix

All runs use tent.yaml defaults for BS/LR (BS=200, LR=1e-3).

| OPTIM.STEPS | SAVE_DIR |
|---|---|
| 1 | output_B/steps/steps01 |
| 2 | output_B/steps/steps02 |
| 4 | output_B/steps/steps04 |
| 8 | output_B/steps/steps08 |
| 16 | output_B/steps/steps16 |

The `STEPS=1` point is the same configuration as the reproduction tent run and the
BS=200 point in Ablation 2 — all three must agree on the 5-corruption mean.

### CLI commands (5 runs)

```bash
cd tent/

# STEPS=1  (reference, matches reproduction tent and bs/bs200)
python cifar10c.py --cfg cfgs/tent.yaml MODEL.ARCH Standard \
  CORRUPTION.SEVERITY [5] \
  CORRUPTION.TYPE "['gaussian_noise','motion_blur','fog','contrast','jpeg_compression']" \
  OPTIM.STEPS 1 SAVE_DIR ../output_B/steps/steps01

# STEPS=2
python cifar10c.py --cfg cfgs/tent.yaml MODEL.ARCH Standard \
  CORRUPTION.SEVERITY [5] \
  CORRUPTION.TYPE "['gaussian_noise','motion_blur','fog','contrast','jpeg_compression']" \
  OPTIM.STEPS 2 SAVE_DIR ../output_B/steps/steps02

# STEPS=4
python cifar10c.py --cfg cfgs/tent.yaml MODEL.ARCH Standard \
  CORRUPTION.SEVERITY [5] \
  CORRUPTION.TYPE "['gaussian_noise','motion_blur','fog','contrast','jpeg_compression']" \
  OPTIM.STEPS 4 SAVE_DIR ../output_B/steps/steps04

# STEPS=8
python cifar10c.py --cfg cfgs/tent.yaml MODEL.ARCH Standard \
  CORRUPTION.SEVERITY [5] \
  CORRUPTION.TYPE "['gaussian_noise','motion_blur','fog','contrast','jpeg_compression']" \
  OPTIM.STEPS 8 SAVE_DIR ../output_B/steps/steps08

# STEPS=16
python cifar10c.py --cfg cfgs/tent.yaml MODEL.ARCH Standard \
  CORRUPTION.SEVERITY [5] \
  CORRUPTION.TYPE "['gaussian_noise','motion_blur','fog','contrast','jpeg_compression']" \
  OPTIM.STEPS 16 SAVE_DIR ../output_B/steps/steps16
```

**Runtime note:** STEPS=16 is by far the heaviest run — 16 backward passes per
batch × 250 batches/corruption × 5 corruptions. Expect ~25–35 min on a T4.
If time-pressed, cap at STEPS=8 or use `CORRUPTION.NUM_EX 2000` for the steps
sweep to cut I/O time:
```bash
# time-pressed fallback (2000 ex per corruption instead of 10000)
CORRUPTION.NUM_EX 2000 OPTIM.STEPS 16 SAVE_DIR ../output_B/steps/steps16_2k
```
The error trend (not the absolute values) is what matters for the ablation.

### Deliverable: mean-error vs. steps line plot

- X axis: OPTIM.STEPS (1, 2, 4, 8, 16) — linear or log scale
- Y axis: mean error (%) over the 5 corruptions
- One line for tent; draw horizontal reference at norm error (no step-count
  dependence) to show when tent over-adapts past the norm baseline
- Annotate the STEPS=1 point as "paper default"

Expected shape: roughly flat from 1→2 steps (small gain or no change), then
rising error at 4→8→16. If STEPS=16 goes above the norm baseline, annotate
that explicitly as "over-adaptation collapse."

**Explanation:** In the online (non-episodic) setting, the BN
affine parameters persist and accumulate updates across the entire test stream.
Multiple gradient steps per batch compound this accumulation: each additional
step pushes γ and β further toward minimising entropy on a single batch, at
the cost of generality across the stream. The entropy loss for a given batch
can become near-zero after several steps (the model becomes highly confident on
those samples), but this peaked confidence does not transfer to subsequent batches
with different corruption realisations. The paper's choice of STEPS=1 is therefore
not just a compute trade-off but a regularisation choice — one step per batch
constrains the per-batch update magnitude and prevents the affine parameters from
over-fitting the stream.

---

## Execution Order and Dependencies

```
Reproduction completes → output_A/Standard/{source,norm,tent}/seed1/ exist
         │
         ├── Ablation 1: zero extra runs; filter results CSV → chart + 2 paragraphs
         │
         └── Ablation 2 + 3: 11 short runs on 5-corruption subset (independent)
                              run in parallel across Colab accounts if needed
```

Recommended run order to hit the consistency checks early:
1. Run BS=200 (Ablation 2 reference) — quick, confirms env matches reproduction
2. Run STEPS=1 (Ablation 3 reference) — should give same number as BS=200
3. Remaining 9 sweep points in any order (BS=8 and STEPS=16 are heaviest; schedule last)

---

## Verification Checklist

| Check | Pass condition |
|---|---|
| Ablation 1: norm−source gap | ~23pp mean on 5-corruption subset (consistent with 15-corruption result) |
| Ablation 1: tent−norm gap | ~1–2pp mean (small but positive for all corruptions) |
| Ablation 2: BS=200 vs reproduction tent | mean error on 5 corruptions agrees to <0.5pp |
| Ablation 3: STEPS=1 vs BS=200 | same number (same config) — cross-check logs |
| Ablation 2: BS=8 worse than norm | small-batch tent error > norm error (collapse) |
| Ablation 3: STEPS=16 worse than STEPS=1 | degradation visible (may be small at BS=200) |

If BS=200 (Ablation 2) does not match A's tent on the same 5 corruptions, check:
- `CORRUPTION.TYPE` override syntax (missing quotes around string list is a common failure)
- `CORRUPTION.SEVERITY [5]` was passed on both runs
- RNG_SEED is 1 on both (tent.yaml default)

---

## Output Directory Structure

```
output/
  A/                          (reproduction results — do not modify)
    Standard/source/seed1/
    Standard/norm/seed1/
    Standard/tent/seed1/
    Standard/tent/seed2/
  B/
    bs/
      bs008/   bs016/   bs032/   bs064/   bs128/   bs200/
    steps/
      steps01/  steps02/  steps04/  steps08/  steps16/
```

---

## Section for Report

**Three Ablations**

### 1 BN-stat vs. Entropy Decomposition
- 1 stacked-bar figure (per-corruption + mean)
- 2 paragraphs: quantitative split + mechanistic explanation
- Key takeaway: norm accounts for ~93 % of Tent's gain; the entropy gradient
  is a modest but consistent incremental improvement

### 2 Batch-Size Sensitivity
- 1 line-plot figure (error vs. BS on log-x axis)
- 1 paragraph: BN-stat noise + gradient noise; min viable BS ≈ 32–64
- Side note: link to Ablation 1 — the BN-stat quality degradation at small BS
  explains why even the norm baseline degrades at tiny batch sizes

### 3 Update-Steps Sensitivity
- 1 line-plot figure (error vs. steps)
- 1 paragraph: over-adaptation in the online (non-episodic) setting; STEPS=1
  as implicit regularisation; comparison to the norm reference line
- Optional: note that episodic mode (MODEL.EPISODIC True) would change this
  picture — per-batch reset would make more steps beneficial up to a point

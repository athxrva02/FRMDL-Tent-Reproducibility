#!/usr/bin/env bash
#
# Student A — "Reproduced" criterion: re-run the three upstream Tent configs
# end-to-end on both WRN architectures and two seeds (12 runs total).
#
# Runs on a CUDA machine (cifar10c.py calls .cuda() unconditionally). Each run
# already sweeps all 5 severities x 15 corruption types, so the severity-trend
# deliverable falls out of the same runs -- no extra runs needed.
#
# Usage (from the tent/ directory):
#   bash repro_a/run_all.sh            # full 12-run matrix
#   bash repro_a/run_all.sh --smoke    # quick <2 min sanity check only
#
# Logs/results land under ./output/A/<arch>/<method>/seed<seed>/.
# The RobustBench checkpoint downloads to ./ckpt and CIFAR-10-C to ./data on
# first run -- Student A runs first and shares these populated dirs with B/C.

set -euo pipefail

# Resolve repo paths so the script works regardless of where it is invoked from,
# but cifar10c.py expects to run from tent/, so cd there.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TENT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$TENT_DIR"

ARCHS=("Standard" "Hendrycks2020AugMix_WRN")
METHODS=("source" "norm" "tent")
SEEDS=(1 2)

if [[ "${1:-}" == "--smoke" ]]; then
  echo "[run_all] smoke test: source / Standard / gaussian_noise sev5 / 1000 ex"
  # Note: YACS coerces list overrides via literal_eval, so string-valued lists
  # must have quoted elements: ['gaussian_noise'], not [gaussian_noise].
  python cifar10c.py --cfg cfgs/source.yaml \
    MODEL.ARCH Standard \
    CORRUPTION.SEVERITY "[5]" \
    CORRUPTION.TYPE "['gaussian_noise']" \
    CORRUPTION.NUM_EX 1000 \
    SAVE_DIR ./output/A/_smoke
  echo "[run_all] smoke test done -- check ./output/A/_smoke for the log."
  exit 0
fi

total=$(( ${#ARCHS[@]} * ${#METHODS[@]} * ${#SEEDS[@]} ))
i=0
for arch in "${ARCHS[@]}"; do
  for method in "${METHODS[@]}"; do
    for seed in "${SEEDS[@]}"; do
      i=$(( i + 1 ))
      save_dir="./output/A/${arch}/${method}/seed${seed}"
      echo "[run_all] (${i}/${total}) arch=${arch} method=${method} seed=${seed} -> ${save_dir}"
      python cifar10c.py --cfg "cfgs/${method}.yaml" \
        MODEL.ARCH "${arch}" \
        RNG_SEED "${seed}" \
        SAVE_DIR "${save_dir}"
    done
  done
done

echo "[run_all] all ${total} runs complete. Next: python repro_a/parse_logs.py"

#!/usr/bin/env bash
#
# Re-run the three upstream Tent configs end-to-end on WRN-28-10 (Standard) at
# severity 5, plus one extra tent run at seed 2 for a variance number
# (4 runs total).
#
# Runs on a CUDA machine (cifar10c.py calls .cuda() unconditionally). Each run
# evaluates all 15 corruption types at severity 5 only (CORRUPTION.SEVERITY [5]
# is passed explicitly below; the cfg default would sweep all 5 severities).
#
# Usage (from the tent/ directory):
#   bash repro_a/run_all.sh            # full 4-run set
#   bash repro_a/run_all.sh --smoke    # quick <2 min sanity check only
#
# Environment knobs (all optional):
#   PY            python to use         (default: python; on Colab set to the
#                                        venv python, e.g. /content/venv/bin/python)
#   OUT_ROOT      output root dir       (default: ./output/A; point at Google
#                                        Drive on Colab so logs survive restarts)
#   SKIP_EXISTING skip a run whose      (default: 0; set to 1 to resume after a
#                 SAVE_DIR already       Colab disconnect without redoing runs)
#                 has a .txt log
#
# Logs/results land under $OUT_ROOT/<arch>/<method>/seed<seed>/.
# The RobustBench checkpoint downloads to ./ckpt and CIFAR-10-C to ./data on
# first run -- subsequent runs reuse the cached dirs.

set -euo pipefail

# Resolve repo paths so the script works regardless of where it is invoked from,
# but cifar10c.py expects to run from tent/, so cd there.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TENT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$TENT_DIR"

PY="${PY:-python}"
OUT_ROOT="${OUT_ROOT:-./output/A}"
SKIP_EXISTING="${SKIP_EXISTING:-0}"

ARCH="Standard"
# (method seed) jobs for the reduced plan: source/norm/tent at seed 1, plus a
# tent seed-2 run for the variance deliverable. Only tent gets a second seed.
JOBS=("source 1" "norm 1" "tent 1" "tent 2")

if [[ "${1:-}" == "--smoke" ]]; then
  echo "[run_all] smoke test: source / Standard / gaussian_noise sev5 / 1000 ex"
  # Note: YACS coerces list overrides via literal_eval, so string-valued lists
  # must have quoted elements: ['gaussian_noise'], not [gaussian_noise].
  "$PY" cifar10c.py --cfg cfgs/source.yaml \
    MODEL.ARCH Standard \
    CORRUPTION.SEVERITY "[5]" \
    CORRUPTION.TYPE "['gaussian_noise']" \
    CORRUPTION.NUM_EX 1000 \
    SAVE_DIR "${OUT_ROOT}/_smoke"
  echo "[run_all] smoke test done -- check ${OUT_ROOT}/_smoke for the log."
  exit 0
fi

total=${#JOBS[@]}
i=0
for job in "${JOBS[@]}"; do
  read -r method seed <<< "${job}"
  i=$(( i + 1 ))
  save_dir="${OUT_ROOT}/${ARCH}/${method}/seed${seed}"
  # Skip only runs that actually COMPLETED. A complete run logs 15 "error %"
  # lines (15 corruptions at severity 5); a crashed run leaves a header-only
  # log, so a plain file-existence check would wrongly skip it.
  if [[ "${SKIP_EXISTING}" == "1" ]]; then
    done_lines=$(cat "${save_dir}"/*.txt 2>/dev/null | grep -c "error %" || true)
    if [[ "${done_lines:-0}" -ge 15 ]]; then
      echo "[run_all] (${i}/${total}) skip (complete, ${done_lines} results): ${save_dir}"
      continue
    fi
  fi
  # Fresh attempt: drop any partial log left by a previous crash so the
  # completion count stays accurate (otherwise two partials could sum to
  # >=15 and be falsely treated as complete on the next resume).
  mkdir -p "${save_dir}"
  rm -f "${save_dir}"/*.txt 2>/dev/null || true
  echo "[run_all] (${i}/${total}) arch=${ARCH} method=${method} seed=${seed} -> ${save_dir}"
  "$PY" cifar10c.py --cfg "cfgs/${method}.yaml" \
    MODEL.ARCH "${ARCH}" \
    RNG_SEED "${seed}" \
    CORRUPTION.SEVERITY "[5]" \
    SAVE_DIR "${save_dir}"
done

echo "[run_all] all ${total} runs complete. Next: python repro_a/parse_logs.py --root ${OUT_ROOT}"

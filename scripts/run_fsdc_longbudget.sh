#!/usr/bin/env bash
# FSDC long-budget sign-flip probe (design doc 22 §8 follow-up): does
# SFS-FF/FSD-Down turn positive given 3x the pilot budget?
#   - none-24k: stock DEIMv2-L architecture + pretrained ViT-S (reference)
#   - l2enc-24k: SFS-FF + embedded FSD-Down (DBFSAF stays off — negative
#     in the pilot)
# Identical recipe except the fsdc switches (same seed, LR cosine, aug
# window, EMA). ~24,266 steps each (~4.4 h/arm on the 4090D).
# Waits for the running follow-up queue (baseline _v2) to release the GPU.
set -u
cd "$(dirname "$0")/.."
PY=.venv/bin/python

echo "[wait] for the current queue to release the GPU..."
while pgrep -f "scripts/visdrone_deimv2_800.py|scripts/visdrone_fsdc_800.py" >/dev/null 2>&1; do
    sleep 60
done
echo "[wait] done at $(date +%H:%M:%S)"

run_arm() {
    local name="$1"; shift
    local log="runs/${name}_console.log"
    if grep -q "Task training finished" "runs/${name}/train.log" 2>/dev/null; then
        echo "[skip] ${name} already finished"
        return 0
    fi
    echo "[run ] ${name} -> ${log}"
    $PY scripts/visdrone_fsdc_800.py "$@" 2>&1 | tee "${log}"
}

run_arm visdrone_fsdc_none-24k_800_30ep \
    --model fsdc-l --tag none-24k --baseline --epochs 30
run_arm visdrone_fsdc_l2enc-24k_800_30ep \
    --model fsdc-l --tag l2enc-24k --no-dbfsaf --epochs 30
echo "long-budget arms done"

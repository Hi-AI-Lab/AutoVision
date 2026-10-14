#!/usr/bin/env bash
# FSDC PR-5 follow-up arms (design doc 22 §8): the fsdc-none stock-
# architecture twin, the fsdc-s warm-start rerun (first attempt crashed
# on a 384-into-192 backbone weights mismatch — fixed in the script),
# and the deimv2-s baseline rerun on the fixed engine (the original run's
# first 1000 steps were trained in eval mode; the old dir is preserved
# as *_full_v1evalmode).
set -u
cd "$(dirname "$0")/.."
PY=.venv/bin/python

run_fsdc() {
    local name="$1"; shift
    local log="runs/${name}_console.log"
    if grep -q "Task training finished" "runs/${name}/train.log" 2>/dev/null; then
        echo "[skip] ${name} already finished"
        return 0
    fi
    echo "[run ] ${name} -> ${log}"
    $PY scripts/visdrone_fsdc_800.py "$@" 2>&1 | tee "${log}"
}

# Arm (s): warm-start sanity — decoder + 192 ViT from the DEIMv2-S COCO
# conversion; no external backbone weights (fixed in this pass).
run_fsdc visdrone_fsdc_s_800_full --model fsdc-s

# Arm (none): stock DEIMv2-L architecture twin — HF 1x1 fusion + original
# encoder, same pretrained ViT-S. Paired with l2/l2-enc it isolates each
# FSDC module's NET contribution.
run_fsdc visdrone_fsdc_none_800_full --model fsdc-l --tag none --baseline

# Arm (a-v2): DEIMv2-S baseline rerun on the fixed engine (clean anchor).
BASELINE_DIR=runs/visdrone_deimv2_dinov3-s_800_full
if grep -q "Task training finished" "${BASELINE_DIR}/train.log" 2>/dev/null; then
    echo "[skip] baseline already finished"
else
    echo "[run ] deimv2-s baseline (fixed engine) -> runs/visdrone_deimv2_dinov3-s_800_full_console.log"
    $PY scripts/visdrone_deimv2_800.py --model deimv2-dinov3-s 2>&1 | tee runs/visdrone_deimv2_dinov3-s_800_full_console.log
fi
echo "all follow-up arms done"

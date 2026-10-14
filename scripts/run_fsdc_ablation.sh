#!/usr/bin/env bash
# FSDC-DETR PR-5 ablation arms on a single GPU (design doc 22 §8).
# Sequential execution, one arm after another; each arm's console output
# lands in runs/<name>_console.log. An arm is skipped when its run
# directory already contains a finished train.log.
set -u
cd "$(dirname "$0")/.."
PY=.venv/bin/python

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

# Arm c-v2: paper full form — pretrained DINOv3 ViT-S backbone + DBFSAF
# fusion + SFS-FF/FSD-Down encoder (doc 22 v1.2 erratum architecture).
run_arm visdrone_fsdc_l2_800_full --model fsdc-l --tag l2
# Arm enc-v2: encoder-only ablation — isolates SFS-FF/FSD-Down.
run_arm visdrone_fsdc_l2_800_enc --model fsdc-l --tag l2-enc --no-dbfsaf
# Arm (s): warm-start sanity — decoder from the DEIMv2-S COCO conversion.
run_arm visdrone_fsdc_s_800_full --model fsdc-s
echo "all fsdc arms done"

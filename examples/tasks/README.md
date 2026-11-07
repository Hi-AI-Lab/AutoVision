# Task examples

One small, runnable example per task family. They live outside `src/` on
purpose: example code is **not** part of the installed wheel, and training
examples import the CLI layer, which the package layering contract forbids
inside `autovision.tasks`.

Every example:

- writes a tiny synthetic dataset when `--data` is omitted (except detection,
  which uses the bundled `coco8` smoke set);
- defaults to the GTR family: inference loads the official MIT checkpoints,
  training runs a tiny random-init GTR configuration for 1-3 steps (needs
  `.[gtr]` + CUDA; only panoptic/classification have no GTR route and stay
  CPU-friendly);
- prints the checkpoint, exported model, validation metrics and class map;
- can render one prediction with `--predict-image` and copy the exported
  `.pt` with `--export`;
- can export the trained model to ONNX with `--export-onnx PATH`
  (parity-gated) and serve that artifact on any device through the
  inference scripts' `--onnx PATH`;
- supports `--dry-run`, which only writes/prints the demo data config.

## Install

```bash
pip install -e ".[gtr,download,export]"   # GTR examples (CUDA + Triton required)
# panoptic/classification only (CPU-friendly):
pip install -e ".[transformers,timm,export]"
```

## Quick start

```bash
# Inference-only: released/official weights, no training
python examples/tasks/detection.py
python examples/tasks/oriented_detection.py
python examples/tasks/instance_segmentation.py
python examples/tasks/semantic_segmentation.py
python examples/tasks/pose_estimation.py
python examples/tasks/depth_estimation.py
# Panoptic/classification need an exported checkpoint:
python examples/tasks/panoptic_segmentation.py \
    --weights runs/examples/panoptic_segmentation_train/exported_models/exported_last.pt

# Training examples (tiny synthetic data / coco8)
python examples/tasks/detection_train.py --steps 2
python examples/tasks/classification_train.py --steps 2
python examples/tasks/instance_segmentation_train.py --dry-run

# After training, render one image and copy the exported model
python examples/tasks/semantic_segmentation_train.py --steps 2 \
    --predict-image path/to/image.png --export runs/semantic.pt

# ONNX round trip: export the trained model (parity-gated), then serve it
# on any device — the forward runs in the ONNX session, pre/post stay in
# the torch network, so a CPU-only machine can serve GTR artifacts
python examples/tasks/classification_train.py --steps 2 \
    --export-onnx runs/classification.onnx
python examples/tasks/classification.py \
    --weights runs/examples/classification_train/exported_models/exported_last.pt \
    --onnx runs/classification.onnx
```

## Subsystem / coverage examples

Beyond the per-task train/infer pairs, these examples exercise subsystems and
code paths the default happy-path scripts do not reach — every detection
backend family, the TensorRT / OpenVINO / INT8 export paths, the FastAPI
reference serving stack, the plugin protocol, and self-supervised pretraining
plus embedding extraction:

```bash
# Every detection backend family (gtr/rfdetr via released weights,
# rtdetr/detr via a 2-step coco8 smoke train + inference)
python examples/tasks/detection_backends.py

# Export a trained .pt to ONNX / TensorRT / OpenVINO / INT8
python examples/tasks/export_demo.py \
    --weights runs/examples/classification_train/exported_models/exported_last.pt --int8

# A GTR detection checkpoint may be refused by the TRT/OpenVINO parity gates:
# the script then exits 1; add --allow-skips to accept those known-unhealthy
# targets explicitly.

# Reference serving: wrap a checkpoint in the FastAPI app, hit /predict + /healthz
python examples/tasks/serving_demo.py

# Plugin protocol: register / enumerate / retrieve a plugin
python examples/tasks/plugin_demo.py

# Tracking primitives: BYTETracker association + line/polygon zones + smoother
python examples/tasks/tracking_demo.py

# Hyperparameter search: random / grid / asha (synthetic objective, offline)
python examples/tasks/hpo_demo.py

# Model registry: filesystem backend + deployment gates + rollback + audit
python examples/tasks/registry_demo.py

# Compliance: license normalization + compliance report + gate
python examples/tasks/compliance_demo.py

# Self-supervised pretraining + embedding extraction
python examples/tasks/pretrain_demo.py
```

> Known framework limits surfaced by these examples: on TensorRT 11.4 the GTR
> pure-ONNX graph is miscompiled, so a detection (`gtr`) TensorRT export is
> rejected by the parity gate; a GTR detection OpenVINO IR export can also
> fail its numerical parity gate on a random-init smoke checkpoint. Those are
> `src/autovision` boundaries, not example bugs (see `docs/export.md` and
> `docs/known_limits.md`). `export_demo.py` prints them as `SKIPPED` and exits
> non-zero unless `--allow-skips` is passed. ONNX artifacts and
> compiled-backend *inference* are unaffected: `decode_detection_outputs`
> accepts the flat `list` a compiled runner returns, so `--onnx` /
> `--openvino` / `--engine` serving works with a matching artifact.

> **`--onnx` needs `--weights` for trained artifacts**: without `--weights`,
> the inference script builds its task's DEFAULT checkpoint network (e.g.
> the 19-class Cityscapes semseg model, the 80-class COCO detector) purely
> for pre/post-processing, and the identity gate rejects an artifact whose
> class count differs ("Artifact has N classes but the loaded network has
> M"). Always pair `--onnx` with `--weights <exported.pt>` from the same
> training run — the default network then only supplies the (class-count-
> agnostic) pre/post contract.

`inference_demo.py` is a compatibility alias for `detection.py`. All inference
examples default to the CC0 `assets/airport_aerial.jpg` (see `assets/README.md`
for provenance); pass `--image` for task-specific imagery. The OBB example uses
the same overhead image because its checkpoint is DOTA-trained.

Outputs default to `runs/examples/<task>/` (inference) or
`runs/examples/<task>_train/` (training); generated demo data goes to
`runs/examples/_demo_data/<task>/` (outside `--out`, so `--overwrite` does not
delete it).

## YOLO baselines (optional, AGPL)

The `yolo/` subdirectory adds five Ultralytics YOLO inference examples
(detection / segmentation / pose / OBB / classification) using the same
images. They require the **AGPL** extra:

```bash
pip install -e ".[ultralytics]"
python examples/tasks/yolo/detection.py
python examples/tasks/yolo/obb.py        # airport_aerial.jpg
```

See [`yolo/README.md`](yolo/README.md) for the full table and flags. Detection,
segmentation and pose weights are auto-downloaded from the ModelScope
`Ultralytics/YOLO11` mirror; OBB/classification fall back to Ultralytics or a
local `--model` path. These examples are separate from AutoVision's
Apache-2.0 core.

## Inference flags (all `<task>.py`)

| Flag | Meaning |
|---|---|
| `--image PATH` | input image (default: bundled `assets/airport_aerial.jpg`) |
| `--weights PATH` | exported `.pt` from `<task>_train.py` |
| `--onnx PATH` | exported `.onnx` artifact: run the forward through it (any device; network supplies pre/post only) |
| `--engine PATH` | **detection.py only**: TensorRT `.engine` artifact via backend=`tensorrt` (needs CUDA) |
| `--openvino PATH` | **detection.py only**: OpenVINO `.xml` IR artifact via backend=`openvino` |
| `--class-names PATH` | optional JSON `{class_id: name}` for exported checkpoints |
| `--threshold F` | score threshold (default: 0.3) |
| `--device auto\|cpu\|cuda` | inference device |
| `--out DIR` | output directory (default: `runs/examples/<task>`) |

`--engine` / `--openvino` are wired only in `detection.py` (and its
`inference_demo.py` alias); the other task scripts expose `--onnx` alone.

The ONNX artifact must match the constructed network (image size, class
count — the loader verifies the artifact's embedded metadata and exits with
a one-line error otherwise). Exported-demo artifacts match the
`--weights` networks trained with the same defaults; serving them against
a released-checkpoint network (different geometry) exits with the mismatch.
Detection additionally accepts `--model KEY` (default `detr/gtrvit-s`;
pass `rfdetr/rf-detr-nano` for a CPU-friendly released checkpoint) and the
GTR tasks accept `--checkpoint NAME` / `--variant s|m|l|x` (official MIT
assets). Panoptic/classification have no released task checkpoint and
require `--weights`.

## Common flags (training examples)

| Flag | Meaning |
|---|---|
| `--data PATH` | real data YAML/JSON config (or directory, where supported) |
| `--out DIR` | run directory (default `runs/examples/<task>_train`) |
| `--steps N` | training steps (default 2) |
| `--batch-size N` | batch size (default 2) |
| `--seed N` | training seed |
| `--num-classes N` | class count for classification / instance / semantic |
| `--overwrite` | replace an existing `--out` |
| `--dry-run` | write/print the demo data config; no training |
| `--predict-image PATH` | run the exported model and save `prediction.png` |
| `--export PATH` | copy the exported `.pt` (ONNX/TRT via `autovision-cli export`) |
| `--export-onnx PATH` | also export to ONNX (parity-gated); serve it with `<task>.py --weights <exported.pt> --onnx <artifact>` — `--weights` is REQUIRED unless the artifact's class count matches that task's default checkpoint (see the note below) |
| `--threshold F` | prediction threshold (default 0.3) |

## Task matrix

| Task | Inference | Training | Default inference weights |
|---|---|---|---|
| Detection | `detection.py` | `detection_train.py` | `detr/gtrvit-s` (GTR, COCO) |
| Oriented detection | `oriented_detection.py` | `oriented_detection_train.py` | `gtrobb_s_dota` (GTR, DOTA) |
| Instance segmentation | `instance_segmentation.py` | `instance_segmentation_train.py` | `gtrseg_s_coco` (GTR, COCO) |
| Panoptic segmentation | `panoptic_segmentation.py` | `panoptic_segmentation_train.py` | requires `--weights` |
| Semantic segmentation | `semantic_segmentation.py` | `semantic_segmentation_train.py` | `gtrsemseg_s_cityscapes` (GTR) |
| Pose estimation | `pose_estimation.py` | `pose_estimation_train.py` | `gtrpose_s_coco` (GTR, COCO) |
| Depth estimation | `depth_estimation.py` | `depth_estimation_train.py` | `gtrdepth_s` (GTR) |
| Classification | `classification.py` | `classification_train.py` | requires `--weights` |

Training defaults: `detection_train.py` uses coco8; the other `_train.py`
scripts generate tiny synthetic data (see `--dry-run`). The GTR training
examples (detection / OBB / instance / semantic / pose / depth) run a tiny
random-init GTR configuration and need `pip install -e ".[gtr]"` + CUDA;
panoptic/classification training stays CPU-friendly. GTR inference also
needs the same stack; official weights download via
`scripts/download_gtr_assets.py`.

## Real data contracts

```python
# detection: COCO / YOLO data.yaml
data = "data.yaml"

# oriented detection: YOLO-OBB
data = {"format": "yolo-obb", "path": "obb_root", "train": "images/train",
        "val": "images/val", "names": {0: "plane", 1: "ship"}}

# instance segmentation: COCO instances (segmentation polygons/RLE)
data = {"train": {"annotations": "instances_train.json", "images": "train2017"},
        "val":   {"annotations": "instances_val.json",   "images": "val2017"}}

# panoptic: masks PNG (segment ids, 0 = unlabeled) + segments JSON
data = {"train": {"images": "train/images", "masks": "train/masks",
                  "segments": "train/panoptic.json"},
        "val":   {"images": "val/images", "masks": "val/masks",
                  "segments": "val/panoptic.json"}}

# semantic: paired PNGs; 255 is ignored by loss and mIoU
data = {"train": {"images": "images/train", "masks": "masks/train"},
        "val":   {"images": "images/val", "masks": "masks/val"}}

# pose: COCO person_keypoints
data = {"train": {"annotations": "person_keypoints_train2017.json",
                  "images": "train2017"},
        "val":   {"annotations": "person_keypoints_val2017.json",
                  "images": "val2017"}}

# depth: .npy (float metres) or 16-bit PNG
data = {"train": {"images": "images/train", "depths": "depths/train"},
        "val":   {"images": "images/val", "depths": "depths/val"}}

# classification: ImageFolder layout
data = {"train": "data/train", "val": "data/val"}
```

## Notes

- Demo metrics after 1-3 steps are meaningless by design; they only prove the
  pipeline runs. Use real data and the recipes in `docs/benchmarks.md` for
  numbers.
- Pose/depth were always GTR; detection / OBB / instance / semantic now
  default to the GTR family too. Without the CUDA stack, use `--dry-run` to
  inspect the data contract, or run the panoptic/classification examples on
  CPU (or any inference script with `--weights` from a non-GTR training run).
- `--export` only copies `.pt`; use `autovision-cli export` for ONNX,
  OpenVINO and TensorRT parity gates.
- `assets/airport_aerial.jpg` is CC0 (see `assets/README.md`); it ships in the
  sdist so the examples run from a source distribution.

# AutoVision

Enterprise-grade visual model training framework: train → evaluate → export →
deploy in one chain. Its flagship model family is **GTR** (MIT): one shared
backbone covering detection, oriented detection, instance / semantic
segmentation, depth and pose, with all 26 official weights registered and
sha256-pinned. The Python API and `autovision-cli` share one strict Pydantic
validation surface.

> **Distribution name**: the PyPI package is `autovision-framework`, while the
> import package stays `autovision` and the CLI stays `autovision-cli`. An
> unrelated `autovision` 0.0.1 package exists on PyPI — make sure you install
> `autovision-framework`.

**Links**: [Source repository](https://github.com/Hi-AI-Lab/AutoVision) ·
[Documentation](https://hi-ai-lab.github.io/AutoVision/) ·
[Changelog](https://github.com/Hi-AI-Lab/AutoVision/blob/main/CHANGELOG.md)

## Installation

Python ≥ 3.10:

```bash
pip install autovision-framework

# GTR route: backbone + official weight download + ONNX export (CUDA required)
pip install "autovision-framework[gtr,download,export]"

# CPU-friendly detection route
pip install "autovision-framework[rfdetr,export]"
```

Model packages load lazily per extra; a missing extra yields an empty registry
listing instead of an error. See the
[repository README](https://github.com/Hi-AI-Lab/AutoVision#readme)
for the full extras table and the GTR task matrix.

## Quick start (GTR)

Download the official MIT weights, then run inference — no training:

```bash
python scripts/download_gtr_assets.py --task det   # or --all for all 26
```

```python
import autovision

model = autovision.detection("detr/gtrvit-s", pretrained=True)  # COCO AP 53.6
result = model.predict("image.jpg")   # {'bboxes': ..., 'scores': ..., 'labels': ...}
model.export("model.onnx")            # ONNX export, parity gate on by default
```

Finetune from official weights (pose shown; five more tasks share the shape):

```python
result = autovision.train_pose_estimation(
    out="runs/pose", data=data,       # COCO person_keypoints JSON + images
    model="gtr/s",
    model_args={"checkpoint": "gtrpose_s_coco"},   # omit for random init
)
print(result.best_exported_model_path, result.metrics)
```

Equivalent CLI (same Pydantic validation):

```bash
autovision-cli list-models --task detection --pretrained
autovision-cli train-pose-estimation out=runs/pose data=pose.yaml model=gtr/s \
    model_args='{"checkpoint": "gtrpose_s_coco"}'
autovision-cli predict model=runs/pose/exported_models/exported_best.pt images=dir/ out=preds/
```

CPU smoke path (no GPU): `rtdetr/resnet18` on the bundled `coco8` dataset via
`autovision.detection("rtdetr/resnet18").train(...)`.

## Common extras

| extra | capability |
|---|---|
| `gtr` | GTR GLA backbone (Triton forward, CUDA required) |
| `download` | on-demand GTR upstream weight download (huggingface_hub) |
| `export` | ONNX export + numerical parity gate |
| `transformers` | DINOv2 / DINOv3 backbones (HuggingFace weights) |
| `timm` | full timm backbone zoo (arbitrary input channels) |
| `rfdetr` | RF-DETR detection family (CPU-friendly released weights) |
| `openvino` | OpenVINO IR export and runtime |
| `video` | video / RTSP reference inference runtime |
| `serving` | FastAPI reference serving (`/predict`, `/metrics`) |
| `dev` | pytest / ruff (same as CI) |

GTR exports ONNX / TensorRT / OpenVINO via the pure-ONNX GLA lowering
(fixed-size graphs); the eager forward needs the `gtr` extra on CUDA.

## License

Apache-2.0 (AutoVision original code; full text:
<https://www.apache.org/licenses/LICENSE-2.0>). Third-party component and
weight provenance ship in `NOTICE` and `THIRD_PARTY_LICENSES.md`. GTR weights
are MIT. Restricted-license weights (DINOv2 CC-BY-NC 4.0 / DINOv3 License)
are not downloaded by default; user-held weights may not be redistributed.

Full documentation (model zoo, benchmarks, export details, known limits) is
available on the
[documentation site](https://hi-ai-lab.github.io/AutoVision/) and in the
`docs/` directory of the source repository.

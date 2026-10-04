#!/usr/bin/env bash
# Container smoke test: 8 synthetic images -> 1 epoch pretrain -> artifacts.
set -euo pipefail
python - << 'PY'
import sys, tempfile, pathlib
sys.path.insert(0, "tests")
from conftest import create_images
tmp = pathlib.Path(tempfile.mkdtemp())
create_images(tmp / "images", num_images=8, size=(64, 64))
import autovision
result = autovision.pretrain(
    out=tmp / "run", data=tmp / "images", model="torchvision/resnet18",
    epochs=1, batch_size=4, num_workers=0,
    method_args={"queue_size": 4, "teacher_pretrained": False},
)
assert result.checkpoint_path.exists() and result.exported_model_path.exists()
print("smoke test OK:", result.out_dir)
PY

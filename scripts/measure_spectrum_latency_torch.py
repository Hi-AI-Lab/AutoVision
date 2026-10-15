# Copyright (c) 2026 AutoVision contributors. Apache-2.0.
"""Torch-backend latency for the spectrum (doc 29 §4.2, torch column).

p50 over ``iters`` after warmup; fp16 = autocast, fp32 = full precision.
Hardware/software get logged into docs/benchmarks.md next to the numbers.
"""

from __future__ import annotations

import json
import statistics
import time

import torch

from autovision.tasks.detection.detr.task_model import DETRObjectDetection

MODELS = ["detr/n", "detr/s", "gtr-s", "gtrvit-s"]
DTYPES = {"fp32": None, "fp16": torch.float16, "bf16": torch.bfloat16}


def measure(model_key: str, batch_size: int, precision: str, iters: int = 50) -> float:
    dtype = DTYPES[precision]
    net = DETRObjectDetection(
        num_classes=80,
        model_name=model_key,
        image_size=(640, 640),
        load_weights=False,  # latency is architecture-bound, not weight-bound
    ).cuda().eval()
    x = torch.randn(batch_size, 3, 640, 640, device="cuda")
    for _ in range(20):
        with torch.no_grad(), torch.autocast("cuda", dtype=dtype, enabled=dtype is not None):
            net.forward_backend(net.preprocess_batch(x))
    torch.cuda.synchronize()
    samples: list[float] = []
    for _ in range(iters):
        start = time.perf_counter()
        with torch.no_grad(), torch.autocast("cuda", dtype=dtype, enabled=dtype is not None):
            net.forward_backend(net.preprocess_batch(x))
        torch.cuda.synchronize()
        samples.append((time.perf_counter() - start) * 1000)
    del net
    torch.cuda.empty_cache()
    return round(statistics.median(samples), 2)


def main() -> None:
    results: dict[str, float | str] = {}
    for key in MODELS:
        for batch_size in (1, 4):
            for precision in ("fp32", "fp16"):
                label = f"{key}|bs{batch_size}|{precision}"
                try:
                    results[label] = measure(key, batch_size, precision)
                except Exception as error:  # noqa: BLE001 - report and continue
                    results[label] = f"FAILED: {type(error).__name__}"
                print(label, "->", results[label])
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()

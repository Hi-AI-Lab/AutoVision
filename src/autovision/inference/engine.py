# Copyright (c) 2026 AutoVision contributors. Apache-2.0.
"""Backend-agnostic inference engine (design doc 08 §12 reference runtime).

One contract, three forwards: the network's own pre/post-processing stays
in Python (``preprocess_image`` -> ``preprocess_batch`` ->
``postprocess``), and only the forward call is swapped — torch eager,
ONNX Runtime, or a TensorRT engine. This is the same contract the export
parity gate verifies (deployed-torch reference vs the ONNX graph), so the
runtime's "same pre/post as evaluation" promise is structural rather than
re-implemented.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Sequence

import torch

from autovision.errors import AutoVisionError
from autovision.keys import (
    BBOXES,
    DEPTH,
    KEYPOINTS,
    LABELS,
    MASK,
    MASKS,
    POLYGONS,
    SCORES,
    SEGMENT_IDS,
)
from autovision.tasks.task_model import TaskNetwork
from autovision.tasks.task_model_helpers import load_task_model
from autovision.types import PathLike

__all__ = ["InferenceEngine"]

logger = logging.getLogger("autovision.inference.engine")

_BACKENDS = ("torch", "onnx", "tensorrt", "openvino")


class InferenceEngine:
    """Run a task network with a swappable forward backend.

    ``model`` is an exported ``.pt`` payload or a loaded :class:`TaskNetwork`;
    the network provides pre/post-processing for every backend. ``artifact``
    is the compiled graph for the ``onnx``/``tensorrt``/``openvino``
    backends (``.onnx`` file / ``.engine`` file / ``.xml`` IR). The torch
    backend mirrors the export-path reference semantics (``eval`` +
    ``deploy`` hooks), so its outputs sit inside the same numerical band as
    the exported graph.

    ``require_metadata=True`` (INFER-04, review P3-6) refuses artifacts
    that carry no AutoVision identity metadata — for the torch backend the
    whole metadata block is skipped and the flag is a no-op. The CLI
    predict and serving entry points pass ``True``; library callers keep
    the lenient default (warning only).
    """

    def __init__(
        self,
        model: PathLike | TaskNetwork,
        *,
        backend: str = "torch",
        artifact: PathLike | None = None,
        device: str | torch.device | None = None,
        providers: Sequence[str] | None = None,
        openvino_device: str = "CPU",
        require_metadata: bool = False,
    ) -> None:
        if backend not in _BACKENDS:
            raise AutoVisionError(
                f"Unknown inference backend '{backend}'; expected one of "
                f"{list(_BACKENDS)}."
            )
        if backend != "torch" and artifact is None:
            raise AutoVisionError(
                f"backend='{backend}' needs the compiled artifact: pass "
                "artifact=<model.onnx|model.engine|model.xml>."
            )
        if isinstance(model, (str, Path)):
            self.network: TaskNetwork = load_task_model(
                model, device=str(device) if device is not None else "cpu"
            )
        else:
            self.network = model
        self.network.eval()
        self.backend = backend
        self.artifact = Path(artifact) if artifact is not None else None
        self.openvino_device = openvino_device if backend == "openvino" else None

        if backend != "torch" and self.artifact is not None and (
            not self.artifact.exists()
        ):
            # Examples-review P2: the identity metadata gate below used to
            # warn FIRST for a non-existent artifact ("carries no metadata
            # — construction proceeds"), followed only then by the real
            # "artifact does not exist" error from the forward factory —
            # the scary wrong message before the useful one. Existence is
            # checked before anything that reads the file.
            raise AutoVisionError(
                f"The {backend} artifact does not exist: '{self.artifact}'."
            )

        if backend != "torch":
            # Identity gate (design D1): the artifact must have been
            # exported from THIS network. Legacy artifacts without
            # metadata pass through unchanged — unless the caller runs a
            # production entry point (INFER-04): ``require_metadata=True``
            # refuses metadata-stripped artifacts outright, because the
            # identity gate is otherwise vacuous for them.
            from autovision.export.artifact_metadata import (
                read_metadata,
                verify_engine_artifact,
            )

            assert self.artifact is not None  # the guard above rejects None
            verify_engine_artifact(self.network, self.artifact, backend=backend)
            if read_metadata(self.artifact, backend=backend) == {}:
                message = (
                    f"The {backend} artifact '{self.artifact}' carries no "
                    "AutoVision identity metadata — re-export it from this "
                    "network (export verify=True writes the metadata)."
                )
                if require_metadata:
                    raise AutoVisionError(message)
                logger.warning(
                    "%s Construction proceeds without the identity gate "
                    "(pass require_metadata=True to refuse such artifacts).",
                    message,
                )

        if backend == "torch":
            if device is not None:
                self.network.to(device=torch.device(device))
            self._check_cpu_eager_blockers()
            if hasattr(self.network, "deploy"):
                self.network.deploy()  # export-path reference semantics
            self._forward = self._torch_forward
        elif backend == "onnx":
            self._forward = self._make_onnx_forward(providers)
        elif backend == "openvino":
            self._forward = self._make_openvino_forward(openvino_device)
        else:
            self._forward = self._make_tensorrt_forward(device)

    # ── forwards ────────────────────────────────────────────────────────
    def _check_cpu_eager_blockers(self) -> None:
        """Readable failure for CPU torch-backend engines over modules whose
        eager forward needs CUDA (duck-typed ``cpu_eager_blocker`` attribute,
        e.g. the GTR GLA fla/Triton kernels) — instead of an opaque Triton
        pointer error at the first forward."""
        if self.network._module_device().type != "cpu":
            return
        for module in self.network.modules():
            blocker = getattr(module, "cpu_eager_blocker", None)
            if blocker:
                raise AutoVisionError(blocker)

    def _torch_forward(self, batch: torch.Tensor) -> Any:
        return self.network.forward_backend(batch.to(self.network._module_device()))

    def _make_onnx_forward(self, providers: Sequence[str] | None):
        import numpy as np
        import onnxruntime as ort

        assert self.artifact is not None  # the constructor guard guarantees it
        if not self.artifact.exists():
            raise AutoVisionError(f"ONNX artifact does not exist: '{self.artifact}'.")
        session = ort.InferenceSession(
            str(self.artifact),
            providers=list(providers) if providers else ["CPUExecutionProvider"],
        )
        input_meta = session.get_inputs()[0]
        input_name = input_meta.name
        input_is_half = input_meta.type == "tensor(float16)"
        if input_is_half and "CUDAExecutionProvider" not in session.get_providers():
            raise AutoVisionError(
                "The ONNX graph takes float16 inputs but the session has no "
                "CUDA execution provider; export an fp32 graph or pass "
                "providers=['CUDAExecutionProvider']."
            )

        def forward(batch: torch.Tensor) -> Any:
            feed = batch.detach().cpu().float()
            if input_is_half:
                feed = feed.half()
            outputs = session.run(None, {input_name: feed.numpy()})
            tensors = []
            for output in outputs:
                tensor = torch.from_numpy(np.ascontiguousarray(output))
                # postprocess runs in fp32 regardless of the graph dtype.
                tensors.append(tensor.float() if tensor.dtype == torch.float16 else tensor)
            return tensors[0] if len(tensors) == 1 else tuple(tensors)

        return forward

    def _make_openvino_forward(self, openvino_device: str):
        from autovision.export.openvino_helpers import OpenVINOEngineRunner

        assert self.artifact is not None  # the constructor guard guarantees it
        if self.artifact.suffix.lower() != ".xml":
            raise AutoVisionError(
                "The OpenVINO backend needs an .xml IR artifact (and its "
                f".bin weights beside it); got '{self.artifact}'."
            )
        if not self.artifact.exists():
            raise AutoVisionError(
                f"OpenVINO IR does not exist: '{self.artifact}'."
            )
        runner = OpenVINOEngineRunner(self.artifact, device=openvino_device)

        def forward(batch: torch.Tensor) -> Any:
            return runner.run(batch)

        return forward

    def _make_tensorrt_forward(self, device: str | torch.device | None):
        from autovision.export.tensorrt_helpers import TensorRTEngineRunner

        assert self.artifact is not None  # the constructor guard guarantees it
        if not self.artifact.exists():
            raise AutoVisionError(
                f"TensorRT artifact does not exist: '{self.artifact}'."
            )
        if device is None:
            if not torch.cuda.is_available():
                raise AutoVisionError(
                    "The tensorrt backend requires a CUDA device."
                )
            device = "cuda"
        runner = TensorRTEngineRunner(self.artifact, device=device)

        def forward(batch: torch.Tensor) -> Any:
            return runner.run(batch)

        return forward

    # ── public API ──────────────────────────────────────────────────────
    @torch.inference_mode()
    def run_batch(
        self, images: Sequence[Any], *, threshold: float = 0.5
    ) -> list[dict[str, Any]]:
        """Run a batch of images; results are in original-frame pixels.

        ``images`` accepts what :meth:`TaskNetwork.preprocess_image`
        accepts: paths, PIL images, or CHW tensors. Pre/post-processing is
        the network's own (same as evaluation); only the forward differs
        per backend.

        Output contract (INFER-05): the backend returns exactly one record
        per input, and every record carries at least one canonical
        ``autovision.keys`` field — a backend graph whose output layout
        disagrees with the torch reference used to surface as silently
        wrong coordinates or an opaque error deep inside postprocess; both
        now fail here with the backend named.
        """
        if len(images) == 0:
            # INFER-09: an empty batch used to die on a bare
            # "stack expects a non-empty TensorList" RuntimeError.
            raise AutoVisionError(
                "run_batch received an empty image list; nothing to infer."
            )
        tensors, metadata = [], []
        for image in images:
            tensor, meta = self.network.preprocess_image(image)
            tensors.append(tensor)
            metadata.append(meta)
        batch = self.network.preprocess_batch(torch.stack(tensors))
        raw = self._forward(batch)
        records = self.network.postprocess(raw, metadata, threshold=threshold)
        if len(records) != len(images):
            raise AutoVisionError(
                f"The {self.backend} backend returned {len(records)} "
                f"record(s) for {len(images)} input image(s) — the graph's "
                "output layout disagrees with the torch reference; "
                "re-export the artifact or check the postprocess contract."
            )
        # INFER-05 (review P3-2): the known-key set comes from keys.py —
        # the same contract the consumers use — so a new canonical key can
        # never be wrongly rejected here. An EMPTY record is equally a
        # layout failure: it means postprocess returned nothing usable.
        known = {
            BBOXES, SCORES, LABELS, MASKS, KEYPOINTS, POLYGONS,
            MASK, DEPTH, SEGMENT_IDS,
        }
        for index, record in enumerate(records):
            if not isinstance(record, dict):
                raise AutoVisionError(
                    f"The {self.backend} backend's postprocess returned "
                    f"{type(record).__name__} at position {index}; expected "
                    "a dict record (INFER-05)."
                )
            if not (set(record) & known):
                raise AutoVisionError(
                    f"The {self.backend} backend's postprocess record at "
                    f"position {index} carries none of the canonical "
                    f"result keys (got {sorted(record)}) — the output "
                    "layout disagrees with the torch reference (INFER-05)."
                )
        return records

    def run(self, image: Any, *, threshold: float = 0.5) -> dict[str, Any]:
        """Single-image convenience (see :meth:`run_batch`)."""
        return self.run_batch([image], threshold=threshold)[0]

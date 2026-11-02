# Copyright (c) 2026 AutoVision contributors. Apache-2.0.
"""GTR integration package (roadmap design doc 31 G2).

MIT re-ports of the Intellindust GTR backbone line + multi-scale
encoder. Importing this package is safe without the ``gtr`` extra; the
fla dependency is required only to *construct* the attention modules
(:func:`gtr_available` reports readiness). ONNX export swaps the eager
GLA kernels for the pure-ONNX lowering in :mod:`gla_onnx` via the
export pipeline's ``prepare_onnx_export`` module-tree hook.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Expose the compiled GLA CUDA extension (csrc/, built in-place via
# ``csrc/build.sh`` or setup.py) so vit_adapter's plain
# ``import gla_torch_ext`` resolves. Must run BEFORE the submodule
# imports below — vit_adapter probes the extension at module load.
_CSRC_DIR = Path(__file__).resolve().parent / "csrc"
if _CSRC_DIR.is_dir() and str(_CSRC_DIR) not in sys.path:
    sys.path.append(str(_CSRC_DIR))

from autovision.tasks.detection.shared.gtr.encoder import (  # noqa: E402
    ConvNormLayer_fuse,
    GTREncoder,
    get_activation,
)
from autovision.tasks.detection.shared.gtr.gla_onnx import (  # noqa: E402
    chunk_gla_pure,
    gla_forward_pure,
)
from autovision.tasks.detection.shared.gtr.spatial_swiglu import (  # noqa: E402
    ViTAdapterSpatialSwiGLU,
)
from autovision.tasks.detection.shared.gtr.vit_adapter import ViTAdapter  # noqa: E402

__all__ = [
    "GTREncoder",
    "ViTAdapter",
    "ViTAdapterSpatialSwiGLU",
    "ConvNormLayer_fuse",
    "chunk_gla_pure",
    "gla_forward_pure",
    "get_activation",
    "gtr_available",
]


def gtr_available() -> bool:
    """True when flash-linear-attention is importable (the gtr extra)."""
    try:
        import fla  # noqa: F401

        return True
    except ImportError:
        return False

# Copyright (c) 2026 AutoVision contributors. Apache-2.0.
"""GTR pose-estimation task model package (roadmap doc 31 §11 S5).

Inference/load slice: :class:`GTRPoseEstimation` loads the official MIT
``gtrpose_*_coco`` checkpoints and predicts person keypoints (COCO
``[x, y, v]`` triplets at the original frame). Training (CDN + keypoint
loss/matcher) is roadmap §11 **S5b**.
"""

from autovision.tasks.pose_estimation.gtr.task_model import (
    GTR_POSE_VARIANTS,
    GTRPoseEstimation,
)

__all__ = ["GTRPoseEstimation", "GTR_POSE_VARIANTS"]

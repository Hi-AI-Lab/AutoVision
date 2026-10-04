# Copyright (c) 2026 Autovision contributors. Apache-2.0.
"""Pinned pretrained-weight manifest (machine-managed).

Regenerate with scripts/update_weight_hashes.py; hand-edits are
overwritten on the next pinning run.
"""

from autovision._models.weight_verification import WeightArtifact

__all__ = ["WEIGHT_MANIFEST"]

WEIGHT_MANIFEST = {
    'facebook/dinov2-base': WeightArtifact(
        repo_id='facebook/dinov2-base',
        license='apache-2.0',
        source='huggingface',
        revision=None,
        sha256={},
    ),
    'facebook/dinov2-giant': WeightArtifact(
        repo_id='facebook/dinov2-giant',
        license='apache-2.0',
        source='huggingface',
        revision=None,
        sha256={},
    ),
    'facebook/dinov2-large': WeightArtifact(
        repo_id='facebook/dinov2-large',
        license='apache-2.0',
        source='huggingface',
        revision=None,
        sha256={},
    ),
    'facebook/dinov2-small': WeightArtifact(
        repo_id='facebook/dinov2-small',
        license='apache-2.0',
        source='huggingface',
        revision=None,
        sha256={},
    ),
    'facebook/dinov2-with-registers-base': WeightArtifact(
        repo_id='facebook/dinov2-with-registers-base',
        license='apache-2.0',
        source='huggingface',
        revision=None,
        sha256={},
    ),
    'facebook/dinov2-with-registers-giant': WeightArtifact(
        repo_id='facebook/dinov2-with-registers-giant',
        license='apache-2.0',
        source='huggingface',
        revision=None,
        sha256={},
    ),
    'facebook/dinov2-with-registers-large': WeightArtifact(
        repo_id='facebook/dinov2-with-registers-large',
        license='apache-2.0',
        source='huggingface',
        revision=None,
        sha256={},
    ),
    'facebook/dinov2-with-registers-small': WeightArtifact(
        repo_id='facebook/dinov2-with-registers-small',
        license='apache-2.0',
        source='huggingface',
        revision=None,
        sha256={},
    ),
    'facebook/dinov3-convnext-base-pretrain-lvd1689m': WeightArtifact(
        repo_id='facebook/dinov3-convnext-base-pretrain-lvd1689m',
        license='dinov3-custom (see repository card)',
        source='huggingface',
        revision=None,
        sha256={},
    ),
    'facebook/dinov3-convnext-large-pretrain-lvd1689m': WeightArtifact(
        repo_id='facebook/dinov3-convnext-large-pretrain-lvd1689m',
        license='dinov3-custom (see repository card)',
        source='huggingface',
        revision=None,
        sha256={},
    ),
    'facebook/dinov3-convnext-small-pretrain-lvd1689m': WeightArtifact(
        repo_id='facebook/dinov3-convnext-small-pretrain-lvd1689m',
        license='dinov3-custom (see repository card)',
        source='huggingface',
        revision=None,
        sha256={},
    ),
    'facebook/dinov3-convnext-tiny-pretrain-lvd1689m': WeightArtifact(
        repo_id='facebook/dinov3-convnext-tiny-pretrain-lvd1689m',
        license='dinov3-custom (see repository card)',
        source='huggingface',
        revision=None,
        sha256={},
    ),
    'facebook/dinov3-vit7b16-pretrain-lvd1689m': WeightArtifact(
        repo_id='facebook/dinov3-vit7b16-pretrain-lvd1689m',
        license='dinov3-custom (see repository card)',
        source='huggingface',
        revision=None,
        sha256={},
    ),
    'facebook/dinov3-vit7b16-pretrain-sat493m': WeightArtifact(
        repo_id='facebook/dinov3-vit7b16-pretrain-sat493m',
        license='dinov3-custom (see repository card)',
        source='huggingface',
        revision=None,
        sha256={},
    ),
    'facebook/dinov3-vitb16-pretrain-lvd1689m': WeightArtifact(
        repo_id='facebook/dinov3-vitb16-pretrain-lvd1689m',
        license='dinov3-custom (see repository card)',
        source='huggingface',
        revision=None,
        sha256={},
    ),
    'facebook/dinov3-vith16plus-pretrain-lvd1689m': WeightArtifact(
        repo_id='facebook/dinov3-vith16plus-pretrain-lvd1689m',
        license='dinov3-custom (see repository card)',
        source='huggingface',
        revision=None,
        sha256={},
    ),
    'facebook/dinov3-vitl16-pretrain-lvd1689m': WeightArtifact(
        repo_id='facebook/dinov3-vitl16-pretrain-lvd1689m',
        license='dinov3-custom (see repository card)',
        source='huggingface',
        revision=None,
        sha256={},
    ),
    'facebook/dinov3-vitl16-pretrain-sat493m': WeightArtifact(
        repo_id='facebook/dinov3-vitl16-pretrain-sat493m',
        license='dinov3-custom (see repository card)',
        source='huggingface',
        revision=None,
        sha256={},
    ),
    'facebook/dinov3-vits16plus-pretrain-lvd1689m': WeightArtifact(
        repo_id='facebook/dinov3-vits16plus-pretrain-lvd1689m',
        license='dinov3-custom (see repository card)',
        source='huggingface',
        revision=None,
        sha256={},
    ),
    'facebook/dinov3-vits16-pretrain-lvd1689m': WeightArtifact(
        repo_id='facebook/dinov3-vits16-pretrain-lvd1689m',
        license='dinov3-custom (see repository card)',
        source='huggingface',
        revision=None,
        sha256={},
    ),
    'stevenbucaille/rf-detr-base': WeightArtifact(
        repo_id='stevenbucaille/rf-detr-base',
        license='apache-2.0',
        source='huggingface',
        revision=None,
        sha256={},
    ),
    'stevenbucaille/rf-detr-large': WeightArtifact(
        repo_id='stevenbucaille/rf-detr-large',
        license='apache-2.0',
        source='huggingface',
        revision=None,
        sha256={},
    ),
    'stevenbucaille/rf-detr-medium': WeightArtifact(
        repo_id='stevenbucaille/rf-detr-medium',
        license='apache-2.0',
        source='huggingface',
        revision=None,
        sha256={},
    ),
    'stevenbucaille/rf-detr-nano': WeightArtifact(
        repo_id='stevenbucaille/rf-detr-nano',
        license='apache-2.0',
        source='huggingface',
        revision=None,
        sha256={},
    ),
    'stevenbucaille/rf-detr-small': WeightArtifact(
        repo_id='stevenbucaille/rf-detr-small',
        license='apache-2.0',
        source='huggingface',
        revision=None,
        sha256={},
    ),
}

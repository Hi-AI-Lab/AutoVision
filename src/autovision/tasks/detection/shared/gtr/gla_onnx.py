# Copyright (c) 2026 AutoVision contributors. Apache-2.0.
"""Pure-ONNX lowering for the GTR Gated Linear Attention (GLA) block.

The eager GLA runs ``fla.ops.gla.chunk_gla`` (Triton, CUDA-only) or the
hand-written CUDA extension — neither has an ONNX representation, which is
why the ``gtr/*`` routes carried an ``export_blocker``. This module
re-implements the same recurrence with plain tensor ops so the deployed
model traces to standard opset-17 ONNX (MatMul / CumSum / Exp / Add) and
runs in onnxruntime, TensorRT and OpenVINO without any plugin.

Recurrence (identical to ``fla.ops.gla.chunk_gla`` called with
``initial_state=None, output_final_state=False``, which is what the GTR
deploy model uses)::

    S_t = exp(g_t) * S_{t-1} + k_t^T v_t        # state, (K, V) per head
    o_t = scale * S_t^T q_t

with a per-key-channel log decay ``g`` (the model feeds
``gk = logsigmoid(proj) / gate_logit_normalizer``). Unrolled over chunks of
``C`` tokens, only decay *differences* inside one chunk are exponentiated:

    o_inter_i = (q_i * exp(gcs_i)) @ S_prev                     # gcs = in-chunk cumsum
    A_ij      = sum_d q_id k_jd exp(gcs_id - gcs_jd),  j <= i   # causal
    S_next    = S_prev * exp(g_last) + (k * exp(g_last - gcs))^T v

so every exponent is <= 0 and nothing can overflow. Two numerical notes
carry hard-won lessons from the upstream deployment work (Intellindust-AI-Lab
GTR ``tensorrt_plugin`` export patches, MIT — see the provenance ledger):

- the intra-chunk decay is kept as an explicit ``(C, C, K)`` tensor, like
  fla's A kernel. GLA's gate is per key channel, so the decay cannot be
  factored into a row and a column term without one factor growing like
  ``exp(C * |g|)``; trained checkpoints produce strongly negative gates and
  that factorization overflows to ``inf * 0 = NaN``.
- the recurrent state is stored in fp32 even for fp16 graphs: it is an
  unnormalized running sum (GLA has no softmax) and can saturate fp16.
"""

from __future__ import annotations

import torch
from torch import Tensor

__all__ = ["chunk_gla_pure", "gla_forward_pure"]

#: Chunk length of the unrolled recurrence. Only affects the operator
#: count, not the math; 64 mirrors fla's default chunk size.
_GLA_CHUNK = 64


def chunk_gla_pure(q: Tensor, k: Tensor, v: Tensor, gk: Tensor, scale: float,
                   chunk_size: int = _GLA_CHUNK) -> Tensor:
    """Chunkwise gated linear attention in plain tensor ops (ONNX-traceable).

    Args:
        q, k, gk: ``(B, T, H, K)``; ``gk`` is the log decay (<= 0) per key channel.
        v: ``(B, T, H, V)``.
        scale: query scale (the GLA modules use ``head_k_dim ** -0.5``, like fla).
        chunk_size: unroll length ``C``.

    Returns:
        ``(B, T, H, V)`` — computed in fp32, cast back to ``v.dtype``.
    """
    batch, seq_len, heads, k_dim = q.shape
    v_dim = v.shape[-1]
    pad = (chunk_size - seq_len % chunk_size) % chunk_size
    if pad:
        # Pad the token axis (decay 0 keeps the state, zero k/v/q contribute nothing).
        q = torch.nn.functional.pad(q, (0, 0, 0, 0, 0, pad, 0, 0))
        k = torch.nn.functional.pad(k, (0, 0, 0, 0, 0, pad, 0, 0))
        v = torch.nn.functional.pad(v, (0, 0, 0, 0, 0, pad, 0, 0))
        gk = torch.nn.functional.pad(gk, (0, 0, 0, 0, 0, pad, 0, 0))
    num_chunks = (seq_len + pad) // chunk_size

    # (B, H, NC, C, *): the recurrence runs per head over the token axis.
    q = q.transpose(1, 2).reshape(batch, heads, num_chunks, chunk_size, k_dim)
    k = k.transpose(1, 2).reshape(batch, heads, num_chunks, chunk_size, k_dim)
    v = v.transpose(1, 2).reshape(batch, heads, num_chunks, chunk_size, v_dim)
    g = gk.transpose(1, 2).reshape(batch, heads, num_chunks, chunk_size, k_dim)

    dtype = torch.float32
    qf, kf, vf, gf = (t.to(dtype) for t in (q, k, v, g))
    gcs = gf.cumsum(-2)                      # in-chunk cumsum, (B, H, NC, C, K)
    g_last = gcs[..., -1:, :]                # chunk-start -> chunk-end decay
    # (i, j) -> -inf for j > i (diagonal kept); exp() turns it into 0.
    causal = torch.triu(
        torch.full((chunk_size, chunk_size), float("-inf"), dtype=dtype, device=q.device),
        diagonal=1,
    )

    state = qf.new_zeros(batch, heads, k_dim, v_dim)
    outs = []
    for chunk in range(num_chunks):
        gc = gcs[:, :, chunk]                # (B, H, C, K)
        gl = g_last[:, :, chunk]             # (B, H, 1, K)
        q_c, k_c, v_c = qf[:, :, chunk], kf[:, :, chunk], vf[:, :, chunk]

        # Inter-chunk: the state is in the chunk-start frame; exp(gc) <= 1.
        o_inter = (q_c * gc.exp() * scale) @ state
        # Intra-chunk: explicit per-key-channel decay, exponent <= 0 after the
        # causal mask (see the module docstring for why this is not factored).
        pair = gc.unsqueeze(3) - gc.unsqueeze(2) + causal.unsqueeze(-1)
        att = ((q_c * scale).unsqueeze(3) * k_c.unsqueeze(2) * pair.exp()).sum(-1)
        o_c = att @ v_c + o_inter

        # State update: decay to the chunk end, exponent <= 0 on both terms.
        k_end = k_c * (gl - gc).exp()
        state = state * gl.transpose(-1, -2).exp() + k_end.transpose(-1, -2) @ v_c
        outs.append(o_c)

    o = torch.cat(outs, dim=2).reshape(batch, heads, seq_len + pad, v_dim)
    o = o[..., :seq_len, :]
    return o.transpose(1, 2).reshape(batch, seq_len, heads, v_dim).to(v.dtype)


def gla_forward_pure(gla: torch.nn.Module, hidden_states: Tensor) -> tuple[Tensor, None, None]:
    """Export-time replacement for :meth:`GatedLinearAttention.forward`.

    Mirrors the deployed eager forward (fused qkv/g/gk_low GEMM -> split ->
    gk_high projection, fp16-safe logsigmoid, RMSNorm-swish output gate) but
    computes the attention with :func:`chunk_gla_pure` instead of the
    Triton/CUDA kernels. Only valid on a ``convert_to_deploy``-ed module.

    Args:
        gla: the deployed :class:`GatedLinearAttention` module.
        hidden_states: ``(B, T, hidden)`` input.

    Returns:
        ``(output, None, None)`` — same tuple shape as the eager forward.
    """
    if not getattr(gla, "fused_qkv_g_gk_low", False):
        raise RuntimeError(
            "gla_forward_pure requires convert_to_deploy() to have fused the "
            "q/k/v/g/gk_low projections (the ONNX export pipeline deploys first)."
        )
    if gla.num_kv_groups != 1:
        raise RuntimeError(
            "gla_forward_pure covers the GTR configs (num_kv_heads == num_heads); "
            f"got num_kv_groups={gla.num_kv_groups}."
        )
    fused = torch.nn.functional.linear(
        hidden_states, gla.qkv_g_gkl_proj.weight, gla.qkv_g_gkl_proj.bias
    )
    q, k, v, g_pre, gkl = fused.split(list(gla._fused_split), dim=-1)
    gk = gla.gk_high_proj(gkl)

    k_dim, v_dim = gla.head_k_dim, gla.head_v_dim
    q = q.view(*q.shape[:-1], -1, k_dim)
    k = k.view(*k.shape[:-1], -1, k_dim)
    gk = gk.view(*gk.shape[:-1], -1, k_dim)
    v = v.view(*v.shape[:-1], -1, v_dim)

    # logsigmoid(x) = min(x, 0) - log1p(exp(-|x|)): stays finite in fp16 for
    # any x, unlike the log(sigmoid(x)) form (sigmoid underflows below x=-17).
    gk = (torch.clamp_max(gk, 0.0) - torch.log1p(torch.exp(-gk.abs()))) / gla.gate_logit_normalizer
    if gla.clamp_min is not None:
        gk = torch.clamp_min(gk, gla.clamp_min)

    o = chunk_gla_pure(q, k, v, gk, float(k_dim**-0.5)).to(hidden_states.dtype)
    if gla.use_output_gate:
        g = g_pre.view(*g_pre.shape[:-1], -1, v_dim)
        if gla.fuse_norm_and_gate:
            norm = gla.g_norm_swish_gate
            y = _rms_norm(o, norm.weight, norm.eps) * torch.nn.functional.silu(g)
        else:
            y = _rms_norm(o, gla.g_norm.weight, gla.g_norm.eps) * gla.gate_fn(g)
        o = y.reshape(*y.shape[:-2], -1)
    else:
        o = _rms_norm(o, gla.g_norm.weight, gla.g_norm.eps).reshape(*o.shape[:-2], -1)
    return gla.o_proj(o), None, None


def _rms_norm(x: Tensor, weight: Tensor | None, eps: float) -> Tensor:
    """RMSNorm matching fla's RMSNorm/FusedRMSNormGated numerics."""
    out = x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + eps)
    return out * weight if weight is not None else out

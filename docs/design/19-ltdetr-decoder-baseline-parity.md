# 19 — LT-DETR decoder baseline-parity: analysis and decision record

Status: **analysis / deferred rewrite** (2026-10-08)
Reference baseline: lightly-train 0.17.0
`ltdetr_object_detection` / `object_detection_components`.

## 1. The four structural deltas

The local `DFINEv2Decoder` (subclass of the shared `RTDETRv2Decoder`) and
the baseline `DFINETransformer` compute the same loss contract (weights,
matcher, FGL/DDF targets, W-bin space, CDN — all verified equal, see the
2026-10-08 audit), but the layer internals differ:

| # | Aspect | Baseline (lightly-train 0.17.0) | Local fork | Effect |
|---|---|---|---|---|
| 1 | Decoder layer fusion | `Gate`-gated residual after cross-attn + FFN, no `norm2`, `norm3` output clamped to ±65504 | plain residual `+dropout→norm2`, no clamp | different parameterization of the same idea; gate bias is an extra learned vector |
| 2 | Deformable attention | no `value_proj`/`output_proj` (value reshaped directly) | both projections present (RT-DETR lineage) | extra parameters + different attention numerics |
| 3 | `layer_scale` / wide layers | supported (`decoder_layer_wide`, post-`eval_idx` width interpolation) | absent | inert at the baseline default `layer_scale=1.0`; local configs cannot express `>1` |
| 4 | `input_proj` when `in_channels == hidden_dim` | `nn.Identity` (no xavier init) | always Conv1x1(+BN) + xavier | local adds a learnable projection the baseline folds away |

Consequences: state dicts are **not interchangeable** with baseline
checkpoints (this is why the COCO weight path consumes a backbone-only
extract), and published baseline/official mAP numbers are not
architecture-matched comparisons.

## 2. Why not rewrite now

1. **Training-semantics parity is already achieved.** Every loss-side
   divergence except the four above was fixed in the 2026-10-08 audit
   (aux L1+GIoU deep supervision, sanitize, `num_boxes_go` DDP
   normalization). The remaining deltas change the *function class*, not
   the *objective*.
2. **The rewrite invalidates every live artifact.** All exported
   checkpoints (pilots, the running 50-epoch run, the Dome twins) carry
   the fork decoder. A baseline-parity decoder is a from-scratch retrain
   of everything, with no resume path.
3. **Comparability gains require retraining anyway.** Since baseline COCO
   head weights cannot load into either architecture (shape mismatch),
   parity only pays off for *future* runs whose value is a matched
   comparison against baseline-published numbers.

## 3. Migration plan (when the decision is taken)

1. Port `Gate` fusion + remove `value_proj`/`output_proj` in
   `rtdetrv2_decoder.py` behind a per-variant flag
   (`decoder_layer="gate" | "residual"`), defaulting to the fork layout
   for checkpoint continuity.
2. Port `layer_scale`/wide layers in the same flag-gated pass (they share
   the decoder-layer surgery).
3. Switch `input_proj` to Identity when channels match, under the same
   flag.
4. Golden-trajectory test: fixed-seed loss sequence on a tiny decoder,
   snapshotted per flag value, so the fork layout stays frozen while the
   baseline layout stabilizes.
5. New recipe tag for parity runs (e.g. model alias suffix), then
   retrain: (a) COCO-reference sanity at 72-epoch recipe, (b) VisDrone
   pilot v5 twin, before any published comparison.

## 4. Related baseline divergences fixed elsewhere (this audit)

- aux-layer L1+GIoU deep supervision (dfine criterion) — fixed.
- degenerate-box sanitize in VFL / FGL quality / DDF weights / matcher
  costs — fixed (`box_ops.sanitize_boxes_cxcywh_normalized`).
- `num_boxes_go` cross-rank normalization — fixed.
- recipe dispatch by model-name prefix (dinov2 V2 args vs LTDETR args),
  `lr_reference_batch` 16/32 — fixed.

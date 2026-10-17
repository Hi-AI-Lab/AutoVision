# 导入第三方 DINOv3 EoMT 权重

`scripts/import_eomt_checkpoint.py` 把第三方训练包产出的 DINOv3 EoMT
checkpoint 转换成 AutoVision 标准导出格式。转换器只做确定性的张量重排，
不依赖也不 import 第三方训练包，符合仓库的 provenance 约束。

## 支持的 checkpoint

源文件需要是下面这种导出布局：

```text
{"model_class_path": "...",
 "model_init_args": {...},
 "train_model": {"model.backbone....": tensor, ...}}
```

支持：

- semantic segmentation：`classes` 类别表，模型类别数 = `len(classes)`；
- panoptic segmentation：`thing_classes` + `stuff_classes`，类别数 =
  `len(things) + len(stuffs)`，`num_thing_classes = len(things)`；
- 已发布的 DINOv3 EoMT 家族：ViT-T/T+、ViT-S、ViT-B、ViT-L（12/24 层，
  3/6/12/16 heads，`mlp_ratio` 4 或 6，512/640/1280 输入，171/133 类）；
- 融合 `attn.qkv`、`bias_mask`、`mask_head.0/2/4`、`upscale.N.norm`
  等源布局。

## 用法

```bash
python scripts/import_eomt_checkpoint.py \
    --checkpoint path/to/source.pt \
    --out path/to/export.pt
```

转换器会：

1. 从 `model_init_args` 和 state dict 形状推导
   `hidden_size / num_hidden_layers / num_attention_heads / mlp_ratio /
   num_blocks / num_upscale_blocks / image_size`；
2. 把 `backbone.blocks.N.*` 映射到 HF `layers.N.*`，融合 qkv 拆成
   q/k/v，`mask_head.0/2/4` 映射到 `fc1/fc2/fc3`，`upscale.N.norm`
   映射到 `layernorm2d`；
3. 应用 `bias_mask`（q/v bias 保留，k bias 必须为零），丢弃未使用的
   `mask_token`；
4. 写入推理用的 `attn_mask_probs` 和保留 key
   `autovision.rope_periods`（加载时转成 `inv_freq`）；
5. 严格回读，并检查关键张量、LayerNorm eps、RoPE 频率和 attention mask；
6. 写出 `<out>.json`：源路径/SHA256、类别表、`layer_norm_eps`、
   `attn_mask_prob` 等元数据。

常用参数：

| 参数 | 默认 | 说明 |
|---|---:|---|
| `--layer-norm-eps` | `1e-5` | 源 backbone 的 LayerNorm eps；DINOv3 EoMT 家族为 1e-5 |
| `--attn-mask-prob` | `0.0` | 推理时关闭 joint-block attention mask；训练时可设 1 |
| `--verify-forward` | 关 | 回读后按原生分辨率跑一次前向 |
| `--no-verify` | 关 | 跳过严格回读（不建议） |
| `--dry-run` | 关 | 只打印推导出的配置，不写文件 |
| `--overwrite` | 关 | 覆盖已存在的输出 |

## 加载转换后的模型

```python
from autovision import load_task_model

model = load_task_model("path/to/export.pt", device="cpu")
result = model.predict("image.jpg")
```

导出的 `model_init_args` 包含 `layer_norm_eps`，`load_task_model` 会用
同一个 eps 重建模型；RoPE periods 通过保留 state key 自动安装。

## 注意

- **输入分辨率**：semantic 模型用 512×512，panoptic 模型用 640×640，
  1280 变体必须用 1280×1280。HF 的 `grid_size` 和 attention mask 依赖
  这个尺寸。
- **类别顺序**：sidecar JSON 里的类别表就是模型内部顺序（源 dict 的插入
  顺序）；semantic 输出标签是连续 0..N-1，映射回原始 ID 时使用该表。
  panoptic 的 thing 在前、stuff 在后，与 `num_thing_classes` 对齐。
- **预处理**：任务模型的归一化是 ImageNet mean/std，转换时会校验源
  `image_normalize` 必须一致。非方形图像的实际预处理策略可能与源训练包
  不同；复现原指标时先用方形原生尺寸。
- **训练**：导出里的 `attn_mask_probs=0` 是推理设置；如果继续训练，
  按训练配方把它设回 1 或使用训练侧的调度。
- **重新导出**：RoPE periods 通过保留 state key
  `autovision.rope_periods` 传递；如果从加载后的模型再次导出，请保留该
  key，否则新导出会用 HF 默认的 `rope_theta` 重算 RoPE。
- **许可/分发**：这些是第三方权重，建议放在外部 user-held 目录，不要
  提交进仓库；转换器不会把权重打包进发布物。

# 规格 S-4：ECViT 轴向 RoPE（Apache 源 re-port）

状态：已实现。**与分册 28 原方案的偏差（已记台账 P0-4）**：原方案建议以
timm/transformers 的 RoPE 为底重写；实际发现 DEIMv2（Apache-2.0，
`engine/backbone/vit_tiny.py`，本地镜像）包含与本仓数值完全同源的轴向
RoPE——以它为移植源比 timm 方案多一层保障：**数值逐比特同源，已发布
checkpoint 的 `rope_embed.periods` buffer 直接兼容**。

## 来源与归属

- 上游：DEIMv2 `engine/backbone/vit_tiny.py`（Copyright (c) 2025 The
  DEIMv2 Authors, Apache-2.0；其自述 "Modified from DINOv3"）。
- 义务：保留上游版权声明、声明修改（Apache-2.0 §4b），在文件头与本台账
  登记。

## 接口（不变，checkpoint 兼容）

- 模块名 `RopePositionEmbedding`，`forward(*, H, W) -> (sin, cos)`。
- persistent buffer `periods`，形状 `(D_head//4,)`——**必须保持名字、形状与
  persistent 属性**：已发布 checkpoint 的 periods 值与初始化公式有 ~1e-4
  漂移，必须经 buffer 往返加载。
- `_init_weights()`：`periods = base ** (2·arange(D_head//4) / (D_head//2))`，
  `base=100.0`。

## 行为规格（数值不变）

1. 坐标：`normalize_coords="separate"`——`coords_h = arange(0.5, H)/H`、
   `coords_w = arange(0.5, W)/W`；meshgrid → 展平为 `(HW, 2)` → 线性映射到
   `[-1, +1]`。
2. 角度：`2π · coords[:, :, None] / periods[None, None, :]` → `(HW, 2, D//4)`
   → 轴维展平 → `(HW, D//2)` → 首尾拼接为 `(HW, D)`。
3. 输出 `sin/cos` 各 `(HW, D)`；调用方自行升维 `(1,1,HW,D)`。
4. 应用侧 `rotate_half`/`apply_rope`：标准旋转位置编码半旋转组合，仅作用于
   patch token（register token 保持原始嵌入）。

## 允许的自由度（不触碰数值）

实现结构、变量命名、辅助函数划分、坐标构造的表达方式。

## 验收

- 既有 ecvit 相关单测全过；ECViT 前向输出与重写前逐元素一致（同一
  随机权重、float32，max abs err == 0）。
- `periods` buffer 在 `state_dict()` 中存在、persistent、形状正确。

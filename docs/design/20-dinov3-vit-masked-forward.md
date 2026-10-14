# VTrain 实现蓝图 · 分册 20：DINOv3 ViT masked 前向（token drop 的 ViT-v3 配方）

| 文档属性 | 内容 |
|---|---|
| 版本 | v1.0（已实施，实现记录见 §8） |
| 状态 | ✅ 已落地：8 例新测试 + 既有 30 例 + 相关回归（weight verification 61 / e2e 训练 16）全过；ruff 全绿 |
| 来源设计 | 分册 17 §4.3.1 勘误（"DINOv3 masked 支持的明确配方……留待 ViT-v3 配方立项"）——**本册即该立项** |
| 参考实现 | `src/autovision/_models/dinov2/dinov2_wrapper.py::forward_multiscale_features_masked`（分册 17 v1.1 已验证路径）；上游 `transformers 5.19` `models/dinov3_vit/modeling_dinov3_vit.py` |
| 受影响模块 | `_models/dinov2/dinov2_wrapper.py`（仅提取共享 helper）、`_models/dinov3/dinov3_wrapper.py`（主要改动）、`ltdetr_object_detection/task_model.py`（门控文案）、`tests/test_window_token_drop.py` |
| 基线兼容承诺 | `token_drop=None` 时全部家族既有行为、权重、导出图**逐比特不变**；DINOv2 masked 路径行为逐比特不变（仅提取 helper，无逻辑改动）；验证/导出路径强制稠密 |

> **标注约定**（承分册 17）：【HF】transformers 5.19 实测事实（本册全部经真机实证，见 §2）；【VTrain】本仓库现状（出处 `文件:行`，相对 `src/autovision/`）；【设计】本文档决策；【差异】相对 DINOv2 masked 路径的行为差异（编号 V1–V2 及理由）。

---

## 1. 背景与目标

分册 17 落地了 window 级 in-backbone token dropping，但实施期实证推翻了"DINOv3 继承即得"的预测（§4.3.1 勘误）：DINOv3 ViT 的 HF 内部布局是 `embeddings/model/norm` 且 encoder 需要显式 rope，v1 按风险预案将其结构门控显式拒绝。当前唯一可用的 token-drop 骨干是 DINOv2 系（`ltdetr/dinov2-vits14-noreg` 等），而 `1280` 分辨率对 patch-14 尺寸不整除（须 1260/1288），实际限制了该特性与现役 1280 ViT 实验线（`runs/visdrone_ltdetr_vits16_1280_cocofull`）的结合。

**本册目标**：给 `TransformersDinov3ViTWrapper` 补齐 masked 多尺度前向，使 `dinov3-vits16 / vitt16 / vitt16plus / vitb16 / vitl16` 全系 ViT-v3 变体支持 `token_drop`；保持 DINOv2 路径与稠密路径逐比特不变。

**非目标**：ECViT wrapper（分册 17 §10 P3，继续挂起）；ConvNeXt（无 token 序列，永久不适用）；rope 增广的逐图/共享语义改动（§4 声明为既定差异，不"修复"）。

---

## 2. 事实基础（transformers 5.19，全部真机实证）

### 2.1 DINOv3 ViT 内部布局与前向编排

```
DINOv3ViTModel
├── embeddings        # DINOv3ViTEmbeddings(pixel_values) → (B, 1+R+N, D)
├── rope_embeddings   # DINOv3ViTRopePositionEmbedding(pixel_values) → (cos, sin)
├── model             # DINOv3ViTEncoder（子模块仅 layer）
│     forward(hidden_states, position_embeddings, output_hidden_states)
│     → hidden_states = [embeddings输出, block_0, ..., block_{N-1}]   # 与 DINOv2 tap 语义一致
└── norm              # final layernorm（只作用于 last_hidden_state，不进 hidden_states）

DINOv3ViTModel.forward(pixel_values):
    hidden = self.embeddings(pixel_values)
    position_embeddings = self.rope_embeddings(pixel_values)
    encoder_outputs = self.model(hidden, position_embeddings=position_embeddings, ...)
    return self.norm(encoder_outputs.last_hidden_state)
```

实证：`m.norm(inner.last_hidden_state) == full.last_hidden_state` 逐元素成立；`hidden_states` 长度 N+1、首项为 embeddings 输出——**tap 语义与 DINOv2 完全一致**（tap `i+1` = 第 `i` 个 block 之后、无终局 layernorm）。

### 2.2 rope 表的三个关键性质（masked 路径的正确性依据）

1. **动态网格**：`rope_embeddings.forward(pixel_values)` 从**输入像素 H/W** 现算 patch 网格坐标（`num_patches_h = H // patch_size`），不依赖 `config.image_size`——1280 输入自然得到 80×80 表，无需插值。
2. **逐位置恒等**：坐标按 `(row, col)` 网格生成，`cos/sin` 形状 `(N, head_dim)`，第 `j` 行对应 patch 网格的第 `j` 个位置（行优先展平，与 patch token 顺序一致）。**对 kept 索引做 `index_select` 即保留每个保留 token 的原始位置编码**。
3. **前缀反推**：`apply_rotary_pos_emb`（`modeling_dinov3_vit.py:249`）用 `num_prefix = q.shape[-2] - sin.shape[-2]` **反推** CLS+registers 前缀数，再对 patch 段逐位置施加 `(q·cos + rotate_half(q)·sin)`。

**推论（本册正确性核心）**：gather 掉 token 后若沿用完整 rope 表，则 (a) 前缀反推得到负数（`3+kept-16 < 0` when `kept < 16`），split 直接崩溃或错位；(b) 保留 token 按收缩后下标取 rope，位置恒等被破坏。因此 **hidden 序列与 rope 表必须用同一份 kept 索引同步 gather**——两者同时收缩后，前缀反推恢复正确（`(3+kept) - kept = 3`），且每个保留 token 拿到自己的原始位置 rope。

### 2.3 rope 增广（【差异 V1】的根源）

`DINOv3ViTRopePositionEmbedding.forward` 在 `self.training` 时对坐标施加 `augment_patches_center_coordinates(shift, jitter, rescale)`；**默认配置 `pos_embed_rescale=2.0` 非空**（log-uniform `[1/2, 2]` 缩放，每次调用独立采样）——即稠密路径训练期本就带 rope 增广。含义见 §4。

### 2.4 门控现状（【VTrain】）

- `TransformersDinov3ViTWrapper(TransformersDinov2Wrapper)`（`_models/dinov3/dinov3_wrapper.py:27`）——是基类子类，继承的 `supports_masked_forward` 检查 `hasattr(model, 'embeddings') and hasattr(model, 'encoder')`；DINOv3 无 `.encoder`（是 `.model`）→ 返回 False → 任务模型 init 期报错（`task_model.py:316-325`），`test_dinov3_vit_layout_rejected` 钉住。
- `TransformersDinov3ConvNextWrapper` **不是** `TransformersDinov2Wrapper` 子类 → 家族/布局双重门控下永远不可达 masked 路径，无需改动。
- 路由零缺口：`ViTDetectionBackbone.forward(window_keep_masks=)` 直接多态调用 `wrapper.forward_multiscale_features_masked`（`backbone_wrappers.py:209-221`），下游 resize/STA/投影无感知。

---

## 3. 设计

### 3.1 DINOv3 masked 编排配方（每图，与 DINOv2 版逐段对应）

```python
image = x[i:i+1]
kept  = _kept_patch_indices(mask, H//ps, W//ps) if mask is not None else None   # 共享 helper，网格语义单点
hidden = self._model.embeddings(pixel_values=image)                             # (1, 1+R+N, D)
cos, sin = self._model.rope_embeddings(pixel_values=image)                      # (N, hd) × 2
if kept is not None:
    prefix, patches = hidden[:, :num_prefix], hidden[:, num_prefix:]
    hidden = torch.cat((prefix, patches.index_select(1, kept)), dim=1)
    cos, sin = cos.index_select(0, kept), sin.index_select(0, kept)             # ← §2.2 推论
encoder_outputs = self._model.model(hidden, position_embeddings=(cos, sin),
                                    output_hidden_states=True)
# 每 tap：hidden_states[layer+1] → 零回填 → _reshape_patch_tokens（与 DINOv2 逐行相同）
```

- `num_prefix = 1 + num_register_tokens`：与基类同公式；vits16 的 register 数由 config 携带（官方 DINOv3 ViT 带 registers），`_reshape_patch_tokens` 无需改动。
- **keep-all 仍走 gather/refill**（基类同款裁定）：全索引 `index_select` 逐比特复现稠密值，不变量在测试中保持诚实。
- 零回填逻辑（`new_zeros` + `index_copy_` + 前缀 cat，梯度经 `index_copy_` 流向保留 token）从基类提取为模块级 helper `_backfill_patch_tokens(hidden, kept, num_prefix, num_h, num_w)`，**两个 wrapper 共用单点实现**；DINOv2 `forward_multiscale_features_masked` 改为调用该 helper（行为逐比特不变，测试守护）。

### 3.2 `supports_masked_forward` 的 DINOv3 判据

```python
# TransformersDinov3ViTWrapper 内 override：
def supports_masked_forward(self) -> bool:
    m = self._model
    return all(hasattr(m, a) for a in ("embeddings", "model", "rope_embeddings")) \
        and hasattr(m.model, "layer")
```

- 三属性 + encoder 层容器齐备才声明支持——版本漂移时退化为配置期报错（分册 17 §9 同款缓解）。
- 基类判据（`embeddings`+`encoder`）不动；`TransformersDinov3ConvNextWrapper` 与 ECViT wrapper 不在继承树内，天然排除。
- 任务模型门控 `task_model.py:316-325` 的**逻辑零改动**（isinstance 基类 + polymorphic supports 检查自动放行 DINOv3 ViT），仅更新错误文案（不再宣称 DINOv3 被拒，改为"该 wrapper 未暴露 masked 前向所需的内部编排面"）。

### 3.3 代码组织

| 文件 | 改动 |
|---|---|
| `dinov2_wrapper.py` | 新增模块级 `_backfill_patch_tokens`；`forward_multiscale_features_masked` 的 tap 回填段改调 helper（逻辑不变）；基类 `supports_masked_forward` 与其 masked 入口报错文案改为中性表述 |
| `dinov3_wrapper.py` | `TransformersDinov3ViTWrapper` override `supports_masked_forward`（§3.2）与 `forward_multiscale_features_masked`（§3.1，复用 `_kept_patch_indices` + `_backfill_patch_tokens` + 基类 `_reshape_patch_tokens`） |
| `task_model.py` | 门控错误文案更新（无逻辑变更） |
| `tests/test_window_token_drop.py` | 翻转 dinov3 拒绝测试 → 放行断言；新增 DINOv3 wrapper 不变量组（§5） |

### 3.4 与分册 17 的关系

- 分册 17 §4.3.1 勘误的"拒绝"裁决由本册**有界取代**：仅 ViT-v3 wrapper 层；分册 17 的其余契约（off-by-default、家族门控、val/导出稠密、mask 构建语义）原样继承，全部既有 30 例测试继续有效。
- 分册 17 §11.2 表中"`--token-drop` 训练脚本开关未加（待 ViT-variant 配方立项）"——本册 §7 P1 一并补上 vits16 脚本开关。

---

## 4. 行为差异声明

### 【差异 V1】rope 增广的采样粒度（仅训练期、仅 masked 路径激活时）

| 场景 | 稠密路径 | masked 路径 |
|---|---|---|
| eval / 导出 | rope 增广关闭 | 同左——**逐比特不变量在 eval 下成立** |
| 训练，batch 前向 | 每批**一次** rescale 采样，批内共享 | 逐图循环 → **每图独立**采样 |

- 不修它：参考实现（windowed fork）在窗口维度同样是逐窗独立语义；检测训练中 rope 增广本就是逐样本意义的随机化，逐图采样是合法（甚至更充分）的增广实现，与 dropout 同类的 RNG 流差异。
- 后果与守护：(a) 训练期"keep-all masked batch ≡ dense batch"**不再逐比特**（ rope 采样不同）——所有逐比特不变量测试固定在 **eval 模式**（与现有 dinov2 套件同做法，其无 rope 不受影响，但测试风格统一）；(b) `token_drop=None` 基线兼容承诺不受影响（mask 缺省根本不进 masked 路径）。
- 若未来需要训练期 batch 一致语义：可在 wrapper 层对齐为"每步一次采样、批内广播"——记为可选项，不在本期实施。

### 【差异 V2】增广配置的版本依赖

`pos_embed_shift/jitter/rescale` 默认值（尤其 `rescale=2.0`）是 transformers 侧事实而非本仓库契约；若上游改变默认，差异 V1 的"激活范围"随之变化，但 eval 路径的逐比特不变量与门控行为不受影响。§5 的不变量测试在 eval 下运行，天然免疫。

---

## 5. 测试与验收（镜像分册 17 §8 的 wrapper 不变量，DINOv3 版）

现有 `tests/test_window_token_drop.py` 30 例全部保持通过；新增/修改：

| # | 测试 | 判据 |
|---|---|---|
| T1 | 翻转 `test_dinov3_vit_layout_rejected` → `test_dinov3_vit_masked_forward_accepted`：`dinov3-vits16-notpretrained` + `token_drop` 构造**成功** | init 不再报错 |
| T2 | keep-all ≡ dense 逐比特（全部 tap，eval） | 逐比特相等 |
| T3 | partial keep：被删 patch 区域每 tap 精确为 0，保留区有限非零 | 同分册 17 §8.2 |
| T4 | 不同 mask 的 batch ≡ 逐图 masked 前向拼接（逐比特） | 同分册 17 §8.3 |
| T5 | mask 全 `None` ≡ dense（masked 入口，eval） | 逐比特相等 |
| T6 | 网格不整除 / mask 数量不符 → 硬报错 | 复用既有匹配文案 |
| T7 | 梯度流经 `index_copy_` 到达保留 token（train 模式，patch embedding 权重 grad 非零且有限） | 同分册 17 §11.2 |
| T8 | `supports_masked_forward` 三态：DINOv2 True / DINOv3 ViT True / ConvNeXt wrapper False（且其 masked 入口报错） | 布尔断言 |
| T9 | 门控文案：构造期对不支持者仍报配置错（用 ConvNeXt wrapper 路径已由 T8/既有测试覆盖，无需新例） | — |

回归面：`tests/test_window_token_drop.py` 全量 + 分册 17 §11.3 的相关回归（ltdetr_object_detection 等）+ `ruff check`。

---

## 6. 风险表

| 风险 | 等级 | 缓解 |
|---|---|---|
| HF 内部属性面再漂移（rope 签名/encoder 容器名） | 中 | §3.2 属性判据 + 配置期报错；判据只认"编排所需最小面"，不锁具体类名 |
| rope gather 位置错位（§2.2 推论的实现错误） | 中 | T2（keep-all 逐比特）+ T4（batch 拼接）双不变量覆盖：任何位置错位都会破坏其一 |
| 共享 helper 重构动到 DINOv2 已验证路径 | 低 | 重构为纯提取（无逻辑变更）；既有 30 例全量回归守护逐比特行为 |
| rope 增广训练期语义争议 | 低 | §4 显式声明 + 测试固定 eval；训练期差异作为既定增广语义记录 |
| `index_select` 带来的额外显存/耗时（rope 表小：N×hd） | 低 | O(N) 张量复制，相对 encoder 注意力可忽略；逐图循环开销已被分册 17 §9 实测否决 |

---

## 7. 实现清单与路线

| 文件 | 改动 | ~行 |
|---|---|---|
| `_models/dinov2/dinov2_wrapper.py` | `_backfill_patch_tokens` 提取 + 两处文案中性化 | ~30 |
| `_models/dinov3/dinov3_wrapper.py` | `supports_masked_forward` + `forward_multiscale_features_masked` | ~85 |
| `ltdetr_object_detection/task_model.py` | 门控文案 | ~6 |
| `tests/test_window_token_drop.py` | T1 翻转 + T2–T8 新增 | ~120 |
| `scripts/visdrone_ltdetr_1280_vits16_baseline_coco.py` | `--token-drop` 开关（`MODEL_ARGS["token_drop"] = {...}`，分册 17 §4.4 同款）| ~15 |

- P0 本册实施 + §5 测试（本次）。
- P1 vits16 VisDrone 脚本开关（本次顺手落地——当前 `runs/visdrone_ltdetr_vits16_1280_cocofull` 基线正在跑，消融 A/B 的现成对照）。
- P2 消融矩阵：± token_drop（vits16 @1280，对齐现役基线）；`token_keep_ratio` 诊断已内建（分册 17 §4.4）。
- P3 G2 骨干占比 profile（`scripts/profile_step_phases.py`，GPU 环境）——分册 17 遗留，适用面随本册扩展到 ViT-v3。

---

## 8. 实施记录（2026-10-10，v1.0 落地）

### 8.1 与 §7 清单的对照

| 文件 | 实际改动 | 对 §3/§7 |
|---|---|---|
| `_models/dinov2/dinov2_wrapper.py` | 模块级 `_backfill_patch_tokens` 提取（masked 前向 tap 回填段改调 helper，纯提取）+ `supports_masked_forward` 与 masked 入口报错文案中性化 | 如计划 |
| `_models/dinov3/dinov3_wrapper.py` | `supports_masked_forward`（embeddings + model + rope_embeddings + model.layer 四点判据）+ `forward_multiscale_features_masked` 全量 override（§3.1 配方；rope 与 token 同 kept 索引 gather） | 如计划 |
| `ltdetr_object_detection/task_model.py` | 门控注释与报错文案更新（isinstance + polymorphic supports 逻辑零改动，DINOv3 ViT 自动放行） | 如计划 |
| `tests/test_window_token_drop.py` | `test_dinov3_vit_layout_rejected` 翻转为 `test_dinov3_vit_masked_forward_accepted`（T1）+ `TestWrapperMaskedForwardDinov3` 七例（T2–T7）+ 三态 layout 测试（T8） | 如计划，共 8 例 |
| `scripts/visdrone_ltdetr_1280_vits16_baseline_coco.py` | `--token-drop` 开关（window 32 / keep_ratio 0.4 / step_stop=AUG_STOP；默认关，baseline 契约不变） | 如计划（P1） |

### 8.2 验证

- `tests/test_window_token_drop.py` 38/38（既有 30 + 新增 8）。
- `tests/test_weight_verification.py` 61/61；`tests/test_train_object_detection.py` + `tests/test_train.py`（e2e 训练）16/16。
- `ruff check` 全部触碰文件绿。

### 8.3 实施期发现与处置

1. **冒烟期抓到 rope 对齐断言写错列范围**（测试脚本自身错误，非实现）——零回填区域断言应为被删窗口对应的 patch 行列；正式测试沿用分册 17 的正确区间，且 T2（keep-all ≡ dense 逐比特）对任何 rope gather 错位都是完备守卫（§6 风险表的缓解得到实证）。
2. **`test_weight_verification.py::test_covers_dinov3_registry` 既有失败**（stash 隔离证实先于本册存在）：工作树中未提交的 vitt16/vitt16plus 注册（`hf=None`，无公开 checkpoint）未同步 manifest 测试。处置：测试对 `hf=None` 条目跳过 manifest 查询（与注册处文档字符串的语义一致），回归信号恢复干净；该跳过逻辑归属既有 vitt16 工作，本册仅恢复绿灯。
3. **train 模式 rope 增广实证**：`test_gradients_flow_to_kept_tokens` 在 `wrapper.train()` 下运行，即默认 `pos_embed_rescale=2.0` 增广激活的路径——masked 前向在增广开启时工作正常，差异 V1 的声明与实现一致。

### 8.4 后续（更新 §7 路线）

- P2 消融（± token_drop @ vits16 1280）就绪：`--token-drop` 开关已落地，对照基线 `runs/visdrone_ltdetr_vits16_1280_cocofull` 正在产出。
- P3 G2 profile 不变。

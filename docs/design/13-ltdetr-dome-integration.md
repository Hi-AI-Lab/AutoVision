# VTrain 实现蓝图 · 分册 13：LT-DETR × Dome-DETR 密度导向检测集成设计

| 文档属性 | 内容 |
|---|---|
| 版本 | v1.1（已实施；实施记录见 §5 D11 与 §6 验收结果） |
| 状态 | 已实现并通过 `tests/test_ltdetr_dome.py`（43 例）与既有检测回归 |
| 上游论文 | Dome-DETR: DETR with Density-Oriented Feature-Query Manipulation for Efficient Tiny Object Detection（ACM MM 2025, arXiv:2505.05741） |
| 参考实现 | `/home/hdjs/Research/Software/Dome-DETR-master`（官方代码, 基于 D-FINE） |
| 受影响模块 | `_task_models/ltdetr_object_detection/`、`_task_models/object_detection_components/`、`tests/`、`scripts/` |
| 基线兼容承诺 | 关闭新开关时，LT-DETR / RT-DETR 全部既有行为、权重、导出图**逐比特不变** |

> **标注约定**：【Dome】Dome-DETR 参考实现中的已验证事实（出处 `文件:行`，路径相对 `/home/hdjs/Research/Software/Dome-DETR-master/`）；【VTrain】本仓库现状（出处 `文件:行`，路径相对 `src/autovision/`）；【设计】本文档决策；【差异】有意偏离 Dome 原实现的决策（附编号 D1–D10 及理由）。

---

## 1. 背景与目标

### 1.1 Dome-DETR 方法摘要

Dome-DETR 针对密集微小目标场景（VisDrone、AI-TOD）中 DETR 类模型的两大浪费——**特征冗余**（背景区域照常参与注意力）与**查询僵化**（固定 top-k 查询数与图像目标密度无关）——提出三个组件，官方成绩（VisDrone2019 val, 800px）：S/M/L = 35.9 / 38.9 / 39.0 AP（13.2M / 23.9M / 36.0M 参数）。

| 组件 | 全称 | 作用位置 | 一句话机制 |
|---|---|---|---|
| **DeFE** | Density-Focal Extractor | encoder 入口（最浅层特征） | 轻量卷积头预测逐像素**密度图** + 逐图**目标数量比值** `reg_value`；用逐框高斯热图与数量比值做辅助监督 |
| **MWAS** | Masked Window Attention Sparsification | encoder stride-16 层 | 密度门控选出"含目标"的窗口子集，窗口注意力（轴置换序列注意力）只在窗口内计算，特征残差写回 |
| **PAQI** | Progressive Adaptive Query Initialization | decoder 查询选择 | top-`max_num_select` 候选中，前 `min_num_select` 直通，其余按密度窗口门控丢弃，再做**密度调制 IoU 阈值的逐类 NMS**——最终查询数逐图可变（250~500） |

辅助损失（【Dome】`src/zoo/dome/dome_criterion.py:491-528`）：

- **密度图损失**：加权 MSE，低估高密度区域额外惩罚
  `L_den = w_d · mean[(d−g)² · (1 + λ·g·1[d<g])]`，VisDrone 配方 `w_d=1, λ=0.3`；
- **数量回归损失**：不对称 MSE，低估罚 2 倍
  `L_cnt = mean[s·(r̂−r)²]`，`s = 2 if r̂<r else 1`，`r = (clamp(n_gt, min, max) − min)/(max − min)`。

### 1.2 集成范围

**做**：将 DeFE / MWAS / PAQI 三个组件移植进 LT-DETR 家族，作为**可选密度分支**（一个配置开关），首战场景 = VisDrone + `ltdetr/dinov3-convnext-tiny` + D-FINE decoder @800px（对齐现有 `scripts/visdrone_ltdetr_800.py` 基线）。

**不做（非目标）**：

- 不改动 RT-DETR / RT-DETRv2 / OBB 家族的任何行为；
- 不引入 Dome 的 HGNetv2 backbone、4 层 stride-4 金字塔、`pre_bbox_head`、encoder aux 损失（【VTrain】decoder 本就没有这些，见 §2.1）；
- 不做 epoch 制 MultiStepLR / Mosaic-MixUp 停增广逻辑（沿用引擎 flat-cosine + 停增广窗口契约）；
- ONNX 导出不支持动态查询（走静态 deploy 模式，见 §4.8）。

### 1.3 预期收益与判据

- 主指标：VisDrone val mAP（`eval_max_dets=500` 已对齐 Dome 评测协议，【VTrain】`config.py:425`）。
- 辅助指标（新）：**查询充裕率**（查询数 ≥ GT 数的图占比）与**平均查询数**（【Dome】`src/solver/det_engine.py:238-254` 的 defe Ample Rate / Average Anchor Number）。
- 成功判据：在 800px ConvNeXt-tiny 基线上，开启密度分支后 mAP ≥ +1.0，且平均查询数显著低于固定 500 上限（省 decoder 计算）；若 mAP 持平但查询数下降 ≥40%，也视为可接受（效率收益）。

---

## 2. 两套代码的现状对照

### 2.1 架构差异表

| 维度 | Dome-DETR【Dome】 | VTrain LT-DETR【VTrain】 | 移植含义 |
|---|---|---|---|
| 金字塔 | 4 层，strides [4,8,16,32]（`configs/dome/Dome-S-VisDrone.yml:33-34`） | 3 层，strides [8,16,32]（`task_model.py:193`） | DeFE 输入层从 stride-4 改为 **stride-8**（D1） |
| encoder 输出契约 | `dict{feats, defe, img_inputs}`（`zoo/dome/hybrid_encoder.py:501-502,658`） | `list[feat]`（`object_detection_components/hybrid_encoder.py:287-327`） | dome encoder 返回 `(outs, defe_info)` 元组，任务模型解包（§4.6） |
| 查询选择 | 密度门控 + 动态 NMS，查询数 250~500 可变（`zoo/dome/dome_decoder.py:767-909`） | 固定 top-`num_queries`(300)（`rtdetrv2_decoder.py:741-776`） | 新增 dome decoder 子类（§4.3） |
| CDN 去噪 | attn mask 已扩展 per-batch padding（`zoo/dome/denoising.py:110-150`） | 2D 共享 mask（`object_detection_components/denoising.py:149-161`） | 扩展为可选 3D mask（§4.4） |
| decoder 精修 | FDR 角点分布 + pre_bbox_head + LQE | FDR 角点分布（`dfine_decoder.py:46-177`，无 pre_bbox_head/LQE） | 只移植查询选择，精修循环不动 |
| 损失掩码 | focal/vfl/mal 乘 `batch_queries_num` 掩码（`dome_criterion.py:91-94` 等） | 无（查询全有效） | `loss_labels_vfl` 加可选 `query_mask`（§4.5） |
| 归一化 targets | 训练时已归一化 cxcywh（collate 内） | 同（归一化 cxcywh，`training_step:115-118`） | GT 热图生成可直接用 `targets["boxes"]`，比 Dome 的 eval 分支更简单 |
| 精度策略 | AMP fp16 + 损失处 fp16 autocast（`dome_criterion.py:510,521`） | bf16 autocast 包住整个 `training_step`（`engine.py:662-680`） | 密度损失强制 fp32 island（D6） |
| 训练制 | epoch 制，MultiStepLR，Mosaic/MixUp | 步数制 flat-cosine + 停增广窗口 | 不移植 Dome 调度（D9） |
| 导出 | 独立 onnx-export 分支 | `torch.export` 严格模式 + `deploy()` 钩子（`_export/export_onnx.py:93-95,106-112`） | 动态查询/NMS 循环不可导出 → `deploy()` 静态模式（D8） |
| DDP | 支持（2026/1 修复多 batch） | DDP + 梯度累积（`engine.py:493-508`） | 每 rank 独立 pad 到本 rank 批内最大值，天然兼容 |

### 2.2 移植主线结论

1. **DeFE/MWAS 是"encoder 内插件"**：只依赖最浅层/中层投影特征与 targets 的归一化框，与 backbone 家族（ViT/ConvNeXt/ECViT）解耦，可安全做成 `HybridEncoder` 子类。
2. **PAQI 是"decoder 查询选择替换"**：需要把"固定 Q"的全链路假设（CDN mask、损失、评测、导出）一并打通，是改动面最大的一块。
3. **移植顺序按依赖链**：DeFE（产出密度图）→ PAQI（消费密度图）→ MWAS（消费密度图、改特征）。MWAS 独立于 1/2，放最后以隔离风险。

---

## 3. 总体设计

### 3.1 模块拓扑

**新增文件**（全部落在共享组件层 + ltdetr 任务层，风格对齐现有"faithful port + 设计文档注释"惯例）：

```
src/autovision/_task_models/object_detection_components/
├── defe.py            # DeFE 密度头 + GT 热图生成器 + 两个密度损失（纯函数）
├── mwas.py            # 轴置换窗口注意力 WindowProcessor + 密度→窗口掩码
└── dynamic_nms.py     # 密度调制逐类向量化 NMS

src/autovision/_task_models/ltdetr_object_detection/
├── dome_encoder.py    # DomeHybridEncoder(HybridEncoder)：DeFE+MWAS 接线
└── dome_decoder.py    # DomeDFINEv2Decoder(DFINEv2Decoder)：动态查询选择

tests/test_ltdetr_dome.py   # 全部单元/性质测试（§7）
scripts/visdrone_ltdetr_dome_800.py  # VisDrone 对比实验脚本（§8）
```

**修改文件**（全部向后兼容，新增参数默认关闭）：

| 文件 | 改动 |
|---|---|
| `ltdetr_object_detection/config.py` | 新增 `LTDETRDensityArgs`；`LTDETRObjectDetectionTrainArgs` 增加 `density: LTDETRDensityArgs \| None = None` |
| `ltdetr_object_detection/task_model.py` | `__init__` 按 `density` 分支构建 dome encoder/decoder；`_forward_train` 传 targets/defe；`forward_backend`/`deploy()` 适配 |
| `ltdetr_object_detection/train_model.py` | `training_step`/`validation_step` 透传 `query_mask` 与 `defe`；日志新增 3 项 |
| `object_detection_components/denoising.py` | `get_contrastive_denoising_training_group` 增加可选 `batch_queries_num`/`num_heads`（None = 原行为） |
| `object_detection_components/rtdetrv2_criterion.py` | `loss_labels_vfl` 增加可选 `query_mask`；`RTDETRCriterionV2.forward` 透传 |
| `object_detection_components/dfine_criterion.py` | `forward` 增加 `query_mask`/`defe_info`，追加两个密度损失 |
| `ltdetr_object_detection/__init__.py` | 导出 Dome 类（可选） |

### 3.2 数据流（三态）

**训练态**（`training_step` → `_forward_train`）：

```
x(B,3,H,W) ──backbone──► feats[S8, S16, S32]
   │
   ▼ DomeHybridEncoder.forward(feats, targets)
   proj = input_proj(feats)                          # 3 × hidden_dim
   density, reg_value = LiteDeFE(proj[0])            # S8 → pool/2 → 主干 → ×8 → 密度图 (B,1,H/2,W/2)
   defc_pooled = adaptive_max_pool2d(density, (⌈H16/w⌉, ⌈W16/w⌉))   # 窗口粒度 w=mwas_window_size
   win_mask    = adaptive_threshold(defc_pooled)     # (B, ⌈H16/w⌉, ⌈W16/w⌉) bool，保证每图≥1 窗口
   proj[1]    += MWAS(proj[1], win_mask)             # 仅窗口内窗口注意力，残差写回
   outs        = AIFI(proj[2]) → FPN → PAN           # 基线路径不变
   gt_heat     = GaussHeatmap(targets.boxes, (H/2, W/2))   # 仅训练
   ► 返回 (outs, defe_info)
   │
   ▼ DomeDFINEv2Decoder.forward(outs, targets, defe_info)
   memory, shapes = _get_encoder_input(outs)          # 基线不变
   候选 = enc_head 打分全网格 → top-max_num_select
        → 前 min_num_select 直通
        → 其余按 win_mask 门控 → 逐类动态 NMS(IoU 阈值 = base + scale·密度)
        → 逐图数量 n_b ∈ [~min, max]，batch 内 pad 到 max(n_b)
   padding 查询：content 置 0、logits 置 −1e4（D5）
   CDN 组（3D attn mask：去噪隔离 + padding 双向屏蔽）→ 拼接 → FDR 精修循环（基线不动）
   ► out{pred_logits, pred_boxes, aux_outputs, dn_*, corners/refs,
         batch_queries_num, query_valid_mask, defe_info}
   │
   ▼ DFINECriterionV2(..., query_mask, defe_info)
   既有损失（VFL×mask + L1 + GIoU + FGL + DDF + DN）＋ L_den + L_cnt（fp32）
```

**推理态**（`forward(x)`）：同上但无 CDN、无 GT 热图；padding 位置 logits 已是 −1e4，`decode_detection_outputs` 的 topk+threshold 天然过滤（`decode.py:38-47`），`batch_queries_num` 仅用于日志。

**导出态**（`deploy()` 后）：encoder 跳过 MWAS（DeFE 头可留作可剪枝旁路），decoder 退化为固定 top-`min_num_select` 的基线 top-k——图是静态形状，`torch.export` 可通过（D8，§4.8）。

### 3.3 开关与兼容策略

- 总开关 = `LTDETRObjectDetectionTrainArgs.density`（`None` = 完全关闭）。关闭时**不构建**任何 dome 模块（state_dict 无新键、参数组无新条目、导出图同基线）。
- 开启时 dome 模式与 decoder 家族的关系：v1 仅支持 `decoder="dfine"`（FDR 已是 Dome 同款）；`decoder="rtdetrv2"` + density 的组合在 dome_decoder 中以同一 `_get_decoder_input` 覆盖实现（RTDETRv2 与 DFINE 共享基类的查询选择，技术上直接可行），但 v1 不测试、不承诺。
- `num_queries` 语义：dome 开启时 `min_num_select` **缺省继承 `num_queries`**（现有脚本改一行即可迁移），`max_num_select` 必须显式给出（默认 500）。

---

## 4. 详细设计

### 4.1 DeFE 密度头与监督（`object_detection_components/defe.py`）

**移植源**：【Dome】`src/zoo/dome/defe.py:15-128`（LiteDeFE）与 `:131-217`（GaussHeatmapGenerator）。

```python
class DepthwiseSeparableConv(nn.Module):     # 3×3 depthwise(dilation) + 1×1 pointwise + ReLU
class LightweightAttention(nn.Module):       # GAP→FC(÷r)→ReLU→FC→Sigmoid 通道注意力
class DeFETrunk(nn.Module):                  # 原 OptimizedDeFE：5 层 DSConv(1,2,3,1,1)+BN，第 3 层后插注意力
class LiteDeFE(nn.Module):
    def __init__(self, channels: int): ...   # D2：通道参数化（Dome 硬编码 256）
    # forward(feats[B,C,H8,W8]) →
    #   conv1(1×1)+AvgPool2 → trunk → density_head(3×3 → ×4上采样 → 1×1 → Sigmoid)
    #   → interpolate ×2 → density[B,1,H/2,W/2]（逐图 max 归一化）
    #   regression_head(GAP→Linear→Sigmoid) → reg_value[B,1]

class GaussHeatmapGenerator:
    def __init__(self, img_size=(Hh, Hw), sigma_ratio=1.2): ...
    # __call__(boxes_normalized_cxcywh[B?,N,4]) → heatmap[1,Hh,Hw]
    # 逐框 6σ 高斯核裁剪叠加，整图 max 归一化（【Dome】defe.py:131-189 原样移植；
    #  AutoVision targets 恒为归一化 cxcywh，不需要 Dome 的 eval 分支换算）
```

**损失函数**（同文件，纯函数，内部 `torch.autocast(enabled=False)` + `.float()`，D6）：

```python
def defe_density_loss(density, gt_heatmap, *, weight=1.0, underestimation_penalty=0.3) -> Tensor
def defe_count_loss(reg_value, gt_counts, *, min_num_select, max_num_select,
                    underestimation_weight=2.0) -> Tensor
```

**设计要点**：

- 密度图分辨率：`proj[0]` 为 stride-8，保持 Dome 的"净 ×4"结构（pool÷2 → 主干 → ×8）落在 **stride-2**（800px 输入下 400×400）。GT 热图按同尺寸生成（归一化框 × img_size，天然分辨率无关）。
- 逐图 max 归一化保留【Dome】`defe.py:122-124`——推理期密度门控依赖它。
- `reg_value` 的监督目标在 criterion 侧由 `gt_counts` 现算（同【Dome】`dome_criterion.py:497-507`），`min/max_num_select` 从 decoder 配置传入，保证单一真源。
- 数值安全：`density.max()==0`（空图）时跳过归一化；损失在 fp32 计算。

### 4.2 MWAS 窗口注意力（`object_detection_components/mwas.py` + `dome_encoder.py`）

**移植源**：【Dome】`src/zoo/dome/get_roi_features.py:89-351`（AxisPermutedEncoderLayer/Encoder、WindowProcessor）。

**模块结构**：

```python
class AxisPermutedEncoderLayer(nn.Module):
    # 同一层内做两次 self-attention：第一次按窗口展平序列，
    # 第二次对序列做轴转置 permute(1,0,2) 后再算一次——
    # 用一维注意力近似覆盖二维窗口交互（【Dome】get_roi_features.py:146-163）

class WindowProcessor(nn.Module):
    def __init__(self, embed_dim, dim_feedforward, num_layers=1, num_heads=8): ...
    # rel_pos_encoder: Linear(2→64→embed_dim)，输入窗口内归一化坐标
    def forward(memory[B,C,H16,W16], win_mask[B,nh,nw], window_size, glob_pos[C,H16,W16])
        → (enhanced[B,C,H16,W16], win_mask)   # 仅掩码窗口残差写回
```

**必须修复/适配的三点**：

1. **pad-to-multiple（D3）**：Dome 断言 `H % window_size == 0`（`get_roi_features.py:209`），而 VTrain 的 scale-jitter ladder 尺寸 ∈ [576, 1024]（步长 32），stride-16 图边长 36..64，对 window=10 不整除（如 576→36）。方案：特征图 `F.pad` 零填充到窗口整数倍 → 计算窗口 → 裁剪回原尺寸；窗口网格数一律 `⌈H16/w⌉ × ⌈W16/w⌉`（`adaptive_max_pool2d` 天然支持非整除）。注意 BN 会见到零填充像素（影响可忽略，窗口内容不受污染，因为注意力只在窗口内部）。
2. **(y,x) 索引统一（D4）**：Dome 在 `dome_decoder.py:847-850` 用 `(cx,cy)` 直索引 `[row,col]`（转置隐患，方形输入下不可见）。移植版全部统一 `(y, x)` 顺序。
3. **去掉调试面（D7）**：`SAVE_INTERMEDIATE_VISUALIZE_RESULT` 开关、`no_valid_windows.log` 追加写（`get_roi_features.py:224-239`）不移植；"零有效窗口"由 `adaptive_threshold` 的降阈兜底（下）保证，不再可能发生。

**密度→窗口掩码**（`dome_encoder.py` 内，替代【Dome】`hybrid_encoder.py:462-498` 的 `adaptive_defe_filter` + 上采样再池化的迂回）：

```python
def adaptive_window_mask(pooled[B,1,nh,nw], *, thresh=0.05, step=0.01) -> BoolTensor[B,nh,nw]:
    # 逐图从 thresh 起按 step 降阈，直到非空；全空（理论上 density 已 max 归一化，不会发生）
    # 则取 argmax 单窗口兜底。语义与 Dome 等价：
    #   "窗口 max 密度 > 阈值" ⇔ "窗口内任一像素 > 阈值"
    # （Dome 先 interpolate 回 (H16,W16) 再逐像素阈值、再 max-pool 回窗口，见
    #   hybrid_encoder.py:516-523——两端都是 max/阈值单调组合，直接在窗口粒度判定即可）
```

**dome encoder 接线**（`DomeHybridEncoder.forward(feats, targets=None) -> (outs, defe_info)`）：

- DeFE 在 `input_proj` 之后、AIFI 之前跑（Dome 同位，`hybrid_encoder.py:511-518`）；
- MWAS 作用于 `proj[1]`（stride-16），窗口注意力 ffn 宽度取 `spec.enc_dim_feedforward`；
- GT 热图生成仅 `self.training and targets is not None` 时执行（对每个图循环逐框生成，与 Dome 一致；框已在 0-1 空间，直接乘目标分辨率）；
- `defe_info = {"density_map", "density_pooled", "window_mask", "reg_value", "gt_heatmap"(训练)}`；
- `deploy()` 语义：`self.static_export = True` 后跳过 DeFE/MWAS，`forward` 退化为基线 `HybridEncoder.forward` 路径（§4.8）。

**成本预估**：800px 下窗口数 ≤ 5×5=25、每窗 10×10=100 token、1 层 d_model=hidden——单图窗口注意力的 FLOPs 相比 AIFI（stride-32 全图 25×25 token 全局注意力）同量级或更小；Python 逐图循环（batch ≤ 8）预计 < 1ms/图，P2 实测后决定是否向量化（§9-R3）。

### 4.3 动态查询选择（`dome_decoder.py` + `dynamic_nms.py`）

**移植源**：【Dome】`src/zoo/dome/dome_decoder.py:767-943`（`_get_decoder_input`/`_select_topk`）与 `src/zoo/dome/dynamic_nms.py:37-73`。

```python
class DomeDFINEv2Decoder(DFINEv2Decoder):
    def __init__(self, *args, density: DensityParams, **kwargs): ...
    # density 携带 min_num_select/max_num_select/nms_iou_base/nms_iou_density_scale

    def _get_decoder_input(self, memory, spatial_shapes, defe_info,
                           denoising_logits=None, denoising_bbox_unact=None):
        anchors, valid_mask = self._generate_anchors(...)          # 基线复用
        output_memory = self.enc_output(valid_mask * memory)
        logits = self.enc_score_head(output_memory)
        coords_unact = self.enc_bbox_head(output_memory) + anchors

        # ① 全网格 top-max_num_select（基线 _select_topk 改 topk 参数）
        mem_k, log_k, anc_k = self._select_topk(output_memory, logits, coords_unact,
                                                self.max_num_select)
        # ② 前 min_num_select 直通；其余按 win_mask[图, ⌊cy·nh⌉, ⌊cx·nw⌉] 门控
        # ③ 逐图（Python 循环，与 Dome 一致）：
        #    bbox = sigmoid(enc_bbox_head(mem) + anc) → xyxy
        #    density_at_center = density_pooled[b, ⌊cy·nh'⌉, ⌊cx·nw'⌉]   # (y,x)（D4）
        #    iou_thr = base + scale · density_at_center        # 0.4 + 0.5·d（【Dome】:852）
        #    keep = per_class_dynamic_nms(boxes, max_class_score, argmax_class, iou_thr)
        #    最终 keep = arange(min_num) ∪ {keep ≥ min_num}    # 前 min 永不剔除
        # ④ pad 到 batch 内 max(n_b)：memory/logits/coords 补零，记录
        #    batch_queries_num: list[int]、query_valid_mask: BoolTensor[B, Q]
        # ⑤ padding 查询 logits 置 −1e4（D5）
        return content(=mem.detach()), coords_unact.detach(), batch_queries_num, query_valid_mask
```

**逐类动态 NMS**（`dynamic_nms.py`，移植【Dome】`dynamic_nms_fast` 向量化版）：按类分组 → 分数降序 → 预计算类内 IoU 矩阵 → 顺序扫描，`suppress = IoU[i, j>i] ≥ thr[i]`。返回保留索引。候选 ≤ max_num_select（VisDrone 500），性能无忧。

**`forward` 改动**（`dome_decoder.py` 覆盖 `DFINEv2Decoder.forward`）：

- CDN 调用传 `num_queries = max(batch_queries_num)` 与 `batch_queries_num`（§4.4）；
- 输出 dict 追加 `batch_queries_num`、`query_valid_mask`、`defe_info` 透传；
- FDR 精修循环、`aux_outputs`、`dn_*`、corners/refs 全部复用基类实现——**不改 `TransformerDecoder`**（变长 Q 对它透明，self-attn mask 是唯一交互点）。

**不变式**（进入 §7 测试）：`min_num_select ≤ n_b ≤ max_num_select`（对每张图）；`query_valid_mask[:, :n_b]` 全真；padding logits ≤ −1e4。

### 4.4 CDN 去噪组扩展（`object_detection_components/denoising.py`）

**移植源**：【Dome】`src/zoo/dome/denoising.py:110-150`。

`get_contrastive_denoising_training_group(...)` 追加两个可选参数：

```python
def get_contrastive_denoising_training_group(
        targets, num_classes, num_queries, class_embed,
        num_denoising=100, label_noise_ratio=0.5, box_noise_scale=1.0,
        batch_queries_num: list[int] | None = None,   # 新增
        num_heads: int | None = None,                 # 新增
):
```

- `batch_queries_num is None`（默认）→ 现有 2D mask 路径逐比特不变；
- 否则构造 3D mask `(B·num_heads, dn+Q, dn+Q)`：在既有"去噪组互不可见"块上，逐 batch 追加 padding 双向屏蔽（有效查询↔padding 互不可见；padding 只见自己，避免 softmax NaN）——`nn.MultiheadAttention(batch_first=True)` 原生支持 3D attn mask，`TransformerDecoderLayer` 把 mask 直通 `self_attn`（`rtdetrv2_decoder.py:425`），**层代码零改动**。

`dn_meta["dn_num_split"]` 的 query 段 = `max(batch_queries_num)`，criterion 的 dn 分支不涉及 padding（dn 段是致密的），无需改动。

### 4.5 损失扩展

**`rtdetrv2_criterion.py`**：

```python
def loss_labels_vfl(outputs, targets, indices, num_boxes,
                    alpha=0.75, gamma=2.0,
                    query_mask: Tensor | None = None):   # 新增，(B, Q) bool
    ...
    loss = (loss * weight)
    if query_mask is not None:
        loss = loss * query_mask.unsqueeze(-1)
    return loss.sum() / num_boxes.clamp(min=1)
```

`RTDETRCriterionV2.forward` 与 `DFINECriterionV2.forward` 增加 `query_mask: Tensor | None = None` 参数并透传到每个 `loss_vfl*` 项（aux 层同乘）。**box 损失不加掩码**——L1/GIoU 只作用于匹配对，与 Dome 一致（`dome_criterion.py:179-198`）。

**`dfine_criterion.py`**：`forward(..., defe_info: dict | None = None)`——非 None 时在全部既有损失之外追加：

```python
losses["defe_density_loss"] = defe_density_loss(defe_info["density_map"],
                                                defe_info["gt_heatmap"], ...)
losses["defe_count_loss"]   = defe_count_loss(defe_info["reg_value"],
                                              [len(t["labels"]) for t in targets], ...)
```

两项权重走 `weight_dict`（`defe_density_loss: 1.0`、`defe_count_loss: 1.0`，与 Dome VisDrone 配方一致，直接进总损失）。注意 `losses["loss"] = Σ` 的既有汇总逻辑自动纳入。

### 4.6 训练 / 验证接线（`task_model.py`、`train_model.py`）

**`task_model.py`**：

```python
def __init__(self, ..., density: LTDETRDensityArgs | None = None):
    ...
    if density is not None and density.enabled:
        self.encoder = DomeHybridEncoder(in_channels, hidden_dim=..., ..., density=density)
        self.decoder = DomeDFINEv2Decoder(**decoder_kwargs, reg_max=..., density=density)
    else:                     # 现状分支，逐比特不变
        self.encoder = HybridEncoder(...)
        self.decoder = DFINEv2Decoder/RTDETRv2Decoder(...)

def _forward_train(self, x, targets=None):
    feats = list(self.backbone(x))
    if self._density_enabled:
        outs, defe_info = self.encoder(feats, targets=targets)
        return self.decoder(outs, targets=targets, defe_info=defe_info)
    outs = self.encoder(feats)
    return self.decoder(outs, targets=targets)
```

推理路径 `forward_backend` 不需要 targets：dome encoder 以 `targets=None` 调用（跳过 GT 热图），dome decoder 的 `defe_info=None` 时仍从 encoder 输出拿密度（见 §3.2 推理态——实现上 encoder 在 dome 模式恒返回元组，`forward_backend` 解包后只取 logits/boxes）。

**`train_model.py`**：

- `__init__`：`density=model_args.density` 传入任务模型；criterion 构造不变。
- `training_step`：从 `out` 取 `query_valid_mask`/`defe_info` 传入 criterion；`log_dict` 自动带上 `defe_density_loss`/`defe_count_loss`。
- `validation_step`：`outputs` 若含 `query_valid_mask` 则传 `query_mask` 给 val 损失；并记录两个新指标（进 `log_dict`，随引擎既有 `val/*` 日志链路输出，`engine.py:694-703`）：
  - `dome_avg_queries = mean(batch_queries_num)`
  - `dome_ample_rate = mean(n_b ≥ n_gt)`
- 优化器分组：dome encoder/decoder 是 `self.model.encoder/decoder` 的子类，`get_optimizer` 的 detector 组（`train_model.py:285-289`）**自动覆盖**新参数；`get_weight_decay_parameters` 对 norm/bias 的豁免同样自动生效。EMA（`ModelEMA` 整模型拷贝）无需改动。

### 4.7 配置面（`config.py`）

```python
class LTDETRDensityArgs(PydanticConfig):
    """密度导向查询/特征分支（Dome-DETR 移植）。None = 关闭。"""
    model_config = {"extra": "forbid"}

    min_num_select: int | Literal["auto"] = "auto"   # "auto" → 继承 num_queries
    max_num_select: int = 500                        # VisDrone 500；AI-TOD 建议 1500
    use_mwas: bool = True
    mwas_window_size: int = 10                       # stride-16 特元窗口
    density_map_weight: float = 1.0                  # Dome defe_density_map_weight
    density_underestimation_penalty: float = 0.3     # Dome density_recall_penalty
    count_loss_weight: float = 1.0                   # 数量回归损失总权重
    count_underestimation_weight: float = 2.0        # Dome reg 损失的低估罚
    window_threshold: float = 0.05                   # 自适应降阈初值
    window_threshold_step: float = 0.01
    nms_iou_base: float = 0.4                        # IoU 阈值 = base + scale·密度
    nms_iou_density_scale: float = 0.5
    gaussian_sigma_ratio: float = 1.2                # GT 热图 σ = ratio·框宽高
```

`LTDETRObjectDetectionTrainArgs` 增加 `density: LTDETRDensityArgs | None = None`（`model_config.extra=forbid` 已有，pydantic 自动完成嵌套校验）。`resolve_auto` 不涉及。

**脚本用法**（§8 给完整脚本）：

```python
model_args = {
    ..., 
    "num_queries": 250,                     # dome 下即 min_num_select
    "density": {"max_num_select": 500},     # 其余字段走默认
}
```

### 4.8 导出契约（deploy 静态模式）

**问题**：`export_onnx` 走 `torch.export` 严格模式（`export_onnx.py:106-112`），而 dome 的查询选择含 `torch.nonzero`/数据依赖循环（窗口门控、动态 NMS），**不可导出**；且输出 Q 逐输入变化，破坏静态 shape 契约。

**设计（D8，实施修正）**：导出路径本就调用 `TaskModel.deploy()`（`export_onnx.py:93-95`），基类实现**递归调用所有模块的 `convert_to_deploy`**——这就是既有导出契约（RepVgg 融合、deformable 采样器切换在基线导出中早已发生）。因此**任务模型无需覆写 `deploy()`**：dome 模块各自实现 `convert_to_deploy` 即可自然参与递归——

- `DomeHybridEncoder.convert_to_deploy()`：置 `_static_export`，forward 退化为基线 `super().forward(feats)` 路径（跳过 DeFE/MWAS）；
- `DomeDFINEv2Decoder.convert_to_deploy()`：置 `_static_export`，查询选择退化为基类固定 top-`num_queries`（= min_num_select）分支，输出形状静态。

- 语义损失：极密帧最多保留 `min_num_select` 个候选（VisDrone 250），配合 `eval_max_dets=500` 的评测协议会有小幅 AP 差——§8 消融中量化（Dome 的 onnx-export 分支同样如此处理）；
- 验证：`verify_onnx_export_outputs`（`task_model.py:308-319`）自动覆盖 deploy 后路径。

### 4.9 DDP / EMA / AMP / 引擎契约

| 项 | 结论 |
|---|---|
| DDP | 每 rank 独立 pad 到本 rank 批内 max，`find_unused_parameters=True`（默认，`engine.py:506`）兜底任何"某步未用模块"的边角（如全空 GT 图跳过热图生成分支）。梯度累积 `no_sync()` 逻辑不受影响。 |
| AMP | 引擎 bf16 autocast 包住 `training_step`（`engine.py:662-680`）。密度损失与 GT 热图生成内部强制 fp32（D6，对齐仓内"fp32 island"惯例，如 `dfine_decoder.py:122-125`）。逐图 max 归一化在 fp32 内做。 |
| EMA | `ModelEMA` 整模型 deepcopy，dome 参数自动纳入。 |
| NaN 监控 | 引擎 `nan_monitor` 检查 loss/梯度；密度损失的 fp32 island + GT 热图有限性（高斯核归一化）保证无新 NaN 源。 |
| 检查点 | 新模块参数随 state_dict 自然进/出；**dome 开关改变架构**，resume 时须与训练时同配置（`config.py` 的 `extra=forbid` + init-args 快照机制已保证）。 |

---

## 5. 与 Dome 原实现的差异清单

| # | 差异 | Dome 现状 | VTrain 决策 | 理由 |
|---|---|---|---|---|
| D1 | DeFE 输入 stride-8（密度图 stride-2） | stride-4 输入、密度图全分辨率 | 【差异→已提供闭环】 | 初版受 3 层金字塔限制；现已支持 `pyramid_levels=4`（strides 4/8/16/32，ConvNeXt 家族专属），该模式下 DeFE 回到 stride-4、密度图全分辨率、MWAS 在 stride-8——与 Dome 完全同构；VisDrone 实验脚本已启用 |
| D2 | 通道参数化 | LiteDeFE 硬编码 256 | 【差异】 | hidden_dim 随变体（192/224/256/384…） |
| D3 | MWAS pad-to-multiple + ⌈·⌉ 网格 | `H % window == 0` 硬断言 | 【差异】 | scale-jitter ladder（576..1024）不整除窗口 |
| D4 | 密度索引统一 (y,x) | `dome_decoder.py:847-850` 存在 (x,y)/(y,x) 混用 | 【差异】（bug 修复） | 方形输入下不可见的转置隐患，防御性修复 |
| D5 | padding 查询 logits 源头置 −1e4 | 仅在损失端掩码，padding 查询可参与匹配 | 【差异】 | 杜绝 padding 查询被 Hungarian 选中/被 decode 输出；训练动力学影响趋近于零 |
| D6 | 密度损失 fp32 island | fp16 autocast 内计算 | 【差异】 | 本引擎为 bf16；对齐仓内 fp32 island 惯例 |
| D7 | 移除调试面 | 可视化开关、`no_valid_windows.log` | 【差异】 | 工程卫生；自适应阈值兜底使"零窗口"分支不可达 |
| D8 | 导出静态模式 | 官方 onnx-export 分支 | 【差异】 | `torch.export` 不支持数据依赖控制流；`deploy()` 是仓内既有钩子 |
| D9 | 步数制 flat-cosine | epoch 制 MultiStepLR + 停增广 | 【差异】 | 引擎契约，超参映射见 §8 |
| D10 | 无 pre_bbox_head / enc-aux / LQE | Dome 有 | 【差异】 | VTrain 的 DFINEv2Decoder 本就没有；查询选择移植不依赖它们 |
| D11 | 数量回归损失用 fp32 目标 | `dome_criterion.py:507` 把 (0,1) 目标存入 **int64** 张量，几乎所有图的监督目标被截断为 0（上游 bug） | 【差异】（bug 修复） | `reg_value` 是纯辅助头，正确目标只改善训练信号；保留 int64 行为没有意义 |
| D12 | GT 热图按框分块向量化 | 逐框 Python 循环贴核 | 【差异】（等价实现） | 数学相同（±3σ 截断 + 逐核 sum 归一化 + 整图 max 归一化），密集帧（>500 框）显著加速 |
| — | 窗口掩码直接在窗口粒度阈值化 | 上采样到像素级再阈值再池化回窗口 | 【简化】 | max→阈值→max 与 max→阈值在窗口粒度语义等价，省两次重采样 |
| — | 无 padding 时 CDN 保持 2D mask | 给定 batch_queries_num 即扩 3D | 【简化】 | 3D repeat(2D) 与 2D 语义相同；省一次掩码膨胀 |

---

## 6. 实施计划（四个 PR，每个独立可合入）

> **实施记录（v1.1）**：四个阶段已合并实现（单分支交付）。验收结果：`tests/test_ltdetr_dome.py` 44 例全绿；全量回归 927+ 通过（3 个 eomt 模块因环境缺 `EomtDinov3*` 无法收集，与本工作无关）。审查修复：OBB criterion 覆写点补 `query_mask` seam、`ema.py` foreach 空列表守卫（既有改动引入）、导出 init-args 快照 pydantic→dict（安全加载器兼容）、密度损失权重单一来源（weight_dict）、deploy 契约回归基类递归。
>
> **实施后记（v1.2，首次真实训练后）**：VisDrone 800px 首跑暴露并修复三项——
> ① **性能**：动态 NMS 的逐候选 torch 循环（每步 ~500 次 GPU 同步）+ 散布的 per-image `.any()`/分支同步使步时翻倍（1.81 s vs 基线 0.89 s）。修复：NMS 贪心扫描移到 CPU numpy（一次批量传输，语义不变，微基准 30.8→8.4 ms/次，GPU 上 keep 集合一致）；密度归一化改无分支 clamp；窗口掩码 any() 批量化。分解基准（800px/batch4/fwd+bwd）：基线 dfine 346 ms → dome 444 ms（**+28%，+97 ms**）。
> ② **指标**：`dome_avg_queries`/`dome_ample_rate` 原挂 validation log_dict 被引擎丢弃，改为按图累计 + 经 `compute_metrics("val")` 落盘（DDP 经 export/merge_state 合并各 rank 分片）。
> ③ **消融纯度**：历史基线脚本实为 rtdetrv2 + 旧优化器（decoder="auto" 解析），与 dome 差 4 个变量。基线脚本重写为严格孪生（同 safetensors COCO 骨干 / dfine / Dome 配方 / 64,710 步 / 同增广窗，唯一缺 `density`），并加程序化孪生一致性测试（`TestScriptTwin`）。
>
> **实施后记（v1.3，数据驱动核查后）**：对 VisDrone 标注的实测统计（每图目标数 P50/P90/P99 = 42/104/212，tiny 占 60–69%，>250 目标的图仅 0.4%）表明：
> ① **D1 结构风险已闭环**——新增 `pyramid_levels: 3|4` 训练参数（ConvNeXt 家族 4-stage 骨干，ViT/ECViT 拒绝），4 层模式下 DeFE@stride-4、密度图全分辨率、MWAS@stride-8、AIFI@最深层，与 Dome 同构；实验脚本已切到 4 层。密闭测试用本地 tiny DINOv3-ConvNeXt 构建覆盖全链路形状。
> ② **查询扩张臂在本文献上基本休眠**——min_num_select=250 高于 P99=212，动态查询将常驻下限；`dome_ample_rate` 健康值应≈1.0，`dome_avg_queries` 预期 ~250–270。数量回归头目标几乎恒 0（reg_value 仅训练辅助，无推理影响），其损失趋零不作为收敛证据。保持 250/500 以维持与 Dome 的配方对齐；如需榨取效率可另做 min=150 附加消融。

### PR-1：DeFE 密度头 + 损失（不改变检测行为）

- 新增 `defe.py`（模块 + 两个损失纯函数）；
- `dfine_criterion.py` 增加可选 `defe_info`（此时任务模型尚不产出，参数恒 None）；
- `task_model.py` 增加 `density` 开关但**仅构建 DeFE 头**（挂 encoder，损失生效，查询仍固定 top-`num_queries`）；
- 测试：热图峰值在框中心、空 GT 图损失为 0、fp32/bf16 数值、开关关闭时 state_dict 无新键。
- **验收**：VisDrone 800px 短程 smoke（2k 步）loss 收敛、`defe_density_loss` 单调下降。

### PR-2：动态查询选择（PAQI）

- 新增 `dynamic_nms.py`、`dome_decoder.py`；`denoising.py` 3D mask 扩展；`rtdetrv2_criterion.py` `query_mask`；
- `config.py` 增 `LTDETRDensityArgs`；`train_model.py` 接线与日志。
- 测试：§7 全部不变式与等价性用例。
- **验收**：单测全绿；VisDrone smoke 中 `dome_avg_queries` ∈ (250, 500) 且与图内 GT 数正相关；`dome_ample_rate` > 0.9。

### PR-3：MWAS 窗口注意力

- 新增 `mwas.py`、`dome_encoder.py`（含 pad-to-multiple）；
- **验收**：576/800/1024 三种 ladder 尺寸前向形状正确；VisDrone 800 完整训练对比（§8 消融 d）。

### PR-4：导出 + 实验脚本 + 文档

- `deploy()` 静态模式、`scripts/visdrone_ltdetr_dome_800.py`、本分册状态更新。
- **验收**：dome 模型 ONNX 导出成功且 `verify_onnx_export_outputs` 通过；静态 vs 动态 AP 差量化记录。

---

## 7. 测试计划（`tests/test_ltdetr_dome.py`）

风格对齐 `tests/test_ltdetr_object_detection.py`（离线 `dinov2-vittest-notpretrained` 变体 + 小图，全部 hermetic）：

| 用例 | 断言 |
|---|---|
| `test_density_off_is_baseline` | `density=None` 时无 `Dome*` 模块、state_dict 键集合与基线一致、前向输出一致 |
| `test_gt_heatmap_peaks_at_centers` | 单框热图 argmax == 框中心像素；两框叠加归一化后 max=1 |
| `test_density_losses_finite_and_backward` | 空GT/正常/满GT 三种 targets 下损失有限、`.backward()` 梯度非 NaN |
| `test_count_loss_asymmetry` | 低估惩罚是高估的 2 倍（同偏差幅度） |
| `test_query_count_invariants` | 构造稀疏/密集混合 batch：每图 `min ≤ n_b ≤ max`；`query_valid_mask` 前缀全真 |
| `test_padding_logits_masked` | padding 位置 logits ≤ −1e4；decode 后无框来自 padding |
| `test_cdn_3d_mask_no_leakage` | 有效查询↔padding 双向屏蔽；去噪组间互不可见；注意力前向无 NaN |
| `test_vfl_masked_equals_manual` | 带掩码 VFL == 逐图切片手算之和（相对误差 < 1e-6） |
| `test_dynamic_nms_matches_reference` | 与朴素逐类循环实现一致；密度↑ → 保留框数单调不减 |
| `test_mwas_pad_to_multiple` | 576/672/1024 输入下 stride-16 图 36/42/64 边长前向成功；掩码窗口外残差为 0 |
| `test_mwas_residual_locality` | 增强只发生在 win_mask 窗口内（重建前后特征差分的支持集 == 掩码窗口集合） |
| `test_export_static_shapes` | `deploy()` 后 `torch.export` 成功、输出 Q == min_num_select、与 EMA 权重一致性校验 |
| `test_checkpoint_roundtrip` | dome 开启训练 2 步 → 存/载 → loss 连续 |

金标回归：`tests/golden/` 若覆盖 LT-DETR 前向，确认 dome 关闭路径不产生新差异。

---

## 8. VisDrone 训练配方与消融

**基座**：`scripts/visdrone_ltdetr_dome_800.py`（复制 `visdrone_ltdetr_800.py`，29k 步 flat-cosine、batch 4 × 累积 4、`eval_max_dets=500` 全部保持——与 640/800 基线可比）。

```python
model_args = {
    "backbone_weights": BACKBONE_WEIGHTS,
    "image_size": (800, 800),
    "num_queries": 250,            # dome 下 = min_num_select（Dome-S VisDrone 同值）
    "backbone_lr_factor": 0.05,
    "ema_warmup_steps": 2000,
    "ema_momentum": 0.999,
    "eval_max_dets": 500,
    "density": {"max_num_select": 500},   # 其余默认：window 10、IoU 0.4+0.5d、w_den 1.0
}
```

**超参对照 Dome**（`configs/dome/Dome-S-VisDrone.yml:26-47`）：`min/max = 250/500`、`mwas_window_size=10`、`defe_density_map_weight=1` 一致；Dome 的 lr/调度不移植（D9），沿用本仓 AdamW + flat-cosine 与 lr 缩放公式（`train_model.py:246`）。

**消融矩阵**（同一 29k 步配方，唯一变量 = density 配置）：

| 组 | density | min/max | MWAS | 目的 |
|---|---|---|---|---|
| a（基线） | off | 300 固定 | – | 现有 800px 基线 |
| b | on | 250/500 | off | DeFE 损失 + 动态查询的净贡献 |
| c | on | 250/500 | on | 完整 Dome 组合 |
| d | on | 300/500 | on | min 抬平后与基线对比（隔离查询数效应） |

记录：mAP@[.5:.95]、APs、`dome_avg_queries`、`dome_ample_rate`、步时延、`deploy()` 静态导出后 mAP。

---

## 9. 风险与开放问题

| # | 风险 | 缓解 |
|---|---|---|
| R1 | padding 查询被匹配器选中、污染匹配 | D5 源头置 −1e4 后分类代价恒定、框代价大，实践中不会被 Hungarian 选中；PR-2 加专项测试 |
| R2 | 动态 Q 造成批内形状抖动（cudnn benchmark / kernel 缓存抖动） | 本仓未用 compile；deformable attn 对 Q 一维动态不敏感；smoke 实测步时延 |
| R3 | MWAS 逐图 Python 循环开销 | 预估 <1ms/图（§4.2）；不达标则 P3 向量化（`torch.gather` 窗口批处理） |
| R4 | 密度头与 SSL backbone 特征域差（DINOv3 vs Dome 的 HGNetv2） | DeFE 从头训练且损失独立，理论上域无关；消融 b 组直接度量 |
| R5 | 逐图 max 归一化在空图/纯背景帧的数值稳定 | fp32 + `max>0` 守卫；`adaptive_window_mask` argmax 兜底 |
| R6 | 静态导出（Q=min）在极密帧掉点 | §8 记录静态/动态 AP 差；必要时把 min_num_select 作为导出参数上调 |
| R7 | RT-DETRv2 decoder + density 组合未验证 | v1 文档标注"实验性"；dome decoder 的查询选择与 decoder 家族解耦，后续补测 |

**开放问题**（评审时定）：

1. `min_num_select` 的 "auto" 继承 `num_queries` vs 独立字段——倾向继承（迁移成本最低）；
2. 密度图 stride-2 分辨率（400×400@800）的 GT 热图生成开销是否可接受——如偏高，降为 stride-4（去掉最后 ×2 插值），对门控/NMS 无影响；
3. `dome_avg_queries`/`dome_ample_rate` 是否纳入 `compute_metrics` 正式指标（进 checkpoint 元数据）还是仅日志。

---

## 10. 参考索引

**Dome-DETR 参考实现**（`/home/hdjs/Research/Software/Dome-DETR-master/`）：

| 内容 | 位置 |
|---|---|
| DeFE 模块 / GT 热图 | `src/zoo/dome/defe.py:15-128` / `:131-217` |
| MWAS 窗口注意力 | `src/zoo/dome/get_roi_features.py:89-351` |
| encoder 侧 DeFE/MWAS 接线、自适应阈值 | `src/zoo/dome/hybrid_encoder.py:462-556` |
| 动态查询选择 / 密度调制 NMS 调用 | `src/zoo/dome/dome_decoder.py:767-943` |
| CDN 的 per-batch padding mask | `src/zoo/dome/denoising.py:110-150` |
| 逐类动态 NMS（向量化） | `src/zoo/dome/dynamic_nms.py:37-73` |
| 密度损失 / 查询掩码损失 | `src/zoo/dome/dome_criterion.py:91-136,491-528` |
| VisDrone 超参 | `configs/dome/Dome-S-VisDrone.yml` |
| 评测辅助指标 | `src/solver/det_engine.py:238-254` |

**VTrain 受影响面**（`src/autovision/`）：

| 内容 | 位置 |
|---|---|
| 任务模型组装 / 前向 | `_task_models/ltdetr_object_detection/task_model.py:88-207,261-273` |
| 训练参数与变体注册表 | `_task_models/ltdetr_object_detection/config.py:374-473` |
| 训练包装（损失接线/优化器/EMA） | `_task_models/ltdetr_object_detection/train_model.py:56-147,224-314` |
| 共享 HybridEncoder | `_task_models/object_detection_components/hybrid_encoder.py:192-327` |
| RT-DETRv2 decoder（查询选择基类） | `_task_models/object_detection_components/rtdetrv2_decoder.py:715-832` |
| D-FINE decoder（FDR 循环） | `_task_models/object_detection_components/dfine_decoder.py:46-177` |
| CDN 去噪组 | `_task_models/object_detection_components/denoising.py:46-168` |
| VFL 损失 / D-FINE criterion | `_task_models/object_detection_components/rtdetrv2_criterion.py:30-62`、`dfine_criterion.py:66-169` |
| 解码 / 引擎 / 导出钩子 | `_task_models/object_detection_components/decode.py:22-57`、`_engine/engine.py:493-508,662-703`、`_export/export_onnx.py:75-112` |

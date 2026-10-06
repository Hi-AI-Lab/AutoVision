# VTrain 实现蓝图 · 分册 14：LT-DETR 逐物体对比 SSL 辅助头（SimCSE 式）集成设计

| 文档属性 | 内容 |
|---|---|
| 版本 | v1.3（已实施，含 Route B、D14 未配对负例、评测探针；实施记录与审查修复见 §6） |
| 状态 | 已实现并通过 `tests/test_ltdetr_ssl.py`（70 例）+ `tests/test_ssl_probes.py`（6 例）+ `tests/test_ltdetr_ssl_command.py`（3 例）与全部受影响回归 |
| 上游方法 | RF-DETR 分类头 SSL v2.1：逐物体、与 box 匹配对齐的 SimCSE 式对比学习（`L = L_det + λ·L_ssl`） |
| 参考实现 | `/home/hdjs/Research/Software/rf-detr-ssl`（RF-DETR v1.9.4 + SSL 扩展；权威设计文档 `docs/learn/train/ssl-per-object-classification-head.md` v2.1，M1–M5 已实施并带 VisDrone-mini 消融） |
| 受影响模块 | `_task_models/object_detection_components/`、`_task_models/ltdetr_object_detection/`、`_data/`、`_transforms/detection_transforms/`、`_commands/train_task/object_detection.py`、`tests/`、`scripts/` |
| 基线兼容承诺 | `ssl=None` 时 LT-DETR 全部既有行为、权重、导出图**逐比特不变**；`ssl={"mode": "off", "view": "two_crop"}` 提供与 SSL 运行**批匹配**的无 SSL 基线（§8） |

> **标注约定**：【RF-DETR-SSL】参考实现中的已验证事实（出处 `文件:行`，路径相对 `/home/hdjs/Research/Software/rf-detr-ssl/`）；【VTrain】本仓库现状（出处 `文件:行`，路径相对 `src/autovision/`）；【设计】本文档决策；【差异】有意偏离 RF-DETR-SSL 的决策（附编号 D1–D12 及理由）。

---

## 1. 背景与目标

### 1.1 方法摘要（RF-DETR-SSL）

【RF-DETR-SSL】该扩展在检测损失之外，对 **decoder 最终层 query 特征**（即分类头 `class_embed` 的输入表示）施加 SimCSE 式对比目标。核心语义映射：

| SimCSE（NLP） | 本方法（检测） |
|---|---|
| 一条句子 | 一个**匹配上的物体实例** `(matched query, GT box, label)`——与 `loss_boxes`/`loss_labels` 用**同一批匈牙利匹配**，零新增匹配 |
| 同一句子的两个 dropout 视图 | 同一物体的两个增广视图（`two_crop`）或投影头两次 dropout 采样（`dropout`） |
| 对角线 arange = 正对 | 跨视图对应由 **box-IoU 匈牙利 + 标签双重校验**恢复（弱对直接丢弃） |
| batch 内其余句子 = 负例 | batch 内其余全部匹配实例（DDP 下跨卡 gather 扩池，槽位替换保梯度） |
| NLI 显式正对 + 硬负列偏置 | `sup` 模式 = SupCon 式同标签软正对扩展（**非** SimCSE-sup 忠实移植）；硬负偏置改造为 **IoU 软对齐**：正对 logit 加 `λ_w·log(iou^γ)`，非正对位恒为 0 偏置 |

总损失 `L = L_det + λ·L_ssl`，λ 沿 `weight_dict` 机制。三种模式：`unsup`（SimCSE InfoNCE）/ `sup`（SupCon 式扩展，同标签互为软正对，可限定同图）/ `both`。训练专用投影头（`Linear+Tanh` 或 `mlp2`）挂在**模型**上（checkpoint/DDP/EMA 自动覆盖），推理/导出自然旁路。

【RF-DETR-SSL】上游消融结论（M4，VisDrone-mini 200 图、RFDETRNano、5 epochs、单种子，设计文档 §16）：

| 配置 | best EMA mAP | Δ vs base |
|---|---|---|
| base（无 SSL，batch 匹配） | 0.1191 | — |
| **unsup two_crop λ=0.1** | **0.1452** | **+0.0261** |
| unsup dropout（投影头噪声，λ=0.5） | 0.1366 | +0.0175 |
| sup two_crop same_image=True | 0.1343 | +0.0152 |
| unsup two_crop λ=1.0 | 0.1122 | **−0.0068（唯一负收益）** |

跨轮稳定结论：**低 λ（0.1–0.25）最稳健；λ=1.0 有毒**；其余排名受单种子噪声影响。推荐起点：`unsup, two_crop, λ=0.1, τ=0.05, iou_match=0.55`。B4 前置测量（同图同类共存占比）在 VisDrone-mini 上为 98.6%——拥挤场景假负例压力近饱和，但 unsup 仍优于 sup（见 §9 开放问题 5）。

### 1.2 集成范围

**做**：

- 将数学组件（投影头、InfoNCE/SupCon 损失、分布式 gather、IoU 偏置、跨视图配对）移植进共享组件层，作为 LT-DETR 家族的**可选 SSL 辅助分支**（一个配置开关 `ssl`）；
- 首战场景 = VisDrone + `ltdetr/dinov3-convnext-tiny` + D-FINE decoder @800px，与分册 13 的 DOME 密度分支**可组合**（§4.9）；
- `two_crop`（双增广视图，主力档）与 `dropout`（零数据改动对照档）两档视图；
- D-FINE decoder 家族 v1 支持（RT-DETRv2 decoder 家族与 OBB 见 §9 开放问题 1/2）。

**不做（非目标）**：

- 不改动 RT-DETR / RT-DETRv2 / OBB 家族的任何行为（v1 边界，配置显式拒绝）；
- 不移植上游未实现的设计项（`ssl_box_feat` R 通道 RoIAlign、`ssl_pos_decompose` 位置分解、λ 预热、硬负列偏置——上游文档 §9 已列为设计-only 或 experimental）；
- 不做 epoch 制调度改动（沿引擎 flat-cosine 契约）；
- Route B（无标注实例源）只给设计草图，v1 不实现（§4.10）。

### 1.3 预期收益与判据

- 主指标：VisDrone val mAP（`eval_max_dets=500`，【VTrain】`config.py:425` 已对齐）。
- 新增诊断指标：`ssl_paired_ratio`（跨视图配对产率，§4.6）与 `train loss_ssl` 首→末走势。
- 成功判据（对齐上游验收标准）：
  1. **批匹配基线不回归**：相对 `mode="off", view="two_crop"` 基线（同 2B 数据管线、无 SSL 损失），mAP 变化 ≥ −0.2 AP；
  2. **正增益**：推荐配置（unsup two_crop λ=0.1）mAP ≥ +0.5 AP（上游小规模 +2.6 AP，全量数据集预期收窄）；
  3. 与 DOME 组合（§4.9）时密度分支收益不被侵蚀。

---

## 2. 两套代码的现状对照

### 2.1 架构差异表

| 维度 | RF-DETR-SSL【RF-DETR-SSL】 | VTrain LT-DETR【VTrain】 | 移植含义 |
|---|---|---|---|
| 查询制 | 固定 `num_queries × group_detr(13)` 组查询，two-stage | 固定 top-`num_queries`(300)；DOME 模式动态 250~500 + 逐图 padding + CDN 去噪段 | 实例池构造不同：上游按 group-0 过滤，VTrain 按主段匹配索引 + `query_valid_mask` 过滤（D1） |
| 分类头输入 | `class_embed(hs[-1])`，最终层隐状态 | `dec_score_head[-1](output)`，最终层隐状态——**当前未暴露**，需在 `_refine_layers` 追加返回（§4.2） | 新增 `out["query_features"]` 发射 |
| 匹配 | `SetCriterion.forward` 内最终层一次匹配，aux/enc 循环复用列表 | `DFINECriterionV2.forward` **每层**各跑一次匹配，`layer_indices[-1]` 即最终层（`dfine_criterion.py:111-114`） | 直接取 `layer_indices[-1]`，比上游更直接 |
| 损失汇总 | `weight_dict` aux 复制循环**之后**追加 `loss_ssl`，杜绝 `_aux_i` 垃圾键 | `weights` 复制循环覆盖全部键，聚合时 `if key in losses` 过滤（`dfine_criterion.py:200-210`）——DeFE 损失已用"算一次、无后缀、weight_dict 单次"模式（`:186-198`） | SSL 走 **DeFE 同款模式**，复制循环自动过滤空键（D7） |
| 实例来源 | `matched_gt` / `encoder_proposals`（two-stage 编码器 top-k token） | `matched_gt`；Route B 对应物 = decoder top-k 候选处的 encoder memory token（content 已 detach，需绕行，§4.10） | Route B 重命名 `encoder_candidates`，v2（D8） |
| two-view 数据 | `TwoViewDataset` 包装 + 专用双视图 collate（view-major 2B、联合 padding）；必须包在 `GradAccumAlignedDataset` 内层（D4 约束） | `ObjectDetectionDataset.__getitem__` 每次 transform 调用独立随机（`_data/object_detection_dataset.py:35-71`）；collate `RTDETRDetectionCollateFunction` 含批级 scale-jitter/mixup；**引擎梯度累积是连续拉取 loader 批**（`engine.py:663-668`），loader 批即微批 | 数据集级包装（item = 二元组）+ collate 扁平化后**单次联合调用**即可，批内天然保持配对，**无 D4 对应问题**（D4） |
| 批级增广 | 无 mixup；多 scale 在数据集侧 | 批级 scale-jitter + mixup（`collate.py:109-149`）+ 逐样本 mosaic | mixup/mosaic 保持开启，配对由 IoU+标签校验天然容忍（D5） |
| 目标格式 | 归一化 cxcywh（transform 管线同步） | 归一化 cxcywh（`training_step:127-130` 从 `batch["bboxes"]` 构建） | 一致，`box_cxcywh_to_xyxy` 直接可用 |
| 精度策略 | bf16/fp16 autocast，SSL 头内 fp32 归一化 | bf16 autocast 包住 `training_step`（`engine.py:656-680`）；仓内 fp32-island 惯例（`dfine_decoder.py:178-181`） | 投影头输出强制 fp32（上游已如此），损失全程 fp32（D6 沿分册 13 惯例） |
| 优化器 | `get_param_dict`：backbone 层级衰减 / decoder / 其余=base lr | 显式组 `backbone / detector`（connector+encoder+decoder），lr×`sqrt(global/default)`（`train_model.py:267-357`） | `ssl_head` 不在 encoder/decoder 之下 → **显式 `ssl` 参数组**（§4.6） |
| 未用参数 DDP | DDP 常规 | `find_unused_parameters=True` 默认（`engine.py:506`，env 可关） | `every_n_steps>1` 时跳步未用参数由默认值兜底 |
| 训练制 | epoch 制（Lightning）+ auto-batch 按 2B 校准 | 步数制 flat-cosine；loader 批=微批 | 无 auto-batch 探测器需要适配；batch_size 语义文档化（§8） |
| 导出 | train-only 头不定义 `export()` 自然跳过；`ssl_head.` 进缺失键白名单 | `forward_backend` 只返回 `(logits, boxes)`（`task_model.py:325-327`），SSL 头不进图；导出 state dict 需过滤 `ssl_head.*` | 导出零影响 + 卫生过滤（D9） |
| EMA | `AveragedModel` 整体平均，自动覆盖头 | `ModelEMA` 整模型 deepcopy（`train_model.py:365-367`） | 自动，无需改动 |

### 2.2 移植主线结论

1. **数学组件是"纯函数插件"**：上游 `ssl_head.py` 仅依赖 torch/scipy/torchvision（scipy 本仓已有，`matcher.py:11`），可近乎逐行移植为 `object_detection_components/ssl_contrastive.py`。
2. **挂载点三处**：decoder 暴露 `query_features`（唯一的行为面改动，训练-only）→ 任务模型持头发射 `ssl_z1/z2` → criterion 最终层追加 `loss_ssl`。推理/导出/验证路径零改动。
3. **数据侧比上游更简单**：引擎的"loader 批=微批"消除了上游 D4（GradAccum 对齐）约束；数据集级二元组包装 + 现有 collate 单次联合调用即可。
4. **实施顺序按依赖链**：数学组件 + criterion/decoder 接线（dropout 视图先全链路打通，零数据改动）→ two_crop 数据管线 → DOME 组合验证与消融。

---

## 3. 总体设计

### 3.1 模块拓扑

**新增文件**：

```
src/autovision/_task_models/object_detection_components/
└── ssl_contrastive.py   # 数学组件 + SSLParams + ssl_contrastive_loss 编排（§4.1/§4.4）

src/autovision/_data/
└── two_view.py          # TwoViewObjectDetectionDataset（§4.5）

scripts/
└── ssl_cooccurrence_stats.py   # B4 假负例压力前置测量（§4.5，自上游 ssl_stats.py 移植）

tests/test_ltdetr_ssl.py # 全部单元/性质测试（§7）
scripts/visdrone_ltdetr_ssl_800.py  # VisDrone 对比实验脚本（§8）
```

**修改文件**（全部向后兼容，新增参数默认关闭）：

| 文件 | 改动 |
|---|---|
| `object_detection_components/dfine_decoder.py` | `_refine_layers` 追加返回最终层隐状态（主段分离后）；`forward` 训练分支发射 `out["query_features"]`（§4.2） |
| `object_detection_components/dfine_criterion.py` | ctor 增加 `ssl_params`；`forward` 增加 `ssl_z1`/`ssl_z2` 可选参数，最终层匹配上追加 `losses["loss_ssl"]`（§4.4） |
| `ltdetr_object_detection/task_model.py` | `__init__` 按 `ssl` 开关构建 `self.ssl_head`；`_forward_train` 发射 `ssl_z1/z2`（§4.3） |
| `ltdetr_object_detection/train_model.py` | 训练步提取 ssl 特征与步门控；`ssl` 显式参数组；checkpoint 兼容（缺 `ssl_head.*` 键以当前值填充，EMA 路径同）；配对产率日志（§4.6） |
| `ltdetr_object_detection/config.py` | 新增 `LTDETRSSLArgs`；`LTDETRObjectDetectionTrainArgs.ssl: LTDETRSSLArgs \| None = None`（§4.7） |
| `_transforms/detection_transforms/collate.py` | 新增 `TwoViewDetectionCollateFunction`（包装现有类，§4.5） |
| `_commands/train_task/object_detection.py` | `ssl.view="two_crop"` 时包装 train 数据集与 collate（§4.5） |
| `ltdetr_object_detection/__init__.py` | 导出（可选） |

### 3.2 数据流（三态）

**训练态 two_crop**（主力）：

```
TwoViewObjectDetectionDataset(train_dataset)
  item_i = (base[i], base[i])      # 同一原图两次独立增广采样
                                   #（photometric/zoomout/IoU-crop/flip/mosaic 逐次随机，框随几何同步）
  ▼
TwoViewDetectionCollateFunction(base_collate)
  items = [v1_0..v1_{B-1}, v2_0..v2_{B-1}]  → base_collate 单次调用
  ► batch：键与单视图完全一致，2B 行，row r ↔ row r+B 同源
    （批级 scale-jitter 对 2B 取同一 scale；mixup 保持既有窗口语义，D5）
  ▼
training_step（不改）: targets = bboxes/classes(2B) → model._forward_train(x, targets)
  backbone(2B) → encoder → decoder._refine_layers
  out["query_features"] = 最终层隐状态（主段，DN 分离后）(2B, Q, C)   ← 仅 training
  out["ssl_z1"] = ssl_head(query_features)   ← 两次独立 dropout 采样
  out["ssl_z2"] = ssl_head(query_features)
  ▼
DFINECriterionV2.forward(outputs_list, targets, ..., ssl_z1, ssl_z2)
  indices_final = layer_indices[-1]            ← 与 box 损失同一批匹配，零新增匹配
  池 = 匹配实例 ∩ query_valid_mask（DOME padding 防御过滤）
  two_crop: 逐图 match_cross_view(IoU 匈牙利 + 标签双校验, iou_match) 恢复跨视图正对
  → padded gather（DDP 全局负例池，槽位替换保梯度，all_reduce 共识防死锁）
  → IoU 偏置（正对位 +λ_w·log(iou^γ)，非正对位 0）
  → InfoNCE [+ SupCon]
  losses["loss_ssl"]（无后缀，weight_dict 单次乘 λ）
```

**训练态 dropout**（对照档，零数据改动）：数据管线完全不变（B 行）；`ssl_z1/z2` 仍为两次 dropout 采样，criterion 里同实例对 = 对角线（IoU=1，`log 1=0` 无偏置），`sup` 的 `same_image_positive_only` 在此档无意义（校验拒绝）。

**推理/验证/导出态**：`decoder.forward` 的 `query_features` 只在 `self.training` 发射；任务模型的 `ssl_z1/z2` 只在 `self.training` 且 `ssl_head` 存在时发射 → 验证损失、`predict`、`forward_backend`、`torch.export` 图、EMA 评测全部与基线逐比特一致。`ssl_head` 参数随 state_dict 存在但不参与任何推理计算。

### 3.3 开关与兼容策略

- 总开关 = `LTDETRObjectDetectionTrainArgs.ssl`（`None` = 完全关闭，不构建任何 SSL 模块，state_dict 无新键、参数组无新条目、导出图同基线）。
- `ssl.mode="off"` + `ssl.view="two_crop"` 是合法组合 = **批匹配基线**（数据管线翻倍、损失关闭），对齐上游 C2/D7 公平性要求【RF-DETR-SSL】auto_batch 对 two_crop 恒按 2B 探测的同类设计。
- 与 `density`（分册 13）正交组合：两者都开时行为见 §4.9。
- v1 边界：`ssl` + `decoder="rtdetrv2"` 组合在 `train_model.__init__` 显式 `ValueError`（§9 开放问题 1）。

---

## 4. 详细设计

### 4.1 数学组件移植（`object_detection_components/ssl_contrastive.py`）

移植源【RF-DETR-SSL】`src/rfdetr/models/ssl_head.py`（444 行，自包含）。五个组件逐一对应：

| 组件 | 上游位置 | 移植要点 |
|---|---|---|
| `SSLProjectionHead` | `ssl_head.py:36-91` | 原样。`"simcse"`=`Linear(hidden→dim)+Tanh`，`"mlp2"`=`Linear→ReLU→Dropout→Linear`；独立 `input_dropout` 是 dropout 档的视图噪声源；`forward` 返回 `F.normalize(z.float(), dim=-1)`——**fp32 归一化是 AMP 稳定性硬约束**，不可改 |
| `gather_distributed` | `ssl_head.py:108-136` | 原样。`all_gather` 后本 rank 槽位替换回原张量——负例池全局、梯度只走本地行（SimCSE 语义） |
| `gather_distributed_padded` | `ssl_head.py:139-180` | 原样。先 gather 各 rank size，pad 到全局最大再 gather + 槽位替换；返回 `PaddedGather(tensor, valid)`，**调用方必须用 `valid` 排除 pad 行**（pad 嵌入为零向量、pad 标签为 0） |
| `simcse_infonce_loss` / `supervised_contrastive_loss` | `ssl_head.py:201-252, 255-336` | 原样。含 `logits=` 显式传入接口（偏置不可被就地修改静默丢弃）与 `valid_mask` 语义：列方向 `masked_fill(-inf)`、行方向重索引 anchors；sup 的 pad 列必须 `masked_fill` 排除（`-inf × 0 = NaN` 陷阱，源码 `:332-333` 有注释） |
| `positive_logit_bias` | `ssl_head.py:339-376` | 原样。`weight = where(positive_mask, iou.clamp(min=1e-12)^γ, 1)`，`logits + bias·log(weight)`——**非正对位权重恒为 1**（A1 修复语义：若非正对位 ŵ=0，`log 0=-inf` 会把全部负例打失效） |
| `match_cross_view` | `ssl_head.py:394-444` | 原样。逐图两视图实例池的 box-IoU 匈牙利（`linear_sum_assignment(maximize=True)`）+ `IoU ≥ iou_match` **与**标签一致的双重校验；失败对直接丢弃，不造噪声正对 |

**编排函数**（新增，取代上游 criterion 方法形态）：

```python
@dataclass(frozen=True)
class SSLParams:
    """Shared-component SSL contract, constructed from LTDETRSSLArgs by the
    train wrapper (keeps the components layer free of ltdetr config imports)."""
    mode: str                 # "unsup" | "sup" | "both"
    temp: float = 0.05
    view: str = "two_crop"    # "two_crop" | "dropout"
    iou_match: float = 0.55
    iou_bias: float = 0.5
    iou_gamma: float = 1.0
    same_image_positive_only: bool = False
    gather_distributed: bool = True
    every_n_steps: int = 1

def ssl_contrastive_loss(
    z1: Tensor, z2: Tensor,                 # (2B|B, Q, D) 投影输出
    outputs_final: dict[str, Tensor],       # 最终层输出（取 pred_logits 推 batch 边界）
    targets: list[dict[str, Tensor]],
    indices: list[tuple[Tensor, Tensor]],   # 最终层匈牙利匹配
    params: SSLParams,
    query_mask: Tensor | None = None,       # (B, Q) bool，DOME padding 过滤
    ssl_step: int = 0,
) -> Tensor
```

内部流程与【RF-DETR-SSL】`criterion.py:684-925` 的 `loss_ssl` + `_pair_two_crop` 等价，适配点见 §4.4。

### 4.2 decoder 特征暴露（`dfine_decoder.py`）

【VTrain】`DFINEv2Decoder._refine_layers`（`dfine_decoder.py:133-223`）的精修循环中，最终层隐状态 `output` 就是 `dec_score_head[-1]`（分类头）的输入，但目前循环结束后被丢弃。改动：

1. `_refine_layers` 返回值从 8 元组扩为 9 元组，追加 `matched_hidden`：最终层 `output`，按与 `corners_list`/`refs_list` 相同的方式做 DN 分离（【VTrain】`:197-208` 的 split 模式：`output` 形状 `(B, Q_total, C)`，`dn_num = dn_meta["dn_num_split"][0]`，主段 `output[:, dn_num:]`）；
2. `DFINEv2Decoder.forward` 训练分支发射：

```python
if self.training:
    ...
    out["query_features"] = matched_hidden   # (B, Q_matched, C)，分类头输入表示
```

- **语义对齐**：这就是【RF-DETR-SSL】设计文档四约束第 1 条"对比项作用在 `class_embed` 的输入特征上"的 VTrain 对应物（`dec_score_head[-1]` 的输入）；
- **梯度通路**：`output` 未 detach（FDR 循环只 detach 参考框，`dfine_decoder.py:165,192`），SSL 梯度经 self-attn/cross-attn 流入 decoder 各层 → encoder memory → backbone，与分类头共享表示；
- **DOME 兼容**：`DomeDFINEv2Decoder` 复用基类 `_refine_layers`（分册 13 §4.3），自动获得 `query_features`；DOME padding 查询包含其中，由 §4.4 的 `query_mask` 过滤；
- **RT-DETRv2 decoder**：`RTDETRv2Decoder` 有自己的精修实现，v1 不动（§9 开放问题 1）；
- 显存：`output` 张量本就在前向图中，暴露只是新增引用，**零额外显存**；`query_features` 不 emit 到 eval（`self.training` 门控）。

### 4.3 任务模型 SSL 头挂载与发射（`task_model.py`）

【设计】头由**任务模型**持有（对齐【RF-DETR-SSL】`lwdetr.py:873-895` 的归属裁定：checkpoint 只存模型 state_dict、DDP/EMA 自动覆盖、criterion 无参数）：

```python
# __init__（density 分支同款条件构建模式，分册 13 §4.6）：
self.ssl_head = (
    SSLProjectionHead(hidden_dim=spec.hidden_dim,
                      projection_dim=ssl.projector_dim,
                      kind=ssl.projector,
                      dropout=ssl.projector_dropout)
    if ssl is not None and ssl.mode != "off" else None
)

# _forward_train 末尾（两个 encoder/decoder 分支汇合后）：
if self.ssl_head is not None and self.training and "query_features" in out:
    out["ssl_z1"] = self.ssl_head(out["query_features"])
    out["ssl_z2"] = self.ssl_head(out["query_features"])   # 两次独立 dropout 采样
```

- `ssl=None` 时不构建头（不建空模块，避免 DDP 未用参数告警，对齐【RF-DETR-SSL】`build_model` 行为）；
- `mode="off"` + `view="two_crop"`（批匹配基线）时**不构建头、不发射** ssl 键——数据管线翻倍照常；
- `ssl_z1/z2` 各是**一次完整头前向**（两次调用 = 两组独立 dropout 掩码），criterion 按 `view` 从中切行：two_crop 取 `z1` 的视图 1 行 + `z2` 的视图 2 行，dropout 取 `z1/z2` 的同一行——与【RF-DETR-SSL】`lwdetr.py:607-611` 语义一致。

### 4.4 criterion 编排（`dfine_criterion.py` + `ssl_contrastive_loss`）

**criterion 改动**（【VTrain】`dfine_criterion.py`）：

```python
def __init__(self, ..., ssl_params: SSLParams | None = None):
    ...
    self.ssl_params = ssl_params
    if ssl_params is not None:
        self.weight_dict["loss_ssl"] = ssl_params 的 loss_coef   # 由构造方传入或挂 SSLParams
```

`forward` 签名追加 `ssl_z1: Tensor | None = None, ssl_z2: Tensor | None = None, ssl_step: int = 0`。在主层循环与 DN/DDF/DeFE 之后、权重聚合之前追加（**DeFE 同款位置与模式**，【VTrain】`:186-198, 200-210`）：

```python
ssl_enabled = (
    self.ssl_params is not None and self.training
)
if ssl_enabled:
    if not (isinstance(ssl_z1, torch.Tensor) and isinstance(ssl_z2, torch.Tensor)):
        # Head/criterion mismatch must fail loud, not silently train a
        # whole run without the SSL objective (upstream parity).
        raise ValueError(...报错指向修复方法...)
    stride_hit = (
        self.ssl_params.every_n_steps <= 1
        or ssl_step % self.ssl_params.every_n_steps == 0
    )
    if stride_hit:
        losses["loss_ssl"] = ssl_contrastive_loss(...)
        self.last_ssl_stats = {...}
    else:
        self.last_ssl_stats = None   # 不泄露上一步统计
```

eval 态（`self.training=False`）双保险静默：验证分支永不 raise、永不进 gather。

- **零新增匹配**：直接用 `layer_indices[-1]`（【VTrain】`:111-114` 每层已各存匹配）；
- **垃圾键安全**：`weights` 复制循环会给 `loss_ssl` 生成 `loss_ssl_aux_i`/`loss_ssl_dn_i` 权重条目，但 losses dict 无对应键，`if key in losses`（【VTrain】`:209`）自动过滤——与 DeFE 机制相同（D7）；
- **eval 安全**：验证路径无 ssl 键且 `self.training=False`，双重门控，eval 分支永不进入 gather collective；
- **步门控**：`ssl_step` 由 train wrapper 每步写入 criterion 属性（对齐【RF-DETR-SSL】`module_model.py:546-549` 把状态放 criterion 之外的做法）。

**`ssl_contrastive_loss` 内部**（对照【RF-DETR-SSL】`criterion.py:684-925`，标注差异）：

1. **池构造**：`batch_sel, src_sel = _get_src_permutation_idx(indices)` 的等价展开（batch 行 + query 列）；**设备对齐**——本仓匹配器同样在 CPU numpy 上跑指派（【VTrain】`matcher.py:75`），索引需 `.to(z1.device)`（【RF-DETR-SSL】`criterion.py:735-737` 同款）；labels 从 `targets[i]["labels"][tgt_idx]` 提取，boxes `box_cxcywh_to_xyxy` 后提取；
2. **query_valid_mask 过滤（D1，取代上游 group-0 过滤）**：`keep &= query_mask[batch_sel, src_sel]`——DOME padding 查询的 logits 已被源头置 −1e4（分册 13 D5），Hungarian 实践中不会选中，此过滤是防御性的；固定 Q 基线无 mask（None）则跳过；
3. **DDP 进入/跳过共识**：本地池 <2 不能直接返回，必须 `all_reduce(MAX)` 全局判断（【RF-DETR-SSL】`criterion.py:752-765`）——各 rank 匹配数不同，纯本地判断必然死锁；全局无池返回 `z1.sum() * 0.0`（**connected zero**，数值 0 但连图，DDP 梯度 all-reduce 不破坏）；
4. **two_crop 配对**：`B = outputs_final["pred_logits"].shape[0] // 2`（**从 batch 维取，不用 rows.max()**，上游 A2 修复）；逐图 `match_cross_view` 恢复正对（本 rank 池上配对后再 gather——垃圾对本地丢弃缩小 padded gather；IoU 偏置仍在 gather 后的全局池上计算，与上游 §5.5 规则一致）；本 rank 无有效对时，分布式下用空张量加入后续全部 collective，单进程返回 connected zero（【RF-DETR-SSL】`criterion.py:779-788`）；
5. **同图 image_id（sup 用）**：`image_id = row % B`（two_crop 行算术，D5 无需 image_id 键）；DDP 下 rank r 偏移 `r·B` 保证全局唯一（【RF-DETR-SSL】`criterion.py:797-810` 同款，否则跨 rank 凭空捏造同图正对——上游 P0 回归）；
6. **gather**：z1/z2/labels/boxes/image_ids 全部 `gather_distributed_padded` 按**同一 rank 顺序**；`valid_mask` 取自 z1 的 gather 结果；
7. **IoU 偏置**（two_crop only）：`positive_mask = eye(配对正对) ⊕ 同标签(sup)`，`iou = box_iou(boxes1, boxes2)` 全池，偏置后的 logits 经 `logits=` 显式传入损失；
8. **分发**：`unsup` → InfoNCE；`sup/both` → SupCon（`both` 相加）；全程 fp32（头输出已 fp32，`cosine_logits` 内再 `float()` 归一化）——满足仓内 fp32-island 惯例（D6）。

### 4.5 two_crop 数据管线

**数据集包装**（`_data/two_view.py`，对齐【RF-DETR-SSL】`two_view.py` 语义）：

```python
class TwoViewObjectDetectionDataset(Dataset):
    """item = (view1_item, view2_item)，各为基数据集的完整 transform 输出。"""
    def __init__(self, dataset): self.dataset = dataset
    def __len__(self): return len(self.dataset)          # 长度不变 → 步数/schedule 不变
    def __getitem__(self, i): return self.dataset[i], self.dataset[i]
    def set_transform(self, t): self.dataset.set_transform(t)   # 委托（命令在包装后注入 transform）
```

- 两次访问触发两次独立 transform 采样（【VTrain】transform 全部为逐调用随机 v2 算子，`transform.py:127-257`）→ 两个视图真实不同；框随几何变换同步是现有管线保证的；
- `len` 不变 → `resolve_step_schedule` 的 `train_num_batches`（【VTrain】`object_detection.py:239`）不变，步窗口契约不受影响。

**collate 包装**（`_transforms/detection_transforms/collate.py` 新增类）：

```python
class TwoViewDetectionCollateFunction:
    def __init__(self, base: RTDETRDetectionCollateFunction): self.base = base
    def __call__(self, items: list[tuple[dict, dict]]) -> dict:
        flat = [item for pair in items for item in pair]        # [v1_0..v1_{B-1}, v2_0..v2_{B-1}]
        return self.base(flat)                                  # 单次联合调用
```

- **单次联合调用是关键决策（D4）**：批内单一 scale-jitter draw、单一 `input_size`，2B 行直接进一个 batch——避免双视图各自 scale 后需要联合 padding（padding 会破坏归一化 cxcywh 坐标语义，框需重算，复杂且无必要）。视图多样性由逐样本算子（photometric/zoomout/IoU-crop/flip/mosaic）保证，与【RF-DETR-SSL】"检测损失照常跑 2B"等价；
- **mixup/mosaic 保持开启（D5）**：mixup 把邻行两图拼接进同一行的 targets，IoU+标签配对天然容忍（同源物体的框仍在两视图 targets 中，其余被配对校验丢弃）；mosaic 的四图拼贴同理。代价是配对产率下降，用 `ssl_paired_ratio` 诊断（§4.6），消融矩阵含"关 mixup/mosaic 的 SSL 组"（§8）验证影响；
- 输出 batch 键与单视图完全一致（`image/bboxes/classes/image_path/original_size/input_size`）→ `training_step`、可视化钩子（`engine.py:638-651`）、`_to_device` 全部零改动。

**命令接线**（【VTrain】`object_detection.py:205-249`，`is_ltdetr` 且 ssl two_crop 时）：

```python
train_dataset = ObjectDetectionDataset(data_args, train_rows)
if ssl_two_crop:
    train_dataset = TwoViewObjectDetectionDataset(train_dataset)
train_loader = build_task_dataloader(train_dataset, ...)       # 包装在前，长度不变
...
train_dataset.set_transform(train_transform)                    # 委托到内层
train_collate = RTDETRDetectionCollateFunction(...)
if ssl_two_crop:
    train_collate = TwoViewDetectionCollateFunction(train_collate)
```

val loader / val collate 永不包装（验证单视图，回归防护进 §7）。

**B4 前置测量**（`scripts/ssl_cooccurrence_stats.py`）：自【RF-DETR-SSL】`datasets/ssl_stats.py:21-77` 原样移植 `same_class_cooccurrence_stats`（读 COCO json、跳过 iscrowd、返回 5 个同类共存指标）。用法：

```bash
python scripts/ssl_cooccurrence_stats.py --annotations data/train/_annotations.coco.json
# 同类伙伴占比 > 0.5 的拥挤数据集：优先消融 sup + same_image_positive_only=True
```

### 4.6 训练-验证接线（`train_model.py`）

- **training_step**（【VTrain】`:125-167`）：提取 `ssl_z1 = out.get("ssl_z1")`、`ssl_z2 = out.get("ssl_z2")` 传入 criterion；`self.criterion.ssl_step = step` 每步写入；`log_dict` 自动带上 `loss_ssl`（【VTrain】`:166-167` 全 loss 键入 log_dict → 引擎 `:709-712` 浮动并交 logger）；
- **配对产率诊断**（D10，补齐上游缺失的 `ssl/dropped_pairs`）：`ssl_contrastive_loss` 在 criterion 上落一个 `self.last_ssl_stats = {"pool_size": n, "paired_size": k}`（纯 Python 数值，无梯度，gather 前的本地计数），training_step 塞进 log_dict：`ssl_pool_size`、`ssl_paired_ratio`。**定义**：`paired_ratio = k / max(pool/2, 1)`（two_crop 一对消耗两个池位，1.0 = 完美配对；dropout 档分母 = pool，恒 1.0）。步门控未命中的步重置为 None，不泄露上一步数值；
- **参数组**（【VTrain】`:267-357` 的 detector 组之后追加）：`ssl_head` 不在 `encoder/decoder` 之下，需显式组——按既有 no-wd 集合（`get_weight_decay_parameters([self.model])`，`_optim/optimizer_helpers.py:87`）把 `ssl_head.parameters()` 划分为 wd/no-wd 两组，lr = 基础 `lr`（与 detector 同档，对齐【RF-DETR-SSL】`param_groups.py:72-77` 把 ssl_head 归入 base lr 的行为）；注意参数不得重复归组；
- **EMA**：`ModelEMA` 整模型 deepcopy 自动覆盖 `ssl_head`（EMA 评测时头惰性，无行为影响）——与【RF-DETR-SSL】一致；
- **导出卫生（D9）**：`get_export_state_dict` 过滤 `ssl_head.` 前缀键——上游靠 checkpoint 加载白名单容忍缺失，本仓从源头不导出更干净；`deploy()`/`torch.export` 不受影响（`forward_backend` 从不触碰 ssl_head）；
- **validation_step**：零改动（无 ssl 键 → criterion 门控跳过；val loss 组成与基线逐比特一致）。

### 4.7 配置面（`config.py`）

```python
class LTDETRSSLArgs(PydanticConfig):
    """逐物体对比 SSL 辅助分支（RF-DETR-SSL 移植）。None = 关闭。"""
    model_config = {"extra": "forbid"}

    mode: Literal["off", "unsup", "sup", "both"] = "unsup"
    loss_coef: float = Field(default=0.1, gt=0.0)      # λ；上游 M4：0.1–0.25 稳健，1.0 有毒
    temp: float = Field(default=0.05, gt=0.0)
    view: Literal["two_crop", "dropout"] = "two_crop"  # D2：two_crop 直接落为默认
    projector: Literal["simcse", "mlp2"] = "simcse"
    projector_dim: int = Field(default=256, ge=1)
    projector_dropout: float = Field(default=0.1, ge=0.0, lt=1.0)
    iou_match: float = Field(default=0.55, gt=0.0, le=1.0)
    iou_bias: float = Field(default=0.5, ge=0.0)
    iou_gamma: float = Field(default=1.0, gt=0.0)
    same_image_positive_only: bool = False             # sup 档自由度；dropout 档下校验拒绝
    gather_distributed: bool = True
    every_n_steps: int = Field(default=1, ge=1)        # 仅控梯度频率，不省 two_crop 管线算力
    instance_source: Literal["matched_gt", "encoder_candidates"] = "matched_gt"  # v2（D8）
```

`LTDETRObjectDetectionTrainArgs` 增加 `ssl: LTDETRSSLArgs | None = None`（pydantic 嵌套校验自动生效，`extra=forbid` 已有）。

**校验器**（对齐【RF-DETR-SSL】`config.py:1153-1169`）：

- `mode != "off"` ⇒ `loss_coef > 0`；`mode == "off"` 允许任意系数（批匹配基线组合 `mode="off"` + `view="two_crop"` 不构建头、不算损失，但数据管线翻倍）；
- `same_image_positive_only=True` 要求 `view="two_crop"`（dropout 档全部实例同图，该开关无意义）；
- `instance_source="encoder_candidates"` → v1 `NotImplementedError`（§4.10）；
- `decoder="rtdetrv2"` + `ssl is not None` → v1 `ValueError`（train_model 构建期拒绝，§9 开放问题 1）。

**用法**：

```python
model_args = {
    ...,
    "ssl": {"mode": "unsup", "view": "two_crop", "loss_coef": 0.1},  # 其余字段走默认
}
# 批匹配基线：{"mode": "off", "view": "two_crop"}
```

### 4.8 DDP / EMA / AMP / 引擎契约

| 项 | 结论 |
|---|---|
| DDP | gather 共识（`all_reduce(MAX)`）+ padded gather + 槽位替换全套移植；`find_unused_parameters=True` 默认（【VTrain】`engine.py:506`）兜底 `every_n_steps>1` 跳步与任何空池边角；梯度累积 `no_sync()`（`engine.py:671-676`）不受影响——`all_gather` 是显式 collective，各 rank 对称执行 |
| 梯度累积 | **loader 批 = 微批**（【VTrain】`engine.py:663-668` 连续拉取），two-view 配对在单 loader 批内永不拆散——上游 D4（GradAccumAlignedDataset 内层约束）在本仓不存在 |
| AMP | bf16 autocast 包住 `training_step`（【VTrain】`engine.py:678-680`）；SSL 头输出强制 fp32 + 损失全程 fp32（fp32 island 惯例，D6）；InfoNCE 的 `-inf` 掩码列 + fp32 余弦保证无新 NaN 源 |
| EMA | 整模型 deepcopy 自动覆盖 ssl_head；EMA eval 时头惰性 |
| NaN 监控 | 引擎 `nan_monitor` 照常覆盖（loss_ssl 进 total） |
| 检查点 | `ssl_head.*` 随 state_dict 自然进出；`ssl` 开关改变架构 → resume 须同配置（`extra=forbid` + init-args 快照机制已保证）；旧 checkpoint 加载：`load_train_state_dict` 的前缀剥离逻辑容忍多余键，`ssl_head.` 缺失键走既有 partial-load 路径（无警告要求进 §7 测试） |
| 引擎日志 | `loss_ssl`/`ssl_pool_size`/`ssl_paired_ratio` 经 `log_dict` → `engine.py:709-712` → logger 链路，零引擎改动 |

### 4.9 与 Dome 密度分支的组合语义（分册 13）

| 交互点 | 语义 |
|---|---|
| CDN 去噪段 | DN 查询在 `_refine_layers` 里已与主段分离（【VTrain】`dfine_decoder.py:197-208`），`layer_indices[-1]` 只覆盖主段 → **去噪查询永不进入 SSL 池**，无泄漏 |
| DOME 动态查询 padding | padding 查询 logits 源头 −1e4（分册 13 D5）→ Hungarian 实践中不选中；`query_mask` 防御过滤双保险；极小 batch 整批无有效对 → connected zero 兜底 |
| DeFE 密度损失 | 2B 批下逐图照算（每行独立监督），密度监督信号翻倍——消融 c 组（§8）验证组合不侵蚀 |
| MWAS/PAQI | 逐图操作，2B 透明；PAQI 的可变查询数只影响每行 Q padding，池构造按行索引天然兼容 |
| `query_features` | Dome decoder 继承基类 `_refine_layers` 自动获得，无需改动 dome_decoder.py |

### 4.10 Route B：`encoder_candidates` 实例源（SSL-only 预训练，已实现）

【RF-DETR-SSL】Route B 用 two-stage encoder proposals 做无标注实例（`ssl_instance_source="encoder_proposals"`，`criterion.py:722-731`；恒 unsup、dropout 视图）。【VTrain】对应物（**已实现**，与上游的差异见 D8）：

- **实例 = decoder top-k 候选特征**：`RTDETRv2Decoder._get_decoder_input` 在 `_select_topk` 之后以惰性属性 stash **detach 前**的 `enc_topk_memory`（`getattr(self, "_emit_candidate_features", False)` 门控，RT-DETR 家族零影响）；`DFINEv2Decoder` 以 `ssl_candidate_features=True` 构建时发射 `out["ssl_candidate_features"]`（train + eval）。梯度经 `enc_output` 流回 encoder memory（绕开 content 已 detach 的限制，D8）——注意 `enc_score_head` 只产生 topk **索引**（不可导），本身无梯度，符合实例判别语义；
- **SSL-only 语义**：`instance_source="encoder_candidates"` ⇒ criterion **跳过匈牙利匹配与全部检测损失**（VFL/box/GIoU/FGL/DDF/DN/DeFE），对比损失即全部损失，且 **train 与 eval 同算**——验证损失 = 留出集对比损失，即阶段一的选模型信号；stride 门控不适用；
- 无标签 → 校验器强制 `mode="unsup"` + `view="dropout"`（跨独立裁剪的候选框不可比，【RF-DETR-SSL】`lwdetr.py:885-889` 同结论）+ `every_n_steps=1`（stride 门控在此无意义，显式拒绝静默空转）；与 density 分支互斥（密度门控的 padding 候选不可作实例）；
- **两阶段工作流**：阶段一 `ssl={"mode":"unsup","view":"dropout","loss_coef":1.0,"instance_source":"encoder_candidates"}` 跑预训练（`watch_metric` 自动切为 `val_ssl_loss`——引擎对缺失 watch key 的 `-val_loss` 兜底即"越低越好"，世界平均，零额外管道）；阶段二直接 `checkpoint=<stage-1 ckpt>` 启动有监督微调——训练包装的 `load_state_dict` 会**丢弃** stage-1 的 `ssl_head.*` 键（预训练专用头，stage-2 无此配置），其余权重严格加载，检测头从随机初始化开始训练（阶段一它们本就无梯度，等价于重初始化）；
- 使用注意：阶段一验证曲线中的 mAP 是噪声（检测头随机初始化），以 `val_ssl_loss` 为准；检测损失前向计算仍执行（代码路径统一、阶段二衔接零成本），推理/导出路径不受影响（`forward_backend` 不触碰 SSL 通道）。

---

## 5. 与 RF-DETR-SSL 的差异清单

| # | 差异 | 上游现状 | VTrain 决策 | 理由 |
|---|---|---|---|---|
| D1 | group-0 过滤 → `query_valid_mask` 过滤 | `group_detr=13` 组查询，池按第 0 组过滤防近重复正对 | 本仓查询无分组；池过滤改为匹配索引 ∩ `query_valid_mask`（DOME padding 防御） | 架构差异；过滤语义等价（都排除"非真实检测查询"） |
| D2 | `view` 默认 two_crop | 文档推荐 two_crop（B1）但代码默认 `dropout` | 默认 `two_crop`，`dropout` 保留为对照档 | 落地上游文档意图，避免"文档-代码默认值漂移"复刻 |
| D3 | loss_ssl 挂 criterion 的 DeFE 模式 | aux 复制循环**之后**追加 weight_dict，杜绝垃圾键 | weight_dict 正常追加，垃圾键由聚合处 `if key in losses` 过滤（DeFE 同款） | 仓内既有机制已安全，少一处特殊逻辑 |
| D4 | two-view 数据 = 数据集二元组 + collate 单次联合调用 | 专用双视图 collate（联合 padding）+ GradAccumAlignedDataset 内层约束 | 复用现有 collate 全部批级行为；引擎 loader 批=微批，无对齐问题 | 单一 scale 避免归一化框重算；少一个专用 collate 维护面 |
| D5 | mixup/mosaic 保持开启；配对行算术替代 image_id | 无 mixup；image_id 显式传递 | IoU+标签配对天然容忍混合增广；同图判定用 `row % B`，不引入新批键 | 管线复用最大化；配对产率用 `ssl_paired_ratio` 监控（D10） |
| D6 | 损失 fp32 island 显式化 | 头内 fp32 归一化 | 同上游 + 仓内 fp32-island 惯例对齐（分册 13 D6 同款理由） | bf16 引擎契约 |
| D7 | criterion 集成形态 | `SetCriterion.loss_ssl` 方法 + `outputs` 携带 ssl_z1/z2 | 独立编排函数 `ssl_contrastive_loss`（组件层），criterion 一行调用 | 组件层不得依赖 ltdetr 任务层；RT-DETRv2/OBB 后续可复用 |
| D8 | Route B 实例源 = decoder 候选处 memory token | two-stage encoder proposals（`hs_enc` 直接可投影） | 重命名 `encoder_candidates`；梯度经 memory token gather 绕开 content detach | 架构差异（无 two-stage 分支）；v1 不实现 |
| D9 | 导出 state dict **保留** `ssl_head.*` 键 | checkpoint 加载白名单（`weights.py:166-169`）容忍缺失 | 原计划过滤；实施时确认 `load_train_state_dict` → `load_state_dict_compat` 对缺失键是严格报错（仅 `mask_token` 豁免），而导出 init-args 含 `ssl` 会重建带头的推理模型——**保留键才能 round-trip 加载**；导出图本就不含头（`forward_backend` 不触碰） |
| D10 | `ssl_pool_size`/`ssl_paired_ratio` 诊断日志 | `ssl/dropped_pairs` 计数为设计-only，未实现 | criterion 落 `last_ssl_stats`，training_step 入 log_dict | 补齐上游缺口；two_crop 配对产率是 §8 消融的关键读数 |
| D11 | 无 λ 预热 / 梯度比诊断 | 均为上游设计-only（§6.5 D1、`ssl_warmup_epochs` 未实现） | 不移植；λ 起点直接用上游 M4 实测推荐 0.1 | 上游未实现且 M4 已给出实测稳健区间，预热价值待证 |
| D12 | 步数制门控 | epoch 制 + `ssl_step=batch_idx` | `ssl_step=step`（引擎步号），语义相同 | 引擎契约 |
| D13 | **翻转感知跨视图配对**（默认开） | 配对只用恒等对齐的归一化 IoU | 候选对在恒等**与水平镜像**两种对齐下取最大 IoU 再做匈牙利 + 双重校验；配对自身的对齐 IoU 传递给正对 logit 偏置（替换对角线上的原始 IoU） | 两视图独立翻转（各 p=0.5）：偏心物体镜像后归一化坐标几乎不重叠 → 恒等 IoU≈0，恒等配对的理论产率被压到 ~50%；`flip_invariant_pairing=false` 可退回上游严格语义。命令层冒烟实证：tiny COCO 上产率 0.17 → 显著改善 |
| D14 | **未配对实例作额外候选列**（默认开） | 配对校验失败的实例两视图都整条丢弃 | 损失推广到矩形 logits（N 锚 × N+M 候选，前 N 列为已配对正对）：未配对实例按自身视图的嵌入加入候选列——unsup 下恒为负例，sup 下同标签 extras 自动升级为软正对；padded gather/共识/DDP 语义不变；`use_unpaired_as_negatives=false` 退回上游严格语义 | 回收难例（R9），对比学习更多负例通常有益；sup 模式额外获得跨实例类别结构 |

---

## 6. 实施计划（三个 PR，每个独立可合入）

> **实施记录（v1.1）**：三个阶段已合并实现（单分支交付）。验收：`tests/test_ltdetr_ssl.py` 57 例全绿（含 2-rank gloo 池不均衡共识/padded gather 测试、EMA 往返、checkpoint 容错加载）；受影响回归全绿（test_ltdetr_dome 43 + test_ltdetr_object_detection 42 + dfine/组件/checkpoint 90）。独立代码审查（对照上游实现 + 2-rank gloo 行为探针）结论：无 P0；已修复——P1① `ssl_paired_ratio` 分母改为 pool/2（原定义上限 0.5，无法达到 §6 验收阈值）；P1② criterion 训练态 ssl_params 置但特征缺失改为**响亮 raise**（对齐上游与 §7 测试计划，消除静默整跑无 SSL 的浪费）；P2③ D9 改为保留导出键（严格加载器要求）；P2④ rtdetrv2 边界收紧为 `ssl is not None`（含 mode="off"）；P2⑤ 步门控 miss 步重置统计；P2⑥ 组件层防御校验 `same_image_positive_only`×dropout；P2⑦ IoU NaN→0 守卫（零面积框）；P2⑧ 逐比特断言改 `torch.equal`；P2⑨ 补 EMA/gloo 测试。
> **实施记录（v1.3，遗留项清偿）**：①**D14 未配对负例**——两个对比损失推广到矩形 logits（valid_rows/valid_cols 拆分），`_pair_two_crop` 返回未配对实例，extras 按自身视图嵌入加入候选列（sup 下同标签 extras 自动升级软正对），config `use_unpaired_as_negatives`（默认开），日志新增 `ssl_candidate_size`；②**探针移植**——`_metrics/ssl_probes.py` 5 项纯函数 + 6 例 hermetic 测试；③**R8 了结**——段式分组 + CPU 化，块对角 batched 指派经渐近分析否决（O((Σn)³) vs ΣO(n³)）。测试 70+6+3 例全绿，全量回归 260 例，ruff 干净。
> **实施记录（v1.2，Route B 追加）**：`instance_source="encoder_candidates"` SSL-only 预训练已实现——decoder 候选特征 stash（`rtdetrv2_decoder.py` 惰性属性，基线零影响）、`DFINEv2Decoder.ssl_candidate_features` 发射开关（train+eval）、criterion SSL-only 分支（跳过匹配与全部检测损失）、`watch_metric` 动态切换（经引擎 `-val_loss` 兜底）、stage-2 加载容忍（丢弃 stage-1 的 `ssl_head.*` 键）。测试 64 例全绿（新增 Route B 组：组件池/拒绝矩阵/损失集纯度/编码器-检测头梯度隔离/验证路径/watch 切换/stage-2 加载/密度互斥）；回归 243 例全绿；ruff 干净。

### PR-1：数学组件 + dropout 视图全链路（零数据改动）

- 新增 `ssl_contrastive.py`（五组件 + `SSLParams` + `ssl_contrastive_loss`）；
- `dfine_decoder.py` 暴露 `query_features`；`dfine_criterion.py` 接线；
- `task_model.py` 挂头发射；`train_model.py` 步门控 + 参数组 + 导出过滤；`config.py` 配置面与校验；
- 测试：§7 全部非 two_crop 用例。
- **验收**：`ssl=None` 基线逐比特回归；VisDrone smoke（2k 步）`dropout` 档 `train/loss_ssl` 收敛、mAP 不崩。

### PR-2：two_crop 数据管线 + 批匹配基线

- `_data/two_view.py`、`TwoViewDetectionCollateFunction`、命令接线；
- `ssl_pool_size`/`ssl_paired_ratio` 日志；`scripts/ssl_cooccurrence_stats.py`（B4）；
- 测试：two-view 数据集/顺序/微批配对不变式、批匹配基线路径（`mode="off"`）。
- **验收**：two-view 批内 row r ↔ r+B 同 `image_path` 断言；VisDrone smoke 记录 `ssl_paired_ratio`（预期 >0.6，若显著偏低先查 mixup/mosaic 交互再进 PR-3）。

### PR-3：VisDrone 实验 + 消融 + 文档收尾

- `scripts/visdrone_ltdetr_ssl_800.py`；§8 消融矩阵执行与结果回填本册；
- 与 DOME 组合验证（§4.9）；（可选）`ssl_probes` 五探针移植与接入评估；
- 本册状态更新为"已实施"。

---

## 7. 测试计划（`tests/test_ltdetr_ssl.py`）

风格对齐 `tests/test_ltdetr_dome.py`（离线 `-notpretrained` 变体 + 小图，全部 hermetic）：

| 用例 | 断言 |
|---|---|
| `test_ssl_off_is_baseline` | `ssl=None`：无 `ssl_head` 模块、state_dict 键集合与基线一致、前向输出逐比特一致 |
| `test_query_features_training_only` | 训练前向含 `query_features`（形状 (B,Q,C)、`requires_grad`）；eval 前向与基线输出键集合一致 |
| `test_projection_head_contract` | 输出 (N,dim)/float32/单位范数；train 两次前向不同（dropout 噪声）、eval 确定；bf16 autocast 下输出仍 float32 |
| `test_gather_distributed_passthrough` | 无进程组原样返回且 `grad_fn` 保留；1-rank gloo 组反向后梯度全 1（槽位替换语义） |
| `test_padded_gather_valid_mask` | 不同 rank 池大小手工对账；pad 行零嵌入 + valid 全真掩码 |
| `test_infonce_hand_computed` | 与手算 `CE(normalize(z1)@normalize(z2).T/τ, arange(N))` 一致；N=1 raise；`logits=` 覆盖可达损失（偏置集成点） |
| `test_supcon_multi_positive` | 全异标签 == InfoNCE；多正样本 == 手算逐锚平均；`same_image_positive_only` 缺 image_id raise |
| `test_positive_logit_bias_a1` | 未标记位不动；标记位随 IoU 单调增；IoU=0 clamp 有限；非正对位绝不出现 -inf 偏置 |
| `test_match_cross_view` | 恒等框→恒等匹配 IoU=1；次优序下指派最优；低于阈值/标签不符的对被丢弃 |
| `test_ssl_pool_construction` | 池 = 最终层匹配 ∩ `query_valid_mask`；单实例返回**带梯度**的 0；缺 ssl_z1 raise 且报错指向修复方法 |
| `test_loss_ssl_final_layer_only` | losses 无 `loss_ssl_aux_i`/`loss_ssl_dn_i` 键；总损失含 `loss_ssl × weight_dict["loss_ssl"]` |
| `test_ssl_step_stride` | `every_n_steps=2` 时 step 0/2 发射、step 1 不发射 |
| `test_eval_never_gathers` | criterion eval 模式 + eval 输出：不进对比路径、无 collectives |
| `test_two_view_dataset_two_draws` | 注入确定性 transform 两次访问：图像不同、框随几何一致同步 |
| `test_two_view_collate_order` | B=2 → 4 行、`bboxes`/`classes` 顺序 `[0,1,0,1]`、批键结构与单视图一致、`input_size` 单一 |
| `test_two_view_pairing_survives_accumulation` | 梯度累积窗口内 loader 批恒含完整配对（row r 与 r+B 同 `image_path`） |
| `test_two_crop_pairing_and_loss` | 构造已知跨视图框：配对正确、损失有限可反传、全被 IoU 剔除时 connected zero |
| `test_config_validation` | `mode≠off`+`loss_coef≤0` raise；`same_image_positive_only`+`dropout` raise；`encoder_candidates` v1 raise；`rtdetrv2`+ssl raise |
| `test_checkpoint_and_ema_roundtrip` | ssl 开启训练 2 步 → 存/载 → loss 连续；EMA state 含 `ssl_head.*` |
| `test_ddp_unbalanced_pools`（可选，gloo spawn） | 2-rank 池 3 vs 1：共识不死锁、双 rank 损失有限、梯度非空 |

金标回归：`tests/golden/` 若覆盖 LT-DETR 前向，确认 ssl 关闭路径零差异。

---

## 8. VisDrone 训练配方与消融

**基座**：`scripts/visdrone_ltdetr_ssl_800.py`（复制 `visdrone_ltdetr_800.py`；29k 步 flat-cosine、`eval_max_dets=500` 全部保持）。

**batch_size 语义**：two_crop 下 loader 批计"图像对"（每步 2B 张图）。与既有单视图基线比图像吞吐需减半 batch_size；与 SSL 运行比算力则用批匹配基线（同 two_crop 管线、`mode="off"`）——**消融以批匹配基线为参照系**（对齐上游 C2 公平性硬要求）。

```python
model_args = {
    "backbone_weights": BACKBONE_WEIGHTS,
    "image_size": (800, 800),
    "num_queries": 300,
    "backbone_lr_factor": 0.05,
    "ema_warmup_steps": 2000,
    "ema_momentum": 0.999,
    "eval_max_dets": 500,
    # 推荐起点（上游 M4）：
    "ssl": {"mode": "unsup", "view": "two_crop", "loss_coef": 0.1},
}
```

**消融矩阵**（同一 29k 步配方，唯一变量 = ssl 配置）：

| 组 | ssl | 目的 |
|---|---|---|
| a（单视图基线） | None | 现有 800px 基线（吞吐参照） |
| b（批匹配基线） | `{"mode":"off","view":"two_crop"}` | 隔离 two_crop 数据管线本身的增广收益 |
| c（推荐） | `unsup two_crop λ=0.1` | SSL 净贡献 |
| d（视图对照） | `unsup dropout λ=0.5` | 复现上游"dropout 档次优"排序 |
| e（sup 两档） | `sup two_crop same_image True/False` | 拥挤数据假负例缓解（B4 先测：`scripts/ssl_cooccurrence_stats.py`） |
| f（λ 扫描） | `unsup two_crop λ ∈ {0.05, 0.25, 0.5, 1.0}` | 复现"低 λ 稳健 / 1.0 有毒" |
| g（DOME 组合） | c + `density={"max_num_select":500}` | 分册 13 组合不侵蚀验证 |
| h（配对产率） | c ± 关 mixup/mosaic | D5 影响量化 |

记录：mAP@[.5:.95]、APs、`train/loss_ssl` 首→末、`ssl_paired_ratio`、步时延。验收：c − b ≥ +0.5 AP 且 c − b ≥ −0.2 AP 底线。

**探针用法**（`autovision._metrics.ssl_probes`，纯 torch CPU 库函数，未接入训练循环）：在短程插桩运行中收集匹配实例的 `query_features`/投影嵌入后离线调用——`position_leakage_r2`（R²>0.6 判位置泄漏，R10）、`cross_view_nn_accuracy`（纯最近邻恢复跨视图对应，two_crop 嵌入质量直接度量）、`linear_probe_accuracy`/`retrieval_map_at_k`/`alignment_uniformity`（类可分性/检索/塌缩守卫）。

**配对产率运维判据**：`ssl_paired_ratio` < 0.3 → 先查增广交互（RandomIoUCrop/ZoomOut 整框删除、mosaic 组成差异），必要时 two_crop 运行下调 `random_iou_crop_prob`；翻转导致的配对损失已由 D13 翻转感知配对消除（默认开，`flip_invariant_pairing=false` 退回上游语义）。

---

## 9. 风险与开放问题

| # | 风险 | 缓解 |
|---|---|---|
| R1 | two_crop 步时延与显存 ×2 | batch_size 减半或接受；无 auto-batch 探测器需适配（步数制）；smoke 实测记录 |
| R2 | mixup/mosaic 下配对产率下降 → 池变小、有效负例减少 | `ssl_paired_ratio` 监控（D10）；消融 h 组量化；必要时 two_crop 窗口内禁用 mixup（配置项后置） |
| R3 | VisDrone 假负例压力（上游实测 98.6% 同类共存）但 unsup 仍最优——机制未完全解释 | B4 先测本数据集；消融 e 组复验；列入后续研究（类原型缓冲） |
| R4 | `every_n_steps>1` + 用户 opt-out `find_unused_parameters` → DDP 报错 | 文档标注组合约束；默认值已兜底 |
| R5 | DOME 极小 batch + 动态查询 → 整批无有效对频率升高 | connected zero 兜底（全 rank 共识）；`ssl_paired_ratio` 观察 |
| R6 | SSL 梯度与 DOME 密度头/检测损失三方竞争 | 上游证据：λ=0.1 量级下无干扰；g 组消融直接度量；λ 扫描复用 f 组 |
| R7 | RT-DETRv2/OBB 家族后续接入时 `_refine_layers` 签名已变（8→9 元组） | 组件内私有 API，PR-1 同步更新全部调用点；§7 金标回归兜底 |
| R8 | ~~`_pair_two_crop` 逐图 Python 循环 + scipy CPU 指派~~ **已了结**：整池一次性下 CPU（每步 1 次同步）+ 段式分组（O(pool) 替代 O(图数×池) 掩码扫描）；**块对角 batched 指派经评估否决**——scipy 指派 O(n³)，合并为 O((Σn)³) 渐近更差，逐图 scipy 即最优 | 已实施；如步时延仍不达标再议 GPU 线性指派松弛（如 Sinkhorn） |
| R9 | **未配对实例完全退出 SSL 池**：跨视图 IoU 不足（遮挡/截断/大形变）的物体两视图嵌入都被丢弃——SSL 课程系统性偏向"易例"，难例恰是检测最需要表示质量的输入 | **D13 翻转感知配对已消除最大根源**（翻转占几何发散的一半）；`iou_match` 扫参（0.45–0.65）+ `ssl_paired_ratio` 监控；剩余未配对（裁剪/遮挡）已由 **D14 未配对负例**回收（`use_unpaired_as_negatives`，默认开） |
| R10 | **位置信息张力**：two_crop 的增广（缩放/裁剪/翻转）使绝对位置跨视图变化，对比目标隐式鼓励位置不变特征；而 `dec_bbox_head` 依赖位置信息回归框——两个目标在共享表示上方向相反 | λ 小（0.1）+ 检测损失主导（锚定作用）；探针已移植（`_metrics/ssl_probes.py`，`position_leakage_r2` 等 5 项，§8）——怀疑此效应时直接量化（R²>0.6 判泄漏）再调 λ |
| R11 | **`both` 模式两项共享偏置 logits，同类对上方向相反**：SupCon 把同标签对拉近（软正对），InfoNCE 把它们当负例推远；且偏置在低 IoU 同类对上进一步压低其 logit——两项在同类对上"拉锯" | 上游同款语义（共享一份 biased logits 是有意设计）；消融 f 组（unsup vs sup vs both）直接度量；实践中 unsup 或 sup 单模式已足够 |
| R12 | **mixup 混合图像上的对比信号**：mixup 行的像素是两图混合，配对仍由 targets 恢复（D5），但实例特征来自混合内容——对比一致性被"污染" | `ssl_paired_ratio` + 消融 h 组（关 mixup）量化；显著则 two_crop 窗口内禁用 mixup |

**开放问题**（评审时定）：

1. RT-DETRv2 decoder 家族 v1 是否一并接线（`RTDETRv2Decoder` 精修循环需同样暴露隐状态；模式与 dfine 相同）——倾向 v1 不做（与分册 13 v1 策略一致）；
2. OBB 家族（`obb_criterion.py` + `obb_decoder.py`）跟进时机；
3. `batch_size="auto"` 在 two_crop 下的语义（计对还是计图）——实施 PR-2 时定；
4. `ssl_probes` 五探针（线性探测/检索 mAP/alignment-uniformity/跨视图 NN/位置泄漏 R²）是否移植并接入验证循环——上游亦未接入；v1 仅 B4 统计；
5. 拥挤数据上 sup 劣于 unsup 的机制（上游 M4 未解释）——若 VisDrone 全量复现，立项类原型缓冲或同图正对默认化研究。

---

## 10. 参考索引

**RF-DETR-SSL 参考实现**（`/home/hdjs/Research/Software/rf-detr-ssl/`）：

| 内容 | 位置 |
|---|---|
| 权威设计文档（v2.1，含 M4 消融 §16、风险 §15、配方 §12） | `docs/learn/train/ssl-per-object-classification-head.md` |
| 数学组件（头/损失/gather/偏置/跨视图匹配） | `src/rfdetr/models/ssl_head.py:36-444` |
| criterion 编排（池构造/共识/配对/偏置） | `src/rfdetr/models/criterion.py:684-925`；发射门 `:1038-1046` |
| 模型挂载（发射/构造/权重） | `src/rfdetr/models/lwdetr.py:592-615, 873-895, 944-948` |
| two-view 数据集 / 双视图 collate | `src/rfdetr/datasets/two_view.py`；`src/rfdetr/utilities/tensors.py:372-396` |
| 数据管线包装 / auto-batch 校准 | `src/rfdetr/training/module_data.py:360-363`；`training/auto_batch.py:553-569` |
| 配置与校验 | `src/rfdetr/config.py:1066-1086, 1153-1169` |
| checkpoint 白名单 | `src/rfdetr/models/weights.py:166-169` |
| 评测探针 / B4 统计 | `src/rfdetr/evaluation/ssl_probes.py`；`src/rfdetr/datasets/ssl_stats.py` |
| 测试基线 | `tests/models/test_ssl_head.py`、`tests/models/test_ssl_criterion.py`、`tests/test_ssl_integration.py`、`tests/training/test_ssl_pipeline.py` |

**VTrain 受影响面**（`src/autovision/`）：

| 内容 | 位置 |
|---|---|
| D-FINE decoder（FDR 循环 / DN 分离） | `_task_models/object_detection_components/dfine_decoder.py:74-223` |
| D-FINE criterion（逐层匹配 / DeFE 模式 / 权重过滤） | `_task_models/object_detection_components/dfine_criterion.py:87-212` |
| 任务模型组装 / `_forward_train` | `_task_models/ltdetr_object_detection/task_model.py:84-403` |
| 训练包装（损失接线 / 参数组 / EMA） | `_task_models/ltdetr_object_detection/train_model.py:56-167, 267-357` |
| 训练参数 / 变体 | `_task_models/ltdetr_object_detection/config.py:416-519` |
| 匈牙利匹配（scipy/CPU 索引） | `_task_models/object_detection_components/matcher.py:11, 75` |
| 检测数据集 / transform / collate | `_data/object_detection_dataset.py:35-71`；`_transforms/detection_transforms/transform.py:127-257`；`collate.py:97-165` |
| 命令装配（数据集/loader/transform/collate） | `_commands/train_task/object_detection.py:184-249` |
| 引擎（累积切分 / DDP / 日志链路） | `_engine/engine.py:493-520, 656-713` |
| 优化器 no-wd helper | `_optim/optimizer_helpers.py:87` |
| 密度分支（组合语义 / 模式先例） | 分册 13 `docs/design/13-ltdetr-dome-integration.md` |

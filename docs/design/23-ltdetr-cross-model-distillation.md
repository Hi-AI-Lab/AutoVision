# VTrain 实现蓝图 · 分册 23：LT-DETR 跨模型蒸馏（GT 锚定的 teacher→student 三路监督）

| 文档属性 | 内容 |
|---|---|
| 版本 | v1.1（v1.0 设计评审稿 → v1.1：**核心已实施**（2026-10-12，实施记录与偏差见 §13）；G1–G3 门槛与 §9 消融待 GPU 环境） |
| 状态 | 🟡 代码落地 + 25 例测试全绿；`self:ema` / 导出 `.pt` teacher 可用，HF 权重 teacher 未实现（§13.2）；GPU pilot 待跑 |
| 上游思想 | DeepSeek-R1（arXiv:2501.12948）：把大模型的**输出**蒸馏给小模型，监督效率优于小模型自己长训/RL——R1-Distill 系列全部由对 R1 轨迹的监督微调得到，同尺寸效果超过自己跑 RL 的模型。【DS】结合本仓已有的 D-FINE **DDF 层间自蒸馏**（`dfine_utils.ddf_loss`），把"末层教中间层"升级为"跨模型互蒸馏" |
| 配套分册 | DeepSeek 迁移四联册：本册（蒸馏）+ 分册 24（FP8）/ 25（token indexer）/ 26（步工程优化） |
| 参考实现 | 无外部参考实现（本仓首创组合）；数学基础完全复用仓内已验证的 `ddf_loss` / `matched_pair_iou` / `HungarianMatcher` |
| 受影响模块 | `ltdetr_object_detection/config.py`、`train_model.py`、`task_model.py`（只读透传）、`object_detection_components/dfine_criterion.py`、`dfine_utils.py`（新增 cross-model DDF）、`deimv2_object_detection/`（只读复用）、`scripts/visdrone_ltdetr_distill_800.py`（新）、`tests/test_ltdetr_distill.py`（新） |
| 基线兼容承诺 | `distill=None`（默认）时全部家族既有行为、权重、导出图**逐比特不变**；teacher 不进 state_dict / 优化器 / EMA / 导出 / 验证路径 |

> **标注约定**：【DS】DeepSeek 论文/仓库中的已验证事实（出处为论文章节或仓库路径）；【VTrain】本仓库现状（出处 `文件:行`，路径相对 `src/autovision/`，行号对应当前工作树）；【设计】本文档决策；【差异】本设计内部的重大取舍（编号 D1–D7 及理由）。

---

## 1. 背景与动机

### 1.1 R1 的教训与本仓的缺口

【DS】DeepSeek-R1 的核心工程结论之一：对算力受限方，**把强 teacher 的输出变成软监督**（蒸馏）是让小模型变强的最便宜路径——R1-Distill-Qwen-1.5B 仅靠对 R1 轨迹的 SFT 就超过同尺寸模型数周 RL 的结果（R1 论文 §3）。对应的视觉语义：与其让 LT-DETR（~36M）在 VisDrone 上从 COCO 热启动硬训 160 epoch，不如同时给它的每个预测一个"同域更强的模型会怎么预测这个物体"的软目标。

【VTrain】本仓蒸馏基础设施的现状盘点：

| 既有机制 | 位置 | 与跨模型蒸馏的关系 |
|---|---|---|
| DDF 层间自蒸馏 | `dfine_utils.py:378-469`（`ddf_loss`）、`dfine_criterion.py:437-470`（接线） | **已经是蒸馏**，只是 teacher = 本模型末层。`ddf_loss` 的签名（teacher_corners/logits/boxes + student 四元组 + indices）天然支持外部 teacher |
| SSL 实例对齐管道 | 分册 14（`ssl_contrastive.py`） | 证明了"复用 Hungarian 匹配做实例级监督、零新增匹配"在本仓可行且已验证（75 例测试） |
| COCO 检测器热启动 | `coco_checkpoint.py`（`detector_checkpoint` 参数） | 权重级"蒸馏"（hard warm start）；本设计是**输出级**软监督，二者正交可叠加 |
| EMA 均值 teacher | `ema.py:23-87` | 免费的 self-teacher 对照臂（§2.3 臂 c） |
| deimv2 强 baseline | `deimv2_object_detection/`（HF `Deimv2ForObjectDetection` 薄包装，`task_model.py:116-254`） | **teacher 候选**：VisDrone 臂训练产物（`scripts/visdrone_deimv2_800.py`）就是同域强模型，目前只用来"对比"，从未被当作监督信号源 |

### 1.2 为什么是 GT 锚定对齐（本设计的核心问题）

跨模型蒸馏的第一难点：**student 和 teacher 的 query 空间互不相通**——两个模型各自 250–500 个 query，没有任何对应关系。LLM 蒸馏是 token 级对齐（同词表同位置），检测没有这个便利。

【设计 D1】**用 GT 做对齐锚**：双方各自对**同一批 targets** 跑自己的 Hungarian 匹配（student 已有 union 匹配 `indices_go`，`dfine_criterion.py:306-319`；teacher 新增一次自己的匹配）。于是每个 GT 目标 j 同时拥有两个模型视图：student 的匹配 query s_j 与 teacher 的匹配 query t_j——以 GT 为主键做 join，得到逐实例的 (student 预测, teacher 软目标) 对。这继承了分册 14 "复用匹配、零额外对齐开销"的已验证模式，且对 D-FINE 特有的 per-side 分布监督（D-FINE 的"推理轨迹"对应物）逐字适用。

**预期收益与判据**：VisDrone 密集小目标上，teacher 的匹配结果本身就是该域最优的"这个 query 该回归到哪、置信度多少"的软标签。文献先例（LD/FGD/MGD 一系检测蒸馏）在同域 teacher 下普遍 +1~3 AP；本仓判据：消融臂（§9）`distill 全开 − 基线 ≥ +0.5 AP`（10-epoch pilot 噪声线 ±0.3 AP），且 SSL/密度分支收益不被侵蚀。

---

## 2. Teacher 候选与选择策略

### 2.1 候选矩阵

| 候选 | 权重来源 | 同域性 | 类空间 | v1 定位 |
|---|---|---|---|---|
| **a. deimv2 VisDrone 臂 best.ckpt** | `scripts/visdrone_deimv2_800.py` 产出的 `runs/.../exported_best.pt` / `best.ckpt` | ✅ 同域同数据 | VisDrone 10 类 = student | **v1 主推**（G1 直接可测） |
| b. deimv2-dinov3-s COCO | HF 官方转换权重（`deimv2_object_detection/task_model.py:143-175` 的 `load_weights` 路径） | ❌ 跨域 | COCO 80 类 ≠ student 10 类 | 对照臂（只跑 box/DDF 腿，cls 腿关闭，D4） |
| c. EMA self-teacher | `ema.py`（student 自己的 EMA） | ✅ | ✅ | 方法论对照（隔离"蒸馏收益中多大比例来自更大模型"vs"来自软目标本身"） |

【设计 D2】teacher 通过**通用检测 checkpoint 加载器**接入，而不是硬编码 deimv2：v1 支持两类 source——`exported_*.pt`（AutoVision 导出格式，`engine.py:733-749`，含 `model_class_path`/`model_init_args`，可无损重建任意任务模型）与 HF 权重名（走 `load_pretrained_verified`）。deimv2 只是第一个实例；未来 FSDC-L 或ConvNeXt-L LT-DETR 臂同样可当 teacher，接口不变。

### 2.2 G1 门槛：teacher 质量前置校验

蒸馏的基本前提是 teacher 在**目标域**确实更强。任何候选接入前先跑一次 val 评测（复用 `COCOMAPEvaluator`），要求：

```
teacher val mAP ≥ student 当前基线 best mAP + 2.0 AP
```

不满足 → 该候选降级为"仅 box/DDF 腿"或直接否决。**teacher 更弱时蒸馏是负迁移**，这条门槛防住最大的静默失败模式。

### 2.3 部署形态

- teacher 前向 **`torch.inference_mode()`**、eval 模式、无梯度；
- teacher **不注册进 nn.Module 树**（`object.__setattr__` 存储，§4.2）→ 不进 `state_dict()`、不被 DDP 广播梯度、不被 `ModelEMA` deepcopy（`ema.py:29` 的 deepcopy 只复制已注册子模块）、不进任何导出图；
- 显存：teacher 权重 bf16 驻留 + 单次前向激活（batch 4+4 @800px，deimv2-dinov3-l 量级 ≈ 数百 MB 权重 + ~2GB 激活），24GB 卡在现役 two_crop 配方（micro 4）下有余量（doc 22 实测训练峰 19.7GB 需复核，G2 门槛实测为准）。

---

## 3. 三路损失的数学设计

记：student 最终层输出 `(z_s, b_s, c_s)` = logits / boxes / corners；teacher 对应物 `(z_t, b_t, c_t)`；GT 类别 `y`；student union 匹配 `indices_go`（`dfine_criterion.py:306-319`）；teacher 匹配 `indices_t`；逐图 GT→teacher-query 映射 `gt2t[i][j] = t_q`。

### 3.1 Leg 1 — 分类软目标 `loss_distill_cls`

对 union 匹配上的每对 (s_q, j)：

```
p_t = σ(z_t[t_q, y_j]).detach()              # teacher 对该 GT 类别的置信度
loss_distill_cls = BCEWithLogits(z_s[s_q, y_j], clamp(p_t, ε, 1−ε))   # ε=0.01
```

语义：student 学的不是"这个 query 该多自信"（那是 VFL 对 GT 的硬监督），而是"**域内最优模型对同一个物体给多少分**"——teacher 的欠置信（难例）与过置信（易例）校准被迁移。归一化：`/ num_boxes_go`（与 box 损失同口径，`dfine_criterion.py:310-319`）。

### 3.2 Leg 2 — 回归软目标 `loss_distill_box`

```
loss_distill_box = ( L1(b_s[s_q], b_t[t_q]) + λ_giou · (1 − GIoU(b_s[s_q], b_t[t_q])) ) / num_boxes_go
```

teacher 的框是"比 GT 更平滑的回归目标"（介于硬 GT 与 student 当前预测之间），对 VisDrone 小目标的边界抖动有稳定作用。`λ_giou=2.0` 对齐 matcher/criterion 的既有比例（`matcher.py:4`）。默认权重低于 GT 损失（§4.1 的 `box_coef=1.0`，对照 `loss_bbox=5.0`）——**GT 仍是主监督**，teacher 只做正则，这是 D5 的消融对象。

### 3.3 Leg 3 — 分布蒸馏 `loss_distill_ddf_cross`（D-FINE 特有，本设计最有新意的腿）

【VTrain】`ddf_loss`（`dfine_utils.py:378-469`）把"teacher 每条 query 的 per-side 分布"以温度 T=5 的 KL 教给学生，matched query 按 IoU 加权、unmatched 按 teacher max-prob 加权。跨模型场景下 unmatched 权重**语义失效**（两模型 query 无空间对应，teacher max-prob 对 student 的 unmatched query 没有意义）。

【设计 D3】**matched-only 形式**：复用 `ddf_loss` 的 KL 数学，但只保留 matched 元素项（`dfine_utils.py:466` 的 `m1`），归一化用被覆盖的 matched 元素数：

```
teacher_corners_q := c_t[gt2t[j]]   （GT 锚定到 teacher 的对应 query）
loss = T² · Σ_pairs IoU(b_s[s_q], y_j 的 GT 框) · KL(c_s[s_q] ‖ teacher_corners_q) / num_covered_elements
```

IoU 权重沿用官方"matched 权重 = IoU"语义（`dfine_utils.py:429-433`），但基准换成 GT 框（`targets=` 路径已在 `ddf_loss` 中存在，直接复用）。温度、T² 缩放与 bin 空间（`reg_max=32, reg_scale=4.0`，33 bins 非均匀 W 空间）与自蒸馏完全一致。

**reg 空间硬校验**：teacher 的 `reg_max/reg_scale` 必须与 student 完全一致（deimv2 与本仓同出 D-FINE v2 官方配方，均 33 bins / 4.0——G3 门槛实测确认）。不一致 → init 期 `ValueError`（bin 空间不可插值重采，静默错位 = 全程无声蒸馏垃圾分布）。

### 3.4 覆盖率诊断

GT 锚定天然存在"teacher 漏检"空洞：teacher 匹配不上某个 GT（漏检/低分）时该 GT 无软目标。设计诊断指标：

- `distill_coverage = 有 teacher 软目标的 matched 对数 / union matched 对数`（val 全集聚合，镜像 `ssl_paired_ratio` 的落点模式，`train_model.py:352-366`）；
- coverage < 0.7 连续 3 个 val 点 → 日志告警（teacher 域适配不足，考虑 λ 降档）。

无软目标的 matched 对：三腿全部跳过（`loss_distill_*` 仅按覆盖对计算），GT 硬监督照常兜底。

---

## 4. 详细设计

### 4.1 配置面（`config.py`，仿 `LTDETRSSLArgs` 模式 `config.py:439-604`）

```python
class LTDETRDistillArgs(PydanticConfig):
    """跨模型蒸馏分支（设计 doc 23）。None = 关闭，逐比特基线。"""
    model_config = {"extra": "forbid"}

    teacher_model: str                       # 导出 .pt 路径 或 HF 权重名（如 "deimv2/deimv2-dinov3-s"）
    teacher_checkpoint: Path | None = None   # exported_*.pt / best.ckpt（AutoVision 导出格式）
    teacher_image_size: tuple[int, int] = (640, 640)   # D6：teacher 分辨率与 student 解耦
    cls_coef: float = Field(default=1.0, ge=0.0)
    box_coef: float = Field(default=1.0, ge=0.0)
    box_giou_coef: float = Field(default=2.0, ge=0.0)
    ddf_coef: float = Field(default=1.5, ge=0.0)       # 对齐 loss_ddf 自蒸馏权重
    ddf_temperature: float = Field(default=5.0, gt=0.0)
    start_step: int = Field(default=1000, ge=0)        # 预热：匹配稳定前不蒸馏（见下）
    every_n_steps: int = Field(default=1, ge=1)
    require_same_class_space: bool = True   # False 仅在跨类空间 teacher（b 臂）下放开，且强制 cls_coef=0

# LTDETRObjectDetectionTrainArgs 新增（:786 ssl 字段同款）：
    distill: LTDETRDistillArgs | None = None
```

校验器（`_validate_ssl_configuration` 同款 `model_validator`）：

- `teacher_model` 与 `teacher_checkpoint` 至少一个；
- `ddf_coef > 0` ⇒ teacher `reg_max/reg_scale` 与 student 一致（init 期构建 teacher 后检查其 config，不一致 raise）；
- 类空间：teacher `num_classes != student num_classes` 时，`require_same_class_space=True` → raise；`False` → 强制 `cls_coef=0`（box/DDF 腿不依赖类空间，归一化坐标下可用，D4）；
- `start_step` 语义：step < start_step 时不计算三腿（criterion 步门控，镜像 `ssl.every_n_steps` 的写法 `dfine_criterion.py:508-509`）。动机：匹配前期剧烈漂移，teacher 软目标噪声大；R1 的 SFT 数据也是收敛后模型产出的。默认 1000 ≈ 1 epoch（808 步/epoch，`visdrone_ltdetr_800.py:66`）。

### 4.2 teacher 管理（`train_model.py`，train wrapper 持有）

```python
# __init__（EMA 构建 :227-235 之前，因为 G2 要把 teacher 显存算进总账；不注册进模块树）：
if model_args.distill is not None:
    teacher = _load_distill_teacher(model_args.distill)   # exported_*.pt → model_class_path/init_args
                                                          # 重建 + load_train_state_dict（strict）
                                                          # HF 名 → load_pretrained_verified
    teacher.requires_grad_(False).eval()
    object.__setattr__(self, "_distill_teacher", teacher)   # 绕过 nn.Module.__setattr__ 的注册
    self._distill_cfg = model_args.distill
```

- **设备就位**：`object.__setattr__` 的模块不随 `train_model.to(device)`（`engine.py:587`）迁移——首次使用时惰性搬运：`training_step` 里检查 `next(teacher.parameters()).device != x.device` 则 `teacher.to(x.device)`（一次性，幂等）；
- **teacher 匹配器**：独立 `HungarianMatcher()` 实例（`matcher.py:135-156`，无状态，成本同 student matcher）；
- **teacher 前向**（`training_step` 内，student 前向之后）：

```python
with torch.inference_mode():
    t_out = teacher._forward_train(teacher_x)   # deimv2: task_model.py:230-239（labels 不传，损失不算）
t_indices = t_matcher(t_outputs, targets)       # teacher 自己的匹配（同一批 targets）
teacher_targets = _build_teacher_targets(t_out, t_indices, targets)  # gt2t 映射 + gather (z_t/b_t/c_t)
```

- **D6（teacher 分辨率解耦）**：targets 是归一化 cxcywh（`train_model.py:284-287`），匹配与全部三腿损失都在归一化空间——teacher 可以在自己更小的画布（默认 640，前向更便宜）上对**同一批归一化 targets** 匹配，其预测与 student 800px 画布上的预测直接可比。teacher 的 `eval_size`/锚点网格由其自身加载参数决定（deimv2 路径 `deimv2_object_detection/task_model.py:166` 的 `eval_size` 覆盖机制已支持任意尺寸）。限制：`teacher_image_size` 须满足 teacher 自身的尺寸整除规则（构建时校验）；
- **two_crop 交互**：two_crop 使 student batch 为 2B 行（`visdrone_ltdetr_800.py:231-233` micro 4 → 前向 8 行）；teacher 前向同样跑 2B 行（逐行独立，targets 已随行复制）。teacher 成本因此 ×2——`every_n_steps` 是现成的节流旋钮，消融矩阵（§9 臂 f）量化。

### 4.3 criterion 扩展（`dfine_criterion.py`）

`forward` 签名追加 `distill_targets: dict | None = None`（携带 `teacher_logits/teacher_boxes/teacher_corners/gt2t 映射/coverage 统计槽`）。接线位置与模式完全对齐 DeFE/SSL 的"算一次、无后缀"先例（`dfine_criterion.py:477-489, 498-566`）：

```python
# 在 union 匹配 indices_go 与 FGL 目标缓存（:351-394）之后、DDF 段（:437）之前：
if self.distill_params is not None and self.training and distill_targets is not None \
        and step 门控命中:
    losses["loss_distill_cls"] = ...   # §3.1，/ num_boxes_go
    losses["loss_distill_box"] = ...   # §3.2，/ num_boxes_go
    if self.distill_params.ddf_coef > 0:
        losses["loss_distill_ddf"] = cross_model_ddf_loss(...)   # §3.3
    self.last_distill_stats = {"covered": k, "union": n}
```

- `weight_dict` 追加三条目（ctor 里做，镜像 `loss_ssl` 的 `dfine_criterion.py:149-150`）；aux/dn 复制循环会生成 `loss_distill_*_aux_i` 垃圾权重键，由聚合处 `if key in losses` 过滤（`dfine_criterion.py:627-629`，D7 既有机制的自动保护）；
- **`cross_model_ddf_loss`** 放 `dfine_utils.py`：从 `ddf_loss` 提取共享的 KL 核心（probs/log_probs/T² 缩放），`ddf_loss` 本体不动（自蒸馏路径零改动、金标回归兜底）；matched-only 掩码由调用方以 GT 锚定对齐表驱动；
- **DN/aux 分支不蒸馏**：DN query 是加噪 GT（teacher 对应物无意义），aux 层经 `loss_ddf` 自蒸馏链路已有层内监督——三腿只在最终层 union 匹配上计算一次；
- **fp32 island**：三腿全程 fp32 计算（teacher 输出 gather 后 `.float()`），对齐仓内 fp32-island 惯例（分册 14 D6 同款理由）；
- eval 态：`self.training=False` 双保险静默（同 SSL 的 `dfine_criterion.py:498` 门控）——验证路径零改动、无 teacher 前向。

### 4.4 train wrapper 接线（`train_model.py`）

- `training_step`（`:262-367`）：step 门控命中时跑 teacher 前向 + teacher 匹配 + `_build_teacher_targets`，把 `distill_targets` 传入 criterion；`log_dict` 经 `last_distill_stats` 落 `distill_coverage`（模式镜像 `ssl_paired_ratio`，`:352-366`）；
- **参数组零改动**：teacher 无梯度参数，`get_optimizer`（`:585-730`）的 `self.parameters()` 遍历不含未注册模块——天然不会被归组；
- **state_dict 卫生**：teacher 键不存在于任何 state_dict（未注册）——`on_save_checkpoint`（`:797-799`）、`get_export_state_dict`（`:816-822`）、EMA deepcopy（`ema.py:29`）全部自动免疫；
- **checkpoint= 热启动路径**：student 从 COCO checkpoint 起训与蒸馏正交（`engine.py:689-694` 的加载只碰 student 权重）。

### 4.5 DDP / AMP / 引擎契约

| 项 | 结论 |
|---|---|
| DDP | teacher 前向无 collective（inference_mode + 无梯度）；`find_unused_parameters=True` 默认（`engine.py:603-616`）不受影响；teacher 显存每 rank 一份（单卡/单机场景无冗余问题） |
| 梯度累积 | teacher 前向在每个微批内做（teacher 目标逐微批有效）；`every_n_steps` 按 optimizer step 计（`step` 参数即引擎步号，`engine.py:771`） |
| AMP | bf16 autocast 包住 training_step（`engine.py:807-811`）；teacher 前向走同一 autocast（bf16 推理，省显存算力）；三腿损失 fp32 island |
| NaN 守卫 | criterion 的 `nan_to_num` 零填充（`dfine_criterion.py:573-575`）覆盖三条新腿——teacher 漏检空洞不产生 NaN |
| EMA | 不含 teacher（未注册），零改动 |
| 引擎日志 | `loss_distill_*` / `distill_coverage` 经 `log_dict` → `_TrainWindow` → logger（`engine.py:61-94, 851-888`），零引擎改动 |

### 4.6 与其他分支的组合语义

| 分支 | 交互 |
|---|---|
| density（分册 13） | 正交：动态 query 的 padding 位 logits −1e4 不会被 union 匹配选中；`query_valid_mask` 防御过滤与 SSL 同款 |
| SSL（分册 14） | 正交可叠加：两者都复用最终层匹配但消费不同信号（SSL 用 student 自身两视图；蒸馏用 teacher 视图）。λ 各自独立，消融矩阵验证不互相侵蚀 |
| token_drop / indexer（分册 17/25） | 正交：teacher 输入独立（无 mask），student 侧 token 稀疏化只影响 student 特征质量——蒸馏软目标反而是稀疏化训练的稳定器 |
| FP8（分册 24） | **协同**：teacher 是纯推理前向（transformer-heavy 的 deimv2），是 FP8 量化收益最确定的位置（§10 R6） |

---

## 5. 差异与取舍清单

| # | 取舍 | 决策 | 理由 |
|---|---|---|---|
| D1 | query 对齐方式 | GT 锚定 join（双方各自匹配 + GT 主键） | LLM 式 token 级对齐在检测中不存在；GT 是唯一天然主键；复用分册 14 已验证的"匹配复用"模式 |
| D2 | teacher 接入形态 | 通用检测 checkpoint 加载器（exported_*.pt / HF 名），非硬编码 deimv2 | 未来 ConvNeXt-L / FSDC-L 臂可零改动接入；teacher 选择成为消融维度而非架构常量 |
| D3 | 跨模型 DDF 形态 | matched-only（丢弃自蒸馏的 unmatched 项） | unmatched 的 teacher max-prob 权重在跨模型下语义失效（query 无对应）；matched-only 保住有效信号 |
| D4 | 跨类空间 teacher | 允许（box/DDF 腿），强制 cls_coef=0 | 归一化坐标与 W 空间分布跨类空间有效；cls 腿依赖类别对齐，v1 不做映射表 |
| D5 | box 软目标的地位 | 低权重正则（1.0 vs GT 的 5.0），非替代 | teacher 框自身有误差；GT 硬监督必须主导——消融臂 d 量化"teacher 框替代 GT"的退化 |
| D6 | teacher 分辨率 | 与 student 解耦（默认 640） | 全链路归一化坐标使分辨率无关；teacher 前向成本降 ~36%（640²/800²）；teacher 原生训练分辨率更稳 |
| D7 | 预热策略 | start_step 步门控而非 λ warmup | 匹配漂移期的软目标是噪声不是信号；阶跃门控实现最简、消融可读（λ warmup 若需要可在 v1.5 加） |

---

## 6. v0 门槛（实施前必过）

| # | 内容 | 工具 | 判据 |
|---|---|---|---|
| G1 | **teacher 质量校验**：候选 checkpoint 在 VisDrone val 的 mAP | 新增 `scripts/eval_distill_teacher.py`（~80 行：加载 teacher → val 集跑 `COCOMAPEvaluator`） | teacher mAP ≥ student 基线 best + 2.0 AP，否则否决/降级 |
| G2 | **成本 profile**：teacher 前向 + 匹配的步时开销（micro 4 two_crop 配方） | `scripts/profile_step_phases.py` 加 `teacher` 相位（~20 行） | 开销 ≤ 45% 步时（超出 → `teacher_image_size` 降 640 以下 / `every_n_steps=2`） |
| G3 | **reg 空间确认**：teacher `reg_max/reg_scale` 实测等于 (32, 4.0) | G1 脚本顺带打印 teacher config | 不一致 → `ddf_coef` 强制 0 并记录（cls/box 腿仍可跑） |
| G4 | G1–G3 通过 → 实施 §7；失败 → 记录归档本册 | — | — |

---

## 7. 实现清单

| 文件 | 改动 | ~行 |
|---|---|---|
| `ltdetr_object_detection/config.py` | `LTDETRDistillArgs` + `distill` 字段 + 校验器 | ~80 |
| `ltdetr_object_detection/train_model.py` | `_load_distill_teacher` / 惰性搬移 / teacher 前向与匹配 / `distill_coverage` 诊断 | ~120 |
| `object_detection_components/dfine_utils.py` | 从 `ddf_loss` 提取 KL 核心 + `cross_model_ddf_loss`（`ddf_loss` 本体零改动） | ~70 |
| `object_detection_components/dfine_criterion.py` | ctor `distill_params` + 三条 weight_dict + forward 的 `distill_targets` 分支 | ~80 |
| `scripts/visdrone_ltdetr_distill_800.py` | 复制 `visdrone_ltdetr_800.py` + `distill` 配置（臂 a teacher） | ~30 |
| `scripts/eval_distill_teacher.py` | G1 工具 | ~80 |
| `tests/test_ltdetr_distill.py` | §8 全部 | ~350 |

总规模 ~2–2.5 天（不含 G1/G2 的 GPU 等待）。分两个 PR：PR-1 = 数学组件 + criterion 接线 + dropout 式最小通路（teacher=EMA self，零外部 checkpoint 依赖）；PR-2 = 通用 teacher 加载器 + deimv2 臂 + 脚本与消融。

---

## 8. 测试与验收（tiny CPU 配置 + gloo，风格对齐 `tests/test_ltdetr_ssl.py`）

1. **off-by-default 契约**：`distill=None` → state_dict 键集合、前向输出与基线逐比特一致（`torch.equal`）；
2. **teacher 不进任何持久化面**：`state_dict()` / `on_save_checkpoint` 产物 / `get_export_state_dict()` 均无 teacher 键；EMA state 无 teacher 键；
3. **GT 锚定对齐单测**：构造已知匹配（teacher 故意漏检 1 个 GT）→ `gt2t` 映射正确、`coverage` 计数正确、无软目标对被跳过且三腿仍有限；
4. **cls 腿手算对账**：与 `F.binary_cross_entropy_with_logits(z_s, clamp(p_t))` 手算一致；`p_t=0/1` 端点被 ε-clamp；
5. **box 腿**：teacher 框 = GT 框时退化为普通 L1+GIoU（与 `loss_boxes` 同公式同归一化）；
6. **cross DDF**：teacher corners = student corners 时损失为 0（KL 自退）；matched-only 掩码正确（unmatched query 不进分子分母）；温度/T² 缩放与 `ddf_loss` 共享核心（抽出函数的单测）；
7. **reg 空间校验**：teacher reg_max 不一致 + `ddf_coef>0` → init raise（含报错文案指向 G3）；类空间不一致 + `require_same_class_space=True` → raise；`False` → `cls_coef` 被强制 0 且可构建；
8. **步门控**：`start_step` 未到 / `every_n_steps` miss 步：无 `loss_distill_*` 键、`last_distill_stats=None`（不泄露上一步）；
9. **eval 安全**：criterion eval 模式无 distill 键、无 teacher 前向被触发（wrapper 侧门控）；
10. **惰性设备搬移**：teacher 在 CPU 构建 → 首个 training_step 后参数在 batch 设备、仅搬移一次；
11. **2-rank gloo**：teacher 前向无 collective、双 rank 损失有限、coverage 逐 rank 统计正确；
12. **two_crop**：2B 行下 coverage 按行独立、损失形状正确；
13. **args 往返**：`model_dump`/re-coerce（density/ssl/token_drop 同款，`task_model.py:166-175` 模式）。

验收金标：`tests/golden/` 若覆盖 LT-DETR 前向，`distill=None` 路径零差异。

---

## 9. VisDrone 配方与消融

**基座**：`scripts/visdrone_ltdetr_distill_800.py`，除 `distill` 外与 `visdrone_ltdetr_800.py` 完全一致（twin 模式）。

```python
MODEL_ARGS = {
    ...,  # visdrone_ltdetr_800.py 全量照抄
    "distill": {
        "teacher_checkpoint": "runs/<deimv2-visdrone-run>/exported_models/exported_best.pt",
        "teacher_image_size": [640, 640],
        "cls_coef": 1.0, "box_coef": 1.0, "ddf_coef": 1.5,
        "start_step": 1000,
    },
}
```

**消融矩阵**（同一配方，唯一变量 = distill 配置；10-epoch pilot，噪声线 ±0.3 AP）：

| 臂 | 配置 | 目的 |
|---|---|---|
| a | 基线（`distill=None`，即现役 ssl+erase 配方） | 参照系 |
| b | 三腿全开（推荐起点） | 净贡献 |
| c | teacher = EMA self（`teacher_checkpoint=None` + `teacher_model="self:ema"`，PR-1 支持） | 隔离"更大模型"vs"软目标本身" |
| d | `box_coef=5.0`（teacher 框加码） | D5：软目标替代 GT 的边界 |
| e | 单腿扫描：cls-only / box-only / ddf-only | 三腿各自贡献 |
| f | `every_n_steps=2` + teacher 640 | 成本-收益曲线（对照 G2） |
| g | b + density（Dome twin） | 与密度分支组合不侵蚀验证 |
| h | b + `ssl` 关闭 | 蒸馏与 SSL 的可替代性 |

记录：mAP@[.5:.95]、APs、`distill_coverage` 首→末、步时延。验收：b − a ≥ +0.5 AP 且各组合臂相对其无蒸馏对应臂 ≥ −0.2 AP 底线。

---

## 10. 风险表

| 风险 | 等级 | 缓解 |
|---|---|---|
| R1 teacher 域适配不足（COCO-only 候选）→ 覆盖率低/负迁移 | 高 | G1 前置硬门槛 + `distill_coverage` 在线监控（<0.7 告警）+ b 臂只推同域 teacher |
| R2 teacher 前向成本（two_crop 下 ×2） | 中 | G2 门槛；D6 分辨率解耦；`every_n_steps`；分册 24 的 FP8 teacher（R6） |
| R3 匹配漂移期软目标噪声 | 中 | `start_step` 阶跃门控（D7）；消融 f 组扫描 |
| R4 三腿与 SSL/density 梯度竞争 | 中 | λ 全部独立可调；g/h 组消融直接度量 |
| R5 checkpoint 误把 teacher 当 student 加载（用户配置错路径） | 低 | teacher 构建后即 `requires_grad_(False)`+eval；加载器校验 `model_class_path` 非 ltdetr 时打日志警告 |
| R6 deimv2-dinov3-l 无 HF 转换权重（`task_model.py:143-153` 只能本地构建） | 低 | v1 主推臂 a（本地 VisDrone 臂产物）；-l 若有本地训练产物同样经 G1 后可用 |
| R7 `exported_*.pt` 格式漂移（engine 导出格式变化） | 低 | 加载器只依赖 `model_class_path`/`model_init_args`/`train_model` 三个既有键（`engine.py:737-746`），加格式版本断言 |

---

## 11. 后续路线

| # | 内容 | 备注 |
|---|---|---|
| P0 | G1–G3 门槛脚本 + deimv2 VisDrone 臂 checkpoint 盘点（`runs/` 下现存 best） | 半天 |
| P1 | PR-1（数学 + EMA self 通路）→ PR-2（通用 teacher + deimv2 臂） | 2–2.5 天 |
| P2 | §9 消融矩阵执行与结果回填本册 | GPU 时间 |
| P3 | v1.5 候选：backbone 特征蒸馏腿（1×1 投影 + 归一化 MSE，金字塔逐层对齐）；λ warmup；COCO teacher 类别映射表 | 视 P2 结果立项 |
| P4 | 与分册 25 联动：蒸馏软目标作为 token indexer 的前景性监督信号（替代 GT 重叠 BCE 的升级选项） | 研究线 |

---

## 12. 参考索引

**DeepSeek 侧（【DS】事实来源）**：

| 内容 | 位置 |
|---|---|
| R1 蒸馏结论（R1-Distill 全系 SFT 自 R1 轨迹，小模型蒸馏优于自跑 RL） | DeepSeek-R1 论文（arXiv:2501.12948）§3；`github.com/deepseek-ai/DeepSeek-R1` |
| 本设计的 LLM 原型：soft target 蒸馏 + 推理轨迹监督 | R1 论文 §2.3（distill 设置）；本仓对应物 = D-FINE per-side 分布（§3.3） |

**VTrain 受影响面**（`src/autovision/`）：

| 内容 | 位置 |
|---|---|
| DDF 自蒸馏（KL 核心 / IoU 权重 / 温度） | `_task_models/object_detection_components/dfine_utils.py:378-469` |
| criterion 的 DeFE/SSL"算一次无后缀"接线模式 | `_task_models/object_detection_components/dfine_criterion.py:477-489, 498-566` |
| union 匹配 `_get_go_indices` | `_task_models/object_detection_components/dfine_criterion.py:306-319, 633-686` |
| HungarianMatcher（teacher 复用） | `_task_models/object_detection_components/matcher.py:38-156` |
| SSL 实例对齐管道（模式先例） | 分册 14 `docs/design/14-ltdetr-ssl-contrastive-head.md` |
| deimv2 teacher 候选（加载/前向/权重门控） | `_task_models/deimv2_object_detection/task_model.py:116-254` |
| 导出格式（teacher 加载源） | `_engine/engine.py:733-749` |
| EMA（self-teacher 臂 / 注册豁免依据） | `_task_models/object_detection_components/ema.py:23-87` |
| VisDrone 基座配方 | `scripts/visdrone_ltdetr_800.py:80-247`；deimv2 臂 `scripts/visdrone_deimv2_800.py` |

---

## 13. 实施记录（2026-10-12，v1.1 核心落地）

PR-1 + PR-2 的核心已实施（teacher 两个来源 + 三腿损失 + 全部接线），测试全绿。版本号 v1.0 → v1.1。

### 13.1 落地面

| 文件 | 内容 |
|---|---|
| `ltdetr_object_detection/config.py` | `LTDETRDistillArgs`（teacher 源/分辨率/三腿系数/start_step/every_n_steps/类空间规则）+ `distill` 字段 + 校验器 + `to_distill_params()` |
| `ltdetr_object_detection/train_model.py` | teacher 构建（`_load_teacher_from_export` / `self:ema` 别名）/ 惰性设备搬移 / `_build_teacher_targets`（inference_mode + 解码器临时 train + `num_denoising=0`）/ GT→teacher-query 映射 / `distill_coverage` 诊断 / DDF bin 空间守卫 |
| `object_detection_components/distillation.py`（新） | `DistillParams` + `cross_model_distill_losses`（GT 锚定 join + 三腿）+ `cross_model_ddf_loss`（matched-only KL） |
| `dfine_criterion.py` | `distill_params` ctor + weight_dict 三条目 + forward 的 `distill_targets` 分支（含 `gated_out` 步门控标记与 eval 双保险） |
| `tests/test_ltdetr_distill.py` | 25 例：config 校验 / cls 腿手工对账 / teacher 漏检跳过 / connected zero / DDF 自退 / 覆盖率映射 / 步门控 / eval 安全 / wrapper 级 state 键集合不变 / 导出 checkpoint teacher 加载（EMA 权重精确对账）/ 类空间两臂 / 解码器门控 |

验收：新套件 25 例全绿 + 受影响回归 365 例全绿 + ruff 干净（与分册 26 的 W1/W2 同批交付）。

### 13.2 与 §7 设计的偏差（如实记录）

| 偏差 | 依据 |
|---|---|
| teacher 源 v1 只有 `self:ema` 与导出 `.pt`；**HF 公开权重名（臂 b）`NotImplementedError`** | deimv2 任务模型以 `num_classes` 重建会触发类头 retile，"真 teacher 的类空间"问题未决（§2.1 臂 b 本就是对照臂，不阻塞主推臂 a） |
| deimv2 HF teacher 的 DDF 腿暂不可用 | 其 `_forward_train` 契约只透传 `pred_logits/pred_boxes`（HF 输出对象虽有 `intermediate_predicted_corners` 字段，未透传）→ bin 空间守卫自动丢弃该腿 + 一次性 warning；HF teacher 本就被上条挡在 v1 之外 |
| 实现 D7 的步门控语义为显式 `gated_out` 标记 | criterion 无法感知步号；wrapper 门控未命中时传标记（而非 None），让 criterion 能区分"调度跳过"与"接线错误"（后者响亮 raise） |
| EMA 块在 `__init__` 中上移至 criterion 之前 | `self:ema` teacher 需要别名 EMA 副本；deepcopy 不消耗 RNG，上移对其它初始化零影响 |
| DDP gloo 多卡用例未包含 | 单卡实验环境；padded-gather 类风险在本设计中不存在（teacher 前向无 collective），多卡冒烟留待 GPU 环境 |
| teacher_image_size 的 resize 用 `F.interpolate` 直达 | 归一化坐标使分辨率解耦成立（D6）；antialias 省略（teacher 输入不需要像素级对拍） |

### 13.3 待办（进入 §9 消融前）

1. GPU 环境：真实 VisDrone 配方 smoke（2k 步）——观察 `distill_coverage` 首→末走势与步时开销（§6 G2）；
2. deimv2 VisDrone 臂 checkpoint 盘点（G1：teacher mAP ≥ student 基线 +2.0 AP）；
3. `scripts/visdrone_ltdetr_distill_800.py` 配方脚本 + §9 消融矩阵执行。

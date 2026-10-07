# VTrain 实现蓝图 · 分册 15：SIoU 盒回归损失（D-FINE criterion GIoU 项替换）集成设计

| 文档属性 | 内容 |
|---|---|
| 版本 | v1.1（已实施，实施记录见 §6.1） |
| 状态 | 已实现并通过 `tests/test_detection_components.py::TestElementwiseSIoU`（11 例）+ `TestDFINESIoUCriterion`（7 例）+ `tests/test_ltdetr_object_detection.py::TestSIoUBoxRegression`（4 例）与全部受影响回归（detection components / ltdetr / ltdetr-ssl / dfine / task-models / rtdetr-oriented / train-pipeline / rfdetr / deimv2 / configs 共 331 例全绿）。**E1 3-ep shakedown 待启动**（§8） |
| 上游方法 | SIoU: More Powerful Learning for Bounding Box Regression（Scylla-IoU，arXiv:2205.12740） |
| 参考实现 | `/home/hdjs/Research/Software/rf-detr-ssl-v2`（RF-DETR v1.9.4 + 本地扩展；权威设计文档 `docs/learn/train/siou-box-loss-integration.md` v1.1，**S1 已实现并全量测试通过**，E0 探针已运行，E1 3-ep shakedown 未启动） |
| 受影响模块 | `_task_models/object_detection_components/`（`box_ops.py`、`rtdetrv2_criterion.py`、`dfine_criterion.py`）、`_task_models/ltdetr_object_detection/`（`config.py`、`train_model.py`）、`tests/`、`scripts/` |
| 基线兼容承诺 | `box_regression_iou="giou"`（默认）时 LT-DETR 全部既有行为、损失键、权重、日志**逐比特不变**；`box_regression_iou="siou"` 仅替换监督 box 损失中的 GIoU 项为 SIoU（emission 互斥），matcher cost / SSL 配对 / FGL / DDF / DeFE 全部不动 |

> **标注约定**：【RF-DETR-SSL-v2】参考实现中的已验证事实（出处 `文件:行`，路径相对 `/home/hdjs/Research/Software/rf-detr-ssl-v2/`）；【VTrain】本仓库现状（出处 `文件:行`，路径相对 `src/autovision/`）；【设计】本文档决策；【差异】有意偏离 RF-DETR-SSL-v2 的决策（附编号 E1–E7 及理由）。

---

## 1. 背景与目标

### 1.1 方法摘要（SIoU）

【RF-DETR-SSL-v2】（`docs/learn/train/siou-box-loss-integration.md` §1，含 2000-case 仿真复核）SIoU 在 IoU 损失上叠加距离与形状两个惩罚项：

```
L_box = 1 − IoU + (Δ + Ω)/2

角度项（不进损失，只调制 Δ）:
  Λ = 2·s·√(1−s²),  s = min(dx,dy) / √(dx²+dy²+ε²)
  （论文原形 1−2·sin²(arcsin(s)−π/4) 的化简形；s ≤ 1/√2 恒成立，
   故无 arcsin 定义域问题、无 45° 梯度奇异——该点导数精确为 0）
距离项:
  Δ = Σ_{t∈{x,y}} (1 − e^(−γ·ρ_t)),  γ = 2 − Λ
  ρ_x = (dx/c_w)², ρ_y = (dy/c_h)²    ← c_w/c_h = 最小外接框宽/高（各向异性归一化）
形状项:
  Ω = Σ_{t∈{w,h}} (1 − e^(−ω_t))^θ,   θ ∈ [2,6]，经验值 4
  ω_w = |w₁−w₂|/max(w₁,w₂), ω_h 同理   ← 无量纲相对偏差
```

三个对实现与预期重要的结构性结论（fork 文档 §1.2 已数值验证）：

1. **角度机制近收敛自熄**：Λ 对梯度的全部贡献正比于 `ρ·e^(−γρ)`，配对趋于重合时贡献 →0，损失平滑退化为 `1 − IoU + Ω/2`，无 CIoU 式 0/0 病理；
2. **无定义域奇异点**：化简形下 `1−s² ≥ 1/2`，唯一需要注意的数值点是 `s` 分母的 `√(dx²+dy²+ε²)`——`ε²` 加在被开方数**内部**（clamp 在 sqrt 之后会把无穷梯度经 `0·inf` 变成 NaN）；fp16 下 `ε²=1e-14` 下溢，该 op 不得用于纯 fp16（AMP fp32/bf16 均安全）；
3. **预期收益是噪声级的**（fork 文档 §1.3 证据审计）：仿真环境理想化；COCO +2.4 来自私有未开源网络；ultralytics main 已移除 SIoU 选项；后续文献（EIoU/WIoU/MPDIoU 时代）普遍结论为无单一 IoU 变体全场景占优。θ=4 下 Ω 的梯度在中等失配处仅约 L1 梯度的 1%。

E0 静态探针已运行（fork 文档 §5.1）：VisDrone 早训状态下 SIoU/GIoU 损失尺度比 **0.953**（46.8% 的 matched 对完全不相交时，GIoU 的外接框修正项均值 0.348 与 SIoU 的 (Δ+Ω)/2 均值 0.288 量级相当）——**权重直接继承 `giou_loss_coef=2.0`，无需下调**。

### 1.2 集成范围（与分册 14/SSL 移植计划的一致性）

【设计】本分册范围 = **LT-DETR + D-FINE decoder** 这一条 AutoVision 自有 criterion 线，即把监督 box 损失中的 GIoU 项替换为 SIoU 的单变量消融开关。明确不含：

- **RF-DETR 线**：criterion 在 HF 库内部（`transformers/loss/loss_lw_detr.py:187-190`），替换需整体换掉 `model.loss_function`——与 RF-DETR SSL 移植方案（拟议分册 16）中的备选方案是同一个杠杆，若未来实施该替换可顺手加开关，单独为之动刀不值当；
- **RT-DETR / PICO-Det / OBB 线**：`loss_boxes` 的共享函数默认值保持 giou，这些 criterion 不传新参数即零改动（§4.2）；
- **matcher cost 替换（fork S2）、SSL 配对一致性（fork S3）、论文 GA 搜 θ（fork S4）**：沿用 fork 判定——缓做/否决，见 §9。

### 1.3 预期收益与判定门

【设计】沿用 fork 定位：这是一次**廉价的单变量消融**（~80 行核心代码 + 1 个 3-ep shakedown，零额外每步开销），主要产出是机制归因数据（SIoU 的各向异性外接框归一化 + 指数饱和是否对 VisDrone 小目标提供了 L1 给不了的监督）。E1 判定门（§8）：

| E1 结果（SIoU − GIoU，绝对 mAP@0.5:0.95） | 判定 | 后续动作 |
|---|---|---|
| ≥ +0.002（+0.2pp） | 有信号 | 进入 S1-v2 机制归因变体（GIoU+Ω、θ=2、监督-only 隔离） |
| \|Δ\| < 0.002 | 噪声级 | 开关保留但默认关闭，记录证伪结论，不进任何默认配方 |
| ≤ −0.002 | 显著负 | 记录并冻结路线，文档写明数据 |

---

## 2. 两套代码的现状对照

### 2.1 落点差异表

| 关注点 | 【RF-DETR-SSL-v2】 | 【VTrain】 |
|---|---|---|
| box 损失所在 | `SetCriterion.loss_boxes`（`src/rfdetr/models/criterion.py:541-577`，分支 `:564-576`），criterion 自有 | **模块级共享函数** `loss_boxes`（`_task_models/object_detection_components/rtdetrv2_criterion.py:106-133`），被 `RTDETRCriterionV2`（方法委托 `:162-171`）与 `DFINECriterionV2`（导入 `dfine_criterion.py:46-48`）共同消费 |
| IoU 项实现 | `box_ops.elementwise_generalized_box_iou`（逐对） | `box_ops.generalized_box_iou_loss`（**成对矩阵**取对角线，`box_ops.py:72-84`） |
| 坐标约定 | 归一化 cxcywh → `box_cxcywh_to_xyxy`（归一化 xyxy）逐对算 | 归一化 cxcywh → `_cxcywh_to_xyxy_unit`（归一化 xyxy），成对矩阵 → `.diagonal()` |
| 匹配语义 | 逐层匈牙利匹配，`num_boxes` 归一化 | final/aux/dn/enc 各自成套；box 损失走 **union 指派**（`_get_go_indices`）+ `num_boxes_go` 归一化（`dfine_criterion.py:209-257`） |
| 权重承载 | 模型侧 `weight_dict`，SIoU 权重在 aux 复制**之前**写入（`lwdetr.py:960-965`） | criterion 自有 `weight_dict`，聚合循环把每个条目复制到 `_aux_i/_dn_i/_enc_i` 后缀（`dfine_criterion.py:456-475`），另有 `_pre`/`_dn_pre` 硬编码三行（`:458-464`） |
| 开关位置 | `TrainConfig.box_regression_iou`（`config.py:1097-1107`） | （新增）`LTDETRObjectDetectionTrainArgs`，criterion ctor 直传（§4.4） |
| 训练日志键 | `loss_siou`（`module_model.py:48` 直通） | `train_loss/loss_siou` 由引擎从 `loss_dict` 自动派生，无引擎改动 |

### 2.2 移植主线结论

【设计】SIoU 本体（~40 行数学）与开关语义（emission 互斥 + 权重继承 `giou`）从 fork **逐位照搬**；差异全部集中在**权重承载位置**与**共享函数的默认值兼容**两处（§4.3、§5-E3/E4）。fork 的实现已经过 TDD 全绿 + mypy 干净 + E0 探针校验，本移植不重新发明数学。

---

## 3. 总体设计

### 3.1 模块拓扑

```
LTDETRObjectDetectionTrainArgs (config.py)          ← 新增 3 个字段 + 校验
        │ box_regression_iou / siou_loss_coef / siou_theta
        ▼
RFDETRLTDETRTrain (train_model.py)                  ← DFINECriterionV2(...) 透传
        ▼
DFINECriterionV2 (dfine_criterion.py)               ← ctor 存 iou_term/theta；
        │                                             weight_dict：siou 模式弹 loss_giou、
        │                                             写 loss_siou（继承 giou 权重）
        ▼
loss_boxes (rtdetrv2_criterion.py, 模块级函数)       ← 新增 keyword-only iou_term/siou_theta，
        │                                             5 个调用点全部透传
        ▼
box_ops.elementwise_siou (box_ops.py, 新增)          ← 逐对 SIoU 损失 [N]，fork 逐位移植
box_ops.elementwise_box_iou (box_ops.py, 新增)       ← 逐对 IoU（fork 同名件移植）
```

### 3.2 开关语义与兼容策略

【设计】

- `box_regression_iou ∈ {"giou", "siou"}`，默认 `"giou"`。`"giou"` 路径的代码路径、损失键、数值**逐比特不变**（新参数全部带默认值，不进默认调用路径）；
- `"siou"` 时 emission **互斥**：损失字典出现 `loss_siou`、不再出现 `loss_giou`（主层、`_pre`、`_dn_pre`、`_aux_i`、`_dn_i`、`_enc_i` 全部后缀同步切换）——与 fork 一致，保证 `train_loss/*` 日志可读、权重聚合不出现双计；
- `siou_loss_coef=None`（默认）语义为**继承 `loss_giou` 的权重**（AutoVision 两条 criterion 均默认 2.0），E0 实测尺度比 0.953 支持直接继承；显式数值覆盖；
- θ 固定默认 4.0，仅作配置暴露，**不做数据集级搜索**（fork S4 否决）。

---

## 4. 详细设计

### 4.1 数学组件（`object_detection_components/box_ops.py`）

【RF-DETR-SSL-v2】`src/rfdetr/utilities/box_ops.py:142-197` 的 `elementwise_siou` 与其依赖的逐对 IoU（`elementwise_box_iou`）。

【设计】新增两个公开函数进 `__all__`：

```python
def elementwise_box_iou(boxes1, boxes2) -> tuple[Tensor, Tensor]:
    """逐对 IoU 与并集面积，xyxy 对齐对 [N,4]×[N,4] → ([N],[N])。"""

def elementwise_siou(boxes1, boxes2, theta: float = 4.0, eps: float = 1e-7) -> Tensor:
    """Scylla-IoU 损失（arXiv:2205.12740），逐对，[N] ≥ 0；相同框精确为 0。"""
```

实现要点（与 fork 逐位一致的硬约束）：

1. `s = min(dx,dy) / sqrt(dx² + dy² + eps²)`——`eps²` 在根号**内**（fork 注释：sqrt 后 clamp 会把无穷反向梯度经 `0·inf` 变 NaN）；
2. 角度项用化简形 `Λ = 2s·√(1−s²)`（恒有 `1−s² ≥ 1/2`，无定义域/45° 奇异）；
3. `c_w/c_h`、`max(w₁,w₂)` 均 `clamp(min=eps)`；
4. 逐对宽高/外接框 `clamp(min=0)` 与 fork 相同（负宽高退化为 0 而非 NaN）；
5. **不加** AutoVision `generalized_box_iou_loss` 式的 `nan_to_num` 收尾（【差异 E5】：fork 无此层、eps clamp 已覆盖退化域、上游 pred 框已过 `_sanitize_boxes_cxcywh_normalized`；criterion 聚合层的既有 `nan_to_num(nan=0.0)` 兜底不变——保持与 fork 的已验证实现逐位一致，同一数学不携带两套防御）；
6. 纯 fp16 下 `eps²=1e-14` 下溢——沿用 fork 约束：该函数**不得用于纯 fp16**（AMP fp32/bf16 安全），函数 docstring 写明。

### 4.2 共享 box 损失扩展（`rtdetrv2_criterion.loss_boxes`）

【VTrain】`loss_boxes(outputs, targets, indices, num_boxes)` 为模块级函数，逐图循环 matched 对：L1 累加 + `generalized_box_iou_loss(...).diagonal().sum()` 累加，返回 `{"loss_bbox", "loss_giou"}`（`rtdetrv2_criterion.py:106-133`）。

【设计】签名追加 keyword-only 参数（默认值保 RT-DETR 线零改动）：

```python
def loss_boxes(outputs, targets, indices, num_boxes, *,
               iou_term: Literal["giou", "siou"] = "giou",
               siou_theta: float = 4.0) -> dict[str, Tensor]:
```

循环体内按 `iou_term` 分派：`"giou"` 走既有路径逐比特不变；`"siou"` 以 `elementwise_siou(_cxcywh_to_xyxy_unit(pred), _cxcywh_to_xyxy_unit(target), theta=siou_theta)` 替换 GIoU 项，损失键为 `"loss_siou"`（归一化 `num_boxes.clamp(min=1)` 不变）。`RTDETRCriterionV2.loss_boxes` 方法委托（`:162-171`）不传新参数 → RT-DETR 线逐比特不变。

### 4.3 criterion 接线（`DFINECriterionV2`）

【VTrain】ctor 现有 `weight_dict` 默认 `{"loss_vfl": 1.0, "loss_bbox": 5.0, "loss_giou": 2.0, ...}`（`dfine_criterion.py:98-108`）；`ssl_params` 模式下 criterion 自写 `weight_dict["loss_ssl"]`（`:120-121`）；聚合层把每个 weight_dict 条目复制到 `_aux_i/_dn_i/_enc_i` 后缀，另有 `_pre`/`_dn_pre` 硬编码三行引用 `loss_giou`（`:458-464`）。

【设计】

1. **ctor 新增** `box_regression_iou: Literal["giou","siou"] = "giou"`、`siou_theta: float = 4.0`、`siou_loss_coef: float | None = None`（【差异 E3】：fork 在模型侧 `lwdetr.py:960-965` 解析权重继承，VTrain 的 weight_dict 归 criterion 自有——继承解析集中到 criterion ctor，与 `ssl_params.loss_coef → weight_dict["loss_ssl"]` 的既有模式同构）；
2. **siou 模式的 weight_dict 处理**（ctor 内一次）：

   ```python
   if box_regression_iou == "siou":
       # 继承基准：弹出前的 loss_giou 值（默认 dict 为 2.0；用户显式传入的
       # weight_dict 亦同源）。先读后弹，顺序不可换。
       inherited = self.weight_dict.get("loss_giou", 2.0)
       self.weight_dict.pop("loss_giou", None)   # emission 互斥，不留死键
       self.weight_dict["loss_siou"] = (
           inherited if siou_loss_coef is None else siou_loss_coef
       )
   ```

   弹键是安全的：聚合层 `if key in losses` 过滤（`:477-479`），残留死键只影响日志观感，弹出后字典与实际损失集保持一致；

3. **五个 box 损失调用点**（pre `:248`、final `:254`、dn_pre `:308`、dn 层 `:320`、enc `:451`）全部透传 `iou_term=self.box_regression_iou, siou_theta=self.siou_theta`——主层与全部深监督分支同步切换，与 fork 的 per-layer 生命周期一致；
4. **`_pre`/`_dn_pre` 权重三行泛化**：`loss_giou_pre`/`loss_giou_dn_pre` 改为按 `self._iou_loss_key`（giou 模式 = `"loss_giou"`，siou 模式 = `"loss_siou"`）取键——【差异 E4】：fork 无 pre/dn_pre 特例（其 weight_dict 复制在 aux 之前一次性覆盖），VTrain 的聚合循环对这两个键是显式硬编码，必须同步泛化，否则 siou 模式下 `_pre` 分支的 box 损失无权重（静默丢项）；
5. aux/dn/enc 后缀复制循环**不需要改**：siou 模式下 weight_dict 的 `loss_siou` 条目自动获得 `loss_siou_aux_i/loss_siou_dn_i/loss_siou_enc_i` 权重，`if key in losses` 过滤天然对齐 emission；
6. FGL/DDF/DeFE/VFL/SSL 完全不动（box IoU 项替换不影响分布精馏与密度分支的输入）。

### 4.4 配置面（`ltdetr_object_detection/config.py`）

【设计】`LTDETRObjectDetectionTrainArgs` 在回归参数区（`reg_max`/`reg_scale`/`dn_fgl_loss_weight`，`:564-594`）旁新增：

```python
#: 监督 box 损失的 IoU 项："giou" 保持上游 1−GIoU emission；"siou" 以
#: loss_siou（Scylla-IoU，arXiv:2205.12740）替换（emission 互斥，见分册 15）。
box_regression_iou: Literal["giou", "siou"] = "giou"
#: loss_siou 权重；None 继承 loss_giou 权重（E0 实测尺度比 0.953，无需下调）。
siou_loss_coef: float | None = Field(default=None, ge=0.0)
#: SIoU 形状项指数（论文经验值 4；不做数据集级搜索）。
siou_theta: float = Field(default=4.0, gt=0.0)
```

无跨字段校验需求（与 SSL/Dome 正交；`ge/gt` 约束由 pydantic 承担）。

### 4.5 训练接线（`ltdetr_object_detection/train_model.py`）

【设计】`DFINECriterionV2(...)` 构造点（`:118-131`）透传三个新字段。日志零改动：引擎从 `loss_dict` 派生 `train_loss/*`，`loss_siou`（及其 `_aux_i/_dn_i` 后缀）自动出现在训练日志；`RTDETRCriterionV2` 分支（`decoder="rtdetrv2"`）不透传 → 如配置了 `box_regression_iou="siou"` 但 decoder 非 dfine，**报错**（与 SSL 的 decoder 门槛同款，防止静默回退 giou 跑了一个“以为开了 SIoU”的 run）：

```python
if model_args.box_regression_iou == "siou" and decoder != "dfine":
    raise ValueError("box_regression_iou='siou' requires the D-FINE decoder in v1.")
```

---

## 5. 与 RF-DETR-SSL-v2 的差异清单

| 编号 | 差异 | 理由 |
|---|---|---|
| E1 | 数学本体（`elementwise_siou`/`elementwise_box_iou`）逐位照搬，无算法差异 | fork 已 TDD 全绿 + E0 校验，不重新发明 |
| E2 | GIoU 项坐标转换用 VTrain 既有 `_cxcywh_to_xyxy_unit`，非 fork 的 `box_cxcywh_to_xyxy` | 同一变换的 VTrain 命名；数值语义相同 |
| E3 | 权重继承（`siou_loss_coef=None` → giou 权重）在 **criterion ctor** 解析，非模型侧 `lwdetr.build` | VTrain 的 weight_dict 归 criterion 自有（`dfine_criterion.py:98`），且与 SSL 权重写入模式同构 |
| E4 | `_pre`/`_dn_pre` 权重硬编码三行需同步泛化为按活跃 IoU 键取值 | VTrain 聚合循环对这两个键显式硬编码（`:458-464`）；fork 无此特例。不泛化会导致 siou 模式 `_pre` 分支 box 损失静默无权重 |
| E5 | `elementwise_siou` 不加 `nan_to_num` 收尾（VTrain 的 `generalized_box_iou_loss` 有） | 与 fork 已验证实现逐位一致；退化域由 eps clamp + 上游 sanitize + criterion 聚合层兜底，不重复防御 |
| E6 | box 损失的匹配/归一化语义保持 VTrain 现状（union 指派 + `num_boxes_go`），非 fork 的逐层匹配 + `num_boxes` | SIoU 只替换 IoU 项的**函数形式**；VTrain 既有聚合语义是全仓库统一约定，不为单变量消融引入第二个变量 |
| E7 | 范围限定 D-FINE decoder；`decoder="rtdetrv2"` + siou 配置报错 | 与 SSL 的 decoder 门槛同款（`train_model.py:111-115` 惯例），防静默回退 |

---

## 6. 实施计划（单 PR，四个提交粒度）

### 6.1 实施记录（v1.1，2026-10-07）

全部落地，与设计一致，无偏差性返工。落点锚点（路径相对 `src/autovision/`）：

- `_task_models/object_detection_components/box_ops.py`：`elementwise_box_iou`（逐对 IoU + union，O(N)）与 `elementwise_siou`（fork `box_ops.py:142-197` 逐位移植，含 `eps²`-in-sqrt 数值约束与 fp16 警告 docstring）；新增私有 `_assert_equal_length`；
- `_task_models/object_detection_components/rtdetrv2_criterion.py`：`loss_boxes` 追加 keyword-only `iou_term/siou_theta`（`Literal` 标注），giou 路径操作顺序逐比特不变，siou 路径 emission 互斥；
- `_task_models/object_detection_components/dfine_criterion.py`：ctor 新增 `box_regression_iou/siou_loss_coef/siou_theta` + 未知值 fail-loud 校验（较设计 §4.3 增补：组件层直接使用者也得到防护）；weight_dict 先读继承基准后弹 `loss_giou`；`_loss_boxes` 绑定方法替换 5 个调用点；`_pre`/`_dn_pre` 权重行按 `_iou_loss_key` 泛化（§4.3-E4 陷阱已锁测试）；
- `_task_models/ltdetr_object_detection/config.py`：三字段（`Field(ge=0)`/`gt=0` 约束）落位于 `dn_fgl_loss_weight` 之后；
- `_task_models/ltdetr_object_detection/train_model.py`：decoder 门槛（`box_regression_iou='siou'` + 非 dfine → ValueError）+ criterion 透传。

测试落点：`tests/test_detection_components.py` 新增 `TestElementwiseSIoU`（fork 十例镜像 + θ 敏感性一例）与 `TestDFINESIoUCriterion`（emission 切换、matched 对数值一致、梯度流、权重继承/显式/外部 weight_dict、`_pre` 分支键泛化、ctor 校验）；`tests/test_ltdetr_object_detection.py` 新增 `TestSIoUBoxRegression`（默认值、pydantic 校验、decoder 门槛、criterion 权重接线）。lint（ruff check）对全部改动文件干净（`tests/test_ltdetr_object_detection.py` 的 I001 导入排序为 HEAD 既有问题，未顺手改动以免污染 diff）。

后续（未实施，待 E1）：`scripts/visdrone_ltdetr_siou_3ep.py` A/B shakedown 脚本按 §8 配方在启动实验时编写。

**二次审计（同日）**：逐条对账 §3–§7 后补齐两个测试缺口——(a) §7-12 的后缀全集此前只有组件级 `_pre` 覆盖，新增端到端测试 `test_siou_training_step_emits_full_suffix_family` 锁定完整键族（实测 tiny 模型训练步发出 `loss_siou` + `_pre/_dn_pre/_dn_0/_dn_1/_enc_0` 且无任何 `loss_giou`）；(b) §7-16 的 `_pre` 权重此前只验证「键存在」，新增加权差值测试 `test_pre_branch_siou_weights_are_actually_applied`（清零 siou 权重前后总损失之差必须恰等于 Σ siou 项 × 权重，权重行若失联会静默缩水）。另核查：全仓库剩余 `loss_giou` 引用均属其他任务族（picodet/OBB/RT-DETR 独立 criterion），符合 §1.2 范围划定；验证路径（validation_step）在 siou 模式下正常。

1. **`box_ops.py`**：`elementwise_box_iou` + `elementwise_siou`（含 `__all__`、docstring 的公式/数值约束），配套 `tests/test_detection_box_ops.py`（或既有 box_ops 测试文件）新增 SIoU 用例；
2. **`rtdetrv2_criterion.py` + `dfine_criterion.py`**：`loss_boxes` keyword-only 扩展；`DFINECriterionV2` ctor 三参数、weight_dict 处理、五调用点透传、`_pre/_dn_pre` 权重泛化；
3. **`ltdetr_object_detection/config.py` + `train_model.py`**：三个配置字段 + decoder 门槛 + criterion 透传；
4. **测试 + 文档**：criterion/config 用例全绿后回写本册状态行（对照 fork 文档 §9 惯例：实施记录小节 + 「已实现并通过 …」状态）。

## 7. 测试计划

沿用 fork 已验证用例集（【RF-DETR-SSL-v2】`tests/utilities/test_box_ops.py:187-310` 十例、`tests/models/test_criterion.py:350-420` 五例），镜像为 VTrain 布局：

**box_ops 层**（`elementwise_siou`）：

1. 相同框精确为 0；
2. 对称性 `L(a,b) == L(b,a)`；
3. 有界且有限（随机框族，损失 < 3）；
4. 轴对称（交换 x/y 通道损失不变）；
5. 共享中心时角度机制贡献自熄（退化为 `1−IoU+Ω/2` 的等式校验）；
6. 已知数值锁公式（移植 fork 的黄金值用例，`test_box_ops.py:247`）；
7. 梯度有限（`autograd.gradcheck` 级别的 finite 检查）；
8. CPU bf16 autocast 前向有限；
9. 空输入返回空；
10. 不等长输入抛 `ValueError`。

**criterion 层**（`DFINECriterionV2`）：

11. 默认模式 emission 为 `loss_giou`、无 `loss_siou`（逐比特回归）；
12. siou 模式 emission 为 `loss_siou`、无 `loss_giou`（含 `_aux_i/_dn_i/_pre/_dn_pre/_enc_i` 后缀全集）；
13. siou 模式数值与 `box_ops.elementwise_siou` 在 matched 对上逐位一致；
14. `siou_theta` 改变形状失配对的损失值、不影响重叠主导对；
15. 梯度流到 `pred_boxes`；
16. 权重契约：`siou_loss_coef=None` 时 `loss_siou` 权重 == 原 `loss_giou` 权重（2.0），显式值覆盖；`_pre/_dn_pre` 权重键同步泛化；
17. config：`siou_theta<=0` / `siou_loss_coef<0` 报错；`decoder="rtdetrv2"` + siou 报错（E7）。

**兼容性**：

18. `box_regression_iou="giou"`（默认）下 LT-DETR 既有测试全绿（`tests/test_ltdetr_object_detection.py` 等全部受影响回归）。

## 8. VisDrone E1 shakedown 配方

【设计】单变量 A/B（fork `m4_full/run_siou_3ep.py` 的 VTrain 等价物）：

- 数据：VisDrone@800，3 epochs；配方族沿用 `scripts/visdrone_*` 既有脚本惯例（新脚本 `scripts/visdrone_ltdetr_siou_3ep.py`，A/B 仅差 `box_regression_iou`）；
- 观察：`val_metric/map`（EMA）+ `train_loss/loss_siou` vs `train_loss/loss_giou` 曲线形态 + `train_loss/loss_bbox`（L1 是否被 IoU 项变化间接拉动）；
- 判定门：§1.3 表。无论成败回写本册状态行与结论（预期失败也是有效结论——在本仓库环境下证伪 fork COCO +2.4 的外推）。

## 9. 风险与开放问题

1. **收益大概率噪声级**（fork §1.3 证据审计 + E0 尺度比 0.953）：本移植的定位是消融开关而非默认改进；**不得**进入任何默认配方，除非 E1 过门；
2. **fp16 下溢**：`eps²=1e-14` 在纯 fp16 中为 0 → `s` 分母可能为 0。VTrain 训练管线为 AMP bf16/fp32，不受影响；若未来引入纯 fp16 路径需在调用点守护（docstring 已写明）；
3. **开放问题（fork 缓做项，结果依赖 E1）**：S1-v2 机制归因变体（GIoU+Ω、θ=2、监督-only 隔离）；S2 matcher cost 同步替换（改匹配动态，高风险，S1 无信号不碰；VTrain 落点为 `object_detection_components/matcher.py` 的 GIoU cost）；S3 SSL 跨视图配对框一致性（增量弱，真正场景是 label-free Route B）；S4 GA 搜 θ（否决）；
4. **RF-DETR 线的联动**：若 RF-DETR SSL 移植（拟议分册 16）决定整体替换 `model.loss_function`，SIoU 可作为同一 AutoVision criterion 内的开关顺带落地——本册不预置该工作。

## 10. 参考索引

- 上游论文：SIoU: More Powerful Learning for Bounding Box Regression（arXiv:2205.12740）
- fork 权威设计文档：`/home/hdjs/Research/Software/rf-detr-ssl-v2/docs/learn/train/siou-box-loss-integration.md`（v1.1，2026-10-07）
- fork 实现锚点：`src/rfdetr/utilities/box_ops.py:142-197`（SIoU 本体）、`src/rfdetr/models/criterion.py:541-577`（loss_boxes 分支）、`src/rfdetr/config.py:1097-1107`（配置）、`src/rfdetr/models/lwdetr.py:960-965`（权重继承）
- fork 测试锚点：`tests/utilities/test_box_ops.py:187-310`、`tests/models/test_criterion.py:350-420`
- VTrain 落点锚点：`_task_models/object_detection_components/rtdetrv2_criterion.py:106-133`（共享 loss_boxes）、`_task_models/object_detection_components/dfine_criterion.py:81-132,456-479`（criterion/权重聚合）、`_task_models/object_detection_components/box_ops.py:45-84`（GIoU 现状）、`_task_models/ltdetr_object_detection/config.py:564-594`（回归参数区）、`_task_models/ltdetr_object_detection/train_model.py:118-131`（criterion 构造）
- 相邻分册：13（Dome 密度分支）、14（LT-DETR SSL 对比头）

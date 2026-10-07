# VTrain 实现蓝图 · 分册 16：Box-Aware Eraser（框感知擦除增广）设计与全家族支持

| 文档属性 | 内容 |
|---|---|
| 版本 | v1.1（检测路径 + P1.5 退火 + P2 R1-A1 SIGReg 已实施；实施记录见 §5/§6） |
| 状态 | 检测路径 19 例 + SIGReg 13 例 + LT-DETR SSL 集成 2 例全过；真实 VisDrone 健全性检查通过 |
| 来源设计 | rf-detr-ssl-v2 `docs/learn/train/levjepa-detr-integration.md` §8（R3b）/ §9（R4），代码 `src/rfdetr/datasets/box_aware_eraser.py`、`src/rfdetr/training/callbacks/erasure_schedule.py` |
| 上游思想 | LeVJEPA（Meta）：随机互补子集丢弃优于结构化 masking；LeJEPA：SIGReg 分布正则 |
| 参考实现 | `/home/hdjs/Research/Software/rf-detr-ssl-v2`（Apache-2.0） |
| 受影响模块 | `_transforms/detection_transforms/`（新增 eraser + args + transform 接线）、`scripts/`（4 个脚本 `--erase` 开关）、`tests/test_box_aware_eraser.py` |
| 基线兼容承诺 | `random_erase_prob=None`（默认）时，全部检测家族的既有行为、权重、导出图**逐比特不变**；验证管线强制密集（val 配方硬关擦除） |

> **标注约定**：【RFSSL】rf-detr-ssl-v2 参考实现中的已验证事实（出处 `文件:行`，路径相对该仓库）；【VTrain】本仓库现状（出处 `文件:行`，路径相对 `src/autovision/`）；【设计】本文档决策；【差异】有意偏离 rf-detr-ssl-v2 的决策（附编号 E1–E3 及理由）。

---

## 1. 背景与目标

### 1.1 为什么"随机擦除 patch"必须框感知

DETR 的匈牙利匹配是**强制性**的：每个 GT 框必须被指派一个 query（【RFSSL】levjepa 文档 §8.1）。若增广把整个目标抹掉，匹配不会失败，而是产生**无证据监督**——box 回归在教 query 幻觉一个看不见的框，且噪声系统性集中在已经落后的短板类别。VisDrone 小目标在 800px / 16px cell 下仅占 1–2 个 cell，普通 RandomErasing 擦掉一个 100×100 块就会直接消灭一串 GT 的监督信号——这是本设计拒绝"随意擦除"、必须框感知的根本原因。

### 1.2 机制摘要（V1 主模式）

图像划分为 16×16 px cell（对齐 backbone patch 粒度）；凡与 **margin 外扩后的 GT 框有任何重叠**的 cell 全部保护；其余背景 cell 均匀随机采样 `round(ratio × |背景|)` 个置 0（ImageNet 归一化空间中 0.0 即数据集均值灰）。两视图（two_crop）各自独立采样，获得互补的背景子集——InfoNCE/检测头被迫依赖物体证据而非背景配对，这是该增广对 SSL 的主要作用机制（LeVJEPA 消融：随机互补子集 50.7 优于结构化 tube masking 39.6）。

### 1.3 V2 约束式前景擦除（"每个物体至少还有 2 Patch"）

V1 只擦背景。V2（`interior=True`）让**框内部也参与擦除**，但每框随机预留 `min_keep=2` 个内部 cell **永不擦除**；外扩环仍全保护（边界定位证据）；内部 cell 数 ≤ `min_keep` 的框自动全保护。【RFSSL】真实数据量化：81% 的框可擦、框内像素保留中位数 91%——这是"遮挡式部分不变性"区间，与 VisDrone 的密集遮挡工况同向。

---

## 2. 参考实现机制详解（rf-detr-ssl-v2）

| 组件 | 文件 | 职责 |
|---|---|---|
| `protected_cell_mask` | `datasets/box_aware_eraser.py:36` | 归一化 cxcywh → 像素 xyxy（**取整防浮点噪声**）→ margin = `max(padding × 边长, 2px)`（padding=0 显式关闭、无下限）→ any-overlap（比 cell-center 保守，单 cell 目标零风险） |
| `sample_erase_mask` | 同上 :80 | 背景集内无放回均匀采样精确 `round(ratio×|背景|)` 个 |
| `reserve_interior_cells` | 同上 :106 | 逐框随机预留 `min_keep` 个内部 cell（**预约束**：预留进入保护掩码后才采样，下限按构造成立；重叠框各自保留自己的下限）；高级索引写入切片视图（非连续内存的 rect 不能 `.view(-1)`） |
| `erase_background` | 同上 :159 | V1/V2 统一入口：`protected = (保护环 & ~内部) | 预留` → 采样 → `repeat_interleave` 展开 cell 掩码 → clone 后置 0（**不改动输入张量**） |
| `BoxAwareEraserDataset` | 同上 :208 | 数据侧 wrapper：per-view 掷币 `prob`；train-only；`set_epoch_ratio` 供退火回调驱动 |
| `ErasureScheduleCallback` | `training/callbacks/erasure_schedule.py` | 逐 epoch ratio 线性衰减到 0（Dome aug-stop 节奏）；constant 模式推荐首跑；persistent_workers 必须为 False |
| `build_window_keep_mask` / `WindowKeepMaskDataset` | 同上 :315/:348 | R4 路线：window 粒度 keep-mask（真省 FLOPs），需模型侧 gather 支持 |

**自限特性**：密集图（数百框）protected cell 可达 40–80%，候选集自动收缩、擦得少——密集场景零伤害、稀疏场景正则更强，无需人工干预。

---

## 3. LeVJEPA 全路线清单与 VTrain 裁定（遗漏盘点）

| 路线 | 内容（rf-detr-ssl-v2 判定） | VTrain 现状 | 裁定 |
|---|---|---|---|
| **R1** per-object SIGReg | LeJEPA 的 SIGReg 分布正则嫁接到 per-object SSL head：A1（InfoNCE+SIGReg，池=微批全部有效匹配实例，M=1024 随机方向、17 节点梯形积分 t∈[0,3]）/ A2（MSE 无负例+SIGReg） | **未实现**（VTrain 的 `ssl_contrastive.py` 只有 InfoNCE，与【RFSSL】的 λ=1.0 toxic 问题同构） | **P2 立项**（设计文档 14 的扩展；预期收益不是 mAP 而是 λ_ssl 鲁棒性——工程解锁更大 SSL 权重；实验矩阵直接抄【RFSSL】§4：E1a 优先，60 ep 止损线） |
| **R2** LeVJEPA-VideoMix 权重当初始化 | 否决（视频时序预训练与静态图域不合） | 不适用 | **否决** |
| **R3a** 多视图 V>2 | 可行缓做（数据/显存线性上涨；V=3 作二期） | two_crop 已有（`two_view.py`），V=3 无 | **P3 观察**（等 V2 擦除消融出数后决定） |
| **R3b** 擦除式增广（本文档） | V1/V2 已实现 | **检测路径已实施**（§4）；SSL two_crop 路径天然兼容（§4.3） | ✅ 本文档 |
| **R3c** block-causal / tubelet | 不适用（静态图无时序） | 不适用 | **否决** |
| **R4** window 级字面 token 丢弃 | 已实现且机制正确（Nano @960 实测 2.4× 歨时收益），但 **VisDrone 最终判定：默认关，不推荐**——num_windows=2 下 87% 的图 4 个窗口全含框，期望可丢窗口仅 4.4% | VTrain 骨干是 HF DINOv3（非【RFSSL】本地 windowed-dinov2 fork），R4 不移植 | **否决（记录依据）**：密集航拍 + HF 骨干双重不支持；若未来骨干原生支持更细窗口划分可重评 |

**结论**：LeVJEPA 集成文档中尚未在 VTrain 落地的实质遗漏是 **R1（SIGReg）**；R3b 的检测路径已补齐；R2/R3c/R4 有明确否决依据。SSL two_crop 擦除消融是当前最便宜的待跑实验。

---

## 4. VTrain 移植设计

### 4.1 注入点（差异 E1：transform 内置 vs 数据侧 wrapper）

【RFSSL】把 eraser 作为 Dataset wrapper 插在 transform 与 TwoViewDataset 之间；【设计 E1】VTrain 改为**transform 内置**——在 `RTDETRDetectionTransform.__call__` 末尾（归一化图像 + 归一化 cxcywh 框处，与【RFSSL】同一工作空间）按概率应用。理由：

1. VTrain 全部 6 个检测家族（rtdetr/dfine/ltdetr/picodet/rfdetr/deimv2）共用同一个 transform（`_commands/train_task/object_detection.py:287`）——**一处接线，全家族生效**，无需逐家族 wrapper；
2. LT-DETR 的家族参数构建器 `build_ltdetr_train_transform_args` 返回同一 `RTDETRDetectionTransformArgs` 且 `overrides` 透传（`_task_models/ltdetr_object_detection/transforms.py:57`）——`random_erase_*` 直接可用；
3. two_crop 路径：`TwoViewObjectDetectionDataset` 把 transform 委托回内层数据集，每次 `__getitem__` 独立调用两次 transform（`two_view.py:17`）——**每次调用独立掷币 = 【RFSSL】的 per-view 采样语义**，wrapper 层无需存在。

### 4.2 配置面（`RTDETRDetectionTransformArgs`）

```python
random_erase_prob: float | None = None   # 总开关，None=关（val 配方硬置 None）
random_erase_ratio: float = 0.4          # 擦除比例 (0, 1]
random_erase_padding: float = 0.15       # GT 框外扩比例（2px 下限）
random_erase_cell: int = 16              # cell 边长（对齐 patch）
random_erase_interior: bool = False      # V2 约束式前景擦除
random_erase_min_keep: int = 2           # V2 每框保留 cell 下限
```

### 4.3 家族覆盖矩阵（实测）

| 家族 | 命令前缀 | transform 路径 | erase 支持 | 端到端验证 |
|---|---|---|---|---|
| RT-DETR | `rtdetr/*` | `RTDETRDetectionTransform` | ✅ | 契约测试 |
| D-FINE | `dfine/*` | 同上（rtdetr 路径继承） | ✅ | 契约测试 |
| LT-DETR / Dome | `ltdetr/*` | 同上 + 家族 builder 透传 | ✅ | **真实 VisDrone 800px 端到端**（6 图擦除率 35.5%，GT 全保留） |
| PicoDet | `picodet/*` | 同上 | ✅ | 契约测试 |
| RF-DETR | `rfdetr/*` | 同上 | ✅ | 契约测试 |
| DEIMv2 | `deimv2/*` | 同上 | ✅ | **真实 VisDrone 800px 端到端**（12 图擦除率 35.1%，38–85 框/图全保留） |
| SSL two_crop | `ssl.view="two_crop"` | 同一 transform，双视图各自掷币 | ✅（结构性） | §6 P1 待跑消融 |

脚本开关：`visdrone_deimv2_800.py`、`visdrone_ltdetr_dome_800.py`、`visdrone_rtdetr_800.py`、`visdrone_rfdetr_800.py` 均有 `--erase`；其余脚本传 `transform_args={"random_erase_prob": ...}` 即可（框架层全家族已支持）。

### 4.4 与 val / 导出的隔离（差异 E2）

`build_val_transform_args` 硬置 `random_erase_prob=None`——验证与导出永远密集，与【RFSSL】"train-side only, dense at inference/export" 约定一致；`erase_background` 本身对无框图像保守跳过（直接返回原张量），不改动输入。

### 4.5 差异 E3：退火暂缓

【RFSSL】的 `ErasureScheduleCallback`（逐 epoch 线性衰减到 0）v1 未移植：官方建议首跑 constant；VTrain 的 `resolve_step_schedule` 机制（scale_jitter 的 `step_stop` 同款）是现成的二期注入点。**P1.5**：若 constant 消融出正信号，加 `random_erase_step_stop`（复用 step 窗口机制，~20 行）。

---

## 5. 已实施记录

| 项 | 内容 | 验证 |
|---|---|---|
| `box_aware_eraser.py`（新） | 4 个纯函数逐行对齐【RFSSL】；box 转换换用 VTrain `box_cxcywh_norm_to_xyxy` | 19 契约测试 |
| args + transform 接线 + val 隔离 | §4.2 / §4.4 | 契约 + 集成测试 |
| 4 个脚本 `--erase` | §4.3 | 语法 + lint |
| 真实数据健全性 | VisDrone 12 图（DEIMv2 路径）/ 6 图（LT-DETR builder 路径）：擦除面积 35.1% / 35.5%（标称 40%，密集自限 ✓），GT 框零丢失 | 已运行 |
| 回归 | `test_box_aware_eraser.py` 19 例 + 检测变换 + deimv2/rfdetr 套件全过；ruff 全绿 | 已运行 |

---

## 6. 后续路线（按优先级）

| # | 内容 | 成本 | 判据 / 备注 |
|---|---|---|---|
| **P1** | **擦除消融跑批**：Dome ± erase（对照 27.37）× DEIMv2-S ± erase（对照 21.82），V2 模式 constant | 各 ~1–2.5 h，已排队 | ±0.3 mAP 为噪声线（单 epoch 波动 ±0.1–0.3）；Dome +erase 若 ≥ +0.5 即转正式配方 |
| **P1.5** | ✅ **已实现**：`random_erase_step_stop`（`"auto"`→共享 step 调度 stop，`None`=全程 constant；transform 按步门控） | ~20 行 | 契约测试 4 例；与 scale_jitter.step_stop 同款语义 |
| **P2** | ✅ **已实现（A1）**：`epps_pulley_sigreg` + `SSLProjectionHead.forward_with_raw` + `SSLParams.sigreg_{coef,slices,knots}` + `LTDETRSSLArgs` 校验（mode≠off / view=two_crop / matched_gt）+ criterion `loss_sigreg`（独立 weight_dict 项）+ 任务模型 raw taps。**实现差异**：参考实现只对 group-0 匹配池做统计，VTrain 对全查询池（batch 展平）——分布正则同时约束未匹配 query 半区，语义记录在案 | ~120 行 | 13 例测试（分布敏感性 6 + raw tap 3 + 配置面 4）；**待办**：只读校准 run 定 λ_sr 网格，随后按【RFSSL】§4 矩阵 E1a→E2 推进（60 ep 止损线） |
| **P3** | R3a V=3 三视图 | 数据/显存线性 | V=2 消融出正信号后再议 |
| 否决 | R2 / R3c / R4 | — | §3 已记录依据 |

---

## 7. 测试与验收

- `tests/test_box_aware_eraser.py`（19 例）：保护 cell 重复采样永不擦、margin 扩保护（含 padding=0 无下限分支）、空框/零 ratio 恒等、精确擦除数、单 cell 目标全 ratio 下存活、擦除置 0 且保护像素逐位不变、输入不变异、生成器复现、min_keep 逐框下限（含重叠框）、小框 interior 全保护、膨胀环保留、interior 开启时背景仍被擦、transform 开/关集成、val 配方隔离。
- 真实数据：VisDrone 12 图端到端（DEIMv2 路径）+ 6 图（LT-DETR builder 路径）。
- 回归：检测变换 / deimv2 / rfdetr 套件全过；`random_erase_prob=None` 默认路径零改动。

---

## 10. 参考文献

1. LeVJEPA / LeJEPA 系列思想与消融数据：rf-detr-ssl-v2 `docs/learn/train/levjepa-detr-integration.md`（§0 路线表、§8 R3b、§9 R4）。
2. 参考实现：rf-detr-ssl-v2 `src/rfdetr/datasets/box_aware_eraser.py`、`src/rfdetr/training/callbacks/erasure_schedule.py`（Apache-2.0）。
3. 关联分册：`13-ltdetr-dome-integration.md`（Dome 密度引导）、`14-ltdetr-ssl-contrastive-head.md`（SSL 对比头，R1 的宿主）。

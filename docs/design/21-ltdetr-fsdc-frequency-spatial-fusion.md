# VTrain 实现蓝图 · 分册 21：LT-DETR × FSDC-DETR 频域-空域协同特征集成设计

| 文档属性 | 内容 |
|---|---|
| 版本 | v1.1（设计修订：按分册 22 重核的论文① Eq.1–13 精确形式修正 MKSRU 通道映射、FSD-Down 门控形式、SFS-FF 落点；组件规格源移交分册 22） |
| 状态 | 待评审。官方代码未放出（"being processed"，见 §10），本文档的模块细节以论文文本/图示为准，歧义处全部落入 §5 差异清单（F1–F9） |
| 上游论文 | FSDC-DETR: A Frequency-Spatial Domain Collaborative DETR for Small Object Detection（ECCV 2026, arXiv:2607.05176v2） |
| 参考实现 | **无官方代码**。辅助参考：FDConv 官方仓库（github.com/Linwei-Chen/FDConv, CVPR 2025）、SFS-Conv（CVPR 2025）的 FPU、HWD（Pattern Recognition 2023）的 Haar 下采样 |
| 受影响模块 | `_task_models/ltdetr_object_detection/`（config、task_model、backbone_wrappers）、`_task_models/object_detection_components/`（新增 4 个文件）、`tests/`、`scripts/` |
| 基线兼容承诺 | `fsdc=None`（默认）时，LT-DETR 全部既有行为、权重、导出图**逐比特不变**；新增模块全部零损失、零 decoder 改动（与分册 13 的 Dome 移植相比改动面显著更小） |
| 宿主决策 | 集成宿主 = **LT-DETR（原生实现）**。DEIMv2 任务模型是 HF transformers 包装（模型本体在库内，见 §2.1），**不做模块级集成**（非目标，理由见 §2.2） |

> **标注约定**：【论文】FSDC-DETR 论文事实（引用 §/图，arXiv 2607.05176v2）；【VTrain】本仓库现状（出处 `文件:行`，路径相对 `src/autovision/`）；【设计】本文档决策；【差异】有意偏离论文原表述的决策（附编号 F1–F9 及理由）。

---

## 1. 背景与目标

### 1.1 FSDC-DETR 方法详析

#### 1.1.1 定位与成绩

FSDC-DETR = **DEIMv2-L + 三个频域感知模块**，ECCV 2026。在 VisDrone-DET2019 test（800×800, maxDets=300）上 AP 24.7 → **31.1**（+6.4）、AP50 43.3 → 52.3、APs 14.2 → **21.0**（相对提升 48%）；AI-TODv2 test 上 AP +6.6、APs +6.9。中大型目标同步受益（论文 §4 主表）。

官方代码仓库已建（`nevereverinsomnia/FSDC-DETR`）但**代码未放出**，数字目前不可独立复现——这是引用其结果前最大的 caveat，也是本文档将"可验证的模块级移植"而非"整体复现"作为目标的直接原因。

#### 1.1.2 动机链（三个观察，各对应一个模块）

| # | 观察 | 对应模块 |
|---|---|---|
| O1 | ViT 因全局注意力呈**低通**特性（过度平滑），CNN 因局部感受野天然保留**高频**；DEIMv2 双分支（DINOv3 ViT + HGNetv2）只用 **concat + 1×1 conv** 融合，对频谱异构完全盲目，可能引入混叠 | **DBFSAF**（backbone 出口融合） |
| O2 | hybrid encoder（RT-DETR 系 CCFF）的跨尺度融合依赖常规卷积块（CSPRepLayer/RepVgg），高频在多尺度聚合中被逐步衰减 | **SFS-FF**（encoder 融合块替换） |
| O3 | 卷积下采样把空间聚合与频率混叠**耦合**在一起；论文点名批评 UAV-DETR 的 frequency-focused downsampling：平均池化本身就是低通滤波器，FFT 之前高频已被抹掉 | **FSD-Down**（下采样替换） |

三者构成"**构建**（DBFSAF 产出频-空互补表征）→ **传播**（SFS-FF 跨尺度无衰减压送）→ **保持**（FSD-Down 下采样不丢高频）"的流水线。

#### 1.1.3 模块机制

**DBFSAF（Dual-Branch Frequency-Spatial Adaptive Fusion）**——论文 §3.1。作用于每个金字塔层 P 上的双分支特征拼接：

```
F_init = FDConv(Cat[F_L, F_H])                    # 频域动态卷积（FDConv, CVPR 2025）
F_FS, F_C = Split_γ(F_init)                       # 通道划分：γC 进精炼支路，(1−γ)C 直通
F1, F2 = Conv1×1(F_FS)                            # 两份拷贝
F_S = MKSRU(F1)                                   # 多核空间精炼单元
F_F = FPU(F2)                                     # 频率处理单元（借自 SFS-Conv, CVPR 2025）
[A_S, A_F] = Softmax(GAP(Cat[F_S, F_F]))          # 通道级注意力
F̂_S = α·A_S·F_S;  F̂_F = β·A_F·F_F                # α、β 为可学习标量
F_out = Conv1×1(Cat[F̂_S + F̂_F, Conv3×3(F_C)])
```

- **MKSRU**（论文自设计）：DWConv 5×5 → 按通道 {1/2C→DW7×7 d3，3/8C→DW7×7 d2，1/8C→恒等} 切分（Eq.8）→ concat → SiLU(Conv1×1(X̂)) ⊙ SiLU(Conv1×1(Conv3×3(X)))（Eq.9，乘性门控，无显式残差）。本质是廉价的多尺度结构聚合（全部 depthwise）。
- **FPU**：rFFT → 频域 1×1 conv → irFFT，全局频谱混合。
- γ=0.5 最优（消融：γ=0 明显掉点 → 精炼支路必要；γ>0.5 轻微退化）。

**SFS-FF（Spatial-Frequency Shunt Feature Fusion）**——论文 §3.2，Eq.10 签名只匹配**自底向上（PAN）方向**：替换 PAN 融合对；FPN 无论文定义、保持基线。签名：本级特征 F_in (C×H×W) 与相邻细尺度特征 F̂_in (C×2H×2W)：

```
X = Conv1×1(Cat[F_in, FSD-Down(F̂_in)])            # 对齐后拼接过 1×1（2C → C）
F_S(C/2) → rFFT2 → Conv3×3 → irFFT2               # 空间表征走频域全局滤波
F_F(3C/8) → SRM                                   # 频率表征走空间精炼（交叉设计）
F´(C/8)  → 恒等残差
F_out = Conv1×1(Cat[F_S, F_F, F´])
```

注意"交叉"语义：空间切分出来的支路用**频域算子**处理（全局性），频率切分出来的支路用**空间算子**处理（局部性）。

**FSD-Down（Frequency-Spatial Dynamic Downsampling）**——论文 §3.3，替换 stride-2 下采样。基于 HWD 改造：

```
DWT(F_in) → [LL, LH, HL, HH]                      # Haar 小波，各 H/2×W/2，信息无损（正交变换）
F_g = Conv1×1(GConv(F_in))                        # 并行支路：分组卷积 k2/s2/g=C → 1×1 → 4C
F_CA = σ(GAP(Cat[DWT(F_in), F_g]))                # Eq.12：8C 共享门（前 4C 门 DWT 支路，后 4C 门 conv 支路）
F_out = Conv1×1(ρ·F_CA[:4C]⊙DWT(F_in) + F_CA[4C:]⊙F_g)   # Eq.13：ρ 为可学习标量，单一输出投影
```

Haar DWT 是无损正交变换——高频子带（LH/HL/HH）原样保留，由通道注意力自适应调权，这是它对 APs 增益的直接来源。

#### 1.1.4 消融与训练配方（【论文】§4）

| 配置 | AP (VisDrone test) | AP50 | APs |
|---|---|---|---|
| DEIMv2-L baseline | 24.7 | 43.3 | 14.2 |
| + DBFSAF | 29.7 | 49.7 | 19.8 |
| + DBFSAF + SFS-FF | 30.3 | — | — |
| + DBFSAF + SFS-FF + FSD-Down（完整） | **31.1** | **52.3** | **21.0** |

训练配方：100 epochs、batch 64、8×H200、AdamW lr 5e-4 / wd 1.25e-4、800×800；DEIMv2 标准增广 + MixUp/Mosaic/CopyBlend；dense O2O 前 50% epochs 后关闭；最后 2 epochs 关闭全部增广。**全部三个模块都是纯结构组件，不引入任何辅助损失**——这是与 Dome 移植（分册 13）最大的工程差异：无需动 criterion/decoder/评测链路。

#### 1.1.5 批判性评估（影响集成预期）

1. **DBFSAF 独占 +5.0 AP，对一个融合模块异常大**。supplement 提到他们还测了"四层多尺度融合的增强版 baseline"，暗示部分增益可能来自比 stock DEIMv2-L 更多的融合层数/容量，而非纯频域建模。**不可把 +6.4 当作本仓库移植的预期收益**（§1.3 给出诚实判据）。
2. SFS-FF（+0.6）和 FSD-Down（+0.8）的边际增益是叠加在 DBFSAF 之上的顺序消融，无排列组合，模块间可能有次序依赖。
3. 参数量/FLOPs/延迟对比在 supplement 且未随主文放出——三个模块的总开销不可精确预知（§4 给出本仓库的逐模块参数估算）。
4. 论文的 baseline 是 DEIMv2-L（双分支 HGNetv2+DINOv3，4 层金字塔）；本仓库主战变体是 ConvNeXt-tiny 单分支（4 层）与 ViT+STA（3 层）——**backbone 拓扑不同，增益不可平移**，这是集成方案按模块拆开的根本原因。

### 1.2 集成范围

**做**：将 FSD-Down / SFS-FF / DBFSAF 三个模块移植进 LT-DETR 家族，作为**可选频率分支**（一个配置开关组），可按模块独立启停（消融需要）；首战场景 = VisDrone 800px：

- ConvNeXt-tiny 4 层金字塔（`scripts/visdrone_ltdetr_800.py` 基线）→ 可用 **FSD-Down + SFS-FF**（encoder 侧）；
- ViT+STA 变体（`use_sta=True` 家族）→ 额外可用 **DBFSAF**（backbone 双分支融合点）。

**不做（非目标）**：

- 不改动 DEIMv2 任务模型的任何行为（HF 包装，见 §2.2）；
- 不在 ConvNeXt/HF backbone 内部替换下采样（HF 内部不可达；ViT 的语义 tap 分辨率对齐是 bilinear 上采样方向，非论文批评的下采样）；
- 不引入论文的 DEIMv2 训练配方（dense O2O 调度、CopyBlend、100ep batch64）——沿用引擎既有 flat-cosine/multistep + 停增广窗口契约（与分册 13 D9 同理；配方差异记为 F9）；
- 不引入 pywt 依赖（Haar 用固定权重 conv 实现，§4.1）；
- 不复刻完整 DEIMv2-L 双分支 backbone（HGNetv2 wrapper、4 层 P2–P5 融合链）——那是独立的"原生 DEIMv2"工程，不属于本分册。

### 1.3 预期收益与判据（诚实版）

论文增益锚定在 DEIMv2-L 双分支 + 100ep 大配方上；本仓库 ConvNeXt-tiny + 自有配方的合理预期按模块分解：

| 模块 | 论文边际增益 | 本仓库预期 | 成功判据（800px，与孪生基线比） |
|---|---|---|---|
| FSD-Down | +0.8（叠加上位） | +0.3 ~ +0.8（小目标直接受益，机制最独立） | AP ≥ +0.3 且 APs 不降 |
| SFS-FF | +0.6 | +0.2 ~ +0.6（且参数量显著低于被替换的 CSPRepLayer） | AP ≥ +0.2 或步时延下降 ≥ 10% 且 AP 持平 |
| DBFSAF | +5.0（疑有容量混杂） | 不可外推，按探索臂对待 | AP ≥ +0.8 才保留；否则降级为负结果记录 |

组合判据：三模块全开（宿主允许的范围内）总增益 ≥ +1.0 AP、步时延增幅 ≤ 15%、`deploy()` 导出后 AP 差已量化。**任一模块独立消融为负即默认关闭该开关**（三个独立开关正是为此设计）。

---

## 2. 两套架构的现状对照

### 2.1 宿主选型：为什么是 LT-DETR 而不是 DEIMv2

| 维度 | FSDC-DETR 论文 | 本仓库 DEIMv2 | 本仓库 LT-DETR | 结论 |
|---|---|---|---|---|
| 模型本体 | 官方原生实现 | **HF `Deimv2ForObjectDetection` 包装**（【VTrain】`deimv2_object_detection/task_model.py:4-24,215`，前向整体转发，criterion 也在 HF 内） | 原生组装：backbone wrapper → HybridEncoder → D-FINE/RT-DETRv2 decoder（【VTrain】`ltdetr_object_detection/task_model.py:330-448`） | DEIMv2 做模块级集成 = 改 HF 库内部或 monkey-patch，违背框架"转发而非重实现"策略；LT-DETR 是唯一原生宿主 |
| 双分支融合点 | HGNetv2 + DINOv3 concat（P2–P5 四层） | （库内） | **ViT 语义 + STA CNN 细节金字塔 concat + 1×1 + BN**（【VTrain】`backbone_wrappers.py:252-276`）——与论文批评的"concat+1×1 盲融合"**结构同位** | DBFSAF 落点（仅 vit 家族 use_sta=True） |
| encoder 融合块 | SFS-FF 替换 PAN 融合对（Eq.10，FPN 无论文定义） | （库内） | FPN/PAN 融合 = `CSPRepLayer`（【VTrain】`hybrid_encoder.py:96-125,257-273`） | SFS-FF 落点 = PAN 融合对（G3/F1） |
| 下采样点 | 全部尺度转换 | （库内） | PAN `downsample_convs` stride-2 conv（【VTrain】`hybrid_encoder.py:264-273`）；ConvNeXt 内部下采样在 HF 不可达；ViT 语义 tap 为 bilinear 对齐 | FSD-Down 落点 = PAN（F2） |
| 金字塔 | 4 层（P2–P5） | 2–3 层 | 3 层 [8,16,32] 或 ConvNeXt 4 层 [4,8,16,32]（`pyramid_levels=4`，分册 13 D1 已闭环） | SFS-FF 对 3/4 层均适用 |
| 参考代码 | 未放出 | — | — | 模块按论文文本 + 上游公开参考实现（FDConv/SFS-Conv/HWD）移植，歧义入 §5 |

### 2.2 DEIMv2 路径的明确排除

DEIMv2 任务模型的模型本体、hybrid encoder、双分支 backbone、criterion 全部在 `transformers >= 5.9` 库内（【VTrain】`deimv2_object_detection/task_model.py:4-24` 的框架策略陈述）。在其上做 FSDC 集成有三条路，均被否决：

1. **monkey-patch / 子类化 HF 内部模块**：随 transformers 版本漂移碎裂，且该任务模型的权重来源（HF 格式转换 checkpoint）与内部结构强绑定；
2. **原生重实现 DEIMv2**（含 HGNetv2 wrapper、双分支、DEIM criterion）：一个独立的整模型工程，规模数倍于本分册，且与本仓库"DEIMv2 = HF 基线锚点"的定位冲突；
3. **只在 LT-DETR 上借用了 DBFSAF 思想**：即本分册实际方案（F3）——`ViTDetectionBackbone` 的 ViT+STA 融合与 DEIMv2 的双分支融合结构同位，是 DBFSAF 在本仓库的诚实映射。

FSDC-DETR 作为**对比数字**（VisDrone 31.1 AP / 21.0 APs @800px, maxDets=300）进入基线分析，不做本地复现；评测协议差异（maxDets=300 vs 本仓 `eval_max_dets=500`，【VTrain】`scripts/visdrone_ltdetr_800.py:158`）在引用时注明。

### 2.3 移植主线结论

1. **三个模块全部是"encoder/backbone 内插件"**：不触碰 decoder、criterion、matcher、评测、CDN——改动面 = 1 个 encoder 子类 + 1 个 backbone wrapper 可选参数 + config/接线。比分册 13（跨 encoder/decoder/criterion/denoising/导出五个面）小一个量级。
2. **无辅助损失** → `train_model.py` 零改动（优化器分组自动覆盖新参数：新模块挂在 `self.model.encoder`/`self.model.backbone` 下，落在 detector/backbone 既有参数组语义内，见 §4.6）。
3. **移植顺序按依赖链**：FSD-Down（无依赖、最独立、预期收益最高）→ SFS-FF（依赖 FSD-Down 做 PAN 对齐）→ DBFSAF（依赖 FDConv 移植，风险最高，独立 PR 隔离）。

---

## 3. 总体设计

### 3.1 模块拓扑

**新增文件**（共享组件层 + ltdetr 任务层，风格对齐既有"faithful port + 设计文档注释"惯例）：

```
src/autovision/_task_models/object_detection_components/
├── wavelet_downsample.py   # HaarDWT2d（可学习小波，Haar 初始化；G5）+ FSDDown（Eq.12-13 形式）
├── sfs_ff.py               # SpatialRefineModule（SRM 本地最小版）+ SFSFF（Eq.10-11：内嵌 FSD-Down，仅落 PAN）
├── fdconv.py               # FDConv 移植（Fourier-disjoint weights 等三组件，§4.4）
└── dbfsaf.py               # MKSRU + FPU + DBFSAF

src/autovision/_task_models/ltdetr_object_detection/
└── fsdc_encoder.py         # FsdcHybridEncoder(HybridEncoder)：PAN 接线（SFS-FF 整对替换 / FSD-Down 独立臂）

tests/test_ltdetr_fsdc.py             # 全部单元/性质测试（§7）
scripts/visdrone_ltdetr_fsdc_800.py   # ConvNeXt-tiny 孪生实验脚本（§8）
```

**修改文件**（全部向后兼容，新增参数默认关闭）：

| 文件 | 改动 |
|---|---|
| `ltdetr_object_detection/config.py` | 新增 `LTDETRFsdcArgs`；`LTDETRObjectDetectionTrainArgs` 增加 `fsdc: LTDETRFsdcArgs \| None = None` |
| `ltdetr_object_detection/task_model.py` | `__init__` 增加 `fsdc` 参数（dict 强转 + 组合守卫）；encoder 分支构建 `FsdcHybridEncoder`；backbone 分支向 `ViTDetectionBackbone` 传 `fsdc_fusion` |
| `ltdetr_object_detection/backbone_wrappers.py` | `ViTDetectionBackbone` 增加可选 `fsdc_fusion: bool`（默认 False；True 时以 DBFSAF 替换每层的 `convs[i]+norms[i]`） |
| `ltdetr_object_detection/__init__.py` | 导出 `FsdcHybridEncoder`、`DBFSAF`、`FSDDown`、`SFSFF`（可选） |

**零改动**：`train_model.py`（无新损失/日志）、`dfine_criterion.py`/`rtdetrv2_criterion.py`、decoder 家族、denoising、decode、引擎。

### 3.2 数据流（三态）

**训练/推理态**（fsdc 全开，vit+STA 变体示例；ConvNeXt 变体无 DBFSAF 段）：

```
x(B,3,H,W)
   │
   ▼ ViTDetectionBackbone（fsdc_fusion=True）
   sem_feats = ViT 多尺度 taps ─┐
   detail_feats = STA 金字塔  ─┴─ 每层 l ∈ {8,16,32}：
       Cat[sem, detail] → DBFSAF(C_in→hidden) → c_l        # 替换原 convs[l]+norms[l]
   │
   ▼ FsdcHybridEncoder(feats)
   proj = input_proj(feats)                                 # 既有
   proj = _run_aifi(proj)                                   # 既有，不动
   _run_fpn_pan：
       FPN（自顶向下）：upsample_feat = nearest 上采样       # 既有
                      inner = CSPRepLayer(Cat[upsample_feat, feat_low])   # 基线保持（G3：FPN 无论文定义）
       PAN（自底向上）：out = SFSFF(f_in=feat_low, f_hat_in=feat_height)
                        # Eq.10-11：SFSFF 内嵌 FSD-Down 对齐细尺度，整对替换 (downsample_convs, pan_blocks)
                        # use_sfs_ff=False 而 use_fsd_down=True 时：仅 down = FSDDown(feat_low)，融合保持 CSPRepLayer
   ► outs（3/4 层 hidden_dim 金字塔，契约与基线完全一致）
   │
   ▼ DFINEv2Decoder / RTDETRv2Decoder（既有，零改动）
```

**导出态**：三个模块均为静态形状、无数据依赖控制流——`torch.export` 直接可走；FFT 算子的 ONNX 落地按 §4.7 验证，失败则 `convert_to_deploy` 提供恒等降级臂并量化 AP 差。`FsdcHybridEncoder`/`DBFSAF`/`FSDDown` 各自实现 `convert_to_deploy`（FFT 降级开关），递归由 `TaskModel.deploy()` 天然覆盖（与分册 13 §4.8 同机制）。

### 3.3 开关与兼容策略

- 总开关 = `LTDETRObjectDetectionTrainArgs.fsdc`（`None` = 完全关闭）：**不构建**任何 fsdc 模块（state_dict 无新键、参数组无新条目、RNG 流不偏移、导出图同基线）。
- 三个子开关独立：`use_fsd_down` / `use_sfs_ff` / `use_dbfsaf`，支持逐模块消融（§8）。叠加语义：`use_sfs_ff=True` 时 PAN 融合对整对替换、FSD-Down 内嵌其中（Eq.10），独立 FSDDown 臂不再单独出现；`use_fsd_down=True` 单开 = 仅换 PAN 下采样、融合保持 CSPRepLayer（§8 b 臂）。
- 守卫（对齐既有组合守卫风格，【VTrain】`task_model.py:215-238,262-267`）：
  - `use_dbfsaf` 要求 **vit 家族且 `use_sta=True`**（ConvNeXt/ECViT 无双分支融合点 → 配置期 ValueError，F3）；
  - **`fsdc` 与 `density` 互斥**（v1）：两者都以 encoder 子类为载体，组合需要第三个 `DomeFsdcHybridEncoder`，v1 不构建该组合（F6）；
  - `fsdc` 与 `ssl`/`token_drop` 正交（ssl 挂 decoder 端、token_drop 在 ViT token 级、fsdc 在特征融合级），允许组合并加前向形状回归测试。

---

## 4. 详细设计

### 4.1 HaarDWT2d 与 FSDDown（`object_detection_components/wavelet_downsample.py`）

**移植源**：【论文】§3.3（FSD-Down）；HWD（Pattern Recognition 2023）的实现方式。

```python
class HaarDWT2d(nn.Module):
    """2D Haar DWT as a fixed-weight grouped conv (no pywt dependency).

    输入 (B, C, H, W) → 输出 (B, 4C, H/2, W/2)，通道顺序 [LL, LH, HL, HH]。
    权重为不可学习 buffer（正交 Haar 滤波器组，stride-2 depthwise conv）：
      LL = ¼(x00+x01+x10+x11)   LH = ¼(x00+x01−x10−x11)
      HL = ¼(x00−x01+x10−x11)   HH = ¼(x00−x01−x10+x11)
    奇数边长：右/下侧 F.pad 零填充一行/列（ceil 语义，与 FSD-Down 的
    ⌈H/2⌉ 输出一致——对齐【VTrain】backbone_wrappers.py:53-55 的 _ceil_half 链）。
    """
    # register_buffer("filters", ...)  形状 (4C, 1, 2, 2)，groups=C
    # 卷积实现 → 纯 conv，torch.export / ONNX 零障碍

def haar_idwt(ll, lh, hl, hh) -> Tensor:
    """逆变换（仅测试用）：完美重构断言的参照实现。"""

class FSDDown(nn.Module):
    """Frequency-Spatial Dynamic Downsampling（论文① Eq.12-13 精确形式，规格源：分册 22 §4.2）。"""
    def __init__(self, channels: int, freeze_wavelet: bool = False):
        self.dwt = HaarDWT2d(channels, learnable=not freeze_wavelet)   # (B,4C,⌈H/2⌉,⌈W/2⌉)
        self.gconv = nn.Conv2d(channels, channels, 2, 2, groups=channels, bias=False)
        self.conv_path = nn.Conv2d(channels, 4*channels, 1, bias=False)   # Eq.13 的 Conv1×1(GConv) → 4C
        self.out_proj = nn.Conv2d(4*channels, channels, 1)                # Eq.13 单一输出投影
        self.rho = nn.Parameter(torch.ones(1))               # 可学习标量 ρ
    def forward(self, x):
        dwt = self.dwt(x)                                    # 4C
        cp = self.conv_path(self.gconv(x))                   # 4C
        f_ca = torch.sigmoid(gap(torch.cat([dwt, cp], 1)))   # Eq.12：8C 共享门
        gate_d, gate_c = f_ca.chunk(2, dim=1)                # 前 4C 门 DWT 支路，后 4C 门 conv 支路
        return self.out_proj(self.rho * gate_d * dwt + gate_c * cp)   # Eq.13
```

**设计要点**：

- **Eq.12–13 精确形式**（E5，规格源：分册 22 §4.2）：门向量是 **8C 共享**（DWT 4C + conv 支路 4C 拼接过 σ），chunk 后分别调制两支；输出是单一 `Conv1×1(4C→C)`。21 册 v1.0 的"两支独立 C 通道投影"备选方案作废。
- DWT 支路输出 4C 全子带（LL+LH+HL+HH 各 C）——高频信息以可学习权重进入主路，这是与"平均池化"的本质区别。
- **可学习小波（G5）**：滤波器以正交 Haar 基初始化为 `nn.Parameter`（论文①摘要 "learnable wavelet decomposition"）；`freeze_wavelet=True` 回退固定 Haar buffer（消融臂）。
- 奇数尺寸填充发生在 DWT 内部（零填充引入的边界效应 ≤ 1 像素行/列，HWD 同款处理）。
- 参数量：`conv_path 4C² + out_proj 4C² + gconv/小波 ≈ 8C²`；被替换物（stride-2 3×3 ConvNormLayer）= `9C²`——大致持平略减（减参收益主要来自 SFS-FF，§4.2）。

### 4.2 SFSFF 融合块与 SRM（`object_detection_components/sfs_ff.py`）

**移植源**：【论文】§3.2（Eq.10-11）；SRM 按 DEIMv2 的 Spatial Refine Module 最小复刻（【VTrain】仓库内无同构模块，已 grep 确认 → 本地实现，F5）。

```python
class SpatialRefineModule(nn.Module):
    """DEIMv2 SRM 最小版：DWConv3×3 → BN → ReLU → PW 1×1 → BN。"""
    # 参数量 ≈ 9C + C²

class SFSFF(nn.Module):
    """Spatial-Frequency Shunt Feature Fusion（论文① Eq.10-11，规格源：分册 22 §4.3）。

    输入契约 = Eq.10 原生签名：(C,H,W) 本级粗特征 + (C,2H,2W) 相邻细尺度特征；
    细尺度对齐由内嵌 FSD-Down 承担（G3/F1：仅落 PAN，FPN 保持基线 CSPRepLayer）。
    """
    def __init__(self, channels: int, freeze_wavelet: bool = False):
        self.down = FSDDown(channels, freeze_wavelet)                 # Eq.10 内嵌对齐：(2H) → (H)
        self.proj_in = ConvNormLayer(2*channels, channels, 1, 1)      # Eq.10：Cat → 1×1
        self.fft = FrequencyGlobalBranch(channels//2)                 # F_S(C/2) × 频域算子（交叉）
        self.srm = SpatialRefineModule(3*channels//8)                 # F_F(3C/8) × 空间算子（交叉）
        self.proj_out = ConvNormLayer(channels, channels, 1, 1)
    def forward(self, f_in, f_hat_in):                                # (C,H,W) + (C,2H,2W)
        x = self.proj_in(torch.cat([f_in, self.down(f_hat_in)], dim=1))   # Eq.10
        x_f, x_s, x_id = x.split([3*channels//8, channels//2, channels//8], dim=1)
        return self.proj_out(torch.cat([self.srm(x_f), self.fft(x_s), x_id], dim=1))   # Eq.11

class FrequencyGlobalBranch(nn.Module):
    """rFFT2 → 3×3 conv on (real‖imag) → irFFT2；内部强制 fp32（F7）。"""
    # fft_fp32(x): x.float() → rfft2 → cat[re,im] → conv3×3 → irfft2 → .to(x.dtype)
```

**设计要点**：

- **通道切分比例硬编码论文值** {C/2, 3C/8, C/8}：`hidden_dim=256` 时 = {128, 96, 32}，整除无余数；`hidden_dim=192/224/384` 亦整除。守卫：`assert hidden_dim % 8 == 0`（config 期校验，LT-DETR 全部变体 hidden ∈ {64,112,128,192,224,256,384,768,1024,1536} 均满足）。
- **交叉语义保留**：F_S 支路（半通道）走频域全局、F_F 支路走 SRM 局部——与论文 Fig.4 一致；不"纠正"为直觉命名。
- **FFT 的 fp32 island（F7）**：`torch.fft` 系列不参与 autocast，bf16 输入下精度风险集中在 radix 计算与 `cat[re,im]` 拼接；对齐仓内 fp32 island 惯例（【VTrain】`dfine_decoder.py:122-125` 等），分支内 `.float()` 进、`.to(dtype)` 出。800px 下 S3 尺度 100×100、S4 50×50、S5 25×25，FFT 开销可忽略（每次 rfft2 在 100×100×128 上 < 0.1ms 量级）。
- 参数量对比（每处，C=256）：被替换的 PAN 融合对 `(stride-2 conv + CSPRepLayer(2C→C, 3×RepVgg))` ≈ 2.8M；SFSFF（含内嵌 FSD-Down ≈ 0.52M）≈ 1.3M——**约 2× 减参**，且频域支路带来全局感受野补偿。这是"轻量化收益"判据（§1.3）的来源：SFSFF 若 AP 持平但步时延下降，也算正结果。
- 与 CSPRepLayer 的行为差异：无多 bottleneck 深度聚合 → 表达容量下降由频域全局支路补偿（这正是论文的设计主张）。

### 4.3 FsdcHybridEncoder 接线（`ltdetr_object_detection/fsdc_encoder.py`）

对齐 `DomeHybridEncoder` 的子类模式（【VTrain】`dome_encoder.py:50-137`）：继承 `HybridEncoder`，在 `__init__` 按开关**替换**（而非附加）FPN/PAN 的融合与下采样模块，`forward` 复用父类 `_run_aifi`、覆写 `_run_fpn_pan`。

```python
class FsdcHybridEncoder(HybridEncoder):
    def __init__(self, in_channels, *, hidden_dim, dim_feedforward, expansion=1.0,
                 depth_mult=1.0, use_encoder_idx=None, fsdc: LTDETRFsdcArgs):
        super().__init__(...)                                 # 全量基线构建（含被替换模块）
        # FPN：论文未定义替换（G3/F1）——fpn_blocks 保持基线 CSPRepLayer，逐行不动
        if fsdc.use_sfs_ff:
            # Eq.10-11：PAN 融合对 (downsample_convs[i], pan_blocks[i]) 整对替换，
            # 细尺度对齐由 SFSFF 内嵌的 FSD-Down 承担（§4.2）
            self.pan_sfsff = nn.ModuleList(
                [SFSFF(hidden_dim, freeze_wavelet=fsdc.freeze_wavelet)
                 for _ in range(len(self.pan_blocks))]
            )
        elif fsdc.use_fsd_down:
            # 独立臂（§8 b 臂）：仅换 PAN 下采样，融合保持 CSPRepLayer
            for i, ds in enumerate(self.downsample_convs):
                self.downsample_convs[i] = FSDDown(hidden_dim, freeze_wavelet=fsdc.freeze_wavelet)
        self.fsdc_args = fsdc
        self._fft_deploy_fallback = False

    def _run_fpn_pan(self, proj_feats):                       # 覆写：仅 PAN 段分叉
        # FPN：逐行等同 HybridEncoder._run_fpn_pan 的 top-down 段
        # PAN：use_sfs_ff → out = self.pan_sfsff[idx](feat_low, feat_height)   # Eq.10-11，跳过 downsample_convs
        #      else       → downsample_feat = self.downsample_convs[idx](feat_low)   # 基线（或 FSDDown 臂）
        ...

    def convert_to_deploy(self) -> None:
        # FFT ONNX 落地验证失败时的降级臂（§4.7）：FrequencyGlobalBranch 置旁路
```

**必须成立的不变式**（进 §7 测试）：

1. `use_fsd_down` 单开：`downsample_convs[i]` 输出形状与 stride-2 conv 逐尺寸一致（800 链：400→200→100 / 4 层 200→100→50→25，含 ceil 语义）；替换前后 encoder 输出金字塔 `[(C,H_l,W_l)]` 契约不变。
2. `use_sfs_ff`：FPN 段与基线逐比特一致（CSPRepLayer 未动）；PAN 段输出 `(B, hidden, H, W)` 与基线同形，decoder 侧零感知。
3. 两开关全关但 `fsdc` 非 None：encoder 行为与基线逐比特一致（允许空壳类存在，方便脚本统一接线）。

**与 Dome 的组合**：v1 互斥（F6）。实现上 `task_model.py` 的 encoder 构建分支顺序为 `density → dome encoder / fsdc → fsdc encoder / else → 基线`，`fsdc.density 互斥守卫` 放在 §3.3 所述的配置期。

### 4.4 FDConv 移植（`object_detection_components/fdconv.py`）

**移植源**：FDConv（CVPR 2025）官方仓库 `Linwei-Chen/FDConv`（MIT），移植其三组件：

1. **FDW（Fourier-Disjoint Weight）**：把 `k×k` 卷积核组在 Fourier 域构造为频带不相交的多组滤波器（参数存频域、DCT/FFT 重建空间核），实现"一个 conv 的参数在频率维天然分带"；
2. **KSM（Kernel Spatial Modulation）**：逐采样点的空间调制门；
3. **FBL（Frequency Band Learning）**：频带注意力。

**v1 范围决策（F4）**：DBFSAF 内 FDConv 出现的位置是 `Cat[F_L,F_H] (2C_in) → C_out` 的单次投影卷积。v1 移植 **FDW + FBL**（论文明确引用的频率多样性来源），KSM 作为可选（`use_ksm: bool = False`）——理由：KSM 的逐点调制是容量组件而非频率组件，v1 消融需要的是"频域建模"本身；FDConv 官方消融显示 FDW 是主贡献组件。若 PR-3 中 FDW 重建开销实测过高（> 2ms/层），降级臂 `fdconv_lite = Conv3×3`（明确标注的负控臂，量化 FDW 贡献）。

- 参数量：FDW 以 `num_param` 组频带存储，典型配置下 ≈ 1–2× 等效 `3×3` conv（C=384、2C_in→C_out 处 ≈ 0.6–1.2M）；PR-3 落地后以实测写入本表。
- 移植等价性测试：与官方实现（vendored 快照）在固定输入上前向输出 allclose（fp32、rtol 1e-5）。

### 4.5 DBFSAF 与 backbone 融合点接线（`object_detection_components/dbfsaf.py` + `backbone_wrappers.py`）

```python
class MKSpatialRefineUnit(nn.Module):
    """MKSRU（论文① Eq.7-9，规格源：分册 22 §4.1）：DW5×5 → {1/2C→DW7×7 d3，3/8C→DW7×7 d2，
    1/8C→恒等}（Eq.8）→ concat → SiLU(Conv1×1(X̂)) ⊙ SiLU(Conv1×1(Conv3×3(X)))（Eq.9，乘性门控）。"""

class FPU(nn.Module):
    """频率处理单元（SFS-Conv, CVPR 2025）：fp32 rFFT2 → 1×1 → irFFT2 + 全局门。"""

class DBFSAF(nn.Module):
    """论文① §3.1 全流程（Eq.1-6，规格源：分册 22 §4.1）；in=Cat[sem,detail] 宽，out=hidden_dim。"""
    def __init__(self, in_channels, out_channels, gamma=0.5, use_ksm=False):
        self.fdconv = FDConv(in_channels, out_channels, ...)
        self.gamma = gamma
        c_fs = int(out_channels * gamma); c_c = out_channels - c_fs
        self.pfsr_in_s = nn.Conv2d(c_fs, c_fs, 1)            # Eq.2-3：两支各自 1×1（"两份拷贝"）
        self.pfsr_in_f = nn.Conv2d(c_fs, c_fs, 1)
        self.mksru = MKSpatialRefineUnit(c_fs)
        self.fpu = FPU(c_fs)
        self.alpha = nn.Parameter(torch.ones(1)); self.beta = nn.Parameter(torch.ones(1))
        self.bypass = ConvNormLayer(c_c, c_c, 3, 1)          # Conv3×3(F_C)
        self.proj_out = ConvNormLayer(out_channels, out_channels, 1, 1)
    def forward(x):                                          # x = Cat[sem, detail]
        f_init = self.fdconv(x)                              # Eq.1
        f_fs, f_c = torch.split(f_init, [c_fs, c_c], dim=1)  # Partial_γ
        f_s = self.mksru(self.pfsr_in_s(f_fs))               # Eq.2
        f_f = self.fpu(self.pfsr_in_f(f_fs))                 # Eq.3
        a = torch.softmax(gap(torch.cat([f_s, f_f], 1)), dim=1)   # Eq.4：2C 通道维整体 softmax
        f_hat_s, f_hat_f = self.alpha * a[:, :c_fs] * f_s, self.beta * a[:, c_fs:] * f_f   # Eq.4-5
        return self.proj_out(torch.cat([f_hat_s + f_hat_f, self.bypass(f_c)], 1))          # Eq.6
```

**`ViTDetectionBackbone` 接线**（【VTrain】`backbone_wrappers.py:252-276`）：

- 现状：每层 `Cat[sem, detail] → self.convs[i] (1×1, C_in→hidden) → self.norms[i] (BN)`。
- `fsdc_fusion=True` 时：构建 3 个 `DBFSAF(C_in[i], hidden)`（`C_in[i] = embed_dim + conv_inplane*{2,4,4}`，沿用现有通道表 `backbone_wrappers.py:169-181`），替换 `convs[i]+norms[i]` 的职责（BN 并入 DBFSAF 的 ConvNormLayer）。
- **RNG 纪律**：fsdc 模块在 backbone wrapper 内构建会平移后续模块的初始化随机流——与 dome 开启时的行为一致（分册 13 已声明"开启即换架构"），不承诺 fsdc-on 与基线 init 逐比特相同；**fsdc-off 必须逐比特相同**（fsdc=None 时零构建，天然满足）。
- 守卫：`fsdc.use_dbfsaf=True` 且 `spec.family != "vit"` 或 `spec.use_sta=False` → 配置期 ValueError（§3.3）。

### 4.6 配置面（`config.py`）

```python
class LTDETRFsdcArgs(PydanticConfig):
    """FSDC-DETR 频率-空域协同模块（分册 21）。None = 关闭。"""
    model_config = {"extra": "forbid"}

    use_fsd_down: bool = True          # PAN 下采样 → FSDDown（use_sfs_ff=False 时的独立臂）
    use_sfs_ff: bool = True            # PAN 融合对 → SFSFF（Eq.10-11，内嵌 FSD-Down；FPN 保持基线）
    freeze_wavelet: bool = False       # G5：False = 可学习小波（Haar 初始化），True = 固定 Haar buffer
    use_dbfsaf: bool = False           # ViT+STA 融合点 → DBFSAF（ConvNeXt 拒绝）
    dbfsaf_gamma: float = 0.5          # 论文消融最优
    fdconv_use_ksm: bool = False       # FDConv KSM 组件（F4）
    fdconv_lite: bool = False          # 负控臂：FDConv → Conv3×3
    fft_export_fallback: bool = False  # deploy 时 FFT 分支置旁路（§4.7）
```

`LTDETRObjectDetectionTrainArgs` 增加 `fsdc: LTDETRFsdcArgs | None = None`。`task_model.py` 的 `__init__` 遵循既有模式：dict → args 强转（对齐 `density/ssl/token_drop` 的强转，【VTrain】`task_model.py:223-224,244-245,280-281`）；`init_args` 快照按 `LTDETRDensityArgs.model_dump()` 同款转 plain dict（【VTrain】`task_model.py:166-171`，weights_only 安全加载器兼容）。

**优化器/EMA 自动覆盖**：fsdc 模块分别挂在 `self.model.backbone`（DBFSAF）与 `self.model.encoder`（FSDDown/SFSFF）下。注意 DBFSAF 属于 backbone wrapper 内的**随机初始化连接件**——按 `ViTDetectionBackbone` 既有语义（【VTrain】`backbone_wrappers.py:8-11`），STA/投影层按 detector LR 训练而非 backbone LR；**DBFSAF 归入 connector 组**（与 STA 同组），实现上确认 `get_optimizer` 的参数分组判定依据（`backbone_model` 属性边界）天然满足——DBFSAF 不是 `backbone_model` 的子模块，自动落 detector 组，无需改 `train_model.py`。

### 4.7 导出契约

与 Dome 不同（动态查询不可导出），**fsdc 全部模块静态形状、无数据依赖控制流**，唯一风险点是 FFT 算子的 ONNX 落地：

| 环节 | 结论 |
|---|---|
| `torch.export` 严格模式 | rfft2/irfft2 可 trace（纯函数算子），静态形状下无特化问题 |
| ONNX 算子 | FFT → ONNX DFT（opset ≥ 17）。**PR-4 实测**：若目标 runtime 链路（onnxruntime 版本）不支持 DFT 或数值不达标 → `convert_to_deploy` 将 `FrequencyGlobalBranch`/`FPU` 置恒等旁路（`fft_export_fallback=True`），语义损失在 §8 消融量化 |
| Haar DWT / FSDDown / SFSFF 非FFT支路 / DBFSAF 非FFT支路 | 纯 conv/BN/逐元素，零障碍 |
| `verify_onnx_export_outputs` | 既有校验自动覆盖（【VTrain】`task_model.py:690-710`） |

### 4.8 DDP / EMA / AMP / 引擎契约

| 项 | 结论 |
|---|---|
| DDP | 新参数全部参与前向（无条件分支跳过），`find_unused_parameters` 无新依赖；无逐图循环（对比 Dome MWAS），无 CPU-GPU 同步点 |
| AMP | 引擎 bf16 autocast 包住 `training_step`；仅 FFT 分支强制 fp32 island（F7）；Haar DWT 是线性 conv，bf16 下数值稳定（滤波器系数 ∈ {±¼}） |
| EMA | `ModelEMA` 整模型 deepcopy，自动纳入；`rho/alpha/beta` 标量参数一并 EMA |
| NaN 监控 | softmax(GAP) 在零输入下输出均匀分布（非 NaN）；sigmoid/FFT 均无奇异点 |
| 检查点 | fsdc 开关改变架构 → resume 须同配置；`extra=forbid` + init-args 快照机制已保证（与 density 同理，分册 13 §4.9） |
| 步时延 | 预期中性偏优：FSDDown/SFSFF 均比被替换物轻（§4.1/§4.2 估算）；PR-2 基准实测（800px/batch4/fwd+bwd），预算 ±10% |

---

## 5. 与论文原表述的差异清单

| # | 差异 | 论文现状 | 本仓库决策 | 理由 |
|---|---|---|---|---|
| F1 | SFS-FF 落点与契约 | Eq.10：SFS-FF 内嵌 FSD-Down 对齐 (C,2H,2W)→(C,H,W)，签名只匹配 PAN 方向；FPN 无论文定义 | **仅 PAN 整对替换 `(downsample_convs, pan_blocks)`**；FPN 保持基线 CSPRepLayer。21 册 v1.0 的"对齐外移 + FPN/PAN 双替换"方案作废（分册 22 §4.3 重核 Eq.10 后修订） | Eq.10 的 Down 内嵌是块定义的一部分，外移会改变感受野与参数共享；官方代码放出后第一个复核项 |
| F2 | FSD-Down 覆盖范围 | 论文替换**全部**尺度转换（含 backbone 内部下采样层） | 仅 encoder PAN 的 `downsample_convs`（【VTrain】`hybrid_encoder.py:264-273`） | ConvNeXt 内部下采样在 HF 内不可达；ViT 语义 tap 是 bilinear 上采样方向非下采样。STA 金字塔内部的 stride-2 conv（`backbone_wrappers.py:73-115`）是下一个候选点，列入 §9 开放问题 |
| F3 | DBFSAF 宿主 | DEIMv2 双分支（HGNetv2+DINOv3）P2–P5 四层 | `ViTDetectionBackbone` 的 ViT+STA 融合点，3 层；ConvNeXt/ECViT 不适用（配置期拒绝） | 结构同位（"concat+1×1 盲融合"），但层数与分支深度不同——增益预期已相应下调（§1.3） |
| F4 | FDConv 范围 | 论文引用完整 FDConv | 移植 FDW+FBL，KSM 可选默认关；另有 `fdconv_lite` 负控臂 | v1 消融需要的是频域建模本身；KSM 是容量组件 |
| F5 | SRM | DEIMv2 内置 SRM | 本地最小实现（DW3×3+BN+ReLU+PW+BN），11 行 | 仓库无同构模块（grep 确认）；行为以 DEIMv2 描述为准 |
| F6 | density × fsdc | 论文无此组合（其 baseline 无 Dome） | v1 配置期互斥 | 两者都需要替换 encoder；组合 = 第三个子类，留待两个分支各自验证收益后再决定是否值得 |
| F7 | FFT 精度 | 未说明（官方 fp16 AMP） | FFT 分支 fp32 island | 本引擎 bf16 autocast；对齐仓内 fp32 island 惯例 |
| F8 | 导出 | 论文未讨论导出 | FFT ONNX 落地 PR-4 实测；失败 → `convert_to_deploy` 恒等旁路 + AP 差量化 | 静态形状模块 `torch.export` 本身无障碍，仅 ONNX 算子生态问题 |
| F9 | 训练配方 | 100ep/batch64/CopyBlend/dense-O2O 调度 | 不移植，沿用本仓配方（孪生脚本只差 fsdc 开关） | 引擎契约（分册 13 D9 同理）；保证与既有全部基线可比 |

---

## 6. 实施计划（四个 PR，每个独立可合入）

### PR-1：小波下采样组件（不改变任何模型行为）

- 新增 `wavelet_downsample.py`（HaarDWT2d + haar_idwt 参照实现 + FSDDown）；
- 纯组件 PR，不接线——`task_model.py` 零改动。
- 测试：DWT 对固定输入 == 手工参照（rtol 1e-6）；`haar_idwt` 完美重构（偶数/奇数尺寸）；FSDDown 形状（800→400、644→322 链）；bf16/fp32 前向有限性；梯度非 NaN。
- **验收**：单测全绿；微基准 FSDDown vs stride-2 conv 步时对比表。

### PR-2：FsdcHybridEncoder（FSD-Down + SFS-FF 上线，ConvNeXt/ECViT 路径完整）

- 新增 `sfs_ff.py`、`fsdc_encoder.py`；`config.py` 增 `LTDETRFsdcArgs`（`use_dbfsaf=False` 冻结）；`task_model.py` 接线 + 守卫（density 互斥、hidden % 8）。SFS-FF 仅落 PAN（G3/F1）；`freeze_wavelet` 贯通。
- 测试：fsdc=None 逐比特基线（state_dict 键集合、前向输出、RNG 流）；两开关独立/组合的形状与输出契约；§4.3 三条不变式；checkpoint 往返。
- **验收**：单测全绿 + 全量回归；800px/batch4 步时延基准（预算 ±10%）；VisDrone 2k 步 smoke loss 收敛正常。

### PR-3：FDConv + DBFSAF（ViT+STA 路径）

- 新增 `fdconv.py`（vendored 移植 + 等价性测试）、`dbfsaf.py`；`backbone_wrappers.py` 增 `fsdc_fusion`；config 解冻 `use_dbfsaf` + 守卫。
- 测试：FDConv vs 官方快照 allclose；DBFSAF γ 切分/门控有限性/退化等价（γ=1、α=1、β=0 时的结构一致性）；`use_dbfsaf` + ConvNeXt → ValueError；`fsdc + ssl + token_drop` 三组合前向形状回归。
- **验收**：单测全绿；ViT 变体 800px smoke；参数量/步时增量记录进本文档 §4.4/§4.5。

### PR-4：导出验证 + 实验脚本 + 文档收口

- `convert_to_deploy` FFT 降级臂；`scripts/visdrone_ltdetr_fsdc_800.py`（ConvNeXt-tiny 孪生，复制 `visdrone_ltdetr_800.py`，唯一差异 = `fsdc` 字段；ViT 臂脚本按 §8 单列）；ONNX 导出 + `verify_onnx_export_outputs`；程序化孪生一致性测试（对齐分册 13 `TestScriptTwin` 模式）；本文档状态更新 + 实测数据回填。
- **验收**：fsdc 全开模型 ONNX 导出成功（或降级臂量化记录）；孪生测试通过。

---

## 7. 测试计划（`tests/test_ltdetr_fsdc.py`）

风格对齐 `tests/test_ltdetr_dome.py`（离线 tiny 变体 + 小图，全部 hermetic）：

| 用例 | 断言 |
|---|---|
| `test_fsdc_off_is_baseline` | `fsdc=None`：无 Fsdc* 模块、state_dict 键集合与基线一致、固定种子前向输出逐比特一致、RNG 流不偏移 |
| `test_haar_dwt_matches_reference` | 随机输入 DWT == 手工逐块计算（rtol 1e-6）；子带通道序 [LL,LH,HL,HH] |
| `test_haar_perfect_reconstruction` | `haar_idwt(DWT(x)) == x`（偶/奇尺寸，pad 区域除外） |
| `test_fsddown_shape_parity` | 800/644/1024 输入下 FSDDown 输出 == stride-2 conv 输出形状（ceil 链一致） |
| `test_fsddown_gradient_flow` | `freeze_wavelet=True` 时 DWT buffer 不接收梯度；False 时小波参数有梯度且初值 == Haar 基；`rho` 与两支 1×1 梯度非 NaN |
| `test_sfsff_channel_split` | 切分 {3C/8→SRM, C/2→FFT, C/8→恒等}；恒等支路直通；Eq.10 内嵌对齐 (C,2H,2W)→(C,H,W) 形状正确 |
| `test_encoder_contract` | 两开关独立/组合下 encoder 输出金字塔 `[(hidden,H_l,W_l)]` 与基线同形；FPN 段与基线逐比特一致（CSPRepLayer 未动）；4 层与 3 层金字塔都通过 |
| `test_dbfsaf_gating_finite` | softmax 门有限、α/β 梯度非 NaN；零输入下输出有限（均匀门） |
| `test_dbfsaf_gamma_split` | γ=0.5、C=256 时两支宽 128/128；γ 守卫拒绝 γ∉(0,1) |
| `test_fdconv_matches_official` | 固定输入 vs vendored 官方快照 allclose（fp32, rtol 1e-5） |
| `test_combination_guards` | fsdc+density → ValueError；use_dbfsaf+ConvNeXt → ValueError；use_dbfsaf+use_sta=False → ValueError；hidden % 8 != 0 → ValueError |
| `test_fsdc_ssl_tokendrop_stack` | fsdc + ssl(两模式) + token_drop 全组合前向形状与损失有限性 |
| `test_export_fft` | `deploy()` 后 `torch.export` 成功；FFT 旁路臂下导出成功且输出一致性校验通过；记录两臂 ONNX 数值差 |
| `test_checkpoint_roundtrip` | fsdc 全开训练 2 步 → 存/载 → loss 连续 |
| `TestScriptTwin` | `visdrone_ltdetr_fsdc_800.py` 与 `visdrone_ltdetr_800.py` 逐字段 diff == {fsdc}（对齐分册 13 模式） |

金标回归：`tests/golden/` 若覆盖 LT-DETR 前向，确认 fsdc 关闭路径无新差异。

---

## 8. VisDrone 训练配方与消融

**基座**：`scripts/visdrone_ltdetr_fsdc_800.py` —— 复制 `scripts/visdrone_ltdetr_800.py` 全部配方（ConvNeXt-tiny COCO safetensors、dfine、`pyramid_levels=4`、`num_queries=250`、hidden 256、800×800、batch 4 × accum 2、lr 1.4e-4 multistep、`eval_max_dets=500`、EMA），唯一差异：

```python
model_args = {
    ...,
    "fsdc": {"use_fsd_down": True, "use_sfs_ff": True},   # ConvNeXt 无 DBFSAF
}
```

**消融矩阵**（同一配方，唯一变量 = fsdc 配置；沿用现有基线脚本作为 a 臂，不重跑）：

| 组 | 变体 | fsdc | 目的 |
|---|---|---|---|
| a | ConvNeXt-tiny | off（现有基线） | 锚点 |
| b | ConvNeXt-tiny | `use_fsd_down` 单开（融合保持 CSPRepLayer） | FSD-Down 净贡献（预期最高置信） |
| c | ConvNeXt-tiny | `use_fsd_down + use_sfs_ff`（= 论文 Eq.10-11 完整形态，SFSFF 内嵌 FSD-Down） | encoder 全家桶；顺带度量 PAN 融合对→SFSFF 的参数/时延节省 |
| d | ConvNeXt-tiny | c + `fdconv_lite` | （FDConv 不在 ConvNeXt 路径上，此臂仅作 FFT 降级臂 AP 差量化） |
| e | ViT（现有 ViT 800 脚本为基座） | `use_dbfsaf` 单开 | DBFSAF 净贡献（探索臂，判据见 §1.3） |
| f | ViT | `use_dbfsaf + use_fsd_down + use_sfs_ff` | 三模块全开（论文完整配置的本仓映射） |
| g（可选） | 最优臂 | γ ∈ {0.25, 0.75} | 复核论文 γ=0.5 结论的可迁移性 |

记录：mAP@[.5:.95]、AP50、APs/APm/APl、步时延、参数量、`deploy()` 后 AP（FFT 两臂）。

**协议注意**：论文用 maxDets=300，本仓 `eval_max_dets=500`——对比论文数字时必须注明；如需直接对比，可加跑一臂 `eval_max_dets=300` 的评测（仅评测期开关，不改训练）。

---

## 9. 风险与开放问题

| # | 风险 | 缓解 |
|---|---|---|
| R1 | **官方代码未放出**，F1（SFS-FF 输入契约）等论文歧义点可能读错 | 全部歧义显式落入 §5 差异清单；代码放出后逐项复核，F1 是第一优先 |
| R2 | FDConv 移植保真度（三组件交互细节） | vendored 快照 + allclose 测试；`fdconv_lite` 负控臂隔离"频域建模 vs 容量" |
| R3 | 论文增益不可平移（DEIMv2-L 双分支 4 层 vs 本仓单分支 3/4 层 + 配方差异） | §1.3 分模块判据（FSD-Down ≥ +0.3、SFS-FF ≥ +0.2、DBFSAF ≥ +0.8 才保留）；负结果照常记录 |
| R4 | FFT 分支 bf16 精度 / ONNX DFT 生态 | fp32 island（F7）；PR-4 两臂验证 + 恒等降级 + AP 差量化 |
| R5 | SFSFF 容量低于 CSPRepLayer 导致深层融合欠拟合 | c 臂直接度量；若 AP 掉 >0.5，可加 `sfs_ff_blocks: int`（每处叠 1–2 个残差细化块）作为后续开关 |
| R6 | 组合矩阵膨胀（density/fsdc/ssl/token_drop） | v1 互斥 density×fsdc（F6）；ssl/token_drop 组合有专项测试（§7）；组合收益未经实验前不加新开关 |
| R7 | DBFSAF 在 ViT 小变体（hidden 192/224）上 γ 切分后支路过窄（96/112） | 守卫只保证整除；§8 g 臂可试 γ=0.4/0.6；不达标即关闭 use_dbfsaf |

**开放问题**（评审时定）：

1. FSD-Down 是否同时替换 STA 金字塔内部 stride-2 conv（F2 的下一候选点）——若 b 臂 APs 增益显著，值得追加；
2. `LTDETRFsdcArgs` 是否需要一个 `apply_levels: list[int]`（只在 S3/S4 等指定层启用 SFS-FF/FSDDown）——v1 全层启用，除非 smoke 显示浅层不稳定；
3. `fsdc` 与 `density` 的组合 encoder（Dome 密度分支 + FSD-Down 下采样在机制上正交，理论可叠加）——待两分支各自收益确认后再立项；
4. 是否为 FSDC-DETR 的 31.1/21.0 对比数字在 `docs/design/11-appendix-baseline-analysis.md` 增补条目（含协议差异注记）。

---

## 10. 参考索引

**论文与上游参考实现**：

| 内容 | 位置 |
|---|---|
| FSDC-DETR 论文（摘要/全文 v2） | arXiv:2607.05176（abs / html） |
| 官方代码仓库（未放出，watch 复核 F1） | github.com/nevereverinsomnia/FSDC-DETR |
| FDConv（DBFSAF 的卷积核，移植源） | github.com/Linwei-Chen/FDConv（CVPR 2025） |
| FPU（SFS-FF/MKSRU 的频率单元来源） | SFS-Conv（CVPR 2025） |
| Haar 下采样（FSD-Down 基础） | HWD, Pattern Recognition 2023（Elsevier） |
| 被批评的先行工作（对照） | UAV-DETR（frequency-focused downsampling） |

**VTrain 受影响面**（`src/autovision/`）：

| 内容 | 位置 |
|---|---|
| ViT 双分支融合点（DBFSAF 落点） | `_task_models/ltdetr_object_detection/backbone_wrappers.py:125-276` |
| ConvNeXt backbone（单分支，无 DBFSAF） | `_task_models/ltdetr_object_detection/backbone_wrappers.py:279-337` |
| HybridEncoder / CSPRepLayer / PAN 下采样 | `_task_models/object_detection_components/hybrid_encoder.py:96-125,192-338` |
| Dome encoder 子类模式（本设计对齐对象） | `_task_models/ltdetr_object_detection/dome_encoder.py:50-154` |
| 任务模型组装与组合守卫惯例 | `_task_models/ltdetr_object_detection/task_model.py:150-267,372-390` |
| init-args 快照（weights_only 兼容） | `_task_models/ltdetr_object_detection/task_model.py:159-175` |
| 训练参数（fsdc 字段落点） | `_task_models/ltdetr_object_detection/config.py`（`LTDETRObjectDetectionTrainArgs`） |
| ConvNeXt-tiny 800px 基线配方（孪生源） | `scripts/visdrone_ltdetr_800.py` |
| DEIMv2 HF 包装（排除集成的依据） | `_task_models/deimv2_object_detection/task_model.py:4-24,215` |
| 组件规格源（论文① Eq.1–13 精确形式与 E/G 勘误表） | `docs/design/22-fsdc-detr-native-dual-branch.md` |
| 参照分册 | `docs/design/13-ltdetr-dome-integration.md`（子类/守卫/孪生测试/导出契约模式） |

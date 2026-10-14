# VTrain 实现蓝图 · 分册 22：FSDC-DETR 原生复现 —— DEIMv2 式双分支骨干 × 频-空协同全栈

| 文档属性 | 内容 |
|---|---|
| 版本 | v1.1（已实施；实施记录见下方实施注记） |
| 状态 | 已实现并通过 `tests/test_fsdc_native.py`（32 例）与全量回归（1409 passed / 0 failed）。PR-5（消融训练）待 GPU 资源 |
| 上游论文 | ① FSDC-DETR（ECCV 2026, arXiv:2607.05176v2）——DBFSAF/SFS-FF/FSD-Down 三模块，Eq.1–13 已逐条核实；② DEIMv2: Real-Time Object Detection Meets DINOv3（arXiv:2509.20787）——双分支骨干、损失配方、变体规格 |
| 权威实现 | HF `transformers/models/deimv2/modeling_deimv2.py`（本机 5.19.0，Deimv2DINOv3ConvEncoder / Deimv2SpatialTuningAdapter / Deimv2HybridEncoder / Deimv2Decoder + 官方 criterion）——**decoder/criterion 白拿，不移植不重写** |
| 与分册 21 的关系 | 21 = LT-DETR 宿主的模块移植路径（对照实验用）；22 = 按源论文的完整原生复现路径（对标 31.1 AP 主结果）。两册**共享组件层**（`wavelet_downsample.py / sfs_ff.py / fdconv.py / dbfsaf.py`，见 §3.1 依赖声明），接线对象不同 |
| 受影响模块 | 新增 `_task_models/fsdc_object_detection/`（6 文件）；共享组件层 4 文件（沿 21 册）；`scripts/visdrone_fsdc_800.py`；`tests/test_fsdc_native.py` |
| 关键勘误 | FSDC 论文把 CNN 分支称作 "HGNetv2"——**经 DEIMv2 原论文核实为误称**：DINOv3 变体（S/M/L/X）的 CNN 分支是 Spatial Tuning Adapter（STA），HGNetv2 只用于 Nano/Pico/Femto/Atto 超轻变体（§2.1 勘误表 E1）。本册以 STA 为准 |

> **实施注记（v1.1，2026-10-10）**：PR-0→PR-4 已合并交付。
> **PR-0 结果**：官方 DEIMv2-L 配置从 `Intellindust/DEIMv2_DINOv3_L_COCO` config.json 钉死（hidden 224 / dec_layers 4 / dec_ffn 1792 / enc_ffn 896 / STA inplanes 32 / expansion 1.0 / taps 5,8,11 / fuse_op sum），已映射为 `deimv2-dinov3-l` 注册表条目；Hub 仅 N/PICO/S 有 transformers 转换 → **fsdc-l 只能 fresh 训练**（`load_weights=False` 家族默认），fsdc-s 保留 warm-start 臂。FDConv 官方源不可达（github 超时、无 gh CLI）→ 按论文语义实现 FDW+FBL（频带掩码不相交性有测试），allclose 对官方快照的测试标记为 vendored 后补。
> **关键实现事实**：① 4 层金字塔必须在 HF 构建**前**注入 config overrides（deformable attention 的 `num_points_scale` buffer 在构造期烘焙，事后改 config 无效）——父类新增 `config_overrides` 参数承载；② DINOv3 权重转移必须做在 **HFBackbone wrapper 层**（embeddings/rope/norm 在 `.model` 之外，`.model` 域加载会静默漏掉 patch embeddings——已由替换面等价测试钉死）；③ **替换面等价证明通过**：DBFSAF/SFS-FF 全关 + 权重对拷时，与 stock DEIMv2 前向逐比特一致（键集合零差）。
> **代码审查修复**：fsc_encoder 未用导入；FPU 重复 weight 计算；`fft_export_fallback` 由 owner 模块的 `convert_to_deploy` 接线（TaskModel.deploy() 递归覆盖）；FSDC 家族 `load_weights` 默认改 False（fsdc-l 无转换，fsdc-s 显式 opt-in）。
> **遗留**：PR-5 消融训练（§8 七臂）；FDConv 官方快照 allclose；官方代码放出后复核 G4（SRM 细节）与 G6（算子分配标注）。
>
> **实施注记（v1.2 勘误，2026-10-11，首次真实训练后）**：fsdc-l v1 臂（`runs/visdrone_fsdc_l_800_full`）暴露 PR-0 映射错误——**官方 config.json 的 `DINOv3STAs.embed_dim: 224` 是 STA wrapper 的投影后宽度，不是 ViT 宽度**。官方 yml（DEIMv2-main `deimv2_dinov3_l_coco.yml`）加载真 DINOv3 ViT-S/16 预训练权重（`dinov3_vits16_pretrain_lvd1689m`，hidden 384 / 6 heads / 4 register tokens），且 `dinov3_adapter.py:103` 用 `embed_dim = self.dinov3.embed_dim` 覆盖构造参数后做 concat+1×1 投影到 224。v1 臂因此是"随机 224 宽 ViT + 预训练级 0.05× LR"，mAP 仅 0.1273（同协议 DEIMv2-S 基线 0.2182；step 1000 时 0.023 vs 0.1206——典型从头训练特征）。官方 L 优化器配方已取回（lr 5e-4 / **dinov3 桶 1.25e-5 = 0.025×** / wd 1.25e-4）。
> **v2 修复**：① registry 骨干改 384/6h/12L/4register + lr_backbone_ratio 0.025；② `FsdcArgs.backbone_weights` + `FsdcDinoDualBranchEncoder.load_dinov3_pretrained`（facebook safetensors `layer.*` → `model.layer.*` 前缀重命名是唯一映射，211 键 1:1 严格加载，测试钉死）；③ 实验脚本默认注入本地预训练权重（`rf-detr-drone/weights/dinov3-vits16-pretrain-lvd1689m`，manifest 已注册）。v1 全开臂数据保留为"窄随机骨干 pilot"记录；v2 队列（l2 全开 → l2-enc → fsdc-s）以正确架构重启，smoke 通过（36.1M 参数、19.7GB、0.7 s/step）。
>
> **引擎 bug（v1.2 分析中连带发现，已修复）**：`engine.py` 的 fit 循环只在 validation 块末尾调用 `train_model.train()`——HF `from_pretrained` 返回 eval 模式模型，导致 **load_weights=True 的 run（deimv2/rfdetr 基线）在第一次 validation 之前的全部步数于 eval 模式下训练**（DN 组不构建、FGL-DN/aux 损失缺失、BN 统计冻结、无 dropout）。实测特征：基线 train_loss 在 step 1000 的 validation 后从 3.8 跳到 25.6、步时 0.34→0.53 s（损失项 6 项 → 全项），loss_dict 分量对比钉死。修复 = 训练循环开头强制 `train()`（`engine.py` fit_task），回归测试 `tests/test_engine_train_mode.py`（撤销修复必失败）。**影响面**：fsdc 臂（load_weights=False，构造即 train 模式）不受影响；已完成的 deimv2-s 基线前 1000 步被污染（best@5000 不受直接影响），基线重跑（`_v2`）排在当前队列之后；rfdetr 历史 run 同源污染，记录不改。
>
> **实施注记（v1.3 消融中期结果，2026-10-11）**：VisDrone 800px pilot（8088 步，global batch 16，与 `visdrone_deimv2_800.py` 孪生）。
>
> | 臂 | 配置 | 最终 mAP | vs 基线 |
> |---|---|---|---|
> | 基线 a-v2 | stock DEIMv2-S（COCO 全预热；**引擎修复后净化重跑**） | **0.2173** @5000 | — |
> | l2 全开 | 384 ViT-S（LVD-1689M）+ DBFSAF + SFS-FF/FSD-Down，decoder fresh | 0.2256 @8000 | +0.8 |
> | **fsdc-none** | **stock DEIMv2-L 架构**（1×1 盲融合 + 原版 encoder）+ 同预训练 ViT-S | **0.2406** @8000 | **+2.3** |
> | l2-enc | fsdc-none + SFS-FF/FSD-Down（DBFSAF 关） | 0.2398 @7000（已平台） | +2.3 |
> | fsdc-s | 192 ViT-T + SFS-FF（encoder fresh）+ COCO warm decoder | 0.1958 @7000（已平台） | -2.2 |
>
> **净化锚点注**：基线 a-v2（0.2173）与旧污染版（0.2182）终值几乎一致——eval 污染对最终 best 影响可忽略，历史对比读数不变。
> **结论一（DBFSAF 负贡献，成对对照确立）**：l2 vs l2-enc 六个验证点全部为 enc 领先（+1.3 ~ +2.3），val_loss 差距从 step 1000 即存在（-0.21）且恒定——DBFSAF 在 8k 步预算下从起点就拖累融合质量。候选根因按置信度：① FDConv 为语义移植（G4，官方源不可达），论文 +5.0 AP 的归因强依赖官方 FDConv 行为；② γ=0.5 直通砍半 + FFT 门控在短预算下优化不完；③ 官方 100ep×64 配方对该模块友好。**处理：默认关闭 `use_dbfsaf`；官方 FDConv 代码获得后重评。**
> **结论二（SFS-FF/FSD-Down 在本仓条件下中性）**：fsdc-none（stock 架构孪生）0.2406 vs l2-enc 0.2398（**-0.1，运行间噪声内**）——l2-enc 相对旧基线的 +2.3 **来自 384 ViT-S 容量而非频域模块**。步时持平（0.61-0.63 s/step）。对照早期"S 宽度 -2.2"读数：SFS-FF 不伤 L 宽度，但在短预算下也无净增益。**处理：默认关闭；代码保留，24k 步符号翻转探针（none-24k vs l2enc-24k）运行中。**
> **结论三（最大杠杆 = 骨干容量）**：fsdc-none（fresh decoder）0.2406 vs 基线 a-v2（COCO warm decoder）0.2173 = **+2.3 纯粹来自换 384 ViT-S**。论文 +6.4 的对比基线本身就在这个容量档——这是增益难以移植的核心解释，与分册 21 §1.1.5 的容量混杂批判互相印证。

> **标注约定**：【论文①】FSDC-DETR 论文事实（§/Eq.）；【论文②】DEIMv2 论文事实；【HF】transformers 5.19 建模代码事实（`文件:行`）；【VTrain】本仓库现状；【设计】本文档决策；【差异】相对论文①表述的有意偏离/勘误（G1–G8）。

---

## 1. 背景与立项理由

### 1.1 分册 21 没有回答的问题

分册 21 把三模块移植进 LT-DETR（ConvNeXt-tiny 单分支 + ViT+STA），但论文的 +6.4 AP 是在 **DEIMv2-L（DINOv3 ViT-S + STA 双分支）** 上取得的。21 册已确认两处不可平移：ConvNeXt 变体没有双分支融合点（DBFSAF 无靶子）；ViT+STA 变体的分支容量与 DEIMv2-L 不对称。**要回答"FSDC-DETR 的完整配方在本仓条件下复现多少"，只有把双分支骨干原生建出来这一条路。**

### 1.2 DEIMv2 官方自陈的缺口（立项背书）

【论文②】§3 原文：DINOv3-based DEIMv2 "gains primarily arise from improvements on medium and large objects, while performance on small objects remains largely unchanged... DINOv3 excels at capturing strong global semantics but has limited ability to represent fine-grained details"——**DEIMv2 作者亲口确认小目标/细粒度短板**，FSDC-DETR 的频-空三模块正是冲着这个缺口来的。本册的实验因此有明确的因果假设可检验：双分支 ViT 主干（全局语义强、高频弱）+ 频域补偿模块 → APs 增益。

### 1.3 目标与非目标

**做**：

1. 原生实现 DEIMv2 式双分支骨干：DINOv3 ViT-S（384 维、12 层）+ STA 细节金字塔，**四层 P2–P5 融合链可选**（G2），每层以 **DBFSAF（论文① Eq.1–6 逐条）** 替换 DEIMv2 的 concat+1×1 盲融合；
2. 原生实现 FSC-Hybrid Encoder：AIFI/FPN 保持 DEIMv2 原样，PAN 融合替换为 **SFS-FF（Eq.10–11）+ FSD-Down（Eq.12–13）**；
3. **decoder + criterion 保持 HF 官方**（MAL 1.0 / FGL 0.15 / DDF 1.5 / L1 5 / GIoU 2 + dense O2O，【HF】`Deimv2ForObjectDetection`，VTrain `deimv2_object_detection/task_model.py:4-24` 的"转发而非重实现"策略原文），通过**子模块替换**接线（§3.2）；
4. 变体注册表 `fsdc-l`（论文锚点）+ `fsdc-s`（d_model 192，可从在库 DEIMv2-S COCO 检查点 warm-start decoder 的廉价健全臂）。

**不做（非目标）**：

- 不重写/不修改 HF decoder、criterion、matcher——dense O2O、MAL、FGL/DDF 全部沿用官方转换实现；
- 不动 LT-DETR 家族（21 册范围）、不动既有 DEIMv2 基线行为；
- 不引入论文①的 100ep×batch64×H200×8 配方（沿 VTrain 引擎契约与 `visdrone_deimv2_800.py` pilot 配方；配方差异记 G7）；
- object 级 Copy-Blend 增广不在 v1 数据管线内（现 DEIMv2 基线脚本亦未启用，G8）。

### 1.4 预期收益与判据

论文锚点：DEIMv2-L 24.7 → FSDC-DETR 31.1 AP（VisDrone test，800px，maxDets=300）。本仓 pilot 配方（8k 步、global batch 16、无 CopyBlend/Mosaic）下的判据分层：

| 臂 | 对照 | 保留判据 |
|---|---|---|
| DBFSAF 单开（骨干） | 同配置 fsdc 关闭（= 增强 baseline） | AP ≥ +1.0（论文归因 +5.0，预期打折） |
| + SFS-FF/FSD-Down（encoder） | 骨干臂 | AP ≥ +0.4 且步时延增幅 ≤ 15% |
| 完整 fsdc-l | DEIMv2-S 现有基线（`visdrone_deimv2_800.py`） | APs 相对提升 ≥ 15% 方可宣称复现方向正确 |
| 3 层 vs 4 层融合链 | 互为对照 | 检验论文①"四融合层"容量混杂假设（§1.1.5 批判） |

---

## 2. 三个权威事实源对照与勘误

### 2.1 勘误表（FSDC 论文表述 vs 权威事实）

| # | FSDC 论文①表述 | 权威事实 | 本册决策 |
|---|---|---|---|
| E1 | 双分支 = "HGNetv2 + DINOv3"（§2.2/§3.1） | 【论文②】DINOv3 变体（S/M/L/X）CNN 分支 = **STA**；HGNetv2 仅超轻变体。【HF】`Deimv2DINOv3ConvEncoder` 内是 `Deimv2SpatialTuningAdapter`（`modeling_deimv2.py:830-847,1160-1204`） | CNN 分支按 **STA** 实现；"HGNetv2 wrapper" 立项需求转译为"STA 的四层原生扩展"（G1） |
| E2 | DBFSAF 作用于 stage P∈{2,3,4,5}（四层） | 【HF】DEIMv2 三层（strides 8/16/32，resize 公式 `modeling_deimv2.py:1181-1187`）；【论文①】supplement 提"四融合层增强 baseline" | `sta_levels: 3\|4` 双臂配置，v1 两臂都跑（默认 3 = DEIMv2 忠实；4 = 论文口径），以消融裁决容量混杂（G2） |
| E3 | MKSRU 通道映射 | 【论文①】Eq.8：{1/2C→DW7×7 d3，**3/8C→DW7×7 d2，1/8C→恒等**} | 按 Eq.8 精确实现（21 册初版映射有误，已在该册 v1.1 修订） |
| E4 | SFS-FF 输入契约 | 【论文①】Eq.10：`Conv1×1(Cat[F_in, Down(F̂_in)])`，F_in (C,H,W)、F̂_in (C,2H,2W)、输出 (C,H,W)；Down = FSD-Down | 精确按 Eq.10–11：**Down 内嵌于 SFS-FF，落点 = PAN 一处**；FPN 论文未定义 → 保持 DEIMv2 原样（G3） |
| E5 | FSD-Down 门控形式 | 【论文①】Eq.12–13：`F_CA = σ(GAP(Cat[DWT(4C), Conv1×1(GConv)(4C)]))`（**8C 共享门**）；`Conv1×1(ρ⊙F_CA⊙DWT + F_CA⊙ConvPath)`（单输出投影） | 按 Eq.12–13 精确实现（21 册初版的双支路独立投影方案作废） |
| E6 | Eq.4–5 门控 | Softmax 在 `Cat[F_S, F_F]` 的 **2C 通道维**整体做，α/β 可学习标量分别乘两半 | 按 Eq.4–5 精确实现 |
| E7 | SRM 引用 [DEIMv2, RT-DETRv1] | 【HF】transformers Deimv2 无 SRM 类（grep 确认）；原框架代码未获得 | 本地最小实现（DW3×3+BN+GELU+PW+BN），代码放出后对齐（G4） |
| E8 | "learnable wavelet decomposition"（摘要） | 小波滤波器是否可学习未明示；HWD 用固定 Haar | `nn.Parameter` 以正交 Haar 基初始化 + `freeze_wavelet` 开关；固定 Haar 为消融臂（G5） |

### 2.2 DEIMv2 双分支骨干的权威结构（【HF】`modeling_deimv2.py`）

```
Deimv2DINOv3ConvEncoder (L1160)
├── backbone = load_backbone(config)          # DINOv3 ViT（本册：ViT-S，hidden 384, 12 层）
├── spatial_tuning_adapter (L830)             # STA：stem_conv(3×3,s2)+MaxPool → conv2/3/4（s2 链）
│     forward → (h2, h3, h4) @ strides 8/16/32，通道 {2i, 4i, 4i}·inplanes
└── fusion_proj[i] = ConvNormLayer(embed + i·{2,4,4}, hidden, 1×1)   # 盲融合 ← DBFSAF 替换点

forward：
  semantic = [resize(ViT_tap_k, patches·2^(S-2-i)) for k in out_indices]   # 双线性，无参数
  detail   = STA(pixel_values)
  out[i]   = fusion_proj[i](Cat[semantic[i], detail[i]])                    # ← DBFSAF 接管
```

四层扩展（G2）：STA 暴露 stem 输出（stride 4，inplanes 通道）为 h1；ViT taps 取 4 个块（如 (3,6,9,11)）分别 resize 到 strides 4/8/16/32（stride-4 层 = patch 网格上采样 ×4，ViTDet 同款）；`fusion_proj` 变 4 项，通道表 `[embed+i, embed+2i, embed+4i, embed+4i]`。

### 2.3 DEIMv2 Hybrid Encoder 的权威结构（【HF】`modeling_deimv2.py:1264-1384`）

```
AIFI（encode_proj_layers=[2]，最深层级）→ 保持
FPN 自顶向下：lateral_conv → nearest×2 上采样 → fuse_op(sum|cat) → RepNCSPELAN5 → 保持
PAN 自底向上：SCDown(3×3,s2) ↓2 → fuse_op → RepNCSPELAN5  → SFS-FF(F_in, F̂_in) 整对替换
```

注意【HF】DEIMv2 融合默认 **element-wise sum**（`encoder_fuse_op`，非 D-FINE 的 concat）——SFS-FF 内部的 `Cat + Conv1×1` 天然吸收该 fuse 操作，替换面是 `(SCDown, RepNCSPELAN5)` 对，共 `len(in_channels)-1` 处。

---

## 3. 总体设计

### 3.1 模块拓扑与依赖

```
src/autovision/_task_models/object_detection_components/     # 分册 21 的组件 PR（共享）
├── wavelet_downsample.py   # HaarDWT2d + FSDDown（按 Eq.12-13 形式，本册为规格源）
├── sfs_ff.py               # SFSFF（Eq.10-11 形式）+ SRM 本地版
├── fdconv.py               # FDConv 移植（FDW+FBL，KSM 可选）
└── dbfsaf.py               # MKSRU（Eq.7-9）+ FPU + DBFSAF（Eq.1-6）

src/autovision/_task_models/fsdc_object_detection/           # 本册新增
├── __init__.py
├── config.py               # FSDC_MODEL_REGISTRY（fsdc-l / fsdc-s）+ FsdcObjectDetectionTrainArgs
├── task_model.py           # FsdcObjectDetection(Deimv2ObjectDetection)：子模块替换接线
├── train_model.py          # 继承 deimv2 train_model（LR 桶表加一条，§4.6）
├── dual_branch_backbone.py # FsdcDinoDualBranchEncoder（HF Deimv2DINOv3ConvEncoder 的原生四层扩展 + DBFSAF）
└── fsc_encoder.py          # FsdcHybridEncoder（HF Deimv2HybridEncoder 原样 + PAN SFS-FF）

tests/test_fsdc_native.py
scripts/visdrone_fsdc_800.py
```

**依赖声明**：本册的组件实现以分册 21 的 PR-1/PR-2/PR-3 为前置（同一批组件文件，两册共用）；差异仅在 21 册按 LT-DETR 接口接线、本册按 HF Deimv2 接口接线。组件层的规格以**本册核实的论文 Eq.1–13 精确形式为准**（21 册 v1.1 已同步修订）。

### 3.2 组装方式：HF 躯干 + 子模块替换（本册核心决策）

**先例**：【VTrain】`deimv2_object_detection/task_model.py:111-113` 已对 `model.model.enc_score_head` 做子模块手术（类头重铺）；`deimv2/train_model.py:36-75` 已按参数名路径（`.conv_encoder.backbone.`）做 LR 桶。子模块替换是既定模式，不违背"转发而非重实现"策略——被替换的只有骨干与 encoder，**decoder/criterion 保持 HF 官方对象不动**。

```python
class FsdcObjectDetection(Deimv2ObjectDetection):
    """DEIMv2 躯干（官方 decoder/criterion/dense-O2O）+ 原生 FSDC 双分支骨干 + FSC encoder。"""

    def __init__(self, num_classes, model="fsdc-l", image_size=(640,640), ..., fsdc: FsdcArgs | None = None):
        super().__init__(...)                       # 完整 HF 模型构建 + COCO 检查点加载（fsdc-s 可命中）
        hf = self._model.model                      # Deimv2Model（task_model.py:173 的 self._model）
        hf.conv_encoder = FsdcDinoDualBranchEncoder(spec, fsdc)   # 原生双分支骨干
        hf.encoder      = FsdcHybridEncoder(spec, fsdc)           # 原生 FSC encoder
        # decoder / decoder_input_proj / enc_*_head / criterion 不动
```

**替换面验证约束**（进 §7 测试）：`fsdc` 两侧均替换为**逐权重拷贝的 HF 原版模块**时，模型前向与原版 DEIMv2 逐比特一致（证明替换面完整、无隐式依赖）；此后再切原生模块。

**warm-start 策略**（三层，PR-0 验证后落定）：

| 层 | 来源 | 条件 |
|---|---|---|
| DINOv3 ViT-S | 公共 DINOv3 权重（VTrain dinov3 wrapper 既有来源链） | 必然可得 |
| decoder + criterion 头 | HF DEIMv2 COCO 转换检查点（宽度匹配时整体拷入） | fsdc-s（d_model 192）可从在库 `deimv2-dinov3-s` 命中；**fsdc-l（384）待 PR-0 核实 harshaljanjani 是否有 DEIMv2-L/DINOv3-M 转换**，无则 fresh |
| STA/DBFSAF/FDConv/FSC-encoder | 随机初始化（fusion_proj 1×1 与 DBFSAF 结构不同，不可映射） | — |

### 3.3 数据流

```
x(B,3,H,W)
  │
  ▼ FsdcDinoDualBranchEncoder
  semantic[s] = resize(DINOv3ViT_blocks(5,8,11[,·]))     # s = 3 或 4 层（G2）
  detail[s]   = STA_4level(x)
  feat[s]     = DBFSAF[s](Cat[semantic[s], detail[s]])    # Eq.1-6 替换 fusion_proj
  │
  ▼ FsdcHybridEncoder
  proj[s]     = input_proj[s](feat[s])                    # ConvNormLayer 1×1，原样
  proj[ deepest ] = AIFI(proj[deepest])                   # 原样
  FPN：lateral → nearest×2 → fuse(sum) → RepNCSPELAN5     # 原样（G3）
  PAN：F_out = SFSFF(F_in=coarse(H×W), F̂_in=fine(2H×2W))  # Eq.10-11；内含 FSD-Down(Eq.12-13)
  │
  ▼ Deimv2Decoder + decoder_input_proj + enc_*_head（HF 原样）
  ▼ criterion（HF 原样：MAL/FGL/DDF/L1/GIoU + dense O2O）
```

---

## 4. 详细设计

### 4.1 DBFSAF（【论文①】Eq.1–6，`dbfsaf.py`，规格以本册为准）

```python
class DBFSAF(nn.Module):
    def __init__(self, in_channels, out_channels, gamma=0.5):
        self.fdconv = FDConv(in_channels, out_channels, ...)          # §4.4
        c_fs = int(out_channels * gamma); c_c = out_channels - c_fs   # Partial_γ，论文消融 γ=0.5 最优
        self.fs_proj_s = nn.Conv2d(c_fs, c_fs, 1)                     # Eq.2-3：两支各自 1×1（论文
        self.fs_proj_f = nn.Conv2d(c_fs, c_fs, 1)                     #   "两份拷贝"读法；若官方代码为共享 1×1，合并即可）
        self.mksru = MKSRU(c_fs)                                      # Eq.7-9
        self.fpu = FPU(c_fs)                                          # SFS-Conv 频率单元，fp32 island
        self.alpha = nn.Parameter(torch.ones(1)); self.beta = nn.Parameter(torch.ones(1))
        self.c_branch = ConvNormLayer(c_c, c_c, 3, 1)                 # Conv3×3(F_C)
        self.out_proj = ConvNormLayer(out_channels, out_channels, 1, 1)

    def forward(self, x):                                             # x = Cat[semantic, detail]
        f_init = self.fdconv(x)                                       # Eq.1
        f_fs, f_c = torch.split(f_init, [c_fs, c_c], dim=1)           # Eq.1 Partial_γ
        f_s = self.mksru(self.fs_proj_s(f_fs))                        # Eq.2：空间精炼支路
        f_f = self.fpu(self.fs_proj_f(f_fs))                          # Eq.3：频率处理支路
        a = torch.softmax(gap(torch.cat([f_s, f_f], 1)), dim=1)       # Eq.4：2C 通道维整体 softmax
        f_hat_s = self.alpha * a[:, :c_fs] * f_s                      # Eq.4
        f_hat_f = self.beta  * a[:, c_fs:] * f_f                      # Eq.5
        return self.out_proj(torch.cat([f_hat_s + f_hat_f, self.c_branch(f_c)], 1))   # Eq.6
```

**MKSRU（Eq.7–9 精确形式，E3）**：

```python
class MKSRU(nn.Module):
    def __init__(self, c):
        self.dw5 = nn.Conv2d(c, c, 5, 1, 2, groups=c)                 # Eq.7：DWConv5×5(d1)
        # Eq.8 切分映射：{1/2C → DW7×7 d3；3/8C → DW7×7 d2；1/8C → 恒等}
        self.dw7_d3 = nn.Conv2d(c//2, c//2, 7, 1, 9, dilation=3, groups=c//2)   # same-pad = d·(k−1)/2
        self.dw7_d2 = nn.Conv2d(3*c//8, 3*c//8, 7, 1, 6, dilation=2, groups=3*c//8)
        self.gate = nn.Conv2d(c, c, 1)                     # Eq.9 上支 Conv1×1
        self.short_conv = nn.Conv2d(c, c, 3, 1, 1)         # Eq.9 下支 Conv3×3
        self.gate_s = nn.Conv2d(c, c, 1)                   # Eq.9 下支 Conv1×1
    def forward(self, x):
        d = self.dw5(x)
        a, b, c_ = torch.split(d, [c//2, 3*c//8, c//8], dim=1)
        x_hat = torch.cat([self.dw7_d3(a), self.dw7_d2(b), c_], 1)    # Eq.8
        return F.silu(self.gate(x_hat)) * F.silu(self.gate_s(self.short_conv(x)))  # Eq.9：SiLU 门控乘积，无残差加
```

Eq.9 语义：`X_out = SiLU(Conv1×1(X̂_D)) ⊙ SiLU(Conv1×1(Conv3×3(X_in)))`——**乘性双支路，无显式残差**。

### 4.2 FSD-Down（【论文①】Eq.12–13，`wavelet_downsample.py`，规格以本册为准）

```python
class FSDDown(nn.Module):
    """Eq.12-13 精确形式：8C 共享门 + ρ 标量 + 单输出投影。"""
    def __init__(self, channels: int, freeze_wavelet: bool = False):
        self.dwt = HaarDWT2d(channels, learnable=not freeze_wavelet)   # (B,4C,H/2,W/2)；G5
        self.gconv = nn.Conv2d(channels, channels, 2, 2, groups=channels, bias=False)
        self.conv_path = nn.Conv2d(channels, 4*channels, 1, bias=False)   # Conv1×1(GConv) → 4C
        self.out_proj = nn.Conv2d(4*channels, channels, 1)                # Eq.13 输出
        self.rho = nn.Parameter(torch.ones(1))
    def forward(self, x):
        dwt = self.dwt(x)                                              # 4C
        cp = self.conv_path(self.gconv(x))                             # 4C
        f_ca = torch.sigmoid(gap(torch.cat([dwt, cp], 1)))             # Eq.12：8C 向量
        gate_d, gate_c = f_ca.chunk(2, dim=1)                          # 前 4C 门 DWT 支路，后 4C 门 conv 支路
        return self.out_proj(self.rho * gate_d * dwt + gate_c * cp)    # Eq.13
```

Haar 滤波器以正交 Haar 基初始化为 `nn.Parameter`（"learnable wavelet"，E8/G5）；`freeze_wavelet=True` 回退固定 Haar（消融臂）。奇数边长右/下零填充一行/列（⌈·⌉ 语义）。

### 4.3 SFS-FF 与 PAN 接线（【论文①】Eq.10–11，`sfs_ff.py` + `fsc_encoder.py`）

```python
class SFSFF(nn.Module):
    """Eq.10-11：PAN 融合点整对替换 (SCDown, RepNCSPELAN5)。"""
    def __init__(self, c):
        self.proj_in = nn.Conv2d(2*c, c, 1)                            # Eq.10：Cat + 1×1
        self.srm = SpatialRefineModule(3*c//8)                         # 频域支路 × 空间算子（交叉）
        self.fft = FrequencyGlobalBranch(c//2)                         # 空间支路 × 频域算子（rFFT2→Conv3×3→irFFT2，fp32 island）
        self.proj_out = nn.Conv2d(c, c, 1)
    def forward(self, f_in, f_hat_in):                                 # (C,H,W) + (C,2H,2W)
        x = self.proj_in(torch.cat([f_in, self.down(f_hat_in)], 1))    # Eq.10；down = FSD-Down
        x_f, x_s, x_id = x.split([3*c//8, c//2, c//8], dim=1)
        return self.proj_out(torch.cat([self.srm(x_f), self.fft(x_s), x_id], 1))   # Eq.11
```

`FsdcHybridEncoder.forward` 的 PAN 段：`pan_maps.append(SFSFF(pan_maps[-1] 对齐后, fpn_maps[idx+1]))`——`downsample_convs` 与 `pan_blocks` 在原生 encoder 中**不再构建**；FPN 段逐行保持【HF】`Deimv2HybridEncoder.forward`（G3）。两支算子分配（x_f→SRM、x_s→FFT 的"交叉"标注）以论文① §3.2 文字为准，官方代码放出后为第一复核项。

### 4.4 FDConv（`fdconv.py`）

沿分册 21 §4.4 决策不变：移植 FDW（Fourier-Disjoint Weight）+ FBL（Frequency Band Learning），KSM 可选（`fdconv_use_ksm=False` 默认）；`fdconv_lite`（Conv3×3）负控臂隔离"频域建模 vs 纯容量"；与 vendored 官方快照 allclose 测试。

### 4.5 变体注册表与配置面（`config.py`）

```python
FSDC_MODEL_REGISTRY = {
    "fsdc-l": dict(   # 论文①锚点：DEIMv2-L = DINOv3 ViT-S(384) + STA
        base_variant="deimv2-dinov3-l",   # PR-0 核实官方 L 配置与 HF 转换可得性；缺省回退 M
        config=dict(d_model=384, encoder_hidden_dim=384, ...),   # 精确值 PR-0 从官方 yaml/Hub 卡钉死
    ),
    "fsdc-s": dict(   # 廉价健全臂：d_model 192，decoder 可从在库 deimv2-dinov3-s COCO 检查点 warm-start
        base_variant="deimv2-dinov3-s",
    ),
}

class FsdcArgs(PydanticConfig):
    model_config = {"extra": "forbid"}
    sta_levels: int = 3                 # 3（DEIMv2 忠实）| 4（论文①口径，G2）
    use_dbfsaf: bool = True
    dbfsaf_gamma: float = 0.5
    use_sfs_ff: bool = True
    use_fsd_down: bool = True           # SFS-FF 关闭时 FSD-Down 无独立落点（内嵌于 Eq.10）
    freeze_wavelet: bool = False        # G5
    fdconv_use_ksm: bool = False
    fdconv_lite: bool = False
    fft_export_fallback: bool = False
```

守卫：`use_sfs_ff=False` ⇒ `use_fsd_down` 强制无效（FSD-Down 只在 Eq.10 内出现）；`sta_levels=4` 与 DEIMv2-S 宽度 warm-start 兼容性由 PR-0 断言。

### 4.6 双分支 LR 分组（`train_model.py`，继承 deimv2）

【VTrain】`deimv2/train_model.py:36-75` 的参数名桶机制直接复用：原生骨干保持参数名片段 `conv_encoder.backbone.*`（DINOv3 ViT）→ `dinov3` 桶（官方 0.05×）；新增 `conv_encoder.spatial_tuning_adapter.*`、`conv_encoder.dbfsaf.*`、`encoder.*` → detector 桶（1×）。仅需在桶表加 STA/DBFSAF 片段（显式落 detector 桶，防止未来改名误入 backbone 桶）。DBFSAF 属随机初始化连接件——与 21 册 §4.6 同判：连接件不进 backbone LR。

### 4.7 训练/验证/导出/引擎契约

| 项 | 结论 |
|---|---|
| 损失 | HF criterion 原样（dense O2O + MAL + FGL/DDF + L1 + GIoU）；`_reinitialize_class_head` 的类头重铺对替换后模型依旧成立（作用于 decoder 侧） |
| EMA/AMP/DDP | 沿 deimv2 既有链路；FFT fp32 island（21 册 F7）；无数据依赖控制流，DDP 无新同步点 |
| 优化器 | AdamW betas 0.9/0.999、clip 0.1、EMA 0.9999 warmup 1000——`Deimv2ObjectDetectionTrainArgs` 既有官方配方自动覆盖新参数 |
| 导出 | 全静态；FFT ONNX DFT（opset≥17）PR 验证 + `fft_export_fallback` 恒等臂（21 册 §4.7 机制） |
| 检查点 | `FsdcArgs` 走 init-args 快照（weights_only 安全加载器兼容，21 册 §4.6 模式） |

---

## 5. 差异清单（相对论文①，G 系列）

| # | 差异 | 决策与理由 |
|---|---|---|
| G1 | "HGNetv2 wrapper" → STA 四层原生扩展 | E1 勘误：DINOv3 变体的 CNN 分支是 STA（论文② + HF 权威）；按论文②/HF 实现即"按源论文"的正确读法 |
| G2 | P∈{2,3,4,5} 四层 vs HF 三层 | `sta_levels` 双臂（默认 3），消融裁决；论文① supplement 的"四融合层增强 baseline"提示四层可能掺容量增益 |
| G3 | SFS-FF 仅替换 PAN；FPN 保持 | Eq.10 签名只匹配 PAN（输出在粗网格）；FPN 无论文定义 |
| G4 | SRM 本地最小实现 | 无权威源码；代码放出后对齐 |
| G5 | 小波滤波器可学习（Haar 初始化）+ 冻结开关 | 论文①摘要 "learnable wavelet"；固定 Haar 为消融臂 |
| G6 | SFS-FF 两支算子分配按论文文字标注 | 官方代码放出后第一复核项（标注性差异，不改结构） |
| G7 | 训练配方：本仓 pilot（8k 步/global batch 16/无 Mosaic-MixUp-CopyBlend）≠ 论文① 100ep×64 | 引擎契约；与在库 DEIMv2 基线孪生可比是第一优先 |
| G8 | object 级 Copy-Blend 未实现 | 现行 DEIMv2 基线亦未启用；如需对齐论文①配方，作为独立增广 PR 立项 |

---

## 6. 实施计划（五个 PR，每个独立可合入）

### PR-0：spike——规格钉死 + 权重可得性验证（不改模型代码）

- 核实官方 DEIMv2-L yaml（Intellindust 仓库/Hub 卡）：d_model、encoder/decoder 宽度、层数、encoder_fuse_op、`spatial_tuning_adapter_inplanes`；
- 核实 HF Hub：DEIMv2-L/DINOv3-M 的 transformers 格式转换是否存在（fsdc-l decoder warm-start 可行性）；
- 产出：`FSDC_MODEL_REGISTRY` 最终数值 + 本册 §4.5 表回填。

### PR-1：组件层（与 21 册共享，规格按本册 Eq.1–13）

- `wavelet_downsample.py`（Eq.12–13 形式）、`sfs_ff.py`（Eq.10–11 形式）、`fdconv.py`、`dbfsaf.py`（Eq.1–9 形式）；
- 测试：DWT 重构/Eq.9 乘性门控有限性/Eq.12 8C 门切分/γ 切分/FDConv allclose。
- **验收**：单测全绿（该 PR 同时满足 21 册 PR-1 验收，两册共享）。

### PR-2：`fsdc_object_detection` 骨架 + 替换面等价证明

- `config.py`/`task_model.py`/`train_model.py` 接线；`dual_branch_backbone.py`、`fsc_encoder.py` 先以"HF 原版逐权重拷贝"形态接入；
- **关键测试**：替换面等价——双侧用 HF 原版模块拷贝时，与 stock DEIMv2 前向逐比特一致；`fsdc=None` 退化为 stock DEIMv2 包装。
- **验收**：单测全绿；`fsdc-s` 在 VisDrone 上 500 步 smoke 与 stock DEIMv2-S 曲线重合。

### PR-3：原生 DBFSAF 骨干上线

- `dual_branch_backbone.py` 切原生实现（STA 3/4 层 + DBFSAF×N）；LR 桶表加片段；
- 测试：四层形状链（800→strides 4/8/16/32）、`sta_levels` 守卫、检查点往返、与 stock 拷贝形态的数值差异仅来自 DBFSAF。
- **验收**：单测全绿；VisDrone 2k 步 smoke（fsdc-l）loss 收敛、`dinov3` 桶 LR 生效。

### PR-4：FSC encoder 上线 + 导出 + 实验脚本

- `fsc_encoder.py` 切原生（PAN SFS-FF，Eq.10–11）；`convert_to_deploy` FFT 降级臂；`scripts/visdrone_fsdc_800.py`（孪生 `visdrone_deimv2_800.py`，唯一差异 = `fsdc` 字段与 model 名）；ONNX 验证。
- **验收**：导出成功（或降级臂量化）；孪生测试（`TestScriptTwin` 模式）通过。

### PR-5：消融执行与文档收口（§1.4 矩阵）

- a：DEIMv2-S 基线（现有）→ b：fsdc-l 关闭 fsdc（= 宽模型基线）→ c：+DBFSAF → d：+SFS-FF/FSD-Down → e：完整 vs `sta_levels=4` → f：`freeze_wavelet`/`fdconv_lite` 负控。
- **验收**：本文档状态更新 + 实测回填（参数量/步时/APs 分解）。

---

## 7. 测试计划（`tests/test_fsdc_native.py`）

| 用例 | 断言 |
|---|---|
| `test_swap_faces_complete` | 双侧原版拷贝形态 == stock DEIMv2 前向逐比特（替换面完整性证明） |
| `test_fsdc_none_is_stock` | `fsdc=None` 时 state_dict/前向与 `Deimv2ObjectDetection` 一致 |
| `test_dbsaf_eq_forms` | Eq.4 softmax 维度=2C；Eq.8 切分映射 {1/2→d3, 3/8→d2, 1/8→恒等}；Eq.9 无残差（短路置零后输出仍非零） |
| `test_fsddown_eq_forms` | Eq.12 门为 8C 共享、chunk 2 后分配正确；Eq.13 单输出投影；`freeze_wavelet` 时 DWT 参数 requires_grad=False |
| `test_haar_reconstruction` | IDWT 完美重构（偶/奇尺寸）；可学习模式下初始化 == Haar 基 |
| `test_sfsff_pan_shapes` | (C,H,W)+(C,2H,2W) → (C,H,W)；通道切分和 = C；FSD-Down 内嵌对齐 |
| `test_fsc_encoder_contract` | encoder 输出层数/宽度/stride 与 `config.encoder_in_channels` 契约一致；AIFI/FPN 段与【HF】逐行语义等价 |
| `test_lr_buckets` | 参数名桶：`conv_encoder.backbone.*`→dinov3(0.05×)；STA/DBFSAF/encoder→detector(1×)；无参数落错桶 |
| `test_registry_guards` | `use_sfs_ff=False` + `use_fsd_down=True` → ValueError；未知变名 → ValueError |
| `test_warmstart_partial_load` | fsdc-s：decoder 侧键命中 COCO 检查点、骨干/encoder 键 fresh（missing-keys 白名单断言） |
| `test_fft_export_two_arms` | `torch.export` 直通；ONNX DFT 可用性探测 + 恒等臂一致性 |
| `test_checkpoint_roundtrip` | 全开训练 2 步 → 存/载 → loss 连续 |
| `TestScriptTwin` | `visdrone_fsdc_800.py` vs `visdrone_deimv2_800.py` 逐字段 diff == {model, fsdc} |

---

## 8. VisDrone 配方与消融（§1.4 矩阵的落地）

**基座**：`scripts/visdrone_fsdc_800.py` 复制 `scripts/visdrone_deimv2_800.py` 全部配方（800×800、global batch 16 = micro 4 × accum 4、cosine、EMA warmup 200、`eval_max_dets=500`、scale jitter 锁 800、无 Mosaic/MixUp），差异仅 `model="fsdc/fsdc-l"` + `model_args["fsdc"]={...}`。

**协议注意**：论文① maxDets=300 vs 本仓 500——对比论文数字时加跑 `eval_max_dets=300` 评测臂；APs 口径（论文① small = 面积 < 32² 的 COCO 定义沿用）在评测脚本中确认一致。

**记录**：mAP/AP50/APs/APm/APl、参数量分解（ViT/STA/DBFSAF/encoder/decoder 五桶）、步时延、EMA-vs-online（复用 `scripts/diag_ema_vs_online.py`）、`deploy()` 两臂 AP。

---

## 9. 风险与开放问题

| # | 风险 | 缓解 |
|---|---|---|
| R1 | 官方代码未放出，G4/G6/E 系列解释性偏差 | 替换面等价测试钉死接线正确性；组件级歧义已收敛到两处（SRM 细节、算子分配标注），放出后逐项对齐 |
| R2 | DEIMv2-L 官方配置/权重不可得（PR-0 落空） | 回退链：L→M→S（fsdc-s 全程可落地）；d_model 384 的宽度假设由 PR-0 断言 |
| R3 | decoder fresh 训练（L 无 warm-start 时）拉低绝对值 | 判据改为相对 fsdc 关闭孪生臂，不与论文①绝对值硬比；fsdc-s 臂提供 warm-start 参考 |
| R4 | ViT 分支随机失活预训练统计（替换骨干后 BN/LN 域差） | DINOv3 为 LN 无 BN 域差；STA 全 fresh 无此问题；smoke 期监控梯度范数 |
| R5 | 8k 步 pilot 不足以显现频域模块收益（论文① 100ep） | c/d/e 臂同预算互比有效；必要时对最优臂追加一次 24k 步长跑 |
| R6 | FFT 数值/导出生态 | fp32 island + 两臂导出验证（21 册机制） |

**开放问题**（评审时定）：

1. `sta_levels=4` 的 stride-4 语义层来自 patch 网格 ×4 上采样，语义密度低——是否改为只扩 STA 不扩 ViT tap（第 4 层 semantic 用最近 tap 复制）；
2. `fsdc-l` 若 PR-0 确认无 COCO 转换，是否直接以 `fsdc-m`（如存在）为主锚；
3. CopyBlend 增广是否单独立项（G8）——若 c→d 增益显著而绝对值距论文①远，它是最优先的配方补齐项；
4. 分册 21 的 LT-DETR 路线与本册的优先级排序（组件共享，但实验资源互斥）。

---

## 10. 参考索引

**权威事实源**：

| 内容 | 位置 |
|---|---|
| FSDC-DETR 论文（Eq.1–13 模块定义、消融、配方） | arXiv:2607.05176v2（abs / html）；官方仓库 `nevereverinsomnia/FSDC-DETR`（代码未放出，watch 复核 G4/G6） |
| DEIMv2 论文（STA 双分支、变体规格、损失权重、dense O2O+CopyBlend） | arXiv:2509.20787 |
| HF 双分支骨干（权威实现） | `transformers/models/deimv2/modeling_deimv2.py:830-847`（STA）、`:1160-1204`（DINOv3ConvEncoder）、`:1181-1187`（resize 公式） |
| HF HybridEncoder（AIFI/FPN/PAN、fuse_op、SCDown） | `transformers/models/deimv2/modeling_deimv2.py:1264-1384` |
| HF Deimv2Model 装配（替换面：conv_encoder/encoder/decoder） | `transformers/models/deimv2/modeling_deimv2.py:1695-1750` |
| FDConv 移植源 | github.com/Linwei-Chen/FDConv（CVPR 2025） |
| HWD（Haar 下采样基础） | Pattern Recognition 2023 |

**VTrain 受影响面**：

| 内容 | 位置 |
|---|---|
| DEIMv2 任务模型（父类；子模块手术先例 enc_score_head） | `_task_models/deimv2_object_detection/task_model.py:95-113,116-173` |
| DEIMv2 LR 桶机制（`conv_encoder.backbone.` 片段） | `_task_models/deimv2_object_detection/train_model.py:36-75` |
| DEIMv2 变体注册表（fsdc-s warm-start 来源） | `_task_models/deimv2_object_detection/config.py:31-138` |
| DEIMv2 VisDrone pilot 配方（孪生源） | `scripts/visdrone_deimv2_800.py` |
| 共享组件层（分册 21 PR 产出） | `_task_models/object_detection_components/{wavelet_downsample,sfs_ff,fdconv,dbfsaf}.py` |
| 姊妹分册 | `docs/design/21-ltdetr-fsdc-frequency-spatial-fusion.md`（LT-DETR 宿主路线；v1.1 已按本册 Eq. 修订） |

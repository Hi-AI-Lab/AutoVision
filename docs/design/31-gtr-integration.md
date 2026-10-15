# VTrain 实现蓝图 · 分册 31：GTR 集成设计——线性注意力骨干、装配兼容性与蒸馏教师线

| 文档属性 | 内容 |
|---|---|
| 版本 | v1.0（设计评审稿，待实施） |
| 状态 | 设计完成；实施计划见 §7，验收 gate 见 §8 |
| 分析对象 | `/home/hdjs/Research/Software/GTR-main`（Intellindust-AI-Lab《Gated Token Recurrence for Efficient Dense Prediction》，代码 **MIT**；本地权重 `Dino-Dome-Research/weight/gtr/`，checkpoint **MIT**，README front-matter 同） |
| 决策依据 | 2026-10-15 方向输入：GTR 代码与 S/M/L/X 权重到位，评估三条集成路径（蒸馏教师 / 装配兼容变体 / 骨干家族引进） |
| 关联分册 | 23（跨模型蒸馏教师协议）、28（清洁溯源与 re-port 纪律）、29（产品化路线 §3.1 谱系、D1b 教师池、D10 冻结纪律）、30（任务优先目录） |
| 上游 | GTR-main（MIT）；`flash-linear-attention` 0.5.0（GTR 的 GLA 算子库，license 随 extras 引入时以 wheel 元数据复核）；DEIMv2 vit_tiny 蒸馏骨干链（Apache-2.0，GTR 权重命名 `vit_tiny*` 所指） |
| 受影响模块 | `tasks/detection/detr/`（骨干家族、decoder 变体、encoder 变体、蒸馏教师）、`registry/`（资产/模型卡）、`export/`（GLA 的 TRT 路径）、`scripts/`、`docs/provenance/`（LEDGER/NOTICE） |

> **标注约定**：【审计】对 GTR-main 源码/权重的实测事实（出处 `文件:行` 或探针脚本输出，可复查）；【设计】本册决策（附编号 G1–G…及理由）；【目标】可验收目标；【风险】风险与缓解。

---

## 1. 执行摘要与定位判断

GTR 是 DEIM/D-FINE 同团队（Intellindust-AI-Lab）的下一代实时全任务架构：softmax-free 线性注意力骨干（GLA）+ Spatial SwiGLU + 双向 token 扫描，COCO 检测 AP **S=53.6 / M=57.3 / L=58.9 / X=59.4**（EMA 口径，README 基准表），四档权重 MIT 就在本地。对 AutoVision 产品线（分册 29）的意义按价值排序：

1. **教师线（性价比最高）**：GTR det/obj365 全档 MIT 权重本地就位，是 D1b N 档蒸馏与未来各档的现成强教师——只需把 doc 23 的教师协议接到 GTR 的 ckpt 格式（`{model, ema, last_epoch}`，EMA 口径）。
2. **装配兼容变体（中等）**：GTRTransformer decoder 与我们 `dfine_official` **键形同构**；hidden-192 + dec-FFN-512 的谱系变体可高覆盖装配 GTR-S decoder 权重（§3.2 实证：104 键已精确匹配，12 个失配全部是 FFN 宽度，spec 对齐即消）。
3. **骨干家族引进（最大，可选）**：GLA 线性注意力骨干是全新家族（fla 依赖 + 可选 CUDA 扩展 + TRT plugin），也是 GTR 全部增益的来源——是否引进是产品方向级决策，本册给出完整蓝图但不默认启动。

【设计 G1】三条路径**独立成段、按 A→B→C 递进**，每段独立可验收；A 不依赖 B/C，B 不依赖 C。谱系承诺（D10）不因 C 的评估而动摇——在 G2 通过验收前，GTR 骨干不进 `SPECTRUM` frozen 档。

## 2. GTR 架构分析（代码实证）

### 2.1 总装配（`engine/gtr/gtr.py`，123 行）

`GTR = backbone → encoder → decoder` 三件套，无任务专用胶水；`deploy()` 递归调 `convert_to_deploy`（与我们 `TaskNetwork.deploy` 同约定）。独有增量：`GTR_AMP_SAFE_MODE` 环境变量支持 `encoder_decoder_fp32` / `decoder_fp32` 三档 AMP 豁免——分配敏感的 query-selection/decoder 数学可强制 FP32（【审计】gtr.py:33-58）。这对我们 AMP 训练 GLA 有直接参考价值（§9 R3）。

### 2.2 骨干核心：GatedLinearAttention（`backbone/vit_adapter.py:255-607`）

【审计】逐参数读出的机制清单：

- **门控线性注意力**（softmax-free recurrent）：`q/k/v/g` 四投影 + **低秩门控** `gk_proj`（`gate_low_rank_dim=16` 的两层 Linear）+ `FusedRMSNormGated`（gate 与 RMSNorm 融合，`gate_fn='swish'`）；`expand_k=0.5`（key 维减半）、GQA 式 `num_kv_heads` 分组、`qk_norm` 开关。
- **三种执行模式**：`chunk`（训练默认，fla 的 `chunk_gla`）/ `fused_chunk` / `fused_recurrent`（推理）。
- **三层算子栈**：① `fla.ops.gla`（flash-linear-attention 0.5.0，训练可用）；② 自研 CUDA 扩展 `gla_torch_ext`（`backbone/csrc/`，**仅部署模式启用**，经 `torch.library.custom_op` 注册 + fake impl 保图可 trace）；③ TensorRT plugin（`tensorrt_plugin/trt_plugin/gla_plugin.cu`）。`GTR_DISABLE_CUDA_GLA=1` 全局降级回 fla。
- 部署优化：`convert_to_deploy` 把 q/k/v/g/gk_low 五个投影**融合为单个大 Linear**（`vit_adapter.py:340` 附近），hidden 只读一次。

### 2.3 SpatialSwiGLU（`backbone/vit_adapter_spatial_swiglu.py`，136 行）

SwiGLU（GLAMLP）的 value 分支恢复为 2D 网格后过 **3×3 depthwise conv** 再门控——位置信息由卷积供给，因此整个变体**无 pos_embed / 无 CLS / 无 mask token**（`use_pos_embed=False`，显式 `del` 三件）。部署路径把打包的 gate/value GEMM 拆开 + `channels_last`，消掉每块一次的转置拷贝（【审计】convert_to_deploy 注释，这是他们实测延迟优化点）。

### 2.4 双向扫描（Gated Token Recurrence 的「Recurrence」）

【审计】`VisionTransformerSpatialSwiGLU._apply_bid_scan`（:88-95）：逐层对 token 序列 `x.flip(1)`（反向扫描），**奇数层且 `if_quad_dir` 时追加空间转置**（`[B,gs,gs,C].transpose(1,2)`，四方向扫描的第三/四向）；tap 输出处用 `torch.argsort(perm)` 把序列还原到规范 patch 顺序。机制 = 让因果的线性注意力在每个块覆盖不同扫描方向，两个方向成对出现时注意力覆盖完整 2D 邻域。置换跟踪 `_update_perm` 与扫描同步维护。

### 2.5 ViTAdapter 与特征金字塔

`ViTAdapter`（:933-1128，DEIMv2 系多层级联 adapter）：`interaction_indexes=[3,7,11]` 三处 tap，从 **同分辨率**（`multi_layer_same_res: True`，patch 网格全程不变）的 12 层骨干取三层特征，输出 **均匀通道**（gtr_s：[192,192,192]）。金字塔化交给 encoder。

### 2.6 GTREncoder（`hybrid_encoder.py:551-687`）

RF-DETR / LW-DETR 风格的 MultiScaleProjector：对每个目标尺度独立做 `scale_factors=[2.0,1.0,0.5]` 重采样（↑2 = ConvTranspose 通道减半；↓2 = stride-2 ConvBNAct），沿通道 concat 后 **C2f(in→hidden) + LayerNorm2d**。比我们 HybridEncoder（AIFI + CSP）轻得多；`force_contiguous` 开关记录了一个真实的部署坑（depth 任务发布权重训练时无 contiguous 拷贝，:580 注释）。

### 2.7 GTRTransformer decoder（`det/decoder.py`，1194 行）

与 `dfine_official` 同一 gateway 谱系（self-attn + deformable cross-attn 经 learned Gate 融合、无 value_proj/output_proj、FGL/DDF 头），增量：`layer_scale`（逐层可学残差缩放）、`activation: silu`、`group_detr`（DEIM 分组训练）、FFN 每档独立（512/1024/1024/2048）。criterion 新增 **loss_mal**（Matchability-Aware Loss，DEIM 系，`gamma=1.5/alpha=0.75`，替换 VFL；matcher 权重 2/5/2、alpha 0.25/gamma 2.0）。

### 2.8 训练配方（`configs/det/gtr_base.yml` + `coco_finetune/gtr_*.yml`）

【审计】base（COCO 30ep）：lr 2e-3（flatcosine，`lr_gamma 0.5`）、total_batch 128、Mosaic `policy: stop_epoch`（mosaic_epoch 15 / stop 26 后移除 photometric+zoomout+iou-crop）、**copyblend** 增强（copy-paste 系，`copyblend_epochs [4,50]`，`with_expand`）、EMA 0.9999/warmups 1000、grad clip 0.1、sync_bn、`find_unused_parameters: True`。finetune 覆盖：backbone lr **1.5e-5**（预训练骨干微调）、`use_amp: False`（S 档！）、`group_detr: 3`、权重从 `weights/vit_tiny*_spatial_swiglu_distilled.pth` 蒸馏骨干初始化。

### 2.9 缩放表（S/M/L/X，全部实证自 finetune 配置）

| 档 | 预训练教师（DEIMv2 链） | embed_dim/heads | enc/dec hidden | dec FFN | dec layers | COCO AP |
|---|---|---|---|---|---|---|
| S | vit_tiny_spatial_swiglu_distilled | 192/3 | 192 | 512 | 4 | 53.6 |
| M | vit_tinyplus_… | 256/4 | 256 | 1024 | 4 | 57.3 |
| L | vit_small_… | 384/6 | 256 | 1024 | 4 | 58.9 |
| X | vit_smallplus_… | 384/6 | 256 | 2048 | 4 | 59.4 |

教师链依赖注意：`vit_tiny*_distilled.pth` 是 **DEIMv2 蒸馏中间产物，不在本地权重目录**——但 `tools/extract_backbone.py` 可从 GTR 全模型 checkpoint 抽取骨干（这就是该工具的用途），因此骨干初始化不依赖外部下载。

## 3. 与 AutoVision 的对照（实证）

### 3.1 构件矩阵

| GTR 构件 | AutoVision 现状 | 关系 |
|---|---|---|
| GatedLinearAttention + csrc + TRT plugin | 无 | **新件**（路线 C 的全部难点所在） |
| SpatialSwiGLU / bid_scan / ViTAdapter | 无（DEIMv2 vit_tiny 谱系近亲，LEDGER P0-4 已有 DEIMv2 re-port 先例） | **新件**（但 adapter 部分与 DEIMv2 同源） |
| GTREncoder | 无（HybridEncoder 是重型近亲） | **新件**（~140 行，独立性强） |
| GTRTransformer decoder | `shared/dfine_official_decoder.py` 同 gateway 谱系 | **近亲**（差异：layer_scale/silu/FFN 每档化/group_detr 挂钩） |
| loss_mal / matcher | criterion 框架同族（VFL 位） | **增量**（MAL 替换 VFL 的 loss 一枚） |
| HybridEncoder（base 配置线） | 已有（同上游 D-FINE/RT-DETR） | 同源 |
| 训练配方（Mosaic stop_epoch/copyblend/EMA） | 增强/eraser 系已有等价物 | 配方差异，非架构差异 |

### 3.2 装配兼容性实证（2026-10-15 探针）

以 `detr/s` spec 强制 `hidden_dim=192, decoder_hidden_dim=192` 构建目标网络，对照 `gtr_s_coco.pth` **EMA** 权重：**502 源键中 104 个精确同名同形，12 个失配全部是 decoder FFN（GTR 512 vs spec 1024）**——spec 增设 `dec_dim_feedforward=512` 变体后这 12 项同名同宽即匹配；decoder 覆盖率随即从 0.55 大幅上升（FFN 参数占 decoder 近半）。骨干键形不同构（GTR 自研块），骨干权重不参与装配（与 dfine_s 路线同构处理：骨干走自己的预训练线）。

### 3.3 依赖面

新增运行时依赖仅 `flash-linear-attention==0.5.0`（GTR pin；训练路径必需）；CUDA 扩展与 TRT plugin 均为**可选部署加速件**，官方已有环境变量降级路径（`GTR_DISABLE_CUDA_GLA=1`）。torch pin 2.11/cu128 是 GTR 基准环境，非我们矩阵（分册 10 §16：torch 2.4–2.7）——fla 0.5.0 在我们 torch 矩阵上的兼容性是 G2 前置验证项（§9 R5）。

## 4. 溯源与许可分析

- **代码 MIT**（`GTR-main/LICENSE`，Copyright 2026 Intellindust-AI-Lab）：允许直接 re-port 保留版权与许可声明——比 AGPL 宽松，**无 clean-room 义务**，但仍按分册 28 纪律执行：逐文件 re-port + 文件头归属（「based on GTR … MIT, modified by AutoVision」）+ LEDGER 登记（来源/许可/整改状态）+ NOTICE 增 GTR 与 fla 条目。
- **权重 MIT**（HuggingFace 仓 front-matter `license: mit`）：教师/装配/重发布均可；模型卡（分册 29 §3.4 D4）license 字段填 `MIT`。
- **依赖链**：`fla`（flash-linear-attention，随 extras 引入时核验 wheel license）、自研 CUDA ext（随 GTR-main MIT 一并归属）。DEIMv2 vit_tiny 骨干链为 Apache-2.0（LEDGER 已有引用先例 P0-4）。
- 【设计 G2-溯源】新增 `extras: gtr = ["flash-linear-attention==0.5.0"]`（惰性加载，缺依赖时骨干清单为空——沿用模型包惯例）。

## 5. 与现有谱系/训练面的接线设计

### 5.1 路线 A：蒸馏教师（G0）

- 教师加载器扩展（doc 23 协议的 GTR 适配）：读 `{model, ema, last_epoch}`，**默认取 `ema.module`**（README 明示基准数字为 EMA 口径），键前缀剥离 `backbone.backbone._model.`（骨干只用于特征教师时）或整网加载（logit/特征双教师时）。
- 教师注册：`registry/` 增本地资产条目（路径指向 `Dino-Dome-Research/weight/gtr/det/*.pth`，sha256 钉住，license=MIT）——沿用 `registry/apache_assets.py` 模式。
- N 档脚本 `train_spectrum_n_distill.py --mode distill` 的拒绝门禁改为可用状态。

### 5.2 路线 B：装配兼容变体（G1）

- `variants.py` 增 `gtr-s` 宽度配方：`hidden_dim=192, enc/dec hidden 192, dec_dim_feedforward=512, dec_layers=4, dec_num_points=(4,4,4)`（实证自 gtr_s EMA 探针），`decoder_default="dfine_official"`；SPECTRUM 增注（**planned，不冻结**——等 G1 验收）。
- 装配器复用 `dfine_official_warm_start.build_dfine_official_warm_start`，新增 `source_format="gtr"` 分支（EMA 取权重 + 前缀剥离 + FFN-512 spec）。
- backbone 侧：GTR-S 骨干权重经 `tools/extract_backbone.py` 同款抽取 → `backbone_weights` 既有入口（`_load_backbone_weights` 已支持 safetensors/plain sd + 前缀漂移重对齐）。

### 5.3 路线 C：GTR 骨干家族（G2，可选，需方向决策）

- 新家族 `family="gtr"`（`backbone_package="gtr"`）：re-port `vit_adapter.py`（1128 行，最大单件）+ `vit_adapter_spatial_swiglu.py` + `GLAMLP`；fla extras；`convert_to_deploy` 链路对齐我们 `TaskNetwork.deploy`；导出面需评估 GLA 的 ONNX/TRT 表达（官方 TRT plugin 是现成答案，但引入 `.so` 分发是新的部署矩阵项）。
- 谱系动作：G2 验收前 `SPECTRUM` 不动；验收后按 D10 以**新 minor + 新权重条目**方式引入候选档（例如 `detr/gtr-s`），不覆盖现有 frozen 档。

## 6. 训练配方映射（GTR → AutoVision）

| GTR 配方项 | 值 | 我们对应物 | 差异处置 |
|---|---|---|---|
| lr / 调度 | 2e-3 flatcosine γ0.5 | flat-cosine 已有（`multistep_schedule`/Dome LR） | 参数化即可 |
| Mosaic stop_epoch | 15/26 移除三增强 | box-aware eraser / step-window 调度（分册 16） | 配方等价，保留我们系 |
| copyblend | copy-paste + expand | eraser 系近似 | 保留我们系（baseline 忠实优先） |
| group_detr 3 | 分组 DN | denoising 已有，分组未开 | decoder 参数化增量 |
| EMA 0.9999/1000 | 同 | 同（engine 既有） | 无差异 |
| use_amp False（S finetune） | — | AMP bf16 默认开 | GLA 数值敏感，G2 需 AMP 矩阵（§9 R3） |

## 7. 实施计划

| 里程碑 | 周期 | 内容 | 退出门禁 |
|---|---|---|---|
| **G0 教师线** | 1 周 | GTR 教师加载器（EMA 口径 + 前缀剥离）接入 doc 23 协议；本地资产注册（sha256/license）；N 档 distill 门禁解锁 | 教师 forward 对拍（logit/特征一致性单测）；N 档蒸馏冒烟（coco8，教师常数 loss 下降）；模型卡可生成 |
| **G1 装配兼容变体** | 1–2 周 | `gtr-s` 宽度配方（hidden 192/FFN 512/4 层/(4,4,4)）；装配器 `source_format="gtr"`；骨干抽取接线 | 装配覆盖率实测 ≥0.9（decoder 参数量口径）并写入报告；detr/gtr-s 谱系键 planned→frozen 评审；coco8 闭环 |
| **G2 骨干家族（可选，需方向决策）** | 3–4 周 | re-port vit_adapter（MIT 归属头）+ SpatialSwiGLU + GLAMLP；fla extras；AMP/离线/导出矩阵 | COCO 短程复现 GTR-S 趋势（12ep 蒸馏 ≥ 官方 30ep 的 85% AP）；TRT FP16 导出 + 对拍门禁过；LEDGER/NOTICE 全套 |

依赖：G0 独立可先行；G1 依赖 G0 的资产注册；G2 依赖 G1 的 decoder 变体验证且需用户拍板（谱系方向）。

### 实施记录（2026-10-15，G0 + G1 代码侧落地）

- **G0**：`registry/gtr_assets.py`——四档 det COCO 资产注册（本地路径 + **sha256 实测钉住** + MIT + `gtr_ckpt_v1` 格式 + COCO AP 元数据，`verify_gtr_asset` 强校验）；`detr/gtr_teacher.py`——教师包加载器（**EMA-first 语义**、键清单报告 hidden/classes/last_epoch、`extract_gtr_backbone` 前缀剥离、`decoder_decoder_state` 装配交接、`teacher_forward` 契约存根：GLA 后端缺位时显式 `GTRBackendUnavailableError` 指向 G2）。教师模型卡可生成（MIT + sha256 + limitation 注记）。
- **G1**：`variants.py` 增 `gtr-s` 宽度配方（hidden 192 / dec FFN 512 / 4 层 / points (4,4,4)，骨干仍走 timm HGNetV2 线）；装配器增 `source_format="gtr"`（EMA 取权重）；`SPECTRUM["gtr-s"]` planned（组件键已存在、档位冻结待评审——与 m/l/x 的「无键 planned」语义区分）。**覆盖率实测 0.9613 ≥ 0.9（G1 门禁达成）**：144 张量传递、3 个形状跳过。
- **测试** `tests/test_gtr_integration.py` ×14（sha256 实核、EMA≠model 抽键断言、骨干前缀剥离、GLA 契约拒绝、装配覆盖门禁、部分装配拒载、detr/gtr-s coco8 训练闭环、教师模型卡；9.65s）。回归 p1/p2/train_object_detection 41 过；ruff/layering/provenance 全绿。
- **G0 gate 诚实修正**：原「教师 forward 对拍（logit/特征一致性单测）」依赖 GLA 推理路径（fla + vit_adapter 重移植），**并入 G2 前置项**；G0 交付状态级对拍（EMA≠model、键集一致、sha256）+ 契约存根。蒸馏冒烟（coco8 教师 loss 下降）同样随 G2 解锁。
- **G2 保持待方向决策**（分册 §1 G1 设计判断），未启动。
- **G2 代码侧实施记录（2026-10-15 同日，用户拍板继续）**：MIT re-port 三件入 `tasks/detection/shared/gtr/`（vit_adapter 1152 行 + spatial_swiglu + encoder 345 行，归属头 + LEDGER 登记 + NOTICE 增补）；fla 惰性守卫（模块可无依赖 import，构造期报错带安装指引）；pyproject `gtr` extras（flash-linear-attention==0.5.0）；家族接线 `family="gtr"` + `detr/gtrvit-s` 组件档（SPECTRUM 不动，G2 验收前不冻结——D10）；task_model 骨干/encoder 双分派（GTRViTBackbone + GTREncoder 金字塔）；`backbone_model` 指向内层 `_model` 使 `extract_gtr_backbone` 键形免重映射直载。**CUDA 实测**：gtrvit-s 全网 forward + coco8 3 步训练闭环 + GTR→GTR 骨干直载（零 unexpected keys）；测试 `tests/test_g2_gtr.py` ×6（fla+CUDA 门控）+ G0/G1 ×14，合计 54 过、三守卫绿。**G2 剩余（GPU 矩阵）**：COCO 短程趋势（≥官方 85% 判据）、TRT FP16 导出对拍、AMP 三档豁免矩阵。fla 0.5.0 已装入 dev venv（uv），pyproject extras 为声明面；uv.lock 再生留待发版流程。
- **G2 GPU 实测记录（2026-10-15，RTX 4090D 24GB）**：① **AMP 探针**：gtrvit-s@640 纯网络 forward+反向在 fp32/bf16/fp16 三档全部数值有限（440 梯度张量）——R1/R3 数值风险实测低于预期，官方 S 档关 AMP 的保守性或与 MAL loss 数学相关，训练期仍建议 AMP 矩阵复验；② **COCO 短程判别实验**（真实 COCO train2017，bf16，bs16；COCO 图片目录曾为空已从本地 zip 恢复 118,287/5,000 张）：`gtrvit-s` 全随机对照 500 步 loss 41.6→42.9 波动、val≈0.0008（ViT 系从零无信号，符合预期）；**`gtrvit-s` + 官方 GTR-S 骨干直载（`extract_gtr_backbone`→`backbone_weights`）2000 步 val 单调升 14.7→19.0 AP / 31.1 AP50**，较同日程 `detr/s`（HGNetV2-b1 + dfine_s 装配，13.0 AP）**+6.0 AP**——GLA 集成正确性强证据，GTR 骨干质量优势兑现（V3 趋势判据短程证据，完整 85% 判据待全程训练）；③ 冒烟配方三偏差定位并修正（EMA warmup 2000 未完即评估→`ema_warmup_steps` 参数化 / 骨干预训练缺失→直载链 / 步数 0.45%）；④ `docs/benchmarks.md` 首填：torch eager 延迟表（4 模型 × bs1/4 × fp32/fp16；GLA fp16 慢于 fp32 的 Triton 核特性入表注）+ 短程趋势冒烟表（明确标注非发布口径）。**仍待**：TRT FP16 对拍（onnx/tensorrt 未装）、全程训练、uv.lock 再生。
- **部署工程迁移记录（2026-10-15 晚，G2 剩余矩阵三项的实测落地）**：① **自研 CUDA GLA 核迁移完成**——`shared/gtr/csrc/`（nvcc 13.0 与 torch cu130 匹配；setup.py `-std=c++17→c++20` 补丁适配 torch 2.14；官方 test_gla **ALL PASS**；`.so` 经包 `__init__` sys.path 注入，vit_adapter 既有 `import gla_torch_ext` 零改动生效）；**骨干级实测 640px FP16 bs1：fla Triton 16.25ms → deploy+自研核 7.97ms（2.04×）**（recursive convert_to_deploy 28 钩子）；② **torch.compile 实测负收益**（默认模式 9.62ms，GLA custom op 图断裂；CUDA Graphs 模式被拒——custom op `mutates_args={"out"}` 原地写需改接口；官方 1.225ms 依赖 py≥3.11 compile+graphs+其引擎层 graph 封装）；③ **整网 half 与 decoder bbox FP32 精修块（fp16 溢出保护）冲突确认**——正确部署形态是混合精度导出（TRT），整网 half eager 不支持。口径差距（33ms eager→官方 1.225ms）分解完毕：自研核（已迁移，2×）+ compile/graphs（需 py3.11 + op 接口改造）+ 预处理/融合（TRT 路径覆盖）。**G2 剩余更新**：TRT 对拍、全程训练、op 接口改造（graphs）、Python 3.11 评估、uv.lock。

- **金标准复核对拍与本轮修复（2026-10-16，官方 GTR-S 权重直载验收）**：对 `gtr_s_coco.pth`（EMA 口径）直载 `gtrvit-s` 并做同机同数据 COCO val 实测，共发现并修复 **6 处谱系偏差**（全部是形状不可见的语义量，shape-driven 装载全部"成功"）：
  1. **`num_points`**：官方两谱系均为 **[3, 6, 3]**（D-FINE `dfine_hgnetv2.yml:62`、GTR `gtr_base.yml:52`），我们配方是 (4,4,4)——展平后 `sum=12` 形状恒等，逐级 offsets 静默错位。三处配方修正 + 注释证据（"split 而非 sum 是装载契约"注记）。
  2. **`query_pos_head`**：GTR 改为 3 层、hidden 宽、带 `act` 激活（D-FINE 为 2 层 2H）——`gtr_compat` 模式按谱系构造。
  3. **`norm2→norm3`** 重命名：装载器按 `source_format="gtr"` apply；同时修正 `_normalize_key` 使 `cv1→conv1` 仅对 dfine 生效（GTR encoder 重移植保留 `cv1` 原键）。
  4. **`enc_output` 不存在于 GTR**：选择直接吃 encoder memory；评估选择用**组头 `enc_score_head_group[0]`/`enc_bbox_head_group[0]`**（group_detr=3 下共享头是未训练模板）——`gtr_compat` 构造组头并跳过 proj/norm。
  5. **`silu` 全链贯通**：层内、各头、LQE 的激活（state_dict 不可见）。
  6. **query_pos_embed 层间冻结**：GTR 在循环前用初始 refs 算一次、全层复用（`decoder.py:543-544`；其 deploy `pos_qk` 融合注释明示 "query_pos_embed is constant across layers at inference"）；D-FINE 为循环内逐层重算（`dfine_decoder.py:406`）。**逐层重算版本实测 COCO val 45.07（−8.5 AP）**——本轮最大的单体损失项；已按 `dec_lineage` 分流（`frozen_query_pos`）。
- **对拍证据链**：骨干+encoder 逐比特一致（0.0）；选择链（memory/anchors/topk）逐比特一致；层模块**同输入 0.0**（权重逐张量 0.0 差，仅 norm2/norm3 键名异）；同输入全网对拍修后 **logits 3.8e-4 / boxes 2.8e-6 / scores ~1e-6**（残余为饱和锚点 `inverse_sigmoid` 对 1e-7 级 fp 噪声的放大——官方实现固有属性，非结构差）。`_sa_pos_qk_w` 融合自注意力与 value-atlas 均为 `convert_to_deploy` 专属（普通 eval 走经典路径），无需移植。
- **验收实测（RTX 4090D，同机同数据）**：官方引擎 `--test-only`（其评估器/其 transform）**AP 53.6 = README 数值同机复现**；我方评估器 × 官方 net：val500 57.97（我方预处理）/ 58.07（官方 transform）；**官方 net × 官方 transform × 我方（crowd 修复后）评估器：全量 53.64**——与我方全链（我方 net × 我方预处理）**53.61** 互证（网络/评估器/预处理三方闭环，预处理全量口径差 0.03 ≤ 噪声）；我方 net × 官方权重：val500 修复前 49.72 → **修复后 57.97，与官方 net 逐位一致**（map/map_50/map_75/mar 全同位）；**全量 val2017：修复前 45.07 → qpe 修复后 52.85 →（叠加 crowd 口径修复）53.61，官方引擎同机 53.6 —— 最终对齐**。
- **附带修复 1（评估侧缺陷，独立于 GTR）**：`TaskNetwork.predict`/`predict_batch` 缺 `@torch.no_grad()`（5000 图 COCO 评估保留计算图 → 24GB GPU OOM）且不把输入搬到模型设备（`.to("cuda")` 后 path 输入崩溃）——均已修复 + 回归测试（`tests/test_p2_usability.py::TestPredictInferenceContract`）。
- **附带修复 2（COCO 协议口径，独立于 GTR）**：评估器对全部 GT 写 `iscrowd: 0`，val2017 的 446 个 crowd 框（1.2%，大框居多）被当普通 GT 计分（漏检计 FN、重叠检测计 FP）。`iscrowd` 现已贯通 `list_image_info → evaluate_object_detection → update_targets → pycocotools`（官方 ignore-region 语义），单测覆盖；训练内 val 走 dataloader 张量（无 crowd 位），仍按普通 GT 计分（保守口径，后续可随样本 schema 一并贯通）。
- **防再犯**：AVWarmStart payload 记录 `dec_num_points`/`dec_lineage`，`load_warm_start_into_network` 拒绝形状恒等的语义不匹配（sum-only 陷阱在装载层不可探测）；`tests/test_gtr_integration.py` 增谱系冻结断言（gtr→同一 qpe 对象跨层 / dfine→逐层新对象）+ payload 记录/拒载测试。
- **注**：我方预处理（tensor bilinear antialias）与官方（PIL bilinear）val500 单点差 0.10 AP、全量口径 0.03（官方 net × 官方 transform 全量 53.64 vs 我方全链 53.61，≤噪声）——预处理为全家族共享语义，不为单模型改；修复前训练的 `gtrvit-s` 冒烟权重语义为逐层 qpe，重训即对齐（非发布物）。

- **S1 实施记录（2026-10-16，GTR 多任务线首切片，详见 §11）**：① 资产面扩至全任务组——`registry/gtr_assets.py` 26 份权重（det 4 + obj365 4 + seg 4 + pose 4 + depth 4 + obb 2 + semseg 4）sha256 实测钉住 + MIT + README 度量元数据（obj365 无发布口径数字如实留空）。② **semseg 家族落地**：`tasks/semantic_segmentation/gtr/`（head re-port + config/task_model/train_model，`variant s/m/l/x` 宽度表与官方配置一致；官方装载器 EMA-first、`encoder.*`/`head.*` 覆盖率门禁 1.0、backbone 经 `extract_gtr_backbone` strict 直载）；CLI dispatch `gtr/<variant>`。③ **金标准对拍（RTX 4090D）**：官方 `GTRSemSeg`（其引擎 + 官方 EMA 权重）与本家族同输入 logits **逐比特一致（maxdiff 0.0；随机 512 / 真实图 512 / 真实图 512 batch=2 / 真实图 1024）**，argmax 一致率 1.0；**逐档复验 s/m/l/x 共 16 组对比全部 0.0**，四档官方权重全覆盖装载（40 张量/覆盖率 1.0）。④ **逐档验证暴露并修正一处宽度陷阱**：L/X 档官方 encoder/head 宽度是 **256**（encoder 把 384 宽 ViT 投影降到 256；官方配置 `GTREncoder.hidden_dim: 256`），S/M 才是 hidden==embed_dim——宽度表已扩为四元组并锁定。⑤ 错配防护：跨档检查点（如 S 权重进 L 网络）以定制 ValueError 拒绝（首块形状差异提示），非 torch 原始刷屏；`predict()` 全链（预处理→forward→argmax→原图尺寸掩码）实测通过。⑥ `tests/test_gtr_semseg.py` ×10（含四档装载参数化 + 跨档拒绝）；LEDGER/NOTICE 增补该 re-port。**S1 剩余**：Cityscapes 滑窗评估（1024/768）与官方 81.5 mIoU 的本地复现（数据到位后）。

## 8. 验收 gate

| # | 判据 | 证据物 |
|---|---|---|
| V1 | GTR-S 教师在 N 档 12ep 蒸馏中可用且增益可测（vs baseline 锚 ≥ +1 AP） | 训练记录 + 基准表新行 |
| V2 | `detr/gtr-s` 装配覆盖率 ≥0.9、coco8 闭环、模型卡齐全 | 装配报告 + 测试 |
| V3 | （若 G2 启动）12ep 蒸馏达官方 30ep AP 的 ≥85%，TRT 导出对拍过 | runs + 对拍报告 |
| V4 | 溯源全套：LEDGER 逐文件、NOTICE 增补、extras 隔离 | 溯源扫描绿 |

## 9. 风险登记册

| # | 风险 | 概率/影响 | 缓解 | 兜底 |
|---|---|---|---|---|
| R1 | GLA 训练数值敏感（官方 S 档 finetune 直接 `use_amp: False`） | 中/高 | G2 引入 `GTR_AMP_SAFE_MODE` 同款三档豁免；AMP 矩阵单列 | bf16 全程 FP32 decoder |
| R2 | fla 0.5.0 与我们 torch 2.4–2.7 矩阵不兼容（GTR pin torch 2.11） | 中/高 | G2 前置 wheel 兼容验证；extras 隔离 | 降级 `fused_recurrent` 纯 PyTorch 路径（官方自带） |
| R3 | CUDA ext / TRT plugin 的 ABI 与分发负担 | 中/中 | 部署加速件保持可选（官方降级开关）；不进默认轮子 | 纯 fla 路径发布 |
| R4 | 教师口径错误（model vs EMA）导致蒸馏增益虚低 | 中/中 | 加载器默认 EMA + README 口径注记；对拍单测 | 双口径 A/B 记录 |
| R5 | MIT 源码演进的同步负担 | 低/中 | re-port 时逐文件钉 commit；LEDGER 记录上游版本 | 不跟踪上游，独立演化 |
| R6 | 谱系方向被 GTR 牵引偏离 D10 冻结 | 低/高 | 本册 G1/G2 产物一律 planned 起步；frozen 变更走 minor + 新权重条目 | 谱系评审会否决 |

---

## 10. 维护约定

- 本册为 GTR 集成的单一定义源；G0/G1/G2 实施记录追加至 §7 表后（附日期），重大偏差按 ADR 落档。
- 谱系相关变更（`gtr-s` 冻结、`detr/gtr-s` 键）同步分册 29 §3/§12 与 `variants.py` 注释。
- 上游（GTR-main）版本钉住记录进 LEDGER；re-port 文件头保留 MIT 归属。

---

## 11. GTR 多任务面分析与集成计划（2026-10-16）

GTR 不止检测：`engine/gtr/` 下另有 seg（实例分割）/pose/depth/obb/semseg 五条任务线，本地权重目录（`weight/gtr`）六任务组 26 份全部齐备（含 obj365 预训练阶段）。本节为分析 + 分级计划；实施记录追加至 §7。

### 11.1 家族结构与共享面（实测）

全任务共享同一 backbone（`ViTAdapterSpatialSwiGLU`）与 `GTREncoder`；差量集中在 head/decoder。与 `gtr_s_coco` 的**键集关系**（2026-10-16 实测）：

| 任务 | 键数 | 与 det 共享 | 新增/差异 | 架构差量 |
|---|---|---|---|---|
| det | 502 | — | — | —（已集成，见 §7） |
| **obb** | 502 | **502（键名完全相同）** | — | 键名与 det 全同，但**框路径宽度不同（实测）**：`dec_bbox_head` 输出 198=6×33、`lqe_layers.reg_conf` 输入 30=6×(k+1)——**6 参旋转框**（det 为 4 参）；注意力/FFN 主干 141 键直通。移植面 = 框路径 4→6 + 旋转数学（`rbox_ops` 243 行 / criterion 468 行 / matcher / postproc） |
| **obj365** | 502 | 502 | — | det 架构的**预训练阶段**（365 类；head 形状随类数变）；loaders/评测无需新代码，作任意任务的微调初值（README 表无发布口径数字，注册表如实留空） |
| **seg** | 537 | 502 | **+35：`decoder.decoder.*` mask 头** | det 全栈 + `mask_downsample_ratio` 版 `SegmentationHead`（DepthwiseConvBlock×层 + query einsum）+ mask CE/Dice 损失与 matcher mask 代价（point-rend 点采样） |
| **pose** | 467 | 365 | 独立 decoder（`enc_pose_embed`/`enc_out_class_embed`…） | 专属 decoder（833 行）+ keypoint 损失/matcher/dn/ms_deform_attn/postproc |
| **depth** | 358 | 319 | DPT head（`layer_rn/refinenet{1,2,3}/output_conv*`，39 键） | **无 decoder**：encoder 金字塔 → Depth-Anything-V2 式 DPT 头 + SiLog 损失 |
| **semseg** | 266 | 258 | FCN head（8 键：`head.{conv,bn,classifier}`） | **单层 stride-8 encoder**（`scale_factors:[2.0]`）+ mmseg FCNHead 式头 + CE（ignore 255） |

README 口径（S 档）：det 53.6 AP → seg 45.0 mask AP → pose 70.1 kp AP → depth NYU δ1 0.946/AbsRel 0.074 → obb DOTA-v1.0 test 80.0 AP50 → semseg Cityscapes 81.5 mIoU（1024 滑窗 / stride 768）。semseg/seg 全程 `use_amp: False`（与 det 相同）；depth 权重训练时 `GTREncoder` **无 contiguity 拷贝**（其源码注释明示；**2026-10-16 实测：`force_contiguous=False` 下共享面与官方逐比特一致**——该陷阱已按实测钉住）。

### 11.2 我方任务面对照

| 任务 | 我方任务面 | 本地数据 | 度量 | 集成差量 |
|---|---|---|---|---|
| semseg | `semantic_segmentation`（linear + eomt 家族） | 无 Cityscapes（成对 images/masks 格式可承接转换） | `SegmentationMetricCollector`（mIoU）✓ | head 8 键；滑窗评估待补（S1 记录单窗口径） |
| seg | `instance_segmentation`（eomt 家族） | **COCO ✓**（instances 含多边形） | `COCOMaskAPEvaluator`（pycocotools segm）✓ | mask 头 + mask 损失/matcher/后处理；decoder 复用（已 gold-check） |
| obb | `oriented_detection/rtdetr` | 无 DOTA（VisDrone 为水平框） | `metrics/oriented_detection.py` ✓ | 旋转框数学全链移植；权重键集≡det，warm-start 直通 |
| depth | `depth_estimation`（**空家族**） | 无 NYU | **缺** AbsRel/RMSE/δ | DPT 头 + SiLog + 新家族/数据/度量/CLI（README 为六视图 + log-affine fit 自研协议） |
| pose | `pose_estimation`（**空家族**） | COCO keypoints ✓ | **缺** OKS/AP | 独立 decoder 全量移植 + kp 数据/度量/CLI（最大切片） |

### 11.3 分级实施计划（S1–S5）

| 切片 | 内容 | 退出门禁 | 状态 |
|---|---|---|---|
| **S1 semseg** | head re-port（8 键）+ 家族四模块（`gtr/s\|m\|l\|x`）+ 官方装载器（覆盖率 1.0 门禁）+ CLI dispatch | 键形状核对 + **金标准对拍逐比特（maxdiff 0.0）**+ CUDA 训练闭环 | **已交付（2026-10-16）**，见 §7 记录 |
| **S2 seg** | `SegmentationHead`（35 键；**实测 decoder 侧 218 键中 181 键由现有代码吸收（84.9% numel），缺口=37 键 mask 头**）+ decoder mask 路径（`_OfficialFDRDecoder` 可选 mask 输出，det 行为不变）+ mask CE/Dice + matcher mask 代价 + mask 后处理 | COCO mask AP 对齐官方 45.0（±0.5，可端到端实测）；det 回归门禁不破 | 计划（推荐下一个） |
| **S3 obb** | `rbox_ops` + 旋转 decoder/criterion/matcher/postproc 移植；权重复用 warm-start（键集≡det 即通） | 旋转框自检（角度周期/边界）+ 无标签对拍（同输入 vs 官方 net）；若获 DOTA 则 AP50 对齐 80.0 | 计划 |
| **S4 depth** | DPT 头（**实测缺口=39 键，无 decoder**）+ SiLog + 新家族；`force_contiguous=False` 已实测钉住 | NYU 六视图 + log-affine 复现 δ1 0.946（±0.005）；对拍在此配置下进行 | 计划 |
| **S5 pose** | 独立 decoder（833 行）+ kp 损失/matcher/dn/postproc + COCO kp 数据面 + OKS 度量 + CLI | kp AP 对齐 70.1（±0.5） | 计划（最大切片，最后做） |

依赖：S2/S3 均可独立先行（S2 有本地数据 vs S3 键集直通）；S4/S5 需先补度量/数据模块。obj365 四档任何时候可作为 backbone-init 资产（`resolve_gtr_asset("gtr_s_obj365")`）。

### 11.4 共享面全任务实测（2026-10-16，RTX 4090D）

对六个任务组的官方 EMA 检查点逐一执行：按**该任务自己的官方配置**构造我们的 GTRViTBackbone + GTREncoder，装载该检查点的 backbone/encoder 状态（strict），与官方引擎同输入（512²）比较共享面输出：

| 任务 | backbone+encoder 前向 | encoder 键装载 | decoder/head 侧吸收（含 norm 重命名） |
|---|---|---|---|
| det | **逐比特一致（0.0）** | 0 missing / 0 unexpected | 173/183（98.7% numel；缺口=anchors/valid_mask 缓冲） |
| seg | **逐比特一致（0.0）** | 0 / 0 | 181/218（84.9%；缺口=37 键 mask 头） |
| pose | **逐比特一致（0.0）** | 0 / 0 | 38/148（39.4%；独立 decoder） |
| obb | **逐比特一致（0.0）** | 0 / 0 | 141/183（90.1%；40 键框路径形状差=6 参旋转框） |
| depth | **逐比特一致（0.0）**（`force_contiguous=False`） | 0 / 0 | 0/39（无 decoder；DPT 头全新） |
| obj365 | **逐比特一致（0.0）** | 0 / 0 | 164/183（81.8%；17 个类目相关形状差）——**可作 det 微调初值**（类头重建） |

即：**六任务组的共享核心（backbone+encoder，占比最大）已在我们的实现里逐一验证到逐比特**；剩余缺口即各切片已列明的 port 面（seg 37 键 mask 头 / obb 框路径 4→6 + 旋转数学 / depth 39 键 DPT 头 / pose 独立 decoder）。obj365 四档作为 det 系微调 init 的装载路径同时得到验证。

### 11.4 补遗风险登记（承接 §9）

| # | 风险 | 概率/影响 | 缓解 | 兜底 |
|---|---|---|---|---|
| R7 | depth `force_contiguous` 形状不可见差 | 高/中 | **已实测（2026-10-16）：False 逐比特一致**，S4 沿用 + 代码注释 | 双配置对拍记录 |
| R8 | pose/S2 移植面的 train-eval 不一致（如 stage 语义） | 中/高 | 对拍脚本沿用 det 方法论（层/阶段级 + 全链） | 逐阶段 bisect（§7 工具链可复用） |
| R9 | 数据集缺口（Cityscapes/NYU/DOTA） | 确定/中 | 加载正确性用**无标签对拍**（input→output 比对）；度量对齐留待数据到位 | 冒烟用合成数据 + 真实图无标签 |
| R10 | semseg 滑窗评估缺位（1024/768 协议） | 中/低 | S1 已记录单窗口径；滑窗作为独立小项 | 单窗口径明示非发布数字 |
| R11 | 家族代码跨任务 import（`semantic_segmentation/gtr` → `detection/shared/gtr`） | 低/低 | layering 契约允许（同层）；后续可归并 `tasks/shared/gtr/` | 归并作为独立重构项 |

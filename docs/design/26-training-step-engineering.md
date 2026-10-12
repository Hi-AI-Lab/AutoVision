# VTrain 实现蓝图 · 分册 26：训练步工程优化（launch-bound 算子消除与 kernel 效率）

| 文档属性 | 内容 |
|---|---|
| 版本 | v1.0（设计评审稿，未实施） |
| 状态 | 设计完成；W1/W2 为 P0 低风险项可直接实施，W3/W4 为实验/评估项各带独立门槛 |
| 上游思想 | DeepSeek 自研 DeepGEMM / FlashMLA 的工程判断：**大量 launch-bound 小算子的融合收益大于算法改进本身**（FlashMLA 对 memory-bound attention 的带宽优化、DeepGEMM 的 JIT 形状特化 GEMM）。【DS】对应到训练侧：每步的 kernel launch 数与同步次数是吞吐的一阶瓶颈，融合/批量化/异步化优先于新算子 |
| 配套分册 | DeepSeek 迁移四联册：本册（步工程）+ 分册 23（蒸馏）/ 24（FP8）/ 25（token indexer） |
| 参考实现 | 无外部移植源；测量基础 = 仓内现成的两个 profile 脚本（`scripts/profile_step_phases.py`、`scripts/profile_criterion_ops.py`） |
| 受影响模块 | `ltdetr_object_detection/train_model.py`、`object_detection_components/dfine_criterion.py`、`dfine_utils.py`、`object_detection_components/defe.py`（heatmap 循环）、新增 `_optim/` fused optimizer 开关、`tests/test_step_engineering.py`（新） |
| 基线兼容承诺 | 全部改动数值等价（浮点重排容差内）或默认关闭；`W1/W2` 落地后为默认开启（有门槛与回退位），`W3/W4` 恒为实验旗标 |

> **标注约定**：【DS】DeepSeek 仓库/技术报告已验证事实；【VTrain】本仓现状（`文件:行` 相对 `src/autovision/`）；【设计】本文档决策；【差异】D1–D5。

---

## 1. 背景：DeepGEMM/FlashMLA 的教训与本仓的瓶颈画像

【DS】DeepSeek 为 V3 训练/推理自研了三个 kernel 库（FlashMLA、DeepEP、DeepGEMM）。其共同方法论：先 profile 定位"launch-bound / memory-bound"的算子群，再以**融合、批量化、形状特化**消除，而不是从算法层重写。本仓已经自发走上了同一条路——两个 profile 脚本的存在就是证据：

- `scripts/profile_step_phases.py`：按 `backbone → encoder → decoder → criterion → backward → optim+ema` 分相计时（`:91-163`），定位粗粒度瓶颈；
- `scripts/profile_criterion_ops.py`：定位 criterion 内 per-image 循环的 launch-bound 小算子。

【VTrain】**已完成的同源工程**（本册的盘点基线，避免重复劳动）：

| 已做项 | 位置 | 机制 |
|---|---|---|
| matcher 批量 D2H | `object_detection_components/matcher.py:117-131` | 全批 cost 一次 `torch.cat(...).cpu()`，scipy 仍逐图——per-image `.cpu()` 的每图一次管线停顿被消除 |
| 匹配索引单次 H2D | `matcher.py:22-35`（`indices_to_device`） | scipy 索引一次性搬上设备，下游 VFL/box/FGL/DDF 消费者零同步 |
| `_get_go_indices` 向量化 | `dfine_criterion.py:633-686` | 官方逐行 dict 循环 → `torch.unique` + 稳定排序；消除随 VisDrone union 规模增长的 per-row Python 循环 |
| EMA foreach 更新 | `object_detection_components/ema.py:79-87` | `torch._foreach_mul_/_add_` 批量化（数百次 launch → 几次）；别名存储去重 |
| TF32 | `_engine/engine.py:588-592` | fp32 余量（EMA/优化器/非 autocast 算子）走 tensor core |
| pin_memory + 异步搬运 | `_commands/common_helpers.py:120`（`pin_memory=cuda`）、`engine.py:413-422`（`non_blocking=True`） | H2D 真异步 |
| 日志窗口在设备累计 | `engine.py:61-94`（`_TrainWindow`） | 损失张量保持 0-dim CUDA 张量，仅日志点一次 `float()` |
| `translate_gt` 权重函数缓存 | `dfine_utils.py:98-112` | 每 FGL 调用重建 CPU 张量 + 拷贝的单次同步被消除 |
| SSL 段式分组 CPU 化 | 分册 14 R8（已实施） | 逐图 Python 循环 → 一次下 CPU + 段式分组；块对角 batched 指派经渐近分析否决 |
| 验证前后显存释放 | `engine.py:911-918, 950-955` | 缓存分配器 `empty_cache` 的对称释放，消除验证 OOM 点 |
| criterion NaN 零填充单遍 | `dfine_criterion.py:573-575` | dict comprehension 一次完成 |

**剩余瓶颈画像**（两个 profile 脚本的定位结论 + 代码走读）集中在四处：criterion 的 per-image 循环群（§3 W2）、优化器 step 的 launch 数（§3 W1）、decoder 前向的算子融合度（§3 W3）、匹配的 CPU 指派（§3 W4，评估后大概率维持现状）。

---

## 2. 度量协议（先于一切改动）

【设计 D1】每个工作项的实施与验收都按同一协议：

1. **基线采样**：`profile_step_phases.py --steps 30`（Dome twin 配方，micro 4 two_crop）跑 3 次取中位数；补记 `torch.profiler` 的 kernel launch 计数（`profile_criterion_ops.py` 已有该能力）；
2. **改动后同协议复测**：步时与 launch 数双指标；
3. **验收门槛**：单项目标步时 ≥ −5% **且**数值等价测试通过（§5）；不达标回退（每项独立 flag/提交，可单独 revert）；
4. **归档**：结果回填本册 §6。

---

## 3. 工作项设计

### W1（P0）：fused AdamW

【VTrain】`get_optimizer`（`ltdetr_object_detection/train_model.py:699-704`，deimv2 同款 `deimv2_object_detection/train_model.py:231-236`）用默认 AdamW（foreach 实现：每参数组多 kernel，全模型数十 param → 每步上百 launch）。

【设计】CUDA 上 `torch.optim.AdamW(..., fused=True)`：

- fused 实现把整步参数更新折叠为每 dtype 组一次 kernel launch——优化器相位的 launch 数下降一个量级；
- **约束**：fused 要求全部参数同一 dtype 且在 CUDA（fp32 master 权重满足）；` capturable`/`fused` 与 `foreach` 互斥；LR scheduler 经 `param_group["lr"]` 更新的兼容性由 torch 保证（fused 组内 tensor lr 支持现状以测试钉住）；
- 开关：`ltetr`/`deimv2` 训练 args 加 `fused_optimizer: bool = True`（CUDA 上默认开，CPU/NPU 自动回退 foreach——`try` 构造 + 回退，warning 可见）；
- 数值语义：fused 与 foreach 在 fp32 下为同公式不同执行序——浮点重排容差内等价（§5 测试 3），**不做逐比特承诺**（D2）；
- 预期收益：optim+EMA 相位（profile 中通常 5–10%）的 30–50%。

### W2（P0）：criterion per-image 循环块对角向量化

【VTrain】criterion 里仍存留四个 per-image Python 循环群（VisDrone 密集批下每图 union 匹配数百对，B=8~16 → 每步数百次小 kernel launch + 数百次 Python 迭代）：

| 循环 | 位置 | 每步迭代数 |
|---|---|---|
| FGL 目标构建（`bbox2distance` 逐图） | `dfine_criterion.py:357-369` | B |
| FGL 逐层施加（逐图 `fg_loss_from_targets`） | `dfine_criterion.py:371-394` | B × num_entries（3 层 decoder → ~24–32） |
| dn 分支 FGL（`fg_loss` 内层循环） | `dfine_utils.py:354-375` | B × len(dn_outputs) |
| DDF 权重构建（IoU 逐图 `matched_pair_iou`） | `dfine_utils.py:420-434` | B × (num_entries−1 + dn) |
| （附带）DeFE GT 高斯热图逐图生成 | `dome_encoder.py:126-133` | B |

【设计】**块对角展平**（单点模式，与分册 14 R8 的 CPU 段式分组同一思想但留在 GPU）：

1. 逐图的 `(src, dst)` 匹配索引拼接为扁平索引，附每图段偏移 `seg_offsets`（`torch.cat([src + offset_i])` 形式）；
2. `bbox2distance` / `matched_pair_iou` / `fg_loss_from_targets` 全部按扁平 (ΣN, …) 张量**一次调用**；`matched_pair_iou` 的逐图 `box_iou` 对（pred_i, gt_i）改为扁平 `box_iou(pred_flat, gt_flat)` 后 `torch.diag`（逐图 NxM 矩阵在匹配对齐后天然是对角提取，语义等价）；
3. 逐图累加 `layer_loss +` 循环 → 扁平求和（`sum / num_boxes_go` 归一化不变）；求和顺序变化 = 浮点重排（D2，容差验收）；
4. 高斯热图：`GaussHeatmapGenerator` 向量化为网格距离计算的一次 batched 实现（`defe.py` 内新增 batched 路径，逐图路径保留给非方阵边界情况）；
5. dn 分支复用同一展平 helper（`_flatten_matches(indices, device)` 返回 `(flat_src, flat_dst, seg_offsets)`）。

**明确不做的**（记录依据，对齐分册 14 R8 先例）：matcher 的 scipy 指派本身不 batch 化——`linear_sum_assignment` O(n³)，合并为 O((Σn)³) 渐近更差（`matcher.py:54-56` 注释已记录）。

### W3（P1，实验旗标）：decoder 子模块 torch.compile

【VTrain】torch.compile 默认关闭（编译训练 step 破坏 metrics 的既有结论）。decoder 前向是纯张量代码（`dfine_decoder.py` 的 `_refine_layers` 循环），是 compile 的合理最小面。

【设计】实验旗标 `compile_decoder: bool = False`：

- `train_model.__init__` 尾部：`self.model.decoder = torch.compile(self.model.decoder, mode="max-autotune-no-cudagraphs")`（cudagraphs 与变长 query/dynamic control flow 冲突，显式排除）；
- 已知图断点：density 分支的 per-image `nonzero`/host 循环（`mwas.py:254-301`）——**density 开启时 compile 只预期部分收益**；Dome-off 配方收益更完整；
- DDP 兼容：compile 包在 DDP 内层模块上（转换发生在 `task_model` 构建期，DDP wrap 在 `engine.py:603`）——torch 官方支持顺序；
- **门槛独立**：G-W3 = 步时 ≥ −8% 且 §5 数值容差通过且 10-epoch pilot mAP 底线（−0.2 AP）；不达标即关闭并归档（实验性质，不承诺默认开启）；
- 与分册 24 的叠加实验留到两者各自独立验收之后（一次一个变量的消融纪律）。

### W4（P2，评估项）：GPU 匈牙利匹配

【VTrain】matcher 每步跑 `num_entries + pre + enc` 次全批匹配（`dfine_criterion.py:273-309`：3 层 decoder + pre + enc_aux），每次 = GPU cost 矩阵 + 单次批量 D2H + scipy 逐图。D2H 已批量化后，剩余成本 = scipy 本身（CPU，与 GPU 计算重叠有限）。

【设计】**评估先行，不直接实施**：

1. 量化 scipy 段实际步时占比（`profile_criterion_ops.py` 扩展 matcher 计时；预期 3–8%，若 <5% 直接归档否决）；
2. 若超阈，候选方案对比：
   - a. **CPU 侧并行**：scipy 释放 GIL 的部分放线程池（成本矩阵已在 CPU）——低风险，收益受 GIL 限制；
   - b. **GPU auction/Sinkhorn 松弛**：近似解破坏与 GT 损失的匹配一致性风险高——检测匹配对损失语义敏感（matcher 即监督定义），**默认否决**，除非先把匹配一致性容差论证清楚；
   - c. **side-stream 重叠**：cost D2H + scipy 放独立 CUDA stream，与下一微批 backbone 重叠——对 accum>1 配方有效，复杂度高；
3. 结论预期：维持 scipy（现状已足够好），评估记录归档——这与分册 14 R8"块对角 batched 指派经评估否决"是同一纪律的延续。

### 明确否决项（记录依据）

- **训练侧 CUDA Graphs（整步捕获）**：动态 query 数（density 分支 per-image padding，`dome_decoder.py`）+ 每步 host 匹配往返 = 图捕获不可行；静态图只在 `convert_to_deploy` 导出路径有意义（已存在）；
- **criterion 整体编译**：metrics/log_dict 的 CPU 交互与 per-image 控制流导致图断裂成碎片，收益为负（W3 只编译 decoder 的原因）。

---

## 4. 与其他分册的交互

| 分册 | 交互 |
|---|---|
| 23 蒸馏 | teacher 前向加相位于 profile 脚本（G2 度量复用 §2 协议）；W2 的展平 helper 被 teacher 匹配目标构建复用 |
| 24 FP8 | 叠加实验顺序：先 W1/W2（确定收益）后 fp8 档（独立 G2）；compile×fp8 最后 |
| 25 indexer | indexer 的 BCE/预算损失接 criterion 时沿用 W2 的展平 helper；indexer per-image top-k 属模型侧（不在本册范围） |

---

## 5. 测试与验收

1. **W1**：fused vs foreach 200 步 loss 轨迹相对差 < 1e-5（fp32）；优化器 state_dict 往返兼容（resume 场景：fused↔foreach 混合加载告警但不崩）；CPU/NPU 回退路径；
2. **W2**：向量化前后同一输入的 FGL/DDF/热图数值相对差 < 1e-6（fp32，torch.testing.assert_close 默认容差）；空匹配图（src.numel()==0）段跳过语义不变；`num_boxes_go`/`dn_num_boxes` 归一化分母逐位一致；金标回归（现有 147 例组件回归全绿）；
3. **W3**：compile 开/关的输出张量容差内一致；DDP 2-rank smoke；图断点数日志（`torch._dynamo` counters）归档；
4. **通用**：全部改动 off 开关下与基线逐比特一致（D3）；profile 脚本输出格式向后兼容（消融历史可比）。

---

## 6. 门槛、风险与实施顺序

**门槛**（§2 协议）：

| # | 项目 | 判据 |
|---|---|---|
| G-W1 | fused AdamW | optim 相位步时 ≥ −30% 且端到端 ≥ −2% |
| G-W2 | criterion 向量化 | criterion 相位步时 ≥ −15% 且 launch 数 ≥ −40% |
| G-W3 | compile decoder | 步时 ≥ −8% 且 pilot mAP ≥ −0.2 AP |
| G-W4 | GPU 匹配评估 | scipy 占比 < 5% → 归档否决；≥ 5% → 进入方案对比 |

**风险表**：

| 风险 | 等级 | 缓解 |
|---|---|---|
| R1 fused AdamW 与 param_group 级 LR 调度交互（FlatCosine/MultiStep 每步改 group lr） | 中 | torch fused 支持组级 python float lr 的每步更新（测试 1 钉住）；不兼容则该组回退 foreach（混合模式） |
| R2 向量化改变求和顺序 → 极小数值漂移跨 64k 步放大 | 低 | fp32 全程 + 容差验收 + pilot mAP 底线复验（与历史 run 可比性声明：W2 合入点之后的新 run 才两两可比） |
| R3 compile 图断裂收益为负 | 中 | 实验旗标默认关；G-W3 独立门槛；density-on 场景预期管理（§3 W3） |
| R4 profile 结论过拟合到 Dome twin 配方 | 低 | §2 协议要求同时记录 micro batch / gt-per-image 两组参数（脚本已支持 `--batch/--gt-per-image`） |

**实施顺序**：W1（半天）→ W2（1–1.5 天）→ G-W4 评估（半天）→ W3（1 天 + GPU）——W1/W2 的收益直接加速后续全部分册实验的迭代速度，故最优先。

---

## 7. 实现清单

| 文件 | 改动 | ~行 |
|---|---|---|
| `ltdetr_object_detection/train_model.py`（+ deimv2 镜像） | `fused_optimizer` arg + AdamW fused 构造与回退 | ~40 |
| `object_detection_components/dfine_utils.py` | `_flatten_matches` helper + `fg_loss`/DDF 权重的扁平路径（原路径保留开关） | ~80 |
| `object_detection_components/dfine_criterion.py` | FGL 目标构建/施加的扁平路径 | ~60 |
| `object_detection_components/defe.py` | `GaussHeatmapGenerator` batched 路径 | ~40 |
| `ltdetr_object_detection/config.py` | `fused_optimizer` / `compile_decoder` 字段 | ~10 |
| `scripts/profile_criterion_ops.py` | matcher/scipy 计时扩展（G-W4） | ~30 |
| `tests/test_step_engineering.py` | §5 | ~250 |

总规模 ~2 天（W1+W2+G-W4）；W3 另计 1 天 + GPU。

---

## 8. 后续路线

| # | 内容 | 备注 |
|---|---|---|
| P0 | W1 + W2（含 §2 基线采样） | 2 天，收益直接加速全部后续实验 |
| P1 | G-W4 评估归档 | 半天 |
| P2 | W3 compile 实验（独立分支，G-W3 门槛） | 1 天 + GPU |
| P3 | 与分册 24 的叠加实验（fp8 × compile） | 两项各自验收后 |
| P4 | 远期：`_msda_src`/deformable attention 的 CUDA 特化（若 ViT@1280 配方成为主力再议） | 独立立项 |

---

## 9. 参考索引

**DeepSeek 侧（【DS】事实来源）**：

| 内容 | 位置 |
|---|---|
| FlashMLA（memory-bound attention 的带宽优化 kernel） | `github.com/deepseek-ai/FlashMLA` |
| DeepGEMM（JIT 形状特化 FP8 GEMM，细粒度 scaling） | `github.com/deepseek-ai/DeepGEMM` |
| DeepEP（MoE all-to-all 通信/计算重叠——单卡不适用，方法论参考） | `github.com/deepseek-ai/DeepEP` |
| "launch-bound 小算子融合 > 算法改进"的工程判断 | DeepSeek 开源周（2025-02）系列仓库 README 与技术报告 §4（基础设施） |

**VTrain 受影响面**（`src/autovision/`）：

| 内容 | 位置 |
|---|---|
| criterion per-image 循环群（W2 对象） | `dfine_criterion.py:357-394`；`dfine_utils.py:354-375, 420-434` |
| DeFE 热图逐图循环 | `ltdetr_object_detection/dome_encoder.py:126-133`；`object_detection_components/defe.py` |
| 优化器构造（W1 对象） | `ltdetr_object_detection/train_model.py:699-704`；`deimv2_object_detection/train_model.py:231-236` |
| 已完成的同源工程（盘点基线） | §1 表格所引各处 |
| 度量工具 | `scripts/profile_step_phases.py`、`scripts/profile_criterion_ops.py` |
| "评估后否决"纪律先例 | 分册 14 R8（块对角指派否决）；本文 §3 W4 / 否决项 |

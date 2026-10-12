# VTrain 实现蓝图 · 分册 25：可学习 token indexer（DSA 思想的背景稀疏化升级）

| 文档属性 | 内容 |
|---|---|
| 版本 | v1.0（设计评审稿，未实施） |
| 状态 | 设计完成；**前置缺口：本仓尚无 ViT 变体的 VisDrone 训练配方**（doc 17 §11.2 记录：现有脚本全为 ConvNeXt），G0 先补配方再过门槛 |
| 上游思想 | DeepSeek-V3.2 的 DeepSeek Sparse Attention（DSA，arXiv:2509.…/官方 repo）：细粒度 token 级稀疏注意力的关键是**可学习的轻量 lightning indexer**——每个 (query, key) 对打分、top-k 选择，"什么重要"由可学习小模块判断而非启发式规则。【DS】 |
| 本册定位 | 把分册 17 的静态 token drop（GT 框规则 + 随机窗口采样）升级为**可学习的 per-token indexer**：规则先验降格为辅助监督信号，选择本身交给打分模块学习。统一分册 13（密度）/ 17（token drop）/ 20（masked forward）三条"token 重要性"线索的第一步 |
| 参考实现 | 无外部移植源；管道完全复用本仓已验证的 masked-forward 基础设施（docs 17/20，`tests/test_window_token_drop.py` 30 例） |
| 受影响模块 | `ltdetr_object_detection/config.py`、`train_model.py`、`task_model.py`、`_models/dinov2/dinov2_wrapper.py`、`_models/dinov3/dinov3_wrapper.py`、`scripts/visdrone_ltdetr_vits16_800.py`（新，G0）、`tests/test_token_indexer.py`（新） |
| 基线兼容承诺 | `token_indexer=None`（默认）逐比特基线；与 `token_drop` 互斥（init 期校验）；验证/导出路径强制稠密（doc 17 E2 同款） |

> **标注约定**：【DS】DeepSeek 论文/仓库已验证事实；【VTrain】本仓现状（`文件:行` 相对 `src/autovision/`）；【设计】本文档决策；【差异】D1–D6。

---

## 1. 背景：从"规则 + 随机"到"可学习 indexer"

### 1.1 现状与缺口

【VTrain】分册 17 已落地 window 级 token 删除（ViT 家族，~46–48% 可丢率，doc 17 §11.1），机制是：**GT 框保护窗全保 + 背景窗随机采样**（`train_model.py:62-116` 的 `_build_window_keep_mask`，选择逻辑复用像素擦除 `protected_cell_mask`/`sample_erase_mask`）。DINOv3 masked forward 由分册 20 补齐（`dinov3_wrapper.py:57-163`，rope 与 token 同索引 gather）。

两个结构性局限：

1. **选择信号是规则的**：窗口"是否前景"由 GT 框重叠决定——GT 只标注已标注物体，遮挡/小目标窗的保护是二值的；背景窗的保留是均匀随机的，与内容无关。可丢率上限被保护规则锁死（窗口含任何 GT 外扩即全保）；
2. **没有内容感知**：一个纯天空窗和一个含未标注目标的窗在机制眼里相同（随机决定去留）。

【DS】DSA 的回答：用一个**轻量打分器**（lightning indexer，小 MLP 级别）对每个 token 打分，注意力只看 top-k——打分器随主任务端到端训练，重要性判据从数据里学出来。V3.2 报告显示该机制在长序列上大幅省算力且精度不掉，关键是**indexer 的参数与训练信号都是廉价的**。

【设计】把同一思想搬进 ViT 检测骨干：**per-token 前景性打分器 + 预算化 top-k 选择 + GT 重叠辅助监督**。与 doc 17 的关系不是替换而是**升级**：GT 框保护从"硬规则"降格为"辅助监督的标签来源"（§3.3），冷启动阶段两者行为一致（§3.5）。

### 1.2 诚实的预期管理

- **家族限定 ViT**：ConvNeXt 主力（现役 VisDrone 配方）是卷积，无 token 序列，本册不适用（doc 17 §3.1 的家族矩阵裁定沿用）。本册的全部价值兑现在 ViT 变体配方上——因此 G0 前置"ViT VisDrone 配方"缺口；
- **序列长度现实**：800px/patch16 → 2500 token（DINOv3 ViT），相比 LLM 的万级序列短——DSA 的收益比例不会完整复现，目标定位为"骨干 encoder 相位 30–50% 的 FLOPs 削减"（mask 粒度从 32px 窗细化到 patch，可丢率上限从 ~47% 提高，G1 实测）；
- **增广语义保留**：doc 17 §5 的裁定（partial keep = 结构化真缺失增广，非无损加速）在可学习选择下**同样成立**——本册同时收获增广与速度，消融按增广实验设计（± indexer × ± erase）。

---

## 2. 现状管道与手术点

### 2.1 可复用的已验证基础设施

| 环节 | 位置 | 复用方式 |
|---|---|---|
| 稠密/masked 双路径分发 | `task_model.py:580-587`（`_extract_window_keep_masks` → `ViTDetectionBackbone.forward`） | indexer 路径挂同一 `window_keep_masks` 入口（签名不变，D1） |
| DINOv2 masked 前向 | `dinov2_wrapper.py`（`forward_multiscale_features_masked`，doc 17 §4.3） | **签名扩展**：keep masks 由外部给定 → indexer 在中间产出（§3.1 的层间插入需要新编排，见 D2） |
| DINOv3 masked 前向 | `dinov3_wrapper.py:57-163`（rope 同索引 gather） | 同上 |
| 零回填 | `_backfill_patch_tokens`（dinov2/dinov3 共享） | indexer 路径出口零回填，下游 tap/resize/STA/投影零改动 |
| 不变量测试 | `tests/test_window_token_drop.py`（keep-all ≡ dense 逐比特、partial 精确零回填、batch ≡ 逐图拼接） | 全套不变量原样继承为 indexer 路径的测试基线 |

### 2.2 关键结构差异：选择点从"encoder 之前"移到"中间 block 之后"

【VTrain】doc 17/20 的删除点在 patch embedding 之后、encoder 之前——整条 encoder 省算力。【设计 D2】indexer 需要看**内容**才能打分，所以选择点后移到第 `select_layer` 个 block 之后（默认 4，即 encoder 前 4 层稠密、后 N−4 层稀疏）：

```
embeddings → block[0..select_layer) 稠密 → indexer 打分 → top-k gather
           → block[select_layer..N) 稀疏（kept-only）→ 每 tap 零回填 → 原出口
```

- 省的 FLOPs = (N − select_layer)/N 比例的 encoder（vits16：N=12、select_layer=4 → 后 8 层受益，encoder 占骨干 ~85% FLOPs（doc 17 G2 同源口径）→ 理论削减 ~57% encoder）；
- **tap 语义不变**：tap 点若落在稀疏段，hidden states 是 kept-only 序列的输出，零回填后与稠密布局一致（现有 masked 前向已验证该等式，`dinov3_wrapper.py:136-148`）；
- HF 内部编排深化一档：需要逐 block 循环驱动 `model.model.layer`（DINOv3）/ `model.encoder.layer`（DINOv2），而不是整段 `encoder(...)` 调用——这是对 doc 17 §4.3.1 教训（HF 属性面漂移）的加重暴露，`supports_masked_forward` 门控扩展为"逐层循环面"校验（§3.6）。

---

## 3. 详细设计

### 3.1 Indexer 模块（挂在 ViTDetectionBackbone 上）

```python
class TokenIndexer(nn.Module):
    """Per-token foreground scorer (design doc 25; DSA lightning-indexer analogue)."""
    def __init__(self, hidden_dim: int, indexer_dim: int = 64):
        self.mlp = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, indexer_dim), nn.GELU(),
            nn.Linear(indexer_dim, 1),
        )
        # 2D sincos 位置投影（可学习线性到 indexer_dim），拼接到输入——
        # 纯内容打分对 patch 网格位置不敏感，位置是前景性的强先验。
    def forward(self, patch_tokens: Tensor, pos: Tensor) -> Tensor:
        # (B, N, D) + (B, N, D_pos) → (B, N) logits
```

参数量 ~hidden_dim×64 级（vits16：384×64×2 ≈ 50K，骨干 22M 的 0.2%）。归入 **detector 参数组**（随机初始化连接件的既有归类先例，`train_model.py:637-642` 的 connector 语义）。

### 3.2 选择机制（D3：预算化 top-k，非阈值）

对每图：`keep_n = clamp(ceil(keep_ratio × N), min_keep, N)`，取 score top-`keep_n` 的 patch token，与**永远保留的集合**取并集：

- patch 网格到 keep mask 的换算沿用 doc 17 的窗口网格语义向下细化到 patch 粒度（`window_size = patch_size` 合法化：`_kept_patch_indices` 的整除校验 `dinov2_wrapper.py` 已支持任意网格）；
- GT 保护不再是硬规则——但冷启动辅助监督（§3.3）会让保护语义以软方式涌现；`min_keep`（默认 max(64, N/16)）防塌缩到全删。

选择不可导——梯度经 gather 只流进 kept token（doc 17 增广语义），indexer 自身的梯度全部来自 §3.3 的显式辅助损失与预算正则（D4，这是与 DSA 的关键差异，见 §5 D3 说明）。

### 3.3 训练信号：三条辅助损失（`loss_indexer` 族，train-only）

1. **GT 重叠 BCE（主监督，`λ_overlap=1.0`）**：以"patch 网格窗是否与 GT 外扩框重叠"（doc 17 的 `protected_cell_mask` 在 patch 粒度重算）为二值标签，`BCEWithLogits(score, label)`——**doc 17 的规则先验从硬规则降格为监督标签**，indexer 学到的是"在训练分布上预测 GT 保护窗"，泛化行为由数据而非规则决定；
2. **预算正则（`λ_budget=0.1`）**：`relu(mean_topk_keep_fraction − target)² + relu(target − …)²` 的软预算——top-k 本身已硬保证预算，此项约束 score 分布的**可分性**（防 score 全平使 top-k 退化为任意选择）；
3. **（v1.5 候选）kept-token 特征自蒸馏**：稀疏段输出与稠密前向的对应 token 余弦对齐（稠密参考前向每 k 步一次）——直接压制分布偏移，成本高，先不做。

三腿合计 `loss_indexer`，在 criterion 里按 DeFE/SSL/蒸馏同款"算一次、无后缀"模式接线（`dfine_criterion.py:477-489` 模式），权重 `weight_dict["loss_indexer"]`。

### 3.4 配置面（`config.py`）

```python
class LTDETRTokenIndexerArgs(PydanticConfig):
    """Learnable per-token foreground indexer (design doc 25). None = off."""
    model_config = {"extra": "forbid"}

    select_layer: int = Field(default=4, ge=1)     # 该 block 之后插入选择点
    keep_ratio: float = Field(default=0.55, gt=0.0, le=1.0)
    min_keep: int = Field(default=64, ge=1)
    indexer_dim: int = Field(default=64, ge=1)
    overlap_coef: float = Field(default=1.0, ge=0.0)   # GT 重叠 BCE 权重 λ_overlap
    budget_coef: float = Field(default=0.1, ge=0.0)
    overlap_padding: float = Field(default=0.15, ge=0.0)  # GT 外扩（doc 16/17 同语义）
    cold_start_steps: int = Field(default=1000, ge=0)     # 0..N 步用静态 doc17 mask，之后线性切换
    step_stop: int | None | Literal["auto"] = "auto"      # 末期回稠密（doc 17 同款）
    use_for_mwas: bool = False                             # v2：分数池化驱动 MWAS 窗掩码

# LTDETRObjectDetectionTrainArgs 新增：
    token_indexer: LTDETRTokenIndexerArgs | None = None
```

校验器：`token_indexer` 与 `token_drop` 互斥（同一稀疏化位的两个实现）；家族门控 `family == "vit"`（复用 `task_model.py:276-295` 的 token_drop 校验模式）；`select_layer < num_hidden_layers`；`use_for_mwas` 要求 density 分支开启。

### 3.5 冷启动与退火（D5）

- **cold_start（0..cold_start_steps）**：keep mask 直接取 doc 17 静态逻辑（GT 保护 + 随机背景采样，预算对齐 keep_ratio）——indexer 的 BCE 监督同期在学；此阶段行为与 doc 17 等价（增广语义连续）；
- **切换（cold_start..cold_start+fade）**：以概率 p 线性从静态切到 indexer 选择（每步每图掷币，RNG 可复现）；
- **step_stop**：复用 doc 17 的步窗机制（`train_model.py:609-621` 的 auto 解析同款），末期回稠密——部署分布一致性裁定不变。

### 3.6 wrapper 门控扩展

`supports_masked_forward`（`dinov3_wrapper.py:40-55` / dinov2 同名）语义从"masked 前向面"扩展为"**逐层驱动面**"：需要 `model.model.layer`（DINOv3）或 `model.encoder.layer`（DINOv2）可迭代——indexer 编排要逐 block 前向。门控失败 → 配置期报错（doc 17 §4.3.1 的结构门控先例），不进训练中途。

### 3.7 forward 编排（DINOv3 版示意，DINOv2 镜像）

```
per-image loop（kept 数每图不同，doc 17/20 既有裁定）:
  emb = embeddings(pixel)          → strip prefix
  cos, sin = rope_embeddings(pixel)
  前段: for block in layer[:select_layer]: hidden = block(hidden, rope=(cos,sin))
  score = indexer(hidden_patches, pos2d)             # (1, N)
  kept  = topk ∪ min_keep（预算化）
  gather hidden/cos/sin（同索引，dinov3_wrapper.py:121-129 的既有语义）
  后段: for block in layer[select_layer:]: hidden = block(...)
  每 tap: 零回填 → _reshape_patch_tokens              # 出口布局与稠密一致
```

训练态 per-image 循环开销与 doc 17 实测结论一致（batch 小，可忽略，doc 17 §9）。**诊断**：`indexer_keep_ratio`（实际 kept/ N，镜像 `token_keep_ratio` 落点 `train_model.py:342-346`）与 `indexer_overlap_auc`（score 对 GT 重叠标签的 AUROC，val 聚合，G1 的在线版）。

### 3.8 v2 方向：统一 MWAS 窗掩码（`use_for_mwas`，本期只留接口）

MWAS 窗掩码现在由 DeFE 密度图阈值化产生（`dome_encoder.py:99-107`）。indexer score 经窗口 max-pool 后可作为掩码来源与密度图**并联**（可学习前景性 ∥ 密度回归）。v1 不实现（density 分支互斥性待消融），仅落配置开关与校验。

---

## 4. 与既有机制的关系

| 机制 | 关系 |
|---|---|
| doc 17 静态 token drop | **互斥升级**：同一稀疏位，init 校验拒绝同开；冷启动期行为等价 doc 17 |
| doc 20 DINOv3 masked forward | 复用其 rope-gather/零回填语义，编排从整段 encoder 调用深化为逐 block |
| 分册 13 Dome/MWAS | 正交（v1）；v2 的 `use_for_mwas` 是统一入口 |
| 分册 16 像素擦除 | 独立可叠加（doc 17 §3.3 裁定沿用） |
| 分册 23 蒸馏 | 潜在协同：teacher 覆盖率高的实例可作 indexer 监督升级（分册 23 §11 P4），v1 不接 |
| 分册 24 FP8 | indexer MLP（64 宽）低于转换下限自然全精度（分册 24 §4.2） |

---

## 5. 差异与取舍清单

| # | 取舍 | 决策 | 理由 |
|---|---|---|---|
| D1 | 对外签名 | indexer 复用 `window_keep_masks` 入口语义（wrapper forward 签名不变） | 下游（STA/resize/投影/encoder/decoder）零感知；doc 17 的透传链测试直接复用 |
| D2 | 选择点位置 | select_layer 后（默认 4），前段稠密 | indexer 需内容打分；全段稠密打分（选择点在 embeddings 后）退化为静态规则的翻版 |
| D3 | indexer 的梯度来源 | 显式辅助损失（BCE + 预算），**不经** top-k 选择反传 | DSA 的 indexer 由主损失经选择反传（注意力权重对分数可导）；top-k gather 不可导，反传不成立。BCE 监督是检测场景下最便宜的有效替代 |
| D4 | 预算形式 | 硬 top-k（每图 keep_n）+ 软可分性正则 | 纯阈值法在 batch 内 kept 数不定，破坏批处理与显存可预期性；预算化与 DSA 的 per-query top-k 精神一致 |
| D5 | 过渡策略 | 静态冷启动 → 概率淡入 | indexer 早期 score 无意义，直接接管 = 任意选择；冷启动同时保住 doc 17 增广语义的连续性 |
| D6 | 辅助标签粒度 | patch 粒度 GT 重叠（非窗口） | 选择粒度即 patch；窗级标签会让边界窗的监督平均化 |

---

## 6. v0 门槛（实施前必过）

| # | 内容 | 工具 | 判据 |
|---|---|---|---|
| G0 | **ViT 变体 VisDrone 配方补缺**：`scripts/visdrone_ltdetr_vits16_800.py`（COCO 热启动 + 现役 twin 配方）跑通 10-epoch 基线 | 新脚本 | 基线 mAP 落盘；无此基线则后续全部消融无参照系 |
| G1 | **可丢率上界**：patch 粒度 GT 重叠标签的背景占比（= indexer 可删上限，对比 doc 17 窗级的 46.7%） | 离线统计（`scripts/visdrone_token_drop_stats.py` 加 `--patch` 模式） | patch 级可删率显著 > 窗级（预期 >60%），否则细化粒度的动机削弱 |
| G2 | **骨干占比**：ViT 变体骨干相位步时占比 | `profile_step_phases.py`（vits16 配方变体） | 骨干 < 35% 步时 → 速度动机降级（doc 17 G2 同阈值），按增广动机继续 |
| G3 | **逐层驱动面确认**：DINOv2/DINOv3 的 layer 迭代前向与整段调用逐位一致（select_layer=N 时） | 单测（tiny 配置） | 逐位一致才允许逐层编排上线（HF 属性面风险的前置排查） |
| G4 | G0–G3 通过 → 实施 §7 | — | — |

---

## 7. 实现清单

| 文件 | 改动 | ~行 |
|---|---|---|
| `ltdetr_object_detection/config.py` | `LTDETRTokenIndexerArgs` + 字段 + 校验器 | ~70 |
| `ltdetr_object_detection/task_model.py` | indexer 构建（挂 backbone）+ 互斥/家族门控 + args 往返 | ~60 |
| `ltdetr_object_detection/backbone_wrappers.py` | `ViTDetectionBackbone` 持有 indexer；`forward` 新分支（indexer 编排） | ~80 |
| `_models/dinov2/dinov2_wrapper.py` | 逐层 masked 前向编排（`forward_multiscale_features_indexed`）+ `supports_masked_forward` 扩展 | ~110 |
| `_models/dinov3/dinov3_wrapper.py` | 同上（rope 版） | ~90 |
| `object_detection_components/` | `TokenIndexer` 模块 + `indexer_losses`（BCE/预算） | ~90 |
| `ltdetr_object_detection/dfine 接线` | `loss_indexer` 进 criterion（DeFE 模式）+ 诊断 | ~50 |
| `scripts/` | G0 脚本 + 统计扩展 | ~120 |
| `tests/test_token_indexer.py` | §8 | ~400 |

总规模 ~3–4 天（G0 基线训练另计 GPU 时间）。

---

## 8. 测试与验收（继承 `tests/test_window_token_drop.py` 的不变量族）

1. **keep-all ≡ dense 逐比特**（indexer 预算 = N 时，含全 tap）——继承不变量 #1；
2. **partial keep 精确零回填**——继承 #2；梯度流达 kept token 与 indexer MLP（BCE 路径）；
3. **逐层编排等价**：`select_layer ≥ num_layers` 时与整段 masked 前向逐位一致（G3 钉住）；
4. **预算硬约束**：任意输入下 kept 数 == clamp(ceil(ratio·N), min_keep, N)；min_keep 兜底生效；
5. **冷启动调度**：cold_start 期内 mask 与 doc 17 静态逻辑一致（同 RNG）；淡入期概率单调；step_stop 后恒稠密；
6. **辅助损失手算对账**：BCE 与预算正则与手算一致；标签生成（patch 粒度 GT 外扩重叠）与 `protected_cell_mask` 窗级语义的自洽；
7. **互斥与门控**：`token_indexer` + `token_drop` 同开 raise；ConvNeXt/ECViT raise；select_layer 越界 raise；DINOv2 属性面缺失 raise（配置期）；
8. **off-by-default**：`token_indexer=None` → 无 `indexer` 模块、state_dict 键集合与基线一致、前向逐比特一致；
9. **val/导出稠密**：验证前向永不带 mask/indexer 选择（E2 同款）；
10. **诊断**：`indexer_keep_ratio`/`indexer_overlap_auc` 的数值管道（val 聚合 + compute_metrics 落点）；
11. **args 往返**：`model_dump`/re-coerce（density/ssl/token_drop 同款）。

---

## 9. 消融与实验设计（G0 之后的 ViT 配方上）

| 臂 | 配置 | 目的 |
|---|---|---|
| a | G0 基线（稠密） | 参照系 |
| b | doc 17 静态 drop（window 级，doc 17 推荐档） | 静态先验参照 |
| c | indexer（keep_ratio = b 的实际 kept 率对齐） | **同预算下可学习 vs 规则+随机**（核心对照） |
| d | c 的 keep_ratio 扫描 {0.35, 0.45, 0.55, 0.7} | 稀疏-精度曲线 |
| e | c ± overlap_coef（监督消融） | GT 监督的必要性 |
| f | c ± cold_start | 冷启动价值 |
| g | c + erase 组合 | 增广叠加（doc 17 §5 的捆绑归因纪律） |

记录：mAP、APs、`indexer_keep_ratio`、`indexer_overlap_auc`、骨干相位步时（G2 工具）。验收：同预算下 c − b ≥ +0.3 AP（可学习选择的内容感知价值）；速度收益按 step 时长直接测（增广/速度捆绑归因，doc 17 §5 纪律）。

---

## 10. 风险表

| 风险 | 等级 | 缓解 |
|---|---|---|
| R1 HF 逐层驱动面漂移（doc 17 §4.3.1 教训的加重版） | 高 | G3 前置逐位等价 + `supports_masked_forward` 结构门控扩展 + 配置期报错；升 transformers 时哨兵测试 |
| R2 indexer 退化为静态规则翻版（AUROC ≈ 1 但无内容泛化） | 中 | G1 在线 AUROC 监控；e 组监督消融；v1.5 自蒸馏腿兜底 |
| R3 top-k 选择的批间显存抖动（kept 数逐图不同） | 低 | per-image 循环编排（doc 17 既有裁定）；预算 clamp 收窄范围 |
| R4 增广语义下 mAP 损伤 | 中 | 本身即消融对象（d 组曲线）；step_stop 末期回稠密；60ep 止损线 |
| R5 ViT 变体精度基线弱于 ConvNeXt 主力 → 结论外推受限 | 中 | 如实定位：本册是 ViT 家族能力 + 方法论验证（DSA-for-detection 论文叙事），不承诺主力配方收益 |
| R6 与 density 组合的 MWAS 掩码竞争（v2） | 低 | v1 互斥不接；v2 单独消融 |

---

## 11. 后续路线

| # | 内容 | 备注 |
|---|---|---|
| P0 | G0 ViT 配方 + G1/G2/G3 门槛 | 1 天 + GPU |
| P1 | §7 实施（DINOv3 先、DINOv2 镜像） | 3–4 天 |
| P2 | §9 消融矩阵执行与回填 | GPU 时间 |
| P3 | v2：`use_for_mwas` 统一窗口掩码（indexer ∥ DeFE 密度并联） | 视 P2 |
| P4 | v1.5：kept-token 特征自蒸馏腿；与分册 23 的 teacher 监督联动 | 研究线 |

---

## 12. 参考索引

**DeepSeek 侧（【DS】事实来源）**：

| 内容 | 位置 |
|---|---|
| DSA 细粒度稀疏注意力 + lightning indexer（可学习打分器、top-k 选择、主损失端到端训练） | DeepSeek-V3.2-Exp 技术报告（arXiv，2025）；`github.com/deepseek-ai/DeepSeek-V3`；vLLM/SGLang day-0 支持博客（2025-09-29） |
| indexer 的训练信号设计（MTP/主损失驱动） | V3.2 报告 §DSA；本仓对应差异见 §5 D3 |

**VTrain 受影响面**（`src/autovision/`）：

| 内容 | 位置 |
|---|---|
| 静态 token drop（被升级对象） | `ltdetr_object_detection/train_model.py:62-116, 272-291`；分册 17 |
| DINOv2 masked 前向 + 零回填 | `_models/dinov2/dinov2_wrapper.py`（`forward_multiscale_features_masked`/`_backfill_patch_tokens`/`_kept_patch_indices`） |
| DINOv3 masked 前向（rope 同索引 gather） | `_models/dinov3/dinov3_wrapper.py:57-163`；分册 20 |
| 家族/结构门控先例 | `ltdetr_object_detection/task_model.py:276-295, 310-328` |
| ViT 骨干 forward（选择点宿主） | `ltdetr_object_detection/backbone_wrappers.py:204-276` |
| MWAS 窗掩码（v2 统一对象） | `object_detection_components/mwas.py:59-92`；`ltdetr_object_detection/dome_encoder.py:99-124` |
| "算一次无后缀"损失接线模式 | `object_detection_components/dfine_criterion.py:477-489` |
| 步窗调度先例（step_stop/cold start） | `ltdetr_object_detection/train_model.py:609-621`；分册 16 P1.5 |
| 可丢率统计工具（G1） | `scripts/visdrone_token_drop_stats.py`；doc 17 §11.1 实测（46.7% 窗级基线） |

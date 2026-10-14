# VTrain 实现蓝图 · 分册 27：LT-DETR 开放词表扩展——文本对比分类头与语言引导查询选择（Grounding DINO 移植）

| 文档属性 | 内容 |
|---|---|
| 版本 | v1.1（v1.0 设计评审稿 → v1.1：§7 v1/v2 已实施，§13 实现记录回填，含审查修复） |
| 状态 | **v1/v2 已实施并通过测试**（`tests/test_ltdetr_open_vocab.py` 37 例 + 相关回归套件全绿，2026-10-12）；剩余为需 GPU/联网的门槛项：G1 文本塔钉扎、G2 真塔预算实测、G3 双臂训练、P3 起的 v3 路线 |
| 上游思想 | Grounding DINO（IDEA Research，ECCV 2024，`github.com/IDEA-Research/GroundingDINO`）：开放集检测不能只在分类头打补丁——把 DINO 式检测器概念上拆成特征增强、查询选择、解码三阶段，**每一阶段都注入语言**。查询选择由"固定类头分数"换成"查询-文本对比相似度"（language-guided query selection），分类由"逐类 Linear"换成"query × 文本 token 点积"（无类别数耦合的对比头）。【GDINO】 |
| 本册定位 | 该思想向 LT-DETR 的移植蓝图，按本仓算力与离线约束裁剪为三段推进：**v1 文本对比分类头**（解耦类别数，本册主体）、**v2 语言引导查询选择 + decoder 文本交叉注意力**（GDINO 三处融合中的后两处）、**v3 encoder 文本融合与 grounding 数据**（真开放集，路线图）。核心工程裁定是"先降维、再复用"（§3.2）：对比头输出 (B,Q,C) 类分数视图，使 criterion / matcher / 解码 / Dome 动态选择**全栈零改动** |
| 参考实现 | HF `transformers` 的 `GroundingDinoForObjectDetection`（≥4.40；本仓 5.19.0 实测 config：900 queries、enc/dec 各 6 层、d_model 256、max_text_len 256、4 特征层）；OWL-ViT 的"冻结 CLIP 文本塔 + 逐类点积"先例【GDINO】 |
| 受影响模块 | `ltdetr_object_detection/config.py`、`task_model.py`、`train_model.py`；`object_detection_components/rtdetrv2_decoder.py`、`dfine_decoder.py`、`dfine_official_decoder.py`、`dfine_criterion.py`、`decode.py`；新组件 `object_detection_components/open_vocab.py`；`scripts/visdrone_ov_token_budget.py`（新）、`tests/test_ltdetr_open_vocab.py`（新） |
| 基线兼容承诺 | `open_vocab=None`（默认）逐比特基线：无文本模块、state_dict 键集合与前向逐位一致；文本塔冻结且不入 state_dict/optimizer/EMA（distill teacher 先例）；验证/导出路径与基线同分布 |

> **标注约定**：【GDINO】Grounding DINO 论文/官方仓库/HF 实现已验证事实；【VTrain】本仓现状（`文件:行` 相对 `src/autovision/`）；【HF】transformers 实现事实；【设计】本文档决策；【差异】D1–D10。

---

## 1. 背景：为什么是"三处融合"而不是"换个分类头"

### 1.1 现状与缺口：类别数在五处焊死了架构

【VTrain】LT-DETR 目前是纯闭集检测器，`num_classes` 在五个位置耦合进网络：

| # | 耦合点 | 位置 | 后果 |
|---|---|---|---|
| 1 | 编码器打分头 | `enc_score_head = Linear(hidden, num_classes)`（`rtdetrv2_decoder.py:643-646`；official `dfine_official_decoder.py:657-660`） | 两段式查询选择按类别 logit 选 top-k |
| 2 | 逐层分类头 | `dec_score_head = ModuleList[Linear(hidden, num_classes) × L]`（`rtdetrv2_decoder.py:650-652`；official `:666-672`） | **头权重形状 = f(类别数)**，换词表必丢头 |
| 3 | 去噪内容嵌入 | `denoising_class_embed = Embedding(num_classes+1, …)`（`rtdetrv2_decoder.py:622-626`） | DN 分支按类索引查嵌入 |
| 4 | 匹配与标签消费 | matcher 对目标类别列做 focal 聚集（`matcher.py:85-97`）；VFL one-hot scatter（`rtdetrv2_criterion.py:79-82`） | 逻辑按"logits 最后一维 = 类别"书写 |
| 5 | 后处理 | `decode.py:38-43`：flatten → topk → `classes = idx % C` | 类别数进解码算术 |

直接后果（【VTrain】既有先例佐证）：`detector_checkpoint` 跨类数加载时**类别相关头只能跳过**（`coco_checkpoint` 经 `train_model.py:229-242`）；蒸馏分类腿在类数不匹配时强制 `cls_coef=0`（`config.py:709-733`）。**换词表 = 丢头重训**，是开放词表路线的结构性障碍。

### 1.2 Grounding DINO 的回答：三处融合

【GDINO】Grounding DINO 把语言注入 DINO 流水线的三个阶段：

1. **Feature Enhancer**：编码器图像/文本双侧交叉注意力（Flamingo 式零初始化门控）；
2. **Language-Guided Query Selection**：两段式查询不再按类头分数选 top-K，而按**编码器 token 与文本 token 的对比相似度**选（900 queries）；
3. **Cross-Modality Decoder + 对比头**：decoder 每层加文本交叉注意力；分类 = query 特征 × 文本特征点积 → [Q, T] 对比 logits（focal α=0.25, γ=2，**无固定类别数、无 no-object 类**——类别集完全由输入文本决定）。

训练配方是 O365 + GoldG 两阶段预训练、再混 Cap4M，BERT-base **可训练**文本塔，sub-sentence 文本表征。效果：COCO zero-shot 52.5 AP（未用 COCO 训练数据）、ODinW-13 26.1、微调后 ~62.6 AP。

### 1.3 诚实的预期管理

【设计】GDINO 级别的 zero-shot 泛化**依赖 O365+GoldG 量级的 grounding 预训练**，超出本仓算力（VisDrone 10-epoch 单卡试点，`config.py:817-873` 的配方尺度）。本册的收益主张按三段分解，不冒进：

- **v1 就能兑现的**：(a) **架构与类别数解耦**——头权重不再随词表变形，COCO→VisDrone 迁移不再丢头（G3 的热启动臂）；(b) **同域类别可扩展**——训 7 类、推理时加 3 类文本嵌入即得 zero-shot 类（VisDrone 类目演化的真实场景）；(c) 多词表联合训练不冲突（v3 的 O365+VisDrone 前置）；
- **v1 兑现不了的**：novel 类 zero-shot AP 不会高（冻结文本塔 + 仅 10 类训练样本的对比头，泛化上界受训练分布锁定）。G4 的 leave-3-out 协议就是用来**度量**这一上界而非承诺它；
- **v2 兑现的**：GDINO 报告的查询选择对齐收益（选择信号从"学出来的类分数"换成"文本对齐度"），在闭集训练上也可能有精度增益（§9 消融臂 d）；
- **v3 才是 GDINO 全集**：encoder 文本融合 + 可训练长上下文文本塔 + grounding 数据（经分册 23 蒸馏 teacher 出伪框，§4/§11）。

### 1.4 与本仓约束的匹配

【VTrain】本仓无任何文本依赖（探查确认：`clip|bert|siglip|text` 全仓仅命中梯度裁剪误报）；HF 权重走 `load_pretrained_verified` 的 sha256 钉扎清单（`_models/weight_verification.py:381`），离线构建以 `-notpretrained` 变体绕开下载（`dinov2_package.py:160-186`）；测试靠 tiny 本地配置保持 hermetic（`tests/conftest.py:12-46`）。【设计】文本塔**冻结 + 每词表一次离线缓存**，训练/推理零文本塔开销；hermetic 测试用确定性 `identity` 伪文本编码器，永不触网（D8）。

---

## 2. 现状管道与手术点

### 2.1 可复用的已验证基础设施

| 环节 | 位置 | 复用方式 |
|---|---|---|
| 可选分支"None = 逐比特基线"契约 | `LTDETRDensityArgs/SSLArgs/TokenDropArgs/DistillArgs`（`config.py:402/443/611/647`，均 `extra="forbid"`） | `LTDETROpenVocabArgs` 同款 |
| 新模块 RNG 隔离（最后构建） | ssl_head 先例 `task_model.py:450-462` | 文本模块最后构建，保住 `open_vocab=None` 逐比特承诺 |
| 文本塔排除导出面 | distill teacher `object.__setattr__` 先例 `train_model.py:285-307` | 冻结塔不入 state_dict/EMA/optimizer |
| 独立参数组（base LR / no-wd） | ssl 参数组 `train_model.py:933-950` | 文本投影/logit_scale 同款分组 |
| `agnostic` 选择先例 | `enc_score_head → Linear(hidden, 1)`（`rtdetrv2_decoder.py:644-646`） | 证明"换选择信号头"是既有合法形态 |
| 权重钉扎下载 | `load_pretrained_verified`（`weight_verification.py:381`）+ `_models/weight_hashes.py` | 文本塔权重进钉扎清单（G1） |
| 逐类 AP 已落盘 | `COCOMAPEvaluator.last_per_class` 逐类 AP（`_metrics/map_evaluator.py:77, 304-307`） | novel AP = 逐类 AP 子集聚合，零 pycocotools 手术（§3.7） |
| 诊断管道 | `log_dict` extras（`train_model.py:457-491`）→ `compute_metrics`（`:650-672`）→ DDP gather（`:682-707`）→ metrics.jsonl | `ov_*` 诊断同款落点 |
| 配置期 args 往返 | 分支配置 → plain dict 快照（`task_model.py:159-175`） | open_vocab 同款 |

### 2.2 查询选择现状（v2 的手术对象）

【VTrain】两段式选择全在 `RTDETRv2Decoder`（fork 两个 decoder 继承）：

- `_get_decoder_input`（`rtdetrv2_decoder.py:782-830`）：`enc_output` 投影 → `enc_score_head` 出 logits（`:795-798`）→ `_select_topk(…, num_queries)`（`:800-802`）→ **内容 = 被选中 memory（detach）**、参考点 = 反编码坐标（`:820-822`）→ DN 组前置；
- `_select_topk`（`:753-780`）：`query_select_method=="default"` 时按 `outputs_logits.max(-1)` 选（`:763`）——**选择信号就是类别头的最大 logit**；`"agnostic"`/`"one2many"` 为既有变体；
- Dome 动态版 `_get_decoder_input_dynamic`（`dome_decoder.py:94-245`）同样按最大 logit 取 `max_num_select`（`:118-120`），密度只决定**数量**与 NMS，不决定排序信号。

【设计】关键观察：GDINO 的选择也是"**对比分数 top-K，内容仍取被选中位置的图像 memory**"——文本不充当 query 内容。因此本仓把 `enc_score_head` 从 `Linear(hidden, C)` 换成文本对比头后，**`_select_topk` / Dome 动态选择一行都不用改**：`outputs_logits.max(-1)` 的语义自动从"类别分数"变为"文本对齐度"。这就是"先降维、再复用"红利的另一半。

---

## 3. 详细设计

### 3.1 总体架构与三段推进

```
v1  文本对比分类头        dec_score_head → ContrastiveClassHead（(B,Q,C) 类分数视图）
v2  语言引导查询选择      enc_score_head → 同款对比头（选择信号 = 文本对齐度）
    decoder 文本交叉注意力  TransformerDecoderLayer 插文本 K/V 交叉注意力（零初始化门控）
v3  encoder 文本融合       AIFI/TransformerEncoderLayer 文本交叉注意力 + 可训练文本塔
    grounding 数据         O365/GoldG/GDINO-teacher 伪框（分册 23 联动）
```

三段各自独立可开、独立消融；v1 是本册实施主体，v2 设计定稿随册落地，v3 只留接口与路线。

### 3.2 核心裁定：reduce-then-reuse（先降维、再复用）

【设计】对比头在**头内部**完成"文本空间 → 类别空间"的降维，对外仍吐 (B, Q, C)：

```
query_feat (B,Q,h) ──proj──► q (B,Q,d) ─┐
                                        ├─ cosine × logit_scale + class_bias ──► (B,Q,C)
class_names ──文本塔(冻结,缓存)──► text (C,d) ─┘
```

- `logits[b,q,c] = logit_scale · cos(q[b,q], text[c]) + bias[c]`；
- **逐类一票**（每类一个池化文本嵌入，非 GDINO 的逐 token 子句）——这是与 GDINO 的最大结构差异，取舍见 D3；
- 该视图喂给既有 `outputs["pred_logits"]` 消费面后，**criterion / matcher / decode / Dome / enc 监督全部零改动**（§2.1 #4/#5 的耦合点原样工作）。`dec_score_head` 从"per-layer Linear"换为"per-layer ContrastiveClassHead"，`ModuleList` 结构与逐层监督节奏不变。

### 3.3 词表资产（`open_vocab.py` 上半部）

【设计】词表是**数据**，不是架构：

- **类名来源**：train args 显式 `class_names: tuple[str, …]`（校验：非空、唯一、`len == num_classes`；数据集 `categories` 做交叉核对，不一致 → 配置期报错）。VisDrone 10 类名单作为常量进 `open_vocab.py`（【VTrain】探查确认：仓内现无任何类名映射，`categories` 仅运行期来自数据集，`train_model.py:342, 604-605`）；
- **prompt**：`prompt_template = "a photo of a {}."`（CLIP 惯例单模板起步）；`prompt_templates` 多模板系综取均值留 v1.5；
- **文本塔**：`text_encoder ∈ {"siglip-base", "clip-b32", "clip-b16", "identity"}`，默认 `siglip-base`（768 维，sigmoid 预训练、迁移性强的现代选择）；`identity` 是确定性伪编码器（类名稳定哈希 → 单位空间嵌入），**仅供 hermetic 测试与离线冒烟**，永不注册为真实选项的替代（D8）；
- **嵌入缓存**：每 (text_encoder, 词表, 模板) 一次前向（冻结塔、`torch.inference_mode`、fp16 存储 fp32 载入），落 `{cache_dir}/ov_text_{encoder}_{hash}.pt`；缓存命中后**文本塔不再进入计算图**——训练与推理的运行时文本开销为零；
- **离线**：siglip/clip 权重走 `load_pretrained_verified` 钉扎清单（G1 补条目）；下载失败 → 配置期报错并提示缓存路径，不静默随机初始化。

### 3.4 `ContrastiveClassHead`（v1 主体）

```python
class ContrastiveClassHead(nn.Module):
    """Text-contrastive classification head (design doc 27).

    Emits the (B, Q, num_classes) reduced view consumed by the unchanged
    criterion/matcher/decode stack; raw per-class cosine logits are the
    only internal representation.
    """
    def __init__(self, hidden_dim, text_embeds: Tensor,
                 logit_scale_init=2.659, logit_scale_max=4.605,
                 use_class_bias=True):
        self.proj = nn.Linear(hidden_dim, text_embeds.shape[-1])
        self.logit_scale = nn.Parameter(torch.tensor(logit_scale_init))  # no-wd 组
        # class bias: 恢复基线 bias_init_with_prob(0.01)≈-4.59 的早期校准
        # （对比头无 bias，裸 cosine 起点 sigmoid≈0.5，focal 代价失衡）；
        # 新类别外推时 bias 以 -4.59 初始化，词表仍可移植。
        self.class_bias = nn.Parameter(torch.full((C,), -4.59)) if use_class_bias else None
        self.register_buffer("text_embeds", F.normalize(text_embeds, dim=-1))
    def forward(self, query_feat):                     # (B,Q,h) → (B,Q,C)
        q = F.normalize(self.proj(query_feat), dim=-1)
        logits = self.logit_scale.clamp(max=self.logit_scale_max) * q @ self.text_embeds.T
        return logits if self.class_bias is None else logits + self.class_bias
```

- **逐层结构**：`dec_score_head = ModuleList[ContrastiveClassHead × L]` 镜像基线（`rtdetrv2_decoder.py:650-652`），每层独立投影（早期层语义未成熟，共享投影会串层）——D9；
- **尺度**：可学习 `logit_scale`，CLIP 惯例 init `ln(1/0.07)≈2.659`、clamp `[0, ln 100]`（D7；GDINO 用裸点积 + focal，本仓保留 VFL，见 D4）；
- **文本缓冲**：`text_embeds` 是 decoder 上的 registered buffer，**随导出面走**——部署侧只需 (C, d) 矩阵 + 投影层，无文本塔依赖；
- **encoder 头（v1 不动）**：`enc_score_head` 保持 `Linear(hidden, C)`，两段式选择与 `_enc_i` 监督语义原样（D5）；v2 才换（§3.5）；
- **official decoder 镜像**：`dfine_official_decoder.py:666-672` 的宽尾结构（`eval_idx` 后层用 `scaled_dim`）需逐层镜像替换，注意 `:332` 处 FDR 内部出 logits 的路径（§7 单列）。

### 3.5 语言引导查询选择（v2）

【GDINO → 设计】enc 头换成同款 `ContrastiveClassHead`（输入 (B,ΣHW,h) → (B,ΣHW,C)）：

- `_select_topk`（`rtdetrv2_decoder.py:753-780`）**零改动**：`max(-1)` 语义自动变为"该位置的文本对齐度"；`"one2many"`/`"agnostic"` 变体同样自动成立；
- Dome 动态版（`dome_decoder.py:118-120`）**零改动**：密度决定数量（doc 13），文本对齐度决定排序——两个信号正交；
- `enc_aux` 监督（`dfine_criterion.py:700-721`）零改动：enc logits 仍是 (B,HW,C) 的 VFL 消费面；
- 与 GDINO 的残差差异：GDINO 的对比相似度是 query×**token**（子句内多 token），本仓 v2 是 query×**类嵌入**（逐类一票）——token 级选择与子句表征一起归入 v3（D3）。

### 3.6 Decoder 文本交叉注意力（v2）

【设计】注入点：`TransformerDecoderLayer.forward`（`rtdetrv2_decoder.py:450-477`）的图像可变形交叉注意力**之后**（GDINO 同序：self → image cross → text cross → FFN）；official 镜像 `_OfficialDecoderLayer.forward`（`dfine_official_decoder.py:249-277`）。

```python
# 残差零初始化门控（Flamingo/GDINO 语义）：
gate = nn.Parameter(torch.zeros(1))          # 训练从 0 淡入
h = h + gate * TextCrossAttention(q=h, kv=self.text_kv)   # text_kv 由 text_embeds 缓冲投影
```

- K/V 来自 `text_embeds` 缓冲的线性投影（冻结源、可学投影），序列长 = C（VisDrone 10 / COCO 80 / O365 600）——对 (300~900 queries × C) 的注意力成本可忽略；
- `decoder_text_attention: bool = False` 控制；开启时对所有 dec 层生效（GDINO 全层）；
- **热启动安全**：零初始化门控保证启用 v2 注意力后的首步前向与未启用逐位一致（§8 不变量 #5），COCO 检测点热启动不被新支路扰动。

### 3.7 novel 类评测协议（G4）

【设计】

- **协议**：VisDrone leave-3-out——训练只用 7 类（`class_names` 7 元组 + 对应过滤的数据集子集），评测对全部 10 类出 AP，`novel_eval_classes = ("awning-tricycle", "bus", "motor")`（3 个 novel 类）；
- **落点**：`COCOMAPEvaluator.last_per_class` 已暴露逐类 AP（`map_evaluator.py:304-307`），novel AP = novel 子集逐类 AP 均值，在 `compute_metrics` 聚合（`train_model.py:656-671` 先例），键名 `val_metric/novel_ap`、`val_metric/base_ap`，DDP gather 走 `export_metric_state/merge_metric_state`（`:682-707`）；
- **正确性门槛**：`novel_eval_classes = all` 时协议必须复现基线 mAP（G4 判据）；
- **如实预期**：v1 的 novel AP 是**测量对象**（预期显著低于 base AP），不是收益承诺（§1.3）；它为 v3 grounding 数据的增量提供参照系。

### 3.8 配置面（`config.py`）

```python
class LTDETROpenVocabArgs(PydanticConfig):
    """Text-contrastive open-vocabulary branch (design doc 27). None = off."""
    model_config = {"extra": "forbid"}

    text_encoder: Literal["siglip-base", "clip-b32", "clip-b16", "identity"] = "siglip-base"
    class_names: tuple[str, ...]                       # 必填；len == num_classes
    prompt_template: str = "a photo of a {}."
    prompt_templates: tuple[str, ...] | None = None    # v1.5：系综均值
    use_class_bias: bool = True
    logit_scale_init: float = 2.659
    logit_scale_max: float = 4.605
    text_dtype: Literal["fp16", "fp32"] = "fp16"       # 仅影响一次性缓存前向
    cache_embeddings: bool = True
    # —— v2 段（随册实现，默认关）——
    language_guided_selection: bool = False            # enc 头 → 文本对比头
    decoder_text_attention: bool = False               # decoder 文本交叉注意力
    text_attn_heads: int = 8
    # —— 评测协议 ——
    novel_eval_classes: tuple[str, ...] | None = None  # leave-out 协议子集

# LTDETRObjectDetectionTrainArgs 新增：
    open_vocab: LTDETROpenVocabArgs | None = None
```

校验器：`class_names` 唯一且 `len == num_classes`；`novel_eval_classes ⊂ class_names`；`language_guided_selection` / `decoder_text_attention` 依赖文本头存在（`open_vocab is not None`）；`identity` 编码器仅允许测试/显式声明非研究用途（D8 的软门）。

### 3.9 训练引擎接线

- **构建顺序**：文本模块在 decoder/encoder/ssl_head 之后最后构建（RNG 隔离先例，`task_model.py:450-462`）——`open_vocab=None` 与基线逐比特（G0）；
- **导出面**：文本塔经 `object.__setattr__` 挂载、不入 `state_dict`/EMA/optimizer（distill teacher 先例，`train_model.py:285-307`）；`text_embeds` buffer、投影与 bias **随导出**；类名随 init-args 快照进导出元数据（`task_model.py:159-175` 先例）；
- **参数组**：文本投影进 detector 组（decoder 子模块自动归类）；`logit_scale`/`class_bias` 进 `ov_no_wd` 组（ssl 参数组先例 `train_model.py:933-950`）；文本塔无参数组（冻结）；
- **诊断**：`ov_scale`（logit_scale 当前值）、`ov_sim_gap`（matched 对的正负相似度差，val 聚合）、`ov_bias_range`；`log_dict` extras 落点同 `ssl_pool_size` 先例（`train_model.py:467-481`）；
- **数据面零改动**：文本是词表级资产而非逐样本输入，collate/transform 不动（`bboxes`/`classes` 契约不变，`collate.py:29-32`）。

---

## 4. 与既有机制的关系

| 机制 | 关系 |
|---|---|
| 分册 13 Dome 密度 | 正交且互补：密度管数量、文本对齐度管排序；`_get_decoder_input_dynamic` 零改动共享（§3.5）；消融矩阵单独设臂 |
| 分册 14 SSL | 无冲突：`encoder_candidates` 路线消费 decoder candidate 特征，与分类头解耦；`(B,Q,C)` 视图下 `last_ssl_stats` 管道不变 |
| 分册 15 SIoU / 分册 23 蒸馏 | 框回归腿不受文本头影响；**分册 23 的类数匹配约束被 v1 解除**：文本头无 `num_classes` 形状，`require_same_class_space` 的 cls-leg 死刑条款对文本头学生失效（v3 跨词表蒸馏的接口前提） |
| 分册 24 FP8 | 文本塔一次性 fp16 缓存前向，低于转换下限自然全精度（doc 24 §4.2 同 indexer 论断）；投影层随 detector 组进 FP8 混精管线 |
| 分册 25 token indexer | 完全正交（图像侧稀疏化 vs 语言侧分类）；可同开 |
| 分册 26 工程面 | 参数组/优化器接线遵守其 W1/W2 的"新组显式命名"纪律 |
| COCO 热启动（`coco_checkpoint`） | v1 启用时类别相关头跳过逻辑扩展为"文本头一律可加载 + `text_embeds` 按目标词表重建"——COCO→VisDrone 首次实现**全头迁移**（G3 热启动臂） |
| 分册 19 decoder parity | official decoder 的文本头/文本注意力需独立镜像（宽尾结构），测试矩阵必须双 decoder 覆盖 |

---

## 5. 差异与取舍清单

| # | 取舍 | 决策 | 理由 |
|---|---|---|---|
| D1 | 文本塔选择 | 冻结 SigLIP/CLIP 文本塔（v3 才考虑可训练 BERT） | GDINO 训 BERT 端到端的前提是 O365+GoldG 大数据；本仓算力下可训练文本塔只会过拟合 VisDrone 10 类；OWL-ViT 谱系证明冻结 CLIP 文本塔 + 对比头是可行形态 |
| D2 | 推进切分 | v1 头先行、v2 选择/注意力、v3 融合与数据 | 头是类别解耦的结构性根因，先解根因；选择信号换轴与注意力是增益项，独立消融；一次全上无法归因 |
| D3 | 逐类一票 vs 逐 token 子句 | v1/v2 逐类池化嵌入；token 级归 v3 | CLIP/SigLIP 上下文仅 77 token，600 类单 prompt 装不下；逐类独立编码天然绕开上下文限制，代价是丢子句内多 token 对齐——该对齐的价值需 O365 量级词表才显影，VisDrone/COCO 词表无损失 |
| D4 | 对比头损失语义 | 保留 VFL（IoU 质量加权）而非 GDINO 裸 focal | 本仓 D-FINE 线的匹配/损失栈全按 VFL 书写（`rtdetrv2_criterion.py`）；换 focal 会连锁改 matcher 代价与全部 aux 腿，违背 reduce-then-reuse；GDINO focal 的收益需在其数据规模下复验 |
| D5 | v1 范围：enc 头不动 | `enc_score_head` 保持闭集 Linear | 训练期选择本来就只在闭集词表内竞争；文本化 enc 头的主要价值在 v3 开放词表训练，提前换只增加消融变量 |
| D6 | DN 残差耦合 | `denoising_class_embed = Embedding(C+1)` 保留 | DN 只在训练词表内工作（含噪声类采样 `denoising.py:136-145`），无跨词表需求；文本化 DN 是纯增量风险，留 v3 |
| D7 | 相似度尺度 | 可学习 `logit_scale`（CLIP init/clamp）而非裸点积 | VFL 的 sigmoid 语义需要可控的 logit 量程；裸点积在 h=192~384 的小投影下早期量程不足 |
| D8 | 测试的文本塔 | `identity` 确定性伪编码器 | hermetic 纪律（`conftest.py` 断网 + tiny 配置先例）；真实塔只在 G1 钉扎清单与 smoke 脚本出现 |
| D9 | 投影层级 | 每层独立 `Linear(h→d)`（镜像 `ModuleList` 结构） | 保留逐层独立监督节奏与基线结构同构；共享投影的省参（≤0.2%）不值得引入跨层耦合变量 |
| D10 | 类名来源 | train args 显式 `class_names`（数据集 categories 仅交叉核对） | 显式资产、随导出元数据走；数据集 categories 运行期才存在（`train_model.py:342`），不适合做构建期输入 |

---

## 6. v0 门槛（实施前必过）

| # | 内容 | 工具 | 判据 |
|---|---|---|---|
| G0 | **基线逐比特**：`open_vocab=None` 时模块不构建、state_dict 键集合一致、前向逐位一致（tiny 配置） | `tests/test_ltdetr_open_vocab.py`（identity 塔） | 与 `tests/test_window_token_drop.py` 不变量 #1 同判据 |
| G1 | **文本塔钉扎**：siglip-base/clip-b32 权重 sha256 进 `_models/weight_hashes.py`；下载-校验-缓存全链路走通一次 | `scripts/` smoke（一次性联网） | 缓存文件落盘 + 断网重跑可命中 |
| G2 | **词表 token 预算**：VisDrone 10 / COCO 80 / O365 600 的逐类 prompt 编码时长与显存 | `scripts/visdrone_ov_token_budget.py`（新，纯统计无训练） | 600 类全词表一次性编码 < 60 s、缓存 < 50 MB；超限则 v1 限定 VisDrone/COCO 词表并记录 |
| G3 | **闭集无损 + 全头迁移**：VisDrone 10-epoch twin：(a) 文本头 from-scratch vs 类头基线，\|ΔmAP\| ≤ 0.5 AP；(b) COCO 检测点热启动：文本头臂 vs 丢头基线臂 | 两个训练臂（GPU） | (a) 无损；(b) 文本头臂显著优于丢头臂（预期 ≥ +1 AP）——全头迁移的价值实证 |
| G4 | **novel 协议正确性**：leave-3-out 评测管线在 `novel_eval_classes = all` 时复现基线 mAP | G2 脚本扩展 | 逐位复现 |
| G5 | G0–G4 通过 → v2 实施（§3.5/§3.6） | — | — |

---

## 7. 实现清单

| 文件 | 改动 | ~行 |
|---|---|---|
| `object_detection_components/open_vocab.py`（新） | 文本塔 wrapper（siglip/clip/identity）+ prompt 编码/缓存 + `ContrastiveClassHead` + decoder 文本交叉注意力模块 + VisDrone 类名常量 | ~350 |
| `rtdetrv2_decoder.py` | `dec_score_head` 头型分发（Linear/对比）；v2：enc 头分发 + 层内文本注意力注入（`:450-477`） | ~120 |
| `dfine_official_decoder.py` | 同款镜像（宽尾 `:666-672` + `_OfficialDecoderLayer` 注入 `:249-277`） | ~100 |
| `dfine_decoder.py` | LQE/出 logits 路径的头透传（`:296-301`） | ~30 |
| `dfine_criterion.py` | 零改动（reduce 视图）；仅 weight_dict 不动、诊断 sink 挂点确认 | ~10 |
| `ltdetr_object_detection/config.py` | `LTDETROpenVocabArgs` + 字段 + 校验器 | ~110 |
| `ltdetr_object_detection/task_model.py` | 文本模块最后构建 + args 往返 + v2 decoder 参数透传 | ~90 |
| `ltdetr_object_detection/train_model.py` | 嵌入缓存构建、塔排除导出面、`ov_no_wd` 参数组、`ov_*` 诊断、novel AP 聚合 | ~130 |
| `_metrics/map_evaluator.py` | novel/base AP 子集聚合出键 | ~25 |
| `scripts/visdrone_ov_token_budget.py`（新） | G2/G4 工具 | ~120 |
| `tests/test_ltdetr_open_vocab.py`（新） | §8 全部不变量 | ~450 |

总规模 ~4–5 天（G3 双臂训练另计 GPU 时间）。criterion/matcher/decode 零手术是本设计的验收红线——任何需要改这三处才能跑通的实现都说明 reduce 视图泄漏了。

---

## 8. 测试与验收（全部 hermetic，identity 塔 + tiny 配置）

1. **off-by-default 逐比特**（G0）：`open_vocab=None` 无 `text_embeds`/投影键、state_dict 集合一致、前向逐位一致；
2. **降维视图契约**：`ContrastiveClassHead` 输出严格 (B,Q,C)；同输入同输出（无随机性泄漏到前向）；
3. **校准不变量**：`logit_scale` clamp 生效（>max 时梯度穿 clamp 到 proj）；`class_bias` 初始化使早期 matched focal 代价与基线 `bias_init_with_prob(0.01)` 同量级（手算对账）；
4. **全栈零改动证明**：启用 v1 后 criterion 的 loss 键集合与基线相同（`loss_vfl/...` 无新键）；matcher 代价手算对账；
5. **v2 热启动安全**：`decoder_text_attention=True` 的第 0 步前向与 `False` 逐位一致（零初始化门控）；训练若干步后分叉；
6. **选择语义切换**：`language_guided_selection=True` 时 `_select_topk` 入参 logits 来自文本头（结构断言）；Dome 动态路径与 v1/v2 组合跑通 tiny 冒烟；
7. **缓存一致性**：同 (塔, 词表, 模板) 二次构建命中缓存（不实例化文本塔）；词表变动 → 缓存键变动重算；
8. **导出面**：文本塔不在 `state_dict`/EMA/optimizer 参数组；`text_embeds` buffer 在导出内；导出模型换 `class_names` 重建 `text_embeds` 后可推理（v1 的同域扩类冒烟）；
9. **校验器**：类数不匹配 / 类名重复 / novel 类越界 / v2 依赖 v1 头 等全部配置期 raise；
10. **novel 协议**（G4）：`novel_eval_classes=all` 复现基线 mAP 逐位一致；leave-3-out 的 base/novel AP 分解正确（构造微型 GT 验证）;
11. **双 decoder 矩阵**：rtdetrv2 / dfine / dfine_official 三 decoder × v1 必测，official 宽尾层单独断言；
12. **args 往返**：`model_dump`/re-coerce（density/ssl 同款）。

---

## 9. 消融与实验设计（VisDrone 800px twin 配方，G3 之后）

| 臂 | 配置 | 目的 |
|---|---|---|
| a | 基线（类头） | 参照系 |
| b | v1 文本头（from-scratch） | 闭集无损验证（G3a 复跑进矩阵） |
| c | b + 模板系综（prompt_templates） | prompt 鲁棒性 |
| d | b + v2 LGS（enc 文本头） | 选择信号换轴的闭集增益（GDINO 对齐收益的本地复验） |
| e | d + v2 decoder 文本注意力 | 第三处融合的增量 |
| f | COCO 检测点热启动：文本头臂 vs 丢头臂 | 全头迁移价值（G3b 正式版） |
| g | leave-3-out：7 类训练 / 10 类评测（b 基础上） | novel AP 基线（v3 grounding 数据的参照系，不承诺为正收益） |

记录：mAP、APs、base/novel AP、`ov_sim_gap`、`ov_scale`。验收线：b − a ≥ −0.5 AP（无损）；d − b ≥ +0.3 AP 则 v2 的对齐收益成立；e − d 单独归因。

---

## 10. 风险表

| 风险 | 等级 | 缓解 |
|---|---|---|
| R1 对比头闭集掉点（VFL 语义 vs cosine 几何不兼容） | 中 | D7 尺度 + D9 校准 bias；G3a 无损门槛前置；回退路径 = use_class_bias 加大权重或放弃对比头（v2/v3 不依赖 v1 成败） |
| R2 novel AP 近乎 0 被误读为"方向失败" | 中 | §1.3 预期管理写死：novel AP 是 v3 的参照系而非 v1 的 KPI；文档与实验记录同步声明 |
| R3 reduce 视图泄漏（某处隐含假设 logits=类别语义而非分数视图） | 中 | §7 验收红线 + §8.4 全栈 loss 键集合断言；Dome per-class NMS（`dome_decoder.py:153-211`）冒烟必测 |
| R4 文本塔下载/离线失败 | 低 | G1 钉扎 + 配置期报错 + `identity` 测试塔隔离；`-notpretrained` 式离线纪律 |
| R5 official decoder 宽尾镜像遗漏（`scaled_dim` 层头形状） | 中 | §8.11 双 decoder 矩阵；official 路径单独实现清单行 |
| R6 多词表缓存键碰撞（词表/模板哈希不含文本塔版本） | 低 | 缓存键 = (encoder, 权重 sha256 前缀, 词表, 模板) 四元组 |
| R7 VisDrone 类名与数据集内部 id 映射错位 | 中 | G4 协议正确性门槛 + `class_names` 与 `categories` 交叉核对校验器（§3.3） |

---

## 11. 后续路线

| # | 内容 | 备注 |
|---|---|---|
| P0 | G0–G2 门槛（1–2 天，含 G1 一次性联网） | 无训练 |
| P1 | §7 v1 实施 + G3 双臂（4–5 天 + GPU） | 本册主体 |
| P2 | G4 novel 协议 + §9 臂 g | GPU |
| P3 | v2：LGS + decoder 文本注意力（§3.5/§3.6）+ §9 臂 d/e | 视 P1 |
| P4 | v3：O365/GoldG 式 grounding 数据——GDINO-HF（transformers 5.19 已实测可导入）作分册 23 的 teacher 出伪框，解锁跨词表蒸馏（文本头解除 `require_same_class_space` 约束） | 研究线，大数据 |
| P5 | v3：可训练文本塔（BERT sub-sentence，GDINO 全集）+ encoder 文本融合 | 视 P4 |

---

## 12. 参考索引

**上游侧（【GDINO】事实来源）**：

| 内容 | 位置 |
|---|---|
| 三阶段融合（enhancer / language-guided query selection / cross-modality decoder）、对比头 [Q,T] + focal α0.25 γ2、sub-sentence 表征、O365+GoldG→Cap4M 两阶段、COCO zero-shot 52.5 AP | Grounding DINO，ECCV 2024；`github.com/IDEA-Research/GroundingDINO` |
| HF 实现（≥4.40）：900 queries / 6+6 层 / d_model 256 / max_text_len 256 / 双阈值（box/text）后处理；本仓 venv 5.19.0 导入实测 | `transformers` `GroundingDinoForObjectDetection`；HF 文档 `docs/transformers/model_doc/grounding-dino` |
| 冻结 CLIP 文本塔 + 逐类点积的先例谱系 | OWL-ViT（arXiv:2205.06230）；GLIP（arXiv:2112.03857） |
| 后续演进（1.5/1.6 权重不开放、API-only） | Grounding DINO 1.5（arXiv:2405.10300） |

**VTrain 受影响面（`src/autovision/`）**：

| 内容 | 位置 |
|---|---|
| 五个类别耦合点 | §1.1 表（`rtdetrv2_decoder.py:622-652`、`matcher.py:85-97`、`rtdetrv2_criterion.py:79-82`、`decode.py:38-43`） |
| 两段式选择（v2 手术对象） | `rtdetrv2_decoder.py:753-830`；Dome 动态版 `dome_decoder.py:94-245` |
| criterion 消费面（零改动证明对象） | `dfine_criterion.py:135-146, 228-749`；VFL scatter `rtdetrv2_criterion.py:79-82` |
| 可选分支契约/RNG 隔离/导出排除/参数组先例 | `config.py:402-744`；`task_model.py:450-462`；`train_model.py:285-307, 933-950` |
| 逐类 AP 与诊断管道（novel AP 落点） | `_metrics/map_evaluator.py:241-244`；`train_model.py:650-707`；`_engine/engine.py:886-963` |
| 权重钉扎/离线纪律 | `_models/weight_verification.py:381`；`_models/weight_hashes.py`；`tests/conftest.py:12-46` |
| hermetic 测试基线 | `tests/test_window_token_drop.py`；tiny 变体 `config.py:250-262` |

---

## 13. 实现记录（2026-10-12，v1.1）

### 13.1 落地面

| 文件 | 内容 |
|---|---|
| `object_detection_components/open_vocab.py`（新） | 文本塔 spec（siglip-base / clip-b32 / clip-b16 / identity）+ 结构化缓存键（§13.2）+ `ContrastiveClassHead`（`logit_scale` clamp + per-class bias，init = `bias_init_with_prob(0.01)`）+ `TextCrossAttention`（零初始化门控）+ `VISDRONE_CLASS_NAMES` |
| `rtdetrv2_decoder.py` | `_make_score_head` 头分发；enc 头 v2 分发（`_ov_lgs`，D5）；`_score_head_logits` **按头类型分发**（§13.2 P0）；`text_kv` 贯穿 `TransformerDecoder`/层；持久 `ov_text_embeds` 缓冲 + `load_ov_text_embeds` + 未填充前 forward 报错 |
| `dfine_decoder.py` / `dome_decoder.py` | `_refine_layers(text_embeds=…)` 贯穿；Dome enc 头分发（密度管数量、文本对齐度管排序，零改动共享） |
| `dfine_official_decoder.py` | 宽尾逐层宽度镜像（`hidden_dim`/`scaled_dim`）；`_OfficialDecoderLayer` gateway 后文本注意力；FDR 内两处调用点分发；`convert_to_deploy` 兼容（§13.2 P0） |
| `config.py` | `LTDETROpenVocabArgs` + 校验器；train args 类数校验；`novel_eval_classes` 允许全集（G4 self-check，§13.2） |
| `task_model.py` | args 快照 dict 往返；`decoder_kwargs` 注入；词表编码在 ssl_head 之后最后执行（`encode_class_names` 内部 `fork_rng`，塔不进模块树） |
| `train_model.py` | `set_categories` 词表交叉核对；`_base_novel_ap`（读 `last_per_class`）；`_fill_missing_ov_keys` checkpoint/EMA 容忍；`ov_logit_scale` 诊断；显式命名 `ov_no_wd` 参数组 |
| `coco_checkpoint.py` | 热启动跳过集扩展：改名的对比头键 + `.text_attn.` 零初始化子层 + 形状不匹配的跨词表 `ov_text_embeds`（按目标词表重建） |
| `scripts/visdrone_ov_token_budget.py`（新） | G2 工具（identity 默认 hermetic；真塔 opt-in） |
| `tests/test_ltdetr_open_vocab.py`（新，37 例） | §8 不变量全量 + 审查修复回归（部署路径/参数组命名/宽尾宽度/键集契约） |

### 13.2 审查修复（实施后审查发现，全部已修）

1. **[P0] official decoder 部署路径崩溃**：`convert_to_deploy` 把非 eval 层头换成 `nn.Identity` 后，按 `text_embeds is not None` 的分发会把 `text_embeds` 传给 Identity → TypeError（FDR 的 layer-0 `pre_scores` 调用在 eval 也无条件执行）。修复：`_score_head_logits` 改为**按头类型**（`isinstance(head, ContrastiveClassHead)`）分发。
2. **[P1] v1 enc 头误换对比头**：v1.0 实现把 `enc_score_head` 在 v1 就换成了对比头，违背 D5（v1 不动 enc 头）——纯 v1 + eval 路径 TypeError。修复：enc 头仅在 `_ov_lgs`（v2）时替换。
3. **[P1] 热启动容忍集**：`coco_checkpoint` 的 tolerated-missing 集扩展到 `.text_attn.`（零初始化子层不扰动热启动的语义保证）与形状不匹配的跨词表 `ov_text_embeds`。
4. **[P1] 缓存写入竞态**：冷缓存下 DDP 各 rank 并发编码，共享 tmp 名可能交错写坏缓存——改为 `tempfile.mkstemp` + `os.replace` 原子发布。
5. **[P1] 缓存键缺权重哈希（R6）**：键结构改为 JSON（encoder, **权重 sha256 前缀**, 模板, 类名列表）——重钉扎权重自动失效旧缓存，类名含分隔符不再碰撞。
6. **[P1] G4 配置不可达**：`novel_eval_classes` 禁止全集与 §3.7 矛盾——改为允许全集（novel_ap = 逐类 AP 均值 = mAP 复现检查；base_ap = -1 哨兵）。
7. **[P1] identity 塔软门**：`encode_class_names` 对 `identity` 发出响亮 warning（研究数字无意义，仅测试/冒烟）。
8. **[P1] 损坏缓存**：`torch.load` 失败 → warning + 重编码（原实现在坏字节上直接崩）。
9. **[P2] `agnostic` × LGS**：decoder ctor 校验拒绝（原实现下 `_enc_class_logits` 会对 Linear 双参调用）。
10. **[P2] `ov_no_wd` 命名参数组**：校准标量（`logit_scale`/`class_bias`，含 enc 头）从 detector 组摘出，独立命名组（doc 26 W1 纪律）。

### 13.3 与 v1.0 设计的偏差记录

- **诊断降采样**：`ov_sim_gap`/`ov_bias_range` 未实施（需要 criterion 配合，违反零手术红线），只保留 `ov_logit_scale`（log 域）。v2 消融若需相似度间隔，从 evaluator 侧另行探针。
- **`text_dtype` 字段未落地**：编码固定 CPU fp32 一次性执行（600 类 × 768 维 fp32 ≈ 2 MB，无降采样价值）。
- **novel/base AP 落点**：§7 原分配给 `_metrics/map_evaluator.py`，实际落在 `train_model._base_novel_ap`（读 `last_per_class`，零 pycocotools 手术）。
- **G2/G4 脚本拆分**：G4 协议正确性由测试套件承担（`novel=all` 复现检查在 `compute_metrics` 聚合层验证），脚本只承担 G2 预算统计。
- **`ltetr_default_batch_size` 拼写**（应为 ltdetr）：既有公共 API，不在本册范围内改名。

### 13.4 门槛执行记录（2026-10-13，v1.1）

- **G1 ✅ 已过**：三个文本塔经 `scripts/update_weight_hashes.py` 钉扎进 `_models/weight_hashes.py`（siglip-base 双格式 + clip-b32/b16 的 bin；`huggingface.co` 直连不可达，经 `AUTOVISION_MIRROR_URL=https://hf-mirror.com` 取 registry 元数据）。镜像桥接修复：`_bridge_mirror_to_hf_endpoint` 同时置 `HF_HUB_DISABLE_XET=1`——Xet CAS 服务器绕过 HF 端点且对匿名请求 401，禁用后回退普通 HTTP resolve（镜像可服务，钉扎 sha256 逐字节验证不变）。依赖落地：`sentencepiece`/`protobuf` 进 `transformers` extra（SigLIP tokenizer 必需，CLIP 为 BPE 不需要）。
- **G2 ✅ 已过**（判据：全词表编码 < 60 s、缓存 < 50 MB）：

| 塔 | 词表 | 嵌入 | 首跑（含加载/下载） | 缓存命中 | 缓存 |
|---|---|---|---|---|---|
| siglip-base | visdrone 10 | (10,768) fp32 | 5.4 s | ~0.05 s | 31.5 KiB |
| siglip-base | coco80 | (80,768) fp32 | 6.9 s | ~0.1 s | 241.5 KiB |
| clip-b32 | visdrone 10 | (10,512) fp32 | 62.7 s（含 605 MB 下载） | ~0.05 s | 21.5 KiB |

  实现勘误两处（本节修复）：`_tower_weight_digest` 对 `artifact.sha256`（dict）做结构化摘要；transformers v5 的 `get_text_features` 返回 ModelOutput——对齐嵌入取 `pooler_output`（CLIP 为 post-projection，SigLIP 无投影矩阵、池化即共享空间），且 SigLIP 必须 `padding="max_length"`（模型卡契约）。
- **G3 ▶ 进行中**（2026-10-13 启动）：OV 臂 = `scripts/visdrone_ltdetr_800_cocofull_openvocab.py`（cocofull 参照跑的逐比特孪生 + `open_vocab` siglip-base，40,443 步 / 50 epoch / batch 8 / 800px，out `runs/visdrone_ltdetr_convnextt_800_cocofull_ov`）。热启动面与参照跑完全一致（backbone 180 + encoder 246 + decoder 206 张量加载，15 个类别相关头跳过——两臂同等 fresh，头类型是唯一变量）。实测 0.73 s/步（参照 0.76，文本头训练期零开销符合设计）；显存 23.3/24.5 GB。**判据回填**：完成时 best mAP 对照参照跑 0.34646，|Δ| ≤ 0.005 即 G3a（闭集无损）通过；G3b 的跨词表迁移面已由 `test_warmstart_decoder_key_contract` + coco_checkpoint 容忍集覆盖。

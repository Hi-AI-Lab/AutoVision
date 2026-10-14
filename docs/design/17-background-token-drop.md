# VTrain 实现蓝图 · 分册 17：动态背景 Token 删除（R4 重评：window 级 in-backbone token dropping）

| 文档属性 | 内容 |
|---|---|
| 版本 | v1.2（已实施；G1 门槛真跑通过，实现记录见 §11，DINOv3 勘误见 §4.3.1；v1.2 = mask 构建 CPU 化，见 §11.5） |
| 状态 | ✅ 检测路径已实施：30 例新测试 + 147 例相关回归全过；G1 可丢率 46.7%（800/32px）/ 47.6%（812/28px）≫ 25% 双动机门槛；G2 骨干占比 profile 待 GPU 环境 |
| 来源设计 | rf-detr-ssl-v2 `docs/learn/train/levjepa-detr-integration.md` §9（R4），代码 `src/rfdetr/datasets/box_aware_eraser.py:315`、`src/rfdetr/models/backbone/dinov2_with_windowed_attn.py` |
| 上游思想 | LeVJEPA：随机互补子集丢弃（与分册 16 像素擦除同源）；R4 把"丢弃"从像素级抬到 token 级，真省 FLOPs |
| 参考实现 | `/home/hdjs/Research/Software/rf-detr-ssl-v2`（Apache-2.0） |
| 受影响模块 | `ltdetr_object_detection/config.py`、`task_model.py`、`train_model.py`、`backbone_wrappers.py`、`_models/dinov2/dinov2_wrapper.py`、`scripts/visdrone_token_drop_stats.py`（新）、`tests/test_window_token_drop.py`（新） |
| 基线兼容承诺 | `token_drop=None`（默认）时全部家族既有行为、权重、导出图**逐比特不变**；验证/导出路径强制稠密 |
| 重评依据 | 分册 16 §3 R4"否决（记录依据）"的两个前提在本设计下均被化解，见 §1.2 |

> **标注约定**：【RFSSL】rf-detr-ssl-v2 参考实现中的已验证事实（出处 `文件:行`，路径相对该仓库）；【VTrain】本仓库现状（出处 `文件:行`，路径相对 `src/autovision/`，行号对应当前工作树）；【设计】本文档决策；【差异】有意偏离 rf-detr-ssl-v2 的决策（编号 D1–D5 及理由）。

---

## 1. 背景与重评理由

### 1.1 机制一句话

数据侧按 GT 框生成每图一份 **window 级 keep mask**（框重叠窗全保、背景窗按 ratio 随机保留），模型侧在 ViT backbone 的 **patch embedding 之后、encoder 之前** 把被删窗口的 token 从序列里 gather 掉——被删窗口的 encoder 计算被真实跳过，输出端把丢弃位置**零回填**成稠密特征图，FPN/PAN/decoder 全链路无感知。train-only；`prob` 掷币、`step_stop` 退火，关掉即基线。

### 1.2 为什么重学分册 16 的 R4 否决

分册 16 §3 对 R4 的否决依据有两条，本设计逐一化解：

| 否决依据（分册 16 §3） | 本设计的化解 | 性质 |
|---|---|---|
| (a) 参考实现窗口 = 整图 2×2 四分之一图，VisDrone 87% 的图 4 窗全含框，期望可丢窗口仅 **4.4%** | 【设计 D4】VTrain 骨干是 HF 全局注意力 ViT，**没有窗口划分几何**，mask 粒度自由可选：32px 窗 @800px → 25×25 = 625 窗（对比参考的 4 窗），box-aware 保护后可丢率由数据决定而非结构锁死 | 前提失效 |
| (b) VTrain 骨干是 HF DINOv2/v3（非【RFSSL】本地 windowed-dinov2 fork），R4 不移植 | 【设计 D5】不移植 fork、也不需要 fork：复用 HF `Dinov2Model` 现成模块（`embeddings`/`encoder`/`layernorm`）重新编排前向即可，~90 行；代价见 §5 分布偏移声明 | 前提失效，代价换动机 |

**动机重定位（【设计 D2】，本设计最重要的一条）**：参考实现依赖"窗口注意力无跨窗依赖"做到**丢弃窗口逐比特一致、零分布偏移**（纯加速）；VTrain 的全局注意力下，删除 token 会改变剩余 token 的 attention 上下文——partial keep **不是**无损加速，而是"**结构化真缺失增广**（强于分册 16 的灰格占位）+ 训练吞吐加速"的复合动机。【RFSSL】文档 §9.7 对此早有定性："V1 灰擦除已覆盖'信息抹除'价值；R4 的边际 = 真缺失（更强）+ 速度"。增广语义下分布偏移不是缺陷而是作用机制，与 dropout 同类；`step_stop` 退火保证训练末期回到稠密分布。

---

## 2. 参考实现机制详解（rf-detr-ssl-v2）

### 2.1 全链路（数据 → 传输 → 模型 → 输出）

| 环节 | 位置 | 机制 |
|---|---|---|
| keep mask 生成 | `datasets/box_aware_eraser.py:315-345` `build_window_keep_mask` | 与像素擦除**同一套选择逻辑**：`protected_cell_mask(cell=window)` 保框窗 → `sample_erase_mask(protected, 1-ratio)` 采丢背景窗 → `~drop` 即 keep mask，形状 `(H//window, W//window)` |
| 数据侧挂载 | 同上 `:348-420` `WindowKeepMaskDataset` | per-view 掷币 `prob`；空框 / 尺寸不整除 window → 稠密跳过；`target["window_keep_mask"]` bool 张量随 target 流动 |
| 训练步接线 | `training/module_data.py:377-388` | `token_drop_prob>0` 才包 wrapper（默认关）；`window = patch_size × num_windows` |
| 模型侧提取 | `models/lwdetr.py:493-515` | 从 targets 读 mask → **pad 到批网格**（`value=False`：pad 窗口 = 丢弃 = 纯节省）→ `backbone(samples, window_keep_masks=...)` |
| 删除点 | `backbone/dinov2_with_windowed_attn.py:469-505` | patch token 重排为 `(B×num_windows², 每窗 token 数, D)`（窗口即 batch 元素，各带独立 CLS+registers）；keep mask `nonzero()` gather = **纯 batch 收缩** `:501-503`；`last_kept_window_indices` 存 self 供出口用 |
| encoder 前向 | 同上 `:1352-1386` | mask 存在时**逐图循环**（每图 B=1，kept 数可不同；小 batch 下循环开销可忽略），全注意力层走 `flatten_windows` 变长合并 |
| 输出端回填 | 同上 `:1414-1485` `_collect_feature_maps` | kept 分支先把 `(k, T_w, C)` **零回填**成全 `(B×num_windows², T_w, C)` `:1454-1462`，再复用原 undo-windowing → 全图稠密特征图，**丢弃窗口区域恰为 0** |
| 配置 | `config.py:1123-1124` | `token_drop_prob=0.0`（关）/ `token_drop_ratio=0.4` |

### 2.2 实测结论（【RFSSL】§9 实现记录，2026-10-06）

- **机制正确性**：`tests/models/test_window_token_drop.py` 10 项，核心不变量 = keep-all 与稠密前向逐比特一致（含全注意力层）、partial keep 精确零回填、batch=逐图拼接、mask 缺省走原路径。
- **效率**：解析 FLOP 占比 encoder ≈ 85%（≫35% 门槛）；Nano @960 forward+backward 实测 dense 346.3 ms → keep~40% 145.4 ms = **2.4×**。
- **数据集判定**：VisDrone 上**默认关**——num_windows=2 的四分之一图粒度对密集航拍太粗；重评条件"未来 backbone 支持更细窗口划分"。

### 2.3 ⚠️ 参考实现的一个潜在接线 bug（本设计的避坑依据）

数据侧 `window = patch_size × num_windows = 16×2 = 32px`（`module_data.py:381`）生成 **30×30** 的 mask（@960），而模型侧 pad 目标是 `(num_windows, num_windows) = (2,2)`（`lwdetr.py:504-513`）——`F.pad` 负填充在本仓库环境实测**静默裁剪**（30×30 → 左上角 2×2），即整图只有左上 4 个窗口参与 keep 语义。模型侧单测直接构造 (2,2) mask，未覆盖数据接线，故盲区。【设计 D1】VTrain 把 mask 生成挪到 **training_step**、在最终画布上网格语义**单点定义**，结构上消灭这类错位。

### 2.4 参考实现期踩过的 6 个坑（移植检查单）

① 透传漏 `Joiner`（backbone 是 Sequential 包装）→ TypeError；② 包装器属性路径（`.patch_size/.num_windows` 而非 `.config`）；③ mask 网格语义（§2.3）；④ 重构 un-merge 时 shape 重读提前破坏 dense 路径依赖；⑤ kept 分支 undo reshape 的 batch 维 `batch_windows//W²` 在 k=1 时为 0；⑥ `lwdetr.py` 缺 `F` 导入。

---

## 3. VTrain 现状与注入点裁定

### 3.1 骨干家族矩阵（token 删除的适用面）

| 家族 | 包装 | token 序列 | R4 适用 | 说明 |
|---|---|---|---|---|
| `vit` | `ViTDetectionBackbone`（`backbone_wrappers.py:125-264`）→ HF `Dinov2Model`/`Dinov3ViTModel`（`_models/dinov2/dinov2_wrapper.py:30`，`_models/dinov3/dinov3_wrapper.py:27` 为其子类） | ✅ patch token（patch 16 @800 → 2500 token） | **✅ 本设计唯一目标** | 全局注意力；HF 无 `attention_mask` → 只能 gather 不能 mask-attention |
| `convnext` | `ConvNeXtDetectionBackbone`（`backbone_wrappers.py:267-325`） | ❌ CNN | ❌ | **VisDrone 800 两个脚本实际用的家族**（`scripts/visdrone_ltdetr_800.py:77`、`visdrone_ltdetr_dome_800.py:123`）；卷积感受野跨空间，删空间块既不省算力也破坏语义 |
| `ecvit` | `ECViTDetectionBackbone`（`backbone_wrappers.py:328-352`） | ✅（pass-through wrapper） | 二期 | wrapper 未暴露内部编排，先不做 |

【设计 D3】配置期门控 `family=="vit"`（复用 `pyramid_levels=4` 的家族检查先例，`task_model.py:198-205`）；其余家族传 `token_drop` 直接报错。

### 3.2 ViT 前向链与手术点

`training_step`（`train_model.py:181-257`）→ `_forward_train`（`task_model.py:455-513`，backbone 调用 `:458`）→ `ViTDetectionBackbone.forward`（`backbone_wrappers.py:204-264`）→ `forward_multiscale_features`（`dinov2_wrapper.py:68-96`：**单次全前向** `output_hidden_states=True`，在 `interaction_indexes` 处 tap `hidden_states[i+1]`，`_reshape_patch_tokens`（`:131-153`）剥 CLS+registers 后 reshape 成 (B,D,H/ps,W/ps)）→ 三 tap 双线性 resize 到 8/16/32 步距（`backbone_wrappers.py:219-238`）+ 可选 STA 细节金字塔 concat（`:240-256`）→ 1×1 conv+SyncBN 投影（`:261-264`）。

【设计】删除点 = **embeddings 之后、encoder 之前**（同【RFSSL】§9.2 裁定：此点删 token，所有 tap 同步受益，resize/STA/投影代码零改动；encoder 出口/projector 之后删则无节省且破坏 deformable 采样语义）。STA 分支独立消费原图，不受影响。

### 3.3 与既有机制的关系（正交性核查）

| 机制 | 层级 | 与本设计的关系 |
|---|---|---|
| 像素擦除（分册 16） | 数据侧像素 | **独立可叠加**：擦除是灰格占位，删除是真缺失；R4 边际 = 真缺失 + 速度（§1.2）。两者保护语义同源（margin 外扩、any-overlap），但采样互不影响 |
| Dome MWAS | encoder stride-16 注意力 | **正交**：MWAS 是注意力计算稀疏化（窗内 attention，特征原地残差，`mwas.py:5-9`），不删 backbone token。协同向：被删骨干 token 输出零特征 → 密度头低响应 → `adaptive_window_mask`（`mwas.py:59-92`）倾向跳过对应 encoder 窗 |
| SSL taps | 同一前向 | raw tap 读 hidden states，被删位置为 0——与"真缺失"增广动机一致，SSL 消融记录在案即可 |
| Dome 查询门控 / 诊断 | decoder / val | `defe_info` 流不受影响；`dome_window_ratio` 诊断（`train_model.py:419-423`）是新增 `token_keep_ratio` 的现成模板 |

---

## 4. VTrain 移植设计

### 4.1 配置面（`config.py`，仿 `LTDETRDensityArgs` 模式 `:381-419`）

```python
class LTDETRTokenDropArgs(BaseModel):      # model_config = {"extra": "forbid"}
    window_size: int = 32        # 窗口边长 px（须被 patch_size 整除；画布不整除→该样本稠密跳过）
    keep_ratio: float = 0.4      # 背景窗保留率 (0, 1]
    prob: float = 1.0            # 每样本掷币通过率；未通过样本稠密
    padding: float = 0.15        # GT 框外扩比例（与像素擦除同语义）
    step_stop: int | Literal["auto"] | None = None   # 复用 resolve_step_schedule（分册 16 P1.5 同款）

# LTDETRObjectDetectionTrainArgs 新增（:693 density 同款）：
    token_drop: LTDETRTokenDropArgs | None = None    # None = 逐比特基线
```

接线义务：`LTDETRObjectDetectionTrain.__init__` 显式透传（`train_model.py:63-93` → `task_model.py:121-148`）；args `model_dump()` 快照 + 重建时 re-coerce（`task_model.py:159-166`、`:206-207`、`:227-228`）；无新参数 → LR 分组（`train_model.py:493-505`）零改动。

### 4.2 数据侧：mask 在 `training_step` 生成（【差异 D1】）

【RFSSL】用 Dataset wrapper；【设计 D1】VTrain 在 `training_step`（`train_model.py:183-187` 组 targets 处）生成——**最终画布确定之后**（batch scale jitter 之后，脚本已锁 800：`visdrone_ltdetr_800.py:157-165`），网格语义单点（消解 §2.3 类错位），且 **transform/collate 零改动**（绕开 `collate.py:234-241` 键白名单与 two-view 展平逻辑）：

```python
# 逐样本：prob 掷币 → 失败/空框/尺寸不整除 window_size → 无 mask（稠密）
# 否则：protected = protected_cell_mask(boxes_norm, H, W, cell=window_size, padding)   # 复用 _transforms/.../box_aware_eraser.py:44
#        keep = ~sample_erase_mask(protected, 1 - keep_ratio, generator)               # 复用 :95——选择逻辑与像素擦除单点统一
#        targets[i]["window_keep_mask"] = keep
```

step 门控（`step_stop`）复用像素擦除的步窗机制；`validation_step` 永不附加（val/导出强制稠密，E2 同款）。criterion 只索引已知键，多余 key 无害（【VTrain】targets 全链路只读不弹出）。

### 4.3 模型侧：提取 → pad → 透传 → per-image 编排

1. **`task_model.py::_forward_train`**（`:455`）：从 targets 收集 `window_keep_mask`（全有或全无，否则报错）→ `F.pad` 到批网格 `value=False`（pad 窗=丢弃=纯节省，`lwdetr.py:507-513` 同款）→ `self.backbone(x, window_keep_masks=...)`；mask 缺省走原路径逐比特不变。
2. **`backbone_wrappers.py::ViTDetectionBackbone.forward`**（`:204`）：新增可选 kwarg 透传进 wrapper；其余家族分支不收该 kwarg（D3 门控保证不会到达）。
3. **`dinov2_wrapper.py`** 新增 `forward_multiscale_features_masked(x, layer_indices, keep_masks)`（dense 路径 `:68-96` 不动）：
   - 逐图循环（kept 数每图不同；HF 无 attention_mask 只能走 gather，与【RFSSL】`dinov2_with_windowed_attn.py:1352-1386` 同裁定）；
   - 每图：`emb = self._model.embeddings(pixel)` → 剥 CLS+registers 前缀 → 按 keep mask 展开到 patch 粒度 gather（`window_size` 对齐 patch 网格；`nonzero` 索引）→ 重接前缀 → `self._model.encoder(..., output_hidden_states=True)`（tap 语义 = 无终局 layernorm，与现路径一致）；
   - 每 tap：**零回填** `full[img, kept_patch_idx] = tokens` 后复用 `_reshape_patch_tokens`（`dinov2_wrapper.py:131-153`）→ 输出形状与稠密路径完全一致，下游 resize/STA/投影（`backbone_wrappers.py:219-264`）零改动；
   - init 期断言 HF 内部属性（`embeddings`/`encoder`/`layernorm`）存在（防坑 ②，DINOv2 与 DINOv3 属性面差异在此暴露）。
4. **DINOv3**：`TransformersDinov3ViTWrapper` 继承即得；唯一分叉是 embeddings 内部（rope），对"整块复用 `.embeddings` 再索引输出"的编排方式透明。

### 4.3.1 勘误（实施期实证，v1.1）：DINOv3"继承即得"不成立

实测（transformers 5.19）推翻 §4.3 第 4 条的预测：`DINOv3ViTModel` 的内部布局是 `embeddings` / `model` / `norm`（属性名不同），且 encoder 需要显式传入 `position_embeddings = self.rope_embeddings(pixel_values)`——rope 不是 embeddings 内部细节，而是前向签名的一部分。按 §9 风险表预案处置：**v1 结构门控限定 DINOv2 布局**——`TransformersDinov2Wrapper.supports_masked_forward()` 检查 `embeddings` + `encoder` 属性，任务模型 init 期对不满足者报配置错（含针对 DINOv3 的说明文案），`tests/...::test_dinov3_vit_layout_rejected` 钉住该行为。DINOv3 masked 支持的明确配方可依 §2.1 同款编排补做：`embeddings(pixel)` → `rope_embeddings(pixel)` → `model(hidden, position_embeddings=rope)` → 出口零回填——留待 ViT-v3 配方立项。

> **后续**：该配方已于分册 20 立项并落地（`docs/design/20-dinov3-vit-masked-forward.md`，2026-10-10）：DINOv3 ViT wrapper 自持 masked 前向（rope 与 token 同 kept 索引 gather），本节门控对 DINOv3 ViT 自动放行，拒绝测试翻转为放行断言。

### 4.4 诊断与消融开关

- `token_keep_ratio`（实际保留窗/总窗，含全样本均值）镜像 `dome_window_ratio`（`train_model.py:419-423`）进 val 日志——这也是 VisDrone 真实可丢率的在线计量。
- 脚本开关仿 Dome 脚本 `--erase` 模式（`visdrone_ltdetr_dome_800.py:262-268`）：ViT 变体脚本加 `--token-drop`，`MODEL_ARGS["token_drop"] = {...}`；inert 键预置仿 `:203`。

---

## 5. 分布偏移声明（【设计 D2】的落地含义）

| 不变量 | 参考实现（windowed fork） | VTrain（HF 全局注意力） |
|---|---|---|
| keep-all ≡ dense | ✅ 逐比特（纯 batch 收缩） | ✅ **仍逐比特**（gather-all + scatter-all = 恒等；测试不变量保留） |
| partial keep 对保留 token | ✅ 逐比特一致（窗内注意力无跨窗依赖） | ❌ attention 上下文变化——**增广语义**（真缺失），非无损加速 |
| 输出端被删区域 | 精确零回填 | 精确零回填（同款；投影后呈 conv bias 响应，与 V1 灰格同款"可学习无信息区"） |

推论：吞吐收益（若有）与增广收益**捆绑出现、无法解耦归因**——消融矩阵按增广实验设计（± token_drop × ± erase），速度收益用 step 时长直接测。`step_stop` 退火保证末期回稠密（参考 §9.4："衰减到 dense 后速度收益同步消失，增广价值优先时这是正确代价"）。

---

## 6. v0 门槛（实施前必过，均为零模型改动）

| # | 内容 | 工具 | 判据 |
|---|---|---|---|
| G1 | **可丢性量化**：VisDrone train @800、window 32px、padding 0.15，离线统计每图 keep 分布（只需 `protected_cell_mask`+`sample_erase_mask`） | 新增 ~60 行数据统计脚本 | 期望可丢窗 <10% → 降级为纯增广动机（只做 ± 消融、不宣传速度）；>25% → 加速+增广双动机成立 |
| G2 | **骨干占比 profile**：ViT 变体（非 ConvNeXt）单步时分解 | `scripts/profile_step_phases.py` | backbone <35% 步时 → 速度动机降级（【RFSSL】§9.7 同阈值） |
| G3 | G1+G2 通过 → 实施 §7；失败 → 记录数据归档本册，R4 维持否决 | — | — |

---

## 7. 实现清单

| 文件 | 改动 | ~行 |
|---|---|---|
| `ltdetr_object_detection/config.py` | `LTDETRTokenDropArgs` + `token_drop` 字段 + D3 家族校验 | ~60 |
| `ltdetr_object_detection/train_model.py` | `training_step` mask 构建（§4.2）+ `token_keep_ratio` 诊断 | ~40 |
| `ltdetr_object_detection/task_model.py` | args 透传/dump/re-coerce + `_forward_train` 提取/pad/传参（§4.3.1） | ~50 |
| `ltdetr_object_detection/backbone_wrappers.py` | `ViTDetectionBackbone.forward` kwarg 透传 | ~10 |
| `_models/dinov2/dinov2_wrapper.py` | `forward_multiscale_features_masked` + 零回填 helper + 属性断言（§4.3.3） | ~90 |
| `scripts/` | ViT 变体脚本 `--token-drop`；G1 统计脚本 | ~80 |
| `tests/test_window_token_drop.py` | 新建，见 §8 | ~200 |

总规模 ~1.5–2 天（对齐【RFSSL】§9.6 v1 账面）。

---

## 8. 测试与验收（tiny CPU 配置，仿【RFSSL】`test_window_token_drop.py` + 本仓库契约测试风格）

1. **keep-all ≡ dense 逐比特**（全部 tap）——核心不变量，参考实现同款；
2. partial keep：每 tap 被删 patch 区域**精确为 0**、保留区有限非零；
3. 不同 mask 的 batch ≡ 逐图前向拼接（逐比特）；
4. mask 缺省 → 与基线路径逐比特一致（回归）；
5. `token_drop=None` off-by-default 契约（`test_off_by_default` 同款）；
6. val 配方隔离：验证前向永无 mask；
7. mask 构建单测：框窗永不丢、精确 `round(ratio×|bg|)` 采样数、空框/不整除→稠密跳过、generator 复现、prob=0 恒等；
8. args 往返：`model_dump`/re-coerce（density/ssl 同款）；D3：convnext+token_drop 配置报错。

---

## 9. 风险表

| 风险 | 等级 | 缓解 |
|---|---|---|
| HF 内部属性面（`embeddings`/`encoder`/`layernorm`）跨 DINOv2/v3 版本漂移 | 中 | init 断言 + 属性访问单点封装；属性断言失败 → 配置期报错而非训练中途 |
| partial keep 分布偏移伤 mAP | 中 | 本身即假设的消融对象；`step_stop` 退火 + `prob` 掷币提供强度旋钮；60ep 止损线 |
| VisDrone 可丢率不足（G1） | 中 | 门槛前置，失败即止，零模型改动沉没成本 |
| ViT 家族非当前 VisDrone-800 主力（ConvNeXt） | 低 | 如实定位为 ViT 变体能力；ConvNeXt 无 token 序列，不做空间块删除（感受野破坏，无算力收益） |
| 逐图循环开销 | 低 | 参考 Nano@960 实测循环开销可忽略（§2.2）；batch 小 |
| per-image 前向 × DDP | 低 | mask 逐图、无 collective（【RFSSL】§9.5）；无新参数，`find_unused_parameters` 不受影响 |

---

## 10. 后续路线

| # | 内容 | 备注 |
|---|---|---|
| P0 | G1/G2 门槛脚本 + 数据统计 | 半天，先于一切模型改动 |
| P1 | §7 实施 + §8 测试 | 1.5–2 天 |
| P2 | 消融矩阵：± token_drop × ± erase（ViT 变体，VisDrone 800） | ±0.3 mAP 噪声线；`token_keep_ratio`/step 时长同步归档 |
| P3 | ECViT wrapper 暴露内部编排后跟进 | 二期 |

---

## 11. 实施记录（2026-10-08，v1.1 落地）

### 11.1 G1 门槛真跑（全量 VisDrone train，6471 图，零模型改动）

| 配置 | 期望可丢窗（mean / median / p90） | 完全保护图 | 判定 |
|---|---|---|---|
| 800px / 32px 窗（25×25=625 窗，patch-16 系） | **46.7% / 48.2% / 55.8%** | 0% | ≫25% → **加速+增广双动机成立** |
| 812px / 28px 窗（29×29=841 窗，patch-14 系，如 vittest/vits14） | **47.6% / 49.2% / 56.2%** | 0% | 同上 |

对照参考实现 num_windows=2 的 4.4%（87% 图四窗全含框）：doc 17 §1.2 的预判——"更细粒度化解密集航拍可丢率"——被全量数据证实。工具：`scripts/visdrone_token_drop_stats.py`（--window/--resolution/--keep-ratio 可调）。

### 11.2 实现清单（与 §7 的偏差标注）

| 文件 | 实际改动 | 对 §7 |
|---|---|---|
| `config.py` | `LTDETRTokenDropArgs`（window_size/keep_ratio/prob/padding/step_stop）+ `token_drop` 字段 + 包导出 | 如计划 |
| `train_model.py` | `_build_window_keep_masks`（training_step 画布上构建，prob 掷币 / 空框 / 不整除 / all-keep 四类稠密跳过）+ `token_keep_ratio` 诊断 + `get_optimizer` 解析 `step_stop="auto"`（复用 `resolve_detection_step_schedule`） | 如计划 |
| `task_model.py` | init 参数/dict 重绑定/`_init_args` 快照 + 家族与 patch 对齐校验 + **结构门控**（§4.3.1）+ `_extract_window_keep_masks`（None→全保 pad、过小 mask→False pad、设备搬运）+ `_forward_train` 分支 | 如计划 + 结构门控加码 |
| `backbone_wrappers.py` | `ViTDetectionBackbone.forward(window_keep_masks=)` 分支到 masked wrapper | 如计划 |
| `dinov2_wrapper.py` | `_kept_patch_indices`（整除硬校验，§2.3 教训）+ `supports_masked_forward` + `forward_multiscale_features_masked`（per-image：embeddings → patch gather → encoder(`output_hidden_states=True`) → 每 tap 零回填 `index_copy_` → 复用 `_reshape_patch_tokens`） | 如计划，DINOv3 改为显式拒绝 |
| `scripts/` | **G1 脚本已落地并真跑**；`--token-drop` 训练脚本开关**未加**——全部现有 VisDrone 脚本均为 ConvNeXt（token_drop 不适用），smoke 脚本为直线式无 argparse；flag 待 ViT-variant VisDrone 配方立项时随配方落地（接线即 `MODEL_ARGS["token_drop"] = {...}`，等价路径已被测试覆盖） | 偏差，如实记录 |
| `tests/test_window_token_drop.py` | 30 例：wrapper 五大不变量（keep-all≡dense 逐比特、partial 精确零回填、batch≡逐图拼接、None≡dense、网格/数量硬报错、梯度流达 kept token）+ mask 构建 7 例（框窗永不丢、精确采样数、空框/不整除/prob=0/all-keep 稠密跳过、种子复现）+ 契约 7 例（off-by-default、dict 重绑定快照、patch 对齐、convnext/ecvit/dinov3-vit 三族拒绝）+ 集成 8 例（E2E all-keep≡dense 逐比特、val/导出稠密、pad 语义、非 ViT 硬 guard、training_step 诊断与 step_stop 门、auto 解析）+ 选择语义复用 1 例 | 如计划 |

### 11.3 验证与回归

- 新套件 30/30 过（`uv run python -m pytest tests/test_window_token_drop.py`）。
- 相关回归 147 过（ltdetr_object_detection / ltdetr_dome / box_aware_eraser / detection_data_transforms）+ e2e 训练 7 过。
- 全仓套件 1289 过 / 1 失败——失败项 `test_methods_distillationv3` golden 快照经 stash 隔离验证**在本改动之前即失败**（工作树既有 hot-path 改动所致），与本期无关。
- `ruff check` 全绿。
- G2（骨干步时占比 profile）待 GPU 训练环境，工具沿用 `scripts/profile_step_phases.py`。

### 11.4 实施期抓出的坑（对照 §2.4 清单）

① 透传链 `targets → _forward_train → extractor → backbone kwarg → wrapper` 六处缺一即 TypeError——以 E2E all-keep≡dense 测试钉住；② `dinov2_wrapper.py` 原本未导入 `torch`（与参考仓库坑⑥同类，冒烟首跑即抓到）；③ 网格语义单点化 + `_kept_patch_indices` 整除硬校验，从结构上排除参考仓库的静默 `F.pad` 裁剪坑③；④ DINOv3 布局差异（§4.3.1）——文档预测被实证推翻，按风险预案拒绝而非沿用。

### 11.5 审查修复（2026-10-08，v1.2）：mask 构建迁到 CPU 侧（P1 性能）

原实现在 `training_step` 里对**已搬运到设备**的 boxes 构建 mask：`protected_cell_mask` 的 `.cpu()`、`sample_erase_mask` 的 `.item()`、`keep.all()`、`keep_fractions` 的 `.item()` 合计每样本 2–4 次 GPU→CPU 同步（B=16 时每步 30–60 次），直接侵蚀 token-drop 的步时收益，且会让 G2 的骨干占比 profile 测出 sync 开销而非 backbone 占比。修复：`_build_window_keep_masks` 改为接收 collate 输出的 **CPU 侧 boxes + 画布 (H, W)**，在 `.to(device)` 之前逐样本构建（宿主侧数学，零同步），返回 CPU mask，由 `_extract_window_keep_masks` 统一搬运；RNG 调用序列不变，同种子输出逐位一致。G2 profile 须在本修复之后执行。

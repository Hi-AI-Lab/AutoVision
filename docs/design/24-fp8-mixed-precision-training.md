# VTrain 实现蓝图 · 分册 24：FP8 混合精度训练（`precision="fp8-mixed"` 档位）

| 文档属性 | 内容 |
|---|---|
| 版本 | v1.1（v1.0 设计评审稿 → **v1.1 实测否决归档**，2026-10-12；实测记录见 §13） |
| 状态 | ❌ **否决实施（有实测依据）**：4090D 上对本项目全部相关 GEMM 形状实测，fp8 相对 bf16 为 0.68–1.03×（无收益乃至倒退，§13.3）。蓝图与门槛保留完整，供 §13.5 的复核条件触发时重启；在此之前**不要**投入 §7 的实施工时 |
| 上游思想 | DeepSeek-V3（arXiv:2412.19437）首次大规模 FP8 训练：①细粒度 tile-wise（1×128 激活 / 128×128 权重）在线量化，拒绝 delayed scaling；②**敏感算子保持高精度**（embedding 输出、attention 核心、gating、归一化、首尾层）；③只对 Linear GEMM 量化，存储/优化器 master 权重不变。【DS】 |
| 配套分册 | DeepSeek 迁移四联册：本册（FP8）+ 分册 23（蒸馏）/ 25（token indexer）/ 26（步工程优化） |
| 参考实现 | `torchao.float8`（`convert_to_float8_training`，官方 PyTorch FP8 训练路径）；torchao **0.18.0** 与本仓 torch 2.14.1+cu130 经 `uv pip install --dry-run` 解析兼容（§13.1，未常驻安装） |
| 硬件前提 | RTX 4090D（sm_89 / compute capability 8.9，FP8 tensor core ✅，实测 `nvidia-smi` 确认） |
| 受影响模块 | `_commands/train_task/object_detection.py`（转换点）、新增 `_accelerator/float8.py`（policy + 转换 helper）、`_engine/engine.py`（precision 表 + 导出卫生）、`pyproject.toml`（可选依赖）、`tests/test_fp8_training.py`（新） |
| 基线兼容承诺 | 不选 `fp8-mixed` 时全部行为逐比特不变；`fp8-mixed` 不改变 master 权重 dtype / 优化器 / EMA 数学 / 导出图结构（仅 state dict 含 fp8 缓冲键，导出面过滤，§4.5） |

> **标注约定**：【DS】DeepSeek 论文中的已验证事实；【VTrain】本仓库现状（出处 `文件:行`，路径相对 `src/autovision/`）；【设计】本文档决策；【差异】重大取舍（D1–D6）。

---

## 1. 背景：DeepSeek-V3 的 FP8 三原则与本仓的同构性

【DS】DeepSeek-V3 的 FP8 方案（技术报告 §3.3）可以压缩为三条原则，每条在本仓都有直接对应物：

| DeepSeek-V3 原则 | 本仓对应物 | 状态 |
|---|---|---|
| 细粒度 tile-wise 在线量化（1×128 / 128×128，per-tile amax） | torchao `Float8LinearConfig` 的 tilewise/rowwise recipe | 待引入（§4.2） |
| **敏感算子高精度隔离**：attention 核心、gating、normalization、首尾层不进 FP8 | **fp32 island 惯例已是既成架构**：decoder box/distribution 头 fp32（`dfine_decoder.py` 的 fp32 island，分册 14 §2.1 引 `:178-181`）、SSL 投影头 fp32 归一化（`ssl_contrastive.py:185` 的设计描述）、密度损失 fp32（分册 13）、criterion 全程 fp32 island | ✅ **本仓设计哲学与 DeepSeek 同构，这是本册可行性的根本依据** |
| master 权重/优化器状态不量化（FP8 只动 GEMM 计算） | autocast 语义下参数保持 fp32（`engine.py:807-811` 的 autocast 只影响计算 dtype） | ✅ 天然成立 |

【VTrain】精度管道现状：

- 命令层解析：`_commands/train_task/object_detection.py:334-340`——`precision="auto"` → CUDA 上 `"bf16-mixed"`；
- 引擎层表：`_engine/engine.py:696-698`——`autocast_dtype = {"bf16-mixed": torch.bfloat16, "16-mixed": torch.float16}`（预训练引擎 `engine.py:289-295` 同构）；
- TF32 已开（`engine.py:588-592`）；无 GradScaler（bf16 无需）；NaN 总损失守卫（`engine.py:833-838`）；
- **无任何 FP8 路径**；量化只在部署端（INT8 PTQ，`_export/quantization.py`）。

---

## 2. 设计目标与非目标

**做**：
- 命令层新增 `precision="fp8-mixed"`：torchao `Float8Linear` 转换 + 名称策略排除敏感 Linear；
- 转换点在命令层 `train_model` 构建后、`fit_task` 之前（含 EMA 副本，§4.3，D1）；
- 与 bf16 autocast 叠加：FP8 只接管被转换 Linear 的 GEMM，其余全部维持 bf16 autocast 行为；
- 完整的 resume/导出卫生（§4.4–4.5）；
- 诚实的受益面声明与 G2 吞吐门槛（§3.1、§6）。

**不做（非目标）**：
- 不做 conv/attention 核心的 FP8（`torch._scaled_mm` 仅 matmul；ConvNeXt 卷积骨干不覆盖——这不是选择而是硬件/库边界，见 §3.1）；
- 不做 FP8 优化器状态 / master 权重量化（DeepSeek 也不做）；
- 不做 FP8 TensorRT 部署（部署侧另有 INT8 PTQ 管线；FP8 部署留待 TRT 路径单独立项）；
- 不动预训练引擎 `Engine.fit`（v1 边界：只接 `TaskEngine.fit_task` 的命令入口）。

---

## 3. 受益面分析（本册最重要的诚实章节）

### 3.1 覆盖矩阵：FP8 能加速什么

`torchao.float8` 只转换 `nn.Linear`（`torch._scaled_mm` 是 matmul 原语）。本仓各模块的覆盖情况：

| 模块 | 主体算子 | FP8 覆盖 | GEMM 规模（主力配方 @800px） |
|---|---|---|---|
| DINOv3 ConvNeXt-tiny 骨干 | Conv2d（depthwise + pointwise） | ❌ conv 不覆盖 | — |
| HybridEncoder CNN/FPN/PAN | Conv2d | ❌ | — |
| AIFI transformer（stride-32 层，25×25=625 token） | Linear（QKV/Proj/FFN 256→1024） | ✅ | 小（625×256×1024 级） |
| D-FINE decoder（3 层，~250–500 query） | Linear（self/cross-attn 投影、FFN 256→1024、分布头） | ✅（头排除） | 小–中 |
| **DINOv3/DINOv2 ViT 变体骨干** | Linear（QKV/FFN，seq ~2500–3481 @800px，width 384–1024） | ✅ | **中–大（本仓最大 GEMM 群）** |
| **deimv2 teacher 前向**（分册 23） | Linear 密集（DINOv3 ViT-S + decoder） | ✅ | 中–大；纯推理无 BN/dropout 顾虑 |
| SSL 投影头 / DeFE 密度头 | Linear | 策略性排除（§4.2） | — |

**结论（v1.1 实测后修订）**：v1.0 的本节按"GEMM 占比 × fp8 理论加速比"推算了 5–40% 的收益区间。**实测推翻了这一推算**：本仓全部相关形状（AIFI/decoder 的 256→1024、ViT-S 的 384→1536、QKV 256→256/384→1152，token 625–6400）上 fp8 前向+反向 GEMM 相对 bf16 为 **0.68–1.03×**——这些 GEMM 处于访存受限区间，fp8 节省的算力不是瓶颈，量化/缩放/padding 开销反而吃掉收益（§13.3 全表）。DeepSeek 的 FP8 收益来自 hidden 7k+、token 10万+ 的算力受限 GEMM，与 36M 检测模型的形状相差两个数量级。**本册据此否决实施**；本节表格保留作为"若形状规模扩大"的复核对照面。

### 3.2 显存

FP8 GEMM 的激活量化发生在算子内部（per-tile scale 广播），主激活仍以 bf16 存储于 autograd 图——**显存收益接近零**。24GB 卡的 micro-batch 不会因此放宽（与 §1 的预期联动：本册的价值在吞吐，不在容量）。

---

## 4. 详细设计

### 4.1 依赖与环境门（`pyproject.toml`）

```toml
[project.optional-dependencies]
# FP8 training via torchao Float8Linear (sm_89+; design doc 24).
# 0.18.0 实测与 torch 2.14.1+cu130 解析通过（doc 24 §13.1）。
fp8 = ["torchao>=0.18"]
```

运行时门（`_accelerator/float8.py`）：

```python
def float8_supported(device: torch.device) -> bool:
    # CUDA + compute capability >= 8.9（Ada/Hopper/Blackwell）; torchao 可导入
```

命令层在解析出 `fp8-mixed` 后立即检查：不满足 → `logger.warning` + 回退 `bf16-mixed`（镜像 `engine.py:696-697` 既有回退风格），**不 raise**（训练连续性优先，回退必须可见）。

### 4.2 转换策略（`_accelerator/float8.py` 新模块）

```python
#: 默认排除的模块名片段（DeepSeek"敏感算子"原则的 VTrain 实例化）。
FP8_EXCLUDE_FRAGMENTS = (
    "class_embed",        # decoder 分类头（fp32 island 源头；VisDrone 10 类
                          # out=10 不被 16 整除，实测 torch._scaled_mm 直接报错）
    "dec_score_head", "dec_bbox_head",   # D-FINE decoder 输出头
    "ssl_head",           # 分册 14 投影头（fp32 归一化语义）
    "defe",               # LiteDeFE 密度头（分册 13 fp32 island）
    "enc_score_head",     # 两段式 proposal 头
    "LQE",                # 分布质量估计器（logit 校准，对数值敏感）
    "query_feature",      # 特征发射通路（如后续出现独立投影）
)

def should_convert(module_name: str, module: nn.Module) -> bool:
    return (isinstance(module, nn.Linear)
            and not any(frag in module_name for frag in FP8_EXCLUDE_FRAGMENTS)
            and min(module.in_features, module.out_features) >= 256   # 小 GEMM 无收益
            and module.in_features % 16 == 0
            and module.out_features % 16 == 0)   # 实测硬约束（§13.2）

def convert_to_float8_training_(root: nn.Module) -> int:
    # torchao convert_to_float8_training(module_filter_fn=...) 的薄包装（§13.2：
    # v0.18 的实参名是 module_filter_fn，不是 filter_fn）；
    # config 固定 Float8LinearConfig(pad_inner_dim=True)——token 维整除约束
    # 的兜底（§4.4）。返回转换的 Linear 数（进日志）。幂等。
```

- **策略按名称片段排除**而非白名单：新增分支模块默认安全（除非恰好撞片段），对齐"基线兼容承诺"；
- **`min_dim >= 256` 尺寸下限**：256 以下的 Linear GEMM 太小，FP8 的 scale 开销必然倒挂（§13.3 实测）；
- **`% 16 == 0` 维度整除（实测硬约束，§13.2）**：`torch._scaled_mm` 前向要求 mat2（权重，形状 K×N）两个维度被 16 整除——`head = nn.Linear(256, 10)`（VisDrone 10 类正是此形状）实测直接 `RuntimeError: mat2 shape (256x10) must be divisible by 16`；
- **转换幂等性**：重复调用安全（§4.3 的调用点设计依赖这一点）。

### 4.3 转换点（命令层，`object_detection.py`）

```python
# :330 train_model = train_cls(resolved_model_args) 之后、:381 fit_task 之前：
if precision_resolved == "fp8-mixed":
    from autovision._accelerator.float8 import convert_to_float8_training_
    n_main = convert_to_float8_training_(train_model)     # 只转训练模型
    logger.info("FP8: converted %d Linear modules (precision=fp8-mixed).", n_main)
```

**D1（v1.1 实测后简化：只转训练模型，EMA 副本保持稠密）**：v1.0 曾要求主模型与 EMA 副本同步转换，依据是"键集合错位会破坏 `ema.update`"。**实测推翻了这个前提**：torchao 0.18 默认配置（tensorwise 动态缩放）的 `Float8Linear` **不新增任何 state_dict 键**——转换后的模块 state_dict 仍只有 `weight`/`bias`（fp32 master 权重原样保留，量化在 GEMM 内即时进行，无持久 scale/amax 缓冲，§13.2）。因此：

- EMA（`ema.py:29` 的 deepcopy 发生在转换之前）与转换后主模型**键集合天然一致**，`ema.update` 的按键更新照常工作；
- EMA 副本不转换反而是**更好的设计**：验证/导出走 EMA 权重（`train_model.py:376`），保持无 fp8 的标准数值——验证结果与 bf16 基线严格可比，导出图无需任何卫生处理（原 D3 的过滤需求消失）。

引擎 `fit_task` 内部的 DDP 包装（`engine.py:603`）发生在转换之后，无额外协调。transform/collate step hooks 与转换无交互。

### 4.4 精度表与 resume 契约

- `engine.py:696-698` 的 `autocast_dtype` 表**不改**：`fp8-mixed` 不在其中 → `autocast_dtype=None`？——不行，会丢 bf16 autocast。【设计】`fp8-mixed` 映射到 **bf16 autocast**（FP8 只在 Float8Linear 内部经 `scaled_mm` 生效，autocast 照旧覆盖其余全部算子）：

```python
autocast_dtype = {
    "bf16-mixed": torch.bfloat16,
    "16-mixed": torch.float16,
    # fp8-mixed: GEMM quantization lives inside torchao Float8Linear
    # (command-layer conversion); everything else keeps the bf16 contract.
    "fp8-mixed": torch.bfloat16,
}.get(precision)
```

  预训练引擎 `engine.py:289-295` 同步加映射（不支持时回退逻辑一并扩展），保持两引擎语义一致；
- **token 维整除约束（实测第二硬约束，§13.2）**：反向的权重梯度 GEMM 要求 mat1 尾维（= batch × token 数）被 16 整除。本仓形状实测：AIFI 在 625 token/图、batch 4 → 2500，`2500 % 16 ≠ 0` → **直接 RuntimeError**。对策：转换 config 固定 `Float8LinearConfig(pad_inner_dim=True)`（实测 2500/5000 token 均通过）——padding 有少量开销（§13.3 的 2500-token 形状已含此开销），换来对任意 batch/分辨率的鲁棒性；v1 不做运行时整除分派；
- **resume 契约（实测后简化）**：默认配置下转换不改变 state_dict 键集合（§13.2）——同精度与跨精度（bf16 checkpoint → fp8 训练）的 checkpoint 互载**均直接工作**，无需任何 shim；v1.0 的"跨精度 strict 报错"风险（R3）撤销。

### 4.5 导出与部署卫生（D3）

`engine.export_model`（`engine.py:733-749`）序列化 `train_model.get_export_state_dict()`。**实测后本节大幅简化**：默认配置下 `Float8Linear` 不新增任何 state_dict 键（§13.2），且 D1 已裁定 EMA 副本不转换——导出载荷就是与 bf16 基线逐键相同的 fp32 权重，**无需任何过滤**。v1.0 设计的 `is_float8_buffer_key` 谓词删除；改为一条**钉住前提的哨兵测试**：转换一个 tiny 模型，断言 `state_dict().keys() == 转换前 keys()`——若未来 torchao 配方引入持久缓冲键，此测试先行失败，提醒重新评估导出面（而非静默泄漏缓冲键到导出文件）。ONNX/TensorRT 导出不受影响：`_export` 链路从 init args 重建任务模型，图结构与数值面同 bf16 基线。

### 4.6 与其他分支的交互

| 分支 | 交互 |
|---|---|
| 分册 23 蒸馏 | v1.0 曾定位为"主要受益者"；**实测后撤销**——teacher 规模的 GEMM（384→1536 @1600–6400 token）实测 0.96–1.03×（§13.3），teacher 前向加速应走部署侧量化（§11 P2/P3）而非训练侧 fp8 |
| 分册 25 indexer | indexer MLP（hidden 64）低于 256 下限自然排除——稀疏选择逻辑必须保持全精度 |
| 分册 26 compile | torch.compile 与 Float8Linear 兼容（torchao 官方路径），但 v1 两者不同时启用（一次只引入一个变量，消融纪律） |
| SSL two_crop | 无交互（batch 维放大与模块转换正交） |
| NaN 守卫 | `engine.py:833-838` 照常覆盖（fp8 引入的新数值面被 Float8Linear 的在线 scale 兜底 + 总损失守卫兜底） |

---

## 5. 差异与取舍清单

| # | 取舍 | 决策 | 理由 |
|---|---|---|---|
| D1 | 转换点 | 命令层构建后，**只转训练模型**（v1.1 简化） | 实测 Float8Linear 默认配置无新增 state_dict 键（§13.2）→ EMA 错位前提不成立；EMA 保持稠密让验证/导出数值与 bf16 基线严格可比 |
| D2 | 小 Linear 不转换（<256） | 策略内建尺寸下限 | 小 GEMM 的 scale/quant 开销必然倒挂（§13.3 实测 0.86–0.96× 即在 ≥256 维形状上）；同时收窄数值面 |
| D3 | 导出面 | **零过滤 + 哨兵测试**（v1.1 简化） | 实测无新增缓冲键；哨兵测试钉住"键集合不变"前提，配方漂移时先行失败 |
| D4 | recipe 默认 tensorwise | rowwise 留作消融 | tensorwise 是 torchao 默认且最快路径；rowwise（per-row/channel scale）更接近 DeepSeek 的细粒度精神，但 sm89 上收益需实测——G2 门槛下两者各测一次 |
| D5 | 不支持时回退而非报错 | warning + bf16-mixed | 镜像 `engine.py:696-697` 既有回退语义；训练连续性优先；回退必须日志可见 |
| D6 | autocast 保持 bf16 | fp8-mixed ≠ 取消 autocast | FP8 是模块级 GEMM 策略；取消 autocast 会把 LayerNorm/softmax/残余 matmul 全推回 fp32，反而变慢 |

---

## 6. v0 门槛（v1.1：G2 已被实测否定，全链终止）

| # | 内容 | 工具 | 判据 | v1.1 状态 |
|---|---|---|---|---|
| G0 | **覆盖面量化**：主力配方单步中 Linear 密集相位占比 | `scripts/profile_step_phases.py` | < 25% → 降级 | 未单独跑——**被 G2 微基准先行否定**（即使占比 100%，形状级收益仍 ≤1.03×，见 §13.3） |
| G1 | **数值漂移**：fp8 vs bf16 各 100 步 | `scripts/smoke_fp8_training.py` | 收敛、无 NaN、漂移 < 20% | **冒烟配置勘误（v1.1）**：不能用 vittest（hidden 64，全部 Linear 低于 256 下限 → 零转换，测了个寂寞）；须用 convnext-tiny @320（AIFI/decoder 的 256/1024 Linears 合格，§13.2 策略实测通过） |
| G2 | **吞吐**：fp8 vs bf16 步时 | 微基准 + 端到端 | 端到端 ≥ +10% | ❌ **实测否定**：形状级 0.68–1.03×（§13.3），端到端必为负收益。全链终止 |
| G3 | **精度底线**：10-epoch twin | — | mAP ≥ −0.3 AP | 不再适用（G2 已否决） |

---

## 7. 实现清单

| 文件 | 改动 | ~行 |
|---|---|---|
| `pyproject.toml` | `fp8` 可选依赖组（`torchao>=0.18`） | ~4 |
| `_accelerator/float8.py`（新） | 支持检测 / 排除策略（片段 + ≥256 + %16）/ 转换 helper（`module_filter_fn` + `pad_inner_dim=True`） | ~110 |
| `_commands/train_task/object_detection.py` | precision 解析扩展 + 转换点（只转训练模型，EMA 不转） | ~15 |
| `_engine/engine.py` | 两处 autocast 表加 `fp8-mixed` 映射 + 回退扩展 | ~12 |
| `scripts/smoke_fp8_training.py` | G1/G2 工具 | ~80 |
| `tests/test_fp8_training.py` | §8 | ~200 |

（v1.1 注：本清单随 §13 否决归档——仅在 §13.5 复核条件触发后执行。）

---

## 8. 测试与验收

1. **off-by-default 契约**：precision 不含 `fp8-mixed` 时命令层零转换调用（mock 断言）、训练路径与基线逐比特一致；
2. **策略单测**：`should_convert` 对 `class_embed`/`dec_score_head`/`ssl_head`/`defe`/`LQE` 全排除；256 以下与不满足 %16 的 Linear 排除（含 `Linear(256, 10)` 类头形状，§13.2）；普通 encoder/decoder Linear 转换；幂等（二次调用零增量）；
3. **键集合不变哨兵（v1.1 重写）**：tiny 模型转换前后 `state_dict().keys()` 逐键相等（torchao 0.18 实测性质，§13.2）——配方升级引入持久缓冲键时此测试先行失败；转换后主模型与未转换 EMA 副本的键集合相等，`ema.update` 跑通且数值有限（D1 依据）；
4. **autocast 映射**：`fp8-mixed` → bf16 autocast context 生效（捕获 autocast dtype 断言）；不支持设备回退 bf16-mixed 且有 warning；
5. **导出面（v1.1 简化）**：转换模型的 `get_export_state_dict` 与未转换基线逐键相同（EMA 稠密 + 无缓冲键），round-trip strict 加载直接通过；
6. **resume 契约（v1.1 简化）**：同精度与跨精度（bf16 checkpoint → fp8 训练）checkpoint 互载均走通（键集合不变，§13.2）；
7. **数值冒烟**：convnext-tiny @320 配置（非 vittest——低于 256 下限零转换，§13.4）转换后 20 步 loss 有限、反向梯度非 None、clip_gradients 正常；
8. **div-16 鲁棒性**：token 维不整除的 batch（如 AIFI 形状 625×4）在 `pad_inner_dim=True` 下前向反向通过（§13.2 实测语义落为回归）。

---

## 9. 使用配方（⚠️ v1.1：被 §13 实测否决，勿按此启用）

```python
# 本节保留仅为文档完整性；当前任何配方都不应启用 fp8-mixed（§13.3）。
train_object_detection(..., precision="fp8-mixed", ...)

# ViT 变体（v1.0 预期的主要受益者；实测 QKV 类形状仅 1.03×，见 §13.3——不要启用）
# visdrone_ltdetr_1280_vits16_baseline_coco.py 变体 + precision="fp8-mixed"

# 分册 23 联动：teacher 前向 fp8（与 student 精度解耦，独立开关语义记入分册 23 §10 R6）
```

---

## 10. 风险表

| 风险 | 等级 | 缓解 |
|---|---|---|
| R1 主力配方（ConvNeXt conv 为主）收益不达标 | **已证实（§13.3）** | 全部形状实测 0.68–1.03×；本册已否决，此风险即否决依据 |
| R2 torchao API/键名漂移（版本耦合） | 中（若重启） | 实测 API 面：`module_filter_fn` 实参名、`Float8LinearConfig(pad_inner_dim=)`、无新增缓冲键（§13.2）——重启实施时以哨兵测试钉住 |
| R3 跨精度 resume 加载失败 | **已撤销** | 实测键集合不变（§13.2），跨精度互载直接工作 |
| R4 fp8 数值噪声伤小目标回归（FGL/DDF 分布头对精度敏感） | 低（若重启） | 分布头已在排除名单（fp32 island）；div-16 + pad_inner_dim 已实测可行 |
| R5 EMA/导出键错位 | **已撤销** | 键集合不变 + D1 只转训练模型（§4.3） |
| R6 4090D 之外的设备面（旧卡 <sm89、NPU） | 低 | `float8_supported` 门 + D5 回退；NPU 本就是软件侧支持（`engine.py:168-196` 注释） |

---

## 11. 后续路线（v1.1：实施线终止，仅留复核触发条件）

| # | 内容 | 备注 |
|---|---|---|
| P0 | ~~G0/G1/G2/G3~~ **已终止**——§13.3 实测否定吞吐前提 | 无实施工时投入 |
| P1 | **复核条件**（任一满足时重启本册，从 §13 的实测脚本续起）：① 引入 hidden ≥ 2048 的骨干变体（ViT-L/B @1280px 级配方）；② 更换 fp8 开销更低的下一代卡（sm1xx+）；③ torchao 的 rowwise/低开销 recipe 在 sm89 上有数量级改进 | 触发后先重跑 §13.3 微基准，形状级 ≥ 1.15× 才进端到端 |
| P2 | 分册 23 teacher 前向的**部署侧**量化（TRT FP8/INT8，走 `_export` 链路） | 与训练侧 fp8 无关，不受本否决影响 |
| P3 | FP8 TensorRT 部署路径（`_export/quantization.py` 扩展） | 独立立项 |

---

## 12. 参考索引

**DeepSeek 侧（【DS】事实来源）**：

| 内容 | 位置 |
|---|---|
| FP8 混合精度框架（细粒度量化、在线 scale、敏感算子高精度、master 权重不量化） | DeepSeek-V3 Technical Report（arXiv:2412.19437）§3.3 |
| 官方 FP8 GEMM kernel（JIT 细粒度 scaling 参考） | `github.com/deepseek-ai/DeepGEMM` |
| FP8 训练栈开源 | `github.com/deepseek-ai/DeepSeek-V3`（pt/转换/推理栈说明） |

**VTrain 受影响面**（`src/autovision/`）：

| 内容 | 位置 |
|---|---|
| 命令层 precision 解析 | `_commands/train_task/object_detection.py:334-340` |
| 引擎 autocast 表（×2 处） | `_engine/engine.py:289-295`（预训练）、`:696-698`（task） |
| autocast 应用点 / 梯度累积 / NaN 守卫 | `_engine/engine.py:807-846, 833-838` |
| EMA deepcopy 时序（D1 依据） | `_task_models/object_detection_components/ema.py:23-58` |
| fp32 island 惯例（设计同构性依据） | `dfine_decoder.py`（box/分布头）、`ssl_contrastive.py`（fp32 归一化）、分册 13/14/15 各册声明 |
| 导出格式（D3 卫生点） | `_engine/engine.py:733-749` |
| 部署端 INT8 PTQ（非目标参照物） | `_export/quantization.py` |
| 相位 profile 工具（G0） | `scripts/profile_step_phases.py:91-163` |

---

## 13. 实测记录（2026-10-12，v1.1 否决依据）

环境：RTX 4090D（sm_89，24564 MiB）、torch 2.14.1+cu130、Python 3.10；torchao 经 `uv pip install` 临时安装、测毕即卸载（工作区环境无残留）。

### 13.1 依赖解析

`uv pip install --dry-run torchao`（对项目 venv）→ `Resolved 1 package … + torchao==0.18.0`，无 torch 重装。**v1.0 所写 `torchao>=0.28` 有误，已钉正为 `>=0.18`。**

### 13.2 API 面与硬约束（全部实测）

| 事实 | 证据 |
|---|---|
| 转换入口：`convert_to_float8_training(module, *, module_filter_fn, config)` | `inspect.signature` 实测——v1.0 草案写的 `filter_fn` 不存在 |
| **权重维整除**：前向 `torch._scaled_mm` 要求权重两维被 16 整除 | `nn.Linear(256, 10)`（VisDrone 类头形状）转换后前向 → `RuntimeError: mat2 shape (256x10) must be divisible by 16` |
| **token 维整除**：反向权重梯度 GEMM 要求 mat1 尾维（batch×token）被 16 整除 | 625 token/图 × batch 4 = 2500 → `Expected trailing dimension of mat1 to be divisible by 16 but got mat1 shape: (256x2500)`（转置后尾维 2500） |
| `Float8LinearConfig(pad_inner_dim=True)` 兜底 token 维 | 2500/5000 token + pad → 前向反向 3 步训练通过；不可 pad 的权重维仍须被策略排除 |
| **state_dict 键集合不变**：默认配置（tensorwise 动态缩放）转换后 state_dict 仍只有 `weight`/`bias`（fp32） | tiny 模型转换 + 前向反向后逐键打印实测；无 scale/amax 持久缓冲 |
| 策略过滤（片段排除 + ≥256 + %16）在真实模块树上工作 | filter 转换 tiny（body 两条 Linear 转换，head/small 保留 `nn.Linear`）+ 3 步 bf16-autocast 训练收敛 |

### 13.3 GEMM 微基准（前向+反向，fp32 master 权重，bf16 autocast；3 轮取最优，µs/step）

| 形状（in→out @ tokens） | 对应模块 | bf16 | fp8（pad_inner_dim=True） | 加速比 |
|---|---|---|---|---|
| 256→1024 @ 625 | AIFI FFN（单图） | 151.5 | 175.9 | **0.86×** |
| 256→1024 @ 2500 | AIFI FFN（batch 4） | 281.2 | 413.2 | **0.68×** |
| 256→256 @ 625 | AIFI/decoder QKV | 442.3 | 458.4 | 0.96× |
| 384→1536 @ 1600 | deimv2-dinov3-s teacher FFN @640px | 299.1 | 302.1 | 0.99× |
| 384→1536 @ 6400 | 同上 @1280px | 428.7 | 446.6 | 0.96× |
| 384→1152 @ 2500 | ViT-S QKV @800px | 490.1 | 477.2 | 1.03× |

**判读**：全部 12 个测量点落在 0.68–1.03×，无一达到 G2 的 +10% 门槛，多数倒退。机理：这些 GEMM 的算术强度远低于 4090D 的算力拐点（2500-token × 256→1024 实测仅 ~4.7 TFLOPS bf16，设备峰值 ~165），属访存/launch 受限——fp8 节省的算力不构成收益，量化/缩放/padding 反而是净开销。**DeepSeek-V3 的 FP8 收益前提（hidden 7k+、token 10 万+ 的算力受限 GEMM）在本项目所有当前与近预期形状上都不成立。**

### 13.4 G1 冒烟配置勘误（保留给重启场景）

v1.0 的 G1 冒烟写"vittest @224 或 convnext-tiny @320"：vittest（hidden 64）在 ≥256 下限下**零转换**，冒烟测不到任何 fp8 路径——勘误为 convnext-tiny @320（AIFI 256→1024、decoder 256 维 Linears 均合格且被 16 整除）。

### 13.5 复核条件（重启本册的唯一入口）

① hidden ≥ 2048 的骨干变体进入配方（ViT-L/B @1280px 级，GEMM 进入算力受限区）；② 更换 fp8 开销更低的卡；③ torchao recipe 在 sm89 上有数量级改进。触发后从 §13.3 的微基准脚本续起，形状级 ≥ 1.15× 才允许进入端到端门槛。

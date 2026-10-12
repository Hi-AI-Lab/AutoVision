# VTrain 实现蓝图 · 分册 24：FP8 混合精度训练（`precision="fp8-mixed"` 档位）

| 文档属性 | 内容 |
|---|---|
| 版本 | v1.0（设计评审稿，未实施） |
| 状态 | 设计完成；G2 吞吐门槛的结果决定是否实施（§6 诚实预期：本仓主力配方受益面有限，见 §3.1） |
| 上游思想 | DeepSeek-V3（arXiv:2412.19437）首次大规模 FP8 训练：①细粒度 tile-wise（1×128 激活 / 128×128 权重）在线量化，拒绝 delayed scaling；②**敏感算子保持高精度**（embedding 输出、attention 核心、gating、归一化、首尾层）；③只对 Linear GEMM 量化，存储/优化器 master 权重不变。【DS】 |
| 配套分册 | DeepSeek 迁移四联册：本册（FP8）+ 分册 23（蒸馏）/ 25（token indexer）/ 26（步工程优化） |
| 参考实现 | `torchao.float8`（`convert_to_float8_training`，官方 PyTorch FP8 训练路径）；**注意：torchao 尚未安装**（本仓环境 torch 2.14.1+cu130，需新增可选依赖） |
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

**结论（决定 G2 门槛与预期管理）**：现役 VisDrone 主力配方（ConvNeXt-tiny + hidden 256）的 Linear GEMM 占步时比例有限（`profile_step_phases.py` 的相位分解可直接量化——**G0 门槛先测 decoder+encoder 相位占比**），预期全步收益 **5–15%**；ViT 变体（`visdrone_ltdetr_1280_vits16_baseline_coco.py` 一族）与 deimv2 训练/teacher 推理受益更大（20–40% 量级）。【设计】因此本册定位为**低风险档位扩展 + teacher 前向加速器（分册 23 联动）**，不是无条件默认值；G2 不达标就只保留给 ViT/deimv2 配方使用，如实记录。

### 3.2 显存

FP8 GEMM 的激活量化发生在算子内部（per-tile scale 广播），主激活仍以 bf16 存储于 autograd 图——**显存收益接近零**。24GB 卡的 micro-batch 不会因此放宽（与 §1 的预期联动：本册的价值在吞吐，不在容量）。

---

## 4. 详细设计

### 4.1 依赖与环境门（`pyproject.toml`）

```toml
[project.optional-dependencies]
# FP8 training via torchao Float8Linear (sm_89+; design doc 24).
fp8 = ["torchao>=0.28"]
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
    "class_embed",        # decoder 分类头（fp32 island 源头）
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
            and min(module.in_features, module.out_features) >= 256)   # 小 GEMM 无收益，跳过降低数值面

def convert_to_float8_training_(root: nn.Module, *, config: Float8LinearConfig | None = None) -> int:
    # torchao convert_to_float8_training(filter_fn=...) 的薄包装；
    # 返回转换的 Linear 数（进日志）。幂等：已是 Float8Linear 的模块跳过。
    # recipe：v1 用 tensorwise（默认最快）；rowwise 作为消融开关。
```

- **策略按名称片段排除**而非白名单：新增分支模块默认安全（除非恰好撞片段），对齐"基线兼容承诺"；
- **`min_dim >= 256` 尺寸下限**：256 以下的 Linear（各处小投影、matcher 无参数、二维 embedding）GEMM 太小，FP8 的 scale 开销可能倒挂——跳过即同时收窄数值面（D2）；
- **转换幂等性**：EMA 副本与主模型都要转换（§4.3），同一函数重复调用安全。

### 4.3 转换点（命令层，`object_detection.py`）

```python
# :330 train_model = train_cls(resolved_model_args) 之后、:381 fit_task 之前：
if precision_resolved == "fp8-mixed":
    from autovision._accelerator.float8 import convert_to_float8_training_
    n_main = convert_to_float8_training_(train_model)
    if getattr(train_model, "ema_model", None) is not None:
        convert_to_float8_training_(train_model.ema_model.model)   # D1：EMA 副本同步转换
    logger.info("FP8: converted %d Linear modules (precision=fp8-mixed).", n_main)
```

**D1（为什么在命令层且必须包含 EMA 副本）**：`ModelEMA.__init__` 在 train wrapper `__init__` 里 `copy.deepcopy(model)`（`ema.py:29`）——若只转换主模型，EMA 的 state_dict 键集合与主模型错位，`ema.update` 的按键 `model_state[key]`（`ema.py:48-58`）直接 KeyError 或静默漏更新。命令层在 deepcopy 之后对**两个副本都转换**，键集合保持一致（torchao 转换对同一结构确定性同名）。引擎 `fit_task` 内部的 DDP 包装（`engine.py:603`）发生在转换之后，无额外协调。transform/collate step hooks 与转换无交互。

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
- **resume 契约**：`fp8-mixed` 的 checkpoint 含 Float8Linear 的 scale/amax 缓冲键；**同精度 resume 键集合一致直接工作**（命令层转换先于 `engine.py:651-688` 的加载）。从 bf16 checkpoint 起训（`checkpoint=` 初始化路径）→ 缺 fp8 缓冲键 → strict 加载报错；【设计】在 `convert_to_float8_training_` 完成后于命令层打一条显式日志声明该约束；v1 不做跨精度加载 shim（使用面窄，报错即正确行为，R3）。

### 4.5 导出与部署卫生（D3）

`engine.export_model`（`engine.py:733-749`）序列化 `train_model.get_export_state_dict()`（= EMA 权重，含 `ema_model.` 前缀的 fp8 缓冲键）。推理加载器从 `model_init_args` 重建**未转换**的任务模型 → fp8 缓冲键成为 unexpected keys。【设计】导出前过滤：

```python
# engine.export_model 内，构造 payload 前：
state = {k: v for k, v in train_model.get_export_state_dict().items()
         if not is_float8_buffer_key(k)}
```

`is_float8_buffer_key` 由 `_accelerator/float8.py` 提供（识别 torchao Float8Linear 缓冲键模式；实现期以实际 torchao 版本的键名集为准并加单测钉住——**这是 torchao 版本耦合面，升 torchao 时此测试是哨兵**）。ONNX/TensorRT 导出不受影响：`_export` 链路从 init args 重建任务模型（从未见过 fp8 转换），图结构与数值面同 bf16 基线。

### 4.6 与其他分支的交互

| 分支 | 交互 |
|---|---|
| 分册 23 蒸馏 | **主要受益者**：teacher 前向是 Linear 密集的纯推理（deimv2/DINOv3 ViT），转换收益最确定且无训练数值风险；teacher 副本单独转换即可（`distill` 路径在 wrapper 中持有 teacher，命令层追加一行） |
| 分册 25 indexer | indexer MLP（hidden 64）低于 256 下限自然排除——稀疏选择逻辑必须保持全精度 |
| 分册 26 compile | torch.compile 与 Float8Linear 兼容（torchao 官方路径），但 v1 两者不同时启用（一次只引入一个变量，消融纪律） |
| SSL two_crop | 无交互（batch 维放大与模块转换正交） |
| NaN 守卫 | `engine.py:833-838` 照常覆盖（fp8 引入的新数值面被 Float8Linear 的在线 scale 兜底 + 总损失守卫兜底） |

---

## 5. 差异与取舍清单

| # | 取舍 | 决策 | 理由 |
|---|---|---|---|
| D1 | 转换点 | 命令层构建后（含 EMA 副本），而非 wrapper 内/engine 内 | EMA deepcopy 时序（§4.3）：wrapper `__init__` 早于一切外部介入点；命令层是唯一能同时看到主模型与 EMA 的位置；引擎保持精度表纯净 |
| D2 | 小 Linear 不转换（<256） | 策略内建尺寸下限 | 小 GEMM 的 scale/quant 开销可能倒挂；同时收窄数值面；DeepSeek 对 GEMM 同样有粒度门槛精神 |
| D3 | 导出面 | 过滤 fp8 缓冲键而非保留 | 推理重建路径无 fp8 模块，保留键 = unexpected keys 报错；`_export` 数值对拍门禁不受影响 |
| D4 | recipe 默认 tensorwise | rowwise 留作消融 | tensorwise 是 torchao 默认且最快路径；rowwise（per-row/channel scale）更接近 DeepSeek 的细粒度精神，但 sm89 上收益需实测——G2 门槛下两者各测一次 |
| D5 | 不支持时回退而非报错 | warning + bf16-mixed | 镜像 `engine.py:696-697` 既有回退语义；训练连续性优先；回退必须日志可见 |
| D6 | autocast 保持 bf16 | fp8-mixed ≠ 取消 autocast | FP8 是模块级 GEMM 策略；取消 autocast 会把 LayerNorm/softmax/残余 matmul 全推回 fp32，反而变慢 |

---

## 6. v0 门槛（实施前必过）

| # | 内容 | 工具 | 判据 |
|---|---|---|---|
| G0 | **覆盖面量化**：主力配方（Dome twin）单步相位分解中 decoder+encoder（Linear 密集相位）占比 | `scripts/profile_step_phases.py`（现成） | Linear 密集相位 < 25% 步时 → 主力配方预期收益 <10%，本册降级为"ViT/deimv2/teacher 专用档"，主力配方不推荐开启（如实记录） |
| G1 | **数值漂移**：tiny 配方（vittest @224 或 convnext-tiny @320）fp8 vs bf16 各 100 步 | 新增 `scripts/smoke_fp8_training.py`（~60 行） | 双臂 loss 曲线均收敛、无 NaN；终点相对漂移 < 20%（fp8 噪声预算，不做逐比特） |
| G2 | **吞吐**：主力配方（或 G0 判定的受益配方）fp8 vs bf16 步时 | G1 脚本放大版 / profile_step_phases | 端到端步时 ≥ +10%（低于 → D4 换 rowwise 复测一次；仍低于 → 按配方归档，不全局推） |
| G3 | **精度底线**：10-epoch VisDrone pilot fp8 vs bf16 twin | `visdrone_ltdetr_dome_smoke.py` 变体 | mAP 差 ≥ −0.3 AP；超出 → 该配方禁用 fp8 并记录 |

---

## 7. 实现清单

| 文件 | 改动 | ~行 |
|---|---|---|
| `pyproject.toml` | `fp8` 可选依赖组 | ~4 |
| `_accelerator/float8.py`（新） | 支持检测 / 排除策略 / 转换 helper / 缓冲键谓词 | ~120 |
| `_commands/train_task/object_detection.py` | precision 解析扩展 + 转换点 + EMA 副本 + resume 契约日志 | ~25 |
| `_engine/engine.py` | 两处 autocast 表加 `fp8-mixed` 映射 + 回退扩展 + `export_model` 键过滤 | ~20 |
| `scripts/smoke_fp8_training.py` | G1/G2 工具 | ~80 |
| `tests/test_fp8_training.py` | §8 | ~200 |

总规模 ~1.5 天（不含 G2/G3 的 GPU 时间；torchao 安装与 sm89 验证另计半天）。

---

## 8. 测试与验收

1. **off-by-default 契约**：precision 不含 `fp8-mixed` 时命令层零转换调用（mock 断言）、训练路径与基线逐比特一致；
2. **策略单测**：`should_convert` 对 `class_embed`/`dec_score_head`/`ssl_head`/`defe`/`LQE` 全排除；256 以下 Linear 排除；普通 encoder/decoder Linear 转换；幂等（二次调用零增量）；
3. **EMA 键集合一致**：转换后主模型与 EMA 副本的 state_dict 键集合相等；EMA update 跑通且数值有限；
4. **autocast 映射**：`fp8-mixed` → bf16 autocast context 生效（捕获 autocast dtype 断言）；不支持设备回退 bf16-mixed 且有 warning；
5. **导出卫生**：`export_model` 产物无 fp8 缓冲键；`is_float8_buffer_key` 对 torchao 实际键名的哨兵测试（真转换一个 tiny 模型取键名）；
6. **resume 契约**：同精度 checkpoint→resume 走通（engine_state 完整）；bf16 checkpoint → fp8 模型给出清晰报错（断言错误信息含 fp8 提示）；
7. **数值冒烟**：tiny 模型转换后 20 步 loss 有限、反向梯度非 None、clip_gradients 正常；
8. **导出链路**：转换后的训练模型 `get_export_state_dict` 过滤后可被推理加载器 strict 加载（round-trip）。

---

## 9. 使用配方

```python
# 主力 ConvNeXt 配方：先过 G0，占比达标才开
train_object_detection(..., precision="fp8-mixed", ...)

# ViT 变体（预期主要受益者）
# visdrone_ltdetr_1280_vits16_baseline_coco.py 变体 + precision="fp8-mixed"

# 分册 23 联动：teacher 前向 fp8（与 student 精度解耦，独立开关语义记入分册 23 §10 R6）
```

---

## 10. 风险表

| 风险 | 等级 | 缓解 |
|---|---|---|
| R1 主力配方（ConvNeXt conv 为主）收益不达标 | **高（结构性）** | G0 前置量化覆盖面；不达标则档位仅限 ViT/deimv2 配方——本册从不承诺主力配方收益 |
| R2 torchao API/键名漂移（版本耦合） | 中 | fp8 锁版本下限；`is_float8_buffer_key` 哨兵测试；`_accelerator/float8.py` 单点封装 |
| R3 跨精度 resume 加载失败 | 低 | v1 声明为约束（同精度 resume）；报错信息指向修复；使用面窄 |
| R4 fp8 数值噪声伤小目标回归（FGL/DDF 分布头对精度敏感） | 中 | 分布头已在排除名单（fp32 island）；G3 mAP 底线门；rowwise 消融作为更细粒度兜底 |
| R5 EMA/导出键错位 | 中 | D1 双副本转换 + 测试 3/5/8 钉住 |
| R6 4090D 之外的设备面（旧卡 <sm89、NPU） | 低 | `float8_supported` 门 + D5 回退；NPU 本就是软件侧支持（`engine.py:168-196` 注释） |

---

## 11. 后续路线

| # | 内容 | 备注 |
|---|---|---|
| P0 | G0 相位占比实测（半天，现成工具） | 决定主力配方是否进入 G1–G3 |
| P1 | torchao 安装 + §7 实施 + G1/G2 | 1.5–2 天 |
| P2 | G3 twin pilot；结果回填本册 | GPU 时间 |
| P3 | 分册 23 teacher 前向 fp8（蒸馏落地后） | 联动项 |
| P4 | rowwise recipe 消融；torch.compile 叠加实验（分册 26 W3 之后） | 视 P2 |
| P5 | FP8 TensorRT 部署路径（`_export/quantization.py` 扩展） | 独立立项，不在本册 |

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

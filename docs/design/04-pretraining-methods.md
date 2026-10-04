# VTrain 实现蓝图 · 分册 04：预训练方法层

| 文档属性 | 内容 |
|---|---|
| 版本 | v3.1（实现蓝图拆分版，内容与总览同步维护） |
| 总览 | [framework_design.md](../framework_design.md) |
| 全局章节 | §7 |

> **标注约定**：【基线】lightly-train 0.17.0 已验证事实（出处 `文件:行`，路径相对 `src/lightly_train/`）；【设计】VTrain 决策；【差异】有意偏离基线的决策（附 ADR 编号）。正文中的 § 编号为**全局章节编号**，章节与分册的映射见总览。

---

## 7. 预训练方法层

### 7.1 方法契约与基类【基线】

`_methods/method.py`（`class Method(LightningModule)`）——VTrain 基类照此设计，**修正 v2.0 的一处错误**：

```python
@dataclass
class TrainingStepResult:
    loss: Tensor                                # 单一已求和标量
    log_dict: Mapping[str, Any] | None = None   # 分量只进日志

class Method(LightningModule):
    def __init__(self, method_args, optimizer_args, embedding_model,
                 global_batch_size: int, num_input_channels: int): ...
    # 子类必须实现的四个钩子：
    method_args_cls() -> type[MethodArgs]           # static
    transform_cls() -> type[MethodTransform]        # static
    trainable_modules() -> TrainableModules         # (modules, modules_no_weight_decay)
    training_step_impl(batch, batch_idx) -> TrainingStepResult
    # 基类具体实现：
    optimizer_args_cls(optim_type)     # "auto" → AdamW；再查优化器注册表
    training_step(batch, batch_idx)    # 调 impl → log("train_loss", sync_dist, batch_size=len(views[0]))
                                       # → log_dict → global_step==0 时落示例增强图
    configure_optimizers()             # 见下方精确公式
    on_train_batch_start/end           # profiling/data_time、batch_time 计时
```

> **修正（对照 v2.0 §7.1）**：基类**不做**「loss 分量字典求和」——`training_step_impl` 返回的 `loss` 已由方法自行求和；多分量方法在 impl 内部加权和（如 DINOv2 的 `dino_global*2/n + dino_local/n + ibot + 0.1*koleo`，`_methods/dinov2/dinov2.py:335-387`）。

`configure_optimizers` 精确数学（`method.py:90-122`，VTrain 照抄）：

```python
lr_scale = global_batch_size / method_args.reference_batch_size     # MethodArgs 默认 256
if lr_scale_method == "sqrt": lr_scale = math.sqrt(lr_scale)        # 蒸馏用 sqrt@1536
optimizer  = optim_args.get_optimizer(param_groups, lr_scale)       # args 内部再 lr *= lr_scale
warmup_epochs = min(10, max_epochs / 10)
warmup_steps  = min(total_steps, total_steps/max_epochs * warmup_epochs)
scheduler  = CosineWarmupScheduler(optimizer, warmup_epochs=warmup_steps,
                                   max_epochs=total_steps)          # interval="step"
```

参数组构建（`_optim/optimizer_helpers.py:56-77`）：`[{"name":"params"}, {"name":"params_no_weight_decay","weight_decay":0}]`。**no-weight-decay 判定清单**（`get_weight_decay_parameters`，`:83-203`，VTrain 照抄并加单测）：norm 层（含类名含 "Norm"）、Embedding/EmbeddingBag、`ndim<=1`、名字含 `bias/bias_v/bias_k`、token 类（`cls_token/mask_token/storage_token/register_token`）、`queries/query`、位置编码族（`pos_embed/position_embedding/.../bias_table`）、logit scale 族（`logit_scale/temperature/tau`）、layer scale 族（`layer_scale/layerscale/res_scale/rezero/gamma`）。

梯度裁剪：基类不设默认，方法各自实现——DINO 3.0、DINOv2 可配（默认 3.0）、蒸馏三版 1.0（均 `algorithm="norm"`）。

### 7.2 方法注册表与清点【基线】

注册即 dict：`densecl`（隐藏）、`dino`、`dinov2`、`distillationv1`、`distillationv2`、`distillationv3`，别名 `distillation→v3`。`dinov31`（PaKA）与 `detcon` 代码在树但**未注册**（0.16.4 已移除 dinov31 的注册）。

【设计】VTrain 注册表为 `dino`、`dinov2`、`dinov3`、`simclr`、`densecl`、`detcons`、`detconb`、`distillationv2`、`distillationv3`（+别名 `distillation→v3`），全部可见（`include_in_list=True`），无 `distillationv1`。蒸馏系列**沿用基线版本号命名**：v2/v3 是 §7.3 配方谱系名（蒸馏方法的演进代际，等价于上游方法名而非实现迭代序号），目录名与基线逐条镜像以维持分册 11 的对照映射，金标准快照（`tests/golden/distillationv3_losses.json`）同锚定。**后继规则**：新增代际（v4）在验收通过后**取代**而非并列旧代际——旧代际保留注册名一个次版本弃用窗口后随目录一并移除，防止版本目录堆积。机制命名方法（dino/simclr/densecl/detcon 等）一律取上游方法名，不适用本条。

### 7.3 蒸馏方法 DistillationV3（P0 对标实现）【基线】

以下为逐条核验的完整规格，VTrain P0 等价实现并保持数值兼容：

```
结构  学生 = EmbeddingModel(骨干+embed头)；教师 = get_teacher(任意 ModelWrapper，默认 "dinov3/vitb16")，
      冻结+eval+no_grad，make_teacher() 钩子，checkpoint 保存时剥离 teacher_embedding_model.* 键
      （on_save_checkpoint 过滤 / on_load_checkpoint 回填，:416-433）
      两个线性投影头: Linear(embed_dim → teacher_dim)，权重 trunc_normal_(std=0.02)
      教师 buffer 队列: register_buffer("teacher_queue", zeros(queue_size, teacher_dim))，per-rank 无 all-gather

变换  单视图 224²；RandomResizedCrop min_scale=0.14；flip h0.5；
      ColorJitter(p0.8, s0.5: b.8/c.8/s.4/h.2)；gray 0.2；GaussianBlur p1.0 sigma(0.0,0.1)；ImageNet 归一化
      mixup 不在变换里，在 training_step_impl 内: λ~U(0,1)，x' = λx + (1-λ)x[randperm]

损失  KLDivLoss(reduction="batchmean")；τ_global = τ_local = 0.07；loss_local_weight = 1.0；无 centering
      global:  s_sim = einsum("b d, c d -> b c", student_g, queue)   # queue 含当前 batch(先更新后算损失)
               t_sim = einsum("b d, c d -> b c", teacher_g, queue)
               L_g = KL(log_softmax(s_sim/τ), softmax(t_sim/τ))
      local:   t_self = einsum("b m d, b n d -> b m n", teacher_p, teacher_p).flatten(0,1)
               s_self = 同理(student_p)；L_l = KL(log_softmax(s_self/τ), softmax(t_self/τ))
      特征均先 L2 归一化；队列存的就是归一化后向量(无 denormalize)
      总损失 = L_g + 1.0 * L_l；日志键 train_loss/global_loss、train_loss/local_loss(.item() 浮点)

队列  auto 桶表（get_bucket_value，首阈值即用，_scaling.py:47-64）:
      数据集 <50→16, <100→32, <250→64, <500→128, <1k→256, <2k→512, <4k→1024,
             <10k→2048, <20k→4096, ≥20k→8192；显式 queue ≥ 数据集大小 → ValueError
      更新: 右移 B 位、头部写入当前 batch 教师特征（:275-290）

优化  optimizer_args_cls: "auto"/AdamW → DistillationV3AdamWArgs(lr=5e-4, wd="auto")；
      LARS → DistillationV3LARSArgs(lr=1.8, momentum=0.9, wd=1e-6, trust_coef=1e-3)
      wd="auto" 按架构: transformer/hybrid 或 conv+layernorm → 0.04；conv+batchnorm → 1e-6；
      未知组合抛错（架构信息来自 ArchitectureInfoGettable，兜底类名启发式 _is_probably_conv）
      lr 缩放 sqrt @ reference_batch_size=1536；梯度裁剪范数 1.0
```

分期：v0 全局项 → v1 补 patch KL/mixup/队列 → v2 域内教师（`method_args={"teacher": "runs/<exp>/exported/best.pt"}`，基线已支持任意 wrapper/路径加载，`:47-80`）。

### 7.4 其他内置方法速览【基线】

| 方法 | 要点 |
|---|---|
| SimCLR | LightlySSL `NTXentLoss(τ=0.1)` + 2 层投影头；双视图共享前向后 chunk；SGD lr0.3 |
| DINO | Siamese+EMA(动量余弦 0.99→1.0)、DINOLoss 带中心化、教师温度预热 37500 步、输出维度按数据集分桶 1024→65536、SGD lr0.03、冻结最后层 1250 步 |
| DINOv2 | DINO+iBOT+KoLeo 全配方；mask 0.1–0.5；层级 lr decay 0.9、patch_embed ×0.2；wd 余弦 0.04→0.4；`default_steps=125_000`；步数低于阈值告警 |
| Distillation v1 | v3 的前身：仅全局队列 KL、单头、教师限 DINOv2/v3 wrapper、auto 优化器为 LARS |
| Distillation v2 | 中间层 MSE 蒸馏（`get_intermediate_layers(n=2)` 拼接），无队列无归一化，LARS lr 9.0 |
| DenseCL | MoCo-v2 型（隐藏方法），全局 NTXent + 局部 DenseCLLoss 各 0.5，memory bank 桶 16→65536 |

### 7.5 VTrain 内置预训练评估【设计】（基线缺失，属必要补强）

- **kNN 探针**：`method_args.knn_eval={enabled:"auto", every_n_epochs, k, split}`；auto = 数据集 ≤ 50 万张开启；独立 dataloader，不进训练路径；指标进 logger 与 run 元数据。
- **线性探针**（P1）：大骨干 detached 特征 + linear head 短训。
- 验收锚点：探针指标随训练单调可追踪，写入 nightly 基线看板。基线对照：预训练期唯一质量信号是 `train_loss`（ EarlyStopping 默认 monitor=train_loss 且 patience=1e12，形同虚设，`_callbacks/callback_args.py:25-28`）。

---

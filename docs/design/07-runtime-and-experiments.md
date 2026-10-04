# VTrain 实现蓝图 · 分册 07：训练运行时 · 实验管理与可复现性

| 文档属性 | 内容 |
|---|---|
| 版本 | v3.1（实现蓝图拆分版，内容与总览同步维护） |
| 总览 | [framework_design.md](../framework_design.md) |
| 全局章节 | §10–§11 |

> **标注约定**：【基线】lightly-train 0.17.0 已验证事实（出处 `文件:行`，路径相对 `src/lightly_train/`）；【设计】VTrain 决策；【差异】有意偏离基线的决策（附 ADR 编号）。正文中的 § 编号为**全局章节编号**，章节与分册的映射见总览。

---

## 10. 训练运行时

### 10.1 引擎形态：基线事实与 VTrain 决策

【基线】双形态并存，且**各自有不可替代的理由**：

- **预训练 = PL Trainer**：`Trainer(default_root_dir=out, max_epochs, accelerator, strategy, precision, loggers, callbacks, sync_batchnorm=(CUDA))`；resume 直接 `trainer.fit(..., ckpt_path="last")`（完整训练态由 Lightning 恢复）；增强调度无 step 级需求。
- **任务微调 = Fabric 手写循环**（`train_task.py:1269-2004`，VTrain 引擎设计的事实参照）：`fabric.launch()` → 顺序组装（§5.3 task 流）→ `fabric.setup(model, optimizer)` → `InfiniteCycleIterator` → 每步：set_step → （活跃态变化则重建 loader）→ `fabric.no_backward_sync(enabled=is_accumulating)` + `training_step` + `backward(loss/grad_accum)` → NaN 检查 → `clip_gradients` → `optimizer.step/scheduler.step` → `on_train_batch_end()`（EMA）→ 周期性 save/export → 周期性验证 + best 跟踪 + 可视化落盘。

【差异】VTrain 单引擎（ADR-0001），但 v2.0 的理由需修正：基线用 Fabric 的**真实动机**是①grad-accum、②step 级增强调度与 dataloader 重建、③step 级 EMA/保存/验证编排、④step 语义（steps 而非 epochs）——这四点在 Trainer 下都别扭。因此 VTrain 的 `_engine` 接口必须内建这四个原语（`fit_loop(steps, grad_accum, step_hooks)`），否则单引擎只是把复杂度搬进 Trainer hack。逃生舱 `trainer_args` 透传并文档声明不承诺兼容。

### 10.2 分布式【基线】

- rank 探测链（`_distributed.py` + `_lightning_rank_zero.py`，VTrain 照抄并补 K8s 变量）：
  - global：`RANK → LOCAL_RANK → SLURM_PROCID → JSM_NAMESPACE_RANK`
  - local：`LOCAL_RANK → SLURM_LOCALID → JSM_NAMESPACE_LOCAL_RANK`
  - node：`NODE_RANK → GROUP_RANK → SLURM_NODEID`
  - `is_global_rank_zero` = rank==0 或（未设且 node0）；`is_local_rank_zero` 未设时视为 True（单进程语义）。
- `strategy="auto"`：多设备 → `ddp_find_unused_parameters_true`（蒸馏头/冻结教师需要）；`sync_batchnorm` 仅 CUDA。
- 全局 batch：用户给 loader_args.batch_size 时 global = per-device × devices；否则 clamp 到数据集大小（告警）且必须被 devices 整除（`ValueError`）。
- rank0 独占写（日志/checkpoint/导出）；跨 rank out 目录一致性握手：`verify_out_dir_equal_on_all_local_ranks`（共享 tmp 目录写 marker 文件互认，超时 30s，`common_helpers.py:159-209`）——多节点共享存储场景防呆，VTrain 照抄。

### 10.3 精度与数值健康【基线→设计】

- pretrain `precision="auto"` → bf16-mixed（`torch.cuda.is_bf16_supported()`）否则 32-true；task **默认 bf16-mixed 且无 auto**。
- fp32 岛：检测 decoder 的 bbox refinement 段落强制 fp32（§8.2）；匹配器 `nan_to_num`；TRT 对 dinov3/vits16+fp16 用 strongly-typed 网络保 FP32 attention。
- `float32_matmul_precision="auto"` = 沿用 torch 当前值（尊重外部设置）。
- 【设计】VTrain 把散落经验收敛为 `fp32_modules=[...]` 配置 + 自动健康监控（NaN/Inf 捕获定位首个异常层、梯度范数记录、loss spike kσ 告警）。基线的 `_debug/` 是现成参考实现：`NaNCaptureMonitor` 在 clip/step 前扫梯度，命中即保存可重放快照（模型 state + microbatch + RNG）并抛 `NaNDetectedError`（`_debug/nan_capture.py`）；`DebugUnderflowOverflow` 为 vendored HF 精简版。

### 10.4 回调与周期产物【基线】

| 回调 | 行为 |
|---|---|
| DataWaitTQDMProgressBar（恒开） | `data_wait = data_time / (batch_time + data_time)`，进度条显示百分比（`tqdm_progress_bar.py:50-58`） |
| ModelExport（默认开，每 epoch） | rank0 写 `out/exported_models/exported_last.pt` |
| ModelCheckpoint（默认开） | `out/checkpoints/`，save_last=True/save_top_k=1/版本计数关闭；`on_save_checkpoint` 注入 `checkpoint["lightly_train"]` 元数据块（§10.5） |
| LearningRateMonitor（默认开） | 修复 PL `_get_optimizer_stats` 返回类型的补丁子类 |
| MLFlowLogging（有 MLflow logger 时） | 启动 SystemMetricsMonitor（CPU/GPU）+ 上传 train.log 为 artifact |
| EarlyStopping（默认开但惰性） | monitor=train_loss, patience=1e12, check_finite=True——实际作用仅 NaN 熔断 |

### 10.5 checkpoint 与状态【基线】

```
预训练 ckpt = {"state_dict": Method 全训练态,
               "lightly_train": {version, date(ISO), license_info,
                                 models: {model, wrapped_model, embedding_model},  # pickle 整对象
                                 normalize_args: {mean, std}}}                      # embed 依赖它
任务 ckpt = TrainTaskState = {train_model, optimizer, scheduler, train_dataloader,
                              step, model_class_path, model_init_args, license_info}
exported .pt = {"model_class_path", "model_init_args",
                "train_model": get_export_state_dict()  # 检测器=仅 EMA 键
                "license_info"}
```

- **无 seed、无配置快照**（seed 只在日志与 logger hyperparams）——VTrain 补 `run.json`/`config.resolved.yaml`（§11）。
- resume 与 finetune 语义分离且实现清晰，VTrain 照抄：
  - `resume_interrupted=True`：只从 last.ckpt 读 `model_init_args`（meta device 加载省内存），随后 `fabric.load(path, state)` 全状态恢复；要求同 out 目录。
  - `checkpoint=<path>`：`fabric.load` 后仅取权重，`strict=False` 并告警 missing/unexpected 键；校验 checkpoint 的 `model_init_args["model_name"]` 与请求模型同家族，否则报「不兼容 task-model 家族」。
  - 模型参数本身可指向本地路径 / 可下载注册名 / 纯架构名，五分支解析顺序固定（`train_task_helpers.py:1331-1399`）。

### 10.6 容错现状与 VTrain 补强【基线→设计】

基线已有：原子写（mmap tmp+rename；ckpt 依赖 torch.save + 外部调度重试）、dataloader timeout 180s、mmap 等待超时、out 目录握手、（恢复）`resume_interrupted`。基线没有：step 看门狗、心跳文件、OOM 引导、自动拉起。VTrain 按 v2.0 §10.4 表格实现（原子 ckpt / auto-resume / 看门狗 / OOM 引导 / 心跳），机制设计不变。

### 10.7 环境变量面【基线】

全部 17 项见附录 C。VTrain 等价物 `VTRAIN_*` 全套 + 新增 `VTRAIN_OFFLINE / VTRAIN_MIRROR_URL / VTRAIN_TELEMETRY(默认0) / VTRAIN_STEP_TIMEOUT_SEC / VTRAIN_DATA_CACHE_DIR`。基线的 `EnvVar[T]` 冻结 dataclass（`name/_default/_type/convert_empty_str_to_default`，空串回退默认值）值得照抄。

---


## 11. 实验管理与可复现性

### 11.1 Run 目录规范【基线→设计】

【基线】（实测产物清单）

```
out/
├── train.log                        # RotatingFileHandler 1GiB×4
├── checkpoints/{last,best}.ckpt     # pretrain 仅 last；任务 best 由 watch_metric 驱动
├── exported_models/exported_{last,best}.pt
├── <logger 目录>                    # tensorboard(默认开) / metrics.jsonl(默认开, JSONLLogger)
│                                    # / mlflow / wandb(默认关)
├── image_examples/                  # 任务: 前 3 步 train/val 可视化网格
└── debug/                           # 任务: debug_args 开启时的 nan_capture/underflow 快照
```

【设计】VTrain 增加（v2.0 §11.1 保持）：`run.json`（git commit/框架版本/torch·cuda·cudnn/GPU 型号/数据指纹/seed/触发人）、`config.resolved.yaml`（审计锚点 + 复跑入口：`vtrain.finetune(config="runs/<exp>/config.resolved.yaml")`）、`heartbeat`、`eval/`。环境捕获参照基线 `_system.py`（platform/python/依赖版本含可选依赖/CPU/GPU properties/**双目录 git 信息**——包目录与 cwd 各查 branch/commit/uncommitted changes）。

### 11.2 Logger 体系【基线】

- 选择机制：Pydantic args 字段非 None 即启用（无 enum）；`loggers={"tensorboard": None}` 显式关闭。
- 预训练默认 `jsonl + tensorboard` 开、mlflow/wandb 关；任务侧无 jsonl，加 `log_every_num_steps/val_every_num_steps/val_log_every_num_steps`（auto：`min(100, max(1,steps//10))` / `min(1000, max(1,steps))` / `min(20, max(1,val_steps))`）。
- MLflow 断点续接：resume 时按 experiment_name+run_name 反查 run_id 续写；MLflow 系统指标监控与日志 artifact 上传由回调完成。
- 统一 `log_image_to_loggers`（TB add_image / W&B·MLflow log_image / JSONL 跳过）。

【设计】VTrain 保持「本地 run 目录自含 + logger 即插件」架构；tensorboard 默认，wandb/mlflow 经 extras；MLflow run 反查续接机制照抄。

---

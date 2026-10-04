# VTrain 实现蓝图 · 分册 11：附录：基线机制对照 · 源码地图 · 环境变量 · 术语表

| 文档属性 | 内容 |
|---|---|
| 版本 | v3.1（实现蓝图拆分版，内容与总览同步维护） |
| 总览 | [framework_design.md](../framework_design.md) |
| 全局章节 | 附录 A–D |

> **标注约定**：【基线】lightly-train 0.17.0 已验证事实（出处 `文件:行`，路径相对 `src/lightly_train/`）；【设计】VTrain 决策；【差异】有意偏离基线的决策（附 ADR 编号）。正文中的 § 编号为**全局章节编号**，章节与分册的映射见总览。

---

## 附录 A：与 lightly-train 的机制对照（含修正清单）

对照源码：`/home/hdjs/Research/Software/lightly-train-0.17.0/src/lightly_train/`（下表路径相对该目录）。

### A.0 v2.0 → v3.0 修正清单（源码核验结论）

| v2.0 表述 | 核验结论 | 依据 |
|---|---|---|
| 「Method 基类统一实现：training_step 求和 loss 分量字典」 | **错**。`training_step_impl` 返回 `TrainingStepResult(loss, log_dict)`，loss 由方法自行求和；基类只做日志与 sync | `_methods/method.py:38-41, 132-149` |
| 「配置 schema 版本化；读取旧版本由迁移器升级」为对照项 | 基线**无任何** schema/迁移机制；version 字符串写入但从不分支。保留为 VTrain 设计 | `_checkpoint.py:58-70`；全库 grep 无 migration |
| 「解析后配置强制落盘 config.resolved.yaml」为增强 | 正确方向；基线配置不落盘（仅日志+hyperparams），复跑无锚点 | `_commands/train.py:583-601` |
| 「albumentations 被旧版锁死」 | 不准确：约束 `>=1.3.1,!=1.4.*`（排除 1.4，允许 1.5+/2.x）；真正动机是 SuperGradients 依赖 1.3 + 1.4 有 bug | `pyproject.toml:27-28` |
| 「step 调度增强…实现简化为 batch 内丢弃」 | **放弃简化**：基线 StepScheduledCompose + dataloader 重建方案完整可用，直接移植 | `_transforms/ltdetr_transforms/components.py:58-231` |
| 「权重下载强制哈希校验」为增强 | 正确且必要：基线三套下载路径全无校验（源码留 TODO） | `dinov3_package.py:409` 等 |
| 「单引擎（对双引擎的简化）」理由 | 需修正：Fabric 循环的动机是 grad-accum/step 调度/EMA/step 语义四原语，单引擎必须内建之 | §10.1 |
| 「YOLO 系模型家族」 | 基线无 YOLO 检测器模型；yolo 仅为数据格式。P0 对标家族修正为 LT-DETR（RT-DETRv2/D-FINE）+ PicoDet | `_task_models/` 目录实况 |
| 「DistillationV3 教师=任意 pkg/model（默认 dinov3/vitb16）」 | 正确（v3 放开了 v1 的 DINOv2/v3 wrapper 限制） | `_methods/distillationv3/distillationv3.py:47-80, 96` |
| 「out 目录握手/rank 探测链/data_wait/env 面/无 wd 清单/lr 公式」等 | 全部核验属实，数值与出处见各章 | 各章【基线】标注 |

### A.1 机制对照总表

| 机制 | lightly-train 位置 | VTrain 采纳方式 |
|---|---|---|
| Pydantic 严格校验 + 错误美化 | `_configs/config.py:23-70`、`validate.py:40-185` | 照抄全套 model_config；prettify 增加拼写建议（§5.1） |
| 三配置方言 | `_commands/train.py:526-580` | 同构三方言 + YAML 外来键预过滤（§5.2） |
| auto 哨兵 + 原地 resolve | `X | Literal["auto"]`、各 `resolve_auto()`、`no_auto/assert_config_resolved` | 同机制；补 resolved 落盘 + UnresolvedAutoError 沿用（§5.3） |
| auto 解析顺序 | pretrain 流 `train.py:292-476`；task 流 `train_task.py:1276-1527` | 顺序固化进命令实现与集成测试（§5.3） |
| 规模分桶 | `_scaling.py` + 各方法桶表 | 组件化 + 桶表审计 + 对照单测（§9.4） |
| mmap 文件清单 | `_data/_serialize/memory_mapped_sequence.py`、`common_helpers.py:394-577` | 整方案复用：IPC/懒加载/引用计数/原子发布/撞目录防护（§9.2） |
| ModelWrapper 协议族 | `_models/model_wrapper.py`（246 行） | 预训练组照抄；任务组补显式 Protocol（§6.1/6.4） |
| Package 注册表与分发顺序 | `_models/package_helpers.py:34-157` | 照抄 + entry-points 插件 + 许可标签（§6.2） |
| DINOv2/v3 vendoring | `_models/dinov2_vit/`、`_models/dinov3/` | 依赖优先，缺口 vendor；权重 sha256 强校验（§6.3） |
| 蒸馏 v3（队列 KL 全局 + patch KL 局部） | `_methods/distillationv3/`（3 文件） | P0 等价实现，数值规格按 §7.3 固化 |
| lr 缩放 + CosineWarmup | `_methods/method.py:90-122` | 基类照抄（公式与 warmup 规则原样） |
| no-wd 参数清单 | `_optim/optimizer_helpers.py:83-217` | 清单照抄 + 全量单测（§7.1） |
| checkpoint 元数据内嵌 | `_checkpoint.py`、`_callbacks/checkpoint.py:89-126` | 扩展为自描述格式 + schema_version（ADR-0002） |
| resume/finetune 双语义 | `train_task_helpers.py:1331-1615` | 语义照抄；`resume_interrupted="auto"` 增强（§10.5） |
| 便携导出 + 动态类加载 | exported 四键、`task_model_helpers.py:165-278`、类路径迁移表 `:52-67` | 加 lineage/metrics/schema_version；迁移表制度化（§12.1） |
| ONNX 导出 Path A + FP16 黑名单 + 对拍 | `_export/export_onnx.py`、`onnx_helpers.py` | 全套照抄；对拍升格双门禁 + 黑名单必填制（§12.2） |
| TensorRT 管线 | `_export/tensorrt_helpers.py` | 照抄（static HW/strongly-typed/fp32 attention/TF32 清除） |
| SAHI 切片推理 | `object_detection_components/tiling_utils.py` | 照抄参数与合并策略（§12.4） |
| StepScheduledCompose step 调度 | `_transforms/ltdetr_transforms/components.py` + `ltdetr_schedule.py` | 整方案移植（compose 缓存 + 窗口公式 + loader 重建）（§9.5） |
| BatchReplayCompose 批级参数共享 | `_transforms/batch_transform.py:30-69` | 照抄（§9.3/9.5） |
| 进度条 data_wait | `_callbacks/tqdm_progress_bar.py:50-58` | 照抄 + 升格一等指标 |
| out 目录多 rank 握手 | `common_helpers.py:159-209` | 照抄（§10.2） |
| rank 环境变量探测链 | `_distributed.py`、`_lightning_rank_zero.py` | 照抄 + K8s 变量（§10.2） |
| env 变量面 | `_env.py`（17 项，附录 C） | 等价 `VTRAIN_*` 全套 + OFFLINE/MIRROR/TELEMETRY（§10.7） |
| 遥测 | `_events/tracker.py`（PostHog，默认开） | 结构采纳，默认关；env 解析容错教训（§14.1） |
| 双引擎 | Trainer（pretrain）/ Fabric 手写循环（task） | 单引擎薄封装（ADR-0001），四原语内建（§10.1/R3） |
| NaNCapture/UnderflowOverflow 调试 | `_debug/` | 收敛为 `debug_args` 健康监控配置（§10.3/§15.1） |
| torch.compile 仅包 forward | `_torch_compile.py:41-56`（明示不编 training_step） | 同策略（§8.1 上下文） |
| （缺失）预训练评估 | — | 自建 kNN/线性探针（§7.5） |
| （缺失）统一评测协议/数据治理 | — | 自建 EvalProtocol/DatasetDescriptor（§8.5/§9） |
| （缺失）发版自动化/GPU CI/覆盖率 | 5 workflow、手动 dist | VTrain 补齐（§13/§16） |


## 附录 B：基线源码地图

路径相对 `src/lightly_train/`；★ = VTrain M0/M1 直接移植对象。

| 路径 | 职责（关键文件） |
|---|---|
| `__init__.py` | 公共 API 出口（§4.1）；import 副作用：cv2 线程置 0、NO_ALBUMENTATIONS_UPDATE |
| `_cli.py` | CLI（仅 pretrain/export/embed/extract_video_frames/list_*） |
| `_commands/` | 命令层：`train.py`(pretrain)、`train_helpers.py`(预训练组装)、`train_task.py`(7 任务 + Fabric 循环★)、`train_task_helpers.py`(任务组装★)、`common_helpers.py`(out/checkpoint/数据/批大小★)、`data_helpers.py`、`embed/export/export_task/predict_task/benchmark_task`、`_warnings.py` |
| `_configs/` | ★ `config.py`(PydanticConfig)、`validate.py`(校验+美化)、`model_registry.py`、`omegaconf_utils.py` |
| `_models/` | ★ `model_wrapper.py`(协议族)、`package.py`+`package_helpers.py`(包注册表)、`embedding_model.py`、`_model_helpers.py`(通道/位置编码 hooks)；子包 `timm/ torchvision/ dinov2_vit/ dinov3/ ultralytics/ rfdetr/ super_gradients/ ecvit/ custom/ rtdetr/` |
| `_methods/` | ★ `method.py`(基类)、`method_args.py`、`method_helpers.py`(注册表)；子包 `distillationv3/ distillationv2/ distillation/ dino/ dinov2/ simclr/ densecl/ (dinov31/ detcon/ 未注册)` |
| `_data/` | ★ `image_dataset.py`、`task_dataset.py`、`object_detection_dataset.py`、`coco/yolo_*_dataset.py`、`file_helpers.py`(解码)、`label_helpers.py`、`infinite_cycle_iterator.py`、`_serialize/memory_mapped_sequence.py`★ |
| `_transforms/` | ★ `transform.py`(参数类+MethodTransform)、`view_transform.py`、`batch_transform.py`、`task_transform.py`(step 钩子)、`ltdetr_transforms/`(检测管线+StepScheduledCompose★)、`mosaic/mixup/copyblend/scale_jitter/random_iou_crop/random_photometric_distort.py` |
| `_task_models/` | ★ `task_model.py`(推理态基类)、`train_model.py`(训练态基类)、`task_model_io.py`(导出类型)、`task_model_helpers.py`(load_model)；子包 `ltdetr_object_detection/`(检测主家族)、`picodet_object_detection/`、`object_detection_components/`(criterion/decoder/matcher/EMA/调度/tiling★)、`ltdetr_instance_segmentation/`、`*eomt*/`(分割)、`image_classification*/`、`depth_estimation/` |
| `_optim/` | `optimizer_helpers.py`(参数组/无 wd 清单★)、`optimizer_args.py`、`adamw/adamw8bit/sgd/lars_args.py`、`trainable_modules.py` |
| `_metrics/` | `task_metric.py`(watch_metric 机制)、`mean_average_precision.py`、`loss_metric_collection.py`、按任务子包 |
| `_callbacks/` | `checkpoint.py`(元数据注入)、`export.py`、`tqdm_progress_bar.py`(data_wait)、`learning_rate_monitor.py`、`mlflow_logging.py`、`callback_helpers.py` |
| `_checkpoint.py` / `_task_checkpoint.py` / `_train_task_state.py` | ★ ckpt 元数据 / 保存参数 / Fabric 状态 schema |
| `_distributed.py` / `_lightning_rank_zero.py` / `_system.py` | ★ rank 链 / 系统信息 |
| `_export/` | ★ `export_onnx.py`(Path A)、`export.py`(ExportMixin)、`onnx_helpers.py`(fp16/metadata/repair)、`tensorrt_helpers.py` |
| `_embedding/` | `embedding_predictor.py`、`embedding_transform.py`、`writers/` |
| `_loggers/` | `logger_args.py`、`jsonl/tensorboard/mlflow/wandb.py`、`task_logger_args.py` |
| `_pre_post_processing/` | `object_detection.py`(pre/decode/post/SAHI 处理) |
| `_debug/` | `nan_capture.py`、`underflow_overflow.py`、`debug_args.py` |
| `_events/` | `tracker.py`(PostHog 遥测) |
| `_env.py` / `_license.py` / `_logging.py` / errors.py / types.py | ★ 环境变量 / 许可文案 / 日志 / 错误层级 / 公共类型 |


## 附录 C：基线环境变量全表

（`_env.py`；VTrain 等价物前缀 `VTRAIN_`）

| 变量 | 默认 | 作用 |
|---|---|---|
| `LIGHTLY_TRAIN_LOG_LEVEL` | INFO | 控制台日志级别；CLI `-v` 置 DEBUG |
| `LIGHTLY_TRAIN_CACHE_DIR` | `~/.cache/lightly-train` | 缓存根（含 userid.txt） |
| `LIGHTLY_TRAIN_MODEL_CACHE_DIR` | `<cache>/models` | 权重/可下载 ckpt 缓存 |
| `LIGHTLY_TRAIN_TMP_DIR` | `<cache>/data` | 临时目录（out 握手 marker 等） |
| `LIGHTLY_TRAIN_DATA_CACHE_DIR` | `<tmp>` | mmap 清单文件 |
| `LIGHTLY_TRAIN_DATALOADER_TIMEOUT_SEC` | 180 | DataLoader timeout（0 关闭） |
| `LIGHTLY_TRAIN_IMAGE_MODE` | None | 图像读取模式（RGB/UNCHANGED） |
| `LIGHTLY_TRAIN_MASK_DIR` | None | 预训练附加掩码目录 |
| `LIGHTLY_TRAIN_MAX_NUM_WORKERS_AUTO` | 8 | num_workers auto 上限 |
| `LIGHTLY_TRAIN_DEFAULT_NUM_WORKERS_AUTO` | 8 | CPU 数不可测时回退 |
| `LIGHTLY_TRAIN_MMAP_TIMEOUT_SEC` | 300 | 等 rank0 mmap 超时（-1 关闭） |
| `LIGHTLY_TRAIN_MMAP_REUSE_FILE` | False | 按 data 哈希复用 mmap（跨 run） |
| `LIGHTLY_TRAIN_VERIFY_OUT_DIR_TIMEOUT_SEC` | 30 | out 目录握手超时 |
| `LIGHTLY_TRAIN_DOWNLOAD_CHUNK_TIMEOUT_SEC` | 180 | 下载分块超时 |
| `MLFLOW_TRACKING_URI` | None | MLflow 默认 tracking uri |
| `SLURM_CPUS_PER_TASK` / `SLURM_JOB_ID` | None | num_workers auto / Slurm 探测 |
| `LIGHTLY_TRAIN_POSTHOG_KEY` / `LIGHTLY_TRAIN_EVENTS_DISABLED` | 内置 key / False | 遥测开关（"1" 关闭） |


## 附录 D：术语表

| 术语 | 定义 |
|---|---|
| Method | 预训练/蒸馏方法实现（LightningModule 子类）：`training_step_impl → TrainingStepResult(loss, log_dict)` |
| TaskModel | 推理态任务模型：preprocess_image/preprocess_batch → forward_backend → postprocess 三段式，纯 torch，可导出 |
| TrainModel | 训练态包装：TaskModel + criterion + metrics + 参数组 + EMA；`training_step(fabric, batch, step) → TaskStepResult` |
| ModelWrapper | 骨干适配协议（forward_features/forward_pool/feature_dim/get_model ± multiscale/patch_size/architecture_info） |
| EvalProtocol | 统一评测协议（resize 规则/匹配规则/指标实现/指纹），训练内 val 与 evaluate/benchmark 共用【设计】 |
| Run | 一次训练/评测的产物目录，自含配置、元数据、产物【设计】 |
| auto 解析 | `"auto"` 哨兵按依赖链在命令流中原地求值（resolve_auto），`no_auto/assert_config_resolved` 兜底 |
| 金标准测试 | 固定 seed 下逐步 loss 序列快照比对，拦截配方回归 |
| StepScheduledCompose | 按 step 活跃状态切换（并缓存）增强子管线的 Compose；配合 dataloader 重建传播到 worker |
| MemoryMappedSequence | PyArrow IPC 文件清单的 mmap 序列：零拷贝、pickle 只带路径、pid 键懒加载 |
| EMA 优先导出 | `get_export_state_dict` 仅保留 `ema_model.*` 键——导出物即 EMA 权重 |
| Promotion | 模型资产 dev→staging→prod 门禁晋升【设计】 |
| SAHI | 切片推理：tile(overlap)+全图 global tile → 坐标偏移合并 + NMS/global-local 抑制 |
| ORIGIN.md | vendor 目录的上游来源/改动/升级说明（vendor 白名单制载体） |

---

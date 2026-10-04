# VTrain 实现蓝图 · 分册 09：质量工程 · 安全合规 · 可观测性

| 文档属性 | 内容 |
|---|---|
| 版本 | v3.1（实现蓝图拆分版，内容与总览同步维护） |
| 总览 | [framework_design.md](../framework_design.md) |
| 全局章节 | §13–§15 |

> **标注约定**：【基线】lightly-train 0.17.0 已验证事实（出处 `文件:行`，路径相对 `src/lightly_train/`）；【设计】VTrain 决策；【差异】有意偏离基线的决策（附 ADR 编号）。正文中的 § 编号为**全局章节编号**，章节与分册的映射见总览。

---

## 13. 质量工程

### 13.1 基线测试体系实况【基线】（VTrain 的起点与超越对象）

- 164 个测试文件**镜像 src 结构**；conftest 三件套：缓存目录逐测试隔离、遥测硬关（`LIGHTLY_TRAIN_EVENTS_DISABLED=1`）、`long_running_test` 标记在 CI 自动跳过。
- `tests/helpers.py`（1128 行）：程序化合成数据工厂（图像/掩码/COCO/YOLO/分类/视频全覆盖，**零二进制测试资产**）+ DummyMethod/DummyModel 假件 + 微型测试骨干（`_vittest14` 等 vendored 微模型）。
- **金标准测试已存在**：如 `DINOLoss` 前向断言 `loss == pytest.approx(1.5565, rel=1e-4)`（`tests/_methods/dinov2/test_dinov2_loss.py:65-88`）；错误信息断言精确到字符串（`tests/_configs/test_validate.py:109-117`）。
- 端到端：真实 `pretrain(epochs=1, batch_size=4, 10 张合成图)` + resume 后 state_dict 对比；**不用 fast_dev_run**。
- 短板（VTrain 补齐）：无 GPU runner（GPU 测试全部 skipif 跳过）、无覆盖率工具链、无性能基准、无多卡/容错注入测试、发版无自动化 workflow（`make dist` + twine 手动）。

### 13.2 VTrain 测试策略（七层）【设计】

| 层 | 内容 | 触发 | 基线对照 |
|---|---|---|---|
| L1 单元 | 纯逻辑，mock 全部 IO | 每 PR < 3min | 已具备（镜像结构沿用） |
| L2 组件 | DataModule/导出管线/协议，合成 tiny 数据集 | 每 PR < 10min | 已具备雏形 |
| L3 冒烟 | finetune 1 epoch toy → evaluate → export → load_model 预测 | 每 PR < 15min | 基线 pretrain 端到端模式移植 |
| L4 **数值金标准** | 固定 seed 训 N=50 步，loss 序列逐点容差比对（0.5% 相对误差） | 涉训练路径的 PR | 基线只有单点 approx；VTrain 升级为序列快照 |
| L5 多卡集成 | 2×GPU DDP + kill -9 resume 一致性 | nightly | 基线缺失 |
| L6 性能基准 | img/s + GPU busy，30 天趋势，劣化 3% 告警 | nightly | 基线有指标无基线库 |
| L7 兼容矩阵 | Python × torch × CUDA | weekly + 发版 | 基线仅 minimal/maximal 两轴 |

辅助设施照抄基线：合成数据生成器（`tests/fixtures/`）、pytest marker 分组、`uv lock --exclude-newer` 冻结解析、conftest 环境隔离。

### 13.3 代码质量门禁【设计】

| 门禁 | 工具 | 规则（基线现状） |
|---|---|---|
| 分层依赖 | import-linter | §3.1 分层；禁 core→`_adapters/[ultralytics|hf]`（基线无自动守卫） |
| 类型 | mypy strict | 基线全量 strict + vendored 目录 ignore——照抄其 per-file 策略 |
| 风格 | ruff + isort(first-party 分组) + mdformat | 基线 `extend-select=["I"]` + Google docstring 约定沿用 |
| 覆盖率 | pytest-cov + diff-cover | core ≥85% 且增量 ≥90%（基线无） |
| 弃用 | `-W error::DeprecationWarning` | 基线无 |
| vendor | CI 脚本 | `_vendor/` 无 ORIGIN.md 即失败（基线用 licenseheaders 查头，升级为查 ORIGIN） |
| 公共 API 快照 | pytest 快照 | 未声明的 API 变更即失败 |

---


## 14. 安全与合规

### 14.1 基线事实（VTrain 的反例清单）【基线】

| 领域 | lightly-train 0.17.0 实况 |
|---|---|
| 许可 | AGPL-3.0 本体；`LICENSE_INFO` 文案写入每个 pretrain ckpt、任务 ckpt/exported、ONNX metadata（无运行时强制） |
| AGPL 隔离 | ultralytics 不进 Docker 镜像（Makefile 显式排除+注释）；vendored DINOv3 为专有许可（"DINOv3 License Agreement" 头） |
| 权重供应链 | 三套下载机制（自研 urllib / torch.hub / 第三方自管）**均无哈希校验**；DINOv3、ECViT 源码留 TODO；DINOv3 文件名里的 hash 串仅装饰（`check_hash=False`） |
| 遥测 | PostHog **默认开启**；仅 2 个事件（training_started/inference_started）+ 属性（os/gpu 名/版本/is_ci/is_container）；持久化匿名 UUID 于缓存目录；30s 事件级限流；rank0 + 守护线程 + 全异常吞掉；`LIGHTLY_TRAIN_EVENTS_DISABLED="1"` 关闭（注意 Makefile 里的引号写法实际不生效，conftest 的裸 `1` 才有效——env 解析要写成容错的，教训） |
| 数据 | 框架对数据只读（mmap/缓存全部落 cache 目录）；telemetry 不含数据路径 |

### 14.2 VTrain 设计【设计】

保留 v2.0 全部措施（Apache-2.0 本体、依赖许可矩阵三区制、extras/适配层隔离、SBOM、cosign、trusted publishing），并新增两条由基线教训直接导出的硬门禁：

1. **权重校验门禁**：所有远程权重（骨干/教师/注册模型）强制 sha256 校验，`Env` 解析写容错单测。
2. **遥测默认关** + 日志脱敏过滤器 + DatasetDescriptor 的 `pii/allowed_purpose` 标记（§9 数据治理表保留）。

---


## 15. 可观测性与运维

### 15.1 基线观测能力【基线】

| 能力 | 实现 |
|---|---|
| 日志 | 包级 logger 挂 4 个来源（自身/pytorch_lightning/torch/py.warnings）；console 无时间戳彩色、文件 `[asctime][level] msg`；RotatingFileHandler 1GiB×4；`LIGHTLY_TRAIN_LOG_LEVEL` + CLI `-v` |
| 训练指标 | logger 体系（§11.2）；loss 分量键约定 `{split}_loss/{name}`、指标键 `{split}_metric/{name}`、classwise `{split}_metric_classwise/{name}` |
| 系统指标 | CUDAUtilization 守护线程 0.2s 采样；TrainingStepTimer 计时（train/val × step/dataload 四相位）；跨 rank 聚合（时间/显存取 max、利用率取 mean）；输出键 `profiling/{split}_gpu_util_perc|gpu_max_mem_mb|step_time_sec|data_time_sec|throughput_img_per_sec`；console 直显 GPU Util/Step Time/Data Time/img/s |
| 数据饥饿 | 进度条 data_wait 百分比（§10.4）——VTrain 升格为一等指标并入门禁 |
| 调试 | `debug_args`：NaNCaptureMonitor（可重放快照）+ DebugUnderflowOverflow（逐模块 abs min/max 帧） |

### 15.2 VTrain 设计【设计】

保留 v2.0 三支柱与 Runbook 表；实施要点更新：structlog JSONL 文件日志（替换基线纯文本格式）；NVML 系统指标沿用采样线程模式但并入统一指标面；OTel spans（P1）埋点位置参照基线四相位计时器（dataload/step/ckpt/validation）。

---

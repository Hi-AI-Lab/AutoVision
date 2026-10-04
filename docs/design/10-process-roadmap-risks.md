# VTrain 实现蓝图 · 分册 10：交付形态 · 工程组织 · 路线图 · 风险

| 文档属性 | 内容 |
|---|---|
| 版本 | v3.1（实现蓝图拆分版，内容与总览同步维护） |
| 总览 | [framework_design.md](../framework_design.md) |
| 全局章节 | §16–§19 |

> **标注约定**：【基线】lightly-train 0.17.0 已验证事实（出处 `文件:行`，路径相对 `src/lightly_train/`）；【设计】VTrain 决策；【差异】有意偏离基线的决策（附 ADR 编号）。正文中的 § 编号为**全局章节编号**，章节与分册的映射见总览。

---

## 16. 支持矩阵与交付形态

### 16.1 基线支持事实【基线】

- Python `>=3.8,<3.14`（3.14 被 Lightning 阻挡），CI 实测 3.8（lowest-direct 最小依赖）与 3.13（pinned 最大）两端。
- torch `>=2.1.0` 下限；CI 锁 2.1.0（minimal）/ 2.4.0+cu118 / 2.7.0（maximal）+ ROCm 6.1/7.2 变体组。
- Lightning `pytorch_lightning>=2.1`（导入名是 pytorch_lightning 非 lightning）；torchmetrics[detection]>=0.8（SuperGradients 拖住下限）。
- pydantic 传递依赖（via lightly）；albumentations `>=1.3.1,!=1.4.*`。
- Docker：单镜像 `pytorch/pytorch:2.5.1-cuda11.8-cudnn9-runtime`，pillow-simd 替换、预下载 DINOv3 权重、冒烟测试脚本（产出/日志/GPU 生效/CSV 行数断言）。
- 无 GPU CI runner；无自动发版 workflow。

### 16.2 VTrain 支持矩阵与交付【设计】

保留 v2.0 §16 全表（Python 3.10–3.13、torch 2.4–2.7 N-2、CUDA 11.8/12.x、Linux 生产/macOS 开发、PyPI extras、双镜像 cosign+SBOM、K8s PyTorchJob/Slurm 模板、离线矩阵项）。工程细节采纳基线：`uv lock --exclude-newer` 冻结、uv conflicts 声明互斥 extras、Makefile 集中 extras 组合、Docker 冒烟脚本模式。

---


## 17. 工程组织与协作流程

【设计】保留 v2.0 §17 全部（monorepo src layout、trunk-based、CODEOWNERS、变更分级、diataxis 文档、release captain）。基线补充实践：

- 版本号：基线为 `__version__` 静态字符串 + Makefile grep 提取（Docker/docs 复用）——VTrain 用 setuptools-scm（tag 即版本源）。
- 文档工程：基线 Sphinx+furo+autoapi；配置参考靠 `docs/prebuild.py` **从 Pydantic 模型导出方法/变换参数默认值页面**（每次构建重生成）——VTrain 用 MkDocs + 等价 schema 导出脚本，保证「配置参考永不与代码漂移」。
- changelog：Keep a Changelog 格式入库、docs 构建时渲染并剥离 Unreleased 段。
- dev 工具：license 头模板 + `make add-header` + CI 头检查；pre-commit（notebook 清洗、大文件 2MB 拦截）。

---


## 18. 路线图与实现映射

| 里程碑 | 周期 | 内容 | 退出门禁 | 基线参照实现（移植起点） |
|---|---|---|---|---|
| **M0 架构骨架** | 3 周 | 包结构、PydanticConfig+validate+registry、`_engine` 接口（fit_loop/grad_accum/step_hooks 四原语）、CI 骨架（lint/type/unit/分层） | 门禁全绿；ADR 0001-0004 落档 | `_configs/`（5 文件全量移植改造）、`_distributed.py`、`_env.py` |
| **M1 Alpha（检测微调闭环）** | 6 周 | coco/yolo 数据 + mmap 清单、LT-DETR 家族（RT-DETRv2/D-FINE criterion/decoder、EMA、参数组、flat-cosine）、torchmetrics 评测、benchmark、单机多卡 | L3/L4 绿；多卡集成过；快速开始 CI 验证 | `_data/`、`_transforms/ltdetr_transforms/`、`_task_models/ltdetr_object_detection/`、`object_detection_components/`、`_metrics/` |
| **M1.5 Beta（部署闭环）** | 4 周 | exported 格式 v2 + load_model、ONNX Path A（含双门禁）、TRT、SAHI、Docker | 故障注入过；性能基线建立；安全扫描零阻断 | `_export/`、`task_model_io.py`、`tiling_utils.py`、docker 冒烟模式 |
| **M2 Beta（预训练）** | 6 周 | plain_images 数据、蒸馏 v0→v1（§7.3 全量）、kNN 探针、权重供应链（哈希/镜像/离线） | 蒸馏金标准基线；探针趋势验收；许可审计 | `_methods/`（基类+distillationv3）、`_embedding/`、`_scaling.py`、dinov2/v3 包 |
| **M3 GA** | 4 周 | 矩阵全验证、runbook、文档全覆盖、注册表（文件系统后端）、发版自动化演练 | NFR 全表验收；法务/安全签字；回滚演练 | —— |
| M4+ | 持续 | 分割/分类、MLflow 注册表、int8 PTQ、Ascend、OTel | 滚动门禁 | `_task_models/*eomt*`、`_optim/adamw8bit_args.py` |

节奏约定：GA 前 MINOR 每 6–8 周；GA 后 semver + 弃用政策（§4.3）。

---


## 19. 风险登记册

| # | 风险 | 概率/影响 | 缓解 | 兜底 |
|---|---|---|---|---|
| R1 | 上游（Lightning/albumentations/torch）破坏性升级 | 高/高 | N-2 窗口 + `--exclude-newer` 冻结 + 矩阵 CI 提前吃版本；albumentations 大版本隔离在适配层 | `_engine`/适配层钉旧版并行验证（基线 `!=1.4.*` 与 SuperGradients 拖拽即为实证） |
| R2 | AGPL 依赖边界失守 | 中/高 | import-linter 门禁 + extras 隔离 + code owner（基线靠 Makefile 注释级纪律，必须自动化） | 依赖替换预案 |
| R3 | **单引擎无法优雅承载 step 级增强调度/grad-accum/EMA 编排** | 中/高 | `_engine.fit_loop` 内建四原语（§10.1）；L4 金标准覆盖 step 调度切换点；dataloader 重建机制（§9.3）随引擎移植 | 退化方案：任务路径走 Fabric 薄循环（基线已验证可行），`_engine` 双后端 |
| R4 | 导出精度静默劣化 | 中/高 | 双门禁（数值对拍 atol5e-3 + 指标对拍）+ 每模型 FP16 黑名单必填（基线 LT-DETR = Softmax+MatMul 实证热点） | per-model 豁免登记 + TRT strongly-typed |
| R5 | 金标准过脆（硬件差异误报） | 中/中 | 相对容差 + 按 GPU 型号分基线 + rebase 流程（基线 `approx(rel=1e-4)` 单点实践起步） | 退化为趋势监控 |
| R6 | vendored DINOv2/v3 上游变更 + 下载源失效 | 中/中 | ORIGIN.md + 本地权重缓存 + sha256（基线零校验的补丁）+ 镜像源 | 教师权重一经缓存冻结可用 |
| R7 | 多团队插件命名冲突 | 低/中 | entry-points 冲突即报错 + 命名空间建议 | priority/别名 |
| R8 | 性能缓慢劣化无人察觉 | 中/中 | nightly 趋势 + 3% 门禁（基线 profiling 键即指标来源） | 季度人工复评 |
| R9 | 配方知识在人（参数组规则、桶表、窗口调度等隐性经验） | 中/高 | 本文档 §7-§9 已显式固化数值；ADR + 金标准快照即文档；每子包 ≥2 owner | —— |
| R10 | 数据合规事件 | 低/高 | 日志脱敏 + 遥测默认关 + pii 标记 | 事件响应 SOP |

---

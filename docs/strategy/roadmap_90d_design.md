# AutoVision 90 天路线：设计与实施手册

| 文档属性 | 内容 |
|---|---|
| 版本 | v1.0 |
| 日期 | 2026-10-17 |
| 状态 | 内部分析 / 实施文档（非对外发布产物；站点经 `exclude_docs` 排除） |
| 输入 | [差距分析](gap-analysis.md) 的 P0/P1/P2 结论 + 当前工作树（0.2.0 基线 + 在途改动）的只读盘点 |
| 范围 | 阶段 0–4：发布开门、黄金路径、部署最后一公里、数据闭环、生态与商业 |
| 读者 | 框架研发、部署工程、数据/算法、产品与合规 |
| 关联 | [架构总览](../framework_design.md)、[导出与部署](../export.md)、[基准](../benchmarks.md)、[发布流程](../release_process.md)、[已知限制](../known_limits.md) |

> 这份文档不重复“缺什么”——[差距分析](gap-analysis.md) 已经写得很细。它回答三个执行问题：
> **先做什么、每一步的具体设计是什么、什么算做完。** 每个阶段都给出代码接入点、
> 数据结构、CLI/API 变更、测试与验收命令，以及明确不做的边界。

---

## 0. 执行摘要

### 0.1 一句话

AutoVision 的工程内核已经足够支撑一次公开发布，但采用漏斗仍为零：
**0 个自研 `released` 权重、PyPI 未发布、无公开远程仓库、站点无 URL、导出图不含后处理。**
接下来 90 天的核心不是“再补一个任务”，而是把已有能力变成三条闭环：

1. **用户发现 → 第一次成功**：`pip install` → 加载干净权重 → 对自己的图出框；
2. **模型 → 设备上的最终框**：导出件 + 版本化后处理 + 端到端 parity；
3. **误差报告 → 下一轮训练**：数据体检 → 难例清单 → 采样/多轮伪标注。

### 0.2 五条实施原则

| 原则 | 含义 | 对设计的约束 |
|---|---|---|
| **发布优先** | 先让外部用户跑通，再补能力 | 阶段 0 不等任何模型训练；用现有 RF-DETR / GTR 权重开门 |
| **一条黄金路径** | 每个角色只记一条最短路径 | 首屏只展示检测：安装 → 加载 → 预测 → 导出；其余能力降级到“进阶” |
| **干净权重先行** | 先用许可干净、权重现成的家族建立采用 | RF-DETR（Apache-2.0）做可部署旗舰；GTR（MIT）做高精度旗舰；自研谱系后置 |
| **证据化验收** | 每个阶段有可执行门禁，不靠叙事 | 验收命令、测试名、报告文件写进本文；CI 能复跑 |
| **不做清单同等重要** | 防止路线图重新膨胀成 15 个后端 | 第 8 节列出明确不做项，新增需求先过该清单 |

### 0.3 总览

| 阶段 | 周期 | 核心交付 | 北极星判据 |
|---|---|---|---|
| 0 开门 | 1 周 | 公开仓库 + 文档站 + PyPI 0.3.0 + 英文入口 + 可执行快速开始 | 干净 venv 一条命令装成功；陌生人 10 分钟内看到第一张结果图 |
| 1 黄金路径 | 30 天 | 预训练检测句柄、模型目录、RF-DETR 模型卡、GTR 检测 m/l/x 公开、训练配方 | 5 分钟出框；10 分钟导出 ONNX |
| 2 部署最后一公里 | 60 天 | 版本化后处理器 + golden vectors、端到端 parity、OpenVINO 运行时、int8 AP、多路视频、C++ 示例 | 同一张图在 torch/ONNX/OpenVINO/TRT 上最终框一致；int8 精度损失有数字 |
| 3 数据闭环 | 90 天 | data-check、难例清单 → 采样、多轮伪标注、CVAT/Label Studio 互通、长尾采样 | 1k 标注 + 100k 无标注的第二轮 AP 提升可复现 |
| 4 生态与商业 | 6 个月 | 插件协议、模型库榜单/verified、企业合规包、私有注册表 + 部署门禁、LTS/SLA、硬件伙伴包 | 外部贡献者不改核心加模型；3 个企业 POC；1 个硬件伙伴 |

### 0.4 人力假设

本文工作量按 **2 名全职工程师 + 0.5 名文档/社区** 估算；若只有 1 名工程师，
时间乘以约 1.8。阶段 2 与阶段 3 可并行（部署工程 / 数据算法两条线），阶段 4
在阶段 1 发布后即可并行启动。

### 0.5 实施进度（2026-10-17）

已落地（代码 + 测试）：

| 条目 | 落点 | 状态 |
|---|---|---|
| 任务级模型目录 / `list-models --task` / `model-info` | `registry/catalog.py`、`cli/_app.py` | ✅ |
| 预训练检测句柄（RF-DETR 全链；GTR 门控装配） | `tasks/detection/pretrained.py`、`cli/model_handle.py` | ✅（GTR 需真机权重验证） |
| 训练配方 `recipe=` | `cli/recipes.py`、`train_object_detection` | ✅（含 1 步真实训练测试） |
| 版本化后处理规格 + NumPy 参考解码 | `export/postprocess.py`、导出三后端 sidecar | ✅（非 DETR 任务跳过） |
| 端到端框级 parity | `export/e2e_parity.py` | ✅（比较最终框，不比较张量求和） |
| OpenVINO 运行时 | `inference/engine.py`、`inference/stream.py`、`serving/app.py` | ✅（接线测试；真机 IR 读数待有兼容 onnx 的环境） |
| 数据体检 `data-check` | `data/quality.py`、`cli/data_check.py` | ✅（核心 + CLI 夹具测试） |
| 难例清单 + 加权采样 | `metrics/error_analysis.py`、`data/hard_examples.py`、`train_object_detection` | ✅（单机；DDP 组合显式拒绝） |
| 多轮自训练编排 | `scripts/self_train_detect.py` | ✅（真实 2 轮测试；逐轮报告 + summary） |
| 长尾采样 | `data/samplers.py`、`train_object_detection` | ✅（单机；与 hard_examples 互斥） |
| 标注工具互通 | `data/interchange/`、`convert-data` | ✅（CVAT bbox/polygon；LS rectangle/polygon/keypoint-as-box） |
| 多路流运行时 | `inference/multi_stream.py`、`predict sources=...` | ✅（共享批处理 / 重连 / 背压 / 每路产物；无跨路跟踪） |
| PicoDet NMS 后处理规格 | `export/postprocess.py`（`family=picodet`） | ✅（与 torch 后处理逐框对拍） |
| int8 vs fp32 AP 报告 | `scripts/measure_int8_accuracy.py`、`export/int8_report.py` | ✅（报告+单测；真实读数待兼容 ONNX/TRT 环境） |
| C++ ONNX Runtime 示例 | `examples/cpp/` | ✅（host postprocess 测试实测；完整 ORT 构建按环境） |
| 插件协议 | `plugins.py`、`examples/plugin-demo/`、`plugins` CLI | ✅（entry point / model_source / 模板） |
| 企业合规包 | `compliance/`、`compliance-report` CLI | ✅（依赖许可 / 模型卡+sha / 数据快照） |
| 私有注册表 + 部署门禁 | `registry/model_registry.py`、`registry/deployment_gates.py`、`registry` CLI | ✅（current / rollback / audit / 6 门禁） |
| LTS / SLA 文档 | `docs/lts_policy.md`、`docs/support_policy.md` | ✅（政策草案） |
| 硬件伙伴包模板 | `deploy/packs/_template/` | ✅（convert + target parity；真机包按客户） |
| 发布准备 0.3.0 | `_version.py`、CHANGELOG、`configure_release.py`、docs workflow | ✅ 构建/校验就绪；DEIMv2 派生 STA 已删除，许可阻塞解除 |
| legacy ONNX 导出参数错位修复 | `export/export_onnx.py` | ✅ |

验证读数：新增 18 个测试文件 **105 passed / 2 skipped**；自训练套件
**3 passed**（含真实 2 轮）；非 ONNX 推理套件 **13 passed**；serving
**9 passed**；ruff / 分层 / 溯源 / 文档断链 / 模型库 `--check` /
`mkdocs --strict` 全绿。ONNX 推理 2 项因当前环境的 numpy 1.23 与
onnx/onnxscript ABI 不兼容而失败（`export` extra 已加 `numpy>=1.25`，
需重装环境后复测）。

未完成 / 明确阻塞：

- **发布动作**：0.3.0 版本 / CHANGELOG / wheel + sdist / twine check /
  Pages workflow 已就绪；公开仓库与 Pages URL 仍需真实 org/域名，PyPI 上传
  需 trusted publisher 或 token（包名在 PyPI/TestPyPI 均为 404 = 可用）。
  许可阻塞已解除，但需用删除 STA 后的源码重新构建产物。
- **许可处置（已完成）**：`SpatialPriorModule` 已删除，DINOv3 ViT 保留并
  改用 no-STA adapter，ConvNeXt 不变；旧 `backbone.sta.*` 检查点会被显式
  拒绝。见 `THIRD_PARTY_LICENSES.md` / `docs/provenance/LEDGER.md`。
- **真实读数**：RF-DETR / GTR 的真实权重 AP、int8 AP delta、四后端实机 parity
  需要网络 / COCO / GPU / 兼容的 ONNX 环境；本轮只交付代码路径与门控测试。
- **环境阻塞**：`export` extra 已提升到 numpy ≥1.25；当前 `.venv` 仍是旧
  numpy 1.23，ONNX 导出集成测试按环境门控跳过；重装 `.[export]` 后复测。
  CI 的 `export` 环境同样需要 numpy ≥1.25（或对齐 onnx 版本）。
- **后续阶段**：GTR 真机装配验证、int8 真实 AP 读数、C++ ONNX Runtime 全量
  构建（依赖 ORT/OpenCV）、企业 POC / 硬件伙伴目标端 parity（商业结果，
  非代码）尚未完成。

---

## 1. 现状基线（只列与路线相关的事实）

### 1.1 发布与入口

- PyPI 包 **尚未上传**：`docs/pypi_publish.md:23` 自述，README 的
  `pip install autovision-framework` 对外部用户目前不可执行。
- 仓库 **无公开 remote**，盘点时工作树有 67 个未提交变更；`[project.urls]` 仍是
  TODO（`pyproject.toml:40-42`）；`mkdocs.yml` 没有 `site_url` / `repo_url`。
- CI 已有 tag 触发的 `publish` job（`.github/workflows/ci.yml:75-97`），
  但 PyPI pending publisher 尚未配置。
- 文档以中文为主；`README_PYPI.md` 也是中文。对全球开源社区，英文入口缺失。

### 1.2 模型与权重

- `RELEASED_SPECTRUM_WEIGHTS` 为空（`src/autovision/registry/spectrum_release.py:48-50`），
  自研检测谱系 **0 个 `released` 档**。
- `SPECTRUM` 中 `detr/n`、`detr/s` 为 `spec-frozen`，`detr/m/l/x` 为 `planned`
  （`src/autovision/tasks/detection/detr/variants.py:418-477`）。
- `detr/n` 与 `detr/s` 的体量几乎相同：17.79M / 59.1 GFLOPs 与
  18.14M / 61.8 GFLOPs（`docs/benchmarks.md:78-83`）。这不是一个有意义的
  小尺度谱系；在 YOLO 语境里两者都接近 M 档。
- **GTR 检测 m/l/x 的规格已经注册**（`gtrvit-m` / `gtrvit-l` / `gtrvit-x`，
  `src/autovision/tasks/detection/detr/variants.py:337-375`），并有构造级测试
  （`tests/test_gtr_detection_variants.py:84-116`）。缺的是公开模型键、模型卡、
  全量 COCO AP 与文档，而不是“从零接线”。
- **RF-DETR 完整权重已注册**：Nano / Small / Medium / Large 走
  `Roboflow/rf-detr-*` + `load_pretrained_verified`
  （`src/autovision/encoders/rfdetr/rfdetr_source.py`）。训练配置默认
  `load_weights=True`（`src/autovision/tasks/detection/rfdetr/config.py:215`），
  但任务模型构造函数默认 `load_weights=False`
  （`src/autovision/tasks/detection/rfdetr/task_model.py:337-340`）——这是
  推理侧“默认不加载权重”的根因。
- `autovision.detection(model)` 只返回训练句柄；没有“从注册表加载预训练权重并
  直接预测”的入口（`src/autovision/cli/model_handle.py:255`）。
- `list_models()` 只是编码器目录名（`src/autovision/encoders/__init__.py:43-45`），
  不包含 `rfdetr/*`、`detr/*`、`rtdetr/*` 等任务模型，也没有“有无预训练权重”
  的状态。

### 1.3 部署与推理

- `InferenceEngine` 的后端只有 `torch` / `onnx` / `tensorrt`
  （`src/autovision/inference/engine.py:27`）。OpenVINO 有导出和 benchmark
  后端、也有 `OpenVINOEngineRunner`
  （`src/autovision/export/openvino_helpers.py:171`），但没有接进推理运行时
  与 serving。
- 导出图只包含 forward：`postprocess` 留在 Python（`TaskNetwork` 契约），
  `export_onnx.py` 没有 decode/NMS 逻辑。parity 门禁比较的是原始输出张量的
  按维求和（`src/autovision/tasks/detection/detr/task_model.py:793-820`），
  不是最终框/分数/类别。
- int8 的验收是张量级容差（atol 2.0 / rtol 0.5），没有“int8 vs fp32 的 AP
  下降”报告。
- `run_stream` 只接受单个 source（`src/autovision/inference/stream.py:116`），
  没有多路调度、断流重连、队列背压。
- 没有 C++ / 移动端 / 边缘端参考运行时；ONNX hub 策略成立，但设备侧仍需自己
  重写前后处理。

### 1.4 数据闭环

- 伪标注 / 自训练是 **单轮**（`scripts/self_train_detect.py`）；误差分析产出
  报告与画廊（`src/autovision/metrics/error_analysis.py:439`），但产物不回灌
  训练。
- 没有数据体检：坏图、重复图、框越界、split 泄漏、类别分布、标签噪声都没有
  系统检查。
- 没有标注工具互通（CVAT / Label Studio）；没有长尾采样 / 重加权。
- 数据快照能力已存在（`src/autovision/data/versioning.py:215`），但训练不会
  自动绑定快照。

### 1.5 生态与注册

- 模型注册表已有文件系统 + dev/staging/prod + sha256（`src/autovision/registry/model_registry.py:87`），
  但无回滚、无 `current` 指针、无部署门禁、无审计日志、未接 serving。
- `ModelCard` 与谱系发布门禁已存在（`src/autovision/registry/model_card.py`、
  `spectrum_release.py`），但只服务自研谱系，尚未覆盖 RF-DETR / GTR 的
  公开模型卡。
- 编码器侧有 `register_source`（`src/autovision/encoders/package_helpers.py:41`），
  但没有 entry-point 插件协议，head/loss/metric/dataset 没有公开扩展点。
- `scripts/gen_model_zoo.py` 从注册表机器生成模型库，但主角是第三方权重，
  没有“已发布/已验证”的产品表。

### 1.6 已具备但未兑现的能力（本路线最重要的杠杆）

| 能力 | 代码现状 | 差的是什么 | 对应阶段 |
|---|---|---|---|
| RF-DETR 完整权重 | 权重已注册、加载器已验证 | 预训练推理入口、模型卡、默认加载策略 | 1 |
| GTR 检测 m/l/x | 变体规格已注册并有测试 | 模型卡、全量 AP、公开文档（模型键已在注册表） | 1 |
| OpenVINO 运行时 | Runner 已存在 | 接进 `InferenceEngine` / serving / CLI | 2 |
| 模型卡 / 发布门禁 | schema 与校验已存在 | 覆盖 RF-DETR/GTR；接模型库与榜单 | 1 / 4 |
| 数据快照 | 快照 + 漂移校验已存在 | 训练自动绑定、模型卡记录 digest | 3 |
| 伪标注 / 误差分析 | 单轮闭环已通 | 多轮编排、难例回灌、数据体检 | 3 |
| 注册表 | 不可变版本 + 晋级 | 回滚、current、部署门禁、审计 | 4 |
| 模型目录 | 编码器目录 | 任务级目录 + 预训练状态 + 许可/可导出 | 1 |

---

## 2. 阶段 0：开门（1 周）

### 2.1 目标

让一个陌生人能在干净环境里完成：**安装 → 跑通 CLI → 打开文档站 → 看到第一张
结果图**。这一阶段不训练任何模型，只做发布工程与入口修正。

### 2.2 现状与阻断

| 阻断 | 证据 | 处置 |
|---|---|---|
| 包未发布 | `docs/pypi_publish.md:23` | 配置 trusted publisher，打 `v0.3.0`，CI 上传 |
| 无公开 remote | `git remote -v` 为空 | 建公开仓库；按发布流程推孤儿快照分支 |
| 工作树脏 | 67 个未提交变更 | 按主题拆分提交；发布快照只含发布树 |
| 项目 URL 缺失 | `pyproject.toml:40-42` | 填 Homepage / Documentation / Repository / Issues / Changelog |
| 文档站无 URL | `mkdocs.yml` 无 `site_url` | 确定 Pages 地址；加 `site_url` / `repo_url` |
| 快速开始不可执行 | README 首条 `pip install` 在包发布前失败 | 发布前改为源码安装；发布后切回 PyPI |
| 英文入口缺失 | README / PyPI README 中文 | 新增 `README.en.md` + 英文 quickstart；中文保持 canonical |

### 2.3 设计

#### 2.3.1 发布物与版本

- 版本：`0.3.0`（MINOR：新增发布入口与黄金路径，不破坏 0.2 公共 API）。
- 发布形态：wheel 纯 Python + sdist（GTR/MSDA 源随 sdist）；PyPI 发行名
  `autovision-framework`。
- 发布快照：按 [发布流程](../release_process.md) 的孤儿单提交分支；tag
  `v0.3.0` 必须与 `src/autovision/_version.py`、CHANGELOG 一致。
- CI：复用现有 `publish` job；首次发布前在 PyPI 配 pending publisher
  （project `autovision-framework`、workflow `ci.yml`、environment `pypi`）。

#### 2.3.2 公开入口

- GitHub 仓库：README 首屏只保留三件事——一句话定位、安装、10 行 quickstart；
  其余能力收进 `<details>` 或文档站。
- 文档站：`docs/` 现有页面已足够；新增 docs/quickstart.md（英文 + 中文各一段），
  从安装到预测到导出，命令可复制。
- Issue / PR 模板已存在，发布后启用 Discussions（或等价渠道），置顶
  “如何报告精度/部署问题”模板，要求附模型键、后端、硬件、复现命令。

#### 2.3.3 快速开始契约

新增 docs/quickstart.md。**0.3.0（阶段 0）的可执行版本**先走示例脚本：

```bash
python examples/quickstart_detect.py --image demo.jpg --out preds/
```

该脚本直接构造 RF-DETR 任务模型并 `load_weights=True`，输出框并导出 ONNX；
它是阶段 0 的临时入口，阶段 1 再切换到下面的稳定 API。

**目标形态（0.4.0 起）**为四步：

```bash
# 1. 安装（发布后用 PyPI；发布前用源码）
pip install "autovision-framework[rfdetr,export]"

# 2. 加载干净权重，直接预测（不训练）
python - <<'PY'
import autovision
model = autovision.detection("rfdetr/rf-detr-small", pretrained=True)
result = model.predict("demo.jpg")
print(result["bboxes"][:3], result["scores"][:3])
PY

# 3. 导出 ONNX（默认 parity 门禁）
python - <<'PY'
import autovision
model = autovision.detection("rfdetr/rf-detr-small", pretrained=True)
model.export("demo.onnx")
PY

# 4. 用自己的数据微调
autovision-cli train-object-detection out=runs/ft data=my_data.yaml \
    model=rfdetr/rf-detr-small recipe=recipes/detection/finetune.yaml
```

> 稳定 API 依赖阶段 1 的预训练句柄；阶段 0 的验收以“示例脚本可跑”为准，
> 不以 API 形态为准。

快速开始必须由 CI 跑：新增 scripts/check_quickstart.py 或
tests/test_quickstart.py，在 CPU 上执行：
- 构造一个 64×64 的合成图；
- 用 `-notpretrained`/随机权重变体跑 `predict` 与 `export(.onnx)`；
- 断言输出字段与 parity 通过。
真实权重下载的版本放在 nightly job（`AUTOVISION_HF_TESTS=1`），避免阻塞 PR。

#### 2.3.4 英文入口

- 新增 `README.en.md`：定位、安装、quickstart、三个差异化（许可干净 /
  数值可验证 / 协议化基准）、文档链接。
- PyPI 页面正文改为英文 + 中文摘要（或英文 `README_PYPI.md` + 中文段落）；
  所有链接用绝对 URL。
- 不在阶段 0 重写全部中文文档；先保证“英文能进门，中文有深度”。

### 2.4 实施步骤（WBS）

| # | 工作 | 产出 | 人日 |
|---|---|---|---|
| 0.1 | 清理工作树，按主题提交；确认 1900+ 测试基线 | 干净分支 + 基线报告 | 1 |
| 0.2 | 填 `[project.urls]`、`site_url`、`repo_url` | pyproject / mkdocs 变更 | 0.5 |
| 0.3 | 建公开仓库；配置 PyPI pending publisher | 仓库 URL / OIDC 配置 | 0.5 |
| 0.4 | 写 docs/quickstart.md + `README.en.md` | 两页文档 | 1.5 |
| 0.5 | 写 tests/test_quickstart.py（CPU 合成图） | CI 门禁 | 1 |
| 0.6 | 收编 CHANGELOG，升 0.3.0，打 tag，CI 发布 | PyPI 0.3.0 / 文档站 | 1 |
| 0.7 | 发布后验证（干净 venv / CLI / 文档 / 安装 smoke） | 验证记录 | 0.5 |
| 0.8 | 发布公告 / 迁移说明（AGPL 用户为什么值得看） | 公告草稿 | 1 |
| 0.9 | 最小 quickstart 示例（阶段 0 临时入口，阶段 1 切换稳定 API） | examples/quickstart_detect.py | 1 |

总计约 **8 人日**。

### 2.5 验收门禁

```bash
# 干净环境
uv venv /tmp/av-release
uv pip install --python /tmp/av-release/bin/python "autovision-framework[rfdetr,export]==0.3.0"
/tmp/av-release/bin/autovision-cli --help
/tmp/av-release/bin/python tests/test_quickstart.py

# 仓库门禁
ruff check src tests scripts
pytest tests -q
python scripts/check_provenance.py
python scripts/check_layering.py
python scripts/check_docs_links.py
python scripts/gen_model_zoo.py --check
mkdocs build --strict
```

完成判据：上述全部通过；文档站 URL 可访问；PyPI 项目页 README / 许可 /
文件列表正常；`pip install autovision` 不会安装本项目（发行名说明生效）。

### 2.6 风险与回退

| 风险 | 概率 | 处置 |
|---|---|---|
| PyPI 名称相似被拒 | 中 | 已确认 `autovision-framework`；不要用 `AutoVision` / `auto-vision` |
| trusted publisher 配置错误 | 中 | 先用手动 token 发 TestPyPI 演练；CI 失败可改 token 发布 |
| 工作树拆分引入回归 | 中 | 发布前跑全量套件；按主题提交，避免混入在途实验 |
| 英文文档质量不足 | 中 | 先短后长：README + quickstart 英文，深度文档保持中文并标注 |
| 发布后问题响应不过来 | 中 | 置顶 issue 模板 + 每周固定 triage 时段 |

回退：若 PyPI 发布失败，保留源码安装路径；文档站可先上线，tag 延后。不要把
未验证的 0.3.0 作为“latest”。

---

## 3. 阶段 1：黄金路径（30 天）

### 3.1 目标与用户体验

三个角色，各一条最短路径：

| 角色 | 目标体验 | 时间 |
|---|---|---|
| 应用开发者 | 安装 → 加载预训练 → 对自己的图出框 → 导出 ONNX | 5 / 10 分钟 |
| 算法工程师 | 自定义数据集 → 一条命令微调 → val → 误差分析 | 30 分钟 |
| 部署工程师 | 拿到带后处理规格的导出件 → 在目标端复现同样的框 | 1 小时 |

### 3.2 现状

- RF-DETR 权重与加载器已就绪，但推理句柄没有预训练入口；训练配置的
  `load_weights=True` 只在训练路径生效。
- GTR 检测 m/l/x 规格已注册，公开文档只强调 `detr/gtrvit-s`；没有 m/l/x 的
  模型卡与全量 AP 行。
- `list_models()` 只列编码器；用户无法发现 `rfdetr/*`、`detr/*`。
- 没有训练配方层：每个参数都要手写；新用户不知道该用多少 lr / epochs / AMP。
- 快速开始目前是“训练 2000 步 + coco8 合成集”，不是“加载权重 → 预测”。

### 3.3 设计

#### 3.3.1 预训练检测句柄

**目标 API：**

```python
import autovision

# 1) 直接预测（不训练）
model = autovision.detection("rfdetr/rf-detr-small", pretrained=True)
result = model.predict("image.jpg")          # bboxes / scores / labels
model.val("coco_val.yaml")                   # 可选：COCO AP
model.export("model.onnx")                   # parity 门禁

# 2) 微调（预训练初始化）
model = autovision.detection("rfdetr/rf-detr-small", pretrained=True).train(
    out="runs/ft", data="my_data.yaml", recipe="finetune",
)
```

**实现设计：**

1. `DetectionModel.__init__(model, *, pretrained: bool | None = None)`：
   - `None`（默认）：按家族能力解析。RF-DETR / GTR 有完整权重 → `True`；
     RT-DETR / D-FINE / PicoDet 无完整任务权重 → `False`（保持现状）。
   - `True`：立即或惰性构造预训练网络；`False`：只保留训练句柄语义。
2. 新增 tasks/detection/pretrained.py，提供
   `build_pretrained_detection(model_key, *, device="cpu") -> TaskNetwork`：
   - `rfdetr/*`：按变体解析原生分辨率（`RFDETR_MODEL_REGISTRY`），读取
     检查点原生 `num_labels`，用 COCO 80 类 + 背景约定构造
     `RFDETRObjectDetection(load_weights=True)`；**不要**先 resize 分类头
     再加载（会破坏官方权重）。
   - `detr/gtrvit-*`：解析 `gtr_{s,m,l,x}_coco` 资产，构造
     `DETRObjectDetection`，通过 `model_args={"checkpoint": ...}` 直载；
     记录 `variant_spec` 与资产 sha256。
   - 其他家族：明确报错，并给出“先训练 / 用 RF-DETR / 用 GTR”的指引，
     不做静默随机初始化。
3. `DetectionModel._ensure_pretrained()`：`predict/val/export` 在无
   `_network` 时尝试一次预训练加载；失败时错误信息必须包含
   “该模型没有可用的预训练权重；请先 `.train(...)` 或选用 rfdetr/gtr”。
4. 导出重建：`load_task_model` 仍然以 `load_weights=False` 重建结构，再读
   state dict；预训练加载只发生在用户入口，不改变导出语义。
5. 依赖门控：`rfdetr` 需要 `transformers>=5.9`；`gtr` 需要 Triton/CUDA。
   CPU 环境优先推荐 RF-DETR；GTR 的 CPU 路径明确报错。

**分类头细节（实施时必须验证）：**

- COCO 检测的类别数约定在各家族不同（80 / 91 / 背景类）。预训练加载器必须
  以检查点原生 `num_labels` 为准，预测输出映射到 COCO 80 类名；不要在
  `_reinitialize_class_head` 之后再用 80 类覆盖。
- 需要一个测试：加载 RF-DETR Nano，在前向输出上断言 `num_classes` 与
  类别名映射一致；GTR 同理。

#### 3.3.2 模型目录（catalog）

新增 registry/catalog.py，把“用户可选的模型”集中成一个机器可读表：

```python
@dataclass(frozen=True)
class ModelEntry:
    key: str                 # "rfdetr/rf-detr-small"
    task: str                # "detection" | "instance_segmentation" | ...
    family: str              # "rfdetr" | "gtr" | "detr" | "rtdetr" | ...
    status: str              # "released" | "spec-frozen" | "planned"
    pretrained: str          # "full" | "backbone" | "none"
    license: str             # SPDX id of the weights
    weights_asset: str | None
    input_size: tuple[int, int] | None
    params_m: float | None
    gflops: float | None
    exportable: tuple[str, ...]  # ("onnx", "tensorrt", "openvino")
    card: str | None         # 模型卡路径或 URL
    notes: str = ""
```

- 数据来源：`DETR_MODEL_REGISTRY` / `SPECTRUM` / RF-DETR variant 表 /
  GTR 资产表 / `ModelCard`；不要手写重复表。
- 新增 `list_task_models(task=None, pretrained_only=False)`，保留现有
  `list_models()` 的编码器语义（向后兼容）。
- CLI：
  - `autovision-cli list-models --task detection` 输出表；
  - `autovision-cli list-models --pretrained` 只列有完整权重的；
  - `autovision-cli model-info <key>` 输出许可、权重、AP、延迟、可导出后端、
    一行加载代码。
- `scripts/gen_model_zoo.py` 从 catalog 生成模型库的“可发布模型”表，避免
  文档与注册表漂移。

#### 3.3.3 双旗舰与模型卡

**旗舰 A：RF-DETR N/S/M/L（Apache-2.0，可部署）**

- 模型键：`rfdetr/rf-detr-{nano,small,medium,large}`。
- 模型卡字段：官方 AP / AP50、参数量、输入尺寸、权重 sha256、上游 repo /
  revision、许可、导出后端、延迟（至少 torch + ONNX；TRT/OV 由阶段 2 补）。
- 明确标注“上游权重，非 AutoVision 训练”，避免与自研谱系混淆。
- 默认输入尺寸取变体原生分辨率（Nano 384 / Small 512 / Medium 576 / Large 704）。

**旗舰 B：GTR s/m/l/x（MIT，高精度）**

- 检测公开键：`detr/gtrvit-{s,m,l,x}`；权重资产
  `gtr_{s,m,l,x}_coco` 已注册。
- 官方 AP：53.6 / 57.3 / 58.9 / 59.4；延迟 1.225 / 1.462 / 1.908 / 2.115 ms
  （官方协议，见 [基准](../benchmarks.md) §3）。
- 模型卡必须写清：**不可导出 ONNX/TRT**（GLA 自定义核），适用训练 / 高精度
  推理 / 研究，不适用边缘部署。
- 阶段 1 的硬交付：用 `scripts/eval_spectrum_coco.py` 或新增
  scripts/eval_gtr_detect_ap.py 在 COCO val2017 全量复跑四档，把读数写进
  [基准](../benchmarks.md) §1，并附复现脚本。这是“协议化基准”最有说服力的
  一次兑现。

**命名纪律：**

- 对外不再用 `detr/n`、`detr/s` 的“n”去暗示边缘档；它们当前约 59–62 GFLOPs。
  在自研谱系真正做出 5–10 GFLOPs 的 nano 之前，模型库把它们标为
  `spec-frozen`，不放进“可选模型”首屏。
- RF-DETR 的 N/S/M/L 用官方命名，不映射成 `detr/n...x`。

#### 3.3.4 GTR 检测 m/l/x 公开与 AP

代码接线已基本完成，剩余工作是“产品化”：

1. 在 `docs/model_zoo.md` 增加 GTR 检测 m/l/x 行：模型键、资产、官方 AP、
   sha256、加载代码、导出边界。
2. 新增/扩展评测脚本，跑全量 5000 图：
   - 若 `scripts/eval_spectrum_coco.py` 已支持 `detr/gtrvit-*`，直接复用；
   - 否则新增 scripts/eval_gtr_detect_ap.py，按“数据面 → 变换 → 网络 →
     官方的后处理 → pycocotools”全链评测，输出 JSON + markdown。
3. 把四档读数写入 `docs/benchmarks.md` §1（自家精度），标注 seed（官方权重
   无训练 seed，标 “official EMA weights, inference reproduction”）、硬件、
   协议。与官方值的 delta 必须记录，不能只写“一致”。
4. 加一个回归测试：`tests/test_gtr_detection_variants.py` 已覆盖 m/l/x 构造；
   新增“公开键可解析 + 模型卡存在 + 基准行存在”的契约测试，防止文档漂移。

#### 3.3.5 训练配方（recipes）

新增 `recipes/` 目录，先做三条，不做配方 DSL：

```text
recipes/detection/finetune.yaml      # 自定义数据微调（RF-DETR / GTR / RT-DETR 通用默认）
recipes/detection/coco-finetune.yaml # COCO 全量微调（发布读数用）
recipes/detection/low-data.yaml      # 小数据：强增广、低 lr、早停
```

配方 v1 只覆盖三个嵌套参数组（`model_args` / `transform_args` /
`trainer_args`）：lr、weight_decay、EMA、增广、评估间隔等。steps / batch_size
等顶层参数保持显式，避免与函数默认值产生“用户是否显式指定”的歧义；后续若
需要顶层配方，单独设计优先级规则。实现方式：

- 新增 cli/recipes.py：`load_recipe(name_or_path)` 读取 YAML，与 CLI 参数
  合并（CLI 优先）；未知键报错（复用现有 Pydantic 严格校验）。
- `train_object_detection(..., recipe="finetune")` 接受内置名或路径。
- 配方不是新配置系统：最终仍落在现有 `model_args` / `transform_args` /
  `trainer_args`，不引入第二套 schema。
- 测试：配方能构造配置；CLI `recipe=` 能跑 2 步；未知键失败。

#### 3.3.6 快速开始与示例

新增 `examples/`：

```text
examples/quickstart_detect.py     # 阶段 0 已建；阶段 1 切到稳定 API
examples/finetune_detect.py       # 自定义数据微调 + val + 误差分析
examples/video_predict.py         # 视频/RTSP 单路推理
examples/export_matrix.py         # torch/ONNX/OV/TRT 同图对比（阶段 2 用）
```

- 每个示例都要能在 README/文档中一行命令运行。
- CI 跑 CPU 合成图版本；真实权重版本放 nightly。
- 示例输出统一为 COCO 风格的 `bboxes/scores/labels`，避免每个示例一套格式。

### 3.4 实施步骤（WBS）

| # | 工作 | 产出 | 人日 |
|---|---|---|---|
| 1.1 | `build_pretrained_detection` + RF-DETR 预训练加载 | 模块 + 单测 | 3 |
| 1.2 | `DetectionModel(pretrained=True)` + 惰性加载 + 错误文案 | API + 测试 | 2 |
| 1.3 | GTR 检测预训练加载（checkpoint 解析） | 模块 + 测试 | 2 |
| 1.4 | registry/catalog.py + `list-models --task/--pretrained` + `model-info` | 目录 + CLI + 测试 | 3 |
| 1.5 | RF-DETR N/S/M/L 模型卡（AP/延迟/sha256/许可） | 卡片 + 文档 | 2 |
| 1.6 | GTR 检测 m/l/x 公开 + 全量 AP 复跑 | 读数 + 基准行 + 脚本 | 3 |
| 1.7 | `recipes/` + cli/recipes.py + 三条配方 | 配方 + 测试 | 2 |
| 1.8 | 四个 examples + CI quickstart | 示例 + 门禁 | 2 |
| 1.9 | 文档：模型库、README、quickstart 重写 | 文档 | 1 |

总计约 **20 人日**。

### 3.5 验收门禁

```bash
# 预训练路径
pytest tests/test_pretrained_detection.py -q
pytest tests/test_model_catalog.py -q
pytest tests/test_recipes.py -q

# 真实权重（nightly / GPU 门控）
AUTOVISION_HF_TESTS=1 pytest tests/test_pretrained_quickstart.py -q

# 基准一致性
python scripts/eval_gtr_detect_ap.py --tier s --data coco.yaml --max-images 5000
python scripts/gen_model_zoo.py --check
```

完成判据：

1. 干净环境 `autovision.detection("rfdetr/rf-detr-small", pretrained=True).predict(img)`
   成功，输出 COCO 80 类名；
2. `model.export("demo.onnx")` 在默认 parity 门禁下通过；
3. `autovision-cli list-models --task detection --pretrained` 列出 RF-DETR
   N/S/M/L 与 GTR s/m/l/x；
4. `docs/benchmarks.md` §1 有 GTR 检测 s/m/l/x 四行全量 AP，且脚本可复跑；
5. 配方能跑 2 步训练；未知键报错；
6. quickstart CI 通过；真实权重 nightly 通过。

### 3.6 风险

| 风险 | 处置 |
|---|---|
| RF-DETR 原生类别数/COCO 映射不一致 | 以检查点 `config.num_labels` 为准；加载测试锁定类别名映射 |
| `transformers>=5.9` 的 RF-DETR 类不稳定 | 锁定下限 + 运行时门控；失败时给出安装指引，不回退到随机权重 |
| GTR 在 CPU 不可用 | 文档明确 CUDA/Triton 要求；CPU 用户走 RF-DETR |
| GTR 全量 COCO AP 复跑耗时长 | 先跑 s/m/l/x 四档，允许夜间任务；每档记录硬件与耗时 |
| 配方层变成第二套配置系统 | 配方只做 YAML 合并，最终值必须过现有 Pydantic 校验 |
| “双旗舰”让用户困惑 | 模型库首屏用两列对照：可部署（RF-DETR）/ 高精度（GTR），并标注导出边界 |

---

## 4. 阶段 2：部署最后一公里（60 天）

### 4.1 目标

把“转换不掉精度”从承诺变成可复现的证据：

1. 设备侧拿到的不只是 forward 图，还有**版本化后处理规格 + golden vectors**；
2. torch / ONNX / OpenVINO / TensorRT 的**最终框**一致，而不只是原始张量；
3. int8 有 **AP 下降**数字，不只是张量容差；
4. 多路视频能稳定跑；目标端有一个最小 C++ 参考实现。

### 4.2 现状

- 导出图只含 forward；后处理在 `TaskNetwork.postprocess` 的 Python 侧。
- parity 比较原始输出按维求和，容易“通过”但最终框不一致。
- OpenVINO 有 runner，但 `InferenceEngine` / serving / CLI 不认识它。
- int8 只有张量级 parity；没有 mAP。
- `run_stream` 单路；无重连、无背压、无每路指标。
- 没有 C++/设备侧参考。

### 4.3 设计

#### 4.3.1 版本化后处理器

新增 `export/postprocess/`（Python + NumPy 参考实现），把“如何从原始输出得到
最终框”从任务模型里抽出来，变成可序列化、可跨语言复现的规格。

**规格（`postprocess.json`，随导出件一起交付）：**

```json
{
  "schema_version": 1,
  "family": "detr",
  "input_size": [640, 640],
  "num_queries": 300,
  "score_threshold": 0.5,
  "top_k": 300,
  "box_format": "cxcywh_normalized",
  "nms": null,
  "classes": ["person", "bicycle", "..."],
  "outputs": ["logits", "boxes"],
  "preprocess": {
    "resize": "letterbox",
    "mean": [0.485, 0.456, 0.406],
    "std": [0.229, 0.224, 0.225],
    "pad_value": 0
  }
}
```

- DETR 家族：sigmoid → top-k → 阈值 → cxcywh→xyxy → 按 metadata 还原原图。
- Anchor/NMS 家族（PicoDet 等）：额外给出 `nms` 字段（IoU 阈值、max dets、
  class-aware 与否）。
- 规格里包含 **preprocess**，因为设备侧最常见的偏差来自 resize/letterbox/
  normalize，而不是 forward。
- 导出时写 `<model>.postprocess.json`；`export_onnx` / `export_tensorrt` /
  `export_openvino` 都写同一份规格。`ModelInputSpec` 已有输入形状，
  规格补后处理与预处理。

**参考实现：**

```python
from autovision.export.postprocess import load_spec, decode

spec = load_spec("model.postprocess.json")
records = decode(raw_outputs, metadata, spec)
```

- 纯 NumPy，不依赖 torch（可在无 torch 的设备环境运行）。
- 与任务模型的 `postprocess` 做数值对拍：同一 raw outputs + metadata，
  框差 ≤ 1e-4（fp32）或 IoU ≥ 0.999。
- 版本号 `schema_version` 一旦发布只增不改；新增字段保持向后兼容。

#### 4.3.2 端到端 parity

新增 export/e2e_parity.py：

```python
def verify_end_to_end(
    network, *, backends, artifact_paths, images, spec,
    box_tolerance=1.0, score_tolerance=1e-3,
) -> ParityReport:
    """Run preprocess -> forward(backend) -> reference postprocess on the
    same images; compare FINAL boxes/scores/labels, not raw tensors."""
```

- 对每个后端：`preprocess` 用同一份图像与 metadata；`forward` 走对应
  backend；`postprocess` 用同一份 spec；比较：
  - 框：匹配后的最大坐标差（像素）与 IoU；
  - 分数：最大绝对差；
  - 类别：完全一致；
  - 数量：允许 ±N（top-k 并列时），超出即失败。
- 接入：
  - `benchmark(verify_parity=True)` 默认跑端到端；保留原始张量 parity 作为
    第二道门禁；
  - `export` 增加 `verify_end_to_end=True`（默认开）；
  - 失败时保留 `parity_report.json`，含逐图逐框差异，便于定位。
- Golden vectors：`tests/golden/e2e/` 存固定图像（或合成张量）+ 期望最终框；
  不存大权重。测试用 tiny/随机权重模型，数值稳定即可。

**为什么不直接把 NMS 导进图：**

- DETR 家族是 NMS-free，top-k/阈值可导；anchor 家族的 NMS 在 TRT/OV 上
  算子支持不一致；
- 把后处理导进图会让不同后端的图分叉，反而增加“转换掉精度”的风险；
- 阶段 2 先交付“图 + 版本化后处理 + golden vectors”，阶段 2 后半段再评估
  对 DETR 家族做 opt-in 的 `export_postprocess=True`。这一点作为 ADR D2
  记录（见第 8 节）。

#### 4.3.3 OpenVINO 运行时

- `InferenceEngine._BACKENDS` 增加 `"openvino"`；`artifact` 接受 `.xml`。
- `_make_openvino_forward` 复用 `OpenVINOEngineRunner`（已存在），
  设备参数 `openvino_device="CPU" | "GPU" | "AUTO"`。
- `serving.create_app(backend="openvino", artifact="model.xml")` 可用；
  `/healthz` 增加 `device`。
- CLI `predict backend=openvino artifact=model.xml` 可用；`benchmark` 已支持
  OpenVINO，不需要改。
- 测试：
  - tests/test_openvino_runtime.py：CPU 上 tiny 模型 `InferenceEngine`
    与 torch 最终框一致；
  - 无 openvino 包时给出安装指引（跳过而非崩溃）；
  - serving 的 `/predict` 用 OV 后端跑一张 base64 图。

#### 4.3.4 int8 精度验收

新增 scripts/measure_int8_accuracy.py 或 `benchmark --accuracy`：

1. 固定校准集：COCO train2017 的 200 张图（与 val 不重叠），记录文件清单
   与种子；
2. 导出 fp32 ONNX → ORT 静态 PTQ → int8 ONNX（现有能力）；
3. 用同一个 `evaluate_object_detection` 在 COCO val2017（全量或固定 5000
   子集）上分别跑 fp32 与 int8，输出 AP / AP50 / 每类 AP；
4. 生成 `int8_accuracy.json` + markdown 表：模型、后端、校准集、AP、delta、
   硬件、日期；
5. 发布门禁建议：检测 AP 下降 ≤ 1.0（可配置）；超过则阻止模型卡标
   “int8 verified”，但不阻止 fp32 发布；
6. 把 `coco_map_int8` 写进 `ModelCard.metrics`，模型库显示 delta。

注意：int8 的 parity 容差（atol 2.0/rtol 0.5）只证明“张量没炸”，AP 才是
验收标准；两者都保留，但叙事以 AP 为准。

#### 4.3.5 多路视频

新增 inference/multi_stream.py：

```python
@dataclass(frozen=True)
class StreamSource:
    name: str
    url: str                  # 文件 / RTSP / HTTP
    reconnect: bool = True
    max_reconnects: int = 10

class MultiStreamRunner:
    def __init__(self, sources, engine, out_dir, *,
                 batch_size=4, queue_size=8, drop_policy="drop_oldest",
                 render=True, track=False, metrics_interval=5.0): ...
    def run(self, max_seconds=None) -> MultiStreamResult: ...
```

架构：

- **每路一个 reader 线程**：OpenCV `VideoCapture` + 超时；读到的帧带
  `(source, frame_index, timestamp)` 放入有界队列；断流按指数退避重连
  （1s → 30s，带 jitter），超过 `max_reconnects` 标记该路失败。
- **一个中央 batcher 线程**：从各队列取帧，凑到 `batch_size` 或超时
  （例如 20ms）后调用 `engine.run_batch`；结果按来源分发。这样多路共享
  一次前向，吞吐不被单路帧率限制。
- **每路一个 writer**：`out/<source>/predictions.jsonl` +
  `annotated.mp4`；写盘不阻塞推理（队列满时丢最旧帧，计数）。
- **指标**：每路 `frames_read / frames_dropped / reconnects / fps /
  inference_latency_ms / end_to_end_latency_ms`，全局汇总；
  结束时写 `summary.json` + 控制台表。
- **背压**：`queue_size` 有界；`drop_policy="drop_oldest" | "block"`；
  默认丢最旧，保证实时性；离线批处理用 `block`。
- **线程安全**：`InferenceEngine` 的 forward 只在 batcher 线程调用；
  postprocess 可在 batcher 线程或 per-source writer 做，但必须保证
  engine 不被并发调用（TRT/OV runner 通常非线程安全）。
- **CLI**：`predict sources=rtsp://a,rtsp://b out=out/ ...`；或
  `sources=streams.yaml`。保留单路 `source=` 不变。
- **测试**：复用现有进程内 MJPEG/HTTP 源；2–4 路跑 30 帧；模拟一路断开，
  断言重连计数与其余路不中断；断言每路 JSONL 行数与帧数一致；无 GPU。

#### 4.3.6 C++ 参考运行时

新增 `examples/cpp/`：

```text
examples/cpp/
  CMakeLists.txt
  README.md
  src/main.cpp
  src/postprocess.hpp
  testdata/
```

- 依赖：ONNX Runtime C++（MIT）+ OpenCV（Apache-2.0）；不引入 TensorRT/OV
  的 C++ 依赖，先做 ONNX Runtime 一条。
- `main.cpp`：加载 `.onnx` + `.postprocess.json` + 图片，执行
  resize/normalize → ORT session → 参考后处理 → 输出 JSON（框/分数/类别）。
- `postprocess.hpp`：按 spec 实现 DETR 家族的 decode；与 Python 参考实现
  共享同一份 spec 与 golden vectors。
- 测试：
  - CI 可选 job 安装 ONNX Runtime + OpenCV，构建并跑 1 张图；
  - 断言输出与 Python 参考实现的框差 ≤ 1px、类别一致；
  - 无依赖环境跳过（`AUTOVISION_CPP_TESTS=1` 门控）。
- 这不是 SDK，不承诺 ABI 稳定；文档明确“参考实现，按需裁剪”。

#### 4.3.7 预处理下沉（可选，按读数决定）

现有基准显示 `detr/n` 的 TRT b1 全链 13.06ms 中 engine 只有 2.70ms，
约 10ms 是 CPU resize/normalize。阶段 2 的验收不强制解决，但给出两条路径：

- **GPU 预处理**：`preprocess_image` 的 resize/normalize 在 CUDA 上做，
  减少 H2D 前后的 CPU 时间；需要保证 letterbox 语义一致。
- **图内预处理**：导出时把 resize/normalize 融进 ONNX（`preprocess.fused`），
  设备侧只做解码 + H2D。风险是不同后端的 Resize 算子差异，必须先过端到端
  parity。
- 建议先测 GPU 预处理；只有在目标硬件上确实收益 > 30% 才做图内融合。

### 4.4 实施步骤（WBS）

| # | 工作 | 产出 | 人日 |
|---|---|---|---|
| 2.1 | 后处理 spec + NumPy 参考实现 + 与 torch 对拍 | 模块 + 测试 | 4 |
| 2.2 | 导出件写 spec；`load_spec/decode` API | 导出管线 + 测试 | 2 |
| 2.3 | 端到端 parity + golden vectors + 报告 | 模块 + 测试 | 4 |
| 2.4 | OpenVINO 接 `InferenceEngine`/serving/CLI | 后端 + 测试 | 2 |
| 2.5 | int8 精度脚本 + AP 表 + 门禁 | 脚本 + 读数 | 4 |
| 2.6 | 多路流 runner + CLI + 重连/背压 + 测试 | 模块 + 测试 | 8 |
| 2.7 | C++ 示例 + golden 对拍 + 可选 CI | 示例 + 测试 | 5 |
| 2.8 | 文档：导出/部署/多路/ C++ 指南 | 文档 | 2 |
| 2.9 | 端到端报告与 README 证据表 | 报告 | 1 |

总计约 **32 人日**（预处理下沉另计 3–5 人日，按读数决定）。

### 4.5 验收门禁

```bash
# 后处理与端到端 parity
pytest tests/test_postprocess_spec.py -q
pytest tests/test_e2e_parity.py -q

# OpenVINO 运行时
pytest tests/test_openvino_runtime.py -q

# int8 AP（GPU + 数据集门控）
AUTOVISION_TRT_TESTS=1 python scripts/measure_int8_accuracy.py \
    --model runs/best.pt --data coco.yaml --calibration-images calib/ \
    --out bench/int8/ --max-images 5000

# 多路视频（CPU/本地文件）
pytest tests/test_multi_stream.py -q

# C++（可选依赖）
AUTOVISION_CPP_TESTS=1 pytest tests/test_cpp_reference.py -q
```

完成判据：

1. 同一张图在 torch / ONNX / OpenVINO / TRT 上，参考后处理后的最终框
   最大坐标差 ≤ 1px、类别一致；报告落盘；
2. 两个模型的 int8 AP 下降表存在，且不超过约定阈值（或明确记录超标）；
3. 4 路视频 30 帧稳定跑；断开一路后自动重连，其余路不中断；每路有
   JSONL 与指标；
4. C++ 示例在 CI 可选 job 中构建并匹配 golden vectors。

### 4.6 风险

| 风险 | 处置 |
|---|---|
| 后处理规格与任务模型漂移 | spec 由导出管线生成；单一 `decode` 实现；对拍测试锁定 |
| 不同后端的 letterbox/Resize 差异 | spec 显式记录 preprocess；端到端 parity 比较最终框 |
| OpenVINO CPU 长尾延迟 | 暴露 `PERFORMANCE_HINT` / 线程配置；记录 p50/p90，不用单次读数 |
| int8 AP 下降超阈值 | 允许 fp32 发布；int8 标“未 verified”；给出校准集/排除层建议 |
| 多路流线程/队列复杂 | 先做“reader 线程 + 单 batcher”最小架构；不做每路独立推理 |
| C++ 依赖矩阵重 | 只做 ONNX Runtime + OpenCV 一条；CI optional；不承诺 SDK |

---

## 5. 阶段 3：数据闭环（90 天）

### 5.1 目标

让客户在“数据少、标注贵、长尾差”时有一条可执行的闭环：

```text
数据体检 → 训练 → 误差分析 → 难例清单 → 重采样/多轮伪标注 → 再训练
```

每个箭头都有命令、产物和可复现报告。

### 5.2 现状

- 伪标注 / 自训练单轮；误差分析有报告与画廊，但没有机器可读的难例清单，
  也不回灌训练。
- 没有数据体检；坏图、重复图、框越界、split 泄漏、类别分布都没有系统检查。
- 没有标注工具互通；客户要从 CVAT / Label Studio 手工导。
- 没有长尾采样 / 重加权；只有伪标注的逐类阈值。
- 数据快照已存在，但训练不自动绑定。

### 5.3 设计

#### 5.3.1 数据体检（data-check）

新增 data/quality.py + CLI `data-check`：

```bash
autovision-cli data-check data=my_data.yaml out=qa/ \
    [near_duplicate=true] [fail_on=error|warning] [max_examples=20]
```

检查项：

| 类别 | 检查 | 严重级别 |
|---|---|---|
| 图像 | 无法解码 / 0 字节 / 截断 / 通道异常 / 尺寸过小或过大 | error / warning |
| 重复 | 精确 SHA256 重复；pHash 近重复（Hamming ≤ 8） | warning / error（跨 split 为 error） |
| 标注 | `x2<=x1`、`y2<=y1`、NaN/Inf、越界（容差可配）、面积过小、重复框 | error |
| 类别 | 类别 id 越界、空 split、验证集缺类、类别极度不平衡 | error / warning |
| 泄漏 | 同一图/近重复图同时出现在 train 与 val/test | error |
| 分布 | 每类计数、imbalance ratio、图像尺寸分布 | info |
| 关键点/分割 | 可见性、多边形有效性、mask 覆盖率（先检测，后续扩展） | warning |

输出：

```text
qa/
  report.json      # 机器可读：summary / issues[] / per_class / per_split
  report.md        # 人读：汇总表 + top 问题文件 + 处置建议
  manifest.json    # 问题文件清单（路径 + 问题类型 + 严重级别）
```

- 结果确定性排序（按 split、路径、问题类型），便于对比与 CI。
- `fail_on=error`（默认）时发现问题退出码 1；`warning` 不阻断，但报告里
  标红。
- 训练前可选自动跑：`trainer_args.data_check=true`（默认关闭，发布/大客户
  建议开启）；报告摘要写入 run 目录。
- 测试：合成 fixture（坏图、重复图、越界框、泄漏）断言每类问题都被抓到；
  coco8 应通过。

#### 5.3.2 难例清单与采样

**清单格式（`hard_examples.json`）：**

```json
{
  "schema_version": 1,
  "source_run": "runs/det1",
  "iou_threshold": 0.5,
  "score_threshold": 0.3,
  "images": [
    {
      "image": "train/0001.jpg",
      "split": "train",
      "tp": 3, "fp": 2, "fn": 5,
      "fp_reasons": {"localization": 1, "background": 1},
      "max_fn_area": 4096.0,
      "classes_fn": [1, 7]
    }
  ]
}
```

- 由 `metrics/error_analysis.py` 的离线分析产出；复用现有匹配结果，不重新
  推理。
- 新增 data/hard_examples.py：
  - `HardExampleManifest.load(path)` 校验 schema；
  - `HardExampleSampler(dataset_size, manifest, *, alpha=1.0, max_repeat=5,
    seed=0)`：权重 = `1 + alpha * normalized_error`，误差由 FN/FP 数与
    漏检面积归一化；`max_repeat` 防止过采样；
  - 只对训练 split 生效；val 不采样。
- 训练入口：`train_object_detection(..., hard_examples="hard.json")`；在
  数据加载器构建时替换 sampler。配置字段放在 detection 数据参数或
  `trainer_args`，需过 Pydantic 校验。
- 测试：分布（高误差图被采样更多）、可复现（同 seed 同序列）、2 步训练
  闭环、val 不受影响。
- 报告：训练结束写 `hard_examples_used.json`（权重摘要 + 采样次数），
  模型卡可记录。

#### 5.3.3 多轮伪标注

把 `scripts/self_train_detect.py` 从“单轮”扩展为“可编排多轮”：

```bash
python scripts/self_train_detect.py \
    --teacher runs/teacher/exported_models/exported_best.pt \
    --images unlabeled/ --val-data val.yaml --out runs/st \
    --rounds 3 --student rfdetr/rf-detr-small \
    --threshold 0.5 --min-improvement 0.5
```

轮次逻辑：

1. 教师对无标注池推理 → 伪标签；
2. 与已标注训练集合并（可选）→ 训练学生；
3. 在标注 val 上评测；
4. 若 AP 提升 ≥ `min_improvement`，学生升级为下一轮教师；否则停止；
5. 每轮写 `rounds/r{i}/`（pseudo / student / report），顶层写
   `summary.md` + `summary.json`。

约束：

- 教师选择必须显式：`--teacher-strategy best|last|ema`；默认 `best`；
- 每轮记录伪标签统计（逐类计数、分数分布）与学生指标；
- 伪标签为空直接失败（已有行为）；
- 不做自动“无限自我训练”；默认最多 3 轮，AP 不升即停；
- 测试用 fake teacher/student（monkeypatch 训练入口）验证轮次、停止条件、
  报告字段；不在 CI 跑真实多轮训练。

#### 5.3.4 标注工具互通

新增 `data/interchange/`：

```text
data/interchange/
  cvat.py           # CVAT for Images 1.1 XML <-> COCO
  label_studio.py   # Label Studio JSON -> COCO（先单向）
  schema.py         # 统一的中间结构（image/annotation/attribute）
```

- CVAT：支持 bbox、polygon；坐标已是像素；类别名映射到 COCO `categories`；
  支持 `difficult`/`occluded` 属性到 `iscrowd` 或自定义字段。
- Label Studio：支持 `rectanglelabels`、`polygonlabels`、`keypointlabels`；
  百分比坐标转像素；支持 `annotations` 与 `predictions` 两种来源。
- CLI：`convert-data from=cvat --cvat-xml ...`、
  `from=label-studio --ls-json ...`；与现有 `convert-data` 共用输出 schema。
- 测试：仓库内 fixture（小 XML/JSON）→ 转换 → 用现有 COCO loader 读回；
  框坐标误差 ≤ 1e-6；类别名一致；空标注/坏坐标报错。
- 不做 UI、不做服务端 API；只做文件互通。

#### 5.3.5 长尾采样

新增 data/samplers.py：

- `ClassBalancedSampler`：按类别频率的 `1/freq^power` 采样，`power` 可配
  （0=均匀，1=完全平衡）；对多标签图像按最大权重或平均权重。
- `RepeatFactorSampler`：LVIS 风格，类别频率低于阈值 `t` 的重复因子为
  `sqrt(t/freq)`，上限 `max_repeat`。
- 训练入口：`trainer_args.sampler={type: "class_balanced", power: 0.5}`；
  与 `hard_examples` 互斥或组合（先做互斥，避免权重语义混乱）。
- 测试：采样后的类别分布接近目标；同 seed 可复现；2 步训练。
- 损失重加权作为后续项：需要改多个检测头的 criterion，先不做；用采样器
  解决大部分长尾问题。

#### 5.3.6 训练自动绑定数据快照

- 训练入口默认写浅快照（路径 + 大小 + mtime 清单），发布/大客户可开
  `deep=true`（内容哈希）。
- 快照 digest 写入 `run_config.json` / `metrics.json`，并在模型卡的
  `provenance` 中记录。
- `compliance-report`（阶段 4）读取该 digest。
- 测试：训练 2 步后快照文件存在、digest 与手动快照一致；快照校验失败时
  训练继续但告警（或按配置 fail-fast）。

### 5.4 实施步骤（WBS）

| # | 工作 | 产出 | 人日 |
|---|---|---|---|
| 3.1 | data/quality.py + `data-check` CLI + 报告 schema | 模块 + 测试 | 5 |
| 3.2 | 难例清单 schema + error_analysis 导出 | 模块 + 测试 | 2 |
| 3.3 | `HardExampleSampler` + 训练接入 + 测试 | 模块 + 测试 | 3 |
| 3.4 | 多轮自训练编排 + 报告 + 停止条件 | 脚本 + 测试 | 3 |
| 3.5 | CVAT / Label Studio 转换 + fixtures | 模块 + 测试 | 5 |
| 3.6 | 长尾 sampler + 训练接入 + 测试 | 模块 + 测试 | 3 |
| 3.7 | 训练自动绑定快照 + 模型卡 provenance | 模块 + 测试 | 2 |
| 3.8 | 端到端案例（1k 标注 + 100k 无标注） | 报告 + 复现脚本 | 5 |
| 3.9 | 文档：数据闭环手册 | 文档 | 2 |

总计约 **30 人日**。

### 5.5 验收门禁

```bash
pytest tests/test_data_quality.py -q
pytest tests/test_hard_examples.py -q
pytest tests/test_self_train_rounds.py -q
pytest tests/test_interchange.py -q
pytest tests/test_samplers.py -q

# 案例复现（数据/GPU 门控）
python scripts/self_train_detect.py --rounds 2 ... --report runs/case/report.json
autovision-cli analyze-errors ... out=runs/case/errors/
autovision-cli train-object-detection ... hard_examples=runs/case/errors/hard_examples.json
```

完成判据：

1. `data-check` 在缺陷 fixture 上逐类报出问题，在 coco8 上通过；
2. 难例清单可被 sampler 消费；训练 2 步成功；采样分布可解释；
3. 多轮脚本能跑 2 轮并产出 `summary.md`；不升即停；
4. CVAT / Label Studio fixture 往返读回一致；
5. 1k 标注 + 100k 无标注案例：第二轮 AP 相对第一轮有可复现变化（正向或
   明确记录的负向结果），报告含数据体检、伪标签统计、每轮指标；
6. 训练产物记录数据快照 digest。

### 5.6 风险

| 风险 | 处置 |
|---|---|
| 数据体检误报（近重复、框越界容差） | 分级 error/warning；`fail_on` 可配；报告给样例图 |
| pHash 近重复在大数据集上慢 | 先精确哈希分桶，再对候选做 pHash；支持采样/分片 |
| 难例过采样导致过拟合 | `max_repeat` 上限；val 不采样；训练结束对比 baseline |
| 多轮伪标注确认偏差 | 最多 3 轮；每轮在标注 val 上评测；不升即停；逐类阈值 |
| 标注格式变体多 | 先覆盖主流 bbox/polygon/keypoint；未知字段保留并警告 |
| 长尾 sampler 改变训练分布 | 默认关闭；配置显式开启；报告采样分布 |

---

## 6. 阶段 4：生态与商业（6 个月）

### 6.1 目标

把“许可干净 + 数值可验证”变成可被外部使用和采购的生态：

- 外部开发者能加 backbone / dataset / metric，不改核心；
- 模型库有可验证的“已发布/已验证”条目，不只是第三方表；
- 企业能拿到合规包、私有注册表、部署门禁、LTS/SLA；
- 至少一个硬件伙伴包，而不是 15 个后端的矩阵。

### 6.2 现状

- `register_source` 存在，但没有 entry-point 发现、没有插件模板、没有
  head/loss/metric/dataset 扩展点。
- 模型库由 `scripts/gen_model_zoo.py` 生成，但主角是第三方权重；没有
  “verified” 状态。
- 注册表有不可变版本与晋级，但无回滚、无 current、无门禁、无审计、
  未接 serving。
- 没有合规报告生成器；`NOTICE` 与资产注册表是原始材料。
- 没有 LTS/SLA 政策；没有硬件伙伴包。

### 6.3 设计

#### 6.3.1 插件协议

新增 `autovision/plugins.py`：

```python
PLUGIN_API_VERSION = 1

@dataclass(frozen=True)
class PluginInfo:
    name: str
    version: str
    kind: Literal["backbone", "dataset", "metric", "loss", "task", "postprocess"]
    api_version: int
    factory: Callable[..., Any]

def discover_plugins(*, strict=False) -> list[PluginInfo]: ...
def register_plugin(info: PluginInfo) -> None: ...
```

- 发现机制：`importlib.metadata.entry_points(group="autovision.plugins")`；
  每个 entry point 返回一个 `PluginInfo` 或 `register(api)` 函数。
- 注册函数：
  - `register_backbone(name, factory, *, arch_traits=None)`：包装现有
    `ModelSource` / `CatalogModelSource`；
  - `register_dataset(name, factory)`：工厂返回带 `list_image_info` 的
    数据参数对象；
  - `register_metric(name, factory)`：返回 `update/compute` 协议；
  - `register_loss(name, factory)`：用于任务级 criterion 注入；
  - `register_task(name, factory)`：高级扩展，要求实现
    `TrainModel` / `TaskNetwork` 契约；
  - `register_postprocess(name, factory)`：阶段 2 的 spec 解码器扩展。
- API 版本：`PLUGIN_API_VERSION` 写进 `PluginInfo`；不兼容时拒绝加载并
  给出明确错误。插件 API 从 1.0 起按 SemVer。
- 发现策略：默认自动发现 entry points；`AUTOVISION_PLUGINS=off` 关闭；
  `strict=True` 时任一插件失败即报错，默认收集错误并继续（列出失败的
  插件名）。
- 安全：插件是本地代码，等同依赖；不提供远程加载；文档写明信任边界。
- 模板：`examples/plugin-demo/`，包含：
  - `pyproject.toml` 的 `[project.entry-points."autovision.plugins"]`；
  - 一个自定义 backbone（包装现有 torchvision 模型）；
  - 一个自定义 dataset（读 CSV 清单）；
  - 一个自定义 metric（top-k）；
  - 测试 + README + “如何发布到 PyPI”。
- CLI：`autovision-cli plugins list`（名称、版本、kind、API 版本）；
  `plugins check`（加载诊断）。
- 测试：用 monkeypatch 伪造 entry point；再用 `PYTHONPATH` 子进程安装
  模板包跑一次 `plugins list`；外部贡献者验收以模板为准。

#### 6.3.2 模型库榜单与 verified

- registry/catalog.py 增加 `verified: bool` 与 `verified_reasons: list[str]`。
- `verified` 的机器判据（全部满足）：
  1. `ModelCard.validate()` 通过；
  2. 权重 sha256 与注册表一致；
  3. 全量 COCO val2017 AP（或任务对应全量指标）已记录；
  4. 至少一条延迟记录，脚本存在且可复跑；
  5. 许可在 allowlist 内；
  6. 至少一个后端做过端到端 parity；
  7. 模型键可一行加载。
- `scripts/gen_model_zoo.py` 生成：
  - “已发布 / 已验证”表：模型键、任务、许可、AP、Params/GFLOPs、延迟、
    后端、可导出、卡片、加载代码；
  - “可用但未验证”表：第三方权重、仅有官方读数；
  - 榜单按任务/硬件分表；数字来源与协议链接到 [基准](../benchmarks.md)。
- docs/leaderboard.md 加入导航；只展示 verified 条目，避免“第三方官方值
  混进自家榜”。
- 测试：`gen_model_zoo.py --check`；每个 verified 条目必须有卡片、sha256、
  AP、延迟脚本；缺一项生成器失败。

#### 6.3.3 企业合规包

新增 `autovision/compliance/`：

```text
compliance/
  inventory.py     # 框架/依赖/模型/数据四类清单
  report.py        # compliance.json + compliance.md
  licenses.py      # SPDX 映射 + allowlist/denylist
```

CLI：

```bash
autovision-cli compliance-report \
    --model runs/ft/exported_models/exported_best.pt \
    --data my_data.yaml --snapshot qa/data.snap.json \
    --out compliance/ [--no-timestamp]
```

报告内容：

- **框架**：版本、commit、Python/torch/平台、安装 extras；
- **依赖 SBOM**：已安装包 + 版本 + 许可（`importlib.metadata` + 策展
  SPDX 映射；未知许可标 `UNKNOWN`，不静默放过）；
- **模型**：模型键、权重 sha256、模型卡、许可、训练配置 hash、教师/暖启
  来源；
- **数据**：数据快照 digest、split 统计、data-check 摘要；
- **一致性**：导出后端、parity 报告、int8 AP delta；
- **已知限制**：模型卡中的 `known_limitations`；
- **审批清单**：可勾选的合规项（再分发、商用、出口管制、隐私）。

实现约束：

- 确定性输出：排序键、固定小数位；`--no-timestamp` 可复现；
- 不联网；不读取受限权重内容，只读元数据与哈希；
- 不重复许可判断：复用 `NOTICE`/资产注册表的许可字段与 allowlist；
- 报告是“证据汇编”，不是法律意见；文档写明。
- 测试：fixture 模型 + 数据；断言 JSON schema、许可字段、sha256、缺失
  资产报错；golden 文件对比。

**开源/商业边界建议：** 报告生成器开源（建立信任），企业版卖“审计服务、
私有注册表、部署门禁、LTS/SLA、目标硬件适配”。不要给报告生成器加人为
限制。

#### 6.3.4 私有注册表与部署门禁

扩展现有 `ModelRegistry`：

```python
registry.register(model_file, name=..., version=..., stage="dev")
registry.promote(name=..., version=..., to_stage="staging", gates=[...])
registry.rollback(name=..., stage="prod", to_version="1.2.0", reason=...)
registry.current(name=..., stage="prod")      # -> RegistryEntry
registry.audit_log()                          # -> list[dict]
```

设计：

- **不可变**：版本目录只增不改；晋级复制并校验 sha256。
- **current 指针**：`<stage>/<name>/current.json` 记录当前版本；promote /
  rollback 原子更新（临时文件 + rename）。
- **审计日志**：`<root>/audit.jsonl` 追加 `register/promote/rollback`，
  含时间、操作者（环境变量/CLI 参数）、版本、sha256、原因；不修改历史。
- **部署门禁**：`DeploymentGate` 协议：
  ```python
  class DeploymentGate(Protocol):
      def check(self, entry: RegistryEntry, context: dict) -> list[str]: ...
  ```
  内置门禁：
  - `Sha256Gate`：文件哈希与 manifest 一致；
  - `ModelCardGate`：卡片存在且 `validate()` 通过；
  - `LicenseGate`：许可在 allowlist；
  - `ParityGate`：端到端 parity 报告存在且通过；
  - `BenchmarkGate`：全量指标 + 延迟记录存在；
  - `DataSnapshotGate`：数据快照 digest 存在且校验通过。
- **serving 集成**：`autovision-cli serve model=registry://prod/detector`
  解析 current 版本并加载；`/healthz` 返回版本、sha256、stage。
- **CLI**：`registry-register`、`registry-promote`、`registry-rollback`、
  `registry-list`、`registry-current`；或统一 `registry <verb>`。命令名在
  实现时定，保持 kebab-case。
- **测试**：晋级/回滚/current/审计；门禁失败不改变 current；serving 读到
  正确版本；并发 promote 的原子性（文件锁）。

#### 6.3.5 LTS 与 SLA

文档（不是代码）：

- docs/lts_policy.md：每 6 个月一个 LTS（如 0.3 LTS、0.6 LTS），
  12 个月安全修复 + 6 个月完整支持；LTS 只接受 bug/security 修复，
  不接受新功能；backport 标准；EOL 公告提前 3 个月。
- docs/support_policy.md：社区支持（GitHub issue，工作日 48h 内响应）、
  企业支持（SLA、私有渠道、版本锁定、部署协助）；支持矩阵
  （Python / torch / CUDA / 后端 / OS）。
- `docs/release_process.md` 增加：发布节奏（月度 MINOR/PATCH）、LTS 分支
  规则、CVE 流程、yank 流程。
- 仓库机制：`release/0.3` 分支；tag `v0.3.x`；CI 在 LTS 分支跑套件；
  CHANGELOG 的 `Security` 段。
- 验收：政策页上线；第一个 LTS 分支/tag 创建；CI 绿。

#### 6.3.6 硬件伙伴包

`deploy/packs/<vendor>/`：

```text
deploy/packs/rknn/
  README.md          # 目标芯片、SDK 版本、许可、限制
  convert.py         # ONNX -> 厂商模型
  postprocess.json   # 与阶段 2 同一份 spec
  parity_vectors/    # 20 张图的 golden 输出
  benchmark.md       # p50/p90、功耗（如可得）、硬件
```

- 统一接口：输入是阶段 2 的 ONNX + postprocess spec；输出是目标端可运行
  模型 + 参考后处理。
- 每个包必须给：转换命令、目标端运行示例、20 张图的 parity 结果、
  延迟/内存、许可说明。
- 只在客户/伙伴明确选型后做；不预研所有厂商；一个伙伴一个包。
- 验收：目标端最终框与参考实现一致（容差按平台）；延迟读数可复现；
  许可审计通过。

### 6.4 实施步骤（WBS）

| # | 工作 | 产出 | 人日 |
|---|---|---|---|
| 4.1 | 插件协议 + entry-point 发现 + 模板 | 模块 + 示例 + 测试 | 8 |
| 4.2 | catalog verified 状态 + 模型库/榜单生成 | 脚本 + 文档 | 5 |
| 4.3 | 合规包（inventory/report/licenses） | 模块 + CLI + 测试 | 8 |
| 4.4 | 注册表回滚/current/审计/门禁 | 模块 + CLI + 测试 | 8 |
| 4.5 | serving 接 registry:// | 模块 + 测试 | 2 |
| 4.6 | LTS/SLA 政策 + 首个 LTS 分支 | 文档 + CI | 3 |
| 4.7 | 一个硬件伙伴包（按客户） | 包 + parity + 读数 | 10 |
| 4.8 | 文档/社区：英文指南、示例库、发布节奏 | 文档 | 5 |

总计约 **49 人日**（不含伙伴硬件排期）。

### 6.5 验收门禁

```bash
# 插件
pytest tests/test_plugins.py -q
autovision-cli plugins list

# 模型库/榜单
python scripts/gen_model_zoo.py --check
pytest tests/test_model_zoo_verified.py -q

# 合规包
pytest tests/test_compliance.py -q
autovision-cli compliance-report --model ... --data ... --out compliance/ --no-timestamp

# 注册表
pytest tests/test_registry_gates.py -q
autovision-cli registry promote <name> <version> --to prod

# LTS / 文档
python scripts/check_docs_links.py && mkdocs build --strict
```

完成判据：

1. 外部贡献者按模板加一个 backbone/dataset/metric，核心零改动；
2. 模型库有 ≥5 个 verified 条目（RF-DETR N/S/M/L + GTR 至少一档），
   每项有卡片、AP、延迟、sha256；
3. 合规报告可为一次发布生成，包含 SBOM、许可、模型/数据 provenance；
4. 注册表支持 promote/rollback/current/审计，serving 可解析
   `registry://prod/...`；
5. LTS 政策与首个 LTS 分支上线；
6. 一个硬件伙伴包完成目标端 parity。

### 6.6 风险

| 风险 | 处置 |
|---|---|
| 插件 API 过早冻结 | 1.0 前标 experimental；只承诺 `PLUGIN_API_VERSION` 兼容 |
| 合规报告许可映射不全 | 未知许可显式 `UNKNOWN`；allowlist 评审；不静默 |
| 私有注册表安全 | 文件权限 + 哈希校验 + 审计日志；签名/HSM 作为企业版 |
| 企业功能侵蚀开源 | 开源报告/注册表基础能力；企业卖审计、SLA、硬件包 |
| 硬件伙伴排期不可控 | 按客户触发；一个伙伴一个包；不预研矩阵 |
| LTS 维护成本 | 只 backport bug/security；明确 EOL；CI 自动化 |

---

## 7. 跨阶段工程约定

### 7.1 分支与版本

| 版本 | 触发 | 内容 |
|---|---|---|
| 0.3.0 | 阶段 0 | 发布入口、quickstart、英文 README |
| 0.4.0 | 阶段 1 | 预训练句柄、模型目录、RF-DETR 卡片、GTR AP、recipes |
| 0.5.0 | 阶段 2 | 后处理 spec、端到端 parity、OV 运行时、int8 AP、多路流 |
| 0.6.0 | 阶段 3 | data-check、难例采样、多轮伪标注、互通、长尾 |
| 1.0.0 | 阶段 4 后 | 公共 API 稳定、插件 API 1.0、LTS |

- 公共 API 变更走弃用政策：MINOR 标注 + 运行时警告，至少跨一个 MINOR
  再移除。
- 模型版本与框架版本独立；权重发布用独立 tag / sha256 / 模型卡。
- 发布快照按 [发布流程](../release_process.md)；LTS 分支从对应 tag 拉出。

### 7.2 测试矩阵

| 层级 | 内容 | CI |
|---|---|---|
| 单元 | 新增模块的行为、边界、错误 | 每次 PR |
| 集成 | 2 步训练、导出、流、注册表、插件 | 每次 PR（CPU） |
| 数值对拍 | 参考后处理 vs torch、端到端 parity、int8 AP | PR（小模型）+ nightly（真实权重） |
| 网络/GPU | HF 权重、GTR、TRT、多路 RTSP | nightly / 门控环境变量 |
| C++ | 构建 + golden 对拍 | optional job |
| 文档 | 链接、mkdocs strict、模型库零漂移 | 每次 PR |

新增测试命名建议：
- tests/test_pretrained_detection.py
- tests/test_model_catalog.py
- tests/test_recipes.py
- tests/test_postprocess_spec.py
- tests/test_e2e_parity.py
- tests/test_openvino_runtime.py
- tests/test_multi_stream.py
- tests/test_data_quality.py
- tests/test_hard_examples.py
- tests/test_self_train_rounds.py
- tests/test_interchange.py
- tests/test_samplers.py
- tests/test_plugins.py
- tests/test_compliance.py
- tests/test_registry_gates.py

> 注：以上为规划文件名；实现时按仓库现有测试命名习惯落地，避免与已有
> `test_deployment.py` / `test_infrastructure.py` 的职责重叠。

### 7.3 文档与 CI 门禁

每个阶段新增/修改文档必须：

1. 注册到 `mkdocs.yml` 导航，或被 `exclude_docs` 显式排除；
2. 相对链接可解析（`scripts/check_docs_links.py`）；
3. 模型库/基准由生成器或脚本更新（`gen_model_zoo.py --check`）；
4. 新增 CLI/API 有示例，示例由测试或 quickstart CI 覆盖；
5. 数字必须带口径、硬件、脚本路径；不把第三方官方值混进自家表。

### 7.4 指标与运营

| 指标 | 定义 | 目标 |
|---|---|---|
| time-to-first-result | 看到项目 → 自己的图出框 | ≤ 10 分钟 |
| time-to-first-export | 训练完 → 通过 parity 的 ONNX | ≤ 10 分钟 |
| time-to-first-finetune | 100 张图 → 可评测微调 | ≤ 30 分钟 |
| issue 首次响应 | 工作日中位数 | ≤ 48h |
| 模型下载量 | PyPI / HF / 资产缓存 | 月度增长 |
| POC 数 | 企业试用 | 阶段 4 内 3 个 |
| 回归门禁通过率 | PR CI | ≥ 99% |

运营节奏：每月一次 release note；每两周一次 issue triage；每季度一次
roadmap 复盘；LTS 每 6 个月。

### 7.5 人力与排期

- 阶段 0：1 周，2 人并行（发布工程 + 文档）。
- 阶段 1：4 周，2 人（模型/API + 文档/评测）。
- 阶段 2：8 周，2 人（部署工程为主，数据/文档 0.5）。
- 阶段 3：8–10 周，1.5 人（数据算法 + 训练集成），可与阶段 2 并行。
- 阶段 4：6 个月，2 人 + 0.5 社区；硬件伙伴按客户。

关键路径：阶段 0 → 阶段 1 的预训练句柄 → 阶段 2 的后处理/parity。
数据闭环与插件可以并行，但不要阻塞发布。

---

## 8. 决策记录（ADR）

| ID | 决策 | 理由 | 替代方案（未选） |
|---|---|---|---|
| D1 | **双旗舰**：RF-DETR 可部署，GTR 高精度 | 前者 Apache-2.0、权重现成、可导出；后者 MIT、AP 强、s/m/l/x 齐全但不可导出 | 只推 GTR（部署叙事断裂）；只推自研（无权重，发布被训练阻塞） |
| D2 | **后处理参考实现优先，入图后置** | 跨后端 NMS/Resize 支持不一致；版本化 spec + golden vectors 更可移植 | 直接把 NMS 导进图（后端分叉、调试困难） |
| D3 | **OpenVINO 做一等运行时** | Runner 已存在，CPU 装机量大；接线成本低 | 只做导出、不做运行时（用户仍要自己写前后处理） |
| D4 | **自研权重后置，不等训练** | 全量 COCO 训练周期长；先用干净第三方权重建采用漏斗 | 等 `detr/n/s` 发布再开门（延迟数月） |
| D5 | **企业版边界：开源报告/注册表基础，付费审计/SLA/硬件包** | 建立信任；不制造人为限制；企业为确定性与支持付费 | 把合规报告做成闭源（削弱“可审计”叙事） |
| D6 | **不做 15 个后端、不做标注 UI、不做生产 serving 网关** | 保持维护面；ONNX hub + 3 个一等后端 + 按客户边缘包 | Ultralytics 式矩阵（维护成本不可控） |
| D7 | **暂停 n/s/m/l/x 作为营销命名，直到体量拉开** | 当前 n/s 约 59–62 GFLOPs，不是边缘档；避免被 YOLO11n 对比击穿 | 继续用 n 命名（短期叙事、长期失信） |

---

## 9. 开放问题

1. **RF-DETR 原生类别数**：COCO 检查点的 `num_labels` 是 80 还是 91？
   需要以实际 config 为准，并锁定类别名映射测试。
2. **GTR 全量 AP 的硬件预算**：四档全量 val2017 需要多少 GPU 小时？
   是否先跑 s/m 两档，l/x 夜间？
3. **`detr/n` 是否重做真正 nano**：5–10 GFLOPs 需要新 backbone/decoder
   配方；是否值得，取决于边缘客户需求。短期用 RF-DETR Nano 覆盖。
4. **英文文档范围**：只做 README + quickstart + model zoo，还是全量翻译？
   建议先前者，按流量决定。
5. **int8 AP 阈值**：检测 AP 下降 ≤ 1.0 是否适用于所有家族？分割/姿态
   需要单独阈值。
6. **企业版形态**：私有注册表/部署门禁是开源还是企业？建议基础开源、
   审计/SLA/硬件包付费。
7. **硬件伙伴选择**：RKNN / Horizon / Ascend 先做哪一个？按现有客户，
   不按热度。
8. **插件 API 冻结时间**：1.0 前保持 experimental；是否需要兼容层？
   建议 1.0 时只承诺 `PLUGIN_API_VERSION=1` 的最小面。

---

## 附录 A：计划文件地图

以下为规划中的新增/主要修改文件（实现时以实际为准；本表不含具体行号）：

```text
src/autovision/
  registry/catalog.py                 # 任务级模型目录 + verified 状态
  tasks/detection/pretrained.py       # 预训练检测构建器
  cli/recipes.py                      # 训练配方加载/合并
  export/postprocess/                 # 版本化后处理 spec + NumPy 解码
  export/e2e_parity.py                # 端到端 parity
  inference/multi_stream.py           # 多路流 runner
  data/quality.py                     # 数据体检
  data/hard_examples.py               # 难例清单 + sampler
  data/samplers.py                    # 长尾 sampler
  data/interchange/                   # CVAT / Label Studio 转换
  plugins.py                          # 插件协议 + entry-point 发现
  compliance/                         # 合规包
  registry/deployment_gates.py        # 部署门禁
scripts/
  eval_gtr_detect_ap.py               # GTR 检测全量 AP
  measure_int8_accuracy.py            # int8 AP delta
  check_quickstart.py                 # quickstart CI（或放 tests/）
examples/
  quickstart_detect.py
  finetune_detect.py
  video_predict.py
  export_matrix.py
  plugin-demo/
  cpp/
recipes/
  detection/finetune.yaml
  detection/coco-finetune.yaml
  detection/low-data.yaml
deploy/packs/<vendor>/                # 硬件伙伴包（按客户）
docs/
  quickstart.md
  leaderboard.md
  lts_policy.md
  support_policy.md
```

## 附录 B：新增/变更 CLI 一览

| 命令 | 阶段 | 作用 |
|---|---|---|
| `autovision-cli list-models --task detection [--pretrained]` | 1 | 任务模型目录 |
| `autovision-cli model-info <key>` | 1 | 许可/权重/AP/延迟/可导出/加载代码 |
| `autovision-cli data-check data=... out=...` | 3 | 数据体检 |
| `autovision-cli predict sources=... out=...` | 2 | 多路视频/流 |
| `autovision-cli compliance-report ...` | 4 | 合规报告 |
| `autovision-cli registry <verb> ...` | 4 | 注册表注册/晋级/回滚/查询 |
| `autovision-cli plugins list/check` | 4 | 插件发现与诊断 |

> 现有 25 条命令保持不变；新增命令走同一套 Pydantic 校验与
> `_parse_kwargs` 约定。命令名在实现时统一 kebab-case。

## 附录 C：JSON schema 草案

### C.1 hard_examples.json

```json
{
  "schema_version": 1,
  "source_run": "string",
  "iou_threshold": 0.5,
  "score_threshold": 0.3,
  "images": [
    {
      "image": "string",
      "split": "train",
      "tp": 0,
      "fp": 0,
      "fn": 0,
      "fp_reasons": {"duplicate": 0, "classification": 0, "localization": 0, "background": 0},
      "max_fn_area": 0.0,
      "classes_fn": [0]
    }
  ]
}
```

### C.2 data-check report.json

```json
{
  "schema_version": 1,
  "data": {"spec": "string", "splits": ["train", "val"]},
  "summary": {"images": 0, "issues": 0, "errors": 0, "warnings": 0},
  "issues": [
    {"split": "train", "path": "string", "kind": "unreadable|duplicate|bad_box|leak|...",
     "severity": "error|warning", "detail": "string"}
  ],
  "per_class": {"0": {"train": 0, "val": 0}},
  "imbalance_ratio": 1.0
}
```

### C.3 model catalog entry

```json
{
  "key": "rfdetr/rf-detr-small",
  "task": "detection",
  "family": "rfdetr",
  "status": "released",
  "pretrained": "full",
  "license": "Apache-2.0",
  "weights_asset": "rf-detr-small",
  "input_size": [512, 512],
  "params_m": 32.1,
  "gflops": null,
  "exportable": ["onnx", "tensorrt", "openvino"],
  "card": "docs/model_zoo.md#rf-detr",
  "verified": true,
  "verified_reasons": ["card", "sha256", "full_val_ap", "latency", "license", "e2e_parity"]
}
```

### C.4 compliance.json（顶层）

```json
{
  "schema_version": 1,
  "generated_at": "2026-10-17T00:00:00Z",
  "framework": {"name": "autovision-framework", "version": "0.4.0", "commit": "..."},
  "dependencies": [{"name": "torch", "version": "...", "license": "BSD-3-Clause"}],
  "models": [{"key": "...", "sha256": "...", "license": "...", "card": "..."}],
  "datasets": [{"spec": "...", "snapshot_digest": "..."}],
  "parity": [{"backend": "onnx", "passed": true, "report": "..."}],
  "known_limitations": "string"
}
```

## 附录 D：阶段验收总表

| 阶段 | 必须通过的测试/命令 | 必须存在的产物 | 失败即回退 |
|---|---|---|---|
| 0 | `pytest tests/test_quickstart.py`；干净 venv 安装；`mkdocs build --strict` | PyPI 0.3.0；文档站；英文 README | 保留源码安装；tag 延后 |
| 1 | 预训练/目录/配方测试；GTR AP 脚本；quickstart CI | RF-DETR 卡片；GTR s/m/l/x AP 行；catalog | 预训练入口 feature flag 关闭 |
| 2 | 后处理 spec、e2e parity、OV runtime、int8 AP、多路流测试 | `postprocess.json` + golden vectors；int8 表；多路 summary | 后端标记 experimental；保留 fp32 路径 |
| 3 | data-check、hard examples、多轮、互通、sampler 测试 | QA 报告；难例清单；案例报告 | 功能开关默认关闭 |
| 4 | 插件、模型库 verified、合规、注册表门禁测试 | verified 表；合规报告；LTS 分支；伙伴包 | 插件 API 标 experimental；企业功能按客户 |

---

## 附录 E：与现有文档的关系

- [差距分析](gap-analysis.md)：记录“缺什么”与 P0/P1/P2 执行历史；本文记录
  “接下来怎么干、什么算完成”。
- [架构总览](../framework_design.md)：分层契约与模块地图；新增模块必须遵守
  rank 表与硬禁（`tasks` 不 import `pretext`，`encoders` 不 import 上层）。
- [导出与部署](../export.md)：阶段 2 的现有能力与边界；新增后处理 spec 必须
  更新该页。
- [基准](../benchmarks.md)：阶段 1 的 GTR AP、阶段 2 的 int8 AP 必须遵守
  “只引官方值 / 自家数字附复现脚本 / 只增不改 + 勘误”协议。
- [发布流程](../release_process.md)：阶段 0 的 PyPI/快照/tag 步骤；阶段 4 的
  LTS 分支规则补进该页。
- [已知限制](../known_limits.md)：每阶段结束时更新；不要把“已修复”留在
  限制页里。

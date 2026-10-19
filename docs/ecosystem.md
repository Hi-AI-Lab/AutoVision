# 行业方案接入（Open-Core 生态设计）

本文定义 hi-agi-lab 组织内 **AutoVision 开源核心**与**行业场景方案仓**的连接
架构、接缝契约、许可边界与分阶段实施计划。目标读者：框架维护者与方案仓负责人。

> 定位一句话：AutoVision 是开源底座（Apache-2.0，PyPI 可装），场景仓是商业
> 方案（Commercial）；场景引擎通过 `pip install autovision-framework` 消费
> 框架稳定接缝，场景验证过的通用件反哺框架，形成"底座获客、方案变现、
> 通用件上浮"的 open-core 闭环。

## 1. 背景与现状盘点

### 1.1 组织仓库现状（2026-10）

| 仓库 | 可见性 | 形态 | 场景定位 |
| --- | --- | --- | --- |
| `AutoVision` | Public | 代码仓（唯一） | 训练/推理/导出/服务开源底座，已发 PyPI `autovision-framework` |
| `ai-traffic` | Public | 展示页（README+图，无代码） | 交通事件三域：遗撒散落物 + 事故复合评分分级 + 拥堵监测 |
| `ai-wildfire` | Public | 展示页 | 林草火险 + 防汛防台：瞭望塔双光谱烟柱、水位尺读数、星空地三源合并 |
| `ai-illop` | Public | 展示页 | 违规操作视频监管：SOP 分解为可训练、可验收的动作清单，注册表驱动 |
| `ai-conflict` | Public | 展示页 | 治安冲突视频分析（打架/斗殴 T3 长尾专项） |
| `ai-perimeter` | Public | 展示页 | 周界防护视频分析 |
| `ai-security` | Public | 展示页 | 安防通用 |
| `ai-life-line` | Public | 展示页 | 生命线工程 |
| `ai-urban` | Public | 展示页 | 城市治理 |
| `hi-ai-vision` | **Private** | 平台（代码） | 视频智能分析平台，商业交付收口 |

场景展示仓的共同特征：README 徽标为 `license: Commercial`、`core deps: numpy`、
`tests: N passed`（代码在私有侧）；无默认语言识别（GitHub 未见代码文件）。
即：**方案资产已存在，缺的是与开源底座的正式连接**。

### 1.2 为什么必须连接

1. **框架需要场景养料**：框架能力（zones、流状态机、事件契约）只有在真实
   场景里被高频消费，才会保持接口不腐烂；脱离场景的开源框架会逐渐失去演化
   方向。
2. **场景需要底座背书**：展示页只有"方案叙事"，没有可运行底座；接入
   AutoVision 后，"训练→评测→导出→流式推理"全链有公开可验证的实现，
   商业可信度显著提升。
3. **商业需要开源获客**：Apache-2.0 底座建立信任与线索，Commercial 方案
   引擎与 Private 平台完成变现——这是 open-core 的标准路径。

## 2. 总体架构

```text
┌───────────────────────── 场景方案层 · Commercial ─────────────────────────┐
│  ai-traffic   ai-wildfire   ai-illop   ai-conflict … （展示页 + 私有引擎）│
│  hi-ai-vision（Private 平台：告警工单、台账治理、多点位编排）              │
└───────────────────────────────┬───────────────────────────────────────────┘
                                │  pip 依赖（唯一正式接缝）
┌───────────────────────────────▼───────────────────────────────────────────┐
│  开源核心层 · Apache-2.0                                                   │
│  hi-agi-lab/AutoVision ──发布──▶ autovision-framework (PyPI ≥ 0.3.2)      │
│  稳定接缝：InferenceResult · Zones · run_stream · registry · export ·     │
│           serving · TaskNetwork                                            │
└───────────────────────────────────────────────────────────────────────────┘
```

连接原则：

- **单一接缝**：场景代码只允许通过 PyPI 包消费框架，禁止任何路径 import
  未发布代码（工作区外源码、git submodule、复制粘贴核心文件均不允许）。
- **稳定契约**：场景代码只依赖本文第 4 节列出的稳定接缝；框架对这些接缝的
  变更遵守弃用政策（`docs/lts_policy.md`：先弃用警告跨一个 MINOR）。
- **上浮单向**：通用件从场景仓流向框架（PR + 提案），框架永不反向依赖场景仓。

## 3. GitHub 层互连（P0，零代码）

### 3.1 场景仓：接入徽章与回链

每个场景展示仓 README 头部徽章区追加（中英双语 README 都要改）：

```markdown
[![powered by AutoVision](https://img.shields.io/badge/powered%20by-AutoVision-0FDC78)](
https://github.com/hi-agi-lab/AutoVision)
[![pypi](https://img.shields.io/pypi/v/autovision-framework)](
https://pypi.org/project/autovision-framework/)
```

并在"技术架构"或结尾处固定一句回链：

```markdown
本方案的检测底座、流式推理运行时与导出链路由开源框架
[AutoVision](https://github.com/hi-agi-lab/AutoVision) 提供
（`pip install autovision-framework`）。
```

同时给仓库添加 topics：`autovision` `video-analytics` `open-vocabulary-detection`
+ 场景专属 topic（如 `road-traffic` / `wildfire` / `industrial-safety`）。

### 3.2 AutoVision 仓：Ecosystem 章节

1. 仓库根 `README.md` 与 `README.en.md` 增设"行业方案 / Industry Solutions"
   小节：8 个场景仓 + `hi-ai-vision` 一句话定位表格，全部外链 GitHub。
2. 文档站新增 `docs/ecosystem.md`（即本页），导航置于"插件协议"之后，
   作为生态的权威页面；README 小节链接到此页获取细节。

### 3.3 组织层

- org profile README（`hi-agi-lab/.github` 仓库）写明组织使命与两条产品线。
- Pin 仓库：`AutoVision` 置顶第一，其余按商业优先级排 4 个。
- 场景展示仓的 CI 徽章（README 中 `ci: pending` 占位）在私有引擎建立 CI 后
  替换为真实状态；过渡期删除占位徽章避免误导。

## 4. 技术接缝层（P1 试点 / P2 推广）

### 4.1 依赖声明

场景引擎包统一声明：

```toml
# 场景引擎 pyproject.toml
dependencies = [
    "autovision-framework>=0.3.2,<0.5",   # 上界到下一个大版本前的双 MINOR
]
```

版本策略：框架发布新 MINOR 时，先跑场景引擎侧的兼容冒烟（场景仓的 40 例
测试 + 关键路径），确认无 breaking 再放宽上界；场景引擎禁止使用
`autovision` 内部模块（`_` 前缀、`experimental` 命名空间）。

### 4.2 六大稳定接缝与场景需求映射

| 稳定接缝 | API 形态 | 场景侧需求（以 ai-traffic 为例） | 使用约束 |
| --- | --- | --- | --- |
| 事件契约 | `autovision.results.InferenceResult`；`to_json(masks=...)`、`to_record()` | 检出对象 → 平台事件上报的标准载体；跨 API 边界传输 | mask 默认 RLE，禁止 dense `tolist()` 进 JSONL/HTTP |
| 区域判定 | `autovision.tracking.zones.PolygonZone` / `LineZone` | 车道多边形过滤、白名单区域（公交站/施工区）、禁停区 | zone 状态机按帧有序，只能在流循环内更新 |
| 流式运行时 | `autovision.inference.stream.run_stream` / `multi_stream` | 点位断流告警（≤30s）、重连退避、多路并发 | tracker/zone 状态不得在主循环外更新 |
| 权重注册 | `autovision.registry`（模型注册表 + model card） | 检测底座版本固定与交付审计；场景模型包登记 | sha256 钉住；受限许可权重不自动下载 |
| 导出链 | `autovision.export`（ONNX/TensorRT/OpenVINO + parity 门禁） | 边缘盒子/工控机部署；版本可复现 | 导出保持 `verify=True`，禁止关闭对拍 |
| 参考服务 | `autovision.serving`（FastAPI + `/predict` `/healthz`） | `hi-ai-vision` 平台接入的原型对齐 | 参考实现不直接承载生产；生产走平台侧 |

### 4.3 ai-traffic 试点改造清单（P1）

以 `ai-traffic` 为试点验证接缝充分性，改造项与验收标准：

| # | 改造项 | 现状 | 目标 | 验收 |
| --- | --- | --- | --- | --- |
| T1 | 检出对象契约化 | 引擎内自定义 dict | 统一走 `InferenceResult`，事件字段经 `to_record()` | 事件 JSON Schema 与平台对拍通过 |
| T2 | 车道过滤 | numpy 手写多边形判断 | `PolygonZone`（注册车道多边形，zone 内过滤） | 与原实现误报/漏报一致（同批回放帧零差异） |
| T3 | 断流告警 | 自研计时 | `run_stream` 的断流口径 + `frames_failed` | 断流 30s 告警演练通过 |
| T4 | 底座权重 | 隐式路径 | `registry` 登记 + model card（含 sha256、许可） | 新环境一键加载，无路径硬编码 |
| T5 | 边缘导出 | 无 | 底座经 `export` 出 ONNX（parity 门禁） | 对拍误差 ≤ 门禁阈值 |

试点产出两份结论回填本文：接缝缺口清单（若有）与实测回放对比数据。

### 4.4 通用件上浮机制（P2）

场景侧被 ≥2 个场景复用且与具体业务无关的组件，按以下流程上浮：

1. **提案**：在 AutoVision 仓开 issue，附场景来源、API 草案、与现有模块的
   关系（避免重复入口——`docs/release_process.md` 的单一入口原则）。
2. **归属**：状态机/时间窗类 → `tracking`（如"持续确认窗"）、区域治理/
   台账类 → `tracking.analytics` 或 `metrics`；纯业务口径（评分权重、
   SOP 清单）**不上浮**，留在场景引擎。
3. **实现**：遵守仓库硬约束（类型提示、矢量化、`AGENTS.md` 测试规范）；
   变更记入 `CHANGELOG.md`。
4. **下线**：框架发布含该组件的 MINOR 后，场景引擎切换到框架实现并删除
   私有副本（一个 MINOR 的并存宽限，之后私有副本必须移除）。

首批候选：持续确认窗状态机（ai-traffic FR-AC-04）、白名单台账治理
（ai-traffic/ai-perimeter 共用）、水位尺读数标定层（ai-wildfire，待第二个
使用者出现再上浮）。

## 5. 商业边界与许可

### 5.1 开源 / 商业分界

| 资产 | 授权 | 判定标准 |
| --- | --- | --- |
| 框架全链（训练/评测/导出/流/参考服务） | Apache-2.0 | 通用能力，无行业口径 |
| 模型 zoo 与公开权重 | 见 `THIRD_PARTY_LICENSES.md` | 逐资产登记 |
| 场景评分引擎、领域提示词、SOP 动作清单 | Commercial | 行业口径是商业价值本体 |
| `hi-ai-vision` 平台 | 商业闭源 | 交付与运维收口 |
| 场景展示页（README/架构图/演示素材） | Public 仓库（© 保留） | 营销物料，不含引擎代码 |

### 5.2 许可红线（强制）

1. 场景引擎只依赖 PyPI 公开包；禁止把框架未发布代码带入场景仓。
2. `ultralytics` 为 AGPL-3.0：仅通过框架 `ultralytics` extra 隔离加载；
   场景仓不得直接引入，不得让 AGPL 代码与 Commercial 代码同文件/同进程
   静态链接（进程隔离由框架 extra 机制保证）。
3. 场景仓引用框架文档时使用链接而非复制粘贴；防止文档漂移与许可混淆。
4. "AutoVision" 名称与徽标仅按第 3.1 节的 "Powered by" 形式使用；改名或
   商业用途表述（如认证、背书）需维护者书面同意。

## 6. 分阶段实施计划

| 阶段 | 内容 | 交付物 | 完成判据（DoD） |
| --- | --- | --- | --- |
| P0 互链 | 3.1–3.3 全部操作 | 8 仓徽章+回链+topics；README Ecosystem 小节；org pin | `check_docs_links.py` 绿；任一展示页可 2 跳内到达框架 |
| P1 试点 | 4.3 ai-traffic 改造 | T1–T5 完成 + 缺口清单 | 回放对比零差异；40 例测试全绿 |
| P2 上浮 | 4.4 首批候选进框架 | 确认窗状态机 + 台账治理合入 | 框架新增模块有测试与文档；≥2 场景切换到框架实现 |
| P3 推广 | 其余场景仓按 4.2 接入 | 各仓接入 PR | 每仓至少 T1（契约化）完成 |
| P4 引流 | 文档站行业方案页打磨、SEO 关键词、案例数据回填 | 站点页面 + 埋点口径 | 站点→商业线索路径存在且可统计 |

节奏建议：P0 立即；P1 两个周内完成试点；P2/P3 随框架 0.4.0 发布窗口；
P4 持续。

## 7. 风险与缓解

| 风险 | 影响 | 缓解 |
| --- | --- | --- |
| 接缝不稳定导致场景仓频繁返工 | 高 | 场景锁 `>=0.3.2,<0.5`；框架弃用政策跨 MINOR；试点先行 |
| 通用件上浮变成业务口径进框架 | 中 | 4.4 第 2 条归属判定；PR 评审否决权在框架维护者 |
| AGPL 传染 | 高 | 5.2 红线 + 框架 provenance 门禁（`check_provenance.py`）双保险 |
| 展示页与引擎实现漂移（承诺了框架没做的） | 中 | 展示页技术叙事以本文第 4 节接缝为边界；商业口径不上文档站 |
| 双语维护负担 | 低 | 展示页中英同改；框架侧英文页按 fallback 机制可延后 |

## 8. 维护

- 本页随每次阶段完成（P0–P4）更新对应章节与交付物状态。
- 接缝清单（4.2）为契约性内容：增删接缝必须同步 `docs/lts_policy.md` 与
  `CHANGELOG.md`。
- 框架侧接口示例以仓库内 `tests/` 与 `examples/` 为准；本页不复制代码，
  只锚定模块名。

# 指标与合规：设计与实施

| 文档属性 | 内容 |
|---|---|
| 版本 | v1.0 |
| 状态 | as-built 设计 + 整改实施设计（待实施） |
| 日期 | 2026-11-06 |
| 审计基线 | v0.6.0 工作区；行号为该快照，实施时以符号名为锚 |
| 读者 | 框架研发、评审人、算法团队（指标消费方） |
| 关联 | [总览与路线](overview.md)、[基准协议](../benchmarks.md)、[架构总览](../framework_design.md)、[加固计划](../hardening_plan.md)（H1-1/H3-1c/H3-6c/d 已实施项不重复） |

## 1. 范围与验收判定

范围：`src/autovision/metrics/`（COCO mAP + 面积分档、OBB、混淆矩阵、
误差分析、分割/姿态/深度/全景指标）与 `src/autovision/compliance/`
（许可、inventory、报告）。
判定：**未达标，但差距集中且可收敛**。检测主链（pycocotools bbox
AP、状态卫生、混淆矩阵、误差分析归因、合规 fail-closed 许可归一）
成熟且有 H 系列回归锚；但 4 个 P1 全部影响**对外指标数字**——OBB
自研 AP 两处正确性 bug、crowd 标志训练链路断链、headline P/R/F1
口径失真。这直接违反 AGENTS.md §3.3"数值改动必须对拍"的纪律
（OBB 恰好无对拍）。

## 2. 现状设计（as-built）

### 2.1 mAP 评估器

pycocotools 后端：`compute()` 把内存预测/GT 组装成 COCO dict 走
`COCOeval("bbox")`（`src/autovision/metrics/map_evaluator.py:279`）；
headline 刻意跳过 `summarize()`（其 stats 钉死 maxDets=100），改从
precision/recall 数组自算；面积分档 small/medium/large 是 additive
keys（-1 表示该档无 GT）；`auto_raise`（maxDets 上限提升）是
per-call 局部变量不污染状态（加固计划 H3-6c 已实施其状态卫生）；
DDP 汇聚：`export_state/merge_state` CPU 张量快照，rank0
`all_gather_object` 汇聚后统一 compute，`merge_state` 重推导
`_max_instances_seen`。crowd：`update_targets` 接受可选 `iscrowds`
并写入 GT annotation（评估器层面正确）。

### 2.2 其余指标

- 混淆矩阵：`(classes+FP) × (classes+FN)` 口径，TP 对角、FN 落
  `[true, FN]` 列、误分类 FP 落 `[true, pred]`；crowd 命中行排除；
  显式 `class_ids` 截断时 warning 一次（H3-1c 已实施）。
- 误差分析：分数降序贪心匹配（H1-1 已实施）、TIDE 风格互斥归因
  （duplicate → classification → localization → background）、
  crowd 语义对齐 pycocotools `dtIg`（H3-6d 已实施）、全局 top-K
  画廊确定性选择；`hard_examples.json` 只被 train split 采样消费。
- 分割：混淆矩阵 mIoU（ignore=255，DDP 矩阵相加）；姿态：官方
  `COCOeval("keypoints")`；实例分割：pycocotools segm；全景：自研
  PQ/SQ/RQ（IoU>0.5 严格匹配）；深度：AbsRel/RMSE/δ。
- OBB：**全仓唯一自研匹配 + 自研 AP**（`src/autovision/metrics/oriented_detection.py`）。
- 合规：许可显式别名表 + SPDX 表达式、任一原子未知整体 UNKNOWN
  （fail-closed）；inventory 校验框架 commit/模型 sha256/快照漂移；
  报告产出 JSON/MD/CycloneDX SBOM/sha256 manifest。

## 3. 关键流程

- **验证 → mAP**：`validation_step` → `model.postprocess(threshold=0.0)`
  → `evaluator.update(decoded)` + `update_targets(...)` → engine 在
  验证点 gather → rank0 compute → `val_metric/*` 进日志/JSONL →
  `validation_report()` 渲染 + HEADLINE_PROTOCOL footer。
- **analyze-errors**：在线（model+data）或离线（predictions+
  annotations）→ `match_detections` → `summarize_matches` →
  `ConfusionMatrix.from_matches` → 画廊 → report.{json,md} +
  hard_examples.json。
- **合规报告**：许可扫描 → 模型 sha256+卡片校验 → 快照重导 →
  `ComplianceReport.save()` 四件套 + manifest 复核 → CLI
  `passes(fail_on)` 门禁。

## 4. 审计发现清单

### 4.1 P1

| 编号 | 问题 | 证据 |
|---|---|---|
| METRIC-01 | OBB mAP 只遍历该图 GT 中出现的类别（`gt_labels.unique()`）：无同类 GT 图像上的预测（含背景 FP）**静默丢弃**，不进 PR 流——AP 系统性虚高（已实证：官方语义 0.5 的场景返回 1.0） | `src/autovision/metrics/oriented_detection.py:147` |
| METRIC-02 | OBB `_average_precision` 包络积分错误：reversed 扫描时"recall 增长点"积分用的是当时已累计的 best_precision，同 recall 值的更高精度点在积分之后才被纳入，无法回补——AP 系统性虚低（已实证：VOC 标准实现 1.0，本实现 0.5）；docstring 声称 all-point (VOC-style envelope) 但实现并非 | `src/autovision/metrics/oriented_detection.py:44-52` |
| METRIC-03 | crowd（iscrowd）训练链路断链：评估器支持、COCO 数据集产出该字段，但 `ObjectDetectionDataset.__getitem__` 只读 bboxes/class_labels 把 iscrowds 丢弃，四个检测任务的 `validation_step` 构造 targets 均不携带——训练期验证把 crowd 框当普通 GT 计 FN，mAP 系统性偏低（val2017 有 446 个 crowd 框）；评估器级单测存在但无端到端 wiring 测试，正是断链漏网原因 | `src/autovision/data/object_detection_dataset.py:94-115`；`src/autovision/tasks/detection/detr/train_model.py:635-639`、`src/autovision/tasks/detection/rtdetr/train_model.py:142-148`、`src/autovision/tasks/detection/rfdetr/train_model.py:301-320`、`src/autovision/tasks/detection/picodet/train_model.py:95-101` |
| METRIC-04 | headline precision/recall/F1 三元组不属于同一操作点：precision 数组的 R 轴是 101 个 recall 阈值（recThrs）而非 score 阈值，当前实现把每个 recall 阈值点的 precision 与 maxDets 全量 recall 配对求 F1 再 argmax；`HEADLINE_PROTOCOL` 文案宣称 "F1-optimal score threshold" 与实现不符，且随 CLI footer / model card 外传 | `src/autovision/metrics/map_evaluator.py:344-379`、`:389-411`、`:26-30` |

### 4.2 P2

| 编号 | 问题 | 证据 |
|---|---|---|
| METRIC-05 | 合规 CLI 默认 fail-open：未传 strict/fail_on 时 `fail_on="none"`，报告有 error 也退出码 0（Python API 默认是 "error"）——企业门禁命令默认恒通过 | `src/autovision/cli/_app.py:349-350` vs `src/autovision/compliance/report.py:73-79` |
| METRIC-06 | OBB AP 协议与 COCO 不可比（自研 exact all-point vs 官方 101-point）且**无对拍金标准**——AGENTS.md §3.3 数值纪律的唯一豁免区，正是 METRIC-01/02 漏网的原因 | `src/autovision/metrics/oriented_detection.py:5-10` docstring 自述 |
| METRIC-07 | `DepthMetricCollector` 全量累积 val 集 (pred, gt) 深度图（merge 再翻倍）：NYU 级数据可达数十 GB；classification 同模式但有"验证集小"声明，depth 无声明也无降维选项 | `src/autovision/metrics/depth_estimation.py:44-61` |
| METRIC-08 | error_analysis report.json 全量 matches 落盘（数十万框 → 数百 MB）且 confusion 内容写两遍 | `src/autovision/metrics/error_analysis.py:656-667`、`:527` |
| METRIC-09 | 分割指标只过滤 prediction 越界；target 为负（部分数据集用 -1 而非 255 表示 ignore）→ bincount 崩溃；target ≥ num_classes → reshape 崩溃 | `src/autovision/metrics/segmentation.py:33-38` |

### 4.3 P3

| 编号 | 问题 | 一句话方向 |
|---|---|---|
| METRIC-10 | COCO dict 的 categories name 写成数值 id（pycocotools 不用 name，语义错误） | 对齐 error_analysis 的正确写法 |
| METRIC-11 | `_max_instances_seen` 跨 `reset()` 保留，跨数据集泄漏 | reset 时清零 |
| METRIC-12 | pose 评估器在 batch 无 index 时用 rank 局部 id（DDP 潜伏冲突） | 断言 index 必备 |
| METRIC-13 | PQ 允许重叠预测段双匹配同一 GT（重复计 TP） | 匹配互斥 |
| METRIC-14 | 画廊文件名碰撞（同 stem 不同目录互相覆盖） | 名含相对路径哈希 |
| METRIC-15 | `_render_overlay` 对灰度/RGBA 崩溃；画廊图缺失使整场分析崩溃且 report.md 已先写 | 类型归一 + 容错 + md 后写 |
| METRIC-16 | `match_detections` 的 zip 静默截断（长度校验只在高层入口） | API 层断言 |
| METRIC-17 | `per_class` 以类名为键，类别重名时计数合并 | 改 id 键或重名报错 |
| METRIC-18 | 合规报告非原子写；SBOM serialNumber 随机 uuid4 不可复现；`timestamp=False` 时 null 不符 CycloneDX | 复用原子写；确定性 uuid（uuid5）；省略字段 |
| METRIC-19 | markdown 表格 cell 未转义（依赖名含 `|` 破坏表） | 统一转义 helper |
| METRIC-20 | 依赖去重未按 name 小写归一（大小写变体重复计行） | 归一化去重 |
| METRIC-21 | `model_inventory`/`dataset_inventory` 对坏快照裸 raise（与"缺快照记 info issue"不一致） | 坏快照进 issue 清单 |
| METRIC-22 | `TaskMetrics._concatenate` 对标量列崩溃（docstring 自认支持标量） | 标量转 (1,) 数组 |
| METRIC-23 | `COCOMAPEvaluator.set_ground_truth` 无调用方的遗留公共 API | 按弃用协议移除 |
| METRIC-24 | DDP 汇聚经 all_gather_object pickle 全量预测 dict（序列化开销） | 评估 0.7.x 优化项 |
| METRIC-25 | 实例分割每验证轮对全量 GT mask 重编码 + json 落盘（无缓存） | GT RLE 缓存 |

## 5. 整改实施设计

> 状态：**METRIC-01..09 已实施 + P3 落地 12 项（10/11/12/14/15/16/17/
> 18/19/20/21/22/23），随 0.6.1 发布**；P3-24（DDP 汇聚 pickle 成本）与
> P3-25（实例分割 GT 重编码）登记为性能批次。
> 实施与设计的偏差：METRIC-04 直接落地方案 A（interim 文案修复被方案 A
> 取代——新文案随单操作点一步到位）；OBB golden 采用"手工可算向量 +
> 独立 oracle 断言"而非 pycocotools 多边形化对拍（旋转框无官方可比
> 基线，设计中的属性测试以 oracle 形式落在 `_average_precision` 上）；
> METRIC-03 的 mAP 侧照旧走 pycocotools（其原生处理 iscrowd），headline
> 侧经 match_detections 的 dtIg 语义忽略 crowd；P3-13 的互斥锁对
> (B,H,W,2) id-map 契约不可达（预测段天然互斥），保留为防御性保险并有
> 回归测试钉住合法输入；P3-23 按"警告一个 MINOR"实施而非直接删除。
>
> **实施后复审结论：复审判定"需修复后合入"，全部发现已闭环。** 复审
> 用运行时实证抓住了两个 P0——恰好都落在单测盲区（本条即审计对
> METRIC-03 的原始教训重演）：① **headline 键名断链**——匹配器只认
> `"boxes"` 而生产 decode 契约是 `"bboxes"`，训练路径 headline 恒为
> 0/0/0 而测试全绿（测试恰好都用 `"boxes"` 键）；② **crowd 链在
> transform 边界断开**——`RTDETRDetectionTransform.__call__` 重建输出
> dict 时丢弃 `iscrowds`，生产 CLI 永远 `set_transform`，修复完全不
> 生效。已修：`_matches_cache` 归一化 `bboxes`/`boxes` 双键（缺键仍走
> 空记录兜底）；transform 三条输出路径（mosaic affine keep / compose
> sanitize / 主返回）全部携带 iscrowds 并与框过滤对齐。复审另抓到
> rfdetr 的接错步骤（training_step 死接线、validation_step 漏接）——
> 已改为 crowd 只进 VALIDATION targets（对齐设计"只影响验证指标"）。
> 补齐三个端到端 wiring 测试（dataset→transform→collate→evaluator、
> 生产 bboxes 键 headline、crowd 全链）。其余：P1-2 合规四件套原子写
> 补实现；P3-a update 失效缓存、P3-b 画廊悬空链接、P3-e 死代码清理、
> P3-d 文案同步为排期项（低风险）。**复审教训已固化为流程注记：单测
> 必须覆盖"生产输入形态"（键名/边界层），否则 wiring 类修复的验收是
> 假的。**

### 5.1 METRIC-01 OBB PR 构建覆盖全类别（P1）

**问题**：`for class_id in gt_labels.unique().tolist()` 使无 GT 图像
上的该类预测永远不进 PR 流。

**设计**：类集合改为"该图预测类别 ∪ 该图 GT 类别"（全局逐图迭代，
按类累积 (score, tp_flag) 列表），语义对齐 pycocotools（某类有 GT 的
数据集中，该类在无 GT 图像上的检测是 FP 并参与排序，受 maxDets
约束）。实现上把"按 GT 类别分桶"改为"按预测类别分桶 + GT 命中标记"，
保持每图每类 top-k 截断不变。

**涉及文件**：`src/autovision/metrics/oriented_detection.py`
（匹配主循环与 PR 累积）。

**测试计划**（与 METRIC-06 的金标准一并落地）：
1. 审计复现用例直接固化为回归：img0 一个 GT + 一个 TP(0.9)；img1
   无 GT + 一个 0.95 背景 FP → `map_50 == 0.5`（修复前 1.0）。
2. 轴对齐子集属性测试：随机 fixture 上 OBB 评估器（框转多边形）与
   pycocotools bbox 评估的 AP 差值在容差内（101-point 对齐见
   METRIC-02 的口径决策）。

**验收标准**：复现用例从 1.0 变 0.5；属性测试绿。

### 5.2 METRIC-02 OBB AP 换标准包络积分（P1）

**设计**：`_average_precision` 改为标准 VOC all-point 实现：
分数降序 → 累计 TP/FP → precision 单调包络
（`np.maximum.accumulate(prec[::-1])[::-1]`）→ recall 网格积分
（`sum((r[i]-r[i-1]) * prec_env[i])`，recall 前置 0）。**口径决策
（推荐方案 A）**：同时提供 `interpolation="all-point"|"coco101"`
两挡（默认 all-point 维持 docstring 语义），coco101 挡用于与
pycocotools 的可比性对拍；两挡差异写进 docstring 与 model card
字段。

**涉及文件**：`src/autovision/metrics/oriented_detection.py`。

**测试计划**：审计复现向量固化：`[(0.9, TP), (0.8, FP)]` 单 GT →
all-point = 1.0（修复前 0.5）；`[(1,0,0) flags]` → 1.0（修复前
0.333）；与独立参考实现（测试内 10 行 numpy oracle）在 200 组随机
fixture 上逐位一致；coco101 挡与 pycocotools 多边形化路径差值容差内。

### 5.3 METRIC-03 crowd 链路端到端接通（P1）

**设计**（纯 wiring，评估器零改动）：

1. `ObjectDetectionDataset.__getitem__` 从行 JSON 读出 `iscrowds`
   并入 target（与 bboxes/class_labels 同形）；collate 规则同步。
2. 四个检测任务（detr/rtdetr/rfdetr/picodet）的 `validation_step`
   把 crowd 标志传给 `evaluator.update_targets(..., iscrowds=...)`
   ——`update_targets` 签名已支持，只差调用方。
3. **范围决策**：crowd 只影响验证指标，不进训练损失（matcher/损失
   行为不变）——与 COCO 官方评估惯例一致，理由写入各 train_model
   docstring；若后续要"损失也忽略 crowd"，另立条目评估。

**涉及文件**：`src/autovision/data/object_detection_dataset.py`、
上述四个 `train_model.py`。

**测试计划**：端到端 wiring 测试（tiny COCO fixture 含一个 crowd
标注 → 走 validation_step → spy 断言 `update_targets` 收到
iscrowds 且该框不计 FN）；无 crowd fixture 的 mAP 数值与现状逐位
一致（防回归锚）；`tests/test_train_object_detection.py` 既有评估器
级 crowd 用例保持。

**验收标准**：wiring 测试通过；含 crowd 数据的训练期 mAP 不再被
crowd 框压低（用 tiny fixture 前后对照值锁定）。

### 5.4 METRIC-04 headline 口径修正（P1）

**设计**（**决策点，推荐方案 A**）：

- **方案 A（推荐）**：headline 改为**单一显式操作点**的可信读数：
  新增 `headline_score_threshold: float = 0.50`（评估器参数），在该
  score 上用 `match_detections` 机制计算 precision / recall / F1
  （三者天然同操作点，实现复用误差分析的匹配器）；`HEADLINE_PROTOCOL`
  文案同步改写为"single operating point at score=X"；footer 与
  model card 字段同步。旧字段名保留一个 MINOR + deprecation warning
  （值语义变化必须在 CHANGELOG 显著标注）。
- **方案 B（更保守）**：headline 只报 AP/AR（官方语义），删除 P/R/F1
  ——最诚实但损失可用性，作为 fallback 记录。

无论何方案：`map_evaluator.py:26-30` 的 `HEADLINE_PROTOCOL` 文案先
修正为当前实现的真实语义（ interim 修复，防止披露继续失真），该
interim 修复可先行独立合入。

**涉及文件**：`src/autovision/metrics/map_evaluator.py`、四个检测
`train_model.py` 的 report 渲染、`docs/benchmarks.md`（协议声明）。

**测试计划**：构造已知匹配的 fixture → P/R/F1 逐位等于手算值；
footer 文本对账测试（基础设施测试风格，防文案再漂移）；cap=100 时
AP 与 `summarize()` 等值性锚定测试（补上加固计划 H3-6c 遗留的"无
测试锚定"缺口）。

### 5.5 METRIC-05 合规 CLI 默认 fail-closed（P2）

**设计**：CLI 默认 `fail_on="error"` 对齐 Python API。定性为安全修复
级默认翻转（fail-open 的门禁是危险默认），沿加固计划 H2-1 的先例：
PATCH 内直接生效 + CHANGELOG 显著标注；脚本依赖旧默认（恒通过）的
用户会开始看到非零退出——错误消息给出 `fail_on=none` 显式旁路。
**测试计划**：默认 + 含 error 报告 → 非零退出；`fail_on=none` 显式
旁路 → 退出 0；`tests/test_compliance.py` 回归。

### 5.6 METRIC-06 OBB 对拍金标准（P2，与 5.1/5.2 同批）

**设计**：`tests/golden/obb_ap.json`（fail-fast 录制策略，与导出
golden 同语义）钉住三类场景的期望 AP：随机 fixture × {全点积分,
coco101} × {有 crowd, 无 crowd}；另加 5.1/5.2 的复现向量。这是
AGENTS.md §3.3 在 OBB 面的补课，此后任何 OBB 数值改动必须过 golden。

### 5.7 METRIC-07 深度指标流式化（P2）

**设计**：`DepthMetricCollector` 改为逐图像累计标量（absRel 和、
平方误差和、δ 计数、样本数），对齐模式（none/median/log-affine）
在逐图阶段完成仿射拟合再取标量；`merge_state` 只聚合标量。内存
O(1) 于数据集大小。**测试计划**：小 fixture 上流式实现与全量实现
数值一致（容差 1e-9）；`merge_state` 跨 rank 幂等；NYU 量级合成
数据的 RSS 断言。

### 5.8 METRIC-08 误差分析报告瘦身（P2）

**设计**：report.json 的 matches 改为摘要计数 + top-K（默认
`max_report_matches=1000`，参数暴露）；confusion 只保留顶层一份
（删除 summary 内嵌副本，读取端兼容注记一个 MINOR）。hard_examples
与画廊不变。**测试计划**：大 fixture → report 体积上界断言；confusion
数值前后一致；`tests/test_error_analysis.py` 回归。

### 5.9 METRIC-09 分割 target 域校验（P2）

**设计**：构建 bincount 索引前校验 target：值 < 0 且 != ignore_index
→ `AutoVisionError`（消息含图像 id 与非法值）；值 ≥ num_classes →
同上。ignore=255 与 ignore=-1 两种数据集口径都显式支持
（`ignore_index` 已是参数，补文档）。**测试计划**：负值 / 越界
fixture → 可读报错；ignore=-1 配置 → 正常出 mIoU。

## 6. 实施顺序

阶段 1 批次（本模块全部 P1 + METRIC-05/06 联动）：

```
METRIC-04-interim（文案修正，独立小 commit 先行）
  → METRIC-01 + METRIC-02 + METRIC-06（OBB 三连：修实现 + 换积分 + 立 golden，一次评审）
  → METRIC-03（wiring 四文件，含端到端测试）
  → METRIC-04 方案 A（口径切换，随 MINOR）
  → METRIC-05（CLI 默认翻转，随 PATCH）
```

P2 其余（07/08/09）进阶段 3。每项独立 commit，标题
`fix(metrics): METRIC-nn <摘要>`。

## 7. 风险与回滚

- METRIC-01/02 修复后**所有历史 OBB 数字不可比**（一虚高一虚低的
  双向失真被同时纠正）：CHANGELOG 显著标注"OBB mAP 口径修正，与
  0.6.x 之前数字不可比"；golden 保证此后稳定；`docs/benchmarks.md`
  的 OBB 行标注口径切换版本。
- METRIC-03 使含 crowd 数据的训练期 mAP 数值上升（修复目标）：
  基准页在切换版本重新对拍一次 bbox 主链（防 headline 连带漂移）。
- METRIC-04 方案 A 的字段值语义变化：deprecation warning 跨一个
  MINOR；footer 与 model card 同步；无静默双轨。

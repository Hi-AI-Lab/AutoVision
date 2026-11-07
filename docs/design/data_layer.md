# 数据层：设计与实施

| 文档属性 | 内容 |
|---|---|
| 版本 | v1.0 |
| 状态 | as-built 设计 + 整改实施设计（待实施） |
| 日期 | 2026-11-06 |
| 审计基线 | v0.6.0 工作区；行号为该快照，实施时以符号名为锚 |
| 读者 | 框架研发、评审人、数据平台 |
| 关联 | [总览与路线](overview.md)、[架构总览](../framework_design.md)、[加固计划](../hardening_plan.md)（其数据层条目已实施项不重复）、[已知限制](../known_limits.md) |

## 1. 范围与验收判定

范围：`src/autovision/data/`（数据集、行缓存、格式转换、interchange、
快照与漂移校验、数据体检）。判定：**部分达标**——检测训练热路径
（mmap 原子发布 + 持锁引用计数 + 扫描缓存 + 带文件名/行号校验）达到
企业级水准；但版本化功能对标准 YOLO data.yaml 不可用（DATA-01/02）、
YOLO 行缓存对图像集合变化不失效（DATA-03）、实例分割全量 mask 驻留
内存（DATA-04），三者阻止验收。

## 2. 现状设计（as-built）

### 2.1 组件地图

按"格式解析 → 行归一化 → mmap 行缓存 → 任务 Dataset → 变换管线"分层：

| 组件 | 文件 | 职责 |
|---|---|---|
| 基类/协议 | `src/autovision/data/task_dataset.py` | transform 注入与 collate 声明 |
| 类别归一 | `src/autovision/data/label_helpers.py` | 任意类别 id → 0 起稠密 id |
| 行契约 | 各 `*DataArgs.list_image_info()` | 归一行迭代器（image_path + JSON 字符串标注），转换/训练/体检共用 |
| mmap 行缓存 | `src/autovision/data/serialize/memory_mapped_sequence.py` | PyArrow IPC 分块写、按 pid 惰性重开 mmap、pickle 只带 path+columns（worker 零拷贝） |
| 缓存命名/发布 | `src/autovision/data/mmap_helpers.py` | sha256 命名、rank0 临时文件 + FileLock 下"先 refcount 后 rename"原子发布、其余 rank 轮询等待、退出引用计数清理 |
| 指纹 | `mmap_helpers.py` 的 `_data_fingerprint` + 各 args 的 `*_mmap_hash` | 标签文件路径+size+mtime_ns 失效 + 参数维度 |
| 扫描缓存 | `src/autovision/data/scan_cache.py` | 逐文件可解码性裁决缓存到 `<root>/.autovision_scan_cache.json`，严格 best-effort |
| 图像缓存 | `src/autovision/data/image_cache.py` | disk（`.npy` sidecar）与 ram（`/dev/shm`）两层 opt-in |
| 格式转换 | `src/autovision/data/convert.py`、`src/autovision/data/interchange/` | YOLO/COCO/VOC/pose/seg 互转、CVAT/Label Studio → COCO（`CocoBuilder` 统一装配） |
| 快照 | `src/autovision/data/versioning.py` | 内容寻址快照（排序清单 sha256 + deep 内容哈希）与漂移校验 |
| 体检 | `src/autovision/data/quality.py` | 复用行契约做离线审计 |

防穿越唯一入口是 `src/autovision/data/file_helpers.py` 的
`resolve_under`；损坏 JSON/缺字段的报错必带文件名。代码中大量
加固计划数据层系列的编号注释是上一轮加固的痕迹（见加固计划 §8.2，
本文不重复）。

### 2.2 契约与不变量

- 行语义与缓存裁决解耦：扫描缓存只裁决不改行（`object_detection_dataset.py` 构造注释）。
- mmap 缓存"裁决不碰数据"：损坏缓存自愈为回源解码。
- 快照写走 tmp + fsync + `os.replace`（`versioning.py`）。
- 类别以 train split 的 categories 为准；val 未声明类别计数告警后丢弃（COCO 侧）。

## 3. 关键流程

- **检测训练**：`list_image_info(split)` → `dataset_mmap_context` 落成
  mmap → `__getitem__` 解析 JSON 行 → `open_image` 解码 → 归一 cxcywh
  转绝对 XYXY → transform。
- **COCO 行构建**：读 JSON → image_id 索引（重复 id 报错）→ 按
  image_id 分组标注 → train categories 定类别表 → val 未声明类别
  计数告警丢弃 → bbox 归一化。
- **YOLO 行构建**：`rglob` 图像目录 → labels 镜像路径 → 逐行解析
  （字段数≠5 计数弃用告警）；OBB 变体字段数为 9。
- **快照**：`_load_spec`（yaml/json）→ `_collect_roots` → 排序清单
  sha256（deep 再加内容哈希）→ 原子写；verify 重导并报
  added/removed/resized/content-changed。

## 4. 审计发现清单

### 4.1 P1

| 编号 | 问题 | 证据 | 后果 |
|---|---|---|---|
| DATA-01 | `dataset_snapshot` 无法处理标准 YOLO data.yaml：顶层字符串值全部归入 `"(shared)"`，类名字符串被当路径收集 | `src/autovision/data/versioning.py:133-141`、`:151-157`（已实证） | strict（默认）直接报错；strict=False 能写但 train/val 分片结构完全丢失——版本化主功能对 YOLO 用户不可用 |
| DATA-02 | `verify_dataset_snapshot` 只遍历快照内记录的 split，快照后新增的整棵 split 不产生差异 | `src/autovision/data/versioning.py:401`（已实证：新增完整 val 仍 ok=True） | 与模块自述 "fail loudly when the data drifts" 相悖，数据漂移漏报 |
| DATA-03 | YOLO mmap 行缓存指纹只含标签文件（`_label_files` 只收 `.txt/.json` 等），不含图像集合；`_mmap_hash` 不含任何 mtime | `src/autovision/data/mmap_helpers.py:93-111`、`src/autovision/data/yolo_object_detection_dataset.py:123-131`、`:146`（已实证：增删图像指纹不变） | 新图**静默不进训练**；删图导致训练中途 `open_image` 崩溃 |
| DATA-04 | 实例分割构造时把全部实例 mask 经 `frPyObjects+merge+decode` 解码为 bool (H,W) 常驻内存，且该类用于训练 split | `src/autovision/data/coco_instance_segmentation_dataset.py:92-162` | COCO train2017 量级需数百 GB 内存；val2017 约 11GB；无懒解码、无上界、无告警 |

### 4.2 P2

| 编号 | 问题 | 证据 |
|---|---|---|
| DATA-05 | disk 图像缓存 sidecar `with_suffix(".npy")` 使 `a.jpg`/`a.png` 共写 `a.npy`：mtime 互不同则缓存永不命中且持续重写；mtime 恰好相等则只校验通道不校验 H/W，可串像素 | `src/autovision/data/image_cache.py:110-146`（已实证抖动） |
| DATA-06 | YOLO loader 对"字段数正确但值非法"的行裸抛 `ValueError`（无文件/行号），绕过自家 `_MalformedLabelLines` 计数协议 | `src/autovision/data/yolo_object_detection_dataset.py:166-169`；对比 convert 路径有文件+行号包装（`src/autovision/data/convert.py:166-173`） |
| DATA-07 | YOLO loader 对未知/被忽略类别直接 `continue`，无计数无告警；COCO 侧同场景有 `undeclared_drops` 计数 + 汇总告警 | `src/autovision/data/yolo_object_detection_dataset.py:167-168`、`src/autovision/data/yolo_oriented_object_detection_dataset.py:67-69` vs `src/autovision/data/coco_object_detection_dataset.py:168-174` |
| DATA-08 | 重复 image_id：检测明确 raise，实例分割/关键点用 dict comprehension 静默覆盖 | `src/autovision/data/coco_instance_segmentation_dataset.py:64-75`、`src/autovision/data/coco_keypoint_dataset.py:85-92` vs `src/autovision/data/coco_object_detection_dataset.py:122-132` |
| DATA-09 | `num_channels=1` 通道契约破坏：非 3 通道走 `UNCHANGED`，RGB 源快路径返回 3 通道、RGBA 返回 4 通道，行为随解码路径漂移 | `src/autovision/data/file_helpers.py:60`、`:99-100` |
| DATA-10 | Label Studio 转换器直接 `Image.open(Path(image_name))`，缺路径逃逸防护；`images_dir / 绝对路径` 在 pathlib 下丢弃前缀，绝对路径指向目录外图片被接受 | `src/autovision/data/interchange/label_studio.py:39-51`；CVAT 有守卫（`src/autovision/data/interchange/cvat.py:43-50`），`resolve_under` 未被 interchange 复用 |
| DATA-11 | 语义/全景分割完全不校验 image–mask 尺寸一致，错误只会以含混形式在 transform 内爆出或静默错位；对比：深度会重采样、实例分割会 resize | `src/autovision/data/semantic_segmentation_dataset.py:115-120`、`src/autovision/data/mask_panoptic_segmentation_dataset.py:216-232` |
| DATA-12 | DDP + `overwrite=True` 时每个 rank 都无条件 unlink mmap+refcount（无 rank 守卫、无锁）；启动时序偏斜下慢 rank 可删掉 rank0 刚发布的文件 | `src/autovision/data/mmap_helpers.py:198-200` |

### 4.3 P3

| 编号 | 问题 | 一句话方向 |
|---|---|---|
| DATA-13 | scan cache 读-改-写无锁，并发可互丢指纹条目 | 并发场景加文件锁或文档声明单写者 |
| DATA-14 | `_has_active_ref` 无锁读 + `exists→read` 竞态，偶发误报 | 移入 FileLock 临界区 |
| DATA-15 | `coco_to_yolo` 静默丢引用未声明 image_id 的标注；不校验 bbox 范围；crowd 框静默变普通标签 | 对齐 `merge_coco` 的告警纪律 |
| DATA-16 | convert 的 YOLO 侧只扫 `images_dir.iterdir()` 顶层，与训练 loader 的 `rglob` 不一致 | 统一 `rglob` |
| DATA-17 | 转换器输出全部非原子直写 | 复用 `_write_atomic` 房规 |
| DATA-18 | VOC XML 解析/数值错误无文件名上下文；split 有但磁盘缺 XML 的 stem 静默跳过 | 包装 + 计数告警 |
| DATA-19 | convert.py 大量裸 `ValueError`，与 `AutoVisionError` 家族并存 | 统一异常家族 |
| DATA-20 | COCO 声明 width/height=0 → 归一化除零裸崩 | 构造期正数校验 |
| DATA-21 | 标签文件二进制/GBK → `UnicodeDecodeError` 裸抛无文件名 | 包装为带文件名的 `AutoVisionError` |
| DATA-22 | 快照 JSON 缺 `splits` 键 → 裸 KeyError | `load_snapshot` 校验 schema |
| DATA-23 | `_data_fingerprint` stat 失败静默降级为纯字符串指纹 | 至少 warning |
| DATA-24 | 深度 `.npy` 损坏裸崩；`.squeeze()` 对 (C,H,W) 歧义 | 校验 + 显式维度处理 |
| DATA-25 | 全景 mask crop 20 次重试全失败后静默返回全空 masks | 至少 warning 计数 |
| DATA-26 | mask 读取宽严不一（多通道报错 vs 静默取 channel 0） | 统一口径并文档化 |
| DATA-27 | `SharedImagePool` 无生产调用方（死代码）；`two_view.py` 每 item 重复解码两次 | 接线或移除 |
| DATA-28 | `list_segmentation_pairs` 列表阶段全量解码所有 mask 求 max_label，大数据集启动 O(N) | 需要时懒求 + 缓存 |
| DATA-29 | COCO 孤儿标注静默忽略无告警（与 merge_coco 不对称） | 计数告警 |
| DATA-30 | `ignore_classes` 含不存在 id 被静默接受 | 校验告警 |
| DATA-31 | CLI 预训练入口 `image_dir=None` 使 H3-2i 的相对路径 mask 匹配永不生效 | 传入相对键或移除该分支 |
| DATA-32 | scan cache 首启在主进程串行解码全数据集，无进度/耗时预估 | 采样/并行/预期提示 |
| DATA-33 | DDP + `overwrite=True` 残余：早于 rank0 到达的非 0 rank 可打开旧 mmap（跨 rank 行集分叉）；挂死已闭合，根治需版本化发布协议 | 实施后复审登记，见 known_limits mmap 缓存键条目 |
| DATA-34 | `resolve_under` 拒绝指向数据根之外的符号链接图像——用符号链接挂共享存储的布局需传真实目录；放宽为词法包含判断属安全设计决策 | 实施后复审登记，见 known_limits 数据加载严格性条目 |

## 5. 整改实施设计

> 状态：**DATA-01..12 已全部实施（工作区，随 0.6.1 发布）**，P3 中的
> DATA-20/21/22/23/29/30 亦已顺手落地；其余 P3 登记在 §4.3 按批次排期。
> 实施与设计的偏差（均已记录在对应小节代码注释与测试中）：
> DATA-03 的主修复落在 `mmap_helpers._data_fingerprint` 的目录分支
> （生产缓存键的实际消费点；`*_data_mmap_hash` 同步对齐）；
> DATA-05 的新鲜度键用来源**字节大小**而非 H/W（读取期无需解码即可
> 校验，语义更强）；DATA-12 按"rank0 删除、非 0 rank 为读者"的最小
> 正确协议实施（残余的读者开到旧文件竞态窗口已在代码注释登记）。
>
> **实施后复审结论（只读代理复审 + 348 项目标回归全绿）：可合入，
> 无 P0/P1。** 复审驱动的追加修复：YOLO 快照纳入镜像 labels 目录
> （原设计只钉 images，会造成"标签已钉"的假信心）；标量 split 值与
> 手改快照条目改为带指引的 `AutoVisionError`；图像缓存截断旗标告警
> 去重为进程级一次；实例分割孤儿标注告警与检测面对齐。
> 复审登记的两条行为边界已写入 `docs/known_limits.md`：
> ① `resolve_under` 拒绝指向数据根之外的符号链接图像（词法包含判断
> 的放宽属设计决策，暂不实施）；② DDP + `overwrite=True` 的跨 rank
> 行集分叉窗口（根治需版本化发布协议，排期 P3）。

### 5.1 DATA-01 快照规格感知标准 YOLO data.yaml（P1）

**问题**：`_load_spec` 把顶层**字符串**值全部归入 `"(shared)"`，只有
顶层 dict 才成独立 split；`_collect_roots` 把类名字符串也当路径。
**目标行为**：标准 YOLO data.yaml（`path/train/val(/test)` + `names`）
能被快照与校验正确解释；不破坏既有 autovision 自有 spec。

**设计**：

1. 新增 `_detect_yolo_spec(payload) -> bool`：顶层 dict 含
   `train` 或 `val` 键且其值为字符串/字符串列表时判为 YOLO 规格。
   显式逃逸口：payload 带 `format: autovision-snapshot` 字段时跳过检测。
2. YOLO 解释规则：`train`/`val`/`test` 各成独立 split（字符串或列表
   均可；相对路径先接 `path`（若存在）再接 spec 文件所在目录）；
   `names`（dict 或列表）记录进快照 JSON 的 `names` 字段但**绝不参与
   路径收集**；其余未知键按现有规则（仅 dict-of-dicts 成 split）。
3. `verify_dataset_snapshot` 走同一解释器（复用 `_load_spec` 结果），
   保证快照与校验对 spec 的理解一致。

**涉及文件**：`src/autovision/data/versioning.py`（`_load_spec`、
`_collect_roots`、`_digest_files` 的 split 组装段）。

**兼容性**：既有 autovision spec（无 train/val 顶层字符串键）行为
不变；`"(shared)"` 分支仅在不识别为 YOLO 时生效。`known_limits.md`
若登记过该限制则同步删除。

**测试计划**（新增，建议名 tests/test_data_versioning_yolo.py）：
标准 data.yaml fixture（path/train/val/names）→ 快照 splits ==
["train","val"]；strict 下缺失目录报错信息只指向 train/val 不指向
类名；names 为 dict（数字键）与列表两种形态；既有 autovision spec
fixture 回归不变。

**验收标准**：新测试全绿；`tests/test_data_versioning.py` 既有用例
全绿。

### 5.2 DATA-02 校验对"快照后新增 split"发声（P1）

**问题**：校验循环只遍历快照内记录的 split（`versioning.py:401`）。
**设计**：

1. 校验入口先解析当前 spec 得到声明 splits 集合，与快照记录集合求
   差集：`added_splits`（spec 有、快照无）、`removed_splits`（快照有、
   spec 无，现状已覆盖为目录差异，保留）。
2. `added_splits` 非空 → `ok=False`，diff 报告新增
   `added_splits: [{name, files, bytes}]`（按 DATA-01 的解释器取
   roots 统计）；strict 语义沿用现有返回值约定。
3. spec 缺失/不可解析时降级为现状行为（只比对快照内 splits）并
   warning 一次。

**涉及文件**：`src/autovision/data/versioning.py`
（`verify_dataset_snapshot`）。**测试计划**：快照只含 train 后新增
完整 val → ok=False 且报告含 added_splits；spec 被删 → 降级路径
warning；现有 verify 用例回归。**验收标准**：漂移漏报场景在测试中
从 ok=True 变为 ok=False。

### 5.3 DATA-03 YOLO 行缓存指纹纳入图像清单（P1）

**问题**：行集合由图像目录 `rglob` 决定，但缓存指纹只看标签文件。
**设计**：

1. `yolo_object_detection_dataset.py` 的 `train_data_mmap_hash` /
  `val_data_mmap_hash` 追加一项图像清单摘要：对 `rglob` 图像集合计算
  sorted `(relpath, size, mtime_ns)` 的 sha256（与 `_data_fingerprint`
  同构）。
2. 摘要在 args 内按 `(images_dir, split)` 进程内缓存（rglob 已在
  构建行集合时执行过一次，复用其结果，避免二次扫盘）。
3. `mmap_helpers.py` 的 docstring 同步改写："YOLO 变体的行集合由图像
  清单定义，标签文件仅提供标注内容"。

**涉及文件**：`src/autovision/data/yolo_object_detection_dataset.py`、
`src/autovision/data/mmap_helpers.py`（docstring）。**测试计划**：
向 train/images 增加图像 → hash 变化；删除图像 → hash 变化；仅改
标签内容 → hash 变化（既有行为保持）；空 images 目录（纯背景）→
hash 稳定。**验收标准**：增删图像后 mmap 缓存失效（重建行集合），
不再出现"新数据静默丢单"。

### 5.4 DATA-04 实例分割 mask 懒解码（P1）

**问题**：构造期全量解码 mask 常驻内存。**设计**（分两步走）：

1. **0.6.x 步（懒解码）**：`self._rows` 改存 RLE（`counts` bytes +
   size），`__getitem__` 按需 `frPyObjects/merge/decode`。内存量级从
   数百 GB 降到 MB 级；解码成本从构造期一次性转移为逐 item（与检测面
   行解析同粒度，DataLoader worker 并行消化）。
2. **0.7.x 步（可选，另立条目排期）**：RLE 行并入 mmap 行缓存，与
   检测面对齐。
3. 解码失败（counts 损坏）→ 包 `AutoVisionError` 带标注索引与
   image_id；多边形 RLE 与 uncompressed RLE 两条路径都覆盖。

**涉及文件**：`src/autovision/data/coco_instance_segmentation_dataset.py`
（构造与 `__getitem__`）、`src/autovision/data/coco_keypoint_dataset.py`
（同构处理不做——其 mask 语义不同，保持现状）。**测试计划**：小
COCO fixture 上懒路径与急切路径 mask 逐位一致（golden 对照）；构造
10 万实例的合成标注，断言进程 RSS 上界（量级断言，非精确值）；损坏
counts → 报错带索引。**验收标准**：COCO val2017 量级数据可在默认
内存配置下完成构造；训练 smoke 不变。

### 5.5 DATA-05 图像缓存 sidecar 以完整文件名区分（P2）

**设计**：sidecar 路径从 `path.with_suffix(".npy")` 改为
`path.with_name(path.name + ".npy")`（`a.jpg` → `a.jpg.npy`）；
`_read_sidecar` 的 meta 增加 H/W 校验，不匹配按 miss 处理并删除
陈旧 sidecar。**兼容性**：旧格式 sidecar 成为孤儿——在缓存目录
docstring 注明一次性手工清理即可（不做自动迁移）。**测试计划**：
`a.jpg` + `a.png` 同目录各自命中；同 stem 不同尺寸两文件互不串
像素。**验收标准**：两用例通过，`tests/test_image_cache.py` 回归。

### 5.6 DATA-06 YOLO 值解析错误并入弃用协议（P2）

**设计**：把 `int(parts[0])` / `float(v)` 包进 try，失败时走与字段数
错误相同的 `_MalformedLabelLines` 计数通道（消息含文件与行号），对齐
加固计划 H3-2a 的两阶段安排：当前版本计数 + 一次 warning；0.7.0 由
计划中的 `on_malformed_line` 参数统一承载 raise 语义。**涉及文件**：
`src/autovision/data/yolo_object_detection_dataset.py`、OBB 变体同改。
**测试计划**：`0 0.5 not_a_number ...` 行 → 计数 + warning、训练
不中断；`tests/test_data_hardening.py` malformed 行用例回归。

### 5.7 DATA-07 YOLO 未知类别显式计数告警（P2）

**设计**：`class_id not in internal` 分支计数（含文件与行号样本），
epoch/构造末尾汇总一次 warning（对齐 COCO `undeclared_drops` 的文案
结构）。OBB 变体同步。**测试计划**：含手误类别号的 fixture → warning
文本含被丢条数；无未知类别时零告警（不噪声化）。

### 5.8 DATA-08 重复 image_id 三面统一为 fail-loud（P2）

**设计**：实例分割与关键点加载器复用检测加载器的重复 id 检测
（构造期 `AutoVisionError`，消息含重复 id 与两个来源文件）——对齐
加固计划 H3-2b 已实施的检测面语义。**测试计划**：三任务面各一条
重复 id fixture 均报错；`tests/test_data_hardening.py` 检测面用例
回归。

### 5.9 DATA-09 通道契约按请求归一（P2）

**设计**：`open_image` 语义改为"返回值通道数由调用方声明的
`num_channels` 决定"：快路径解码后若与请求不符（RGB 源 + 1 通道请求
/ RGBA 4 通道），按现有 PIL 兜底路径的转换规则补齐（1 → "L"，
3/4 → "RGB"），并在 docstring 写明映射表。`num_channels=None` 保持
现状（不转换）。**测试计划**：jpg/png 各配 num_channels=1 → 形状
(1,H,W)；RGBA PNG 默认请求 → 3 通道；`tests/test_data_hardening.py`
回归。**验收标准**：`num_input_channels=1` 的模型不再收到 3 通道
张量。

### 5.10 DATA-10 interchange 统一走 `resolve_under`（P2）

**设计**：Label Studio 转换器的图像定位改为
`resolve_under(images_dir, image_name)`；绝对路径或逃逸 → 报
`AutoVisionError`（消息含原始 image_name）。CVAT 保持现有守卫，
common 装配层加一条注释指明唯一入口。**测试计划**：相对路径正常；
`../` 逃逸与绝对路径（目录外真实文件）均报错；`tests/test_interchange.py`
回归。

### 5.11 DATA-11 语义/全景分割尺寸校验（P2）

**设计**：在配对列表阶段或 `__getitem__` 校验 mask 的 (H, W) 与图像
一致，不一致抛 `AutoVisionError`（消息含两尺寸与文件对路径）。分割
标签不做插值 resize（有损且语义错误），与深度（重采样合理）区分的
理由写入 docstring。**测试计划**：不匹配 fixture → 报错带双尺寸；
`tests/` 语义/全景既有用例回归。

### 5.12 DATA-12 overwrite 删除收敛到 rank0（P2）

**设计**：`mmap_helpers.py` 的 overwrite 分支改为：FileLock 内仅
rank0 unlink mmap+refcount；非 0 rank 跳过删除（其角色是读者）。
**测试计划**：模拟 2 rank 时序偏斜（rank1 先进入 overwrite）→
rank0 发布的文件存活；`tests/test_mmap.py` 双进程用例回归。

## 6. 实施顺序

DATA-01 → DATA-02（同一文件连续改，一次评审）；DATA-03；DATA-04
（独立分支，涉及内存语义）；P2 按 05→12 任意顺序，建议 10、12 先行
（安全与正确性敏感）。每项独立 commit，标题
`fix(data): DATA-nn <摘要>`。

## 7. 风险与回滚

- DATA-01/02 改变快照 JSON 的 splits 结构（YOLO 场景从 `["(shared)"]`
  变为真实 split 名）——旧快照文件校验时按其记录内容比对，不受影响；
  回滚即 revert。
- DATA-04 懒解码改变构建耗时分布（构造变快、首 epoch 略慢）——
  golden 对照保证数值不变；回滚 revert。
- DATA-03 使 YOLO 缓存失效更频繁（图像集变动的 run 重建行集合）——
  这是修复目标本身，不是回归；指纹计算复用 rglob 结果，额外开销一次
  哈希。

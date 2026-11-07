# 变更日志

本文件记录 AutoVision 的对外变更。格式遵循 [Keep a Changelog](https://keepachangelog.com/zh-CN/1.1.0/)，
版本语义遵循 [Semantic Versioning](https://semver.org/lang/zh-CN/)；发布流程与版本政策见
`docs/release_process.md`（站点页「发布流程」）。

`0.1.x` 为内部开发线（未对外发布，无变更记录）；本文件自 `0.2.0` 起对外。

## [0.6.1] - 2026-11-07

### 安全修复

- **serving 认证默认 fail-closed（加固计划 H2-1）**：`serve()` 绑定非回环
  地址且未提供 `api_key=` / `AUTOVISION_SERVING_API_KEY` 时拒绝启动；显式
  `AUTOVISION_SERVE_ALLOW_INSECURE=1` 可接受开放端点。`create_app` 作为库
  入口保持 opt-in 语义不变。
- **资源上界（H2-2）**：`serve()` 暴露 `limit_concurrency`（默认 64）与
  `timeout_keep_alive`（默认 30）；`create_app` 新增
  `max_request_pixels`（默认 `max_batch × max_image_pixels`），批量解码
  累计像素超限返回 413；`max_image_pixels` 默认 40M → 16M。
- **api_key 不落环境变量（H2-4）**：`workers>1` 且显式传入 `api_key=` 时
  拒绝启动（多 worker 路径会把配置序列化进同 UID 可读的环境变量），key
  必须走 `AUTOVISION_SERVING_API_KEY`。
- **4xx 错误信息脱敏（H2-5c）**：跨越引擎/序列化边界的 `AutoVisionError`
  不再原文回传（可能含服务器路径），映射为稳定短语 + request_id，原文进
  服务端日志；纯客户端输入错误（base64、图片解码）保留原文。
- **`X-Request-ID` 清洗（H2-5b）**：请求 ID 先校验
  `[0-9a-zA-Z-]{1,64}`，不合格截断并中性化后再进日志/响应头/A-B 分桶，
  消除日志伪造面。
- **A/B 分桶键不可客户端操控（H2-5a）**：分桶键改为服务端逐请求签发的
  routing key（此前用客户端可控的 `X-Request-ID`，可通过重放同一 id 粘住
  指定变体）；`X-AB-Variant` 定向改为 opt-in——`create_app` 新增
  `allow_variant_override`（默认 `False`，请求头被忽略），测试场景显式
  开启后未知变体名仍返回 400。
- **可下载资产下载加固（H3-4i）**：`resolve_downloadable` 改为分块流式
  写入（`.part` + `os.replace`，常量内存）+ per-target FileLock 互斥
  （并发 resolve 同一资产不再互踩）+ (path, size, mtime, expected) 摘要
  缓存（进程内重复 resolve 不再重复全量哈希）；哈希不匹配保持
  fail-closed 删除并重下。

### 修复

- **误差分析匹配器 IoU 行错位（H1-1）**：输入预测分数未按降序排列时
  （离线 JSON 导入的常见形态），TP/FP 归因系统性错位并污染难例采样；
  IoU 行索引改为经 `sorted_idx` 映射。
- **crowd-ignore 类别一致性（H3-6d）**：误差分析的 ignore 判定只考虑
  与预测**同类**的 crowd GT（对齐 pycocotools）——异类 crowd 框不再把
  真实 FP 掩蔽为 ignored。
- **run_stream 永久挂起（H1-2）**：reader 结束时队列满 >5 秒会丢失
  sentinel 使主循环永久空转；sentinel 改为无条件重试（只阻塞 reader
  线程），主循环增加"reader 已死且队列空且无 sentinel"守卫。
- **run_stream 丢帧策略缺陷（H3-5b）**：新增 `max_lag_frames` 参数
  （默认 None=现状），设置后积压超限即 drop_oldest 并计入
  `StreamRunResult.frames_dropped`——修复初版实现"积压检查放在
  `queue.Full` 异常之后永不触发"的缺陷。
- **ByteTrack 两处无界泄漏（H1-3）**：匹配失败的 unconfirmed 轨迹现在
  立即移出 `self.tracks`（此前每帧对其做 Kalman 预测且永不清理）；
  `run_stream` 每批调用 `drain_removed_tracks()`（D2 契约的首个 src
  消费方），退役轨迹计数暴露为 `StreamRunResult.tracks_removed`。
- **断流重连后状态重置（H3-5c）**：重连成功后 tracker/smoother/zone
  状态重置（断流是场景不连续，ID 从 1 重新开始、计数清零）；
  `reconnects=0` 在"断流后未尝试重连即结束"时输出 warning，不再与
  clean EOF 不可区分。
- **multi_stream 加固（H3-1e/H3-5a/d/e/f）**：逐源序列化异常隔离；
  `batch_size` 成为每源每轮实际批组装上限；重连预算随连续健康运行
  重置（累计 reconnects 统计保留）；分辨率变化轮换 writer 分段文件
  （不再覆盖旧段）；每源滑动窗口失败率熔断（最近 20 帧 >50% 失败即
  中止该源并标记 `aborted`），持续 OOM 不再假成功。
- **DDP 种子三件套（H1-4）**：`DistributedSampler` 现在接收运行 seed
  （此前固定 seed=0，多卡数据顺序与 seed 脱钩）；`Engine.fit` 的
  epoch loader 重建补齐 `worker_init_fn/generator/prefetch_factor/
  persistent_workers`；CLI loader generator 按 rank 偏移。多卡训练的
  数据顺序与 0.6.0 不同（复现性修复的预期结果）。resume 现在恢复
  sampler epoch（H3-3b，旧 checkpoint 缺键回退 0）。
- **DDP 验证指标精确性（H3-3c/e）**：验证集分片改为无填充跨 rank 划分
  （`_StridedValSampler`）——padding 复制样本不再被计两次，DDP 验证
  指标与单进程完全一致；`world_size > len(dataset)` 显式报错（此前
  空 epoch 静默通过）；per-rank RNG 状态随 checkpoint 分片保存/恢复
  （H3-3a，旧 checkpoint 回退 rank0 流）。
- **`export_weights` 原子化（H1-5）**：导出权重走 checkpoint 同款
  tmp+fsync+`os.replace`；周期/早停导出补 rank 写失败屏障，rank0 写
  失败不再让其余 rank 挂到 NCCL watchdog 超时。
- **TensorRT 引擎 / OpenVINO IR 原子发布（H1-6/H1-7）**：两者改为
  staged 写入 + parity 门禁通过后 `os.replace` 发布——门禁失败只删
  暂存件，旧好工件不再被"先覆盖再删除"；sidecar 同样原子化。
- **TRT 精度容差与暂存件（H3-4a/b）**：`precision="auto"` 按构建决策点
  回读的**实际**精度选 parity 容差（平台降级到 fp32 不再被 fp16 宽容差
  掩盖）；fp32/fp16 路径的 staged ONNX 在引擎构建成功后清理（未对拍的
  `.onnx` 不再与已验证 `.engine` 并存误导部署）。
- **fp16 门禁 CUDA-EP 路径兑现（H1-8）**：onnxruntime 带 CUDA EP 时
  parity 会话实际路由 CUDA EP（此前警告承诺的路径在代码中不存在）。
- **verify 钩子 precision 中央兜底（H3-4c）**：不接受 `precision`
  参数的任务钩子改为不传该参调用 + 每钩子一次 warning（此前直接
  TypeError 或以 fp32 容差静默校验 fp16）。
- **导出工件携带工具链版本（H3-4d）**：`autovision_toolchain`
  （torch/onnx/opset/onnxruntime/tensorrt/openvino）进 ONNX in-graph
  元数据与 TRT/OV sidecar；`InferenceEngine` 加载时版本不匹配输出
  warning。
- **截断图像全局开关隔离（H1-10）**：掩码解码的
  `LOAD_TRUNCATED_IMAGES` 宽松旗标收进作用域上下文，不再泄漏到图片
  解码——修复分割任务中截断 JPEG 被静默解码进训练。
- **数据加载健壮性（H3-2e/f/g/h）**：COCO/VOC/YOLO 各加载器的损坏/
  截断 JSON、缺键、bbox 长度错统一包装为带文件名的 `AutoVisionError`；
  `yolo_to_coco` 增加归一化坐标校验（越界/非正宽高报错含行号），标签
  文件显式 `utf-8-sig` 编码（BOM 兼容）；mmap refcount 的
  exists→open→increment 竞态移入 FileLock 临界区（新增 4 进程并发
  用例）；版本快照对文件消失报 `AutoVisionError` 且快照 JSON 原子写。
- **掩码匹配跨目录错配（H3-2i）**：semantic/panoptic/depth/ImageDataset
  的掩码匹配键优先相对路径，嵌套目录中同名 stem 不再互相错配（回退
  stem 时对歧义告警一次）；扁平目录行为逐位不变。
- **空验证 shard 按 0 平均（H3-3d）**：DDP 验证损失改为分子/分母分别
  归约，某 rank 零 batch 不再稀释全局均值。
- **registry 版本混排崩溃（H3-4g）**：stage 混入非语义版本号（如
  `rc1`）时 `get(version=None)` 不再 TypeError；数值最新优先并告警。
- **registry 晋升失败审计（H3-4j）**：promote 门禁失败也写
  `promote_failed` 审计条目（含 gate 名与原因摘要，hash-chain 不变）。
- **许可证判定统一（H3-4h）**：deployment_gates 的 allowlist 收敛为
  `compliance/licenses` 单一来源；模型卡 license 统一过
  `normalize_license`（"Apache Software License" 等别名不再误报）。
- **契约层健壮性（H3-1b/d/f/h）**：`(H,W,2)` 缺 `segment_ids` 的
  panoptic 图不再被误判为 N 个实例（box-less 形状启发式 + 构造期
  warning）；`_summarize` 对非数值数组降级为 shape/dtype/length 摘要
  （此前崩溃）；`select()` 空结果抛 `AutoVisionError`；新增
  `ResultValidationError`（`AutoVisionError` 与 `ValueError` 双继承）
  替换 results.py 全部裸 `ValueError`（`except ValueError` 调用方
  不受影响），annotators 的骨架校验改抛 `ConfigValidationError`。
- **可视化一致性（H3-1i/j）**：MaskAnnotator 对分辨率不匹配的 mask
  栈做 NEAREST resize（此前静默跳过全部实例），与语义路径对齐；
  NaN score 渲染为 `n/a`；`filter_scores` 对含 NaN 输入 warning 一次。
- **CLI 与指标（H3-6a/b/c/e/f）**：`val` 的 `AutoVisionError` 输出单行
  错误 + exit 1（此前裸 traceback）；`plot-curves` 缺 `run=` 改为
  exit 2 用法错误；`download`/`predict` 接入未知键拒绝；`auto_raise`
  不再永久改写 `max_dets` 属性（compute 内局部抬升，两次 compute
  读数可比，抬升时告警一次）；val 输出声明 headline 口径
  （argmax-F1 点读数，非 COCO summary）；analyze-errors 支持
  predictions 信封格式（`categories` 显式映射）并在裸列表时告警排序
  假设；HPO 逐 trial 失败聚合透出首个原始异常（拼错键不再只报
  "no successful trial"）。
- **serving 摘要 depth 可见性（H2-5f）**：`/predict` 响应中 depth 深度图
  因 `max_json_elements` 被摘要（`summarized: true`）时，每条结果附加
  `depth_truncated: true`——形状/摘要不是像素，客户端可据此回退到
  文件/对象存储通道而非静默拿到不可用数据。
- **数据快照原生支持 Ultralytics data.yaml（DATA-01/02/22）**：
  `dataset_snapshot` / `verify_dataset_snapshot` 把 `path/train/val` +
  `names` 解释为 YOLO 规格（此前 train/val 被折叠进 `"(shared)"`、类名
  被当路径收集，strict 下直接报错）；镜像 `labels/` 目录一并纳入快照
  （标签内容同样被钉住）；校验对"快照后新增的 split"报
  `added_split` 差异（此前完全失明）；截断/手工编辑的快照（缺
  `splits`/`spec` 或条目非对象）响亮报错而非裸 `KeyError`。YOLO 快照
  额外记录 `names` 类别表。
- **YOLO/OBB 行缓存指纹纳入图像清单（DATA-03）**：mmap 缓存键的目录
  指纹现包含图像文件（路径/大小/mtime）——增删图像正确失效缓存，此前
  新图静默不进训练、删图训练中途崩溃；指纹 stat 失败的降级路径输出
  warning（DATA-23）。
- **COCO 实例分割懒解码（DATA-04）**：行保存紧凑 RLE、`__getitem__`
  按需解码（此前构造期把全部实例 mask 解码为 bool 数组常驻内存，COCO
  train2017 量级需数百 GB）；损坏 RLE 报 `AutoVisionError` 并指明行与
  图像。
- **图像 disk 缓存 sidecar 修复（DATA-05）**：sidecar 命名从
  `<stem>.npy` 改为 `<完整文件名>.npy`（`a.jpg` 与 `a.png` 不再共用
  `a.npy` 互相踩踏）；新鲜度键加入来源字节大小（`cp -p` 保 mtime 的
  内容替换可检出）；sidecar meta 版本升为 2，旧格式 sidecar 自动失效
  重建（旧文件成为孤儿，可手工删除）。
- **YOLO/OBB 标签加载加固（DATA-06/07/21）**："字段数合法但值非法"的行
  改走与字段数错误相同的计数 + deprecation warning 通道（此前裸
  `ValueError` 无文件/行号直接中断训练）；未知类别行计数并汇总
  warning（此前静默丢标注；`ignore_classes` 命中的行保持静默）；二进制
  /GBK 标签文件报错带文件名。
- **重复 image_id 三面统一（DATA-08）**：实例分割与关键点加载器对重复
  `image id` 报错（带两个来源文件名），与检测面 H3-2b 语义一致。
- **`open_image` 通道契约（DATA-09）**：返回通道数强制等于
  `num_channels`——`num_channels=1` 对 RGB/RGBA 源现在返回 1 通道
  （此前随解码路径返回 3/4 通道）；`num_channels` 非 1/3 报
  `AutoVisionError`。
- **Label Studio 路径防逃逸（DATA-10）**：图像定位统一走
  `resolve_under`，绝对路径或 `..` 穿越指向 `images_dir` 之外时报错
  （此前 pathlib 前缀丢弃使外部文件被打开并原文写入 COCO `file_name`）。
- **分割图-掩码尺寸校验（DATA-11）**：语义/全景分割在读取期校验
  image/mask 空间尺寸一致（带两个文件名与尺寸），不再以 transform
  深处的含混错误暴露。
- **mmap overwrite 删除收敛 rank0（DATA-12）**：`overwrite=True` 下仅
  rank0 删除 mmap + refcount（DDP 时序偏斜下慢 rank 不再删掉 rank0
  刚发布的文件并拖满超时）。
- **COCO 加载校验补强（DATA-20/29/30）**：非正 `width/height` 在检测/
  实例分割面构造期报错（带文件名，此前裸 `ZeroDivisionError`）；孤儿
  标注（`image_id` 未在 `images` 声明）计数并汇总 warning（与
  `merge_coco` 对齐）；`ignore_classes` 中未声明 id 输出一次 warning。

- **预训练 resume 漂移守卫 + 调度器两段式恢复（TRAIN-01）**：预训练
  checkpoint 现携带 fit_args/数据指纹快照，resume 时复用 D17 漂移检查
  （此前完全静默）；`CosineWarmupScheduler` 的恢复改为两段式——步计数
  随 state dict 回灌、调度形状（max_epochs/warmup/end_value/base_lrs）
  永远来自当前调用，续训调大 `max_epochs` 后 LR 按新终点继续衰减（此前
  被旧形状冻结在 end_value）。resume 时更换 base LR 现在按"当前调用
  governs"生效：首步仍用 checkpoint 的参数组 LR，其后按新 base 衰减
  （修复前换 LR 被完全忽略）。fit_task 侧调度器形状漂移（resume 扩展
  steps 而调度形状被 checkpoint 固定）输出指名 warning。
- **pretrain 导出原子化（TRAIN-02）**：`exported_last.pt` 改走
  tmp + fsync + `os.replace`（对齐任务导出的 H1-5 纪律），写入失败保留
  旧导出且无 tmp 残留；导出段抽为 `_export_embedding_model` helper。
- **多 rank resume 一致性对账（TRAIN-03）**：resume 后对
  (start_step, best_metric) 做 MIN/MAX all_reduce——非共享文件系统导致的
  跨 rank 状态分叉从"挂满集合通信超时"变为指名错误（含共享盘指引）。
- **`best_metric.json` sidecar 原子写（TRAIN-04）**：改走与 checkpoint
  相同的原子写路径，崩溃不再留下撕裂的 JSON（读取端本就容错，但撕裂
  会静默重置 best 基线）。
- **warm-start/resume 结构不匹配友好报错（TRAIN-05）**：类别数/query 数
  变更后加载 checkpoint 报 key 级 diff 摘要（缺失/多出/形状不匹配计数与
  首批键名），替代 torch 原生多行 size mismatch；删除了永远落空的
  `load_train_state_dict` 实例属性赋值（hook 按是否有该属性保留）。
- **resume 已完成 run 的可见性（TRAIN-12）**：预训练与 fit_task 对
  "已跑满全部 epoch/step"的 resume 输出 info 提示与扩展指引（此前静默
  返回 0 步结果）。
- **杂项（TRAIN-07/09/11/15）**：删除 `accumulated_loss` 死代码；
  `CheckpointMetadata` 新增无歧义的 `step` 字段（fit_task 的 `epoch`
  历史上承载 step，字段向后兼容）；合并重复的引擎常量定义；
  `Engine.fit` 对 batch_sampler 型 DataLoader 报 `AutoVisionError`
  （此前在装配深处裸 TypeError/除零）。

- **multi_stream 视频写手隔离（INFER-01）**：单源的
  `AnnotatedVideoWriter` 创建/写入失败（磁盘满、编解码器故障）不再穿透
  `run()` 终止整个多源运行——该源按既有熔断语义 abort、其余源继续，
  summary.json 保证落盘（含未包容失败的最后兜底）。
- **run_stream 逐帧隔离（INFER-03）**：新增 `on_frame_error`（默认
  `"skip"`）——单帧畸形记录或非有限 tracker 输入记录到 JSONL
  （`{"frame", "error"}` 行）并计入 `StreamRunResult.frames_failed`，
  24/7 源不再因一帧死亡；`"raise"` 保留历史 fail-fast；连续超过 10 帧失败
  仍中止（管线整体坏死不静默空转）。track=True + NaN 分数场景即此修复的
  靶心。
- **推理工件元数据闸门可选 fail-closed（INFER-04）**：`InferenceEngine`
  新增 `require_metadata`——无元数据工件在 CLI predict 与 serving 入口
  被拒绝（此前身份闸门对其整体空转），库用途默认保持宽松但升级为
  warning。
- **后端输出契约运行时校验（INFER-05/09）**：`run_batch` 校验记录数与
  输入一致、每条记录至少携带一个 canonical 结果键、记录必须是 dict——
  后端图与 torch 参考布局不一致从"静默错误坐标或晦涩异常"变为指名
  backend 的错误；空批量报 `AutoVisionError`（此前裸 RuntimeError）。
- **TraceAnnotator 历史上界（INFER-06）**：新增 `max_tracks`（默认
  512，`None` 显式不设限）——长时服务每个出现过的 track_id 不再永久
  占据一条轨迹历史；驱逐按"最后触达"顺序，当前帧 id 受保护。
- **推理杂项（INFER-02/07/08/11/12/13/16）**：float 语义 label map 渲染
  按四舍五入统一两条分支（此前随分辨率一崩一活，与序列化侧 H1-11 的
  弃用路径对齐）；stream.py 的记录键改用 `autovision.keys` 常量；
  label-less 记录的 Track 重建保留 `det_index`（join-back 契约修复）；
  16-bit 帧按自身满量程归一（此前 /255 启发式回绕损坏）；
  TraceAnnotator docstring 撤销"stream 会写 tracker_id 进 extra"的失真
  声明；TrackSmoother 对同帧重复 id 报错；VideoFrameStream 并发二次
  迭代报错（此前共享 capture 未定义行为）。

- **registry CLI 先校验后变更（SERVE-01）**：`registry` 命令的未知键
  校验移到分发之前（按动作白名单）——typo 键此前会**先执行
  register/promote/rollback 再报 usage error**，留下已生效的变更与
  重复审计风险。
- **服务端故障返回 5xx（SERVE-02）**：跨越引擎/序列化边界的
  `AutoVisionError` 从 400 改为 500（稳定短语 + request_id 的脱敏不变）；
  客户端输入错误保持 400。修复重试策略、SLO 错误率区分与告警分诊语义。
- **serve CLI 数值与布尔参数守卫（SERVE-03/11）**：`port=`/`workers=`
  非法值从真实终端的 traceback 变为 usage error（exit 2，含合法范围）；
  `strict=`/`maximize=`/`force=` 的布尔解析改为严格（`strict=False`
  此前被 `bool("False")=True` 静默反转）。
- **解码预算与推理预算分离（SERVE-04）**：新增 `decode_timeout_s`（默认
  `min(10, inference_timeout_s / 2)`）——解码超时返回 504 并计数
  （`autovision_decode_timeouts_total`），**不再毒化健康引擎**（此前纯
  解码超时把 variant 永久置为 503 直到重启）；毒化契约保留给卡死的
  forward（H2-3 语义不变）。
- **warmup 超时（SERVE-05）**：新增 `warmup_timeout_s`（默认 60s）——
  引擎在 warmup 中挂死（CUDA 停滞、NFS 工件）从"启动永久不进入
  serving 状态"变为指名 variant 的启动失败；`0`/`None` 显式禁用
  warmup（挂死线程无法取消，进程退出可能被阻塞——见 known_limits）。
- **限流拓扑正确性（SERVE-06/08）**：桶键按端点命名空间化（`/metrics`
  抓取不再与 `/predict` 互挤）；`trust_proxy_headers` 布尔弃用（0.7.0
  移除），新参数 `trusted_proxy_hops=N` 按标准 XFF 链语义从右选第 N 项
  为客户端（反代下全体客户端不再坍缩为一个桶）；predict 的 429 计入
  延迟直方图（`error_class="rate_limited"`）。
- **workers>1 父进程预校验（SERVE-07/09）**：新增
  `validate_serve_config`（纯函数，覆盖 create_app 全部纯可校验参数）——
  workers>1 的坏配置从"每个 worker 内爆炸 + uvicorn 重启循环"变为父进程
  单条干净错误（未知键同样指名）；多 worker 启动后清理
  `AUTOVISION_SERVE_CONFIG` 环境残留。
- **CLI 辅助层异常家族统一（SERVE-12/13）**：`common_helpers`/
  `train_helpers` 的配置错误统一抛 `ConfigValidationError`（此前裸
  `ValueError`，调用方无法用单一 `except AutoVisionError` 收口）；
  `epochs=` 的浮点/非法值从静默截断/裸崩变为指名错误。

- **OBB mAP 两处正确性修复（METRIC-01/02，数字口径修正）**：PR 流改按
  每图"GT∪预测类别并集"构建——无 GT 图像上的背景 FP 此前被静默丢弃，
  AP 系统性虚高（审计复现：官方语义 0.5 的场景返回 1.0）；AP 积分改为
  标准的"全序列单调包络 → 前向积分"（旧实现交错进行，同一 recall 的
  高精度点无法回补，AP 系统性虚低，0.5 vs 1.0）；新增
  `interpolation="all-point"|"coco101"` 两挡并建 golden 基线
  （tests/golden/obb_ap.json，含两个审计复现向量）；预测/GT 记录数不匹配
  响亮报错（zip 静默截断曾使审计场景成立）。**修复后所有历史 OBB 数字
  与 0.6.x 之前不可比**（双向失真同时被纠正）。
- **crowd（iscrowd）验证链路端到端接通（METRIC-03）**：COCO 行的 crowd
  标志此前在 `__getitem__` 被丢弃、四个检测任务的 `update_targets` 均不
  携带——评估器能力存在但主路径失效，训练期 mAP 把 crowd 框当普通 GT
  计 FN（val2017 有 446 个）。现在 dataset→collate（含 mixup/copyblend/
  小框过滤三处）→batch→train_model 全链携带；直接构造的 batch 缺键时
  回退全零（pre-fix 语义）。**含 crowd 数据的训练期 mAP 数值上升
  （修复目标）**。
- **headline P/R/F1 改为单操作点读数（METRIC-04，设计 A）**：新增
  `headline_score_threshold`（默认 0.50）——precision/recall/F1 在同一
  score 操作点计算（IoU 0.5、crowd 忽略、按有 GT 类宏平均，复用误差分析
  匹配器）；旧实现扫 pycocotools 的 recall 阈值轴与固定 recall 配对，
  F1 系统性高于其配对 recall 且披露文案失真。`HEADLINE_PROTOCOL` 文案
  重写为真实语义；**headline 三个数与 0.6.x 之前不可比**（AP/AR 不变）。
- **指标健壮性（METRIC-07/08/09 + P3）**：深度指标改流式标量累计
  （对齐先于累计，`merge_state` 加法合并——此前全量 (pred,gt) 深度图
  驻留内存，NYU 级数十 GB）；误差分析 report.json 的 matches 改 top-K
  截断（默认 1000，`matches_total`/`matches_truncated` 标记）并删除
  重复存储的 confusion；分割指标对 target 越界/负值响亮报错（此前裸
  bincount 崩溃）；COCO gt.json 的 categories 写真实类名；auto_raise
  高水位随 reset 清零；`match_detections` 对 per-image 数量不匹配报错；
  画廊文件名含父目录去重 + 单张失败跳过（此前互相覆盖/整体崩溃）；
  混淆矩阵/误差分析拒绝重复类名；pose 评估在 DDP 下缺 `image_ids`
  响亮报错；`TaskMetrics` 标量列可拼接；全景指标加 TP 匹配互斥（有效
  输入下不可达，防御性保留）。
- **合规包加固（METRIC-05 + P3）**：`compliance-report` CLI 默认
  fail-closed（`fail_on="error"`，此前默认恒通过；`strict=` 弃用——
  strict=False 不再关闭门禁，用 fail_on=none 显式旁路）；合规四件套
  原子写；SBOM serialNumber 改确定性 uuid5（同内容同编号）；SBOM 无
  timestamp 时省略字段（CycloneDX RFC-3339）；markdown 表格单元全面
  转义；依赖清单按 PyPI 大小写不敏感名称去重；缺失模型文件/损坏快照
  进 issue 清单而非中断报告。

- **注册表孤儿条目自愈（EXPORT-01，P1）**：register/promote 被 kill 于
  "目录已建、manifest 未写"窗口时留下的残目录此前永久卡死该
  (name, version)（"versions are immutable" 误判）。现在三态处理：
  已发布（manifest 存在）→ 不可变拒绝；新鲜孤儿（<10 分钟）→ 提示
  可能有并发注册进行中；陈旧孤儿 → 自动清理 + warning 并重试成功。
  `get()` 对无 manifest 目录报 "Incomplete registry entry"（与 list 的
  跳过行为一致）。
- **fp16 验证强度进工件元数据（EXPORT-02）**：新增
  `autovision_verify_strength`（`full` / `smoke-only`）——fp16 在无 CUDA
  EP 机器上 verify=True 只跑冒烟门禁时，工件携带机器可读的"未做数值
  对拍"标记，加载侧输出 warning；新增 opt-in 的 `VerifyStrengthGate`
  部署门禁（要求 full 才可晋升）。验证门禁先于元数据写入执行。
- **注册表补齐 TRT/OV 身份 sidecar（EXPORT-03）**：`_copy_sidecars`
  现按加载器的 `with_suffix` 语义复制 `model.meta.json`（TRT）与
  `model.json`（OV）身份载体——此前从注册表路径加载 TRT/OV 工件时身份
  门禁整体静默失效。
- **注册表读路径瘦身（EXPORT-04）**：`get`/`current`/`rollback` 新增
  `verify=` 参数（默认 True 保持现语义）；digest 记忆化
  （(path,size,mtime) 键，与 downloadable 同方案）消除重复全量哈希。
- **门禁稳健性（EXPORT-05/06/07）**：dynamic-spatial 第二探针输入种子化
  （与第一输入同源，门禁不再 flaky）；int8 parity 门禁对输出数不匹配
  显式报错 + `equal_nan=False`（NaN 参考不再过闸）；`gates=[]` 旁路
  晋升在审计日志携带 `bypass: true` 与门禁清单（"explicit, audited
  bypass" 承诺兑现）。
- **发布原子性（EXPORT-09）**：三载体 sidecar 发布顺序统一为
  "sidecar 先行、工件 rename 收尾"（TRT/OV 带旧 sidecar 备份恢复；
  OV 为 json → bin → xml，entry-point 最后）——并发读者不再看到
  新工件配旧身份 sidecar 的窗口。
- **下载加固（EXPORT-08/18/19）**：`register_downloadable` 名称拒绝
  穿越/NUL/绝对路径（HF 式命名空间斜杠保留）；OFFLINE 模式不再删除
  校验失败的缓存文件（保留现场供人工修复）；单源下载增加有界重试
  （3 次指数退避）后再换 mirror。
- **杂项（EXPORT-10 + P3）**：>2GB external-data 图在 onnxslim 后
  fail-fast（此前深陷 value_info 修复晦涩报错）；`list_entries` 跳过
  损坏条目 + warning（此前整体崩溃）；`promote` 新增 `set_current=False`
  （历史版本重晋升不再回拉生产指针，审计携带该标记）；benchmark 的
  p90 改最近秩；GTR 根路径改为惰性读取（post-import env 生效）并移除
  与 docstring 矛盾的 research 树默认候选；审计追加改尾部读取
  （O(1) 而非 O(n²)）；动态 batch 上限 128 常量化并标注。

### 变更

- **数据加载严格性（H3-2a/b/d，弃用协议）**：COCO 重复
  `image id` 加载期报错（此前静默覆盖）；未声明的 `category_id` 计数
  并汇总 warning（此前静默丢弃）；YOLO/OBB 畸形行保持跳过但输出带
  文件/行号的 deprecation warning（0.7.0 起默认报错）；目录穿越防护
  统一到检测/实例分割/关键点加载器。COCO 实例分割加载器从
  O(图×标注) 双重循环改为预分组（H3-2c）。
- **float 语义 mask 弃用（H1-11）**：`to_json(masks="rle")` 对 float
  label map 输出 deprecation warning（行为暂不变，0.7.0 拒绝）；
  `masks="dense"` 增加 `max_elements×100` 资源硬闸（H3-1a）；
  `mask_to_bool` 整数路径 `!= 0` → `> 0`（-1 unassigned 对齐 panoptic
  约定，H3-1g）。
- **训练看门狗与提示（H3-3f/g/h）**：`AUTOVISION_STEP_TIMEOUT_SEC`
  接线为告警型看门狗（步超阈边沿告警，非取消机制）；resume 比对
  数据集指纹（漂移告警不阻断）；DDP + batch_size=1 + BatchNorm 输出
  SyncBN 建议 warning。
- **混淆矩阵/质量报告（H3-1c/2j）**：显式 `class_ids` 遇未见类别输出
  一次 warning（此前静默吞计数）；quality 报告 markdown 单元格转义
  扩展到换行/反引号（异常 detail 不再破坏表格）。
- **错误路径可观测性（H2-5d/e）**：429/4xx/503/504 全部计入延迟
  直方图（`error_class` label）；优雅停机默认 30s（传 None 恢复无限
  等待）。
- **推理 poison 按变体隔离（H2-3）**：推理超时只标记被路由的 variant
  （并发信号量同样按 variant 划分）——A/B 部署中非关键 variant 卡死不再
  拖垮全进程；`/predict` 在健康变体内路由，`/readyz` 仅在**全部** variant
  中毒时 503，响应体与 `/healthz` 携带 `poisoned_variants` 明细。
- **相机索引语义澄清（H3-5g）**：`run_stream`/`VideoFrameStream`
  docstring 明确相机索引必须传 int、字符串一律按文件/URL 解析。

### 文档

- 新增「工程加固计划」（`docs/hardening_plan.md`）：2026-11 全仓库审计的
  修复设计与实施计划，含逐项设计、测试计划、验收标准、兼容性与回滚策略；
  批次 0–3 的实施项均在文档中标注状态。
- `docs/known_limits.md` 同步 serving 认证姿态、数据加载严格性、推理
  序列化弃用项。
- **命令数文档对账（H3-6h/H4-6）**：AGENTS.md/README/docs 的"30 条命令"
  更正为 31 条（README 全表补漏 `tile-dataset`）；新增
  `tests/test_infrastructure.py` 对账用例——AGENTS.md 声明的命令数与
  `cli/_app.py` 实际注册数不一致即测试失败，防文档静默漂移。
- **导出侧 golden 数值基线（H4-5）**：新增 `tests/test_export_golden.py`
  与 8 个任务面的 ORT 输出快照（`tests/golden/export_<face>.json`，
  atol 1e-5）——结构化 parity 门禁只比对同权重的 torch-vs-ORT，无法
  捕获跨版本数值漂移；快照钉住 decode/后处理、opset、fp16 黑名单变化。
  录制即失败一次（杜绝 golden 误删后 CI 首轮放行）。

### 测试与工程

- **恢复 8 个被嵌套吞掉的流式管线测试（H0-1）**：`tests/test_stream_pipeline.py`
  的缩进事故使 zone 防护、NaN→null 等核心用例从未执行。
- **机制化（批次 4）**：`scripts/check_test_layout.py`（测试布局守卫，
  防嵌套事故复发）与 `scripts/check_design_anchors.py`（设计编号→测试
  锚定检查，allowlist 带过期机制）进入 CI lint job；`tests/test_soak.py`
  长时运行老化套件（`AUTOVISION_RUN_SOAK=1` 门控）；byte_track golden
  缺失时从"录制并跳过"改为"录制并失败一次"（H4-5）。
- 新增 `scripts/compute_dataset_pins.py`（H1-9 配套：从已下载工件计算
  sha256/size 并输出 registry 片段；pin 值需有网环境填充）。
- **批次 3 补齐约 90 个回归测试**：契约层 panoptic 启发式/异常类型/
  可视化一致性、数据层损坏标注参数化用例/mmap 4 进程并发/快照原子性/
  嵌套目录掩码匹配、引擎 strided sampler 划分数学/DDP 指标与单进程
  一致性/空 shard 守卫/SyncBN 提示/step 看门狗、导出 Dim 缓存/多输入
  拒绝/staged 清理/精度容差 6 场景/工具链元数据、registry 下载并发/
  摘要缓存/晋升审计/许可证别名一致性、运行时丢帧计数/重连重置/
  预算恢复/分段轮换/熔断、CLI 信封映射/HPO 透错/auto_raise 跨轮。

## [Unreleased]

（无。下一个开发周期从本段开始记录。）

### 修复

- **AMP 预检探针（D16）此前从未真正运行**：引擎把 `TrainModel` 包装器传给
  探针，而 `model_input_spec` 只在内部 TaskNetwork 上（包装器未转发）——
  所有任务面的探针都以 "inconclusive" 收场且从不给出真判定。修复两层：
  `TrainModel.model_input_spec` 转发内部网络的声明；探针经
  `get_task_model()` 解包到真正的网络再前向（包装器的 `forward` 是
  training_step 契约，喂张量会崩）。探针 autocast dtype 现随请求精度
  （bf16-mixed 探 bf16，16-mixed 探 fp16），判定对齐实际训练精度。
- **探针可声明弃权（`amp_probe = "skip"`）**：GTR（GLA 双向扫描）栈的
  跨路径 logit 方差在随机初始化下已达 ~1.0（CPU 参考与 CUDA Triton 的
  纯 fp32 对比实测 0.57/0.96，与 autocast "发散"同量级），绝对 atol 探针
  在该家族上结构性无判别力——连它建议回退的 fp32 路径都"不过关"。DETR
  （含 gtrvit-* 变体）任务网络声明 `amp_probe = "skip"`，探针返回
  inconclusive 并保持请求精度，不再据此强制 2 倍慢的 fp32。

### 变更

- **目标检测训练默认开启中间验证**：`trainer_args.val_every_num_steps`
  未设置时解析为 `min(500, steps)`（显式 0 仍为关闭）。此前裸跑默认
  12000 步全程零验证、"best" checkpoint 静默等于 "last"；内置 recipe
  （finetune/coco-finetune/low-data）显式设置不受影响。
- **非 deterministic CUDA 训练开启 cudnn autotune**：`fit_task` 在
  CUDA 上且未设 `AUTOVISION_DETERMINISTIC=1` 时启用
  `cudnn.benchmark`（新增 `engine.enable_cuda_math_modes`，与既有 TF32
  开关同域）。检测器的固定/少档输入形状下 conv 族受 kernel 发射瓶颈
  限制，autotune 按 shape 缓存挑选更快的 conv 算法；
  `AUTOVISION_DETERMINISTIC=1` 保持关闭。

### 性能

- **DETR 家族训练吞吐开关三件套（gtrvit-s @640px batch16 RTX 4090，真机
  ~1.25 s/step → ~0.54 s/step）**：既有 `criterion_flat_losses`（W2 扁平
  匹配索引路径）与 `fused_optimizer`（W1）此前默认关闭；新增 opt-in
  `compile_regions`——matcher 固定 eager（Triton LSAP + 数据依赖索引会
  碎裂 dynamo），criterion 经 `try_compile` 走 inductor（AOTAutograd
  连带融合其 backward；criterion/matcher 无参数且不进导出，checkpoint
  键与导出权重不变）。
- **真机 A/B 定论（2026-11-05，512 张活数据 120 步）**：纯 eager
  `criterion_flat_losses + fused_optimizer` 稳态 0.40-0.45 s/step；开启
  `compile_regions` 触发重编译风暴（step-32 窗口 17.8 s/step），真实数据
  上被 eager 双 flag 完全压制——`compile_regions` 保持 opt-in 并在启用时
  打印真实数据警告，冠军真机配置就是两个 eager flag（~2.9x vs 基线
  1.25 s/step）。静态填充天花板探针（targets 统一填充到 8 框 + 静态
  编译，内容逐批变化）实测 240 ms/step，比 dynamic 275 好 ~12%——
  「入口填充 targets + inf 代价 pad 行 matcher + dn/主损失全掩码」的
  生产改造预估再 +10-15%，之后的墙是 eager backbone（fla/HF）与
  引擎侧固定开销；0.1 s/step 需要整步 CUDA-graph 级工程。
- **compile_regions 以 `dynamic=True` 编译（真实数据必要性，实测）**：
  真实数据每图框数逐批变化 → 匹配对形状逐批变化 → 静态编译图按签名
  反复重编译，触顶 dynamo cache 后整帧回退 eager——实测 309 ms/step，
  比 289 的纯 eager 还差；`dynamic=True` 实测 275 ms/step（约 +5%），
  但 eager matcher 边界传回的匹配对是具体张量，内层
  flatten_match_indices 帧仍按具体尺寸特化并触顶 recompile_limit——
  dynamic 标记无法穿越该边界，故真机收益约持平偏正。确定性形状管线（目标填充到固定容量）下静态编译
  才有大头：criterion 195 / +decoder 164 / +encoder 151 / GPU 下限
  89 ms/step——真机兑现路线是「目标填充到数据集级最大对容量」（涉及
  matcher/loss 契约，未在本轮实施）。
- **真机 vs 台架差距归因（各假设逐一实测排除）**：scale-jitter 形状
  方差（固定形状真机仍 ~0.50s，排除）；worker 每 epoch 重启（512 张
  数据集重启频率减半仍 ~0.51s，排除）；增广吞吐（单样本实测 5.7ms，
  8 worker 下每批 ~12ms，排除）；剩余主因为**逐批目标形状方差**（见上）
  加上 harness 未覆盖的项（num_classes 2 vs 真实 80 的 criterion 宽度、
  denoising 组形状随批变化、EMA/引擎侧每步开销）。整网 torch.compile
  因 fla GLA / HF 外壳 / denoising nonzero 等 graph break 碎裂而无收益
  （whole/per_module/reduce-overhead 实测 1111/1149/2706 vs eager
  1169，已记录在 torch_compile.py docstring）。

## [0.6.0] - 2026-11-05

（发布于 PyPI；发布提交 138c2a3，未走 tag/CI 流程——见 known_limits 流程注记。）

### 沿革

本节正文与 0.6.1 前的 [Unreleased] 合并记录在上方 0.6.1 节（同一工作区两轮加固连续实施）。

## [0.5.0] - 2026-11-04

### 新增

- **任务示例的 ONNX 导出与推理闭环**：全部 8 个训练示例新增
  `--export-onnx PATH`（走 `autovision-cli export` 同一 parity 门禁管线，
  分类/全景在 CPU 即可，GTR 族经导出安全 GLA 降级无需 CUDA），训练结束
  自动导出并在摘要打印工件路径；全部 8 个推理示例新增 `--onnx PATH`——
  前向换 ONNX session（任意设备可用）、前/后处理留在 torch 网络，
  工件元数据（几何/类别数）与构造网络不一致时一行干净退出；`--onnx`
  同时跳过发布 checkpoint 下载与 eager-only CUDA 要求（网络仅作
  前/后处理，随机初始化即可）。examples README 记录完整
  训练→导出→ONNX 推理走查；工件与网络几何不匹配属 fail-closed 设计
  （导出 demo 是 128px、发布网络是 640px 时会明确报错并给出指引）。
  实测八族全部闭环（detection/OBB/instance/pose/depth/semantic GTR +
  panoptic EOMT + classification），eager vs ONNX 对拍一致（检测框
  IoU>0.5 全匹配、分数差 ~1e-5；近平局候选的 top-k 行序互换与
  `docs/known_limits.md` 记录的 decode 行为一致）。

- **训练自愈：NaN/Inf 回滚恢复（第三轮 D32）**：预训练 `Engine.fit` 与
  `TaskEngine.fit_task` 遇非有限 loss（或非 AMP 路径的非有限梯度范数）不再
  直接失败——回滚到 `last.ckpt`（model/optimizer/scheduler/scaler/RNG，
  task 面另含 EMA/best_metric/早停计数）并从中断处重跑；DDP 下 step 级
  4 字节判定 all_reduce / epoch 末判定屏障使全体 rank 一致进入恢复；
  连续 3 次失败硬失败并附 last.ckpt 路径与次数，预算在干净窗口后复位；
  回滚加载前逐张量校验 checkpoint 有限性（中毒文件拒载）。恢复默认开启；
  数据性 NaN 在恢复的 RNG 下必然复现并按预算失败，瞬时毛刺则恢复成功。
- **EMA 中毒修复（D33）**：`ModelEMA.repair_from` 在写 checkpoint 前把
  非有限 EMA 张量从 live 权重重同步（EMA 是滑动平均，单点 NaN 永久污染，
  此前会原样进入 fp32 checkpoint）；经 `get_task_model()` 解析对齐键空间；
  FSDP 下跳过（live 分片不可逐张量对照，记录偏差）。
- **条件性显存清理与 checkpoint 写入重试（D34）**：验证边界改为仅在
  缓存分配器 reserved 占比超阈值（0.5）时 `empty_cache`（MPS 恒清、CPU
  不清、占用查询失败退化为恒清——饱和卡行为与此前逐位一致，小模型不再
  支付全量 cudaFree 往返）；checkpoint 原子写新增 3 次指数退避重试
  （杀毒扫描/设备 flush 类瞬时 OSError），每次重试全新临时文件，耗尽后
  `CheckpointError` 且旧文件完好。
- **活流捕获契约（D35）**：`VideoFrameStream` 新增 `reconnect_attempts`/
  `reconnect_delay_s`（默认 0 = 既有"断流即结束"语义不变）：非文件源读
  失败时重开 capture，帧序号延续，重开失败计入次数；本地文件短读仍视为
  截断报错。`info.fps` 消毒：原始值不在 (0, 1000) 开区间（0/NaN/90kHz
  RTP 时钟）记 0（下游既有默认分支覆盖）。`run_stream` 同名透传
  （CLI `key=value` 壳自动可达）。
 - **NSA-Kalman 置信度缩放（D36，opt-in）**：`BYTETracker`/
  `BOTSORT` 新增 `confidence_scaling=False` 参数——启用后 Kalman 观测
  噪声按 `max(1 - score, 0.05)` 缩放（StrongSORT NSA），高置信检测更强
  地牵引状态；默认关闭，ByteTrack 官方固定噪声数值逐位不变（golden
  生命周期快照锚定）。
- **断流重连可观测（D35 加固）**：`VideoFrameStream` 新增
  `reconnects_used`（最近一次迭代消费的重连次数），`run_stream` 透传到
  `StreamRunResult.reconnects` 并在非零时进入 CLI 摘要行——中途断过流
  （或重连耗尽后提前结束）的运行不再与「干净 EOF」在结果对象上不可
  区分，7×24 监控可据此告警。
- **周期存档与 checkpoint 序列化卫生（设计提案 D20）**：预训练
  `Engine.fit` 新增 `save_period_num_epochs`——命中周期时单次序列化、
  原子写 last + `epoch{N}.ckpt` 双份并滚动保留最近 3 份；所有 checkpoint
  写盘前做 fp16 溢出钳制（克隆上，不动 live 权重）与 4-d 权重
  contiguous 归一。
- **首轮 OOM 自动降批（D21，单进程）**：训练循环捕获分配器类 OOM
  （类型 + 四类驱动错误串），per-device batch 减半并重建 dataloader
  （sampler 保续、optimizer/scheduler 保留），OOMed step 跳过、至多 3 次；
  batch=1 或 DDP/FSDP 下不降级（DDP 的非对称 OOM 无法安全同步）直接
  报错并给出人工建议。
- **引擎级回调注册表（D22）**：`engine/callbacks.py::CallbackRegistry`
  （6 个命名事件 + add/set/clear/adapt），旧 `callbacks=` 鸭子列表继续
  接受；逐 handler 异常隔离（回调失败告警不中断）；新事件
  `on_checkpoint_save`；预训练 `Engine.fit` 新增 `callbacks` 参数。
- **训练基建杂项（D23）**：dataloader 级 worker 播种
  （`build_task_dataloader`/`get_dataloader` 新增 `seed=`，修复 numpy/
  random 在 fork worker 中同源的静默复现陷阱；object_detection 与预训练
  CLI 已接线）；`runtime/model_info.py`（参数分层 + thop 惰性 GFLOPs，
  fit 横幅新增 GFLOPs 行）；`runtime/fuse.py`（通用 Sequential
  Conv+BN 折叠 + `assert_fused_parity` 数值门禁，QAT 出范围）。
- **增广四目标协议（D24，架构前置）**：`augment/core.py::GeoTransform`——
  一次采样、四目标钩子（image/instances/semantic_mask/depth）共享同一
  冻结参数；值约定入协议（语义最近邻+255、深度最近邻+0、关键点出界
  visibility 置零）；mosaic/mixup 移植时的实现层，方法制公开面不变。
- **指标聚合容器（D27，容器部分）**：`metrics/containers.py::TaskMetrics`
  ——逐 batch 只 append 原始行、`process()` 期末归并、
  `export_state/merge_state` 保持既有 DDP 鸭子协议；
  `empty_empty_score` 固化"无 GT + 无预测 = 满分"约定；现有
  Collector/Evaluator 行为不变。
- **数据与评估小件（D29a/b）**：`data_check` 新增存储体检（抽样 ≤5 文件
  测 stat ping 与读吞吐，10ms/50MB/s 阈值告警建议本地存储，只告警不判罚）；
  `COCOMAPEvaluator` 新增 `auto_raise`（跟踪最密图实例数、compute 期自动
  抬升 maxDets 并告警，零额外 IO），`evaluate_object_detection` 的
  `max_det=None` 默认启用。
- **handle 级编译后端引擎缓存与参数二分（设计提案 D15）**：
  `detection(...).predict(image, backend="onnx", artifact=...)` 现已可用；
  引擎按构建期键（backend/artifact/device/providers/openvino_device）缓存
  复用（LRU-2 槽，交替双配置不重建），阈值类运行期参数每次调用传入；
  `train()` 后缓存失效，handle 反序列化时丢弃进程级缓存。
- **AMP 事前数值探针（D16）**：混合精度训练启动前用调用方自有模型跑
  FP32 vs autocast 对照（不下载任何外部权重），输出漂移即 fail-closed
  降级 fp32；探测不可判（模型无探针输入）时保持请求精度；DDP 下 rank0
  决策 int8 广播。`trainer_args.amp_check=false` 可关（object_detection
  面已接线，其余任务面随后跟进）；`Engine.fit` / `TaskEngine.fit_task`
  新增 `amp_check` 参数。
- **resume 加固与早停状态持久化（D17）**：task checkpoint 现在携带
  早停耐心计数（resume 续接而非清零）与 fit-args 快照；resume 时白名单外
  的超参漂移逐键告警（现值治理），调度类键（steps/patience 等）仅 info；
  接近早停阈值时提前一次验证（possible_stop），早停决策不再被动等下一
  个调度点。
- **空闲 GPU 自动选卡（D18）**：device 语法新增 `-1` 占位符——
  `resolve_device("-1")` 选最空闲卡，`runtime.autodevice.resolve_device_spec`
  展开任意 spec（`"0,-1"` → `"cuda:0,cuda:3"`）；排序键
  （利用率, -空闲显存, 随机 tiebreak），`CUDA_VISIBLE_DEVICES` 物理 id
  双向翻译；逗号 spec 在 `resolve_num_devices` 计入进程数。降级两段式：
  空闲阈值筛光时回退最不忙的可见卡（告警）；可见卡数少于占位符数时
  **报错**（配置不可满足）——不再静默产出 `cpu` 混搭 rank；NVML 缺失
  降级顺序取卡并告警。
- **COCO 导入健壮性（D19）**：实例分割数据集对畸形 segmentation 多边形
  白名单校验（长度/偶数/有限数值），不合格退 bbox 矩形（每文件聚合一条
  告警），无 bbox 可退时丢弃实例并告警；修复 compressed RLE（string
  counts，真实 COCO crowd 标注形态）被误送 `frPyObjects` 而崩溃的既有
  缺陷；`geometry/polygons.py` 新增 `is_valid_polygon` / `polygon_from_bbox`。
- **导出工件自描述与加载端一致性校验（设计提案 D1）**：ONNX 图内
  `metadata_props` 与 OpenVINO / TensorRT 旁路 JSON 现在携带完整身份
  （task、类别数、`image_size`、类别名、输入布局、schema 版本）；
  `InferenceEngine` 构造时校验 network↔artifact 一致，跨任务面误配在加载
  期报出两侧实际值而非首帧形状错误；旧的无元数据工件照常加载。TensorRT
  引擎新增 `<stem>.meta.json` 身份旁路，导出完成日志回显身份摘要。
- **导出产物最小尺寸护栏**：`export_onnx` 在 parity 门禁后断言产物
  > 4 KiB，"只写出了头部的成功导出"直接失败（D14b）。
- **追踪器生命周期 API（D2）**：`Track.det_index` 回写检测索引（跨验证
  丢框保持原位）、`BYTETracker.drain_removed_tracks()` 发布退役轨迹一次、
  `reset()` 换源重置（id 从 1 重新计数）；Kalman 状态改为实例私有。
- **追踪器钩子化与 BoT-SORT（D7）**：`update()` 重构为固定模板 +
  可覆写阶段钩子（golden 生命周期快照逐位锚定），`BOTSORT` 以 XYWH 运动模型
  子类落地（默认开启相机运动补偿）；ReID 外观匹配按设计提案明确出范围。
- **相机运动补偿 GMC（D6）**：`tracking/gmc.py` 提供
  sparse_opt_flow / ORB / SIFT / ECC 五种估计（`cv2` 惰性导入，属 `video`
  extra），检测框前景剔除、失败回退不补偿并告警一次；
  `run_stream(..., motion_compensation=...)` 接线，平移量按降采样比例
  还原到全分辨率坐标。
- **`LineZone` chord 穿越补偿（D13a）**：在既有无限线符号翻转之外增加
  前后锚点连线与线段的真交点判定（端点触碰不算），补计"丢失期间跨线"
  与"锚点正落在线上"两类漏计；每 track 每帧至多计一次。
- **`box_candidates` 增广劣化门（D3）**：mosaic 整画布仿射后按
  最小宽高 / 长宽比上限 / 变换前后面积比过滤劣化目标并同步标签
  （阈值进 `RTDETRDetectionTransformArgs`，默认宽松：2 px / 100 / 0.1）；
  `geometry/polygons.py` 提供纯 numpy 多边形画布裁剪（Sutherland–Hodgman）。
- **autobatch 训练批量探测（D4）**：`engine/autobatch.py` 以真实训练 step
  闭包逐候选剖析峰值显存、线性拟合外推到可用显存的目标比例、实测 OOM 回退
  上一安全档；纯拟合策略与 mock CUDA 探测循环均可离线测试（CLI
  `batch_size="auto"` 按任务面逐个接线，见模块 docstring）。
- **基准计时方法论与指标下限门禁（D5）**：`export/timing.py` 提供 2σ 迭代
  剔离群与按目标测量窗自适应步数；`benchmark_inference(..., robust=True)`
  在 `extra` 输出稳健统计（默认关闭，基准表可比性不变）；registry 新增
  opt-in `MetricFloorGate`，模型卡指标低于声明下限即 promotion 失败。
- **数据面四件（D8–D11）**：数据集扫描缓存（`.autovision_scan_cache.json`
  + `AUTOVISION_SCAN_CACHE=0` 旁路 + `data_check no_cache=true`）；图像
  读取缓存（`.npy` 边车 staleness 协议 / 单张量 RAM 共享池，
  `ImageDataset(image_cache=...)` 选择性启用）；数据集下载断点续传
  （Range 续传 + 5xx 指数退避 + 4xx fail-fast + 磁盘预检，sha256 校验与
  原子发布不变）；训练侧大图切片 `tile-dataset`（IoF≥0.7 窗口-目标分配、
  窗口保留率剪枝、`tiles.jsonl` 溯源清单，推理侧 SAHI 的 complement）。
- **serving lifespan 全管线预热（D14a）**：服务启动时对每个引擎跑一次
  完整 `run_batch`（解码 + forward + 后处理），首个请求不再承担惰性导入与
  会话预热；预热失败仅告警，readiness 探针仍为流量准入依据。

### 修复

- **CI 首曝修复（v0.5.0 tag 门禁）**：dev 线近两个大提交新增的测试此前从未
  在 CI 上运行过（私有仓库只对发布快照触发 CI），首次曝光暴露四处问题。
  - **`Engine.fit` 成功路径销毁进程组后再调 `get_rank()`**（真实回归）：
    finally 的 destroy 对成功路径同样生效，随后 `on_train_end` 的
    rank 判定在 torchrun 多进程下直接
    `ValueError: Default process group has not been initialized`——rank
    门冻结在进程组存活期（`is_rank_zero`），与 `fit_task` 既有做法对齐；
    `AUTOVISION_DDP_TESTS=1` 门控使其在本机全量中被跳过，仅 CI 曝光。
  - **AMP 数值探测在无 CUDA 主机上虚假通过**（实现缺陷）：torch 会静默
    禁用 `autocast("cuda")`，探测退化为 fp32-vs-fp32 的空转平局并误报
    "supported"——cuda 类设备在无 CUDA 主机上改为 fail-closed 返回
    False，并发探测用例补 `is_available` patch 使比较逻辑在 CPU-only
    CI 上真实可达。
  - **`test_autodevice` 五测漏了 GPU 可见性 patch**：只注入了 NVML
    统计，未注入 `torch.cuda.device_count`，GPU-less CI 上可见性门
    先行拒绝；按注入拓扑补齐（文件自身契约即"不依赖硬件"）。
  - **`test_detection_rejects_cpu_for_gtr_eager` 假定 fla 已装**：无
    `gtr` extra 的机器上依赖检查先于 CUDA 检查触发，断言按 fla 可用性
    分支。
- **评审加固（D32–D36 收尾）**：自审查发现的六处边界收口。
  - **回滚读侧校验扩展到 engine_state**：`_assert_state_dict_finite` 改为
    递归遍历整个载荷——NaN 藏在 optimizer moments / EMA sidecar /
    `best_metric` 标量里（权重本身干净）时，恢复同样拒载，不再静默装回
    毒状态；RNG 字节张量与整型状态不受影响。
  - **NaN 恢复预算改为干净 step 复位**：`TaskEngine` 此前只在干净验证后
    复位预算，无验证集的运行三次预算全程共享；现按干净 step 复位（数据性
    NaN 在回放中必然连续复现、预算照常耗尽，与「预算在干净窗口后复位」
    的既有表述一致），预训练面的每干净 epoch 复位不变。
  - **首个验证评估不再接受非有限基线**：`best_metric is None` 分支此前
    无条件 `improved=True`，NaN 指标会成为基线并冻结整个 best 跟踪、
    写出 NaN `best.ckpt`；现要求 `isfinite(watch_value)`。
  - **验证 loss 非有限逐次告警**：验证侧 NaN 是测量故障（训练窗口已逐步
    校验），不触发回滚（重放只会复现它并丢弃干净训练进度），按既有早停
    节奏计入 no-improve；此前完全静默。
  - **恢复日志标注 NaN 来源**：DDP 下远端 rank 中毒时，本地 rank 的恢复
    日志此前仍打 "Non-finite loss at step N"——现区分「detected on this
    rank / reported by a peer rank」，干净 rank 排障不再被引向不存在的
    本地 NaN。
  - **示例摘要防御 + EMA 快路径 + mypy 清零**：`summarize_prediction`
    对 labels/scores 为 None 的记录退化为 outputs 摘要而非 `list(None)`
    TypeError；`ModelEMA.repair_from` 用单次堆叠 amax 归约替代每张量
    device 同步（健康 EMA 的每次 checkpoint 写入只付一次同步）；
    `byte_track.py` 存量 13 个 mypy 错误清零（`_std` 参数收窄、
    `_STrack.update` 调用点显式 float/int、`Track.bbox` 经 `_xyxy_tuple`
    精确四元组、`classes_array` 注解），`VideoFrameStream.__iter__` 注解
    修正为真实的 `Generator` 类型（stream.py 消费 `.close()` 的存量
    mypy 错误一并消除）。
- **GTR 实例分割导出无法重载（`--predict-image` 崩溃）**：GTR seg 是
  `DETRObjectDetection` 子类，导出载荷携带的是 DETR 父层展平的
  init-args——重载时任务级 model key 盖写的 `model_name` 与本类硬编码
  转发的 `model_name`/`decoder` 重复传参直接 `TypeError`，且载荷缺
  `variant`（重载会按默认 "s" 重建几何导致 state dict 形状不匹配）。
  子类现在剥离 stamp 键并在 init-args 记录 `variant`；加载端
  （`load_task_model`）同时确立契约：重载从不取外部权重——载荷缺省时
  对暴露 `load_checkpoint` 形参的类强制 `False`（此前 GTR 五族导出的
  重载都会再下载官方权重再被训练权重覆盖，离线直接失败）。新增
  `tests/test_gtr_seg.py` 导出回环回归测试。
- **任务示例脚本（examples/tasks）的错误路径改为可操作退出**：GTR 推理
  示例在显式 `--device cpu` 下不再落入 fla 的 Triton 深栈崩溃
  （`Pointer argument cannot be accessed from Triton`）——`gtr_requirement`
  新增 `device` 参数校验解析后的设备选择（而非仅机器能力），一行给出
  CUDA/`--onnx` 替代方案；`--weights` 加载的 checkpoint 若属 GTR 家族
  （导出元数据无家族标记，按加载后网络的模块树内省识别）同样在首个
  `predict` 前被干净拒绝，六个推理示例统一接入
  （`gtr_eager_device_guard`）；`*_train.py` 重复运行且 `--out` 已存在时
  捕获 `get_out_dir` 的 `ValueError`，打印指向 `--overwrite` 的一行提示
  而非裸 traceback（新增 `ensure_out_available` 助手，8 个训练示例统一接入）。
  `detection.py` 补齐兄弟示例都有的 `--class-names`（导出 checkpoint 不携带
  类名，此前只能渲染裸类别 id）；`detection_train.py` 的 `lr_warmup_steps`
  钳制到 `--steps`（此前 `--steps 1` 冒烟跑 warmup 长于全程、唯一一步只到
  峰值 LR 的 25%）；示例输出改为任务感知摘要（`summarize_prediction`）——
  分类打 `top1 = <类> (分数)`、全景打 `segments = N`、语义/深度描述稠密图
  尺寸（此前一律按检测措辞 `detections = 1` / `n/a (image-level output)`），
  训练 `--predict-image` 保留其加长版零检出提示；忽略 `--num-classes` 的
  5 个训练示例（detection/OBB/pose/depth/panoptic）改为在误传该参数时
  一行退出说明适用范围，而非静默无效。
- **engine.py 类型注解修复（mypy 归零）**：`np.random.get_state()` 返回值
  按 numpy 桩类型与运行时元组的差异显式标注 `Any`（状态字节不变）；
  LR schedule 的两个分支与验证触发改用 `is not None` 显式收窄（运行时
  语义不变），`TaskEngine.fit_task` 的 best-metric 首评/提升分支同步消除
  Optional 算术报错。此前该模块累计 9 个 mypy 错误，现全部消除。
- **fit 日志语义**：best-metric 行区分首次评估与后续提升——首评打
  `baseline set to`（此前 2 步冒烟跑会打出误导性的
  `improved to 0.0000`），其后仍为 `improved to`；LR schedule 行只列
  非空阶段（冒烟跑此前打出退化的
  `warmup 2 → flat until 2 → cosine until 2 → tail to 2`），tail 相位
  正确标注保持 `min_factor × peak`（旧文案的 "at peak LR" 与实现不符）。
- **存档期 `last.ckpt` 元数据 epoch 改为 0-based（D20 修正，随 D32 生效）**：
  save_period 存档路径此前把 `last.ckpt` 的元数据 epoch 写成 1-based（与
  `epoch{N}.ckpt` 文件名对齐），而 resume 约定是"元数据 +1"——存档期之后
  resume（以及新的 D32 回滚）会静默跳过一个 epoch。现统一为元数据 0-based，
  仅存档文件名保持 1-based；含跨存档边界回归测试。
- **GTR 路线的 OpenVINO 导出不再在 parity 门禁崩溃**：OpenVINO 参考被强制
  deepcopy 到 CPU，而 eager GLA 是 fla 的 CUDA-only Triton 核——五条 GTR
  路线（det/obb/pose/depth/semseg）在 `verify=True`（默认）下 IR 转换成功后
  必然以 `ValueError: Pointer argument cannot be accessed from Triton` 失败。
  参考现在经 `prepare_onnx_export` 钩子回放 trace 期同一导出安全路径（即导出
  图自身的语义），CPU 上即可完成对拍；新增 OpenVINO 端到端门禁测试。
- **GTR 路线的 TensorRT 引擎校验在模型驻留 CPU 时不再崩溃**：engine 校验的
  eager 参考跑在模型所在设备上，而导出 CLI 默认 `device=cpu`——校验触发同一
  Triton 错误并删除已构建的引擎。现在 CPU 驻留且存在 `cpu_eager_blocker`
  模块时回退导出安全参考路径并告警；GPU 驻留模型保持 eager-vs-engine 强校验。
- **GTR ONNX parity 门禁的参考侧显式挂钩**：此前 CPU 参考之所以走导出安全
  路径，靠的是「deepcopy 恰好发生在 trace 窗口内、继承 `_onnx_trace=True`」
  这一副作用——窗口外的任何 re-verify（如后置校验）都会落到 eager 核上崩溃。
  参考现在显式挂钩（行为不变，不再依赖继承时序），新增窗口外 re-verify 回归
  测试。
- **GTR SpatialSwiGLU 骨干的 deploy 静态 bid_scan 置换生效**：两层原因使该
  优化对所有路线都是死代码——`GTRViTBackbone` 从未向骨干传递
  `eval_spatial_size`（`convert_to_deploy` 直接 return），且
  `VisionTransformerSpatialSwiGLU._apply_bid_scan` 覆写不检查
  `_deploy_static_perm`。现在五个构建点传入 `image_size`，覆写遵守标记（静态
  `index_select` 与 flip 链逐位一致，含 CLS-free 置换 buffer 构建），deploy /
  导出图里的 Flip+Reshape+Transpose 链与常量 `argsort` 簿记被单个 Gather 替代；
  顺带移除从未被读取的 per-layer 置换 buffer 注册。
- **修正 `docs/export.md` 动态 H/W 支持面表**：GTR 全线此前被误标为
  「支持」，实际 `dynamic_spatial=True` 被 `dynamic_spatial_blocker` 可读拒绝。
- **GTR 五条任务路线（depth / instance seg / obb / pose / semseg）默认构造
  不再静默返回随机权重**：`load_checkpoint=True`（默认）+ `checkpoint=None`
  （默认）时旧的 `if checkpoint and load_checkpoint:` 守卫什么都不加载、
  不告警——随机权重模型可以跑出看似合理的预测。现在默认解析该变体的
  sha256 固定官方资产（`resolve_gtr_asset`，缺文件/sha 不符时报错并给出
  路径、来源与下载命令）；显式 `checkpoint` 路径仍优先，`load_checkpoint=False`
  保持随机初始化契约（导出门禁测试依赖）。新增默认加载与显式退出的回归测试。
- **GTR 全线真实尺寸（多 chunk）下的 fp16 ONNX 导出修复**：GLA causal 掩码
  trace 出的 `ConstantOfShape` 保持 fp32，而 ORT float16 转换器把周围算子
  改成 fp16——token 数超过一个 GLA chunk（>64 token，即输入大于 128²，
  640² 实际导出必中）后产生混合 dtype 的 `Mul`，`onnx.checker` 拒绝导出。
  `ConstantOfShape` 现在随 MatMul/Resize 一起钉在 fp32；既有 fp16 测试的
  64² 输入恰好单 chunk 所以从未触发。新增 160²（最小失败尺寸）多 chunk
  回归测试。
- **TensorRT 引擎构建前的版本预警**：GTR 路线在检测到 TensorRT ≥ 11 时，
  `export_tensorrt` 现在会在构建前主动告警（上述 Myelin miscompile 已知
  问题）——`verify=False` 的构建不再静默；`verify=True` 的对拍门禁行为
  不变（仍然 fail-closed 拒绝并删除数值错误的引擎）。
- **GTR 预训练句柄 CPU-only 机器可构造**：`build_pretrained_detection
  ("detr/gtrvit-*")` 不再在无 CUDA 时硬拒绝（构造不执行 GLA 核，只有
  eager 前向需要 CUDA），改为告警并正常装配官方权重——CPU-only 机器经
  `InferenceEngine(model.network, backend='onnx', artifact=...)` 走文档
  写明的 ONNX CPU 部署路径（640² det S 实测：ONNX CPU 13/13 检测与
  eager CUDA 一致，score Δ≤0.004，~0.3 s/张）；CPU eager 前向仍被
  `cpu_eager_blocker` 可读拒绝。`DetectionModel` 新增公开只读属性
  `.network`；`examples/tasks/detection.py` 新增 `--onnx <artifact>`
  直跑导出工件（`gtr_requirement` 相应增加 `need_cuda` 开关）。

### 文档

- **移除设计提案 `docs/design_ultralytics_adoption.md`**（D1–D36 的内部<!-- docpath-ok -->
  论证文档，从未随版本发布）：退出 MkDocs nav 并删除文件，全文保留于
  git 历史（提交 63fe135）。D 编号继续作为设计词汇在 CHANGELOG、
  `docs/framework_design.md` 与溯源台账中使用；`docs/provenance/LEDGER.md`
  的净室声明改为指向 git 历史，四处源码注释的设计引用统一收敛到
  「裸 D 编号 + 台账指针」惯例。

### 已知边界（文档）

- **fla 版本横幅与导出家族元数据**：`docs/known_limits.md` 新增两条说明——
  fla 在 Python < 3.11 / Triton < 3.3 导入时打印的升级建议 WARNING 来自
  `fla.utils.check_environments()`（第三方包的环境建议，Triton 提示为真实
  风险信号，勿静默）；导出的 `.pt` 不携带模型家族元数据，示例层对
  `--weights` 的 GTR-on-CPU 预检基于加载后网络的模块树内省，属示例护栏
  而非格式契约。
- **TensorRT 版本边界**：GTR 展开纯图的 fp32 引擎在 TensorRT 11.4
  （11.4.0.106，RTX 4090D 实测，det 与 depth 双路线确认）构建成功但引擎
  数值错误（输出系统性偏离 torch/onnxruntime 参考）——疑似 Myelin 对
  万节点级 CumSum/Exp/MatMul 展开链的 miscompile；上游受支持环境为
  TensorRT 10.14。`verify=True` 引擎对拍门禁会拒绝并删除此类引擎
  （fail-closed，不会静默产出坏引擎）；该 TRT 版本上勿对 GTR 关闭
  verify。见 `docs/export.md` / `docs/known_limits.md`。

## [0.4.0] - 2026-11-02

### 修复

- **ONNX 导出不再遗留孤儿外部数据文件**：dynamo 导出器把全部 initializer 写到
  `<name>.onnx.data`，onnxslim 内联回图后成功路径仍会把这个 sidecar 移到最终
  产物旁（实测每个 GTR fp32/fp16 产物旁遗留 ~45MB 死文件，且 fp32/fp16 的
  `.data` 逐字节相同）。导出收尾现在检查最终图：仍有外部 initializer 引用
  （如超过 ~2GB protobuf 上限的图）则保留 sidecar，否则删除。
- **FP16 导出在无 CUDA execution provider 的 onnxruntime 构建上不再零验证**：
  此前 `verify=True` 的 fp16 导出在该场景只打告警就跳过（`_verify_onnx` 直接
  return），数值完全不设防。现退化为「执行 + 有限性」smoke 门禁——在 CPU EP
  上以 parity 同款种子随机输入实际执行图，任一输出含非有限值即判导出失败；
  精确数值 parity 仍需 CUDA EP（ORT CPU 跑 fp16 图会内部转 fp32，逐元素
  parity 会误报）。
- **GTR parity 门禁按精度与 decode 形态分治**，消除行求和归约的误差抵消盲区：
  新增 `export/parity_checks.py` 共享帮助模块（贪心行匹配 + 匹配后逐元素
  比较）。pose（图内无 top-k，固定顺序输出全部 query）与 depth / semseg
  （稠密图）在 fp32 下从「`sum(dim=…)` 后比较」升级为逐元素比较——历史上
  ±0.5 成对漂移可在求和中完全抵消并绕过门禁，现在必然失败；合法的行重排
  （近简并分数下的 top-k 翻转）经行匹配后仍放行。fp16 下的 pose/depth/
  semseg 与任意精度下的 det / obb 保留行求和不变式：fp16 图在简并权重上
  存在实测 ~0.8 的 decode 相邻真实漂移、det/obb 图内 top-k 在任意精度都会
  整行换入不同候选（观察过 ~0.5 的 box-sum 差异），逐元素对齐不存在。
  `verify_onnx_export_outputs` 钩子新增 keyword-only `precision` 参数
  （默认 fp32，向后兼容），ONNX 与 OpenVINO 门禁均传入。

### 新增

- **GTR 全线支持 ONNX/TensorRT/OpenVINO 导出**：新增 GLA 纯 ONNX 降级
  （`tasks/detection/shared/gtr/gla_onnx.py`，fla 同款显式逐通道衰减的
  chunkwise 递推，标准 opset-17 算子，fp32 状态防 fp16 饱和）；导出管线
  新增 `prepare_onnx_export` 模块树钩子，trace 期把 eager GLA（Triton/
  CUDA 核）换成降级实现并在 `finally` 中还原（encoder 并行流、semseg 头
  einsum 同步切换），eager 训练/推理路径不变。解除 `GTR_ONNX_EXPORT_BLOCKER`，
  五条 GTR 路线（`detr/gtrvit-*`、`obb/gtr`、`pose/gtr`、`depth/gtr`、
  `semseg/gtr`）的模型目录行 `exportable` 翻开；每条路线补最小端到端
  导出门禁（CPU 可跑），GLA 降级配 fp64 朴素递推 oracle 与 fla CUDA
  对拍（含强负门控防溢出回归）。顺带修复 `ConvNormLayer_fuse
  .convert_to_deploy` 不幂等（验证阶段二次 deploy 会崩）与 pose 后处理
  gather 维度错配（按关键点张量秩双分支：部署布局 (B,N,K,2) 走上游
  `deploy_mode` 逐点 gather，扁平训练布局 (B,N,K·2) 保持原路径——此前
  部署后的每次真实姿态推理都会在 `torch.gather` 崩溃）。GTR 导出面同步
  补齐：fp16 导出可用（GLA 经 `onnx_fp32_op_types` 钉住 MatMul/Resize
  保持图一致；GEMM 为主的 GTR 在 ORT CPU 上 fp16 更慢，见 known_limits）；
  `dynamic_spatial=True` 与 CPU `backend='torch'` 两类不适用场景由
  duck-typed 模块树守卫快速可读拒绝（`dynamic_spatial_blocker` /
  `cpu_eager_blocker`），不再出现符号推理 grinding 或晦涩的 Triton 报错。

### 文档

- **任务示例默认全面切换到 GTR**：`examples/tasks/` 六条 GTR 路线的推理与
  训练脚本统一默认 GTR——`detection.py` 默认权重从 `rfdetr/rf-detr-nano`
  改为官方 `detr/gtrvit-s`（MIT，含 `--model rfdetr/rf-detr-nano` 的 CPU
  友好回退），`detection_train.py` 从 RT-DETR tiny 改为 `detr/gtrvit-s`
  tiny（64px / 10 queries / 随机初始化），`oriented_detection_train.py` /
  `instance_segmentation_train.py` / `semantic_segmentation_train.py` 从
  rtdetr-obb / dinov2-eomt / torchvision 线性头改为 `gtr/s` tiny
  （随机初始化）；GTR 训练/推理脚本统一前置 `gtr_requirement()` 守卫
  （CUDA + fla 缺失时给出可读指引）。`_common` 的 RT-DETR tiny 常量随调用
  方移除；GTR 部署边界（tiny 导出 / `--dry-run` 离线检查）写入脚本 docstring。
  已验证：8 个训练示例 `--dry-run` 全过，4 个改动脚本在 CUDA 上 2 步真实
  跑通（train → val → export），6 个 GTR 推理脚本加载官方权重全部出框/
  出图（机场图上 `detr/gtrvit-s` 检出 12飞机 + 1巴士）；tests/test_task_examples.py
  33 项全过。`examples/tasks/README.md` 与 `docs/task_examples(.en).md`
  的安装 / 任务矩阵 / 默认权重表同步更新。
- **双语 README 重构为「GTR 优先」**：GTR（MIT）提为首个正式章节——一个共享
  骨干覆盖六任务的定位、26 个官方权重的逐任务指标表与部署边界；快速开始
  全程 GTR 化（安装 → `download_gtr_assets.py` 下载 → `pretrained=True`
  推理/导出 → `model_args.checkpoint` 官方权重直载微调）；任务入口表只列
  GTR 路线（全景分割 / 分类标注暂无 GTR）；折叠能力区新增「GTR 训练与部署
  细节」；移除「开发」章节与安装节的 extras 一览折叠块（安装命令改为 GTR
  组合 `.[gtr,download,export]`，其余 extras 指向 `pyproject.toml` 与权重
  下载文档；贡献 / 发布 / 安全披露入口并入「许可与溯源」）；能力子系统、
  检测谱系与用户自持权重、导出部署、视频推理、serving、进阶功能收敛为折叠节。顺带同步两处过时表述：GTR 导出状态改为「ONNX/TensorRT/OpenVINO
  全支持（GLA 纯 ONNX 降级，固定尺寸图）」，与已知限制页及目录行为一致；
  版本号更新为 0.3.2。
- **P0 生态互链落地**：双语 README 新增「行业方案 / Industry solutions」
  小节，链接 8 个行业场景方案仓与生态设计页；8 个场景仓 README 已加
  "Powered by AutoVision" 徽章、PyPI 徽章与框架回链，并统一添加
  `autovision` / `video-analytics` topics。
- **新增《行业方案》页**（`docs/ecosystem.md`）：以解决方案介绍为主——
  通用三模型流水线范式、八个场景方案的图文介绍（外链各方案仓实景图）、
  底座稳定接缝一览；生态治理（Open-Core 边界、许可红线、P0–P4 计划）
  以折叠节保留。文档站导航新增对应条目。
- 溯源文档测试随 README 调整同步：`DEIMv2` 字样断言移出 README 检查
  （删除记录仍由溯源台账与已知限制页的测试守住）。

## [0.3.2] - 2026-10-19

### 文档

- **PyPI 页面移除 0.3.0 licensing status 说明段**（该信息保留在仓库
  `CHANGELOG.md` 与 `docs/provenance/LEDGER.md`，不再出现在发行页正文）。

## [0.3.1] - 2026-10-19

### 文档

- **PyPI 页面改为英文**：`README_PYPI.md`（发行包长描述）整体改写为英文；
  包内中文 README（`README.md` / `README.en.md`）不变，仓库仍为双语入口。

## [0.3.0] - 2026-10-19

### 许可与兼容性

- **删除 DEIMv2 派生的 STA（2026-10-17）**：`SpatialPriorModule` 从代码库
  移除，`DETRVariantSpec` 不再携带 `use_sta` / `conv_inplane`；DINOv3 ViT
  变体保留并改用 no-STA adapter，DINOv3 ConvNeXt 与其他家族不变。发行物
  不再包含 DEIMv2 代码，许可阻塞解除。
- **旧 STA-era ViT 检查点不再兼容**：含 `backbone.sta.*` 的基线 `.pt` 与
  用户自持的 `dinov3-vitt16` / `dinov3-vitt16plus` / `dinov3-vits16` 会被
  加载器显式拒绝；需使用重新训练/导出的 no-STA 检查点，或改用 ConvNeXt
  资产。用户自持的 4 个 ConvNeXt 资产不受影响。

### 新增

- **许可与致谢文档**：README 新增 `致谢（Acknowledgements）`；
  新增 `THIRD_PARTY_LICENSES.md`（GTR MIT 全文）与
  `docs/provenance/LEDGER.md`；`NOTICE` 补齐 RF-DETR / PicoDet /
  ByteTrack / SAHI 归属，并记录 DEIMv2 派生 STA 的删除与兼容性变化。
- **发布依赖**：`export` extra 下限提升为 `numpy>=1.25`（onnx / onnxscript
  在导入时使用 `numpy.exceptions`，numpy 1.23/1.24 下会直接失败）；核心训练
  安装仍保持 `numpy>=1.23`，与 super-gradients 的互斥由既有 extras 冲突声明
  处理。

- **黄金路径 / 部署契约（roadmap phase 1–2 代码基础）**：
  - **任务级模型目录**：`autovision.registry.catalog` 汇总 RF-DETR N/S/M/L、
    GTR s/m/l/x、自研 DETR 谱系与 RT-DETR 的许可 / 权重 / AP / 可导出状态；
    `list-models --task detection --pretrained`、`model-info <key>` 可查。
  - **预训练检测句柄**：`autovision.detection(<key>, pretrained=True)` /
    `DetectionModel.from_pretrained` 惰性加载发布权重（RF-DETR 全链；GTR 在
    GLA/CUDA 可用时走官方 EMA 权重装配，否则给出可读安装/硬件指引）；
    `predict / val / export` 不再要求先训练。
  - **训练配方**：`recipe=finetune | coco-finetune | low-data` 或 YAML 路径，
    只覆盖 `model_args` / `transform_args` / `trainer_args`，显式参数优先。
  - **版本化后处理规格**：ONNX / OpenVINO / TensorRT 导出件旁写
    `<artifact>.postprocess.json`（schema_version=1；top-k、阈值、原图映射、
    resize/mean/std），附带 NumPy 参考解码器；分割 / 姿态 / 深度 / 分类导出
    不写 sidecar。
  - **端到端框级 parity**：`autovision.export.verify_end_to_end_parity`
    比较最终框 / 分数 / 类别（宽松匹配 + 像素容差 + missing/extra 计数），
    可直接输出 JSON 报告；不再只比张量求和。
  - **OpenVINO 运行时**：`InferenceEngine(backend="openvino")`、`run_stream`、
    serving 与 CLI `predict` 均支持 `.xml` IR（`openvino_device=CPU|GPU|AUTO`）。
  - 修复 legacy ONNX 导出回退时多传一个位置参数导致的
    `TypeError`（dynamo 不可用时的真正回退路径）。
  - 新增测试：模型目录、预训练句柄、配方（含 1 步真实训练）、后处理规格、
    框级 parity、OpenVINO 运行时接线。

- **数据体检与难例闭环（phase 3 代码基础）**：
  - **数据体检**：`autovision.data.quality` + CLI `data-check`：坏图、精确/近重复
    图、跨 split 泄漏、框越界/非法尺寸、原始类别 id、类别分布与 val 缺类；
    产出 `report.json` / `report.md` / `manifest.json`，`fail_on` 控制退出码。
  - **难例清单**：`analyze-errors` 额外写 `hard_examples.json`（逐图 TP/FP/FN、
    FP 归因、漏检面积）；`split=` 记录来源，采样器只消费 `train`，避免验证集
    泄漏进训练。
  - **难例采样**：`autovision.data.hard_examples` 的 manifest + 确定性
    `WeightedRandomSampler`；`train_object_detection(..., hard_examples=...)`
    按 `min(max_repeat, 1 + alpha × 归一化误差)` 加权。v1 单机；DDP 组合被
    显式拒绝（DistributedSampler 会覆盖权重）。
  - **多轮自训练**：`scripts/self_train_detect.py --rounds N --min-improvement`；
    逐轮产物（`rounds/r{i}/`）、学生提升后升级为下一轮教师、不升即停；
    顶层 `summary.json` / `summary.md` 记录 stop_reason 与最优模型。
    `--rounds 1` 保持原有单轮布局。
  - **长尾采样**：`autovision.data.samplers` 的 `class_balanced` /
    `repeat_factor`；`train_object_detection(..., long_tail_sampler=...)`
    接入训练，与 `hard_examples` 互斥，v1 单机。
  - **标注工具互通**：`autovision.data.interchange` 的 CVAT XML（bbox /
    polygon）与 Label Studio JSON（rectangle / polygon / keypoint-as-box）→
    COCO；`convert-data from=cvat | label-studio`，含 CLI 与 loader 往返测试。
  - 新增测试：数据体检（核心 + CLI 夹具）、难例 manifest/采样/误差分析输出/
    1 步真实训练接入、多轮自训练（真实 2 轮）、长尾采样权重/确定性/训练接入、
    CVAT / Label Studio 转换。

- **生态与企业能力（phase 4 代码基础）**：
  - **插件协议**：`autovision.plugins` 的 `autovision.plugins` entry-point
    发现、`PluginInfo` / `PLUGIN_API_VERSION`、kind 校验、失败收集 /
    strict 模式；`model_source` 自动接入编码器目录；CLI
    `plugins action=list|check`；`examples/plugin-demo` 可安装模板。
  - **企业合规报告**：`autovision.compliance` 汇总框架/依赖许可/模型卡+
    sha256/数据快照；CLI `compliance-report` 输出 `compliance.json` +
    `compliance.md`，支持 `strict` / `no_timestamp`。
  - **私有注册表**：`current` 指针、`rollback`、append-only `audit.jsonl`、
    模型卡/后处理边车随产物复制；部署门禁
    `Sha256Gate` / `ModelCardGate` / `LicenseGate` / `BenchmarkGate` /
    `ParityGate` / `DataSnapshotGate`；CLI
    `registry action=register|promote|rollback|current|list|audit`。
  - **LTS / SLA 文档**：`docs/lts_policy.md`、`docs/support_policy.md`
    （6 个月 LTS 节奏、12/6 个月支持窗、P1–P3 响应目标）。
  - **硬件伙伴包模板**：`deploy/packs/_template/` 的 `convert.py`（校验
    ONNX/spec、执行厂商命令）与 `parity.py`（目标端 JSONL 对拍）；一伙伴
    一包，不预研厂商矩阵。

- **谱系发布治理（P0）**：
  - `SPECTRUM` 档位状态改为三态 `planned / spec-frozen / released`；
    `detr/n`、`detr/s` 明确为 spec-frozen（规格冻结，预训练权重 / AP 待发布）。
  - 新增 `autovision.registry.spectrum_release` 发布门禁：`released` 必须有
    已发布资产、模型卡（许可 + sha256）、训练 seed、全量 COCO AP、延迟记录
    与存在的复现脚本。
  - 新增 `scripts/train_spectrum.py`、`scripts/eval_spectrum_coco.py` 骨架
    （支持 `--dry-run` 与 coco8 smoke 路径）。
  - 新增 RF-DETR 五档模型卡（官方 COCO 指标 + pinned sha256，
    `autovision.registry.model_cards`；Base 标记上游弃用）；基准表补
    RF-DETR 官方 T4 口径（N/S/M/L）。
- **数据闭环（P1）**：
  - 伪标注 / 自训练：`autovision.tasks.detection.shared.pseudo_label` +
    `scripts/self_train_detect.py`（教师 `.pt` → 伪标注 → 学生训练 → 标注 val 评测
    → 报告），CLI `pseudo-label`（逐类阈值、SAHI 可选、批量推理）。
  - 误差分析：`autovision.metrics.error_analysis` + CLI `analyze-errors`（在线 /
    离线两模式）：TP/FP/FN 匹配、FP 归因（duplicate / classification /
    localization / background）、按类别与 COCO 面积分档切片、top-FP/FN 画廊
    （`report.md` / `report.json`）。
  - 转换扩展：PASCAL VOC 解析器、YOLO-pose → COCO 关键点、YOLO-seg 多边形 →
    COCO 分割、COCO 合并（`convert-data from=voc | yolo-pose | yolo-seg | merge`）。
  - pose 增广：关键点感知 mosaic + mixup（默认关闭、步窗口可调），mosaic 几何与
    检测族共享同一实现。
- 检测 mAP 指标新增 `map_small` / `map_medium` / `map_large`（COCO 面积分档，
  此前固定 `"all"`；headline 数字不变）。

- **部署广度（P1）**：
  - ONNX 导出支持**动态 H/W**（`export_onnx(dynamic_spatial=True)`，含第二尺寸
    torch↔ONNXRuntime parity 门禁；`input_dynamic_shapes` 规格位被导出器消费），
    并要求 dynamo 导出器（legacy 路径会把尺寸烘成常量——fail-closed）。
  - **OpenVINO** 一等后端：`export_openvino`（ONNX→IR + 编译后对拍门禁 +
    metadata 边车）、`export out=model.xml`、benchmark 后端
    `openvino`（设备可切 CPU/GPU/AUTO，纳入 parity 门禁）。
  - **TensorRT int8**（QDQ 路线）：`export_tensorrt(precision="int8",
    calibration_data=...)`（QDQ ONNX 先过 int8↔fp32 门禁，再以 strongly-typed
    网络构建引擎），benchmark 的 `tensorrt` 后端支持 `precision="int8"` +
    `quantization={...}`（首层 fp32 配方用 `nodes_to_exclude`）。
  - **多路流运行时**：`MultiStreamRunner` / `run_multi_stream`（`predict
    sources=...`）：每路 reader 线程 + 单一共享 batcher、每路
    `predictions.jsonl` / `annotated.mp4`、断流指数退避重连、队列
    `drop_oldest|block` 背压、`summary.json` 每路帧数/丢弃/重连/错误。
    v1 不做跨路跟踪。
  - **PicoDet NMS 后处理规格**：`postprocess.json` 新增 `family="picodet"` +
    class-offset NMS；NumPy 参考解码与 torch 后处理逐框对拍（含重复框抑制）。
  - **int8 vs fp32 AP 报告**：`scripts/measure_int8_accuracy.py` +
    `autovision.export.int8_report`：同一 split/前后处理下导出 fp32/int8、
    跑全量 COCO AP、输出 `report.json`/`report.md`（delta、门禁、校准集、
    硬件）；`--max-ap-drop` 控制退出码。
  - **C++ ONNX Runtime 示例**：`examples/cpp/`（header-only decode + host
    test + ORT/OpenCV main + CMake）；host 测试由 g++ 真实编译运行，
    完整 runtime 构建依赖 ORT/OpenCV 开发库。
- int8 QDQ 链与 TensorRT 的互操作修复：int32 常量 `DequantizeLinear` 折叠回
  fp32（TRT 11 解析器要求）；激活量化默认**对称**（`activation_symmetric`，
  非对称 zero-point 被 TRT 拒绝）。
- `export` extra 显式声明 `onnxscript`（dynamo 导出器的依赖，缺失时静默回退
  legacy）；新增 `openvino` extra。
- **int8 首层 fp32 配方自动建议**：`suggest_first_layer_nodes`（导出自
  `autovision.export`）枚举 fp32 图的前导 Conv 节点；TensorRT int8 构建失败
  的提示直接给出可粘贴的 `nodes_to_exclude` 配方，复现脚本
  `scripts/measure_new_backends.py --quantization-exclude auto` 一键套用；
  OpenVINO IR 支持 `precision="fp16"`（编译后对拍门禁照常）。

- **超参搜索（B2）**：`autovision.hpo`（`SearchSpace` 分布规格 + `run_search`
  随机 / 网格 / ASHA 驱动 + `task_trial` 复用训练入口与 Pydantic 覆盖面）与
  CLI `hpo`；产出 `report.md` + `trials.jsonl` + `best.json`，已完成试验目录
  自动复用（断点续跑），ASHA 逐 rung 以框架原生 resume 续训而非重训。
- **VOC 可直接训练（B3）**：`download → convert-data from=voc → train` 三步配方
  入 README；转换器保持 VOC 规范类名（排序 id）、`difficult → iscrowd` ignore
  区；端到端训练测试（CPU）在案。

- **推理侧能力（P2 战略项的一部分）**：
  - **ByteTrack 跟踪**（`autovision.tracking`，A3）：卡尔曼运动模型 + 三阶段
    IoU 关联（高分池融合检测分数 / 低分池 / 未确认轨迹），class-aware；官方
    门限齐备（新轨迹需 `track_thresh + 0.1` 且连续命中才发布，低分下限 0.1，
    首帧激活例外）；`predict source=... track=true` → JSONL 逐帧 `tracks`
    （`age` = 创建以来帧数）+ 渲染 id 标注，tracker 按源视频真实帧率折算
    丢失窗口。
  - **参考 serving**（`autovision.serving`，F2；`serving` extra）：FastAPI
    `/predict`（批处理按输入顺序、与评测同源前后处理）+ `/healthz` +
    Prometheus `/metrics`（请求按端点与状态、图片数、延迟 summary + 峰值；
    5xx 同样计数）；显式边界（无鉴权/TLS/编排，漂移检测等按产品触发）。
  - **数据版本化**（`autovision.data.versioning` + `data-snapshot` CLI）：内容
    寻址快照（根相对清单 sha256，`deep=true` 含内容哈希）+ 漂移报告
    （added/removed/resized/content_digest）+ 搬迁副本校验；spec 内相对路径按
    spec 文件目录解析，写入默认不覆盖（`overwrite=true`），deep 校验要求
    deep 快照（浅快照显式报错）。

### 变更

- **PyPI 打包**：发行名改为 `autovision-framework`（import 名仍 `autovision`，
  CLI 仍 `autovision-cli`）；新增 PyPI 专用 README、PEP 639 许可元数据、
  sdist 的 GTR / MSDA CUDA-C++ 源清单，以及 tag 触发的 trusted-publishing
  CI job。`super-gradients` 与 `export` / `transformers` / `rfdetr` / `video`
  的 extras 互斥已显式声明。
- **STA-era 用户自持资产**：`resolve_user_held_asset` 默认在 sha256 校验前
  拒绝 3 个含 `backbone.sta.*` 的 ViT 资产；新增
  `USER_HELD_LOADABLE_ASSETS` / `USER_HELD_UNSUPPORTED_ASSETS` 区分可加载与
  仅审计记录，错误消息列出 4 个可加载 ConvNeXt 替代项。审计场景需显式传
  `allow_unsupported=True`；`verify_user_held_asset` 仍可离线核验 identity。

### 修复

- **分类 / 线性语义分割的 `auto` 默认值取错**：`default_batch_size` /
  `default_steps` 此前声明为 Pydantic 字段，但 CLI 解析 `auto` 时读取类属性，
  实际回退到父类的 16 / 10000（分类应为 128，线性语义分割应为 32 / 20000）。
  现改为与其它任务一致的 `ClassVar`，并加回归测试。
- **HF 骨干（DINOv2/DINOv3 系）的 ONNX 导出在 transformers ≥5 环境下全线失败**：
  首个 eager 前向会先安装 output-capturing hooks（在 `threading.Lock` 下），
  而 dynamo 无法 trace `with <lock>`，严格导出直接失败——现于 trace 前做一次
  warm-up 前向（数值透明）。同批修复：严格 trace 失败且空间轴为**静态**时回退
  legacy TorchScript 导出（此前不回退、直接抛 dynamo 内部错误——文档承诺的
  回退路径此前只覆盖第二步之后）；动态轴失败改为**可读的 AutoVisionError**
  （含"模型在输入尺寸上有 Python 级分支/守卫"的处置指引）。token→patch 网格的
  折叠改由**输入尺寸**推导（`H // patch`），动态 H/W 下不再对符号形状调
  `math.isqrt`：HF/ViT 线的 eager 报错与既有行为一致；timm 包装器在 patch size
  可发现时按输入推导网格（`dynamic_img_size` 类模型由"报错"变为可用，token 数
  与网格不匹配仍报错）。
- int8 首层 fp32 配方此前需手工枚举 fp32 图节点名（见上「自动建议」）。

- CLI 训练入口的显式设备串（`devices="cpu"` / `"cuda:1"`）此前在 9 个任务入口以
  `int(devices)` 崩溃——现按单设备解析（与 `engine.resolve_device` 语义一致）。
- `convert-data` 的文档示例此前无法执行（`out=` / `images=` / `annotations=` 与
  函数参数名不匹配、连字符键直接报错）——已修并加回归测试。
- 新版 CLI 键名统一支持 `--kebab-case`（自动映射到下划线参数名）。
- **步窗口增强在 `num_workers>0` 下静默不生效**（先存缺陷，检测 / 旋转框 / 姿态三族同存）：
  含步窗口调度的训练 loader 现以非持久 worker 构建，窗口边界跨越时把当前 step
  重新下发到 worker。

### 视觉增强与视频工具（迁移自 supervision 0.30）

#### Added

- **增强标注器**（`autovision.visualize.effects`）：`TraceAnnotator`（按
  `tracker_id` 累积轨迹）、`HeatMapAnnotator`（时序热力累积）、
  `BlurAnnotator` / `PixelateAnnotator`（隐私模糊/马赛克）、
  `BoxCornerAnnotator` / `RoundBoxAnnotator`（角标/圆角框）、
  `HaloAnnotator` / `EdgeAnnotator`（mask 光晕/轮廓）、
  `BackgroundOverlayAnnotator`（mask 外压暗）。全部为 Pillow + NumPy
  实现，mask 存在时优先按像素处理。
- **骨架注册表**（`autovision.visualize.skeletons`）：COCO-17 与 GHUM-33
  边表；`KeypointAnnotator(skeleton=...)` 支持注册名或自定义零基边序列，
  默认行为保持 COCO-17 不变。
- **检测几何工具**（`autovision.geometry.boxes` / `autovision.geometry.masks`）：
  `clip_boxes` / `pad_boxes` / `denormalize_boxes` / `scale_boxes` /
  `move_boxes`、`xywh` / `xcycwh` / `xcycarh` 转换、`box_iou`、
  `box_non_max_suppression`、`box_non_max_merge`、`mask_to_xyxy` /
  `xyxy_to_mask`、`polygon_to_mask`、`mask_to_polygons`、
  `approximate_polygon`（NumPy RDP）、`filter_polygons_by_area`。
  `mask_to_polygons` 惰性依赖 OpenCV（`video` extra），其余仅需 NumPy/Pillow。
- **视频工具**（`autovision.inference.video`）：`get_video_frames_generator`、
  `process_video`（逐帧回调 + 可选编码）、`VideoSink`、`ImageSink`
  （按间隔落盘）、`FPSMonitor`。沿用 `InferenceResult` 的 RGB 通道约定，
  与 `run_stream` 的单路/多路管线并存。
- **测试**：`tests/test_geometry_ops.py`、`tests/test_visualize_effects.py`、
  `tests/test_video_utils.py` 覆盖转换、NMS/NMM、mask/polygon、隐私效果、
  轨迹/热力、骨架注册与视频 sink。
- **示例**：`examples/tasks/detection.py` 新增
  `--effect {none,corner,round,blur,pixelate}`，用上述标注器演示检测结果的
  角标/圆角框与隐私模糊/马赛克；默认 `none` 行为不变。
- **溯源**：`THIRD_PARTY_LICENSES.md`、`NOTICE`、`docs/provenance/LEDGER.md`
  登记 Roboflow supervision 0.30 的 MIT 参考关系；实现为 AutoVision 重写。

### 任务示例（examples/tasks）

- 为检测、旋转框、实例/全景/语义分割、姿态、深度、分类各新增一个可运行示例；
  默认生成 2–4 张图的合成数据并训练 1–3 步，支持 `--dry-run`、
  `--predict-image`、`--export`。
- `src/autovision/tasks/README.md` 提供任务到示例的索引；MkDocs 新增
  “任务示例”页面。
- **任务示例统一拆分**：每个任务提供 `<task>.py`（纯推理：RF-DETR/GTR 官方
  权重或 `--weights` 导出的 `.pt` → 预测 → 标注）与 `<task>_train.py`
  （训练/验证/导出）；`panoptic` / `classification` 无官方任务权重，推理需
  `--weights`。
- 新增 `examples/tasks/detection.py`（纯推理：加载 `rfdetr/rf-detr-nano`
  官方权重 → 预测 → 标注）与 `detection_train.py`（原 coco8 训练示例）；
  `inference_demo.py` 保留为兼容别名。
- 移除来源/许可未知的本地街景图；所有推理示例改用 CC0
  `examples/tasks/assets/airport_aerial.jpg`（San Diego 机场俯视图，
  Wikimedia Commons / Mertbiol），并随 sdist 发布。
- 新增 `examples/tasks/yolo/`：Ultralytics YOLO 检测 / 分割 / 姿态 / 旋转框 /
  分类 5 个推理示例（AGPL `ultralytics` extra，权重缓存到
  `~/.cache/autovision/yolo`，复用 AutoVision 的 `InferenceResult` 与标注器）。
  检测/分割/姿态权重默认从 ModelScope `Ultralytics/YOLO11` 下载。
- 新增 `tests/test_task_examples.py`：逐个校验 `--help` 与 `--dry-run`
  数据契约。

### 推理侧：统一结果、任务感知渲染与分析（借鉴 supervision 设计）

#### Added

- **`InferenceResult`**（`autovision.results`）：推理边界统一容器，
  覆盖检测 / 实例分割 / 旋转框 / 姿态 / 分类 / 语义 / 全景 / 深度；
  支持 `select` / `filter` / `to_numpy` / `to_record`，以及
  `to_json(masks="rle"|"dense"|"none", max_elements=...)`。
- **键常量**（`autovision.keys`）：`bboxes` / `scores` / `labels` /
  `masks` / `keypoints` / `polygons` / `mask` / `depth` / `segment_ids`。
- **任务感知渲染**（`autovision.visualize.annotators`）：Box / Label / Mask /
  Polygon(OBB) / Keypoint(COCO 骨架) / 分类 / 语义 / 全景 / 深度；
  `draw_prediction` 现在按结果字段分派，pose / OBB / 分类不再因缺少 `bboxes`
  抛 `TypeError`。
- **区域统计与平滑**（`autovision.tracking.analytics`）：`LineZone`（按类别
  in/out 穿越）、`PolygonZone`（当前区域内计数）、`TrackSmoother`（按轨迹
  滑动平均）；`run_stream(line_zones=..., polygon_zones=..., smooth=True)`
  将 `zones` 写入 JSONL 并叠加到渲染画面。
- **混淆矩阵**（`autovision.metrics.confusion`）：`analyze-errors` 的
  `report.md` / `report.json` 新增 `(类别 + FP) x (类别 + FN)` 矩阵，
  classification FP 落在 `[真值类, 预测类]`。
- **SAHI 合并策略**：`predict_sahi(..., batch_size=N, merge="nms"|"nmm",
  overlap_metric="iou"|"ios")`；tile 间先统一 per-label 过滤，再与全局结果
  合并；NMM 用并集框 + 面积加权分数。

#### Changed

- **单路视频管线**：`run_stream` 改为 reader / infer / writer 三段式
  （`prefetch` / `writer_buffer` 有界队列）；新增 `decode_s` / `infer_s` /
  `render_s` / `write_s` / `fps`；`StreamRunResult.zones` 汇总区域计数。
- **mask 序列化**：serving `/predict`、`predict_to_files`、stream JSONL 默认
  输出 COCO RLE；新增 `mask_format` / `max_json_elements`（后者约束图像级
  mask/depth 与扩展字段，实例行不截断）。
  实例分割 dense mask 不再被直接 `tolist()` 成数十 MB 的响应。
- **依赖**：核心依赖新增 `defusedxml>=0.7.1`；VOC / CVAT 标注 XML 改用
  `defusedxml` 解析，移除 `# nosec B314` 豁免。

#### Fixed

- `draw_prediction` 对 pose / OBB / 分类 / 深度结果的 `TypeError`；mask-only
  实例分割（EOMT）现在渲染 mask 与类别标签。
- `run_stream(line_zones=..., render=True)` 的 zone 叠加崩溃（PIL/NumPy 类型
  不匹配）；writer 失败或超时不再静默返回截断视频。
- `run_stream` 单路解码、推理、编码串行导致的吞吐损失；停止时保证 reader
  sentinel 与 VideoCapture 释放。
- 大图 SAHI 一次性堆叠全部 tile 的显存峰值风险；默认 NMS 走
  `torchvision.ops.nms`，NMM/IOS 为显式选项。
- 非有限浮点（NaN/Inf）在 serving / JSONL / predictions.json 中输出为
  `null`，严格 JSON 解析器不再失败。
- **RF-DETR 预训练类别映射**：官方权重是 91 类 COCO category-id 头
  （0=N/A，1..90 含缺口），此前按 80 类连续索引截断，导致预测标签与类名
  整体错位；现保持 91 类并为每个 id 返回正确类名（GTR/DETR 仍为 0..79
  连续头）。
- CVAT XML 的 `file_name` 路径穿越（缺失 width/height 时）；mask 的
  uint8/float 类型统一按非零 / `>0.5` 归一化。
- 全景分割的 ignore 像素不再被涂色；`filter_scores` 对全景结果不再抛错。

### 修复（企业级审计轮次）

- **DDP 正确性**：D-FINE / RT-DETR 系 criterion 不再对训练 loss 调用
  `reduce_dict`（它会 detach）；2-rank backward 恢复正常，并新增多 rank
  回归测试。日志 reduction 与反传图解耦。
- **训练引擎**：无验证集时 `steps_run` 正确计数；`gradient_accumulation_steps<1`
  在入口报错；每个 rank 调用 `set_device`；实验 logger 故障不再中断训练；
  `save_last=False` 不再写 last checkpoint；确定性模式下不再强制 TF32。
- **指标尺寸**：depth 验证统一在 GT canvas 上计算；instance segmentation 将
  GT mask 重采样到原图尺寸后再做 COCO mask AP，非方形/原图≠网络输入不再崩溃
  或误评分。
- **HPO**：trial 复用要求 `params + budget` 指纹 sidecar 匹配；空间或预算变化
  会重跑，ASHA rung 同样生效。
- **数据完整性**：下载改为临时文件 + 校验 + `os.replace`；缓存压缩包在解压前
  重新校验 size/sha256；zip/tar 解压拒绝路径穿越。
- **合规**：`normalize_license` 改为锚定别名 + AND/OR/WITH 表达式归一化，
  修复 BSD-2、Apache 全名、专有许可证被误判为 MIT 等问题；未知原子 fail-closed。
- **日志**：文件 handler 在多次 run 之间替换而非累积（HPO 不再互相污染）；
  非 rank0 不再构造 MLflow/W&B/TensorBoard run；`finalize_loggers` 隔离单后端故障。
- **serving**：`/docs`/`/openapi.json` 默认关闭；新增有界排队
  `queue_timeout_s`（饱和返回 503 + `Retry-After`）与
  `timeout_graceful_shutdown`；非 ASCII 凭据返回 401 而非 500。
- **导出**：TensorRT runner 检查 `set_input_shape` / `set_tensor_address` /
  `execute_async_v3` 返回值；`export_tensorrt(verify=True)` 默认执行
  eager-vs-engine 数值门禁，失败删除 engine。
- **模型加载**：`load_task_model` 恢复导出时保存的 `normalize_args`；
  `TaskNetwork` 不再把 `__class__` 写进 `init_args`。旧 teacher checkpoint
  的完整 unpickler 回退改为显式 opt-in
  （`AUTOVISION_TRUST_TEACHER_CHECKPOINT=1`）。
- **CI**：测试作业安装 `serving`/`video`/`transformers` 等测试依赖，
  修复 collection 阶段的 ModuleNotFoundError；新增 CI 依赖契约测试。

### 第二轮企业级加固

- **统一冒烟入口**：新增 `scripts/smoke_enterprise.py`（quick/full 两档）与
  `docs/smoke_test.md`，覆盖训练→ONNX/OpenVINO/TensorRT→serving→registry→
  compliance→数据快照→K8s→DDP/FSDP，并作为 CI 门禁；full 在本机 RTX 4090D
  全绿（TensorRT 使用最终框级 parity，int8 保持张量级门禁）。

- **训练规模**：新增 `trainer_args.gradient_checkpointing`（HF 原生 / 自定义
  ViT-GTR blocks）与 `parallel="fsdp"` 任务训练：full-state checkpoint/optimizer
  gather、所有 rank resume、rank0 导出、验证期 `summon_full_params`、EMA 按
  `AUTOVISION_FSDP_EMA_INTERVAL` 更新、rank0 写失败 barrier 与进程组超时。
  `docs/parallelism.md` 记录 DDP/FSDP 及 DeepSpeed/ZeRO/弹性训练的明确边界。
- **EoMT**：补 `forward`（ONNX 导出可用）；加载 hub 权重强制 DINOv2/DINOv3
  许可 opt-in，未钉 sha256 时默认拒绝（`AUTOVISION_ALLOW_UNPINNED_WEIGHTS=1`
  显式放行）；`adapt_eomt_head` 已接入构建流程。
- **注册表**：`promote(gates=None)` 在文件系统和 MLflow 后端都默认执行标准
  门禁；MLflow 注册/晋级保存并取回 sidecar、`get` 重校验 sha256、
  `list_entries` 跳过非 AutoVision 模型版本。
- **数据**：mmap 指纹覆盖标签文件路径/大小/mtime 与 `ignore_classes`；
  `overwrite=True` 强制重建；user-held 权重解析强制 sha256 校验（摘要缓存）。
- **serving**：`/metrics`、`/drift` 在设置 API key 后要求鉴权
  （`protect_observability_endpoints`，探针保持公开）；新增
  `deploy/k8s/` Deployment/Service/HPA/PDB/PVC 参考清单。
- **合规/质量**：`compliance-report fail_on=error|warning|none`、
  CycloneDX SBOM、报告 sha256 清单、数据快照重算与漂移检测；CI 新增
  coverage 下限、bandit（medium/high）与核心模块 mypy 门禁。
- **TensorRT**：新增 `scripts/verify_tensorrt_engine.py` 与 self-hosted
  `trt-nightly.yml`，在目标 GPU 上执行真实 engine build/parity。

### 安全与生产加固

- **serving**：新增可选 `api_key`（Bearer / `X-API-Key`，常量时间比较）、
  `max_body_bytes` 流式请求体上限、`max_image_pixels` 解压炸弹防护、
  `/readyz` 就绪探针、`X-Request-ID` 透传；推理改在线程池执行并以
  `max_concurrency`（默认 1）限制并发，事件循环不再被 forward 阻塞。
- **serving 生产化**：新增 TLS 与多 worker 启动（`serve(workers=...)`，
  多 worker 走 import-string factory）、进程内 token-bucket 限流
  （`rate_limit_per_minute` / `burst` / 可信代理头）、预测分布漂移
  （`/drift`，标签分布 + 分数直方图 PSI，基线可由
  `build_drift_baseline` 生成）以及 A/B/canary 分流
  （`variants=` / `variant_weights=`，按 request ID 稳定哈希，支持
  `X-AB-Variant` 强制）。每个 variant 的请求数与 PSI 进 `/metrics`。
- **训练早停/回调**：task training 新增 `early_stopping_patience` /
  `early_stopping_min_delta`（CLI `trainer_args` 与 Python API），触发时保存
  latest checkpoint 并在 `TaskTrainResult` 返回 `stopped_early` / `stop_step`；
  `fit_task(callbacks=...)` 提供 `on_train_start` / `on_step_end` /
  `on_validation_end` / `on_early_stop` / `on_train_end` 钩子（rank 0，
  callback 异常只告警不中断训练）。
- **fp16 训练**：`16-mixed` 在 CUDA/NPU 上接入 `GradScaler`，scaler 状态随
  checkpoint 保存/恢复；梯度检查与裁剪在 unscale 之后进行。旧 checkpoint
  无 scaler 键时保持兼容。
- **确定性**：新增 `AUTOVISION_DETERMINISTIC=1`，开启 torch 确定性算子和
  cuDNN deterministic（默认关闭，部分算子会报错）。
- **下载完整性**：`AUTOVISION_OFFLINE=1` 现在同时拒绝数据集与可下载模型联网；
  新增 `AUTOVISION_REQUIRE_DATASET_HASHES=1` 拒绝无 sha256 的数据集条目；
  COCO 数据集 URL 改为 HTTPS；数据集/模型下载传入超时。
- **模型注册表**：key 校验拒绝 `.`、`..`、路径分隔符、NUL 和空白键；
  读写路径使用跨进程文件锁（读不会看到半拷贝条目，并发注册有唯一胜者）；
  MLflow 后端接受 `gates=` 并显式 fail-closed（不会静默跳过部署门禁）。
- **运行清单**：训练输出目录新增 `run_manifest.json`（命令、git commit、
  环境白名单、依赖版本、GPU）；resume 写独立 `run_manifest.resume-*.json`。

### 工程与发布

- 新增 `SECURITY.md`（私有漏洞披露流程、支持范围、响应目标）与
  `.github/dependabot.yml`（pip / GitHub Actions 每周更新）。
- 新增 `py.typed`（PEP 561），包数据随 wheel 发布。
- GitHub CI 测试矩阵覆盖 Python 3.10 / 3.11。
- **双语文档站点**：MkDocs Material + `mkdocs-static-i18n`，中文为默认站点、
  英文构建在 `/en/`；新增 `README.en.md` 与首页 / 任务示例 / 已知限制 /
  用户自持权重 4 个英文页，未翻译页自动回退中文。
- **发布元数据**：`pyproject.toml` 补齐 `[project.urls]`，MkDocs 配置写入
  `site_url` / `repo_url`；新增 `scripts/publish_clean_cut.sh` 生成并校验
  单提交发布快照。
- 新增回归测试：serving 鉴权/体积限制/并发/就绪、TLS/多 worker 启动参数、
  限流、漂移 PSI、A/B/canary 分流、fp16 scaler、确定性开关、早停/回调、
  下载门禁、注册表并发与路径校验、运行清单、打包/安全政策。
- 测试稳定性：panoptic 端到端确定性用例固定 CPU，避免 GPU bf16 归约顺序
  造成的假失败。

### 发布前可靠性加固（2026-10-19）

- **serving / 流**：新增 `inference_timeout_s`（默认 120s；超时返回 504 并标记引擎
  不可再用，后续请求 503，避免复用卡死的非线程安全后端）；未处理异常统一返回带
  `request_id` 的结构化 500；探针不再返回绝对路径，RTSP 凭据在错误/统计中脱敏；
  `/metrics`、`/drift` 与 `/predict` 共用限流；writer 失败即时中止；截断视频
  解码失败不再被当成 EOF；多路流新增 `frames_failed` 计数；`track=True` 用于
  非检测结果时显式报错；区域/平滑边界与零面积框不再产生误计数或 Kalman 崩溃。
- **训练 / 数据**：有限 loss + NaN/Inf 梯度现在会中止训练；checkpoint 使用唯一
  临时文件 + fsync，并保存/恢复 Python/NumPy/Torch RNG 状态；pretrain 与
  best-export/sidecar 写失败统一走 rank-0 barrier；空训练/验证 loader 显式报错；
  mmap refcount 损坏可恢复；数据快照对缺失路径默认严格失败；COCO `file_name`
  做目录逃逸校验；JPEG EXIF 方向归一化，截断图片不再静默进入训练；DDP 下自定义
  采样器显式拒绝而不是被静默替换。
- **导出 / 注册表**：ONNX 与 int8 导出改为临时文件 + 原子替换，失败保留上一版；
  `ParityGate` 默认要求真实 parity 报告（可绑定 sha256），不再接受自报标志；
  注册表读取时重新校验模型与 sidecar sha256、拒绝 manifest 路径逃逸；manifest
  原子写入；审计日志拒绝畸形行、使用 hash chain + fsync；插件拒绝混合迭代返回、
  严格发现失败时回滚本次已注册插件并加锁。
- **故障恢复与验收补强（同日第二轮）**：checkpoint / ONNX / int8 / 注册表
  manifest / run manifest 写入前清扫同目标超过 1 小时的崩溃残留临时文件
  （`runtime/fs_utils.sweep_stale_temp_files`，年龄保护绝不触碰在写文件）；
  checkpoint 磁盘满（ENOSPC）统一包装为 `CheckpointError` 且旧文件完好；
  加载跨版本 checkpoint 记录 warning；mmap 缓存损坏/截断改为带重建指引的
  明确报错（运行中被外部截断属 OS 层 SIGBUS，已写入 known_limits）；serving
  引擎毒化后 `/readyz` 返回 503（`poisoned=true`）、`/healthz` 携带
  `poisoned` 字段，编排器可据此摘流重启；新增 kill -9 中途击杀训练 + 断点
  续训的故障注入测试（`tests/test_crash_recovery.py`）。

## [0.2.0] - 2026-10-16

首个对外发布版本（clean-provenance 发布线）。

### 新增

- **任务面 8 个**：检测、旋转框、实例 / 全景 / 语义分割、姿态、深度、分类；
  Python API 与 `autovision-cli`（20 条命令）共享同一套 Pydantic 严格校验。
- **检测谱系 `detr/*`**（39 个注册变体）：DINOv2 / DINOv3 ViT 与 ConvNeXt、
  HGNetV2 装配、GTR 解码器装配；档位谱系 `n` / `s` 冻结为公开档位。
- **GTR 多任务家族**（MIT，26 个官方权重，sha256 全钉住）：检测 / 实例分割 /
  语义分割 / 深度 / 旋转框 / 姿态（s/m/l/x，旋转框 s/x）；官方 EMA 权重直载，
  各任务加载与前向数值对拍（检测解码器谱系 bit-equal）。
- **姿态估计任务面**：COCO 关键点数据面、OKS 指标与官方损失（matcher / 损失 /
  CDN 与官方实现逐位一致）。
- **自监督预训练与蒸馏**：DistillationV3、SimCLR、DINO、DINOv2、DINOv3、
  DenseCL、DetCon B/S、训练中 kNN 探针。
- **运行时**：DDP（预训练 + 任务训练双引擎）、混合精度、原子 checkpoint +
  resume、自描述 checkpoint 格式、NaN 捕获、JSONL / TensorBoard / MLflow / W&B
  记录器、OTel（opt-in，默认关闭）。
- **导出与部署**：ONNX（fp32 / fp16，默认数值对拍门禁）、TensorRT（batch-only
  profile）、int8 QDQ 量化链、`export` CLI、benchmark 三后端（torch / ONNX /
  TensorRT）parity 门禁。
- **参考推理运行时** `autovision.inference`：视频文件 / RTSP / HTTP 流 + batch
  聚合、torch / ONNX / TensorRT 后端、与评测同源的前后处理、
  `predict --source` 输出逐帧 JSONL + 渲染视频。
- **数据层**：11 类数据集、检测增广全套（mosaic / mixup / copyblend /
  box-aware eraser，逐 step 窗口调度）、mmap 文件清单（内容指纹 / 原子发布 /
  碰撞守卫）。
- **权重供应链**：31 个 HuggingFace 资产（revision + sha256 钉住 + 许可门控）、
  GTR 26 个官方权重（MIT）、用户自持资产库（仓库外，不可再分发，见模型库页面）。
- **文档**：公开文档站点与模型库页面（每模型许可 / sha256 / 官方指标 / 加载
  代码）、协议化基准表 `docs/benchmarks.md`。


### 变更

- 版本号对齐 `0.2` 线：`0.1.0` → `0.2.0`；打包版本改为单一源
  （`src/autovision/_version.py`，pyproject 动态读取）。
- 文档树重构为站点导航结构（as-built 架构总览 + 模型库 + 基准 + 贡献 + 发布
  流程）；历史设计分册不再随公开树分发。

### 安全与合规

- 溯源与分层门禁（`scripts/check_provenance.py` / `scripts/check_layering.py`）
  在 CI 强制。
- 受限许可权重默认不下载（DINOv2 CC-BY-NC 4.0 / DINOv3 License 需显式自证
  接受条款后启用）。
- AGPL 组件（`ultralytics`）仅允许 extras 隔离加载：不进核心依赖、不在模块级
  import。

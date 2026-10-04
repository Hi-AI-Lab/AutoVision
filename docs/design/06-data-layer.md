# VTrain 实现蓝图 · 分册 06：数据层与数据治理

| 文档属性 | 内容 |
|---|---|
| 版本 | v3.1（实现蓝图拆分版，内容与总览同步维护） |
| 总览 | [framework_design.md](../framework_design.md) |
| 全局章节 | §9 |

> **标注约定**：【基线】lightly-train 0.17.0 已验证事实（出处 `文件:行`，路径相对 `src/lightly_train/`）；【设计】VTrain 决策；【差异】有意偏离基线的决策（附 ADR 编号）。正文中的 § 编号为**全局章节编号**，章节与分册的映射见总览。

---

## 9. 数据层与数据治理

### 9.1 格式与统一标注表示【基线】

- **内部表示全局唯一**：`bboxes (N,4)` = **归一化 cxcywh**（与 albumentations `BboxParams(format="yolo")` 对齐），`classes (N,)` int64，类别表为有序 dict；COCO 解析时即由 `[x,y,w,h]` 像素除以图宽高完成归一化（`_data/coco_object_detection_dataset.py:186-195`）。
- 行清单 = JSON 可序列化 dict（`{"image_path", "bboxes", "class_labels"}`，bbox/labels 为 JSON 字符串列），生成器产出、mmap 落盘，**解析发生在列清单阶段一次**而非每 epoch。
- 类别解析：COCO 以 **train 注释文件的 `categories`** 为准；YOLO 要求显式 `names: dict[int,str]`；**没有 "auto" 类别**。忽略类经 `get_class_id_to_internal_class_id_mapping` 重映射为连续内部 id，模型内持 `internal_class_to_class` buffer 映射回用户 id。
- mtime 进指纹：COCO 的 mmap 哈希含注解文件 `st_mtime`（编辑即失效缓存）；YOLO 哈希只有路径+配置（无 mtime）——VTrain 统一为「内容哈希 + mtime」。
- 图像解码快路径：jpg/png → `torchvision.decode_image`，失败（截断图）回退 PIL（`ImageFile.LOAD_TRUNCATED_IMAGES=True`）；`.dcm` 走 pydicom extras；非 uint8 按原 dtype 极值缩放为 float32 [0,1]。
- 划分：train/val 必须显式提供，**无自动划分**。

【差异】VTrain 保留统一表示；新增 `classes: auto`（从 train 注解推断并写入 resolved 配置）、`tiles` 切片训练格式（预处理命令生成带重叠 tile 数据集，中心点归属去重；基线的 tiling 仅存在于推理侧 SAHI）。

### 9.2 高性能数据路径：mmap 文件清单【基线】（VTrain 整方案复用）

```
写入  PyArrow IPC new_file 流式写，chunk=10000 行/表；schema 由首行推断(int64/float64/bool/string)
      (_data/_serialize/memory_mapped_sequence.py:34-48, 161-207)
读取  MemoryMappedSequence: pa.memory_map + open_file().read_all()；table 按 os.getpid() 懒加载缓存
      （fork/spawn 均安全：pickle 只携带 {path, columns}，数据零拷贝共享）；切片走 arrow slice/take
      (:51-158)
编排  文件名 = sha256(标识) → <DATA_CACHE_DIR>/<hash>.mmap；标识默认 (out_dir, data_hash)，
      LIGHTLY_TRAIN_MMAP_REUSE_FILE=1 时仅 data_hash（跨 run 复用）
      rank0(local, 非共享盘时每节点) 写 .<out_hash>.temp 后原子 rename；其余 rank 0.2s 轮询等待，
      LIGHTLY_TRAIN_MMAP_TIMEOUT_SEC=300(-1 关闭)；rank0 写期间重置超时计时
      引用计数 <hash>.ref_count + FileLock(<hash>.lock, timeout=300)；归零且非复用模式则删除
      pretrain 侧: 非 resume/overwrite 时 mmap 文件已存在 ⇒ RuntimeError("multiple runs using
      output directory ...") —— 多 run 撞目录的防护（common_helpers.py:439-479）
```

【差异】VTrain 增补：fsspec 远程数据集按需拉取 + 本地二级缓存（`AUTOVISION_DATA_CACHE_DIR` 语义扩展）；清单内容参与数据指纹。

### 9.3 批组装与 dataloader【基线】

- 检测 collate **不做 padding**：`ObjectDetectionBatch = {image_path: list[str], image: (B,3,H,W), bboxes: list[(n,4)], classes: list[(n,)], original_size: list[(w,h)]}`（`types.py:151-157`）；组装顺序：mixup（≥2 样本）**elif** copyblend → BatchReplayCompose(scale_jitter，**同 batch 同尺寸**使 stack 合法) → ToTensorV2 → 过滤 `min_bbox_size_px=4.0` 退化框。
- 构建顺序（step 级增强的先决条件，VTrain 引擎设计必须保留）：①无 transform 建 dataset/loader（拿到 `len(train_dataloader)`）→ ②解析 transform_args（需要 total_steps/train_num_batches/grad_accum）→ ③`dataset.set_transform()` + 装 collate → ④`fabric.setup_dataloaders`。
- train loader：`shuffle=True, drop_last=True`；val：`shuffle=False, drop_last=False`；batch 语义 = **全局 batch**，per-loader = `global // world_size`；timeout 180s（workers>0 时）；`num_workers="auto"` = `SLURM_CPUS_PER_TASK-1` 否则 `cpu//每节点卡数-1`，上限 `LIGHTLY_TRAIN_MAX_NUM_WORKERS_AUTO=8`。
- step 感知 worker：`InfiniteCycleIterator` 包装 train loader；每 step 调 `transform.set_step(step)`/`collate.set_step(step)`，若活跃状态在 step 间变化（`StepActivationTracker.requires_dataloader_reinitialization()`）则 `reset()` 重建迭代器，并把 `persistent_workers` 强制为 False（`train_task_helpers.py:723-738`、`train_task.py:1735-1746`）。

### 9.4 规模分桶（auto 超参的数值来源）【基线】

`_scaling.py` 仅三件套：`IMAGENET_SIZE=1_000_000`（无大小数据集的默认）、`ScalingInfo(dataset_size, epochs)`、`interpolate()` 线性插值、`get_bucket_value()`（首阈值即用）。全部桶表：

| 消费方 | 字段 | 桶/插值 |
|---|---|---|
| Distillation v1/v3 | queue_size | §7.3 桶表 16→8192 |
| DINO | head 输出维度 | <20k→1024, <50k→2048, <100k→4096, <200k→16384, <500k→32768, →65536 |
| DINO | teacher_temp / momentum_start | 20k→1M 上 0.02→0.07 / 0.99→0.996 线性插值 |
| DenseCL | memory_bank_size | 13 桶 16→65536 |
| epochs="auto" | — | 非桶：`ceil(default_steps × batch_size / dataset_size)`，方法无 default_steps 时用固定 default_epochs（通常 100） |

【设计】VTrain 将其组件化为公共 `scaling` 模块（桶表进配置审计），新增「数据规模 → 预训练 epochs」分桶供 VTrain 自己的方法使用，每张表配对照单测。

### 9.5 检测变换与 step 调度【基线】（VTrain P0 配方基准）

**单样本管线**（`_transforms/ltdetr_transforms/utils.py:277-301` 固定顺序）：
`channel_drop → photometric_distort → random_zoom_out → random_iou_crop → hflip → vflip → rotate90 → rotate → resize → ToFloat(255) → normalize`；BboxParams `format="yolo", min_area=1.0, filter_invalid_bboxes=True, clip=True`。

**StepScheduledCompose 机制**（`ltdetr_transforms/components.py:58-231`，v2.0 所述「思路采纳；实现简化为 batch 内丢弃」的判断**需要修正**——基线方案完整可用且已解决 worker 问题，VTrain 直接移植而非简化）：

- 每个 op 的 `ActivationPolicyArgs(step_start, step_stop)` 提供 `is_active(step)`（`"auto"` 未解析 → False；越界窗口 → False）。
- 每 step 计算活跃元组 `(photometric, zoom_out, iou_crop, mosaic)`；按元组**缓存 Compose 实例**（最多 8 种组合）——调度只是换 pipeline，零运行时开销。
- collate 侧独立跟踪 `(scale_jitter, mixup, copyblend)`；活跃状态变化 → dataloader 重建（§9.3 机制）。
- **窗口解析**（`ltdetr_schedule.py:13-106`，常量 `REFERENCE_WARMUP_EPOCHS=4 / SHORT_RUN=12 / REFERENCE_TOTAL=72 / REFERENCE_NO_AUG=12`）：`no_aug = min(12, round(total_epochs×12/72))`；短跑（≤12 epoch）`start = total/3`；长跑 `start=4`；mosaic/mixup 在 `[start, flat)` 关闭（flat = start + total/2），photometric/zoomout/ioucrop/copyblend 延到 `stop = total - no_aug`；空窗口直接置 None 而非钳位。

**默认 op 参数**（`ltdetr_object_detection/transforms.py`，VTrain P0 数值基准）：

| op | 默认 |
|---|---|
| RandomPhotometricDistort | p0.5, brightness(0.875,1.125), contrast/saturation(0.5,1.5), hue(-0.05,0.05) |
| RandomZoomOut | p0.5, side_range(1.0,4.0) |
| RandomIoUCrop | p0.8, scale(0.3,1.0), trials 40/1000 |
| Flip | h0.5 / v0 |
| ScaleJitter | 30 项目标尺寸 480–800@32，**640 重复 19 次**偏置采样，`divisible_by=2×patch` |
| Mosaic | p0.5, output 320, cache 50, affine(rotate10°, trans .1, scale .5–1.5) |
| MixUp | p0.5, β~U(0.45,0.55)，框/标签拼接 |
| CopyBlend | p0.5, ≤3 物体, area≥100 |
| Resize | auto → 上取整到 2×patch 倍数（默认 640²） |

val 变换全部增强置 None，仅 Resize+Normalize。

**预训练视图变换**（`ViewTransform` 固定顺序）：channel_drop → RRC → flip → rotate → 光度(ColorJitter→Gray→Blur→Solarize) → Normalize → ToTensorV2；多视图 = 多个 ViewTransform 串联输出 list（DINO 2 全局+6 局部、DINOv2 2+8@98px、SimCLR 2 同参、蒸馏 1 视图）。batch 级参数共享用 `BatchReplayCompose`（首样本 ReplayCompose 记录、余样本 replay）。

---

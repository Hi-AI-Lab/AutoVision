# VTrain 实现蓝图 · 分册 08：模型资产管理与部署

| 文档属性 | 内容 |
|---|---|
| 版本 | v3.1（实现蓝图拆分版，内容与总览同步维护） |
| 总览 | [framework_design.md](../framework_design.md) |
| 全局章节 | §12 |

> **标注约定**：【基线】lightly-train 0.17.0 已验证事实（出处 `文件:行`，路径相对 `src/lightly_train/`）；【设计】VTrain 决策；【差异】有意偏离基线的决策（附 ADR 编号）。正文中的 § 编号为**全局章节编号**，章节与分册的映射见总览。

---

## 12. 模型资产管理与部署

### 12.1 便携模型格式【基线→设计】

【基线】exported .pt 四键（§10.5）；`load_model()` 流程：设备解析（cuda→mps→cpu）→ 本地路径或注册名下载（S3 基址 + sha256 字段仅展示）→ `torch.load(weights_only=False)` → 动态 import `model_class_path`（已有**类路径迁移表**雏形：`DINOv2LTDETRObjectDetection → LTDETRObjectDetection`，`task_model_helpers.py:52-67`）→ `model_init_args` 强制 `load_weights=False` → 构造 → `load_train_state_dict`（EMA 前缀剥离）→ eval。

【设计】VTrain 格式升级（ADR-0002）：`{schema_version, model_class_path + 迁移表, model_init_args, state_dict, normalize_args, classes, input_spec, metrics 快照, lineage{run_id, data_fingerprint, git_commit, license_info}}`；`load_model()` 只依赖该文件。

### 12.2 ONNX 导出管线【基线】（VTrain 照抄 Path A 并升格为门禁）

```
TaskModel ──► module.eval() + fp32 + deploy()(递归 convert_to_deploy)
        ──► prepare_for_onnx_export(遍历子模块调 ONNXExportConvertible 钩子)
        ──► torch.export.export(strict=True, dynamic_shapes=ModelInputSpec 声明,
                                仅 batch 维动态, 例: Dim("batch_size",1,128)×3×STATIC)
        ──► torch.onnx.export(program, dynamo=True, optimize=True, verify=True,
                              input_names=input_specs.keys(),
                              output_names=BaseModelOutput 字段名)
        ──► [fp16] onnxruntime.transformers.float16.convert_float_to_float16(
                op_block_list = ORT 默认表 + 模型策略(fp32_onnx_op_types)
                # LT-DETR 策略 = ("Softmax","MatMul") —— attention 数值热点
                ) → 去重 Cast → 去冗余 Cast → 拓扑修正
        ──► onnxslim.slim(skip_optimizations=["constant_folding"])   # 防模型膨胀
        ──► repair_value_info(删 value_info 重新形状推断; dynamo 导出器的标注 bug 规避)
        ──► write_onnx_metadata(version/license/normalize/classes/model_name)
        ──► 门禁1 verify: onnx.checker(full_check=True) +
                torch vs onnxruntime 对拍 —— 比较输出沿 dim=1 的和,
                atol=5e-3, rtol=1e-1, equal_nan=True（fp16 且无 CUDA 时跳过并告警）
```

【设计】VTrain 增加门禁2：导出模型在 val 子集上跑 EvalProtocol，mAP 差 < 0.2 才放行；FP16 算子黑名单为**每个模型配置必填项**（基线是单模型 property，VTrain 制度化）。

### 12.3 TensorRT【基线】

`export_tensorrt` 是任务模型方法（非顶层命令）：先出 ONNX → `trt.OnnxParser`（EXPLICIT_BATCH，可选 STRONGLY_TYPED）→ 输入名固定 `"images"`、**动态 H/W 报错**（仅 batch 维 profile min/opt/max）→ workspace 1GB、清除 TF32 → fp16 时附加 OBEY_PRECISION_CONSTRAINTS → `_force_fp32_for_attention_scores`（Softmax 层及喂它的 MatMul 强制 FP32）→ debug 模式可把全部中间张量标记为输出（polygraphy 排查法）。

### 12.4 推理与 SAHI【基线】

- 三段式 predict：`preprocess_image`（resize + 元数据 orig_h/w）→ `preprocess_batch`（归一化）→ `postprocess`（阈值过滤，默认 0.6；输出 `{"labels","bboxes":xyxy 绝对像素,"scores"}`）。
- SAHI：`tile_image(image, overlap, tile_size)`——步长 `max(1, int((1-overlap)·tile))`、必补末尾对齐瓦片、小图上采样或 pad；检测合并 = 全图「global tile」前置 + 瓦片框坐标偏移 + torchvision NMS(0.3) + 同标签 global/local IoU>0.1 抑制；实例分割合并 = 类感知 mask NMS(0.5)。
- `predict_semantic_segmentation` 命令：Fabric 单卡循环，逐图 `model.predict`，掩码 uint8/uint16 PNG 镜像目录结构落盘。

### 12.5 embed【基线】

单视图 Resize+Normalize（归一化参数来自 checkpoint 的 `normalize_args`）；`EmbeddingPredictor` 取 `batch["views"][0]`；`Trainer.predict(inference_mode=True)` + `EmbeddingWriter` 按 batch 累积 CPU 端；输出格式 torch（`{"filenames", "embeddings"}`）/ csv / lightly_csv（无 h5/npz）。**embed 必须能从 checkpoint 恢复完整预处理**——这印证 VTrain 把 normalize_args 等推理元数据放进 checkpoint 的设计。

### 12.6 模型注册表与 promotion【设计】

保留 v2.0 §12.2（dev→staging→prod 门禁、后端抽象、不可变版本）。基线对照物仅为 `DOWNLOADABLE_MODEL_URL_AND_HASH` + `ModelRegistry(downloadable_checkpoint=url+sha256)`。

---

# VTrain 实现蓝图 · 分册 18：RF-DETR 家族的 SSL two_crop 对比分支

| 文档属性 | 内容 |
|---|---|
| 版本 | v1.0（已实施；smoke 全路径 paired_ratio 0.8–1.0） |
| 状态 | ✅ 检测路径已实施：34 例 rfdetr 套件 + 62 例相关回归全过；真实数据 two_crop smoke 通过 |
| 来源设计 | rf-detr-ssl-v2（ssl-per-object-classification-head.md，M4 起点 unsup/two_crop/λ=0.1）；共享组件 `object_detection_components/ssl_contrastive.py`（分册 14/16 已移植） |
| 受影响模块 | `rfdetr_object_detection/{config,task_model,train_model}.py`、`_commands/train_task/object_detection.py`（two_crop 几何增广禁用）、`scripts/visdrone_rfdetr_800.py --ssl/--micro-batch` |
| 基线兼容承诺 | `ssl=None`（默认）逐比特不变；mode="off" = batch-matched 基线（双倍管线跑、无头无线）；val/export 永不触碰 |

> 标注：【设计】本文档决策；【实证】本轮调试实测。

## 1. 设计要点

1. **消费边界不变**：HF criterion 原样消费（IA-BCE+L1+IoU），SSL 损失在训练包装器外侧相加：`total = hf_loss + λ · loss_ssl`（系数只乘一次）。
2. **实例池 = 最终层 Hungarian 匹配的 group-0**：用与 HF box loss **相同的 matcher/costs/config** 重算匹配（确定性 ⇒ 零监督漂移）。Group-DETR 的辅助组匹配被过滤（参考语义 "group-0 matched pool"，池 = GT 数量级而非 13×）。
3. **头与 tap**：`SSLProjectionHead`（simcse/mlp2，dropout 双 draw）挂在 decoder `last_hidden_state`（B,300,256）上；SIGReg raw tap 经 `forward_with_raw` 同款支持。train-only，val/export 不触碰。
4. **命令层零新机制**：two_crop 数据管线按 `resolved_model_args.ssl.view` 自动接线（家族无关，已有代码）。

## 2. 实施期发现与修复（【实证】，价值最高的一条）

**症状**：真实数据 smoke 中 `ssl_paired_ratio = 0`（LT-DETR 同配方 run = 0.99），SSL 完全退化为 connected-zero。

**逐层排查**（探针链）：matcher 池 ✓ → 传输 ✓ → pairing 输入 ✓ → **`RandomIoUCrop`（p=0.8）独立裁剪两视图**：VisDrone tiny 框（<32px）对独立几何增广极端敏感——同一 GT 在两视图的 IoU 几乎全为 0（实测 iou_max 0.02–0.35，view2 匹配数 17→2），`iou_match=0.55` 下配对全灭。

**修复**（`_commands/train_task/object_detection.py`）：`ssl_two_crop` 时对训练 transform 禁用 `random_iou_crop_prob` / `random_zoom_out_prob`（photometric / flip / 擦除保留；flip 由 `flip_invariant_pairing` 恢复）。这与 LeVJEPA 的视图差异语义对齐——**InfoNCE 需要的视图差异来自框感知擦除（doc 16："被迫依赖物体证据而非背景配对"），不是几何裁剪**。

**修复后实测**：paired_ratio **0.8–1.0**（LT-DETR 同量级），loss_ssl 2.46→3.9（正常 InfoNCE 初值）。

## 3. 使用

```bash
python scripts/visdrone_rfdetr_800.py --ssl --erase --micro-batch 2   # M4 起点配方
python scripts/visdrone_rfdetr_800.py --ssl --erase --sigreg-coef 0.1 --micro-batch 2  # +SIGReg 加权
```

`--micro-batch 2`：two_crop 每 forward 2B 图，减半保显存（grad-accum 自动补回全局 batch 8）。诊断指标：`train_loss/loss_ssl`、`train_loss/ssl_pool_size`、`train_loss/ssl_paired_ratio`（健康值 ≈0.9+，持续 0 = 配对管线异常）。

**SIGReg 校准工作流**（参考实现"先只读校准定 λ_sr"）：raw projector taps 在每个活跃头训练步**常发**；`--sigreg-coef 0`（默认）时未加权统计量以 `train_loss/loss_sigreg_stat` 记录——从普通 `--ssl` run 的日志直接读统计量量级，选 λ_sr 使 `λ_sr · stat` 与 `λ_ssl · loss_ssl` 可比，再带 `--sigreg-coef <λ_sr>` 重跑（加权腿 `train_loss/loss_sigreg`）。参考的预期收益是 λ_ssl 鲁棒性而非直接 mAP（doc 16 R1-A1）。

## 4. 测试

config 校验 6 例（off-by-default / 系数门 / two_crop 门 / SIGReg 门 / params 往返）；管线 7 例（head 构建、dict 重绑定快照、ssl 损失发射与梯度、every_n_steps 门、mode=off 基线、state_dict 随行、真实 forward two_crop 端到端）。

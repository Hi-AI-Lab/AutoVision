# 规格 S-1：`sample_high_overlap_box`（PaKA 局部裁剪采样）

状态：已实现（clean-room）。行为契约由 `tests/test_methods_dinov3.py::TestConstrainedCrop`
与本规格共同锁定。本文件是实现的唯一规格来源；实现者不得参考任何第三方
同功能实现的代码文本。

## 接口

```
sample_high_overlap_box(
    parent_box: tuple[int, int, int, int],  # (x0, y0, x1, y1)，原始图像像素
    min_iou: float,                          # 保留参数：包含关系使其恒满足
    rng: random.Random,
) -> tuple[int, int, int, int]               # (x0, y0, x1, y1)
```

## 行为规格

1. 在 `parent_box` 内部采样一个轴对齐整数框，满足：
   - **包含**：`px0 <= x0 < x1 <= px1` 且 `py0 <= y0 < y1 <= py1`（包含关系
     即 PaKA 的重叠约束：局部框完全落在全局裁剪内时重叠率为 1.0）。
   - **面积**：`area / parent_area` 目标落在 `[0.05, 0.40]`；整数化与边界
     截断允许轻微偏离（验收带宽 `[0.04, 0.41]`，与既有单测一致）。
   - **长宽比** `w/h` 从 `[3/4, 4/3]` 抽取（log-uniform，与 torchvision
     RandomResizedCrop 的抽样惯例一致）。
2. 面积抽样为均匀分布；位置抽样为合法区间内的均匀整数分布。
3. 退化父框（任一边 ≤1 像素）：原样返回 `parent_box`。
4. 确定性：完全由 `rng` 决定，不得触碰全局随机源。

## 数值路径约定

目标面积 → 由 `w = sqrt(area·aspect)`、`h = sqrt(area/aspect)` 取整得到
（面积–长宽比族标准数学，torchvision BSD-3 同族）；`w/h` 截断到父框边长
并至少为 1。**已知偏差**：极端长宽比父框（如 3×1000）下，宽高各自独立
截断可使面积比降到带宽容限以下（~0.014）；真实管线中父框来自
`RandomResizedCrop`（长宽比 ∈ [3/4, 4/3]），不会触发，验收测试覆盖
长宽比 0.5/2.0 两种情形。

## 验收

- 既有 `TestConstrainedCrop` 全过（包含、面积带宽、退化框）。
- `tests/test_clean_provenance_p0.py::TestSampleBox`：1000 次采样全部落在
  约束内；两种极端长宽比父框（0.5 与 2.0）下不越界。

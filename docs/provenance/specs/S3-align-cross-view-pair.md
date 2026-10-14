# 规格 S-3：`_align_cross_view_pair`（student 局部 × teacher 全局对齐）

状态：已实现（clean-room）。规格来源：分册 14（LT-DETR SSL 对比头）与
分册 20（DINOv3 ViT masked forward）的 PaKA 机制描述 + 行为契约
（`tests/test_methods_dinov3.py`）。实现者不得参考任何第三方同功能实现
的代码文本。

## 接口

```
_align_cross_view_pair(
    student_tokens: Tensor,   # (B, C, s_h, s_w)  局部视图 patch 特征
    student_geom: Tensor,     # (B, 8)
    student_h, student_w: int,
    teacher_tokens: Tensor,   # (B, C, t_h, t_w)  全局视图 patch 特征
    teacher_geom: Tensor,     # (B, 8)
    teacher_h, teacher_w: int,
    out_h, out_w: int,        # 输出网格 = student 网格
) -> (student_aligned, teacher_aligned, valid)
    # 前两者 (B, C, out_h, out_w)；valid (B, out_h*out_w) bool
```

## 行为规格

1. **共享区域**：两视图裁剪框（原图像素）的交集
   `∩ = [max(x0), max(y0), min(x1), min(y1)]`。
2. **有效性**：交集在两个维度上都至少跨一个输出单元格
   （`交宽 ≥ 学生裁剪宽/out_w`、`交高 ≥ 学生裁剪高/out_h`）才算有效；
   无有效信号的重叠（细缝）不参与对齐损失。
3. **无效兜底**：无效样本的交集框替换为**学生视图自身裁剪框**（保证
   坐标映射恒有定义），由 `valid` 掩码在损失端剔除。
4. **对齐**：student 与 teacher 各自按规格 S-2 重采样到 `(out_h, out_w)`；
   teacher 侧在 no_grad 下执行。
5. 返回 `valid` 展开到 token 级 `(B, out_h*out_w)`。

## 验收

- 全重叠（局部 ⊂ 全局）时 valid 全真；不相交时 valid 全假且损失端正确
  跳过（既有 `TestPaika::test_invalid_pairs_skipped_with_zero_grad_path`
  覆盖损失端）。
- 细缝重叠（宽度不足一格）判无效。
- 既有方法级单测（PaKA 项进入总损失、梯度只达 student 侧）全过。

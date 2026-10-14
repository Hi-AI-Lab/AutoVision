# 规格 S-2：`_roi_align_view`（跨视图 ROI 特征重采样）

状态：已实现（clean-room）。行为契约由 `tests/test_methods_dinov3.py` 与
本规格锁定；数值由 `tests/test_clean_provenance_p0.py::TestRoiAlignReference`
用独立组合实现交叉验证。

## 接口

```
_roi_align_view(
    tokens: Tensor,        # (B, C, H, W) 特征图（B 个样本各对应一个裁剪视图）
    geom: Tensor,          # (B, 8) 视图几何 [x0, y0, x1, y1, img_w, img_h, hflip, vflip]
    region: Tensor,        # (B, 4) 共享区域 xyxy（原图像素坐标；见规格 S-3）
    in_h, in_w: int,       # tokens 的空间尺寸
    out_h, out_w: int,     # 输出网格尺寸
) -> Tensor                # (B, C, out_h, out_w)
```

## 行为规格

1. **去翻转**：视图变换可能记录了 h/v 翻转（`geom[:,6]/geom[:,7]` > 0.5）。
   重采样前先把每个样本的特征图按记录翻转还原到原始图像方向，使网格坐标
   与原图坐标系一致。逐样本选择翻转组合（4 种），不得影响未翻转样本。
   梯度必须可穿过（student 侧调用）。
2. **坐标映射**：共享区域框从原图像素坐标映射到该视图特征网格坐标：
   `gx = (ix - crop_x0) / crop_w * in_w`（y 同理），其中 crop_* 来自
   `geom[:,0:4]`，`crop_w/crop_h` 以 ≥ε 下截断防零。
3. **重采样**：映射后的 xyxy 框（特征网格坐标，连续角点语义）经双线性
   重采样到 `(out_h, out_w)`，采样点取每个输出单元格中心（恒等框 → 恒等
   重采样）；框外零填充，`align_corners=False`。实现使用本仓
   `paika._roi_resample_to_grid`（该原语独立成篇，见其自身规格）。
4. 坐标超界（数值误差）以 `[0, in_w]/[0, in_h]` 截断兜底。

## 验收

- 恒等情形：无翻转、共享区域=整幅裁剪、out=in 时输出与输入逐元素一致
  （atol 1e-5）。
- 交叉验证：与「`grid_sample` 中心采样组合」独立参照实现一致
  （float64，atol 1e-8）；含翻转/非整除网格/部分重叠三种情形。
- 既有 PaKA/方法级单测全过；梯度到达 student 特征。

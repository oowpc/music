# 工作日志：距离矩阵增量计算

**日期**: 2026-06-08

## 背景

原有流程每次点击“计算距离矩阵”都会对所有曲线执行全量两两距离计算。随着 MIDI 数量增加，新增一个文件也会触发已有曲线之间的重复计算。

增量计算的目标是：已有 N×N 矩阵时新增 K 条曲线，只计算新增曲线与全部曲线之间的距离，保留旧的 N×N 左上角矩阵。

## 关键前提

原先 `normalize_minmax()` 使用全局 min/max。新增曲线后，全局 min/max 可能变化，旧曲线归一化坐标也会变化，此时旧距离矩阵不能复用。

本次将默认 Min-Max 改为项目规范中的稳定策略：

- 时间：每条曲线减去自身起始时间，再除以自身总时长。
- 音高：除以固定 MIDI 最大值 127。
- 力度：除以固定 MIDI 最大值 127。

这样新增曲线不会改变旧曲线坐标，距离矩阵可以安全扩展。

`normalize_zscore()` 也改为每条曲线独立 Z-Score，避免新增曲线改变旧曲线标准化结果。

## 本次改动

1. 更新 `src/processing/normalization.py`。
   - Min-Max 改为曲线独立时间尺度 + 固定 MIDI pitch/velocity 尺度。
   - Z-Score 改为曲线独立统计量。

2. 新增 `src/analysis/distance_matrix.py::extend_matrix()`。
   - 输入已有矩阵、当前曲线列表、旧曲线数量和距离算法。
   - 保留旧矩阵左上角。
   - 只计算新增曲线所在行/列。
   - 支持 `standard`、`modified`、`frechet` 三种距离。

3. 更新 `src/ui/main_window.py`。
   - 记录当前矩阵对应的距离算法和归一化方法。
   - 新增 MIDI 后，如果算法/归一化未变化且已有矩阵形状匹配，则自动增量扩展矩阵。
   - 删除曲线或切换归一化时，旧矩阵失效，要求重新计算。

4. 补充测试。
   - 归一化稳定性：新增曲线后旧曲线坐标不变。
   - 矩阵扩展：旧矩阵块被保留，扩展结果与全量重算一致。
   - 主窗口增量路径：新增曲线时调用 `extend_matrix()`。

## 验证

已运行：

```bash
pytest tests/test_main_window_incremental.py tests/test_normalization.py tests/test_distance_matrix.py -v --tb=short
```

结果：

```text
16 passed
```

## 行为说明

- 新增 MIDI：如果已有矩阵有效，自动增量计算新增行/列。
- 删除 MIDI：矩阵失效，因为旧矩阵行列索引已不匹配。
- 切换归一化：矩阵失效。
- 切换距离算法：下次计算使用新算法全量计算；已有矩阵不会跨算法复用。

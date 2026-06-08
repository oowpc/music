# 工作日志：多音轨列表选择 UI

**日期**: 2026-06-08

## 背景

此前已经在 MIDI loader 层支持了多轨检查、指定轨道加载，以及 `all`、`highest`、`strongest` 三种提取方式。但 UI 侧只是简单下拉文本，不适合用户查看每条轨道的名称、乐器、音符数量和音域，也不够符合“UI 列表选择”的设计目标。

## 本次改动

1. 新增 `src/ui/track_selection_dialog.py`。
   - 使用 `QDialog + QTableWidget` 展示轨道列表。
   - 表格列包括：轨道编号、名称、乐器、音符数、音高范围、开始时间、结束时间。
   - 提供“合并所有轨道”和“使用选中轨道”两种来源选项。
   - 提取方式支持“最高音线”“最强力度线”“全部音符”。
   - 默认策略为“合并所有轨道 + 最高音线”。

2. 更新 `src/ui/file_panel.py`。
   - 多轨 MIDI 导入时调用 `TrackSelectionDialog`。
   - 用户确认后将 `(track_index, extraction_mode)` 传给 `load_midi()`。
   - 单轨 MIDI 仍默认使用最高音线，减少无意义弹窗。

3. 新增 `tests/test_track_selection_dialog.py`。
   - 验证默认选择返回 `(None, "highest")`。
   - 验证选中轨道并切换模式后返回指定轨道和提取方式。

4. 更新 README 核心功能描述。

## 当前交互流程

导入多轨 MIDI 时：

1. 解析所有轨道元数据。
2. 弹出轨道选择对话框。
3. 用户选择合并所有轨道或某个轨道。
4. 用户选择提取方式。
5. loader 只提取对应音符并生成 `MelodyCurve`。

## 验证

已运行：

```bash
pytest tests/test_track_selection_dialog.py -v --tb=short
```

结果：

```text
2 passed
```

主窗口 offscreen 创建检查通过：

```text
MainWindow created OK
```

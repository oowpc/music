# 工作日志：MIDI 主旋律提取改进

**日期**: 2026-06-08

## 背景

在测试 `C:\Users\kcl\Downloads\夜曲-周杰伦.mid` 时，原始 loader 会把所有轨道、和弦、伴奏、低音一起提取为三维曲线点。该文件解析得到 2152 个音符事件，其中包含 bass、guitar、ahhs、drums 等轨道。这会让 Hausdorff 距离更多反映编曲/伴奏结构，而不是主旋律相似性。

## 本次改动

1. 在 `src/io/midi_loader.py` 增加 `MidiTrackInfo` 和 `inspect_midi_tracks(filepath)`。
   - 返回每个 MIDI 轨道的名称、乐器名、音符数、音高范围、起止时间。
   - 非 MIDI 或解析失败时返回空列表。

2. 扩展 `load_midi()` 参数。
   - `track_index`: 可指定只加载某一个轨道。
   - `extraction_mode`: 支持 `all`、`highest`、`strongest`。
   - `all`: 保留所有音符，兼容旧行为。
   - `highest`: 同一时间点只保留最高音，更适合粗略主旋律线。
   - `strongest`: 同一时间点只保留力度最大的音。

3. 更新 UI 导入逻辑。
   - 多轨 MIDI 导入时弹出选择框。
   - 默认选项为“全部轨道 - 最高音线（推荐）”。
   - 也可选择“全部轨道 - 全部音符”“全部轨道 - 最强力度线”，或单独选择某个轨道的最高音线。

4. 补充测试。
   - 多轨 MIDI 轨道摘要。
   - 指定轨道加载。
   - 同时音最高音过滤。
   - 最强力度过滤规则。
   - 无效轨道和无效过滤模式处理。

## 夜曲 MIDI 检查结果

文件：`C:\Users\kcl\Downloads\夜曲-周杰伦.mid`

识别到 9 个轨道：

| index | name | instrument | notes | pitch range | time range |
|---:|---|---|---:|---|---|
| 0 | vocal | Piano | 104 | 55-79 | 23.93s-215.56s |
| 1 | subvocal | Recorder | 297 | 65-84 | 40.62s-216.07s |
| 2 | guitar | guitar | 181 | 53-77 | 2.36s-215.73s |
| 3 | bass | bass | 270 | 32-41 | 12.81s-218.43s |
| 4 | special | Sampler | 76 | 72-80 | 66.40s-194.16s |
| 5 | ahhs | Voice | 310 | 58-72 | 13.48s-215.73s |
| 6 | guitar 2 | guitar 2 | 902 | 34-68 | 2.70s-216.57s |
| 7 | rings | Electric Guitar | 12 | 63-68 | 109.89s-131.80s |
| 8 | drums | Percussion | 0 | - | - |

提取结果：

- `all`: 2152 个音符点。
- `highest`: 890 个旋律点。
- `strongest`: 890 个旋律点。

## 验证

已运行：

```bash
pytest tests/test_midi_loader.py -v --tb=short
```

结果：

```text
11 passed
```

## 后续建议

1. 在 UI 中增加轨道详情预览表，比当前弹窗更适合多文件批量处理。
2. 增加“自动推荐主旋律轨道”规则，例如优先选择 name 含 vocal/melody、音域较高、音符数适中的轨道。
3. 给距离计算增加维度权重，默认提高 pitch 权重、降低 velocity 权重。
4. 支持片段截取，比如只比较副歌或指定时间窗口。

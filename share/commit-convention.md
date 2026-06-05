# Commit Message 规范

## 格式

```
<type>(<scope>): <subject>

[body]
```

## Type（必填）

| type | 说明 |
|------|------|
| `feat` | 新功能 |
| `fix` | 修复 bug |
| `refactor` | 重构代码（不改变功能） |
| `perf` | 性能优化 |
| `style` | 代码格式（空格、分号等，不影响逻辑） |
| `test` | 添加或修改测试 |
| `docs` | 文档变更 |
| `chore` | 构建、依赖、辅助工具等杂项 |
| `revert` | 回滚之前的 commit |

## Scope（可选）

指明影响的模块，例如：`midi_loader`, `hausdorff`, `ui`, `normalization`, `clustering`

## Subject（必填）

- 使用中文或英文，保持一致
- 不超过 50 个字符
- 使用祈使语气（"添加" 而非 "添加了"）
- 结尾不加句号

## Body（可选）

- 描述变更原因和细节
- 每行不超过 72 个字符
- 与 subject 之间空一行

## 示例

```
feat(midi_loader): 支持批量导入 MIDI 文件

新增多文件选择对话框，自动解析为 MelodyCurve 列表。
```

```
fix(hausdorff): 修复空曲线导致的除零错误
```

```
docs: 添加 commit message 规范
```

```
test(normalization): 添加归一化边界值测试
```

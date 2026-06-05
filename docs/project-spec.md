# 项目文件结构与代码框架规范

## 一、顶级目录与文件要求

1. 必须包含以下顶级文件：
   - `.gitignore`
   - `README.md`
   - `requirements.txt`（或 `environment.yml`）
   - `LICENSE`

2. 必须包含以下六个一级目录：
   - `docs/`
   - `ai_conversations/`
   - `share/`
   - `src/`
   - `tests/`
   - `report/`

## 二、docs/ 目录规范

3. 用途：存放正式的、稳定的项目计划、设计文档、过程记录。

## 三、ai_conversations/ 目录规范

4. 用途：导出备份与 AI 助手的完整对话记录（JSON 格式）。

## 四、share/ 目录规范

5. 用途：存放不正式的、需要临时写入以便后续引用或留档的文档。
6. 允许任何文本或 Markdown 文件，禁止提交二进制大文件。
7. 清理规则：每个合作者可自由编辑，版本合并前自行解决冲突。

## 五、data/ 目录规范

8. 该目录必须被 `.gitignore` 完全忽略，禁止任何数据文件提交至 Git 仓库。

## 六、src/ 目录规范

9. 所有 Python 源代码必须放在 `src/` 下，按功能模块拆分文件。
10. 辅助函数统一放在 `src/utils/` 下。
11. 每个函数必须包含 docstring，说明输入输出形状与含义。
12. 禁止在 `src/` 目录下存放数据、结果、配置密文或未使用的旧代码。

## 七、tests/ 目录规范

13. 所有单元测试文件放在 `tests/` 下，文件名以 `test_` 开头。
14. AI agent 临时运行代码写入的临时文件也放此目录，同步前需清除。

## 八、report/ 目录规范

15. 该目录仅留空，用于存放最终的毕设报告及相关附件、图片。
16. 合作者不得在 `report/` 内创建子目录，除非经所有人同意。
17. 提交报告前，该目录应为空或仅包含一个 `.gitkeep` 文件。

## 九、results/ 目录

18. 可用于保存少量总结性结果（`.csv`、`.png`、`.txt` 日志摘要）。
19. 大量处理后的数据（超过 5 MB）必须放在 `data/` 子文件夹中。

## 十、.gitignore 必须包含的条目

```gitignore
# Python
__pycache__/
*.py[cod]
*.so
.Python
venv/
.venv/
env/
ENV/

# 数据目录（完全忽略）
data/

# 临时文件
*.log
*.tmp
*.aux
*.out

# IDE
.vscode/
.idea/
*.swp
.DS_Store

# Jupyter
.ipynb_checkpoints/

# 配置文件（本地）
*.local.cfg
.env
```

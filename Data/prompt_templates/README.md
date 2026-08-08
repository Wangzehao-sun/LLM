# `Data/prompt_templates/` — Prompt 模板的唯一事实源

## 这是什么

rephrase / teacher prompt 的模板全部存放在这里，一个模板一个 `.txt` 文件，用**名字**寻址。

需要 prompt 的程序都从这里加载，不再各自持有副本。

## 为什么需要它

整条链路（构建 SFT 数据 → SFT 训练 → 评测）的前提是每一跳 prompt 完全一致，否则测出的准确率变化混杂了 prompt 漂移而非模型能力。之前这个前提是破的：

- `DEFAULT_TEMPLATE` 在主仓库和 Inferapi 各有一份，**同名但内容完全不同**（相似度 0.199：一份写 `## Reference Reasoning Draft:`，另一份写 `## Partial Reasoning Draft:`）。已有的 512 行 SFT 数据用的是 Inferapi 那份，而它在主仓库根本不存在。
- 一处模板把 `\boxed{}` 误写成 `\boxed{{}}`，而该模板走替换渲染、不像 `str.format` 会折叠双括号 —— 双括号原样发给了 API，16 条调用记录全部中招。

现在模板只有一个来源，每份有唯一名字，改动有 git diff 可查。

## 现有模板

| 名字 | 字符数 | 含 `{style_example_1}` |
|---|---|---|
| `rephrase_main_v1` | 1959 | 否 |
| `rephrase_main_v2` | 1960 | 否 |
| `rephrase_inferapi_v1` | 1346 | 否 |
| `rephrase_shared_gold_v1` | 1386 | 否 |
| `teacher_continue_v1` | 1657 | 是 |
| `teacher_repair_v1` | 2962 | 是 |
| `teacher_continue_v1_boxedbug` ⚠️ 已弃用 | 1659 | 是 |

来源与区别写在 `registry.py` 每条的 `note` 里。几点值得知道：

- **`rephrase_inferapi_v1`** 渲染了已有的 512 行 SFT 数据。
- **`rephrase_main_v1` 和 `rephrase_inferapi_v1` 曾经同名**，就是它们的冲突促成了这个包。
- **`teacher_continue_v1_boxedbug`** 保留了带 bug 的字节，仅用于复现修复前那 16 行批次，不要用于新数据。默认列表里不显示。

## 用法

```python
import prompt_templates as pt

# 1) 解析模板：接受名字，也接受 .txt 文件路径
text, name = pt.resolve("teacher_continue_v1")

# 2) 构造 messages（三个消费方共同的起点）
messages = pt.build_messages(system_msg, question, draft, text)
#   -> [{"role": "system", ...}, {"role": "user", ...}]

# 3) 把模板名记进产出物，供下游核对
meta = pt.provenance(name)     # {"prompt_id": ..., "rendered_at": ...}
```

三个消费方**只在 `build_messages` 之后**分叉：

| 场景 | 之后做什么 |
|---|---|
| 调 API | 直接把 `messages` 发出去 |
| SFT | 追加 `{"role": "assistant", "content": target}` |
| 评测 | 追加 generation prompt 后生成 |

### 完整 API

| 函数 | 作用 |
|---|---|
| `load(name)` | 返回模板文本 |
| `resolve(spec)` | `spec` 为名字或文件路径 → `(text, name)` |
| `render(text, question, prefix, style_examples=None)` | 填充占位符。**唯一的渲染路径** |
| `build_messages(system_msg, question, prefix, text, ...)` | 构造 `[system?, user]` |
| `provenance(name)` | `{prompt_id, rendered_at}`，跟产出物一起存 |
| `template_names(include_deprecated=False)` | 已注册模板名 |

### 占位符

只有三种，其余内容（含字面 `\boxed{}`）原样通过：

| 占位符 | 何时填 |
|---|---|
| `{question}` | 离线渲染时 |
| `{prefix}` | 离线渲染时（expert reasoning 草稿） |
| `{style_example_N}` | **不传就保持原样** —— RL trainer 在线用学生同题的错误 rollout 填入 |

## 从 `Data/` 下的脚本导入

`Data/prompt_templates/` 是 `Data/` 内部的包目录，同目录脚本直接 import，无需改 `sys.path`：

```python
import prompt_templates as pt
```

从仓库根运行时（`python Data/xxx.py`），`sys.path[0]` 即 `Data/`，与该目录下脚本互相 import 的方式一致。

外部目录（如 Inferapi）需要指定路径：

```python
import sys
sys.path.insert(0, "/path/to/LLM/Data")
import prompt_templates as pt
```

加载器是**纯 stdlib**（`pathlib`/`re`/`datetime`），不依赖 pandas/numpy/yaml，所以可以被外部仓库 vendor 后独立使用。

## 新增或修改模板

**已经用于生成数据的模板，不要原地改。** 改了之后，同一个名字在不同时间指向不同内容，下游就无法判断某批数据到底用的是哪个 prompt。要改就加新版本：

1. 写 `rephrase_main_v3.txt`（结尾恰好一个换行）
2. 在 `registry.py` 的 `TEMPLATES` 里加一条，填 `note` 说明来源与区别
3. 跑测试：
   ```bash
   python3 -m pytest Myverl/tests/custom/test_prompt_templates_on_cpu.py -q
   ```

命名规则 `<role>_<lineage>_<version>`。**`default` 这个名字被禁用**（import 时检查）—— 正是它让两个不同模板藏在同一个标识符后面。

## 为什么是 `.txt` 而不是 YAML / Python 字符串

`\boxed{{}}` 那个 bug 本身就是 Python 字符串转义的产物。YAML 会重新引入一层转义（多行散文需块标量 `|`，缩进敏感，一次误缩进就静默改变字节）。`.txt` 没有任何转义层 —— `\boxed{}` 就是这几个字符 —— git 能逐行 diff，`Path.read_text()` 零依赖。

已验证：`ruff format` 会重排本目录的 `.py` 文件，但**碰不到 `.txt`**。

## 单一渲染路径

模板以**字面形式**存储（`\boxed{}`，不是 `\boxed{{}}`），用一条 `re.sub` 渲染。原先有两条路径（`str.format` 和 `str.replace`），只因为 `str.format` 处理不了字面 `{}`。

实测这条路径与 `str.format` **逐字相同**，且对已有 512 行 SFT parquet 逐行反解重渲染 **512/512 一致**。所以 `is_teacher` 那个开关被删掉了 —— 一条路径不会和自己不一致。

## 范围

本包只管**模板文本**。

输出侧的泄漏过滤词表**不在这里** —— RL trainer 过滤的是策略模型在 rephrase prompt 下的 rollout，离线后处理过滤的是强模型在 teacher prompt 下的输出，两者筛的东西不同，本就该各自特化，各自留在消费方。

## 测试

```bash
python3 -m pytest Myverl/tests/custom/test_prompt_templates_on_cpu.py -v
```

31 个测试。核心几条：单一渲染路径与 `str.format` 逐字一致、512 行 parquet 重渲染一致、名字不重复、文件换行约定、花括号回归（修好的模板含 `\boxed{}`，`_boxedbug` 变体保留 `\boxed{{}}`）。

涉及仓库外路径（Inferapi、桌面上的 parquet）的测试在文件缺失时 skip，裸 checkout 也能全绿。

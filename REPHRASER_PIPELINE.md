# Rephraser 全链路使用说明

从构建 SFT 数据到训练 rephraser 到评测的完整流程。

**核心约束**：每一跳的 prompt 必须来自同一个模板名。否则测出的准确率变化分不清是模型变了还是 prompt 变了。模板统一在 `Data/prompt_templates/`（见该目录的 `README.md`），Inferapi vendor 了一份副本。

---

## 涉及的两个仓库

| 仓库 | 位置 | 职责 |
|---|---|---|
| `LLM` | `~/LLM` | 模板事实源、SFT 数据构建、训练、评测 |
| `Inferapi` | `~/Desktop/Inferapi` | 调强模型 API 生成 rephrase 结果 |

Inferapi 是独立 git 仓库（`Wangzehao-sun/Inferapi`），跑在可能没有 `LLM` checkout 的机器上，所以它 vendor 模板而非 import。

---

## 两条构建 SFT 数据的路线

先决定用哪条 —— 它们的 SFT target 来源不同，训出来的 rephraser 行为也不同。

| | 路线 A：强模型蒸馏 | 路线 B：学生自身正确 rollout |
|---|---|---|
| SFT target | 强模型（如 `deepseek-v4-flash`）的输出 | 学生模型自己答对的 rollout |
| 工具 | Inferapi + `Data/prepare_sft.py` | `Data/prepare_rephraser_sft.py` |
| 数据量 | 大（每题都能产出） | 小（只有答对的题有） |
| 用意 | rephraser 学会写正确解答 | rephraser 贴近学生分布，降低 IS ratio 方差 |
| 已有产物 | 512 行（`rephrase_inferapi_v1`） | 92 行（`rephrase_main_v2`） |

两条都走同一个模板库，可以对比或混合。

---

## 路线 A：强模型蒸馏

### A1. 挂上学生 rollout（仅 teacher 模板需要）

teacher 模板的 `{style_example_1}` 要填一条学生同题的**错误** rollout，所以先把 rollout dump 挂到 parquet 上。用非 teacher 模板（如 `rephrase_inferapi_v1`）可跳过这步。

```bash
cd ~/Desktop/Inferapi/rephrase_rollout

python3 attach_rollout_responses.py \
  --input-parquet input/deepmath_hard_solonly.parquet \
  --rollout-jsonl input/80.jsonl \
  --output        input/deepmath_hard_solonly_rollout80.parquet
```

默认丢弃没有错误 rollout 的题（teacher 模板需要那条错误回答）；要保留加 `--keep-without-incorrect`。

### A2. 渲染 prompt

```bash
# teacher 路线（续写学生的部分推理）
python3 prepare_summarize_prompts.py \
  --input  input/deepmath_hard_solonly_rollout80.parquet \
  --output input/deepmath_hard_rollout80_teacher.parquet \
  --teacher-only \
  --teacher-template teacher_continue_v1 \
  --teacher-ratio 0.5 --teacher-attempt-ratio 0.5 \
  --workers 16

# 或：summarize 路线（复述草稿再续写，不需要错误 rollout）
python3 prepare_summarize_prompts.py \
  --input  input/deepmath_hard_thinkonly1024.parquet \
  --output input/deepmath_hard_thinkonly1024_summarize0_5.parquet \
  --mode full --full-ratio 0.5 \
  --template rephrase_inferapi_v1 \
  --workers 16
```

会打印用了哪个模板，**记下这个名字**，后面每一步都要用同一个：

```text
[template] summarize: rephrase_inferapi_v1 (1346 chars)
[template] teacher: teacher_continue_v1 (1657 chars)
```

产出物带 `prompt_id` 列（teacher 模式还有 `teacher_prompt_id`）。

不传 `--tokenizer-path` 时按**字符**比例截断，无需装 transformers；传了则按 token 截断。

### A3. 调强模型 API

```bash
# run.sh 默认读 teacher_prompts 字段
DASHSCOPE_API_KEY="sk-..." ./run.sh input/deepmath_hard_rollout80_teacher.parquet
```

**summarize 路线要先改 `run.sh` 里的 `INPUT_FIELD`** —— 它是写死的（`run.sh:47`），不是环境变量：

```bash
sed -i '' 's/^INPUT_FIELD="teacher_prompts"/INPUT_FIELD="summarize_prompts"/' run.sh
DASHSCOPE_API_KEY="sk-..." ./run.sh input/deepmath_hard_thinkonly1024_summarize0_5.parquet
```

常用环境变量：`MODEL`（默认 `deepseek-v4-flash`）、`THREADS`（100）、`RETRY`、`RESUME`、`MAX_TOKENS`、`TEMPERATURE`。本地 vLLM 端点用 `API_URL="" ./run.sh`。

结果追加到 `datav1/output/<输入名>_success.json`（后缀是 `.json`，内容是 JSONL）。`RESUME=true` 会跳过已完成的样本。失败记录在 `datav1/error_out/`。

### A4. 过滤

```bash
python3 postprocess_outputs.py \
  --input  datav1/output/deepmath_hard_rollout80_teacher_success.json \
  --output datav1/output/deepmath_hard_rollout80_teacher_filtered.parquet \
  --output-format parquet \
  --filters correct format
```

- `correct` —— 用 `math_verify` 比对 `reward_model.ground_truth`
- `format` —— 拦截元叙述泄漏（提到 draft/expert guidance、复述模板指令）、长重复字符和重复子串

只看统计不写文件加 `--no-write`。

### A5. 转成 SFT 格式

过滤后的 parquet **还没有** `messages` 列，需要这一步补上：

```bash
cd ~/LLM

python3 Data/prepare_sft.py \
  --in  ~/Desktop/Inferapi/rephrase_rollout/datav1/output/deepmath_hard_rollout80_teacher_filtered.parquet \
  --out Data/rephraser_sft_teacher.parquet \
  --prompt-key teacher_prompts \
  --target-key output
```

`--prompt-key` 要和 A2/A3 用的字段一致（teacher 路线是 `teacher_prompts`，summarize 路线是 `summarize_prompts`）。产出 `messages` = `[system, user, assistant]`，正是 `MultiTurnSFTDataset` 消费的格式。

---

## 路线 B：学生自身正确 rollout

一步到位，从 rollout dump 直接构建：

```bash
cd ~/LLM

python3 Data/prepare_rephraser_sft.py \
  --rollout ~/Desktop/rollout_data/80.jsonl \
  --parquet Data/deepmath_hard_solonly_split_summarize_teacher.parquet \
  --output  Data/rephraser_sft_student.parquet \
  --template rephrase_main_v2 \
  --draft-ratio 0.5 --pick median --per-question 1
```

按 question 文本把 dump 和 parquet join 起来（dump 没有 uid），取 expert reasoning 前 50% 作 draft 渲染 prompt，配上该题的正确 rollout 作 target。直接产出 `messages` 列，无需 A5 那步。

常用参数：

| 参数 | 说明 |
|---|---|
| `--template` | 模板名或 `.txt` 路径。默认 `rephrase_main_v2` |
| `--draft-ratio` | expert reasoning 切多少作 draft。默认 0.5 |
| `--pick` | 每题选哪条正确 rollout：`median`（默认）/ `shortest` / `longest` / `all` |
| `--per-question` | 一题出几行。默认 1 |
| `--rollout` | 可传多个 dump 扩量（桌面上 8 个 dump 两两不重叠） |
| `--dry-run` | 只看统计不写文件 |

默认会丢弃没有 `\boxed{}` 的「正确」rollout —— 那些是被 response-length cap 截断在推导中途、只是碰巧判对的（实测 449 条里有 2 条）。要保留加 `--allow-unboxed`。

---

## 训练

```bash
GPU_DEVICES=4,5,6,7 \
TRAIN_PATH=$HOME/LLM/Data/rephraser_sft_student.parquet \
TRAIN_BSZ=16 LR=1e-5 EPOCHS=3 TEST_FREQ=5 \
bash Myverl/examples/custom/train_sft.sh Qwen3-4b-base
```

跑之前的 preflight 会打印行数和步数，`messages` 列缺失或步数为 0 直接退出。

要注意的几点：

- **`TRAIN_BSZ` 必须远小于行数。** 两个 dataloader 都是 `drop_last=True`，步数 = `floor(行数 / TRAIN_BSZ)`。92 行配 `TRAIN_BSZ=64` 只有 1 步/epoch，配 128 则**一步都不训且不报错**。92 行建议 16。
- **`LR` 默认 1e-6 偏小**（继承自 RL 脚本）。纯 SFT 建议先试 `1e-5`（stock verl 的默认）。
- **`TEST_FREQ` 只算 teacher-forcing 的 `val/loss`**，这条链路没有生成能力，测不了 rollout 准确率 —— 那要靠下面的离线评测。小数据上默认 200 永远触发不到，想看设 5。
- 每个 epoch 存一个 ckpt（`save_per_epoch=True`），落在 `$LOG_DIR/ckpt/global_step_*`，标准 HF 格式，vLLM 能直接加载。

其他可调：`MICRO_BSZ`、`MAX_LENGTH`（16384）、`SAVE_FREQ`、`WORKER_DIR`/`MODEL_DIR`/`CODE_DIR`/`LOG_ROOT`。Hydra 参数可直接追加在命令末尾。

---

## 评测

`fsdp_sft_trainer` 没有生成能力，所以准确率靠离线评测每个 ckpt。

### E1. 用同一个模板渲染验证集

```bash
python3 Data/prepare_summarize_prompts.py \
  --input  <你的验证集>.parquet \
  --output Data/eval_rephrase.parquet \
  --template rephrase_main_v2 \
  --tokenizer-path /home/shared/Qwen3-4b-base \
  --single-mode full --full-ratio 0.5 \
  --list-mode custom --list-ratios 0.5
```

注意这是 **LLM 仓库版**的脚本，参数是 `--single-mode` / `--list-mode`，不是 Inferapi 版的 `--mode`。

**模板名必须和训练数据的 `prompt_id` 一致。** 核对一下：

```bash
python3 -c "
import pandas as pd
print('train:', pd.read_parquet('Data/rephraser_sft_student.parquet')['prompt_id'].unique())
print('eval :', pd.read_parquet('Data/eval_rephrase.parquet')['prompt_id'].unique())"
```

### E2. 逐 ckpt 生成 + 打分

```bash
DATA_PATH=$HOME/LLM/Data/eval_rephrase.parquet \
N_SAMPLES=4 MAX_STEPS=100 \
bash Myverl/examples/custom/generation_deepmath.sh <ckpt 目录名>
```

vLLM 采样 `N_SAMPLES` 条 → `math_verify` 打分（`reward_impl_version=4`，no-think）→ 写出带 `test_score` 列的 parquet（含 `scores_per_response` / `mean_score` / `max_score`），并打印每批和总的平均分。

`generation_deepmath.sh` 读的是 `data.prompt_key=prompt`，而上一步渲染出的 rephrase prompt 在 `summarize_prompts` 列（长度 K 的数组，每个元素是一份 messages）。所以要先把想评的那一份摊到 `prompt` 上：

```bash
python3 -c "
import pandas as pd
d = pd.read_parquet('Data/eval_rephrase.parquet')
d['prompt'] = d['summarize_prompts'].map(lambda a: a[0])   # 取第一份（--list-ratios 只给了一个）
d.to_parquet('Data/eval_rephrase_flat.parquet', index=False)
print(len(d), 'rows; prompt roles:', [m['role'] for m in d.iloc[0]['prompt']])"
```

然后拿 `eval_rephrase_flat.parquet` 去跑评测。

### 建议对比的基线

- 未训练的 instruct 模型
- SFT 后的 rephraser
- 生成数据的那个强模型

三者用同一份验证集、同一个模板、同一套打分。

---

## 模板对齐的检查点

| 环节 | 怎么保证一致 |
|---|---|
| 渲染 prompt | `--template <名字>`，脚本打印实际用的名字 |
| 产出物 | 自动写 `prompt_id` 列 |
| 转 SFT | `Data/prepare_sft.py` 保留源 parquet 全部列，`prompt_id` 跟着走 |
| 训练 | preflight 打印，可人工核对 |
| 评测 | 用同一个模板名渲染，比对 `prompt_id` |

Inferapi 的模板副本要和上游同步：

```bash
diff -r ~/LLM/Data/prompt_templates ~/Desktop/Inferapi/rephrase_rollout/prompt_templates
```

有差异就从 `~/LLM` 重新 copy。**只在 `~/LLM` 改模板**，否则两边对同一个名字的理解会分叉 —— 这正是当初 `DEFAULT_TEMPLATE` 同名不同文（相似度 0.199）的成因。

---

## 已知的坑

**已有的两份数据模板不同。** 512 行那份用 `rephrase_inferapi_v1`，92 行那份用 `rephrase_main_v2`。要混用或对比，先想清楚这个差异。

**teacher 模板修过一个 bug。** 修复前 `\boxed{}` 误写成 `\boxed{{}}`，而该模板走替换渲染、不折叠双括号，双括号原样发给了 API（影响 16 条调用记录）。现已修正，所以修复前后生成的 teacher 数据 prompt 不逐字可比。

**`Data/*.parquet` 被 gitignore**，不会进版本库，靠脚本重新生成。

**`prepare_summarize_prompts.py` 有两个不同的模式参数**：`--mode`（Inferapi 版，`full`/`multi`）和 `--single-mode`/`--list-mode`（LLM 版）。两个仓库的这个脚本是分开演进的，参数不完全一样，照各自的 `--help` 来。

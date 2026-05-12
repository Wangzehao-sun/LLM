# LLM Codebase 说明

本仓库基于 `verl` 进行改装。相比原始 `verl`，当前主要新增和修改集中在两处：

- `Data/`：训练数据预处理、prompt 构造与 target/token 辅助信息生成。
- `Myverl/verl/custom/`：定制化 PPO/GRPO 训练流程、off-policy/SFT 混合训练逻辑、数学 reward、vLLM rollout、数据集读取与配置。

除上述目录外，`Myverl/verl/` 的其他部分基本沿用 `verl` 原有实现，本文不展开说明。

## Data 目录

`Data/` 主要用于把原始 parquet 数据加工成训练所需格式。

### `Data/prompt_rewrite.py`

用于重写数据中的 `prompt` 字段。脚本会从原始 `prompt` 中提取题目文本，并按指定模板生成新的两轮对话格式：

- system：默认 `"You are a helpful assistant."`
- user：可选 `rewrite_problem_only` 或 `qa_style`

如果未开启 `--thinking_mode`，脚本还会从 `target` 中提取 `<think>...</think>` 内部的推理过程，只保留 thinking 内容作为新的 target。常用于构造“只学习推理过程”的数据。

示例：

```bash
python Data/prompt_rewrite.py \
  --input_file input.parquet \
  --output_file output.parquet \
  --prompt_key rewrite_problem_only \
  --backup_old_prompt \
  --backup_old_target
```

### `Data/se_template.py`

用于生成 `se_prompt` 字段。它从样本的 `prompt` 中读取题目，从 `target` 中解析专家推理过程，并套用 SE（self-edit / semantic equivalence 风格）提示词模板。

主要模板包括：

- `se_usr`：要求模型在不改变语义、步骤和结构的前提下，用更自然熟悉的表达重写专家推理。
- `se_usr_translate`：将给定推理文本翻译为准确、流畅的英文。

输出 parquet 会新增 `se_prompt` 字段，并删除若干中间评测字段，如 `responses`、`test_score`、`gemini_*`、`deepseek_*` 等。

### `Data/add_token_split_points.py`

用于为每条样本新增 `token_split_points` 字段。脚本会读取 `target` 或兼容字段中的 thinking 内容，按段落累计长度计算多个阶段的 token 切分点。

关键参数：

- `--num-stages`：切分阶段数，默认 `8`。
- `--curve-power`：控制切分比例曲线，`1.0` 为线性，大于 `1` 时更偏向后段。
- `--tokenizer-path`：用于计算 token 数的 HuggingFace tokenizer。
- `--workers`：多进程处理 worker 数。

该字段用于训练时按推理进度构造 prefix、阶段监督或 off-policy 数据。

## Myverl/verl/custom 目录

`Myverl/verl/custom/` 是对 `verl` 训练主流程的核心扩展目录。

### 训练入口与 Trainer

`calculate_probs_entory.py` 是一个 Hydra/Ray 入口脚本。它初始化 Ray、加载 tokenizer/processor、构造 worker 和 reward manager，最后调用 `NewRayPPOTrainer.calculate_probs_entropy()`，用于计算样本概率、熵等统计信息。

`new_ray_trainer.py` 定义 `NewRayPPOTrainer`，继承并扩展原始 `RayPPOTrainer`。该文件是自定义训练逻辑的中心，主要负责：

- 构造自定义 dataloader 和 worker。
- 合并 on-policy rollout 与 target/off-policy 数据。
- 构造 SE off-policy、hybrid off-policy、失败样本回收等 batch。
- 计算 advantage、KL、ratio 指标和训练 metrics。
- 支持保存 batch tensor、日志落盘、概率/熵统计等调试功能。

### Actor、Loss 与 Metrics

`new_dp_actor.py` 定义 `NewDataParallelPPOActor`，在原始 `DataParallelPPOActor` 基础上增加 off-policy 数据计算、ESS 统计和自定义 policy update。

`new_core_alg.py` 放置定制 loss 和 advantage 算法，包括：

- `compute_grpo_outcome_advantage_split`
- `compute_token_on_off_policy_loss`
- `compute_token_on_off_sft_loss`
- `compute_token_on_off_policy_loss_luffy`
- ESS 相关函数

这些函数支持 on-policy/off-policy token 级混合训练、prefix mask、SE mask、reward mask、importance ratio clipping、SFT 辅助 loss 等。

`new_metrics.py` 提供 ratio/log-ratio 的统计函数，按 on-policy 与 off-policy token 分别统计最小值、最大值、均值和分布区间，便于观察训练稳定性。

### Worker 与 Rollout

`fsdp_workers_new.py` 定义自定义 FSDP worker：

- `NewActorRolloutRefWorker`：负责 actor、rollout、reference policy 的初始化、生成、log prob 计算和 actor 更新。
- `CriticWorker`：负责 critic 初始化、value 计算与 critic 更新。
- `RewardModelWorker`：负责 reward model 打分。
- `AsyncActorRolloutRefWorker`：支持异步 rollout 场景。

`new_vllm_rollout.py` 定义 `NewvLLMRollout`，在原始 vLLM rollout 上调整 prompt 预处理、右填充输入处理、重复采样输出拼接等逻辑。它同时保留 `vLLMAsyncRollout` 包装类，用于异步 worker 调用 vLLM engine。

### 数据集扩展

`rl_dataset_with_target.py` 定义 `RLHFDatasetWithTarget`，继承 `verl` 原始 `RLHFDataset`，在标准 prompt 处理之外额外返回：

- `tgt_input_ids`：target 推理文本 token。
- `tgt_input_ids_lst`：多个候选 target。
- `target_probs`：target token 概率。
- `se_input_ids`、`se_tgt_input_ids`、`se_tgt_probs`：SE prompt/target 相关字段。

该文件还包含：

- `collate_fn`：合并 tensor 与非 tensor 字段。
- `BufferedDataLoader`：用于缓存和回收 batch。
- `ResumableRandomSampler`：支持保存/恢复随机采样状态。

### Reward 相关

`math_reward_manager.py` 注册了名为 `"math"` 的 reward manager。它根据 `data_source` 和 `reward_impl_version` 选择 GSM8K、MATH 或 math-verify reward，并将最终 reward 写到 response 最后一个有效 token 上。

`math_verify_reward.py` 使用 `math_verify` 的 `parse` 与 `verify` 判断模型答案和标准答案是否等价。它支持两种模式：

- `reward_fn_math_verify`：要求输出包含 `<think>...</think>` 格式。
- `reward_fn_math_verify_no_think`：不要求 thinking 标签，直接验证答案。

`deepscaler/` 目录包含从 DeepScaler 风格迁移来的数学 reward 工具：

- `deepscaler/rewards/reward_types.py`：reward 输入、输出、配置和问题类型定义。
- `deepscaler/rewards/math_reward.py`：基于 boxed answer、SymPy、mathd normalize 的数学判分逻辑，可选 LLM ORM。
- `deepscaler/rewards/math_utils/utils.py`：LaTeX/boxed answer 抽取、表达式规范化、符号等价判断等工具函数。
- `deepscaler/globals.py` 和 `system_prompts.py`：thinking 标签、模型名和 ORM prompt 常量。

### 配置文件

`Myverl/verl/custom/config/` 保存自定义训练和推理配置：

- `ppo_trainer.yaml`：基础 FSDP PPO/GRPO 配置。
- `rlplus_ppo_trainer.yaml`：扩展配置，包含 target 采样、off-policy loss、SFT reward、adaptive temperature、失败样本处理等开关。
- `ppo_megatron_trainer.yaml`：Megatron 策略配置。
- `sft_trainer.yaml`：SFT 训练配置。
- `generation.yaml`：vLLM 批量生成配置。
- `evaluation.yaml`：评测数据字段配置。

## 主要运行脚本

项目的主要训练脚本位于 `Myverl/examples/custom/`。三个脚本都会：

- 进入 `$HOME/LLM/Train/verl/` 目录执行。
- 设置 `RAY_DEDUP_LOGS=0` 和 `WANDB_MODE=offline`。
- 默认从 `/home/shared/Qwen2.5-Math-7B-16k-think` 加载模型。
- 使用 1 个节点、4 张 GPU、vLLM rollout、GRPO advantage。
- 将日志写入 `$HOME/LLM/Train/verl/logs/<PROJECT_NAME>/`。

第一个命令行参数会作为模型目录名拼到 `/home/shared/` 后面；剩余参数会直接透传给 Hydra。例如：

```bash
bash Myverl/examples/custom/train_grpo.sh Qwen2.5-Math-7B-16k-think trainer.total_epochs=1
```

### `Myverl/examples/custom/train_grpo.sh`

基础 GRPO/off-policy 训练脚本。默认训练数据为：

```bash
$HOME/LLM/Train/data_0306/openr1.parquet
```

验证数据默认为：

```bash
$HOME/LLM/Train/data/valid_with_aime25_new.parquet
```

该脚本设置 `name="vanilla"`，开启 `use_off_policy_loss=True`，使用 `reward_impl_version=4`，即不强制 thinking 标签的 math-verify reward。rollout 使用 `n=8`，`n_prefix=8`，`prefix_mode=linear`，适合做 vanilla GRPO 加 prefix/off-policy 回收的基线实验。

### `Myverl/examples/custom/train_hype.sh`

HYPE 风格训练脚本，主要用于带 token split prefix 的 off-policy/SFT 混合实验。默认训练数据为：

```bash
$HOME/LLM/Train/data_0306/se_prompt/openr1_seprompt_split.parquet
```

该数据通常需要先经过 `Data/se_template.py` 和 `Data/add_token_split_points.py` 处理。脚本设置：

- `off_policy_strategy="rl-sft"`
- `off_policy_reshape="low_sft_other_rl"`
- `prefix_mode=token_split`
- `reward_impl_version=3`

因此它更依赖数据中的 `se_prompt`、thinking target 和 `token_split_points` 等字段，用于比较不同 off-policy reshape 与 recycle loss 策略。

### `Myverl/examples/custom/train_luffy.sh`

LUFFY 风格训练脚本，默认训练数据为：

```bash
$HOME/LLM/Train/data_0306/openr1_prompt_rewritten_nothinking.parquet
```

该数据通常由 `Data/prompt_rewrite.py` 生成。脚本设置 `name="luffy"`，开启 `se_model.enable=True`，`se.target='standard_dynamic'`，并为 SE 分支配置 `actor_rollout_se`。主 rollout 使用 `n=8`，同时设置 `n_off=1`、`n_prefix=1`，用于结合标准 target shaping、SE 生成和 LUFFY 风格 off-policy loss。

常见覆盖参数示例：

```bash
bash Myverl/examples/custom/train_luffy.sh Qwen2.5-Math-7B-16k-think \
  data.train_batch_size=64 \
  trainer.test_freq=10
```

### 辅助文件

- `test.py`：读取目录下 parquet 文件并汇总打印 `test_score`，用于离线检查评测结果。
- `graph.ipynb`：实验分析 Notebook。
- `__pycache__/`：Python 运行生成的缓存文件，不属于源码逻辑。

# PICO 交接文档

PICO 是一个 GRPO + 「总结替换」(summarize-replacement, SR) 的训练框架：对模型全部做错的题，用**正在训练的 actor（student）**与一个**冻结的 teacher** 逐 token 联合解码出一条候选解，答对就替换掉该题的一条错误 rollout，作为 off-policy 样本参与训练，loss 中带上 teacher 约束的代价 Z_t。

代码在 `Myverl/verl/PICO/`，是从 `Myverl/verl/custom/` 中抽出并精简的独立副本。`verl/custom` 没有改动，其他走 `verl.trainer.main_ppo_new` 的旧脚本照常可用。

---

## 1. 快速上手

```bash
cd Myverl
# 训练（默认 4 卡 0-3，可用 GPU_DEVICES=4,5,6,7 覆盖；第一个参数是 Model/ 下的 student 名）
bash examples/h20/train_PICO.sh                 # 默认 Qwen3-4B-Base
bash examples/h20/train_PICO.sh Qwen3-8B-Base

# 离线联合解码扫参（训练前用来挑 joint_decode 参数）
LIMIT=4 bash examples/h20/run_joint_decode.sh   # 小规模冒烟
bash examples/h20/run_joint_decode.sh

# CPU 测试
pytest tests/PICO/test_PICO_on_cpu.py -v
```

入口：
- 训练：`python -m verl.PICO.main_ppo_new <hydra overrides>`
- 离线解码：`torchrun ... -m verl.PICO.joint_decode ...`（须在 `Myverl/` 下运行）

环境：conda 环境 `verl`（torch 2.6.0 / vllm 0.8.5.post1 / transformers 4.51.3）。verl 以 editable 方式安装，指向 `LLM/Myverl`，PICO 在包内，**无需重新安装**。换机器恢复环境见 `conda_envs/restore_env.sh`——注意它的默认值目前指向 ReLIFT，恢复 PICO 环境需显式指定：

```bash
ENV_NAME=verl VERL_SRC=<共享盘>/LLM/Myverl bash conda_envs/restore_env.sh
```

---

## 2. 一个训练 step 的流程

`new_ray_trainer.PICORayPPOTrainer._train_step`，结构与原版 `RayPPOTrainer.fit()` 循环体一一对应，PICO 新增部分在代码中标 `# PICO:`。

| 步骤 | 做什么 | 代码位置 |
|---|---|---|
| 1. rollout | 每题短 prompt 重复 n=8 份，vLLM 采样 | `_train_step` |
| 2. SR 替换 | 选出 8 条全错的题 → 联合解码候选 → 打分 → 每题挑一条答对的，覆盖其第一条错误 rollout，标 `prefix_mask=1`，附 `off_log_z` | `_summarize_replace` / `_generate_candidates` / `build_off_rows` |
| 3. reward | math_verify 打 0/1 分；按题分组统计；生成 `reward_mask` | `_train_step` reward 块 |
| 4. old_log_prob | actor 自己的 log prob。off 行分母也是 actor 本身（student 就是 actor），不做替换 | `_train_step` |
| 5. advantage | 标准 GRPO（按 uid 分组） | 原版 `compute_advantage` |
| 6. update_actor | `loss_mode` 决定 on/off 行怎么训 | `new_dp_actor` → `new_core_alg` |

**联合解码（`agree` 模式）**，`joint_decode_core.agree_select`：
- 候选 token 必须同时在 student top-k（50）和 teacher top-k（脚本设 5）里，且 teacher 概率 ≥ `agree_teacher_min_prob`（0.05）。
- 在交集内按 student 的分布采样。
- 交集为空时，按 `agree_fallback`（默认 `student`）回退。
- 每个 token 记录 Z_t（交集覆盖的 student 概率质量）和 log Z_t。
- student 是 actor 权重的一份不分片镜像，每步同步一次；teacher 是纯 HF 模型，懒加载，用时搬到 GPU。

**动态 top-k**（`agree_dynamic_top_k=True`，默认关闭）：
1. 先用固定 teacher top-k 取交集，计算 Z_t；
2. Z_t ≥ `agree_target_z` 的 token 保持原交集；
3. 不达标的 token 逐步扩大 teacher 的 k，直到 `agree_dynamic_max_top_k`，取第一个让 Z_t 达标的 k；
4. 扩到上限仍不达标的回退。

所以非回退 token 的 Z_t 一定 ≥ target，fallback 率可能比固定宽度更高。

---

## 3. 文件说明

所有文件名与 `verl/custom/` 中的原文件对应。

| 文件 | 内容 |
|---|---|
| `main_ppo_new.py` | 训练入口，以原版 `verl/trainer/main_ppo.py` 为底稿。差异：actor worker、reward、训练集换成 PICO 版本；合并 `config/joint_decode.yaml` |
| `new_ray_trainer.py` | `PICORayPPOTrainer`：配置检查、pass@k 验证、SR 替换、训练循环、日志/dump/save_tensors |
| `new_fsdp_workers.py` | `PICOActorRolloutRefWorker`：原版 worker + `generate_joint`（联合解码）。只支持 FSDP + vLLM 同步模式 |
| `new_dp_actor.py` | `PICODataParallelPPOActor.update_policy`：结构照原版，loss 有三个分支：`vanilla`（原版 GRPO）/ PICO 模式 / 注册表 loss |
| `new_core_alg.py` | `compute_token_on_off_sft_loss`：on/off 行分开计算 loss，以及 loss 组成的监控指标 |
| `joint_decode_core.py` | 联合解码核心（融合规则、采样、KV cache 循环）。训练和离线共用这一份 |
| `joint_sr.py` | 训练侧调用联合解码的胶水：选题上限、每题候选数 K、结果整形 |
| `joint_decode.py` | 离线联合解码评测脚本（原 `Data/joint_decode.py`），输出每卡一个 parquet |
| `rl_dataset_with_target.py` | 训练集：在原版 `RLHFDataset` 上增加 `summarize_input_ids`（teacher 的长 prompt）和 `tgt_input_ids`（参考解，只用于 dump） |
| `math_reward_manager.py` | 规则 reward，合并自 custom 的三个文件，已在 2048 条真实回复上与原版逐条对齐 |
| `new_metrics.py` | 验证集的精确 pass@k |
| `config/joint_decode.yaml` | 联合解码全部参数，命令行用 `+joint_decode.<key>=` 覆盖 |

脚本：
- `examples/h20/train_PICO.sh`：训练，60 个参数，只保留 PICO 实际读取的。
- `examples/h20/run_joint_decode.sh`：离线扫参，对 `teacher_min_prob` / `target_z` / `fuse_weight` 逐值运行并汇总 `summary.tsv`。

---

## 4. 关键参数

### loss_mode（`actor_rollout_ref.actor.policy_loss.loss_mode`）

格式为 `"<on 行>-<off 行>"`：

| 取值 | on 行 | off 行（被替换的行） |
|---|---|---|
| `rl-rl`（当前） | PPO dual-clip | PPO dual-clip，比值由 `off_policy_reshape` 决定 |
| `rl-sft` | PPO dual-clip | `-log π` |
| `sft-sft` | 正 advantage 的 token 上做 `-log π` | `-log π` |
| `none-sft` | 不训练 | `-log π` |
| `vanilla` | 原版 GRPO，所有行一视同仁，忽略 `prefix_mask`/`off_log_z` | 同左 |

### off_policy_reshape（off 行为 rl 时）

| 取值 | off 行的比值 |
|---|---|
| `clip_with_z`（当前） | r = π_θ/μ_t = Z_t·(π_θ/p_t)，Z_t 放进比值里一起 clip |
| `clip_without_z` | r = π_θ/p_t 正常 clip，clip 之后再乘 Z_t |
| `clip` | 普通 PPO，**不使用 Z_t**（旧注释说它等于 `clip_without_z`，是错的，已改正） |
| `p_div_p_0.1` | LUFFY 写法 p/(p+0.1) |

### SR 替换

| 参数 | 当前值 | 说明 |
|---|---|---|
| `rollout.summarize_replace` | `wrong_only` | `all` 为每题都生成候选，固定覆盖最后一条 rollout（可能覆盖掉正确的） |
| `rollout.accuracy_threshold` | 0.1 | n=8 时等于只选 8 条全错的题 |
| `rollout.summarize_replace_select` | `shortest` | K 条候选里取第一条合格的；`longest` 取最后一条 |
| `algorithm.trajectory_filter.enable` | False | 打开后选候选时额外过滤引用草稿、复述指令、重复输出的回答 |
| `data.recycle_sr_disable_after` | 未设置 | 第 N 步起关闭 SR |
| `policy_loss.mask_off_inject_group` | 未设置 | `True` 时被替换的题整组不进 loss；`on_only` 时只去掉其 on 行 |

### 联合解码（`joint_decode.*`，默认值见 yaml）

当前脚本覆盖了 `agree_teacher_top_k=5`、`agree_dynamic_top_k=False`，其余用默认：`agree_student_top_k=50`、`agree_teacher_min_prob=0.05`、`agree_fallback=student`、`max_new_tokens=5120`、`batch_size=32`（每卡行数）、`max_k=2`、`temperature=1.0`（应与 `rollout.temperature` 一致）。

### 换成标准 GRPO 对照

```bash
loss_mode="vanilla"
    +data.recycle_sr_disable_after=1 \             # 必须：否则 SR 仍会把候选替换进来
    actor_rollout_ref.actor.clip_ratio_high=0.28 \  # 保持 clip 上界和 PICO 一致
```

说明：yaml 中 `clip_ratio_low/high` 默认 0.2，PICO loss 取 `max(clip_ratio_high, clip_ratio)=0.28` 作上界，而 vanilla 直接用 0.2，不加第三行会让上界从 1.28 变成 1.2。

---

## 5. 监控指标

| 指标 | 看什么 |
|---|---|
| `batch/sr_wrong_questions` / `sr_target_questions` | 本步全错的题数 / 实际生成候选的题数 |
| `batch/sr_accepted` | 实际完成替换的题数 |
| `batch/sr_rollout_acc` | 联合解码候选的正确率 |
| `batch/sr_joint_dropped` | 超出行预算被丢弃的题数，经常非 0 时调大 `joint_decode.batch_size` |
| `batch/sr_joint_fallback_frac` | 回退 token 比例，接近 1 说明约束没起作用 |
| `batch/sr_joint_keep_ratio` | 采样时 Z_t 的均值（回退记 0） |
| `batch/sr_joint_truncated` | 生成到 `max_new_tokens` 被截断的候选数（历史上约 0.6%） |
| `actor/off_z_mean` | 实际乘进 loss 的 Z_t 均值 |
| `actor/off_ratio_mean` | off 行比值均值，`clip_with_z` 下首个 mini-batch 应约等于 Z_t |
| `actor/on_pg_clipfrac` / `off_pg_clipfrac` | on/off 行被 clip 的比例 |
| `batch/solve_none` / `solve_one` | 只统计 on-policy 行，不受替换影响 |

其余输出：`<LOG_DIR>/training_metrics.jsonl`（每步指标）、`rollout_data/normal/<step>.jsonl`（含 `is_replaced`、`target`、`original_index`）、`save_tensors/tensors_step_*.pt`（`Data/load_saved_tensors.py` 和 `Data/plot_*` 读取）。

---

## 6. 与旧代码（verl/custom）的行为差异

在保留的选项下训练结果一致（见第 7 节的测试），以下是有意的差异：

1. **替换行的 EOS 现在参与训练**。Qwen3 的 EOS 和 pad 是同一个 id，旧代码按 pad 判断长度会把 EOS 去掉；现在用联合解码返回的长度构造 mask，与 vLLM 行一致。
2. **`actor/on_pg_clipfrac` 现在会计算**，旧代码写死为 0。
3. **动态 top-k 逻辑改变**（见第 2 节），旧逻辑不回退、可能比固定宽度更窄，且忽略 `teacher_min_prob`。
4. **`batch/sr_joint_truncated` 修正**：旧代码和 10240 比较，永远为 0；现在和 `max_new_tokens` 比较。
5. **`sft-rl` / `none-rl` 按名字执行**：旧代码会悄悄退回 `rl-sft`。
6. **删掉的功能会直接报错**，不会悄悄忽略：recycle/failure buffer、warmup、n_off、SE、rephraser、离线 SR、rephrase-KL、affinity、off 行的 batch_mean_norm / group_ess_weight / low_clip / 各类 SFT 变体、`summarize_replace_select=logp`。
7. 删掉恒为常数或重复的指标：`solve_none_format`、`sr_k`、`on_solved_none`、`off_ratio_max/min_clip_frac`、`off_ratio_scale`、`off_distill_loss`。

---

## 7. 测试

`tests/PICO/test_PICO_on_cpu.py`，21 个测试，CPU 上约 10 秒。主要是**与旧代码在相同输入上逐位对照**：

- SR 替换：4 种配置下，替换后的 batch 每个张量和每个 `batch/sr_*` 指标与旧 trainer 一致。
- loss：7 种 mode/reshape × `all_max_clip` 有无 × `off_log_z` 有无，loss、梯度、指标逐位一致；`on_pg_clipfrac` 与原版 `compute_policy_loss` 一致。
- actor：小模型上完整跑一次 `update_policy`，权重、`log_prob`、指标与旧 actor 一致；`vanilla` 与 verl 原版 actor 一致。
- reward：与原版 `MathRewardManager` 一致。
- EOS 修正、动态 top-k 四种情况、配置检查。

---

## 8. 当前状态与待办

**状态**
- 分支 `PICO`（从 `summarize-explain` 分出）。18 个文件**已暂存、尚未提交**：提交时被权限检查拦下，需要手动执行 `git commit`。
- **还没有在 GPU 上完整跑过**。CPU 测试覆盖了 trainer 的数据处理、loss、actor 更新和 reward，没有覆盖 Ray worker、FSDP、vLLM、联合解码在 GPU 上的实际运行。

**建议的首次运行检查**
1. 用 `train_PICO.sh` 和 `train_hype_summarize.sh` 各跑几步，对比 `batch/sr_accepted`、`actor/off_z_mean`、`actor/pg_loss`。
2. 确认 `actor/off_ratio_mean` 在首个 mini-batch 上约等于 `actor/off_z_mean`（`clip_with_z` 下）。若明显偏离，可能是温度不一致或 token 对齐出错。
3. 看 `batch/sr_joint_truncated` 和 rollout dump 中 `is_replaced=True` 的行，确认候选正常结束。

**已知问题 / 可选改进**
- `conda_envs/restore_env.sh` 默认值指向 ReLIFT（见第 1 节）；解压中断后会跳过重新解压。
- 离线扫参默认 `teacher_min_prob=0.01`、`max_new_tokens=8192`，训练是 0.05、5120。用离线结果指导训练前先对齐。
- `wrong_only` 下每步多算一次 on-policy reward（用于选题）。
- 若开启动态 top-k，yaml 默认 `target_z=0.9`、上限 20 可能造成大量回退，建议先离线扫参。
- `summarize_prompt` 数据只有 1 条，但 `max_summarize_prompts` 默认 8，每题重复渲染 8 遍；可加 `+data.max_summarize_prompts=2`。
- `verl/custom/joint_decode_core.py` 注释中仍写着 `Data/joint_decode.py`（旧代码未改）。

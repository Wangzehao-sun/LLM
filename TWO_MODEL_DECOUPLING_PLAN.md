# 拆分 reasoner / rephraser 为两个独立模型

## Context

当前 `summarize-explain` 分支上，**一个模型交替扮演两个角色**，且两个角色更新**同一份权重**：

- **reasoner** = normal step（`_train_step_internal(is_failure_recycle_step=False)`）：短 question prompt 下的标准 GRPO。
- **rephraser** = recycle/extra step（`is_failure_recycle_step=True`）：长 summarize prompt 下 rollout，由 `fit()` 在失败缓冲区攒满时触发（`new_ray_trainer.py:1738`）或 `warmup_steps`（`:1671`）。

两者共用 `self.actor_rollout_wg` 做 generate（`:2071`）/ compute_log_prob（`:2787`）/ update_actor（`:2986`）；唯一的角色差异是 loss mode 覆盖（`:2980-2984`）与采样温度覆盖（`extra_temperature`，`fsdp_workers_new.py:713`）。

目标：改成两个模型各自持有权重、优化器、rollout 引擎和 checkpoint，rephraser 的更新不触碰 reasoner 权重，反之亦然。

### 已确认的关键约束（探索阶段验证）

1. **`Role.ActorRolloutSE` 这个先例不能照抄。** 它在 `fsdp_workers_new.py:130-134` 只设 `_is_ref=True`，而 `update_actor` / `generate_sequences` / `compute_log_prob` / `save_checkpoint` 分别断言 `_is_actor`（`:615`）、`_is_rollout`（`:661`）、`_is_actor`（`:696`）、`_is_actor`（`:785`）。即 SE 角色**既不能训练也不能生成**；`:2298` 那句 `actor_rollout_se_wg.generate_sequences` 会直接踩断言，只因所有脚本都写 `+se_model.enable=False` 才没暴露。**第二个模型必须建成完整的 actor+rollout 角色。**
2. **tokenizer 只有一份**，由 `main_ppo_new.py:86` 从 `actor_rollout_ref.model.path` 构造，被 dataset（`summarize_input_ids` 是**离线预分词**的）、reward manager、以及 `_build_hybrid_off_policy_output`（把一个模型 rollout 出的 raw token id 直接拼进另一个模型的 `input_ids`）共用。rephraser 换模型时**必须同 vocab/tokenizer 家族**（Qwen3-4b-base ↔ Qwen3-8b 可以），否则静默数据损坏。
3. **`use_kl_loss` 必须保持 False**（现有脚本已是）。rephraser worker 的 `_is_ref=False`，走 `compute_ref_log_prob` 会踩 `:750` 的断言。

### 本次已定的四个决策

| 决策 | 选择 |
|---|---|
| rephraser 梯度来源 | **只靠 recycle step**（先跑通；SR 候选行只训 reasoner） |
| recycle loss 形态 | **只支持 `summarize_loss_on_rollout_prompt=True`**，另一分支启动时断言拒绝 |
| 显存分配 | **共享 4 卡 + 双向 offload**，`gpu_memory_utilization` 各降到 ~0.42 |
| 初始权重 | **rephraser 用不同模型**（须同 tokenizer 家族） |

---

## Step 0（前置，必须先做）：单进程双 vLLM 引擎冒烟测试

这是整个方案唯一的高风险未知项，**在写任何 trainer 代码之前先验证**。

`create_colocated_worker_cls`（`single_controller/ray/base.py`）会把同一资源池里所有 worker 合并进**一个** Ray actor 进程（`max_colocate_count=1`）。于是一个 OS 进程内要构造两个 `LLM(enable_sleep_mode=True, distributed_executor_backend="external_launcher")`。而 vLLM 的 sleep 机制建立在**进程级全局 `CuMemAllocator` 单例**上，并行状态 `_TP`/`_WORLD` 也是全局的。

在 GPU 机器上跑约 20 行脚本：同进程内建两个 `LLM(..., gpu_memory_utilization=0.42, tensor_parallel_size=1)`，各自独立 sleep/wake，各自 generate，确认互不干扰。

- **通过** → 按本方案走共享池。
- **失败**（re-init 断言，或 A 的 `sleep()` 释放了 B 的显存）→ 退回 2+2 分池，Step 1c/3a/7 需要改，并额外修 4 处 `world_size` 相关代码（`_balance_batch`、`_validate_summarize` 的 `pad_dataproto_to_divisor`、`ppo_mini_batch_size` 归一化除数、`_validate_config` 整除检查）。

顺带确认 host RAM：两个模型双向 offload，7B 约需 `2 × (28GB params + 56GB Adam) ≈ 168GB`。不够则 7B 不可行（4B 尚可）。

---

## Step 1：Role 枚举与注册

**`Myverl/verl/trainer/ppo/ray_trainer.py`**，`Role` 枚举（`:66-78`）：加 `ActorRolloutRephraser = 8`。保留 `ActorRolloutSE = 7` 不动（它是那个 inference-only 的死先例）。

**`Myverl/verl/trainer/main_ppo_new.py`**，在 SE 那段（`:158-160`）之后，照同样形状加：

```python
if config.get("rephraser", {}).get("enable", False):
    role_worker_mapping[Role.ActorRolloutRephraser] = ray.remote(actor_rollout_cls)
    mapping[Role.ActorRolloutRephraser] = global_pool_id   # 共享池
```

复用同一个 `actor_rollout_cls`（`NewActorRolloutRefWorker`），不需要新 worker 类。门控用新的顶层 `rephraser.enable`，默认 False ⇒ 所有现有脚本零改动、行为不变。

资源池沿用 `global_pool_id`（4 卡），这样 `world_size` 对两个 worker group 都是 4，`_validate_config` 的整除检查、`_balance_batch`、`pad_dataproto_to_divisor`、`ppo_mini_batch_size //= device_mesh.size()` 全都不用改。角色本来就是**步级严格串行**（`fit()` 从不并发跑两个角色），分卡没有并发收益，offload 才是对的手段：闲置模型整步驻留 host RAM，重载成本可摊薄。

---

## Step 2：`fsdp_workers_new.py` — 让第二个模型成为真正可训练的 actor+rollout

`NewActorRolloutRefWorker.__init__`，`:130-134`。把三个字面量列表提到模块级常量（放在 `get_sharding_strategy` 旁边），并**只**把新角色加进前两个：

```python
_ACTOR_ROLES   = ("actor", "actor_rollout", "actor_rollout_ref", "rephraser_actor_rollout")
_ROLLOUT_ROLES = ("rollout", "actor_rollout", "actor_rollout_ref", "rephraser_actor_rollout")
_REF_ROLES     = ("ref", "actor_rollout_ref", "se_rollout_ref")   # se_* 故意只在这里
```

`:131-134` 改用这些常量。提取常量的额外收益：角色→flag 映射变成 CPU 可单测（Step 8），并且用代码本身说明「`se_rollout_ref` 是刻意 ref-only」。

`rephraser_actor_rollout` ∈ actor+rollout 且 ∉ ref，正好给到需要的东西：

- `_is_actor=True` ⇒ `init_model` 建参数**和**优化器+LR scheduler（`:517-521`）、`NewDataParallelPPOActor`（`:562`）、自己的 `FSDPCheckpointManager`（`:587-593`）；四个断言全过。
- `_is_rollout=True` ⇒ `_build_rollout`（`:565`）⇒ 自己的 vLLM 引擎 + sharding manager。
- `_is_ref=False` ⇒ 不多加载第三份权重。

本文件其他地方不改。

---

## Step 3：`init_workers()` 建第二个 worker group

**`new_ray_trainer.py`**，`init_workers`（`:540-654`）。

**3a. 类注册** — 在 `if self.hybrid_engine:` 块内、SE 段（`:552-578`）之后，照它的 `deepcopy` + `OmegaConf.merge` 形状：

```python
if Role.ActorRolloutRephraser in self.role_worker_mapping:
    pool_rp = self.resource_pool_manager.get_resource_pool(Role.ActorRolloutRephraser)
    rp_config = OmegaConf.merge(
        deepcopy(self.config.actor_rollout_ref),
        self.config.get("actor_rollout_rephraser", OmegaConf.create()),
    )
    self.resource_pool_to_cls[pool_rp]["rephraser"] = RayClassWithInitArgs(
        cls=self.role_worker_mapping[Role.ActorRolloutRephraser],
        config=rp_config, role="rephraser_actor_rollout",
    )
```

`deepcopy` 是必须的（SE 段也这么做）：`fsdp_workers_new.py:155-168` 会**原地修改** `config.actor.ppo_mini_batch_size` / `ppo_micro_batch_size`，共享节点会被两个 worker 重复归一化。

配置子树 `actor_rollout_rephraser` 合并在 `actor_rollout_ref` **之上**，只写覆盖项：`model.path`、`actor.optim.lr`、`actor.fsdp_config.{param_offload,optimizer_offload}`、`actor.policy_loss.loss_mode`、`rollout.{gpu_memory_utilization,temperature}`。其余（max 长度、dtype、remove_padding）继承 —— 在共享 tokenizer 的约束下这正是想要的。

**3b. worker group 实例化** — 在 `:635-641` 之后。先无条件 `self.rephraser_wg = None`，再按 key 取，`_role_wg` 才不会 `AttributeError`：

```python
self.rephraser_wg = None
if "rephraser" in all_wg:
    self.rephraser_wg = all_wg["rephraser"]
    self.rephraser_wg.init_model()
```

顺序上让 reasoner 先 `init_model`（`:636`），保持主模型行为与现状一致。

**3c. `_validate_config` 加断言**（`rephraser.enable=True` 时）：

- `rollout.summarize_loss_on_rollout_prompt == True` —— 即已定的 D2，另一分支直接拒绝启动而非静默跑错。
- `not actor_rollout_ref.actor.use_kl_loss` —— rephraser 无 ref 模型。
- `rollout.mode == 'sync'` —— `async_rollout_manager` 只连了 reasoner（`:653`），异步模式下 `_validate_summarize` 会从错的模型生成。
- `actor_rollout_rephraser.model.path != actor_rollout_ref.model.path` 时**大声警告**共享 tokenizer 的要求（本次选了不同模型，这条一定会触发，属预期）。

**3d. LR schedule（需要你定）** — `_create_dataloader`（`:530-538`）把 `total_training_steps` 塞进 `actor_rollout_ref.actor.optim`。但 recycle step 的**总数是数据驱动的、事前不可知**（取决于 buffer 填充速率），拿它当 rephraser 的 LR 衰减分母没有意义。建议给 rephraser **恒定 LR**（`lr_warmup_steps_ratio=0` + 一个大的 sentinel `total_training_steps` 让衰减近似平坦）。

---

## Step 4：`_train_step_internal` 的分发

**命名决策：`self.actor_rollout_wg` 保持指向 reasoner。** 所有继承来的父类方法 —— `_validate`（`ray_trainer.py:600`）、`_balance_batch`（`:893`）、`_save_checkpoint`（`:807`）、`_load_checkpoint`（`:842`）—— 都引用 `self.actor_rollout_wg`，而它们**本来就该**打到 reasoner。改名会波及 stock verl。

加一个 helper：

```python
def _role_wg(self, is_recycle: bool):
    """normal step -> reasoner; recycle/extra step -> rephraser."""
    if is_recycle and self.rephraser_wg is not None:
        return self.rephraser_wg
    return self.actor_rollout_wg
```

在 `_train_step_internal` 开头（`:1793` `metrics = {}` 之后）绑一个**局部**变量，而不是实例属性：

```python
step_wg = self._role_wg(is_failure_recycle_step)
metrics['debug/step_role'] = 1 if step_wg is self.rephraser_wg else 0
```

用局部量的理由：`_train_step_internal` 末尾会调 `_validate()` 和 `_validate_summarize()`（`:3069-3071`），这两个必须分别钉在 reasoner 和 rephraser 上，与本步角色无关。隐式实例属性会泄漏进去。

路由表（行号为当前 `new_ray_trainer.py`）：

| 行 | 调用 | 路由到 | 备注 |
|---|---|---|---|
| 1785, 3118 | start/stop_profile | **两者** | 补 rephraser |
| 2071 | 主 `generate_sequences` | `step_wg` | |
| 2146 | `compute_log_prob` → `noprefix_logp` | **reasoner** | 语义升级，见 Step 6 |
| 2165 | `compute_log_prob` → `summarize_long_log_prob` | `step_wg` | D2 断言后不可达 |
| 2787 | `compute_log_prob(batch)` → `old_log_prob` | `step_wg` | |
| 2986 | `update_actor(batch)` | `step_wg` | **解耦点** |
| 2251/2289/2400/2883 | n_off / REMAX / ref_in_actor | `step_wg` | 现有配置下均不可达，一并路由 |
| 2298, 2322 | `actor_rollout_se_wg` | **不动** | 死代码，会踩 ref-only 断言 |
| 1295, 1302 | SR 候选生成 + 长 prompt logp | **恒 rephraser** | Step 5 |
| 1496/1499/1521 | `_validate_summarize` | **恒 rephraser** | Step 7 |

`_summarize_replace_normal_step` 加一个显式 `gen_wg` 形参（默认 `None` → reasoner），调用点（`:2216`）传 `gen_wg=self.rephraser_wg or self.actor_rollout_wg`。这是个 normal-step 子路径却必须用 rephraser，正是最不该用隐式状态的地方。

`_validate_summarize` 不加形参，函数开头 `val_wg = self.rephraser_wg or self.actor_rollout_wg`，三处（含 `.world_size`）都用它。

---

## Step 5：`_summarize_replace_normal_step` 的 IS ratio 语义（最微妙的一步）

**现状（单模型 θ）**：候选 `y ~ π_θ(·|x_long)`（`:1295`）；`long_log_prob = log π_θ(y|x_long)`（`:1302`）存为 `target_probs` 与 `off_old_log_probs`（`:1338-1339`）；行的 `input_ids` 变成 `[x_short, y]`、`prefix_mask=1`；loss 里 `off_ratio = π_θ(y|x_short) / π_θ(y|x_long)`。这是从 proposal `q=π_θ(·|x_long)` 到 target `p=π_θ(·|x_short)` 的教科书式 IS 修正。它成立依赖一个巧合：**proposal 和 target 是同一份权重**。

**解耦后**：候选由 rephraser φ 生成，proposal 变成 `q=π_φ(·|x_long)`，target 仍是 `p=π_θ(·|x_short)`（被更新的是 reasoner）。正确权重是

```
w(y) = π_θ(y|x_short) / π_φ(y|x_long)
```

**结论：不变式仍然成立，但含义变了。** `exp(logp_short − logp_long)` 依然是合法的 IS 权重，**前提是 `logp_long` 由 rephraser 计算** —— 即 `:1295` 和 `:1302` 必须**都**打到 `rephraser_wg`。若 `logp_long` 来自 reasoner，得到的是 `π_θ(y|x_short)/π_θ(y|x_long)`，proposal 是错的（y 从未由 θ 采出），估计量被**静默**引入偏差、不报错。

所以代码改动只是两处换个 worker，但数学上从「纯 prompt-shift 修正」变成了「**prompt-shift + model-shift 联合修正**」。四个实际后果：

1. **方差 / ESS 是真正的风险。** 现在两个密度共享权重，逐 token ratio 贴近 1；θ 与 φ 分化后 ratio 系统性偏移，序列级权重随长度（~10k token）指数退化。预期 `off_ratio_ess` 崩、`off_ratio_max_clip_frac` 饱和。缓解手段代码里已有，优先级：`off_policy_reshape='batch_mean_norm'` → `'group_ess_weight'` → 收紧 `off_policy_max_clip`。**把 ESS 监控当作信任任何结果前的闸门**；`:2801-2806` 已有的 `batch/long_prob_off_standard` vs `batch/old_prob_off_standard` 差距正是首要健康指标。
2. **`off_policy_reshape='p_div_p_0.1'` 会让解耦在数学上失效** —— 该分支（`new_core_alg.py`）直接覆写 `off_ratio = exp(log_prob)/(exp(log_prob)+0.1)`，完全丢弃 `old_log_prob`，proposal 密度不进梯度。不必改代码，但要知道 IS 叙事只在 `reshape ∈ {clip, vanilla, batch_mean_norm, group_ess_weight}` 时成立。`train_hype_summarize.sh` 用 `clip`，有效。
3. **`off_distill_coef` 变成真正的跨模型蒸馏**（`new_core_alg.py:673` 的 `teacher_prob` 变成 rephraser 的长 prompt 概率）。现有脚本未启用。
4. **`summarize_replace_select='logp'` 语义漂移** —— `cand_logp_mean`（`:1348`）来自 `long_log_prob`，解耦后衡量的是「rephraser 对哪条最自信」，而非 `:1341-1346` 注释声称的「对当前策略最亲和」。**建议保持默认 `'shortest'` 并改注释**；真要 reasoner-affinity 选择需要对 W*K 行做一次 reasoner forward（最多 128×8 序列），成本接近整轮 rollout，另立开关。

另需一条断言：`off_old_log_probs → old_log_probs` 的 swap 在 `reshape ∈ {'dynamic_clip','vanilla'}` 时被跳过（`:2812`），此时 SR 行的 `old_log_probs` 仍是 reasoner 短 prompt 值、ratio 退化成 ≈1。`rephraser.enable=True` 时应断言 `target_probs` 与 `off_old_log_probs` 两条通道**至少一条生效**。

---

## Step 6：`noprefix_logp` 改由 reasoner 计算（`:2146`）

这份 log-prob 喂两个东西：`rephrase_kl` 信任项（`new_dp_actor.py:495-509`）与 `reasoner_affinity_coef` reward shaping（`:2921-2949`）。注释写的意图是「reasoner 的接受度」`sg[π_θ(·|x)]`，但单模型下它其实是 rephraser 自己的短 prompt 快照 —— 一个自 KL，只是意图的代理。

**解耦后应打到 reasoner**，`reasoner_affinity` 才真的名副其实：`f(y) = mean_t log π_θ(y_t|x_short, y_<t)`，即 reasoner 对改写结果的长度归一化对数似然。**这是解耦带来的最干净的收益。**

代价（需要你签字）：`rephrase_kl` 变成 `KL(π_φ(·|x,prefix) ‖ π_θ(·|x))` —— 从「φ 自身移动的信任域」变成「把 φ 拉向 θ 的跨模型拉力」。本次 rephraser 用**不同模型**，step 0 两者权重就不同，这个项从一开始就非零且随分化增长，强度会漂移。`rephrase_kl_coef=0.001` 几乎肯定要重调。**建议初期直接设 0，只用 affinity。** 该张量是 forward-only `compute_log_prob` 的产物、天然 detached，不会有梯度漏进 reasoner。

---

## Step 7：checkpoint / resume / validation

**`_save_checkpoint`**（override 父类 `ray_trainer.py:807-840`）：actor 存完后加 rephraser，路径 `global_step_{n}/rephraser`。reasoner 保持 `.../actor` 以便旧 checkpoint 仍可 resume。同时持久化 `self.extra_steps`（现在是 `fit()` 作用域的局部整数，resume 后归零会打乱 recycle 日志 x 轴）到 `rephraser_state.pt`。

> 附带发现（本次不修）：`failed_questions_buffer` 也没有进 checkpoint。解耦后它门控着 rephraser 的**全部**学习信号，这个既有缺口的后果被放大了。

**`_load_checkpoint`**（override `:842-889`）：rephraser 的加载必须**带存在性判断** —— 否则 resume 一个拆分前的 run 会在 `FSDPCheckpointManager.load_checkpoint` 里崩。找不到就打印提示、从 `model.path` 初始化。

**validation**：
- `_validate()` 继承不动，用 `self.actor_rollout_wg` = reasoner ✓（保留命名的回报）。
- `_validate_summarize()`（`:1428-1592`）改用 `val_wg`。
- **建议新增一个跨模型指标**：函数里已有 `cand_out`，多一次 `self.actor_rollout_wg.compute_log_prob(cand_out)` 就能得到 reasoner 对 rephraser 输出的 log-prob → 记 `val_summarize/reasoner_affinity` 与 `is_ratio_ess`。这和 `batch/long_prob_off_standard` 是判断解耦是否健康的两个核心数字。

---

## Step 8：配置与训练脚本

**`Myverl/verl/custom/config/rlplus_ppo_trainer.yaml`** 加两块（免得处处 `+`）：

```yaml
rephraser:
  enable: False
actor_rollout_rephraser: {}     # 合并在 actor_rollout_ref 之上，只放覆盖项
```

空节点 merge 是 no-op，故 `enable=False` 与今日行为逐字节一致。

**新脚本 `Myverl/examples/custom/train_hype_summarize_2model.sh`**（复制 `train_hype_summarize.sh`，不改原脚本，保留单模型基线做 A/B）：

```bash
+rephraser.enable=True \
+actor_rollout_rephraser.model.path=$REPHRASER_MODEL_PATH \
+actor_rollout_rephraser.actor.optim.lr=1e-6 \
+actor_rollout_rephraser.actor.fsdp_config.param_offload=True \
+actor_rollout_rephraser.actor.fsdp_config.optimizer_offload=True \
+actor_rollout_rephraser.rollout.gpu_memory_utilization=0.42 \
actor_rollout_ref.rollout.gpu_memory_utilization=0.42 \
actor_rollout_ref.actor.fsdp_config.param_offload=True \
actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
+actor_rollout_ref.actor.policy_loss.rephrase_kl_coef=0.0 \
```

保留 `+actor_rollout_ref.rollout.summarize_loss_on_rollout_prompt=True`（D2 要求）与 `+se_model.enable=False`。

---

## 验证

**无 GPU 可做**：
1. `Myverl/tests/workers/test_worker_roles_on_cpu.py` — 断言 `rephraser_actor_rollout` ∈ actor+rollout 且 ∉ ref；`se_rollout_ref` 仅 ref。（Step 2 提常量的意义）
2. `Myverl/tests/trainer/test_rephraser_config_on_cpu.py` — 覆盖项生效、未覆盖项继承、且改动 merge 结果不影响原节点（`deepcopy` 保证）。
3. `_role_wg` 是 3 行纯函数，用两个 sentinel 打 4 种组合。
4. 静态审计：`grep -n "self\.actor_rollout_wg" new_ray_trainer.py`，确认剩余每一处都属于 `init_workers` / `_role_wg` / reasoner 钉死点（2146、`_validate` 继承、`_balance_batch`）/ 死分支。1200 行的方法没法单测，这个 grep 加 `debug/step_role` 面包屑是务实替代。
5. `cd Myverl && pre-commit run --all-files` + `pytest tests/**/test_*_on_cpu.py`。

**必须 GPU**：
6. Step 0 的双 vLLM 冒烟测试（**最先做**）。
7. 显存与 host RAM 余量：看 `perf/max_memory_allocated_gb`、`perf/cpu_memory_used_gb`。
8. **解耦证明**：(a) 在一个 recycle step 前后各跑一次 `_validate()`，断言 reasoner 准确率**逐位相同** —— 没人碰它的权重；(b) 每个模型记一个 param-norm，断言首个 recycle step 后开始分化。(a) 更强。
9. **IS ratio 健康度**：前 ~50 步盯 `actor/off_ratio_ess`、`off_ratio_max_clip_frac`、`long_prob_off_standard` vs `old_prob_off_standard`。ESS 崩就先换 `batch_mean_norm`，再谈方法有效性。
10. checkpoint 往返：存→杀→resume，两模型都加载、`extra_steps` 恢复、拆分前 checkpoint 走提示分支而非崩溃。

## 实施顺序

先落 Step 1-4 且保持 `rephraser.enable=False`，确认单模型行为逐字节不变，再翻开关。Step 5、6 是唯一改**数学**的两步（其余是管线），放在最后单独验证。

| # | 文件 | 改动 |
|---|---|---|
| 0 | — | 双 vLLM 冒烟测试（阻塞后续） |
| 1 | `trainer/ppo/ray_trainer.py`, `trainer/main_ppo_new.py` | Role 枚举 + 注册 |
| 2 | `custom/fsdp_workers_new.py` | 角色常量化 + 新角色进 actor/rollout |
| 3 | `custom/config/rlplus_ppo_trainer.yaml` | 两个配置块 |
| 4 | `custom/new_ray_trainer.py` | `init_workers` + `_role_wg` + 断言 + 路由表 |
| 5 | `custom/new_ray_trainer.py` | SR 候选生成/logp → rephraser（数学） |
| 6 | `custom/new_ray_trainer.py` | `noprefix_logp` → reasoner（数学） |
| 7 | `custom/new_ray_trainer.py` | checkpoint override + `_validate_summarize` |
| 8 | `examples/custom/train_hype_summarize_2model.sh` | 新脚本 |
| 9 | `Myverl/tests/**` | 三个 CPU 测试 |

## 遗留的设计问题（本方案不动，供你决定）

1. **混合策略的 GRPO group**：SR normal step 里一个 group 含 7 条 reasoner rollout + 1 条 rephraser 行，baseline 取组均值（`norm_adv_by_std_in_grpo=False`）。跨策略组的均值 baseline 没有干净解释 —— 今天已如此，但解耦后那条注入行从「同模型的 prompt 变体」变成真正的外来样本。可能的修法：baseline 只在 on-policy 行上算。
2. **rephraser 学习信号稀疏**（本次选的 D1-a）：跑通后若发现 rephraser 几乎不动，再考虑「normal step 也训 rephraser」。
3. **rephraser 的 LR schedule**（Step 3d）：建议恒定 LR，待你确认。

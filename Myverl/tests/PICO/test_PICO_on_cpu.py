"""CPU checks for verl/PICO.

PICO is a trimmed copy of verl/custom, so the checks compare it against the original on
identical inputs:

  * ``PICORayPPOTrainer._summarize_replace`` vs ``NewRayPPOTrainer._summarize_replace_normal_step``
    with a fake joint decoder and a fake reward: the resulting batch must match on every key
    the old one writes (except ``target_probs``, which the joint path never reads).
  * ``new_core_alg.compute_token_on_off_sft_loss`` vs ``new_core_alg``'s on random inputs.
  * ``PICODataParallelPPOActor.update_policy`` vs ``NewDataParallelPPOActor.update_policy`` on a
    tiny model: same updated weights, returned log-probs and metrics.
  * ``PICOMathRewardManager`` vs ``MathRewardManager`` on a few responses.

Run:
    pytest tests/PICO/test_PICO_on_cpu.py -v
"""

from __future__ import annotations

import unittest
from contextlib import ExitStack
from copy import deepcopy
from unittest import mock

import numpy as np
import torch

try:
    from omegaconf import OmegaConf

    from verl import DataProto
    from verl.custom.new_ray_trainer import NewRayPPOTrainer
    from verl.PICO.new_ray_trainer import PICORayPPOTrainer, build_off_rows
    from verl.trainer.ppo.ray_trainer import RayPPOTrainer
except Exception as error:  # pragma: no cover - checkout without the training deps
    IMPORT_ERROR = error
else:
    IMPORT_ERROR = None

PAD = 0
GOOD = 7  # a response starting with this token is "correct"
N_QUESTIONS, N, L_SHORT, L_LONG, WIDTH = 6, 4, 5, 9, 12


class FakeTokenizer:
    pad_token_id = PAD
    eos_token_id = 2

    def batch_decode(self, ids, skip_special_tokens=True):
        # Rows whose second token is 9 read as referencing the draft, so the trajectory filter rejects them.
        return ["see the draft " * 20 if int(row[1]) == 9 else "a full solution " * 20 for row in ids]

    def decode(self, ids, skip_special_tokens=True):
        return self.batch_decode([ids])[0]


def fake_reward(data, return_dict=False):
    responses = data.batch["responses"]
    reward = torch.zeros(responses.shape, dtype=torch.float32)
    reward[:, 0] = (responses[:, 0] == GOOD).float()
    return {"reward_tensor": reward} if return_dict else reward


class FakeJointWorkerGroup:
    """Stands in for actor_rollout_wg.generate_joint: candidate j of a question is correct iff
    the question id (first real student-prompt token) plus j is even; candidates of odd
    questions additionally carry token 9 (rejected by the trajectory filter)."""

    world_size = 2

    def generate_joint(self, data):
        ids = data.batch["student_input_ids"]
        rows = ids.size(0)
        width = data.meta_info["max_response_length"]
        responses = torch.full((rows, width), PAD, dtype=torch.long)
        seen = {}
        for r in range(rows):
            q = int(ids[r][ids[r] != PAD][0])
            j = seen.get(q, 0)
            seen[q] = j + 1
            length = 3 + (r % 4)
            responses[r, :length] = 5
            responses[r, 0] = GOOD if (q + j) % 2 == 0 else 3
            responses[r, 1] = 9 if q % 2 else 5
        lengths = (responses != PAD).sum(-1)
        off_log_z = -0.1 * torch.arange(1, width + 1, dtype=torch.float32).repeat(rows, 1) * (responses != PAD)
        return DataProto.from_dict(
            tensors={
                "responses": responses,
                "off_log_z": off_log_z,
                "response_lengths": lengths,
                "fallback_frac": torch.zeros(rows),
                "teacher_keep_ratio": torch.ones(rows),
                "student_mean_logp": torch.zeros(rows),
            }
        )


def make_config(summarize_replace="wrong_only", select="shortest", max_k=2, trajectory_filter=False):
    return OmegaConf.create(
        {
            "data": {"max_response_length": WIDTH, "use_summarize": True},
            "algorithm": {"trajectory_filter": {"enable": trajectory_filter}},
            "actor_rollout_ref": {
                "rollout": {
                    "n": N,
                    "name": "vllm",
                    "temperature": 1.0,
                    "mode": "sync",
                    "summarize_replace": summarize_replace,
                    "summarize_replace_select": select,
                    "summarize_replace_k": 1,
                    "accuracy_threshold": 0.1,
                    "sr_logprob_prompt": "short",
                },
                "actor": {"policy_loss": {"loss_mode": "rl-rl", "off_policy_reshape": "clip_with_z"}},
            },
            "joint_decode": {
                "enable": True,
                "teacher_model_path": "/teacher",
                "temperature": 1.0,
                "batch_size": 2,
                "target_rows": 0,
                "max_k": max_k,
                "student_prompt": "short",
                "teacher_prompt": "long",
            },
        }
    )


def make_inputs():
    """gen_batch (repeated short prompts), gen_batch_output (rollouts), batch (per question)."""
    short = torch.full((N_QUESTIONS, L_SHORT), PAD, dtype=torch.long)
    for q in range(N_QUESTIONS):
        short[q, 2:] = torch.tensor([q + 10, 11, 12])  # first real token identifies the question
    long = torch.full((N_QUESTIONS, 1, L_LONG), PAD, dtype=torch.long)
    for q in range(N_QUESTIONS):
        long[q, 0, 3:] = q + 20

    gen_ids = short.repeat_interleave(N, dim=0)
    attn = (gen_ids != PAD).long()
    gen_batch = DataProto.from_dict(tensors={"input_ids": gen_ids, "attention_mask": attn, "position_ids": attn.cumsum(-1)})

    # Questions 0,1 all wrong; 2 one right (accuracy 0.25 > 0.1); 3,4,5 all wrong.
    responses = torch.full((N_QUESTIONS * N, WIDTH), PAD, dtype=torch.long)
    responses[:, :4] = 4
    responses[2 * N + 1, 0] = GOOD
    full = torch.cat([gen_ids, responses], dim=-1)
    full_attn = (full != PAD).long()
    gen_batch_output = DataProto.from_dict(
        tensors={
            "prompts": gen_ids,
            "responses": responses,
            "input_ids": full,
            "attention_mask": full_attn,
            "position_ids": full_attn.cumsum(-1),
            "prefix_mask": torch.zeros(responses.shape, dtype=torch.bool),
        }
    )
    batch = DataProto.from_dict(
        tensors={"summarize_input_ids": long},
        non_tensors={
            "data_source": np.array(["math"] * N_QUESTIONS, dtype=object),
            "reward_model": np.array([{"ground_truth": "1"}] * N_QUESTIONS, dtype=object),
        },
    )
    return gen_batch, gen_batch_output, batch


def bare(cls, config):
    trainer = object.__new__(cls)
    trainer.config = config
    trainer.tokenizer = FakeTokenizer()
    trainer.reward_fn = fake_reward
    trainer.actor_rollout_wg = FakeJointWorkerGroup()
    trainer.rephraser_wg = None
    return trainer


@unittest.skipIf(IMPORT_ERROR is not None, f"training deps unavailable: {IMPORT_ERROR}")
class TestSummarizeReplaceMatchesOldTrainer(unittest.TestCase):
    def run_both(self, config):
        gen_batch, gen_out, batch = make_inputs()
        old_metrics, new_metrics = {}, {}
        old = bare(NewRayPPOTrainer, config)._summarize_replace_normal_step(
            gen_batch=deepcopy(gen_batch),
            gen_batch_output=deepcopy(gen_out),
            batch=deepcopy(batch),
            train_batch_size=N_QUESTIONS,
            timing_raw={},
            metrics=old_metrics,
            logp_wg=None,
        )
        new = bare(PICORayPPOTrainer, config)._summarize_replace(deepcopy(gen_batch), deepcopy(gen_out), deepcopy(batch), {}, new_metrics)
        return old, new, old_metrics, new_metrics

    def assert_same(self, config):
        old, new, old_metrics, new_metrics = self.run_both(config)
        self.assertEqual(set(old.batch.keys()) - {"target_probs"}, set(new.batch.keys()))
        for key in new.batch.keys():
            if key == "position_ids":
                # PICO numbers the padding after a replaced response on, as vLLM does; the old
                # code froze it at the last position. Padding never reaches the model
                # (attention_mask 0), so only valid tokens have to agree. The test's fake
                # tokens contain no EOS, so the masks are unchanged by the EOS fix.
                valid = new.batch["attention_mask"].bool()
                self.assertTrue(torch.equal(old.batch[key][valid], new.batch[key][valid]), key)
                continue
            self.assertTrue(torch.equal(old.batch[key], new.batch[key]), key)
        for key, value in new_metrics.items():
            self.assertEqual(old_metrics[key], value, key)
        return new, new_metrics

    def test_wrong_only_shortest(self):
        new, metrics = self.assert_same(make_config())
        self.assertGreater(metrics["batch/sr_accepted"], 0)
        self.assertTrue(new.batch["prefix_mask"].any())
        self.assertTrue((new.batch["off_log_z"] < 0).any())

    def test_all_longest(self):
        self.assert_same(make_config(summarize_replace="all", select="longest"))

    def test_k1_with_question_cap(self):
        """max_k=1 and 5 failed questions over a 4-row budget: one question is dropped."""
        _, metrics = self.assert_same(make_config(max_k=1))
        self.assertEqual(metrics["batch/sr_joint_dropped"], 1)

    def test_trajectory_filter(self):
        _, metrics = self.assert_same(make_config(trajectory_filter=True))
        self.assertGreater(metrics["batch/sr_rej_filter"], 0)

    def test_spliced_row_keeps_eos(self):
        """When the decoder reports one more token than the non-pad ones (an EOS that shares the
        pad id), the spliced row's masks must cover it -- the old code dropped it."""

        class EosDecoder(FakeJointWorkerGroup):
            def generate_joint(self, data):
                reply = super().generate_joint(data)
                reply.batch["response_lengths"] = reply.batch["response_lengths"] + 1
                return reply

        gen_batch, gen_out, batch = make_inputs()
        trainer = bare(PICORayPPOTrainer, make_config())
        trainer.actor_rollout_wg = EosDecoder()
        out = trainer._summarize_replace(gen_batch, gen_out, batch, {}, {})
        replaced = out.batch["prefix_mask"].any(-1)
        self.assertTrue(replaced.any())
        tokens = (out.batch["responses"][replaced] != PAD).sum(-1)
        self.assertTrue(torch.equal(out.batch["prefix_mask"][replaced].sum(-1), tokens + 1))
        self.assertTrue(torch.equal(out.batch["attention_mask"][replaced][:, L_SHORT:].sum(-1), tokens + 1))


@unittest.skipIf(IMPORT_ERROR is not None, f"training deps unavailable: {IMPORT_ERROR}")
class TestHelpers(unittest.TestCase):
    def test_build_off_rows_pads_and_masks(self):
        prompts = torch.tensor([[0, 0, 5, 6], [0, 7, 8, 9]])
        responses = torch.tensor([[3, 4, 0], [3, 0, 0]])
        rows = build_off_rows(prompts, responses, torch.tensor([2, 1]), width=5, pad_token_id=0, mask_dtype=torch.long)
        self.assertEqual(rows["responses"].shape, (2, 5))
        self.assertEqual(rows["prefix_mask"].sum(-1).tolist(), [2, 1])
        self.assertEqual(rows["position_ids"][0].tolist(), [0, 0, 0, 1, 2, 3, 4, 5, 6])

    def test_build_off_rows_keeps_eos_equal_to_pad(self):
        """Qwen3's EOS and pad are both <|endoftext|>. The EOS must stay in the masks, as it does
        for vLLM rows (get_response_mask keeps the first EOS); position_ids must match vLLM's."""
        from verl.utils.torch_functional import get_response_mask

        pad = eos = 0
        prompts = torch.tensor([[pad, pad, 5, 6], [pad, 7, 8, 9]])
        # row 0: two tokens then EOS; row 1: ran to the width without one
        responses = torch.tensor([[3, 4, eos, pad, pad], [3, 4, 5, 6, 7]])
        rows = build_off_rows(prompts, responses, torch.tensor([3, 5]), width=5, pad_token_id=pad, mask_dtype=torch.long)
        vllm_resp_mask = get_response_mask(response_id=responses, eos_token=eos, dtype=torch.long)
        self.assertEqual(rows["prefix_mask"].long().tolist(), vllm_resp_mask.tolist())
        self.assertEqual(rows["attention_mask"][:, 4:].tolist(), vllm_resp_mask.tolist())
        self.assertEqual(rows["attention_mask"][:, :4].tolist(), (prompts != pad).long().tolist())
        prompt_pos = (prompts != pad).long().cumsum(-1).sub(1).clamp(min=0)
        self.assertEqual(rows["position_ids"].tolist(), torch.cat([prompt_pos, prompt_pos[:, -1:] + torch.arange(1, 6)], dim=-1).tolist())


@unittest.skipIf(IMPORT_ERROR is not None, f"training deps unavailable: {IMPORT_ERROR}")
class TestValidateConfig(unittest.TestCase):
    def check(self, config):
        trainer = bare(PICORayPPOTrainer, config)
        with mock.patch.object(RayPPOTrainer, "_validate_config"):
            trainer._validate_config()

    def full_config(self, **overrides):
        config = make_config()
        config.reward_model = {"launch_reward_fn_async": False}
        config.algorithm.adv_estimator = "grpo"
        for dotted, value in overrides.items():
            OmegaConf.update(config, dotted, value)
        return config

    def test_script_config_passes(self):
        self.check(self.full_config())

    def test_sft_loss_modes_pass(self):
        for mode in ("rl-sft", "sft-sft", "none-sft"):
            self.check(self.full_config(**{"actor_rollout_ref.actor.policy_loss.loss_mode": mode}))

    def test_stock_loss_mode_passes(self):
        """vanilla switches back to plain GRPO; it is allowed, only announced."""
        self.check(self.full_config(**{"actor_rollout_ref.actor.policy_loss.loss_mode": "vanilla"}))

    def test_rejects(self):
        for dotted, value in {
            "actor_rollout_ref.rollout.summarize_replace": "off",
            "actor_rollout_ref.rollout.summarize_replace_select": "logp",
            "joint_decode.enable": False,
            "algorithm.adv_estimator": "remax",
            "reward_model.launch_reward_fn_async": True,
        }.items():
            with self.subTest(dotted), self.assertRaises(ValueError):
                self.check(self.full_config(**{dotted: value}))


@unittest.skipIf(IMPORT_ERROR is not None, f"training deps unavailable: {IMPORT_ERROR}")
class TestLossMatchesOld(unittest.TestCase):
    """PICO's compute_token_on_off_sft_loss is a trimmed rewrite of verl/custom's. For every
    option it kept it must give the same loss, gradient and metrics -- bar on_pg_clipfrac, which
    the old one hard-coded to 0 and which must now match stock verl's pg_clipfrac."""

    def setUp(self):
        torch.manual_seed(0)
        rows, width = 8, 16
        self.old_log_prob = -torch.rand(rows, width) * 3
        self.response_mask = (torch.arange(width) < torch.randint(4, width + 1, (rows, 1))).float()
        self.prefix_mask = torch.zeros(rows, width, dtype=torch.bool)
        self.prefix_mask[[1, 5]] = self.response_mask[[1, 5]].bool()
        self.off_log_z = torch.where(self.prefix_mask, -torch.rand(rows, width), torch.zeros(rows, width))
        self.advantages = torch.randn(rows, 1).expand(rows, width)
        self.log_prob = self.old_log_prob + 0.3 * torch.randn(rows, width)

    def kwargs(self, mode, reshape, all_max_clip, off_log_z):
        return dict(
            old_log_prob=self.old_log_prob,
            advantages=self.advantages,
            response_mask=self.response_mask,
            cliprange=0.28,
            cliprange_low=0.28,
            cliprange_high=0.28,
            prefix_mask=self.prefix_mask,
            reward_mask=torch.ones_like(self.old_log_prob),
            all_max_clip=all_max_clip,
            off_policy_strategy=mode,
            off_policy_reshape=reshape,
            off_log_z=off_log_z,
        )

    def test_same_outputs(self):
        from verl.custom.new_core_alg import compute_token_on_off_sft_loss as old_loss
        from verl.PICO.new_core_alg import compute_token_on_off_sft_loss as new_loss

        cases = [("rl-rl", r) for r in ("clip_with_z", "clip_without_z", "clip", "p_div_p_0.1")]
        # sft-rl / none-rl are left out: the old code matched only these four names and silently
        # ran anything else as rl-sft.
        cases += [("rl-sft", "clip"), ("sft-sft", "clip"), ("none-sft", "clip")]
        for mode, reshape in cases:
            for all_max_clip in (10, None):
                for with_z in (True, False):
                    with self.subTest(mode=mode, reshape=reshape, all_max_clip=all_max_clip, with_z=with_z):
                        kwargs = self.kwargs(mode, reshape, all_max_clip, self.off_log_z if with_z else None)
                        lp = self.log_prob.clone().requires_grad_(True)
                        new = new_loss(log_prob=lp, **kwargs)
                        (new_grad,) = torch.autograd.grad(new["pg_loss"], lp)
                        lp = self.log_prob.clone().requires_grad_(True)
                        old = old_loss(log_prob=lp, se_mask=None, **kwargs)
                        (old_grad,) = torch.autograd.grad(old["pg_loss"], lp)
                        self.assertTrue(torch.equal(new_grad, old_grad))
                        for key, value in new.items():
                            if key == "on_pg_clipfrac":
                                continue
                            self.assertTrue(torch.equal(value, old[key]), key)

    def test_on_pg_clipfrac_matches_stock(self):
        """With no replaced rows, the on rows are all rows: PICO's rl on-loss and its clipfrac
        must equal stock compute_policy_loss."""
        from verl.PICO.new_core_alg import compute_token_on_off_sft_loss
        from verl.trainer.ppo.core_algos import compute_policy_loss

        kwargs = self.kwargs("rl-rl", "clip_with_z", None, None)
        kwargs["prefix_mask"] = torch.zeros_like(self.prefix_mask)
        new = compute_token_on_off_sft_loss(log_prob=self.log_prob, **kwargs)
        pg_loss, pg_clipfrac, ppo_kl, _ = compute_policy_loss(self.old_log_prob, self.log_prob, self.advantages, self.response_mask, cliprange=0.28, cliprange_low=0.28, cliprange_high=0.28)
        self.assertGreater(pg_clipfrac.item(), 0)
        self.assertTrue(torch.allclose(new["on_pg_clipfrac"], pg_clipfrac))
        self.assertTrue(torch.allclose(new["pg_loss"], pg_loss))
        self.assertTrue(torch.allclose(new["ppo_kl"], ppo_kl))

    def test_rejects_removed_options(self):
        from verl.PICO.new_core_alg import compute_token_on_off_sft_loss

        for mode, reshape in (("rl-rl", "batch_mean_norm"), ("rl-rl", "low_clip_0.1"), ("rl-sft", "weighted_sft"), ("rl-sft", "low_sft_other_rl")):
            with self.subTest(reshape=reshape), self.assertRaises(AssertionError):
                compute_token_on_off_sft_loss(log_prob=self.log_prob, **self.kwargs(mode, reshape, None, None))


class _TinyLM(torch.nn.Module):
    """Stands in for the HF model: embedding + linear head, returns ``.logits``."""

    def __init__(self, vocab=11, hidden=8):
        super().__init__()
        self.emb = torch.nn.Embedding(vocab, hidden)
        self.head = torch.nn.Linear(hidden, vocab)

    def forward(self, input_ids, attention_mask=None, position_ids=None, use_cache=False, **kwargs):
        from types import SimpleNamespace

        return SimpleNamespace(logits=self.head(self.emb(input_ids)))


@unittest.skipIf(IMPORT_ERROR is not None, f"training deps unavailable: {IMPORT_ERROR}")
class TestActorMatchesOld(unittest.TestCase):
    """PICO's update_policy on one batch, from identical weights:

    * against verl/custom's with the script's settings (rl-rl, clip_with_z, dynamic bsz,
      all_max_clip=10). The script's other actor flags are now optional with the same defaults,
      so the PICO side runs without them.
    * against stock verl's with loss_mode=vanilla, which must be plain GRPO.

    Updated weights, returned log-probs and every metric the reference reports must match."""

    @staticmethod
    def cpu_patches():
        return [
            mock.patch("verl.workers.actor.dp_actor.get_device_id", return_value="cpu"),
            mock.patch("verl.PICO.new_dp_actor.get_device_id", return_value="cpu"),
            mock.patch("verl.custom.new_dp_actor.get_device_id", return_value="cpu"),
            # GPUMemoryLogger reads CUDA memory stats.
            mock.patch("verl.utils.debug.performance._get_current_mem_info", return_value=("0", "0", "0", "0")),
            # flash-attn's triton cross entropy is CUDA-only; use the plain torch path.
            mock.patch("verl.utils.torch_functional.FLAH_ATTN_CROSS_ENTROPY_LOSS_AVAILABLE", False),
        ]

    def run_actor(self, cls, actor_cfg, data):
        torch.manual_seed(0)
        model = _TinyLM()
        actor = cls(config=actor_cfg, actor_module=model, actor_optimizer=torch.optim.SGD(model.parameters(), lr=0.5))
        actor.device_name = "cpu"
        with ExitStack() as stack:
            for patch in self.cpu_patches():
                stack.enter_context(patch)
            metrics, tensors = actor.update_policy(deepcopy(data))
        return model, metrics, tensors["log_prob"]

    def setUp(self):
        import torch.distributed as dist

        if not dist.is_initialized():
            dist.init_process_group("gloo", init_method="tcp://127.0.0.1:29571", rank=0, world_size=1)

    @staticmethod
    def make_data():
        torch.manual_seed(1)
        rows, prompt_len, resp_len, vocab = 8, 3, 6, 11
        responses = torch.randint(1, vocab, (rows, resp_len))
        resp_attn = (torch.arange(resp_len) < torch.randint(2, resp_len + 1, (rows, 1))).long()
        input_ids = torch.cat([torch.randint(1, vocab, (rows, prompt_len)), responses], dim=-1)
        attention_mask = torch.cat([torch.ones(rows, prompt_len, dtype=torch.long), resp_attn], dim=-1)
        prefix_mask = torch.zeros(rows, resp_len, dtype=torch.bool)
        prefix_mask[[2, 6]] = resp_attn[[2, 6]].bool()
        return DataProto.from_dict(
            tensors={
                "responses": responses,
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "position_ids": (attention_mask.cumsum(-1) - 1).clamp(min=0),
                "old_log_probs": -torch.rand(rows, resp_len) * 2,
                "advantages": torch.randn(rows, 1).expand(rows, resp_len).contiguous(),
                "prefix_mask": prefix_mask,
                "se_mask": torch.zeros(rows, resp_len, dtype=torch.bool),
                "reward_mask": torch.ones(rows, resp_len, dtype=torch.bool),
                "off_log_z": torch.where(prefix_mask, -torch.rand(rows, resp_len), torch.zeros(rows, resp_len)),
            },
            meta_info={"temperature": 1.0},
        )

    BASE = {
        "ppo_mini_batch_size": 4,
        "ppo_micro_batch_size_per_gpu": 2,
        "ppo_epochs": 1,
        "use_dynamic_bsz": True,
        "ppo_max_token_len_per_gpu": 20,
        "clip_ratio": 0.28,
        "clip_ratio_low": None,
        "clip_ratio_high": None,
        "entropy_coeff": 0,
        "loss_agg_mode": "token-mean",
        "use_kl_loss": False,
        "grad_clip": 1.0,
        "ulysses_sequence_parallel_size": 1,
        "entropy_from_logits_with_chunking": False,
        "entropy_checkpointing": False,
        "use_torch_compile": False,
        "use_remove_padding": False,
    }

    def assert_same(self, old, new):
        old_model, old_metrics, old_lp = old
        new_model, new_metrics, new_lp = new
        for (name, p_old), (_, p_new) in zip(old_model.named_parameters(), new_model.named_parameters()):
            self.assertTrue(torch.equal(p_old, p_new), name)
        if old_lp is not None:
            self.assertTrue(torch.equal(old_lp, new_lp))
        # Metrics PICO stopped logging: constants for every option it kept (the hard clips and
        # off-distillation are gone, off_ratio_scale only moved under batch_mean_norm /
        # group_ess_weight / low_clip).
        removed = {"actor/off_ratio_max_clip_frac": 0.0, "actor/off_ratio_min_clip_frac": 0.0, "actor/off_ratio_scale": 1.0, "actor/off_distill_loss": 0.0}
        for key, value in old_metrics.items():
            # The old loss hard-coded on_pg_clipfrac to 0; PICO computes it (checked against
            # stock verl in TestLossMatchesOld).
            if key == "actor/on_pg_clipfrac":
                continue
            if key in removed:
                self.assertTrue(all(v == removed[key] for v in value), key)
                continue
            self.assertIn(key, new_metrics)
            self.assertEqual(value, new_metrics[key], key)

    def test_same_update(self):
        from verl.custom.new_dp_actor import NewDataParallelPPOActor
        from verl.PICO.new_dp_actor import PICODataParallelPPOActor

        data = self.make_data()
        base = {**self.BASE, "all_max_clip": 10, "policy_loss": {"loss_mode": "rl-rl", "off_policy_reshape": "clip_with_z"}}
        old_cfg = OmegaConf.create({**base, "off_policy_max_clip": -1, "off_policy_min_clip": -1, "loss_remove_token_mean": False, "loss_remove_clip": False, "use_off_policy_loss": False, "off_policy_loss_impl": "token", "use_off_policy_probs": False})
        self.assert_same(
            self.run_actor(NewDataParallelPPOActor, old_cfg, data),
            self.run_actor(PICODataParallelPPOActor, OmegaConf.create(base), data),
        )

    def test_vanilla_is_stock_grpo(self):
        """loss_mode=vanilla must train exactly like stock verl's actor, prefix_mask / off_log_z
        notwithstanding."""
        from verl.PICO.new_dp_actor import PICODataParallelPPOActor
        from verl.workers.actor.dp_actor import DataParallelPPOActor

        data = self.make_data()
        cfg = OmegaConf.create({**self.BASE, "policy_loss": {"loss_mode": "vanilla"}})
        stock_model, stock_metrics, _ = self.run_actor_stock(DataParallelPPOActor, cfg, data)
        self.assert_same((stock_model, stock_metrics, None), self.run_actor(PICODataParallelPPOActor, cfg, data))

    def run_actor_stock(self, cls, actor_cfg, data):
        """Stock update_policy returns metrics only."""
        torch.manual_seed(0)
        model = _TinyLM()
        actor = cls(config=actor_cfg, actor_module=model, actor_optimizer=torch.optim.SGD(model.parameters(), lr=0.5))
        actor.device_name = "cpu"
        with ExitStack() as stack:
            for patch in self.cpu_patches():
                stack.enter_context(patch)
            metrics = actor.update_policy(deepcopy(data))
        return model, metrics, None


@unittest.skipIf(IMPORT_ERROR is not None, f"training deps unavailable: {IMPORT_ERROR}")
class TestDynamicTopK(unittest.TestCase):
    """agree_select with dynamic_top_k: fixed intersection first; rows below target_z widen the
    teacher's k up to the cap; rows still short there fall back."""

    TARGET = 0.6

    def setUp(self):
        # vocab of 6. The teacher ranks tokens 0 > 1 > 2 > 3 > 4 > 5 in every row.
        log_pb = torch.log(torch.tensor([[0.40, 0.25, 0.15, 0.10, 0.06, 0.04]] * 4))
        # Student mass per row:
        #   row 0 -- agrees with the teacher: fixed top-2 already keeps 0.9 >= target
        #   row 1 -- top-2 keeps 0.2; ranks 3-4 bring it to 0.7 >= target at k=4
        #   row 2 -- mass sits on token 5, beyond the cap of 4: still short at the cap
        #   row 3 -- top-2 keeps 0.1; k=3 reaches 0.75
        p_a = torch.tensor(
            [
                [0.50, 0.40, 0.04, 0.03, 0.02, 0.01],
                [0.10, 0.10, 0.20, 0.30, 0.20, 0.10],
                [0.05, 0.05, 0.05, 0.05, 0.05, 0.75],
                [0.05, 0.05, 0.65, 0.10, 0.10, 0.05],
            ]
        )
        self.logits_a, self.logits_b = torch.log(p_a), log_pb
        self.p_a = p_a

    def select(self, dynamic, fallback="student", teacher_min_prob=0.0):
        from verl.PICO.joint_decode_core import agree_select

        torch.manual_seed(0)
        return agree_select(
            self.logits_a,
            self.logits_b,
            student_top_k=6,
            teacher_top_k=2,
            student_min_prob=0.0,
            teacher_min_prob=teacher_min_prob,
            temperature=1.0,
            top_p=1.0,
            fallback=fallback,
            dynamic_top_k=dynamic,
            target_z=self.TARGET,
            dynamic_max_top_k=4,
        )

    def test_rows(self):
        _, fell_back, _, z_t, _ = self.select(dynamic=True)
        # row 0 keeps the fixed set: Z_t = p(0) + p(1)
        self.assertAlmostEqual(z_t[0].item(), 0.9, places=5)
        # row 1 widens to k=4, the smallest k at or above target
        self.assertAlmostEqual(z_t[1].item(), 0.7, places=5)
        # row 3 widens to k=3
        self.assertAlmostEqual(z_t[3].item(), 0.75, places=5)
        # row 2 is still short at the cap: it falls back, Z_t reported as 0
        self.assertEqual(fell_back.tolist(), [False, False, True, False])
        self.assertEqual(z_t[2].item(), 0.0)
        # every non-fallback row keeps at least the target
        self.assertTrue((z_t[~fell_back] >= self.TARGET).all())

    def test_fixed_path_unchanged(self):
        """dynamic_top_k=False is the plain fixed intersection, no fallback here."""
        _, fell_back, _, z_t, _ = self.select(dynamic=False)
        self.assertFalse(fell_back.any())
        expected = self.p_a[:, :2].sum(-1)
        self.assertTrue(torch.allclose(z_t, expected))

    def test_fixed_floor_applies_to_first_set(self):
        """teacher_min_prob bounds the fixed set: at 0.3 only token 0 (0.40) clears it, so row 0
        keeps 0.5 < target and widens to k=2 (0.9)."""
        _, fell_back, _, z_t, _ = self.select(dynamic=True, teacher_min_prob=0.3)
        self.assertFalse(fell_back[0].item())
        self.assertAlmostEqual(z_t[0].item(), 0.9, places=5)

    def test_fallback_samples_from_full_distribution(self):
        """Row 2 falls back: under 'student' its density is the full student distribution, so
        log Z_t = log p_student - log mu is 0 there."""
        chosen, fell_back, log_pa, _, log_q = self.select(dynamic=True, fallback="student")
        step_logp = log_pa.gather(-1, chosen).squeeze(-1)
        self.assertTrue(fell_back[2].item())
        self.assertAlmostEqual((step_logp[2] - log_q[2]).item(), 0.0, places=5)
        # a constrained row's log Z_t is log of its kept mass
        self.assertAlmostEqual((step_logp[1] - log_q[1]).item(), torch.log(torch.tensor(0.7)).item(), places=5)


@unittest.skipIf(IMPORT_ERROR is not None, f"training deps unavailable: {IMPORT_ERROR}")
class TestRewardMatchesOld(unittest.TestCase):
    def test_same_scores(self):
        # Through the package, as training does: importing verl.custom.math_reward_manager
        # first is circular (it imports verl.workers.reward_manager, which imports it back).
        from verl.PICO.math_reward_manager import PICOMathRewardManager
        from verl.workers.reward_manager import MathRewardManager

        texts = ["so the answer is \\boxed{57}", "the answer is \\boxed{56}", "<think>x</think> \\boxed{57}", "no answer"]

        class CharTokenizer:
            def decode(self, ids):
                return "".join(chr(int(i)) for i in ids if int(i) != 0)

        width = max(len(t) for t in texts)
        responses = torch.zeros(len(texts), width, dtype=torch.long)
        for i, t in enumerate(texts):
            responses[i, : len(t)] = torch.tensor([ord(c) for c in t])
        prompts = torch.ones(len(texts), 3, dtype=torch.long)
        data = DataProto.from_dict(
            tensors={"prompts": prompts, "responses": responses, "attention_mask": torch.cat([prompts, (responses != 0).long()], dim=-1)},
            non_tensors={
                "data_source": np.array(["math"] * len(texts), dtype=object),
                "reward_model": np.array([{"ground_truth": "57"}] * len(texts), dtype=object),
            },
        )
        for version in (3, 4):
            with self.subTest(reward_impl_version=version):
                old = MathRewardManager(CharTokenizer(), 0, version)(data)
                new = PICOMathRewardManager(CharTokenizer(), 0, version)(data)
                self.assertTrue(torch.equal(old, new))
                self.assertGreater(new.sum().item(), 0)


if __name__ == "__main__":
    unittest.main()

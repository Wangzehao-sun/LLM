"""The rephraser config merge: overrides apply, the rest inherits, nothing leaks back.

``init_workers`` builds the rephraser's config as

    OmegaConf.merge(deepcopy(actor_rollout_ref), actor_rollout_rephraser)

Three properties have to hold, and two of them fail SILENTLY when broken:

1. **Overrides win.** Otherwise the rephraser loads the reasoner's weights and the whole
   split is a no-op that still costs a second engine's worth of GPU memory.
2. **Unspecified keys inherit.** Max lengths, dtype and remove_padding must match, since
   the two models exchange raw token ids without re-tokenizing.
3. **The reasoner's node is untouched.** This is why the ``deepcopy`` is there rather
   than being defensive: the worker's ``__init__`` mutates
   ``config.actor.ppo_mini_batch_size`` in place, so a shared node gets normalized once
   per worker group -- halving the mini-batch of whichever is constructed second, with no
   error anywhere.

Skips when omegaconf is unavailable so a bare CPU checkout stays green.

Run:
    pytest Myverl/tests/trainer/test_rephraser_config_on_cpu.py -v
"""

from __future__ import annotations

import unittest

try:
    from omegaconf import OmegaConf
except ImportError:  # pragma: no cover - CPU-only checkout without omegaconf
    OmegaConf = None


def base_config():
    """A cut-down actor_rollout_ref carrying the keys the merge has to get right."""
    return OmegaConf.create(
        {
            "model": {"path": "/models/reasoner", "use_remove_padding": True},
            "actor": {
                "ppo_mini_batch_size": 64,
                "ppo_micro_batch_size": 64,
                "optim": {"lr": 1e-6},
                "fsdp_config": {"param_offload": False, "optimizer_offload": False},
            },
            "rollout": {
                "gpu_memory_utilization": 0.7,
                "temperature": 1.0,
                "prompt_length": 2048,
                "response_length": 14336,
                "n": 8,
            },
            "ref": {"fsdp_config": {"param_offload": True}},
        }
    )


def merge(base, override):
    """Reproduce init_workers' merge exactly, deepcopy included."""
    from copy import deepcopy

    return OmegaConf.merge(deepcopy(base), override)


@unittest.skipIf(OmegaConf is None, "omegaconf not installed")
class TestRephraserConfigMerge(unittest.TestCase):
    def test_empty_override_is_a_no_op(self):
        """This is what makes rephraser.enable=False byte-identical to before."""
        base = base_config()
        merged = merge(base, OmegaConf.create({}))
        self.assertEqual(OmegaConf.to_container(merged), OmegaConf.to_container(base))

    def test_overrides_take_effect(self):
        base = base_config()
        merged = merge(
            base,
            OmegaConf.create(
                {
                    "model": {"path": "/models/rephraser-sft"},
                    "actor": {"fsdp_config": {"param_offload": True}},
                    "rollout": {"gpu_memory_utilization": 0.42},
                }
            ),
        )
        self.assertEqual(merged.model.path, "/models/rephraser-sft")
        self.assertTrue(merged.actor.fsdp_config.param_offload)
        self.assertAlmostEqual(merged.rollout.gpu_memory_utilization, 0.42)

    def test_unspecified_keys_inherit(self):
        """The shared-tokenizer constraint depends on these staying equal.

        summarize_input_ids are pre-tokenized offline with the reasoner's tokenizer, and
        the rephraser's rollout token ids are spliced into the reasoner's input_ids with
        no re-tokenization, so lengths and padding behaviour must not diverge.
        """
        base = base_config()
        merged = merge(base, OmegaConf.create({"model": {"path": "/models/rephraser-sft"}}))
        self.assertEqual(merged.rollout.prompt_length, base.rollout.prompt_length)
        self.assertEqual(merged.rollout.response_length, base.rollout.response_length)
        self.assertEqual(merged.rollout.n, base.rollout.n)
        self.assertEqual(merged.model.use_remove_padding, base.model.use_remove_padding)
        self.assertEqual(merged.actor.optim.lr, base.actor.optim.lr)

    def test_partial_override_of_a_nested_node_keeps_its_siblings(self):
        """Setting param_offload must not wipe optimizer_offload out of the same node."""
        base = base_config()
        merged = merge(base, OmegaConf.create({"actor": {"fsdp_config": {"param_offload": True}}}))
        self.assertTrue(merged.actor.fsdp_config.param_offload)
        self.assertIn("optimizer_offload", merged.actor.fsdp_config)
        self.assertFalse(merged.actor.fsdp_config.optimizer_offload)

    def test_base_config_is_not_mutated_by_the_merge(self):
        base = base_config()
        merge(base, OmegaConf.create({"model": {"path": "/models/rephraser-sft"}, "rollout": {"gpu_memory_utilization": 0.42}}))
        self.assertEqual(base.model.path, "/models/reasoner")
        self.assertAlmostEqual(base.rollout.gpu_memory_utilization, 0.7)

    def test_mutating_the_merged_config_does_not_reach_the_base(self):
        """The failure the deepcopy actually prevents.

        The worker rewrites ppo_mini_batch_size in place during __init__. Without the
        deepcopy that write would land on the shared node and the reasoner's worker would
        normalize an already-normalized value -- a quietly halved mini-batch.
        """
        base = base_config()
        merged = merge(base, OmegaConf.create({"model": {"path": "/models/rephraser-sft"}}))
        merged.actor.ppo_mini_batch_size = 999
        merged.actor.fsdp_config.param_offload = True
        self.assertEqual(base.actor.ppo_mini_batch_size, 64)
        self.assertFalse(base.actor.fsdp_config.param_offload)


if __name__ == "__main__":
    unittest.main()

"""The rephraser's resource pool: shared vs dedicated, and what each implies.

`rephraser.n_gpus_per_node` is not a memory knob. verl merges every worker of a resource
POOL into one Ray actor process (`create_colocated_worker_cls`), and vLLM's sleep mode is
built on a PROCESS-GLOBAL `CuMemAllocator` whose tags are the hardcoded strings
"weights"/"kv_cache" -- so two engines in one process free each other's memory. A separate
pool is therefore the difference between "the rephraser can generate" and "it cannot".

Three properties, all of which fail silently when broken:

1. **n_gpus_per_node=0 must leave the spec exactly as it is today.** A stray second pool
   would reserve GPUs Ray then cannot give the reasoner -- and the failure would look like
   an unrelated resource error.
2. **n_gpus_per_node>0 must ADD a pool without touching the reasoner's.** The reasoner's
   entry is what `trainer.n_gpus_per_node` sizes; shrinking it would silently change the
   divisibility `_validate_config` checks.
3. **Every role must map to a pool that exists.** `get_resource_pool` does a bare dict
   lookup, so a missing key is a KeyError deep inside worker construction.

The spec construction is re-implemented here rather than imported: importing
`main_ppo_new` pulls in ray, torch and the whole verl stack, which is not available in a
CPU-only environment. The source is checked textually alongside, so drift between this
model and the real code is caught rather than assumed away.

Run:
    pytest Myverl/tests/trainer/test_resource_pool_split_on_cpu.py -v
"""

from __future__ import annotations

import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]  # .../LLM/Myverl
MAIN_PPO = REPO_ROOT / "verl" / "trainer" / "main_ppo_new.py"
PPO_CONFIG = REPO_ROOT / "verl" / "trainer" / "config" / "ppo_trainer.yaml"

GLOBAL_POOL = "global_pool"
REPHRASER_POOL = "rephraser_pool"


def build_pools(n_gpus_per_node, nnodes=1, rephraser_enable=False, rephraser_gpus=0):
    """Mirror of the spec/mapping construction in main_ppo_new.TaskRunner.run.

    Returns ``(resource_pool_spec, mapping)`` keyed by role NAME (the real code keys by
    the Role enum, which needs the verl import).
    """
    resource_pool_spec = {GLOBAL_POOL: [n_gpus_per_node] * nnodes}
    mapping = {"ActorRollout": GLOBAL_POOL, "Critic": GLOBAL_POOL}

    if rephraser_enable:
        if rephraser_gpus > 0:
            resource_pool_spec[REPHRASER_POOL] = [rephraser_gpus] * nnodes
            mapping["ActorRolloutRephraser"] = REPHRASER_POOL
        else:
            mapping["ActorRolloutRephraser"] = GLOBAL_POOL
    return resource_pool_spec, mapping


class TestResourcePoolSplit(unittest.TestCase):
    def test_disabled_is_identical_to_a_single_pool(self):
        """Every existing script must be unaffected until the feature is turned on."""
        spec, mapping = build_pools(4)
        self.assertEqual(spec, {GLOBAL_POOL: [4]})
        self.assertEqual(set(mapping.values()), {GLOBAL_POOL})

    def test_shared_pool_adds_no_pool(self):
        """enable=True with n_gpus_per_node=0 keeps ONE pool -- the logprob-only setup."""
        spec, mapping = build_pools(4, rephraser_enable=True, rephraser_gpus=0)
        self.assertEqual(spec, {GLOBAL_POOL: [4]}, "a shared-pool rephraser must not reserve GPUs")
        self.assertEqual(mapping["ActorRolloutRephraser"], GLOBAL_POOL)

    def test_dedicated_pool_is_additive(self):
        """The reasoner's entry must be untouched -- it is what n_gpus_per_node sizes."""
        spec, mapping = build_pools(4, rephraser_enable=True, rephraser_gpus=1)
        self.assertEqual(spec[GLOBAL_POOL], [4], "the reasoner's pool must not shrink")
        self.assertEqual(spec[REPHRASER_POOL], [1])
        self.assertEqual(mapping["ActorRolloutRephraser"], REPHRASER_POOL)
        self.assertEqual(mapping["ActorRollout"], GLOBAL_POOL)

    def test_total_gpus_is_the_sum_of_both_pools(self):
        """get_n_gpus() sums the spec; it feeds the throughput metrics."""
        spec, _ = build_pools(4, rephraser_enable=True, rephraser_gpus=2)
        total = sum(n for per_node in spec.values() for n in per_node)
        self.assertEqual(total, 6, "CUDA_VISIBLE_DEVICES must expose this many GPUs")

    def test_multi_node_replicates_per_node(self):
        """Both pools are [gpus] * nnodes, so a 2-node run doubles each."""
        spec, _ = build_pools(4, nnodes=2, rephraser_enable=True, rephraser_gpus=1)
        self.assertEqual(spec[GLOBAL_POOL], [4, 4])
        self.assertEqual(spec[REPHRASER_POOL], [1, 1])

    def test_every_mapped_pool_exists(self):
        """get_resource_pool does a bare dict lookup: a missing key is a late KeyError."""
        for gpus in (0, 1, 2):
            spec, mapping = build_pools(4, rephraser_enable=True, rephraser_gpus=gpus)
            for role, pool in mapping.items():
                self.assertIn(pool, spec, f"role {role} maps to pool {pool!r}, which is not in the spec")


class TestSourceMatchesTheModel(unittest.TestCase):
    """Guards against this file's build_pools drifting from the real construction."""

    @classmethod
    def setUpClass(cls):
        cls.src = MAIN_PPO.read_text(encoding="utf-8")

    def test_pool_is_added_only_when_gpus_are_requested(self):
        self.assertIn('rephraser_pool_id = "rephraser_pool"', self.src)
        self.assertIn("if rephraser_gpus > 0:", self.src)
        # The additive form. A plain assignment to resource_pool_spec would replace the
        # reasoner's entry instead of adding alongside it.
        self.assertIn("resource_pool_spec[rephraser_pool_id] =", self.src)

    def test_registration_is_still_gated_on_enable(self):
        self.assertIn('config.get("rephraser", {}).get("enable", False)', self.src)

    def test_config_default_keeps_the_single_pool(self):
        """n_gpus_per_node must default to 0, or every existing run gains a pool."""
        import yaml

        cfg = yaml.safe_load(PPO_CONFIG.read_text(encoding="utf-8"))
        self.assertIn("rephraser", cfg)
        self.assertIs(cfg["rephraser"]["enable"], False)
        self.assertEqual(cfg["rephraser"]["n_gpus_per_node"], 0)


if __name__ == "__main__":
    unittest.main()

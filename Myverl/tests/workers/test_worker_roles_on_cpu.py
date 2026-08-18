"""Role -> capability flags for the worker, and the rephraser config merge.

Two invariants that would otherwise only be observable after a multi-GPU job has
started, and whose violations are silent rather than loud:

1. **The frozen rephraser must be neither an actor nor a rollout.** Not an actor is what
   freezes it: ``update_actor`` and ``save_checkpoint`` both ``assert self._is_actor``, so
   a role outside ``_ACTOR_ROLES`` cannot be trained or saved even by mistake. Not a
   rollout is what lets it exist at all: a second vLLM engine in one process shares a
   global ``CuMemAllocator``, so one engine's ``sleep()`` frees the other's memory.
   Adding it to either tuple breaks something with no test failing. Hence this.

2. **The rephraser's config must be a private copy.** The worker's ``__init__`` mutates
   ``config.actor.ppo_mini_batch_size`` in place, so a config node shared with the
   reasoner would be normalized twice -- halving the mini-batch of whichever worker is
   built second, with no error.

The flag constants are read with ``ast`` rather than imported: importing
``fsdp_workers_new`` pulls in torch, vllm and the whole verl worker stack, which is not
available in a CPU-only environment. Reading the literals keeps this test runnable
anywhere, which is the point of a ``*_on_cpu.py`` test.

Run:
    pytest Myverl/tests/workers/test_worker_roles_on_cpu.py -v
"""

from __future__ import annotations

import ast
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]  # .../LLM/Myverl
WORKERS_FILE = REPO_ROOT / "verl" / "custom" / "fsdp_workers_new.py"
TRAINER_FILE = REPO_ROOT / "verl" / "custom" / "new_ray_trainer.py"
PPO_CONFIG = REPO_ROOT / "verl" / "trainer" / "config" / "ppo_trainer.yaml"
MAIN_PPO = REPO_ROOT / "verl" / "trainer" / "main_ppo_new.py"
RAY_TRAINER = REPO_ROOT / "verl" / "trainer" / "ppo" / "ray_trainer.py"


def module_constants(path: Path, prefix: str = "_") -> dict:
    """Evaluate a module's top-level literal assignments without importing it."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    found = {}
    for node in tree.body:
        if not isinstance(node, ast.Assign) or not isinstance(node.targets[0], ast.Name):
            continue
        name = node.targets[0].id
        if not name.startswith(prefix):
            continue
        try:
            found[name] = ast.literal_eval(node.value)
        except ValueError:
            # e.g. _ALL_ROLES, built by a call over the other constants. Recompute it
            # in a namespace holding what we already read.
            try:
                exec(compile(ast.Module(body=[node], type_ignores=[]), "<roles>", "exec"), found)  # noqa: S102
            except Exception:
                pass
    return found


class TestRoleFlags(unittest.TestCase):
    """The role -> (actor, rollout, ref, frozen) mapping."""

    @classmethod
    def setUpClass(cls):
        cls.const = module_constants(WORKERS_FILE)
        for name in ("_ACTOR_ROLES", "_ROLLOUT_ROLES", "_REF_ROLES", "_FROZEN_LM_ROLES", "_ALL_ROLES"):
            if name not in cls.const:
                raise AssertionError(f"{name} missing from {WORKERS_FILE.name}; the role flags were inlined again")

    def flags(self, role: str) -> tuple:
        return (
            role in self.const["_ACTOR_ROLES"],
            role in self.const["_ROLLOUT_ROLES"],
            role in self.const["_REF_ROLES"],
            role in self.const["_FROZEN_LM_ROLES"],
        )

    def test_preexisting_roles_are_unchanged(self):
        """Extracting the literals into constants must not have moved any existing role.

        These four tuples are the pre-refactor behaviour, transcribed from the original
        inline lists. If one drifts, an existing training script changes silently.
        """
        expected = {
            #                    actor, rollout, ref,   frozen
            "actor": (True, False, False, False),
            "actor_rollout": (True, True, False, False),
            "actor_rollout_ref": (True, True, True, False),
            "rollout": (False, True, False, False),
            "ref": (False, False, True, False),
        }
        for role, want in expected.items():
            self.assertEqual(self.flags(role), want, f"role {role!r} changed capability")

    def test_se_role_stays_inference_only(self):
        """se_rollout_ref is deliberately ref-ONLY -- do not "fix" it into a rollout.

        It cannot generate or train: generate_sequences asserts _is_rollout and
        update_actor asserts _is_actor, neither of which it has. The one call site that
        tries (`actor_rollout_se_wg.generate_sequences`) is dead code, reachable only if
        se_model.enable were flipped to True.
        """
        self.assertEqual(self.flags("se_rollout_ref"), (False, False, True, False))

    def test_frozen_rephraser_computes_logprobs_but_neither_trains_nor_generates(self):
        """The load-bearing assertion of the whole split.

        Both omissions matter. Not an actor -> no optimizer, so update_actor and
        save_checkpoint reject it. Not a rollout -> no vLLM engine, and THAT is what lets
        it coexist with the reasoner at all: vLLM's sleep mode uses a process-global
        CuMemAllocator with hardcoded "weights"/"kv_cache" tags, so a second engine in the
        same process would have its memory freed by the first one's sleep(). Log-probs need
        no engine -- one forward pass, the same path critic/reward scoring already takes.
        """
        is_actor, is_rollout, is_ref, is_frozen = self.flags("rephraser_logprob")
        self.assertTrue(is_frozen, "the rephraser must be flagged frozen so compute_log_prob admits it")
        self.assertFalse(
            is_actor,
            "rephraser_logprob must NOT be in _ACTOR_ROLES -- that is what makes update_actor "
            "and save_checkpoint reject it. Freezing is structural, not a convention.",
        )
        self.assertFalse(
            is_rollout,
            "rephraser_logprob must NOT be in _ROLLOUT_ROLES: a second vLLM engine in one "
            "process shares a global CuMemAllocator and the engines free each other's memory. "
            "Candidate generation is done offline precisely to avoid needing an engine here.",
        )
        self.assertFalse(is_ref, "the rephraser is not a reference policy; a third copy of the weights is pointless")

    def test_all_roles_covers_every_flag_list(self):
        """The __init__ assert uses _ALL_ROLES, so a role missing from it is unusable."""
        union = set(self.const["_ACTOR_ROLES"]) | set(self.const["_ROLLOUT_ROLES"]) | set(self.const["_REF_ROLES"]) | set(self.const["_FROZEN_LM_ROLES"])
        self.assertEqual(set(self.const["_ALL_ROLES"]), union)

    def test_frozen_roles_are_disjoint_from_rollout_roles(self):
        """A frozen LM must NOT be a rollout: that is what keeps it off the vLLM engine.

        The inverse of what an earlier revision asserted. The rephraser used to generate
        candidates online, which required an engine -- until it turned out two engines
        cannot share a process. Generation moved offline and the role lost _is_rollout;
        putting it back would reintroduce the allocator conflict.
        """
        self.assertEqual(set(self.const["_FROZEN_LM_ROLES"]) & set(self.const["_ROLLOUT_ROLES"]), set())

    def test_frozen_and_actor_are_disjoint(self):
        """The two are contradictory: frozen means no optimizer, actor means one exists."""
        self.assertEqual(set(self.const["_FROZEN_LM_ROLES"]) & set(self.const["_ACTOR_ROLES"]), set())


class TestFrozenWorkerWiring(unittest.TestCase):
    """Source-level checks on the branches the frozen role depends on."""

    @classmethod
    def setUpClass(cls):
        cls.src = WORKERS_FILE.read_text(encoding="utf-8")
        # Scope to NewActorRolloutRefWorker: this file also defines CriticWorker and
        # RewardModelWorker, whose save_checkpoint legitimately has no _is_actor guard.
        tree = ast.parse(cls.src)
        cls.methods = {}
        for node in tree.body:
            if isinstance(node, ast.ClassDef) and node.name == "NewActorRolloutRefWorker":
                for item in node.body:
                    if isinstance(item, ast.FunctionDef):
                        cls.methods[item.name] = ast.get_source_segment(cls.src, item) or ""
                break
        if not cls.methods:
            raise AssertionError("NewActorRolloutRefWorker not found in fsdp_workers_new.py")

    def test_update_actor_and_save_checkpoint_still_require_is_actor_alone(self):
        """If these ever accept _is_frozen_lm, the model stops being frozen."""
        for name in ("update_actor", "save_checkpoint"):
            body = self.methods[name]
            self.assertIn("assert self._is_actor", body, f"{name} lost its _is_actor guard")
            self.assertNotIn(
                "_is_frozen_lm",
                body,
                f"{name} must NOT admit a frozen LM -- that is the freeze guarantee",
            )

    def test_compute_log_prob_admits_the_frozen_lm(self):
        """Forward-only, so it is safe -- and required, since it is the proposal density."""
        self.assertIn("self._is_actor or self._is_frozen_lm", self.methods["compute_log_prob"])

    def test_frozen_role_gets_no_optimizer(self):
        """optim_config=None is what skips the AdamW + LR scheduler construction."""
        init_model = self.methods["init_model"]
        self.assertIn("elif self._is_frozen_lm:", init_model)
        # The frozen branch must pass the REAL actor fsdp_config, not the empty node the
        # standalone-rollout branch uses: FSDP wrapping reads wrap_policy /
        # mixed_precision, and the fsdp2 path reads offload_policy.
        frozen_branch = init_model.split("elif self._is_frozen_lm:", 1)[1].split("else:", 1)[0]
        self.assertIn("optim_config = None", frozen_branch)
        self.assertIn("fsdp_config = self.config.actor.fsdp_config", frozen_branch)

    def test_frozen_role_builds_the_actor_object_for_log_probs(self):
        """compute_log_prob delegates to self.actor, so the frozen role needs one too.

        Its actor_optimizer is None -- _build_model_optimizer returns None when
        optim_config is None -- which is why compute_log_prob (no_grad, no optimizer)
        works on it while update_policy could not.
        """
        self.assertIn("if self._is_actor or self._is_frozen_lm:", self.methods["init_model"])

    def test_frozen_role_offloads_params_but_not_an_optimizer(self):
        """There is no optimizer state to offload; only params move to host RAM."""
        init = self.methods["__init__"]
        frozen_branch = init.split("elif self._is_frozen_lm:", 1)[1].split("elif self._is_ref:", 1)[0]
        self.assertIn("self._is_offload_param = self.config.actor.fsdp_config", frozen_branch)
        # Match the assignment, not the word: the branch's comment mentions the flag.
        self.assertNotIn("self._is_offload_optimizer =", frozen_branch)


class TestRephraserPlumbing(unittest.TestCase):
    """Config defaults and the trainer-side wiring, checked without importing verl."""

    def test_role_enum_has_the_rephraser(self):
        src = RAY_TRAINER.read_text(encoding="utf-8")
        self.assertIn("ActorRolloutRephraser", src)
        # A reused enum value would make two roles collide in role_worker_mapping.
        values = {}
        for line in src.split("class Role(Enum):", 1)[1].split("\n\n", 1)[0].splitlines():
            if "=" in line and not line.strip().startswith("#"):
                name, _, value = line.partition("=")
                value = value.split("#")[0].strip()
                if value.isdigit():
                    self.assertNotIn(int(value), values, f"Role value {value} reused by {name.strip()} and {values.get(int(value))}")
                    values[int(value)] = name.strip()

    def test_config_defaults_to_disabled(self):
        """Every existing script must be unaffected until the switch is flipped.

        Parsed rather than grepped: the file has several `enable: False` lines, so a
        regex would pass even with the rephraser's own flipped to True.
        """
        import yaml

        cfg = yaml.safe_load(PPO_CONFIG.read_text(encoding="utf-8"))
        self.assertIn("rephraser", cfg, "the rephraser config block is missing")
        self.assertIs(cfg["rephraser"]["enable"], False, "rephraser.enable must default to False")
        # An empty override node merges to a no-op, which is what makes enable=False
        # byte-identical to the pre-split behaviour. A populated default would leak
        # rephraser settings into every run.
        self.assertIn("actor_rollout_rephraser", cfg)
        self.assertIn(cfg["actor_rollout_rephraser"], ({}, None), "actor_rollout_rephraser must default to empty")

    def test_registration_is_gated_on_the_switch(self):
        src = MAIN_PPO.read_text(encoding="utf-8")
        self.assertIn('config.get("rephraser", {}).get("enable", False)', src)
        self.assertIn("Role.ActorRolloutRephraser", src)

    def test_worker_config_is_deepcopied_before_merge(self):
        """A shared node would be batch-size-normalized once per worker group."""
        src = TRAINER_FILE.read_text(encoding="utf-8")
        block = src.split("if Role.ActorRolloutRephraser in self.role_worker_mapping:", 1)
        self.assertEqual(len(block), 2, "the rephraser registration block is missing")
        block = block[1][:1200]
        self.assertIn("deepcopy(self.config.actor_rollout_ref)", block)
        self.assertIn("actor_rollout_rephraser", block)
        self.assertIn('role="rephraser_logprob"', block)

    def test_rephraser_wg_is_always_defined(self):
        """The routing sites read `self.rephraser_wg or ...`, so it must always exist."""
        src = TRAINER_FILE.read_text(encoding="utf-8")
        self.assertIn("self.rephraser_wg = None", src)

    def test_proposal_density_comes_from_the_designated_worker(self):
        """The one error here that produces no exception, only a biased gradient.

        The ratio is pi_theta(y|x_short) / pi_phi(y|x_long). If the candidates come from
        phi (generated offline by Data/aggregate_sr_responses.py) but the denominator is
        computed by theta, the proposal density belongs to a distribution that never
        emitted y -- an invalid importance weight, and nothing raises or NaNs.
        """
        src = TRAINER_FILE.read_text(encoding="utf-8")
        tree = ast.parse(src)
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == "_summarize_replace_normal_step":
                body = ast.get_source_segment(src, node) or ""
                self.assertIn("logp_wg.compute_log_prob(logp_in)", body)
                # Online fallback (no rephraser worker) still generates through the same
                # handle, so sampling and density share a source by construction.
                self.assertIn("logp_wg.generate_sequences(cand_gen)", body)
                self.assertNotIn(
                    "self.actor_rollout_wg.compute_log_prob",
                    body,
                    "the proposal density must come from logp_wg -- the model that produced "
                    "the candidates -- not from the model being updated",
                )
                self.assertNotIn(
                    "self.actor_rollout_wg.generate_sequences",
                    body,
                    "candidate generation must go through logp_wg, not the step's own model",
                )
                return
        self.fail("_summarize_replace_normal_step not found")

    def test_offline_path_reads_the_column_instead_of_generating(self):
        """sr_use_offline must take the column branch, and force K=1 before the reshape.

        K is used to flatten long_prompt to [W*K, L]; the offline column holds ONE
        candidate per question, so a K>1 left in place would leave long_prompt with more
        rows than cand_resp -- a shape error at best, misaligned pairs at worst.
        """
        src = TRAINER_FILE.read_text(encoding="utf-8")
        tree = ast.parse(src)
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == "_summarize_replace_normal_step":
                body = ast.get_source_segment(src, node) or ""
                self.assertIn("sr_response_ids", body)
                head = body.split("long_rows = []", 1)
                self.assertEqual(len(head), 2, "the long_prompt flattening loop moved")
                self.assertIn("K = 1", head[0], "K must be forced to 1 BEFORE long_prompt is flattened")
                return
        self.fail("_summarize_replace_normal_step not found")

    def test_short_logprob_prompt_reuses_the_loss_time_prompt_tensor(self):
        """sr_logprob_prompt='short' must use the SAME tensor the loss is computed under.

        The point of 'short' is that the ratio's numerator and denominator share a prompt.
        Rebuilding the short prompt separately (a second slice of gen_batch, a different
        repeat order) would silently break exactly the property being asked for -- the two
        prompts would be nearly identical, so nothing would look wrong.

        Hence short_wk is computed ONCE, before the log-prob step, and passed to
        _build_hybrid_off_policy_output as loss_prompts afterwards.
        """
        src = TRAINER_FILE.read_text(encoding="utf-8")
        tree = ast.parse(src)
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == "_summarize_replace_normal_step":
                body = ast.get_source_segment(src, node) or ""
                self.assertEqual(
                    body.count("short_prompts = gen_batch.batch['input_ids'][::n]"),
                    1,
                    "short_wk must be derived once; a second derivation can drift from the "
                    "tensor the loss uses",
                )
                # Built before the log-prob call, and fed to the loss rows after it.
                pre, sep, post = body.partition("logp_wg.compute_log_prob(logp_in)")
                self.assertTrue(sep, "the log-prob call moved")
                self.assertIn("short_wk = short_prompts[wq_idx]", pre)
                self.assertIn("'prompts': short_wk", pre, "the short branch must use short_wk")
                self.assertIn("loss_prompts=short_wk", post, "the loss rows must reuse short_wk")
                return
        self.fail("_summarize_replace_normal_step not found")

    def test_validate_summarize_is_disabled_with_a_logprob_only_rephraser(self):
        """It needs to generate under the summarize prompt, which that worker cannot do.

        Falling back to the reasoner would silently retitle theta's accuracy as the
        rephraser's, so the function returns {} instead.
        """
        src = TRAINER_FILE.read_text(encoding="utf-8")
        tree = ast.parse(src)
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == "_validate_summarize":
                body = ast.get_source_segment(src, node) or ""
                self.assertIn('if getattr(self, "rephraser_wg", None) is not None:', body)
                self.assertNotIn(
                    "rephraser_wg.generate_sequences",
                    body,
                    "the logprob-only rephraser has no engine; it cannot generate here",
                )
                return
        self.fail("_validate_summarize not found")

    def test_startup_rejects_the_biased_loss_configuration(self):
        """summarize_loss_on_rollout_prompt=True has no importance correction at all."""
        src = TRAINER_FILE.read_text(encoding="utf-8")
        self.assertIn("_validate_rephraser_config", src)
        tree = ast.parse(src)
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == "_validate_rephraser_config":
                body = ast.get_source_segment(src, node) or ""
                for guard in (
                    "sr_use_offline",                    # the rephraser's only reason to exist
                    "summarize_loss_on_rollout_prompt",  # would drop the IS correction
                    "summarize_replace",                 # rephraser built but unused
                    "warmup_steps",                      # recycle steps, not routed yet
                    "collect_failures",                  # ditto, via the failure buffer
                    "off_policy_reshape",                # reshapes that discard the proposal
                ):
                    self.assertIn(guard, body, f"_validate_rephraser_config does not check {guard}")
                return
        self.fail("_validate_rephraser_config not found")


if __name__ == "__main__":
    unittest.main()

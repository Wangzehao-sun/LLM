"""A Ray worker that decodes with TWO frozen models at once, fusing per token.

Its only job is to produce the ONE rollout that summarize-replacement splices back per
question, together with the per-token density that rollout was actually sampled from.
``joint_sr.py`` drives it; ``joint_decode_core.py`` holds the decoding itself, shared
verbatim with the offline evaluator ``Data/joint_decode.py``.

Deliberately NOT built on ``NewActorRolloutRefWorker``. That class exists to shard a
TRAINABLE model -- FSDP mesh, Ulysses sequence parallel, optimizer, checkpoint manager,
a vLLM engine -- and every one of those is either useless or actively in the way here:

  * Both models are frozen. There is no optimizer, no update, nothing to save. Adding a
    role to ``_FROZEN_LM_ROLES`` would have inherited an FSDP wrap and a sharding
    manager for weights that never change.
  * The decode loop needs ``past_key_values`` threaded across ~10k sequential forwards
    and ``batch_select_indices`` called on the cache to drop finished rows. Both are
    plain-HF-module operations; an FSDP-wrapped module hands back a cache that does not
    survive that treatment cleanly.
  * vLLM cannot do this at all -- no per-token hook (see ``joint_decode_core``'s
    docstring). So the engine machinery is pure overhead.

The upshot is that ``fsdp_workers_new.py`` is untouched by this feature: no new role
string, no change to ``_ACTOR_ROLES`` / ``_ROLLOUT_ROLES`` / ``_FROZEN_LM_ROLES``, and
the tests pinning those tuples (``tests/workers/test_worker_roles_on_cpu.py``) keep
passing unchanged.

MEMORY. Two unsharded bf16 models sit resident on every rank for the whole run. A 4B
pair is ~16GB before any KV cache, and the cache is the term that grows with batch and
length. That is why the default is a DEDICATED resource pool: sharing the reasoner's
pool means sharing its process, where this would compete with the vLLM engine's
pre-reserved arena. ``joint_decode.n_gpus_per_node = 0`` (shared) still works and is
still the cheaper option for a small model, but it is not the default.
"""

from __future__ import annotations

import logging
import os

import torch
from omegaconf import DictConfig

from verl import DataProto
from verl.custom.joint_decode_core import joint_generate
from verl.single_controller.base import Worker
from verl.single_controller.base.decorator import Dispatch, register
from verl.utils import hf_tokenizer
from verl.utils.device import get_device_name, get_nccl_backend, get_torch_device
from verl.utils.fs import copy_to_local

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


class JointDecodeWorker(Worker):
    """Holds a frozen student + frozen teacher and decodes them jointly.

    ``config`` is the ``joint_decode`` node (see ``config/joint_decode.yaml``), already
    merged with any CLI overrides by the trainer.
    """

    def __init__(self, config: DictConfig):
        Worker.__init__(self)
        self.config = config

        # Ray's worker does not initialise the process group for us, and the dispatch
        # layer needs a world size to chunk by. Matches CriticWorker's own guard
        # (fsdp_workers_new.py:913).
        if not torch.distributed.is_initialized():
            torch.distributed.init_process_group(
                backend=f"cpu:gloo,{get_device_name()}:{get_nccl_backend()}",
                init_method=os.environ.get("DIST_INIT_METHOD", None),
            )

        self.student = None
        self.teacher = None
        self.tokenizer = None
        self._eos_ids = None
        self._pad_id = None

    # ------------------------------------------------------------------
    def _load(self, path: str, tag: str):
        """Load one frozen model, honouring the configured attention backend.

        ``attn_impl`` is tried and then FALLEN BACK from rather than demanded, because
        flash-attention-2 is an optional dependency and a model whose architecture lacks
        a kernel raises at construction time. Failing the whole run over an attention
        backend would be the wrong trade when sdpa produces identical numbers, so the
        substitution is made and announced.
        """
        from transformers import AutoModelForCausalLM

        local_path = copy_to_local(path)
        wanted = ([self.config.attn_impl] if self.config.attn_impl != "auto"
                  else ["flash_attention_2", "sdpa", "eager"])
        errors = []
        for impl in wanted:
            try:
                model = AutoModelForCausalLM.from_pretrained(
                    local_path,
                    torch_dtype=torch.bfloat16,
                    trust_remote_code=self.config.get("trust_remote_code", True),
                    attn_implementation=impl,
                )
            except (ValueError, ImportError, RuntimeError) as error:
                errors.append(f"{impl}: {type(error).__name__}: {error}")
                continue
            if self.rank == 0:
                print(f"[joint_decode] {tag} attn_implementation={impl}", flush=True)
            # No FSDP wrap and no device_map: one whole model per rank, on that rank's
            # own GPU. device_map='auto' would shard it across visible devices and
            # collide with the data-parallel split the dispatch layer already does.
            return model.to(get_torch_device().current_device()).eval()
        raise RuntimeError(
            f"could not load {tag} from {path} with any of {wanted}:\n  " + "\n  ".join(errors)
        )

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def init_model(self):
        self.tokenizer = hf_tokenizer(
            copy_to_local(self.config.student_model_path),
            trust_remote_code=self.config.get("trust_remote_code", True),
        )
        teacher_tokenizer = hf_tokenizer(
            copy_to_local(self.config.teacher_model_path),
            trust_remote_code=self.config.get("trust_remote_code", True),
        )
        # HARD failure, not a warning. Fusing logits position-by-position assumes index
        # i means the same token to both models. If the vocabularies differ, every step
        # silently intersects unrelated tokens and the run produces fluent nonsense with
        # no error anywhere -- the worst possible failure mode. The trainer's
        # rephraser path only prints a notice here (new_ray_trainer.py:664-676) because
        # it exchanges token IDS, which is a weaker requirement than exchanging logit
        # POSITIONS.
        if self.tokenizer.get_vocab() != teacher_tokenizer.get_vocab():
            raise ValueError(
                "joint decoding needs identical vocabularies: per-token logit fusion "
                "compares the two models' logits index by index, so a mismatch makes "
                f"every fused step meaningless.\n  student={self.config.student_model_path}"
                f"\n  teacher={self.config.teacher_model_path}"
            )
        del teacher_tokenizer

        self.student = self._load(self.config.student_model_path, "student")
        self.teacher = self._load(self.config.teacher_model_path, "teacher")

        eos = self.tokenizer.eos_token_id
        self._eos_ids = list(eos) if isinstance(eos, (list, tuple)) else [eos]
        self._pad_id = self.tokenizer.pad_token_id
        if self._pad_id is None:
            self._pad_id = self._eos_ids[0]
        if self._eos_ids[0] is None:
            raise ValueError(f"{self.config.student_model_path} has no eos_token_id; decoding cannot terminate")

        if self.rank == 0:
            reserved = get_torch_device().memory_reserved() / 1e9
            print(f"[joint_decode] both models resident, reserved={reserved:.1f}GB per rank", flush=True)

    # ------------------------------------------------------------------
    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
    @torch.no_grad()
    def generate_joint(self, data: DataProto) -> DataProto:
        """Decode this rank's shard of the questions and return responses + densities.

        In:  ``student_input_ids`` / ``student_attention_mask`` (left-padded, the prompt
             the LOSS will use) and ``teacher_input_ids`` / ``teacher_attention_mask``
             (left-padded, may be a different length -- only the generated suffix has to
             align, and it does because both models are fed the same sampled token).
        Out: ``responses`` [B, max_response_length] right-padded token ids, and
             ``off_log_probs`` [B, max_response_length] the per-token log density each
             token was drawn from, plus three per-row diagnostics.

        The caller pads the batch to ``world_size`` before dispatch; this method just
        decodes whatever rows it is handed.
        """
        cfg = self.config
        device = get_torch_device().current_device()
        width = int(data.meta_info["max_response_length"])

        student = {
            "input_ids": data.batch["student_input_ids"],
            "attention_mask": data.batch["student_attention_mask"],
        }
        teacher = {
            "input_ids": data.batch["teacher_input_ids"],
            "attention_mask": data.batch["teacher_attention_mask"],
        }
        n_rows = student["input_ids"].size(0)

        # Right-padded output buffers at the FULL width the trainer expects, filled in
        # per micro-batch. _build_hybrid_off_policy_output assigns off_old_log_probs at
        # max(max_response_length, responses.size(1)), so returning a narrower tensor
        # would either break there or, worse, be broadcast into the wrong columns.
        responses = torch.full((n_rows, width), self._pad_id, dtype=torch.long)
        off_log_probs = torch.zeros((n_rows, width), dtype=torch.float32)
        lengths = torch.zeros(n_rows, dtype=torch.long)
        fallback_frac = torch.zeros(n_rows, dtype=torch.float32)
        keep_ratio = torch.zeros(n_rows, dtype=torch.float32)
        student_logp = torch.zeros(n_rows, dtype=torch.float32)

        # Micro-batch inside the worker rather than relying on the dispatch split: the
        # dispatch chunk is however many questions this rank got, which is data-
        # dependent under wrong_only, while the batch that FITS is a memory property.
        # Two unsharded models plus a KV cache at these lengths is the binding
        # constraint, so it gets its own knob.
        micro = max(1, int(cfg.batch_size))
        n_narrow = 0
        for begin in range(0, n_rows, micro):
            stop = min(begin + micro, n_rows)
            out_ids, out_len, fb_row, logp_row, z_row, logq, narrow = joint_generate(
                self.student, self.teacher,
                {k: v[begin:stop] for k, v in student.items()},
                {k: v[begin:stop] for k, v in teacher.items()},
                eos_ids=self._eos_ids, pad_id=self._pad_id, args=cfg,
            )
            n_narrow += narrow

            # joint_generate returns only as many columns as it actually ran, which is
            # <= max_new_tokens (it breaks early once every row has finished). Truncate
            # to the trainer's width as well: max_new_tokens above max_response_length
            # would otherwise overflow the buffer, and silently dropping the tail is
            # correct -- the trainer cannot represent a longer response anyway.
            keep = min(out_ids.size(1), width)
            responses[begin:stop, :keep] = out_ids[:, :keep]
            off_log_probs[begin:stop, :keep] = logq[:, :keep]
            # Clamp the reported length to what was actually stored, so a truncated row
            # does not claim tokens that are not in the tensor.
            out_len = out_len.clamp(max=keep)
            lengths[begin:stop] = out_len

            # Per-row means. A row that emitted nothing has no tokens to average over;
            # 0 (not NaN) is used here because these go into a DataProto that gets
            # padded, chunked and concatenated, and NaN would poison the batch means the
            # trainer computes over it. The offline script reports NaN instead precisely
            # because it averages for a human, not for a reduction.
            denom = out_len.clamp(min=1).to(torch.float32)
            fallback_frac[begin:stop] = fb_row.to(torch.float32) / denom
            keep_ratio[begin:stop] = (z_row / denom.double()).to(torch.float32)
            student_logp[begin:stop] = (logp_row / denom.double()).to(torch.float32)

        # Zero the density past each row's end. joint_generate already writes 0 for dead
        # rows, and the buffer starts at 0, so this is belt-and-braces -- but the
        # invariant is load-bearing (a stray density on a pad token becomes a gradient
        # on a pad token), so it is enforced rather than assumed.
        valid = torch.arange(width).unsqueeze(0) < lengths.unsqueeze(1)
        off_log_probs = off_log_probs * valid
        assert responses.size(1) == width, f"responses width {responses.size(1)} != {width}"
        assert off_log_probs.shape == responses.shape, "density must align with tokens 1:1"

        if n_narrow and self.rank == 0:
            print(f"[joint_decode] top-p nucleus exceeded the candidate cap on {n_narrow:,} "
                  f"row-steps; sampling was narrower than top_p={cfg.top_p} asked for", flush=True)

        return DataProto.from_dict(
            tensors={
                "responses": responses.to(device),
                "off_log_probs": off_log_probs.to(device),
                "response_lengths": lengths.to(device),
                "fallback_frac": fallback_frac.to(device),
                "teacher_keep_ratio": keep_ratio.to(device),
                "student_mean_logp": student_logp.to(device),
            }
        )

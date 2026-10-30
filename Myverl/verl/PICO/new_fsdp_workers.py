# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
PICO actor / rollout / ref worker. Copied from verl/custom/fsdp_workers_new.py.
"""

import json
import logging
import os
import warnings
from dataclasses import asdict
from typing import Optional

import psutil
import torch
import torch.distributed
import torch.distributed as dist
from codetiming import Timer
from omegaconf import DictConfig, OmegaConf, open_dict
from peft import LoraConfig, TaskType, get_peft_model
from safetensors.torch import save_file
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

from verl import DataProto
from verl.models.transformers.monkey_patch import apply_monkey_patch
from verl.PICO.new_dp_actor import PICODataParallelPPOActor
from verl.single_controller.base import Worker
from verl.single_controller.base.decorator import Dispatch, register
from verl.utils import hf_processor, hf_tokenizer, omega_conf_to_dataclass
from verl.utils.activation_offload import enable_activation_offloading
from verl.utils.checkpoint.fsdp_checkpoint_manager import FSDPCheckpointManager
from verl.utils.debug import DistProfiler, DistProfilerExtension, ProfilerConfig, log_gpu_memory_usage, simple_timer
from verl.utils.debug.performance import reduce_timing
from verl.utils.device import get_device_id, get_device_name, get_nccl_backend, get_torch_device
from verl.utils.flops_counter import FlopsCounter
from verl.utils.fs import copy_to_local
from verl.utils.fsdp_utils import (
    CPUOffloadPolicy,
    MixedPrecisionPolicy,
    apply_fsdp2,
    fsdp2_load_full_state_dict,
    fsdp_version,
    get_fsdp_full_state_dict,
    get_fsdp_wrap_policy,
    get_init_weight_context_manager,
    init_fn,
    layered_summon_lora_params,
    load_fsdp_model_to_gpu,
    load_fsdp_optimizer,
    offload_fsdp_model_to_cpu,
    offload_fsdp_optimizer,
)
from verl.utils.import_utils import import_external_libs
from verl.utils.py_functional import convert_to_regular_types
from verl.workers.sharding_manager.fsdp_ulysses import FSDPUlyssesShardingManager

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

device_name = get_device_name()


def create_device_mesh(world_size, fsdp_size):
    if fsdp_size < 0 or fsdp_size >= world_size:
        device_mesh = init_device_mesh(device_name, mesh_shape=(world_size,), mesh_dim_names=["fsdp"])
    else:
        device_mesh = init_device_mesh(device_name, mesh_shape=(world_size // fsdp_size, fsdp_size), mesh_dim_names=["ddp", "fsdp"])
    return device_mesh


def get_sharding_strategy(device_mesh):
    from torch.distributed.fsdp import ShardingStrategy

    if device_mesh.ndim == 1:
        sharding_strategy = ShardingStrategy.FULL_SHARD
    elif device_mesh.ndim == 2:
        sharding_strategy = ShardingStrategy.HYBRID_SHARD
    else:
        raise NotImplementedError(f"Get device mesh ndim={device_mesh.ndim}, but only support 1 or 2")
    return sharding_strategy


_ACTOR_ROLES = ("actor", "actor_rollout", "actor_rollout_ref")
_ROLLOUT_ROLES = ("rollout", "actor_rollout", "actor_rollout_ref")
_REF_ROLES = ("ref", "actor_rollout_ref")
_ALL_ROLES = tuple(dict.fromkeys(_ACTOR_ROLES + _ROLLOUT_ROLES + _REF_ROLES))


class PICOActorRolloutRefWorker(Worker, DistProfilerExtension):
    """
    This worker can be instantiated as a standalone actor or a standalone rollout or a standalone reference policy
    or a hybrid engine based on the config.rollout.

    Copied from verl/custom/fsdp_workers_new.NewActorRolloutRefWorker without the frozen
    rephraser / SE roles. Adds ``generate_joint`` (joint decoding with a frozen teacher).
    """

    def __init__(self, config: DictConfig, role: str):
        Worker.__init__(self)

        self.config = config
        import torch.distributed

        if not torch.distributed.is_initialized():
            rank = int(os.environ.get("RANK", 0))
            world_size = int(os.environ.get("WORLD_SIZE", 1))
            torch.distributed.init_process_group(backend=f"cpu:gloo,{get_device_name()}:{get_nccl_backend()}", rank=rank, world_size=world_size, init_method=os.environ.get("DIST_INIT_METHOD", None))

        # build device mesh for FSDP
        world_size = torch.distributed.get_world_size()
        #print("GPU设备数量:", world_size)
        # TODO(sgm): support FSDP hybrid shard for larger model
        self.device_mesh = create_device_mesh(world_size=world_size, fsdp_size=self.config.actor.fsdp_config.fsdp_size)

        # build device mesh for Ulysses Sequence Parallel
        self.ulysses_device_mesh = None
        self.ulysses_sequence_parallel_size = self.config.actor.get("ulysses_sequence_parallel_size", 1)
        dp = world_size // self.ulysses_sequence_parallel_size
        if self.ulysses_sequence_parallel_size > 1:
            self.ulysses_device_mesh = init_device_mesh(device_name, mesh_shape=(dp, self.ulysses_sequence_parallel_size), mesh_dim_names=["dp", "sp"])

        self.ulysses_sharding_manager = FSDPUlyssesShardingManager(self.ulysses_device_mesh)
        self._lora_rank = self.config.model.get("lora_rank", 0)
        self._is_lora = self._lora_rank > 0

        self.role = role
        assert self.role in _ALL_ROLES, f"unknown role {self.role!r}, expected one of {_ALL_ROLES}"

        self._is_actor = self.role in _ACTOR_ROLES
        self._is_rollout = self.role in _ROLLOUT_ROLES
        self._is_ref = self.role in _REF_ROLES
        profiler_config: Optional[ProfilerConfig] = None
        if self._is_actor:
            profiler_config = omega_conf_to_dataclass(config.actor.get("profiler", {}), ProfilerConfig)
        if self._is_rollout:
            profiler_config = omega_conf_to_dataclass(config.rollout.get("profiler", {}), ProfilerConfig)
        if self._is_ref:
            profiler_config = omega_conf_to_dataclass(config.ref.get("profiler", {}), ProfilerConfig)

        DistProfilerExtension.__init__(self, DistProfiler(rank=self.rank, config=profiler_config))

        self._is_offload_param = False
        self._is_offload_optimizer = False
        if self._is_actor:
            self._is_offload_param = self.config.actor.fsdp_config.get("param_offload", False)
            self._is_offload_optimizer = self.config.actor.fsdp_config.get("optimizer_offload", False)
        elif self._is_ref:
            # TODO: it seems that manual offload is slowly than FSDP offload
            self._is_offload_param = self.config.ref.fsdp_config.get("param_offload", False)

        # normalize config
        if self._is_actor:
            self.config.actor.ppo_mini_batch_size *= self.config.rollout.n
            self.config.actor.ppo_mini_batch_size //= self.device_mesh.size() // self.ulysses_sequence_parallel_size
            assert self.config.actor.ppo_mini_batch_size > 0, f"ppo_mini_batch_size {self.config.actor.ppo_mini_batch_size} should be larger than 0 after normalization"
            # micro bsz
            if self.config.actor.ppo_micro_batch_size is not None:
                self.config.actor.ppo_micro_batch_size //= self.device_mesh.size() // self.ulysses_sequence_parallel_size
                self.config.actor.ppo_micro_batch_size_per_gpu = self.config.actor.ppo_micro_batch_size

            if self.config.actor.ppo_micro_batch_size_per_gpu is not None:
                assert self.config.actor.ppo_mini_batch_size % self.config.actor.ppo_micro_batch_size_per_gpu == 0, f"normalized ppo_mini_batch_size {self.config.actor.ppo_mini_batch_size} should be divisible by ppo_micro_batch_size_per_gpu {self.config.actor.ppo_micro_batch_size_per_gpu}"
                assert self.config.actor.ppo_mini_batch_size // self.config.actor.ppo_micro_batch_size_per_gpu > 0, f"normalized ppo_mini_batch_size {self.config.actor.ppo_mini_batch_size} should be larger than ppo_micro_batch_size_per_gpu {self.config.actor.ppo_micro_batch_size_per_gpu}"

        # normalize rollout config
        if self._is_rollout and self.config.rollout.log_prob_micro_batch_size is not None:
            self.config.rollout.log_prob_micro_batch_size //= self.device_mesh.size() // self.ulysses_sequence_parallel_size
            self.config.rollout.log_prob_micro_batch_size_per_gpu = self.config.rollout.log_prob_micro_batch_size
        # normalize ref config
        if self._is_ref and self.config.ref.log_prob_micro_batch_size is not None:
            self.config.ref.log_prob_micro_batch_size //= self.device_mesh.size() // self.ulysses_sequence_parallel_size
            self.config.ref.log_prob_micro_batch_size_per_gpu = self.config.ref.log_prob_micro_batch_size

    def _build_model_optimizer(
        self,
        model_path,
        fsdp_config,
        optim_config,
        override_model_config,
        use_remove_padding=False,
        use_fused_kernels=False,
        enable_gradient_checkpointing=False,
        trust_remote_code=False,
        use_liger=False,
        role="actor",
        enable_activation_offload=False,
    ):
        from torch import optim
        from torch.distributed.fsdp import CPUOffload, MixedPrecision
        from transformers import AutoConfig, AutoModelForCausalLM, AutoModelForVision2Seq

        from verl.utils.model import get_generation_config, print_model_size, update_model_config
        from verl.utils.torch_dtypes import PrecisionType

        assert role in ["actor", "ref"]

        log_gpu_memory_usage(f"Before init {role} from HF AutoModel", logger=logger)
        local_path = model_path

        # note that we have to create model in fp32. Otherwise, the optimizer is in bf16, which is incorrect
        # TODO(zhangchi.usc1992): 1. support create from random initialized model. 2. Support init with FSDP directly
        self.tokenizer = hf_tokenizer(local_path, trust_remote_code=trust_remote_code)
        self.processor = hf_processor(local_path, trust_remote_code=trust_remote_code)

        if self.config.model.get("custom_chat_template", None) is not None:
            if self.processor is not None:
                self.processor.chat_template = self.config.model.custom_chat_template
            else:
                self.tokenizer.chat_template = self.config.model.custom_chat_template

        torch_dtype = fsdp_config.get("model_dtype", None)
        if torch_dtype is None:
            torch_dtype = torch.float32 if self._is_actor else torch.bfloat16
        else:
            torch_dtype = PrecisionType.to_dtype(torch_dtype)

        # override model kwargs
        actor_model_config = AutoConfig.from_pretrained(local_path, trust_remote_code=trust_remote_code, attn_implementation="flash_attention_2")

        # patch for kimi-vl
        if getattr(actor_model_config, "model_type", None) == "kimi_vl":
            actor_model_config.text_config.topk_method = "greedy"

        self.generation_config = get_generation_config(local_path, trust_remote_code=trust_remote_code)

        override_config_kwargs = {
            "bos_token_id": self.tokenizer.bos_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
            "pad_token_id": self.tokenizer.pad_token_id,
        }
        override_config_kwargs.update(override_model_config)
        update_model_config(actor_model_config, override_config_kwargs=override_config_kwargs)
        if self.rank == 0:
            print(f"Model config after override: {actor_model_config}")

        # NOTE(fix me): tie_word_embedding causes meta_tensor init to hang
        init_context = get_init_weight_context_manager(use_meta_tensor=not actor_model_config.tie_word_embeddings, mesh=self.device_mesh)

        with init_context(), warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if type(actor_model_config) in AutoModelForVision2Seq._model_mapping.keys():
                actor_module_class = AutoModelForVision2Seq
            else:
                actor_module_class = AutoModelForCausalLM

            actor_module = actor_module_class.from_pretrained(
                pretrained_model_name_or_path=local_path,
                torch_dtype=torch_dtype,
                config=actor_model_config,
                trust_remote_code=trust_remote_code,
            )

            # Apply Liger kernel to the model if use_liger is set to True
            if use_liger:
                from liger_kernel.transformers.monkey_patch import _apply_liger_kernel_to_instance

                _apply_liger_kernel_to_instance(model=actor_module)

            fused_kernel_options = self.config.model.get("fused_kernel_options", None)
            fused_kernels_backend = fused_kernel_options.get("impl_backend", None) if fused_kernel_options is not None else None

            apply_monkey_patch(
                model=actor_module,
                use_remove_padding=use_remove_padding,
                ulysses_sp_size=self.ulysses_sequence_parallel_size,
                use_fused_kernels=use_fused_kernels,
                fused_kernels_backend=fused_kernels_backend,
            )

            # some parameters may not in torch_dtype. TODO(zhangchi.usc1992) remove this after we switch to fsdp2
            actor_module.to(torch_dtype)

            if enable_gradient_checkpointing:
                actor_module.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
            if self._is_lora:
                print("Applying LoRA to actor module")
                actor_module.enable_input_require_grads()
                # Convert config to regular Python types before creating PEFT model
                lora_config = {"task_type": TaskType.CAUSAL_LM, "r": self.config.model.lora_rank, "lora_alpha": self.config.model.lora_alpha, "target_modules": convert_to_regular_types(self.config.model.target_modules), "bias": "none"}
                actor_module = get_peft_model(actor_module, LoraConfig(**lora_config))
        torch.distributed.barrier()

        if self.rank == 0:
            print_model_size(actor_module)

        log_gpu_memory_usage(f"After init {role} from HF AutoModel", logger=logger)

        # We wrap FSDP for rollout as well
        mixed_precision_config = fsdp_config.get("mixed_precision", None)
        if mixed_precision_config is not None:
            param_dtype = PrecisionType.to_dtype(mixed_precision_config.get("param_dtype", "bf16"))
            reduce_dtype = PrecisionType.to_dtype(mixed_precision_config.get("reduce_dtype", "fp32"))
            buffer_dtype = PrecisionType.to_dtype(mixed_precision_config.get("buffer_dtype", "fp32"))
        else:
            param_dtype = torch.bfloat16
            reduce_dtype = torch.float32
            buffer_dtype = torch.float32

        mixed_precision = MixedPrecision(param_dtype=param_dtype, reduce_dtype=reduce_dtype, buffer_dtype=buffer_dtype)

        auto_wrap_policy = get_fsdp_wrap_policy(module=actor_module, config=fsdp_config.get("wrap_policy", None), is_lora=self.config.model.get("lora_rank", 0) > 0)

        if self._is_rollout and self.config.rollout.name == "hf":
            # TODO(zhangchi.usc1992, shengguangming) fix me. Current, auto_wrap_policy causes HFRollout to hang in Gemma
            auto_wrap_policy = None

        if self.rank == 0:
            print(f"wrap_policy: {auto_wrap_policy}")

        fsdp_mesh = self.device_mesh
        sharding_strategy = get_sharding_strategy(fsdp_mesh)

        # TODO: add transformer policy
        # We force reference policy to use CPUOffload to save memory.
        # We force turn off CPUOffload for actor because it causes incorrect results when using grad accumulation
        cpu_offload = None if role == "actor" else CPUOffload(offload_params=True)
        fsdp_strategy = self.config.actor.strategy
        if fsdp_strategy == "fsdp":
            actor_module_fsdp = FSDP(
                actor_module,
                cpu_offload=cpu_offload,
                param_init_fn=init_fn,
                use_orig_params=False,
                auto_wrap_policy=auto_wrap_policy,
                device_id=get_device_id(),
                sharding_strategy=sharding_strategy,  # zero3
                mixed_precision=mixed_precision,
                sync_module_states=True,
                device_mesh=self.device_mesh,
                forward_prefetch=self.config.actor.fsdp_config.forward_prefetch,
            )
        elif fsdp_strategy == "fsdp2":
            assert CPUOffloadPolicy is not None, "PyTorch version >= 2.4 is required for using fully_shard API (FSDP2)"
            mp_policy = MixedPrecisionPolicy(param_dtype=param_dtype, reduce_dtype=reduce_dtype, cast_forward_inputs=True)
            if role == "actor" and fsdp_config.offload_policy:
                cpu_offload = CPUOffloadPolicy(pin_memory=True)
                self._is_offload_param = False
                self._is_offload_optimizer = False
            else:
                cpu_offload = None if role == "actor" else CPUOffloadPolicy(pin_memory=True)

            fsdp_kwargs = {
                "mesh": fsdp_mesh,
                "mp_policy": mp_policy,
                "offload_policy": cpu_offload,
                "reshard_after_forward": fsdp_config.reshard_after_forward,
            }
            full_state = actor_module.state_dict()
            apply_fsdp2(actor_module, fsdp_kwargs, fsdp_config)
            fsdp2_load_full_state_dict(actor_module, full_state, fsdp_mesh, cpu_offload)
            actor_module_fsdp = actor_module
        else:
            raise NotImplementedError(f"not implement {fsdp_strategy}")

        if enable_activation_offload:
            enable_activation_offloading(actor_module_fsdp, fsdp_strategy, enable_gradient_checkpointing)

        log_gpu_memory_usage(f"After {role} FSDP init", logger=logger)

        # TODO: add more optimizer args into config
        if role == "actor" and optim_config is not None:
            from verl.utils.torch_functional import get_constant_schedule_with_warmup, get_cosine_schedule_with_warmup

            actor_optimizer = optim.AdamW(
                actor_module_fsdp.parameters(),
                lr=optim_config.lr,
                betas=optim_config.get("betas", (0.9, 0.999)),
                weight_decay=optim_config.get("weight_decay", 1e-2),
            )

            total_steps = optim_config.get("total_training_steps", 0)
            num_warmup_steps = int(optim_config.get("lr_warmup_steps", -1))
            warmup_style = optim_config.get("warmup_style", "constant")
            min_lr_ratio = optim_config.get("min_lr_ratio", 0.0)
            num_cycles = optim_config.get("num_cycles", 0.5)
            if num_warmup_steps < 0:
                num_warmup_steps_ratio = optim_config.get("lr_warmup_steps_ratio", 0.0)
                num_warmup_steps = int(num_warmup_steps_ratio * total_steps)

            if self.rank == 0:
                print(f"Total steps: {total_steps}, num_warmup_steps: {num_warmup_steps}")

            if warmup_style == "constant":
                actor_lr_scheduler = get_constant_schedule_with_warmup(optimizer=actor_optimizer, num_warmup_steps=num_warmup_steps)
            elif warmup_style == "cosine":
                actor_lr_scheduler = get_cosine_schedule_with_warmup(optimizer=actor_optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=total_steps, min_lr_ratio=min_lr_ratio, num_cycles=num_cycles)
            else:
                raise NotImplementedError(f"Warmup style {warmup_style} is not supported")

            log_gpu_memory_usage(f"After {role} optimizer init", logger=logger)
        else:
            actor_optimizer = None
            actor_lr_scheduler = None

        return actor_module_fsdp, actor_optimizer, actor_lr_scheduler, actor_model_config

    def _build_rollout(self, trust_remote_code=False):
        from torch.distributed.device_mesh import init_device_mesh

        # TODO(sgm): support FSDP hybrid shard for larger model
        infer_tp = self.config.rollout.tensor_model_parallel_size
        dp = self.world_size // infer_tp
        assert self.world_size % infer_tp == 0, f"rollout world_size: {self.world_size} is not divisible by infer_tp: {infer_tp}"
        rollout_device_mesh = init_device_mesh(device_name, mesh_shape=(dp, infer_tp), mesh_dim_names=["dp", "infer_tp"])
        rollout_name = self.config.rollout.name
        if rollout_name == "vllm":
            from verl.workers.rollout.vllm_rollout import vllm_mode, vLLMRollout
            from verl.workers.sharding_manager.fsdp_vllm import FSDPVLLMShardingManager

            log_gpu_memory_usage(f"Before building {rollout_name} rollout", logger=logger)
            local_path = copy_to_local(self.config.model.path, use_shm=self.config.model.get("use_shm", False))
            lora_kwargs = {"lora_kwargs": {"enable_lora": True, "max_loras": 1, "max_lora_rank": self._lora_rank}} if self._is_lora else {}
            if vllm_mode == "customized":
                rollout = vLLMRollout(actor_module=self.actor_module_fsdp, config=self.config.rollout, tokenizer=self.tokenizer, model_hf_config=self.actor_model_config, trust_remote_code=trust_remote_code, **lora_kwargs)
            elif vllm_mode == "spmd":
                # verl/custom/new_vllm_rollout.NewvLLMRollout overrides nothing, so the stock class is used.
                rollout = vLLMRollout(model_path=local_path, config=self.config.rollout, tokenizer=self.tokenizer, model_hf_config=self.actor_model_config, device_mesh=rollout_device_mesh, trust_remote_code=trust_remote_code, **lora_kwargs)
            else:
                raise NotImplementedError("vllm_mode must be 'customized' or 'spmd'")

            log_gpu_memory_usage(f"After building {rollout_name} rollout", logger=logger)
            full_params = torch.distributed.get_world_size() == 1
            rollout_sharding_manager = FSDPVLLMShardingManager(
                module=self.actor_module_fsdp,
                inference_engine=rollout.inference_engine,
                model_config=self.actor_model_config,
                full_params=full_params,
                device_mesh=rollout_device_mesh,
                offload_param=self._is_offload_param,
                load_format=self.config.rollout.load_format,
                layered_summon=self.config.rollout.get("layered_summon", False),
            )
            log_gpu_memory_usage("After building sharding manager", logger=logger)
        else:
            raise NotImplementedError(f"PICO only supports rollout.name=vllm, got {self.config.rollout.name!r}")

        return rollout, rollout_sharding_manager

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def init_model(self):
        from verl.workers.actor import DataParallelPPOActor

        # This is used to import external_lib into the huggingface systems
        import_external_libs(self.config.model.get("external_lib", None))

        override_model_config = OmegaConf.to_container(self.config.model.get("override_config", OmegaConf.create()))

        use_remove_padding = self.config.model.get("use_remove_padding", False)
        use_shm = self.config.model.get("use_shm", False)
        use_fused_kernels = self.config.model.get("use_fused_kernels", False)

        if self._is_actor or self._is_rollout:
            # we need the model for actor and rollout
            if self._is_actor:
                optim_config = self.config.actor.optim
                fsdp_config = self.config.actor.fsdp_config
            else:
                optim_config = None
                fsdp_config = OmegaConf.create()

            local_path = copy_to_local(self.config.model.path, use_shm=use_shm)
            (
                self.actor_module_fsdp,
                self.actor_optimizer,
                self.actor_lr_scheduler,
                self.actor_model_config,
            ) = self._build_model_optimizer(
                model_path=local_path,
                fsdp_config=fsdp_config,
                optim_config=optim_config,
                override_model_config=override_model_config,
                use_remove_padding=use_remove_padding,
                use_fused_kernels=use_fused_kernels,
                enable_gradient_checkpointing=self.config.model.get("enable_gradient_checkpointing", False),
                trust_remote_code=self.config.model.get("trust_remote_code", False),
                use_liger=self.config.model.get("use_liger", False),
                role="actor",
                enable_activation_offload=self.config.model.get("enable_activation_offload", False),
            )

            # get the original unwrapped module
            if fsdp_version(self.actor_module_fsdp) == 1:
                self.actor_module = self.actor_module_fsdp._fsdp_wrapped_module

            if self._is_offload_param:
                offload_fsdp_model_to_cpu(self.actor_module_fsdp)
                log_gpu_memory_usage("After offload actor model during init", logger=logger)

            if self._is_offload_optimizer:
                offload_fsdp_optimizer(optimizer=self.actor_optimizer)
                log_gpu_memory_usage("After offload actor optimizer during init", logger=logger)
        if self._is_actor:
            OmegaConf.set_struct(self.config.actor, True)
            with open_dict(self.config.actor):
                self.config.actor.use_remove_padding = use_remove_padding
                self.config.actor.use_fused_kernels = use_fused_kernels
            self.actor = PICODataParallelPPOActor(config=self.config.actor, actor_module=self.actor_module_fsdp, actor_optimizer=self.actor_optimizer)

        if self._is_rollout:
            self.rollout, self.rollout_sharding_manager = self._build_rollout(trust_remote_code=self.config.model.get("trust_remote_code", False))

        if self._is_ref:
            local_path = copy_to_local(self.config.model.path, use_shm=use_shm)
            self.ref_module_fsdp = self._build_model_optimizer(
                model_path=local_path,
                fsdp_config=self.config.ref.fsdp_config,
                optim_config=None,
                override_model_config=override_model_config,
                use_remove_padding=use_remove_padding,
                use_fused_kernels=use_fused_kernels,
                trust_remote_code=self.config.model.get("trust_remote_code", False),
                use_liger=self.config.model.get("use_liger", False),
                role="ref",
            )[0]
            OmegaConf.set_struct(self.config.ref, True)
            with open_dict(self.config.ref):
                self.config.ref.use_remove_padding = use_remove_padding
                self.config.ref.use_fused_kernels = use_fused_kernels
            self.ref_policy = DataParallelPPOActor(config=self.config.ref, actor_module=self.ref_module_fsdp)

        if self._is_actor:
            self.flops_counter = FlopsCounter(self.actor_model_config)
            self.checkpoint_manager = FSDPCheckpointManager(
                model=self.actor_module_fsdp,
                optimizer=self.actor.actor_optimizer,
                lr_scheduler=self.actor_lr_scheduler,
                processing_class=self.processor if self.processor is not None else self.tokenizer,
                checkpoint_config=self.config.actor.checkpoint,
            )

        if not self._is_actor and self._is_rollout:
            # If ActorRolloutRefWorker is initialized as a standalone rollout,
            # create a checkpoint manager for FSDP model to allow loading FSDP checkpoints for rollout.

            checkpoint_contents = OmegaConf.create({"load_contents": ["model"], "save_contents": []})
            self.checkpoint_manager = FSDPCheckpointManager(
                model=self.actor_module_fsdp,
                optimizer=None,
                lr_scheduler=None,
                processing_class=self.processor if self.processor is not None else self.tokenizer,
                checkpoint_config=checkpoint_contents,
            )

    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
    @DistProfiler.annotate(color="red")
    def update_actor(self, data: DataProto):
        # Support all hardwares
        data = data.to("cpu")  # data will to device with each micro batch on actor.update_policy

        assert self._is_actor
        if self._is_offload_param:
            load_fsdp_model_to_gpu(self.actor_module_fsdp)
        if self._is_offload_optimizer:
            load_fsdp_optimizer(optimizer=self.actor_optimizer, device_id=get_device_id())

        with self.ulysses_sharding_manager:
            data = self.ulysses_sharding_manager.preprocess_data(data=data)
            # perform training
            with Timer(name="update_policy", logger=None) as timer:
                metrics, tensors_dict_to_return = self.actor.update_policy(data=data)
            delta_time = timer.last
            global_num_tokens = data.meta_info["global_token_num"]
            estimated_flops, promised_flops = self.flops_counter.estimate_flops(global_num_tokens, delta_time)
            metrics["perf/mfu/actor"] = estimated_flops * self.config.actor.ppo_epochs / promised_flops / self.world_size
            metrics["perf/max_memory_allocated_gb"] = get_torch_device().max_memory_allocated() / (1024**3)
            metrics["perf/max_memory_reserved_gb"] = get_torch_device().max_memory_reserved() / (1024**3)
            metrics["perf/cpu_memory_used_gb"] = psutil.virtual_memory().used / (1024**3)

            lr = self.actor_lr_scheduler.get_last_lr()[0]
            metrics["actor/lr"] = lr
            self.actor_lr_scheduler.step()

            # TODO: here, we should return all metrics
            # 1. 准备要返回的张量字典
            #tensors_dict_to_return =
            output = DataProto.from_dict(tensors=tensors_dict_to_return, meta_info={"metrics": metrics})
            # 如果需要将 log_prob_shard 传回 trainer，可以这样做：
            output = self.ulysses_sharding_manager.postprocess_data(data=output)
            output = output.to("cpu")

        if self._is_offload_param:
            offload_fsdp_model_to_cpu(self.actor_module_fsdp)
            log_gpu_memory_usage("After offload actor model during update_actor", logger=logger)
        if self._is_offload_optimizer:
            offload_fsdp_optimizer(optimizer=self.actor_optimizer)
            log_gpu_memory_usage("After offload actor optimizer during update_actor", logger=logger)

        return output

    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
    @DistProfiler.annotate(color="red")
    def generate_sequences(self, prompts: DataProto):
        # Support all hardwares
        prompts = prompts.to(get_device_id())

        assert self._is_rollout

        meta_info = {
            "eos_token_id": self.generation_config.eos_token_id if self.generation_config is not None else self.tokenizer.eos_token_id,
            "pad_token_id": self.generation_config.pad_token_id if self.generation_config is not None else self.tokenizer.pad_token_id,
        }
        prompts.meta_info.update(meta_info)
        timing_generate = {}
        with self.rollout_sharding_manager:
            log_gpu_memory_usage("After entering rollout sharding manager", logger=logger)

            prompts = self.rollout_sharding_manager.preprocess_data(prompts)
            with simple_timer("generate_sequences", timing_generate):
                output = self.rollout.generate_sequences(prompts=prompts)

            log_gpu_memory_usage("After rollout generation", logger=logger)

            output = self.rollout_sharding_manager.postprocess_data(output)

        timing_generate.update(self.rollout_sharding_manager.timing)
        # We calculate the average timing across all ranks
        # to make sure meta_info["timing"] is the same
        timing_generate = reduce_timing(timing_generate)
        output.meta_info["timing"] = timing_generate
        output = output.to("cpu")

        # clear kv cache
        get_torch_device().empty_cache()
        return output

    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
    @DistProfiler.annotate(color="blue")
    def compute_log_prob(self, data: DataProto):
        # when is_lora is True, we use the actor without lora applied to calculate the log_prob
        # which is mostly used for ref log_prob calculation
        assert self._is_actor
        if self._is_offload_param:
            load_fsdp_model_to_gpu(self.actor_module_fsdp)

        # Support all hardwares
        from contextlib import nullcontext

        is_lora = data.meta_info.pop("is_lora", False)
        adapter_ctx = self.actor.actor_module.disable_adapter() if is_lora else nullcontext()
        data = data.to(get_device_id())
        # we should always recompute old_log_probs when it is HybridEngine
        data.meta_info["micro_batch_size"] = self.config.rollout.log_prob_micro_batch_size_per_gpu
        data.meta_info["max_token_len"] = self.config.rollout.log_prob_max_token_len_per_gpu
        data.meta_info["use_dynamic_bsz"] = self.config.rollout.log_prob_use_dynamic_bsz
        temperature = self.config.rollout.temperature
        data.meta_info["temperature"] = temperature
        # perform recompute log_prob
        with self.ulysses_sharding_manager:
            data = self.ulysses_sharding_manager.preprocess_data(data)
            with adapter_ctx:
                output, entropys = self.actor.compute_log_prob(data=data, calculate_entropy=True)
            output = DataProto.from_dict(
                tensors={"old_log_probs": output, "entropys": entropys},
                meta_info={"temperature": temperature},
            )
            output = self.ulysses_sharding_manager.postprocess_data(output)

        output = output.to("cpu")

        # https://pytorch.org/docs/stable/notes/fsdp.html#fsdp-notes
        # unshard the root FSDP module
        if self.world_size > 1 and fsdp_version(self.actor.actor_module) == 1:
            self.actor.actor_module._handle.reshard(True)

        if self._is_offload_param:
            offload_fsdp_model_to_cpu(self.actor_module_fsdp)
            log_gpu_memory_usage("After offload actor model during compute_log_prob", logger=logger)

        return output

    # ------------------------------------------------------------------
    # joint decoding: the LIVE actor is the student, a frozen model is the teacher
    # ------------------------------------------------------------------
    def _joint_teacher(self, cfg):
        """Return the frozen teacher, loading it on first use.

        Plain HF, deliberately NOT FSDP-wrapped. Per-token fusion needs both models' full
        logit rows every step; a sharded teacher would all-gather per layer per token, and
        the teacher never receives a gradient so there is nothing to shard FOR.

        Loaded lazily rather than in init_model so that a run with joint decoding disabled
        pays nothing, and so the load happens after vLLM has sized its kv cache.

        ``cfg`` is the joint_decode config node, handed in by generate_joint -- this worker is
        constructed with ``config.actor_rollout_ref``, so it cannot reach the top-level
        ``config.joint_decode`` itself.
        """
        if getattr(self, "_teacher_module", None) is not None:
            return self._teacher_module

        from transformers import AutoModelForCausalLM

        local_path = copy_to_local(cfg.teacher_model_path)

        # Vocab equality is a HARD requirement, stricter than the rephraser path's. That one
        # exchanges token IDS; this one intersects the two models' logits INDEX BY INDEX, so
        # a mismatch silently fuses unrelated tokens and produces fluent nonsense with no
        # error anywhere.
        teacher_tokenizer = hf_tokenizer(local_path, trust_remote_code=cfg.get("trust_remote_code", True))
        if teacher_tokenizer.get_vocab() != self.tokenizer.get_vocab():
            raise ValueError(
                "joint decoding needs the teacher and the actor to share a vocabulary: "
                "per-token fusion compares their logits index by index, so a mismatch makes "
                f"every fused step meaningless.\n  actor:   {self.config.model.path}\n"
                f"  teacher: {cfg.teacher_model_path}"
            )

        # Try the requested attention backend, then fall back. flash-attention-2 is an
        # optional dependency and some architectures have no kernel; failing the whole run
        # over it would be the wrong trade when sdpa gives identical numbers.
        wanted = [cfg.attn_impl] if cfg.attn_impl != "auto" else ["flash_attention_2", "sdpa", "eager"]
        errors = []
        for impl in wanted:
            try:
                model = AutoModelForCausalLM.from_pretrained(
                    local_path,
                    torch_dtype=torch.bfloat16,
                    trust_remote_code=cfg.get("trust_remote_code", True),
                    attn_implementation=impl,
                )
            except (ValueError, ImportError, RuntimeError) as error:
                errors.append(f"{impl}: {type(error).__name__}: {error}")
                continue
            if self.rank == 0:
                print(f"[joint_decode] teacher attn_implementation={impl}", flush=True)
            # Starts on CPU when offloading is on; generate_joint moves it per step.
            self._teacher_module = model.eval()
            if not cfg.get("teacher_offload", True):
                self._teacher_module = self._teacher_module.to(get_device_id())
            return self._teacher_module
        raise RuntimeError(
            f"could not load the joint-decoding teacher from {cfg.teacher_model_path} with any "
            f"of {wanted}:\n  " + "\n  ".join(errors)
        )

    def _joint_student(self, cfg):
        """Return the plain-HF mirror of the actor, and refresh its weights from the actor.

        THIS IS THE GATHER-ONCE STEP, and it is the whole reason this path is affordable.

        The actor is FSDP1 FULL_SHARD with per-layer wrapping, so each rank holds 1/world_size
        of every layer. Decoding straight through the FSDP wrapper means each layer all-gathers
        and reshards inside EVERY forward -- roughly the whole model across the interconnect per
        TOKEN. (The obvious dodge, summon_full_params(recurse=True) around the loop, does not
        work: running a forward inside the context fires FSDP's own post-forward hooks, which
        reshard the nested handles, and the exit path then asserts they are still unsharded --
        "Expects tensor to be unsharded with size N but got N/world_size".)

        So instead: gather the full weights ONCE per step into an unsharded module and decode
        against that, with zero FSDP involvement for the thousands of steps that follow. This
        is exactly what fsdp_vllm.py does to hand weights to the vLLM engine, and it puts the
        student in the same state the offline evaluator's model is in.

        ``summon_full_params`` is safe HERE because no forward runs inside it -- the only thing
        happening is a state_dict read, the same use layered_summon_lora_params makes of it.

        Costs one full copy of the weights resident per rank (~8GB for a 4B model in bf16) plus
        one gather + copy per step. Both are paid once per step rather than once per token.
        """
        from transformers import AutoConfig, AutoModelForCausalLM

        if getattr(self, "_student_module", None) is None:
            # from_config, NOT from_pretrained: every parameter is about to be overwritten from
            # the actor, so reading the checkpoint off disk would be wasted I/O -- and stale,
            # since the actor has been training.
            #
            # No init_empty_weights/meta device here. It would skip the real allocation, but
            # this repo consistently guards that on `not tie_word_embeddings` (see
            # get_init_weight_context_manager and its three call sites) because materialising a
            # meta-tensor model breaks tied embeddings, and load_state_dict(strict=True) would
            # then fail or silently untie them. from_config allocates for real, which costs one
            # extra allocation ONCE per run -- not per step.
            hf_config = AutoConfig.from_pretrained(
                copy_to_local(self.config.model.path),
                trust_remote_code=self.config.model.get("trust_remote_code", False),
            )
            wanted = [cfg.attn_impl] if cfg.attn_impl != "auto" else ["flash_attention_2", "sdpa", "eager"]
            errors = []
            for impl in wanted:
                try:
                    model = AutoModelForCausalLM.from_config(
                        hf_config, torch_dtype=torch.bfloat16, attn_implementation=impl
                    )
                except (ValueError, ImportError, RuntimeError) as error:
                    errors.append(f"{impl}: {type(error).__name__}: {error}")
                    continue
                if self.rank == 0:
                    print(f"[joint_decode] student mirror attn_implementation={impl} "
                          f"(unsharded copy of the actor, refreshed each step)", flush=True)
                self._student_module = model.to(get_device_id()).eval()
                break
            else:
                raise RuntimeError(
                    "could not build the joint-decoding student mirror with any of "
                    f"{wanted}:\n  " + "\n  ".join(errors)
                )

        # Refresh EVERY step: the actor's weights change with each update, and a stale mirror
        # would silently decode against an old policy -- which would also break the
        # off_ratio == Z_t identity that is this path's only end-to-end check.
        #
        # get_fsdp_full_state_dict rather than summon_full_params + state_dict(). The hand-rolled
        # version returns DTensors, and copy_ refuses to mix them with plain tensors
        # ("got mixed torch.Tensor and DTensor"). This helper sets FULL_STATE_DICT for the read,
        # which is what makes the tensors whole and plain, and it handles FSDP1 and FSDP2.
        #
        # rank0_only=False: every rank decodes its own shard of the questions, so every rank
        # needs the full weights, not just rank 0. offload_to_cpu=False keeps the round trip on
        # the GPU -- the mirror lives there and this runs once per step, not per token.
        full_sd = get_fsdp_full_state_dict(self.actor_module_fsdp, offload_to_cpu=False, rank0_only=False)
        self._student_module.load_state_dict(full_sd, strict=True, assign=False)
        del full_sd
        return self._student_module

    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
    @DistProfiler.annotate(color="magenta")
    @torch.no_grad()
    def generate_joint(self, data: DataProto) -> DataProto:
        """Decode with the live actor as student and a frozen teacher as constraint.

        Produces the summarize-replacement candidate, plus the per-token log Z_t (the
        teacher constraint's cost). Runs on the SAME GPUs as everything else, after the
        rollout, when vLLM is asleep.

        ``_is_actor``: the student must be the model receiving gradients.

        In:  ``student_input_ids`` / ``student_attention_mask`` (left-padded; the prompt the
             LOSS will use) and ``teacher_input_ids`` / ``teacher_attention_mask``
             (left-padded, may differ in length -- only the generated suffix has to align,
             and it does because both models are fed the same sampled token each step).
        Out: ``responses`` and ``off_log_z``, both [B, max_response_length] and aligned
             index-for-index, plus three per-row diagnostics.
        """
        from verl.PICO.joint_decode_core import joint_generate

        assert self._is_actor, (
            f"joint decoding needs the trained actor as its student, but this worker's role "
            f"is {self.role!r}"
        )
        # The joint_decode settings arrive through meta_info, not through self.config: this
        # worker is constructed with config.actor_rollout_ref, so the top-level
        # config.joint_decode node is not reachable from here. joint_sr.generate puts it on
        # the request, and DataProto.chunk copies meta_info to every shard.
        cfg = OmegaConf.create(data.meta_info["joint_decode_config"])
        device = get_device_id()
        width = int(data.meta_info["max_response_length"])

        if self._is_offload_param:
            load_fsdp_model_to_gpu(self.actor_module_fsdp)
        # nn.Module.to() moves params in place and returns self, so this and
        # self._teacher_module stay the same object -- which is what lets the finally block
        # below send it back to CPU without having to thread the local through.
        teacher = self._joint_teacher(cfg)
        if cfg.get("teacher_offload", True):
            teacher.to(device)

        data = data.to(device)
        student_batch = {
            "input_ids": data.batch["student_input_ids"],
            "attention_mask": data.batch["student_attention_mask"],
        }
        teacher_batch = {
            "input_ids": data.batch["teacher_input_ids"],
            "attention_mask": data.batch["teacher_attention_mask"],
        }
        n_rows = student_batch["input_ids"].size(0)

        responses = torch.full((n_rows, width), self.tokenizer.pad_token_id, dtype=torch.long)
        off_log_z = torch.zeros((n_rows, width), dtype=torch.float32)
        lengths = torch.zeros(n_rows, dtype=torch.long)
        fallback_frac = torch.zeros(n_rows, dtype=torch.float32)
        keep_ratio = torch.zeros(n_rows, dtype=torch.float32)
        student_logp = torch.zeros(n_rows, dtype=torch.float32)

        eos = self.tokenizer.eos_token_id
        eos_ids = list(eos) if isinstance(eos, (list, tuple)) else [eos]
        pad_id = self.tokenizer.pad_token_id
        if pad_id is None:
            pad_id = eos_ids[0]

        # The student is the plain-HF mirror, refreshed from the actor here. Everything below
        # therefore runs with ZERO FSDP involvement -- no summon context, no per-layer
        # all-gather, no train()/eval() or gradient-checkpointing state to juggle on the
        # actor. See _joint_student for why the alternatives do not work.
        student = self._joint_student(cfg)

        # All of this rank's rows in one decode. The rows are already a split: generate_joint
        # is DP_COMPUTE_PROTO, so DataProto.chunk gave this rank W/world_size of the
        # questions. joint_decode.batch_size is what that per-rank count is set against.
        n_narrow = 0
        try:
            with torch.autocast(device_type=get_device_name(), dtype=torch.bfloat16):
                # The trailing per-token Z_t is for offline sweeps taking a quantile over it;
                # the row mean below is all the training metrics need.
                out_ids, out_len, fb_row, logp_row, z_row, log_z, n_narrow, _z_tok = joint_generate(
                    student, teacher, student_batch, teacher_batch,
                    eos_ids=eos_ids, pad_id=pad_id, args=cfg,
                )

                # joint_generate returns only the columns it actually ran (it breaks
                # early once every row finished), and max_new_tokens may exceed the
                # trainer's width. Truncating is correct: a longer response cannot be
                # represented in the batch anyway.
                keep = min(out_ids.size(1), width)
                responses[:, :keep] = out_ids[:, :keep]
                off_log_z[:, :keep] = log_z[:, :keep]
                out_len = out_len.clamp(max=keep)
                lengths[:] = out_len

                # 0 rather than NaN for an empty row: these go into a DataProto that
                # gets padded, chunked and concatenated, and a NaN would poison every
                # batch mean computed over it.
                denom = out_len.clamp(min=1).to(torch.float32)
                fallback_frac[:] = fb_row.to(torch.float32) / denom
                keep_ratio[:] = (z_row / denom.double()).to(torch.float32)
                student_logp[:] = (logp_row / denom.double()).to(torch.float32)
        finally:
            # The actor's own train()/eval() and gradient-checkpointing state are untouched by
            # this path -- the mirror is what decodes -- so there is nothing to restore there.
            # Only the teacher goes back to CPU, even if decoding raised.
            if cfg.get("teacher_offload", True) and self._teacher_module is not None:
                self._teacher_module.to("cpu")
            get_torch_device().empty_cache()

        # Zero past each row's end. exp(0) = 1, so a token that was never emitted gets a
        # neutral coefficient rather than a spurious one -- and the loss masks it out anyway,
        # but a stray value here would be a weight on a pad token, so it is enforced.
        valid = torch.arange(width).unsqueeze(0) < lengths.unsqueeze(1)
        off_log_z = off_log_z * valid
        assert off_log_z.shape == responses.shape, "log Z_t must align with tokens 1:1"

        if n_narrow and self.rank == 0:
            print(f"[joint_decode] top-p nucleus exceeded the candidate cap on {n_narrow:,} "
                  f"row-steps; sampling was narrower than top_p={cfg.top_p} asked for", flush=True)

        output = DataProto.from_dict(
            tensors={
                "responses": responses,
                "off_log_z": off_log_z,
                "response_lengths": lengths,
                "fallback_frac": fallback_frac,
                "teacher_keep_ratio": keep_ratio,
                "student_mean_logp": student_logp,
            }
        )

        # https://pytorch.org/docs/stable/notes/fsdp.html#fsdp-notes
        # Reshard the root handle. summon_full_params in _joint_student unshards it, and
        # although its own exit path reshards, compute_log_prob does this explicitly for the
        # same reason (the root handle can be left unsharded after a gather) -- following the
        # existing convention rather than assuming.
        if self.world_size > 1 and fsdp_version(self.actor.actor_module) == 1:
            self.actor.actor_module._handle.reshard(True)
        if self._is_offload_param:
            offload_fsdp_model_to_cpu(self.actor_module_fsdp)
            log_gpu_memory_usage("After offload actor model during generate_joint", logger=logger)

        return output

    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
    @DistProfiler.annotate(color="olive")
    def compute_ref_log_prob(self, data: DataProto):
        if self._is_lora:
            # if _is_lora, actor without lora applied is the ref
            data.meta_info["is_lora"] = True
            data = self.compute_log_prob(data)
            # this old_log_probs is in fact ref_log_prob
            data = DataProto.from_dict(tensors={"ref_log_prob": data.batch["old_log_probs"]})
            return data
        assert self._is_ref
        # else:
        # otherwise, the class have a standalone ref model
        # Support all hardwares
        data = data.to(get_device_id())

        micro_batch_size = self.config.ref.log_prob_micro_batch_size_per_gpu
        data.meta_info["micro_batch_size"] = micro_batch_size
        data.meta_info["temperature"] = self.config.rollout.temperature
        data.meta_info["max_token_len"] = self.config.ref.log_prob_max_token_len_per_gpu
        data.meta_info["use_dynamic_bsz"] = self.config.ref.log_prob_use_dynamic_bsz
        with self.ulysses_sharding_manager:
            data = self.ulysses_sharding_manager.preprocess_data(data)
            output, _ = self.ref_policy.compute_log_prob(data=data, calculate_entropy=False)
            output = DataProto.from_dict(tensors={"ref_log_prob": output})
            output = self.ulysses_sharding_manager.postprocess_data(output)

        output = output.to("cpu")

        # https://pytorch.org/docs/stable/notes/fsdp.html#fsdp-notes
        # unshard the root FSDP module
        if self.world_size > 1 and fsdp_version(self.ref_policy.actor_module) == 1:
            self.ref_policy.actor_module._handle.reshard(True)

        return output

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def save_checkpoint(self, local_path, hdfs_path=None, global_step=0, max_ckpt_to_keep=None):
        from verl.utils.logger import log_with_rank

        # only support save and load ckpt for actor
        assert self._is_actor

        if self._is_offload_param:
            load_fsdp_model_to_gpu(self.actor_module_fsdp)

        self.checkpoint_manager.save_checkpoint(local_path=local_path, hdfs_path=hdfs_path, global_step=global_step, max_ckpt_to_keep=max_ckpt_to_keep)
        dist.barrier()

        if self._is_lora and hasattr(getattr(self, "actor_module", self.actor_module_fsdp), "peft_config"):
            lora_save_path = os.path.join(local_path, "lora_adapter")
            peft_model = getattr(self, "actor_module", self.actor_module_fsdp)
            peft_config = {}
            if dist.get_rank() == 0:
                os.makedirs(lora_save_path, exist_ok=True)
                peft_config = asdict(peft_model.peft_config.get("default", {}))
                peft_config["task_type"] = peft_config["task_type"].value
                peft_config["peft_type"] = peft_config["peft_type"].value
                peft_config["target_modules"] = list(peft_config["target_modules"])
            try:
                if fsdp_version(self.actor_module_fsdp) > 0:
                    self.actor_module_fsdp = self.actor_module_fsdp.to(get_device_name())
                    lora_params = layered_summon_lora_params(self.actor_module_fsdp)
                    if dist.get_rank() == 0:
                        save_file(lora_params, os.path.join(lora_save_path, "adapter_model.safetensors"))
                        with open(os.path.join(lora_save_path, "adapter_config.json"), "w", encoding="utf-8") as f:
                            json.dump(peft_config, f, ensure_ascii=False, indent=4)
            except Exception as e:
                log_with_rank(f"Save LoRA Adapter Error ({e})", rank=dist.get_rank(), logger=logger, log_only_rank_0=True)

            dist.barrier()
            log_with_rank(f"[rank-{self.rank}]: Saved LoRA adapter to: {lora_save_path}", rank=dist.get_rank(), logger=logger, log_only_rank_0=True)

        if self._is_offload_param:
            offload_fsdp_model_to_cpu(self.actor_module_fsdp)

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def load_checkpoint(self, local_path, hdfs_path=None, del_local_after_load=False):
        assert self._is_actor or (not self._is_actor and self._is_rollout), f"Checkpoint loading is only supported for Actor or standalone Rollout Workers, but got {self._is_actor} and {self._is_rollout}"

        if self._is_offload_param:
            load_fsdp_model_to_gpu(self.actor_module_fsdp)

        self.checkpoint_manager.load_checkpoint(local_path=local_path, hdfs_path=hdfs_path, del_local_after_load=del_local_after_load)

        if self._is_offload_param:
            offload_fsdp_model_to_cpu(self.actor_module_fsdp)

        if self._is_offload_optimizer:
            offload_fsdp_optimizer(self.actor_optimizer)

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def start_profile(self) -> None:
        """Start profiling for the current rank in the current training step."""
        self.profiler.start()

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def stop_profile(self) -> None:
        """Stop profiling for the current rank in the current training step."""
        self.profiler.stop()

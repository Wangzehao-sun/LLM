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
Note that we don't combine the main with ray_trainer as ray_trainer is used by other main.
"""

import os
import socket

import hydra
import ray
from omegaconf import OmegaConf

from verl.trainer.ppo.ray_trainer import RayPPOTrainer
from verl.trainer.ppo.reward import load_reward_manager


def _load_joint_decode_config(config):
    """Merge verl/custom/config/joint_decode.yaml into ``config.joint_decode``.

    Joint decoding's knobs live in their own file, not in ppo_trainer.yaml. The point is
    that ppo_trainer.yaml stays UNMODIFIED by this feature: nothing changes for any
    existing script, and the whole subsystem is one file plus one flag.

    That rules out a Hydra ``defaults:`` entry (which would mean editing ppo_trainer.yaml
    to list it), so the file is loaded explicitly and merged with whatever the user passed
    as ``+joint_decode.*``. Merge order puts the CLI last, so overrides win.

    The result is written back into ``config`` under ``joint_decode``, so the trainer can
    read ``self.config.joint_decode.*`` like any other node instead of being handed a
    second config object to thread around.

    Returns the merged node. Absent file plus absent overrides yields ``enable: False``,
    which is the inert path.
    """
    from pathlib import Path

    from omegaconf import open_dict

    defaults_path = Path(__file__).resolve().parent.parent / "custom" / "config" / "joint_decode.yaml"
    defaults = OmegaConf.load(defaults_path) if defaults_path.exists() else OmegaConf.create()
    merged = OmegaConf.merge(defaults, config.get("joint_decode", OmegaConf.create()))
    # struct mode is on by default for a Hydra config, so a new top-level key has to be
    # added through open_dict.
    with open_dict(config):
        config.joint_decode = merged
    return merged


@hydra.main(config_path="config", config_name="ppo_trainer", version_base=None)
def main(config):
    run_ppo(config)


# Define a function to run the PPO-like training process
def run_ppo(config) -> None:
    # Check if Ray is not initialized
    if not ray.is_initialized():
        # Initialize Ray with a local cluster configuration
        # Set environment variables in the runtime environment to control tokenizer parallelism,
        # NCCL debug level, VLLM logging level, and allow runtime LoRA updating
        # `num_cpus` specifies the number of CPU cores Ray can use, obtained from the configuration
        ray.init(
            runtime_env={"env_vars": {"TOKENIZERS_PARALLELISM": "true", "NCCL_DEBUG": "WARN", "VLLM_LOGGING_LEVEL": "WARN", "VLLM_ALLOW_RUNTIME_LORA_UPDATING": "true"}},
            num_cpus=config.ray_init.num_cpus,
        )

    # Create a remote instance of the TaskRunner class, and
    # Execute the `run` method of the TaskRunner instance remotely and wait for it to complete
    if OmegaConf.select(config.trainer, "profile_steps") is not None and len(OmegaConf.select(config.trainer, "profile_steps")) > 0:
        nsight_options = OmegaConf.to_container(config.trainer.controller_nsight_options)
        runner = TaskRunner.options(runtime_env={"nsight": nsight_options}).remote()
    else:
        runner = TaskRunner.remote()
    ray.get(runner.run.remote(config))

    # [Optional] get the path of the timeline trace file from the configuration, default to None
    # This file is used for performance analysis
    timeline_json_file = config.ray_init.get("timeline_json_file", None)
    if timeline_json_file:
        ray.timeline(filename=timeline_json_file)


@ray.remote(num_cpus=1)  # please make sure main_task is not scheduled on head
class TaskRunner:
    def run(self, config):
        # Print the initial configuration. `resolve=True` will evaluate symbolic values.
        from pprint import pprint

        from omegaconf import OmegaConf

        from verl.utils.fs import copy_to_local

        print(f"TaskRunner hostname: {socket.gethostname()}, PID: {os.getpid()}")

        pprint(OmegaConf.to_container(config, resolve=True))

        OmegaConf.resolve(config)

        # Download the checkpoint from HDFS to the local machine.
        # `use_shm` determines whether to use shared memory, which could lead to faster model loading if turned on
        local_path = copy_to_local(config.actor_rollout_ref.model.path, use_shm=config.actor_rollout_ref.model.get("use_shm", False))

        # Instantiate the tokenizer and processor.
        from verl.utils import hf_processor, hf_tokenizer

        trust_remote_code = config.data.get("trust_remote_code", False)
        tokenizer = hf_tokenizer(local_path, trust_remote_code=trust_remote_code)
        # Used for multimodal LLM, could be None
        processor = hf_processor(local_path, trust_remote_code=trust_remote_code, use_fast=True)

        # Version validation for vllm.
        if config.actor_rollout_ref.rollout.name in ["vllm"]:
            from verl.utils.vllm_utils import is_version_ge

            if config.actor_rollout_ref.model.get("lora_rank", 0) > 0:
                if not is_version_ge(pkg="vllm", minver="0.7.3"):
                    raise NotImplementedError("PPO LoRA is not supported before vllm 0.7.3")

        # Define worker classes based on the actor strategy.
        if config.actor_rollout_ref.actor.strategy in ["fsdp", "fsdp2"]:
            assert config.critic.strategy in ["fsdp", "fsdp2"]
            from verl.single_controller.ray import RayWorkerGroup
            from verl.workers.fsdp_workers import ActorRolloutRefWorker, AsyncActorRolloutRefWorker, CriticWorker
            from verl.custom.fsdp_workers_new import NewActorRolloutRefWorker
            #actor_rollout_cls = AsyncActorRolloutRefWorker if config.actor_rollout_ref.rollout.mode == "async" else ActorRolloutRefWorker
            actor_rollout_cls = AsyncActorRolloutRefWorker if config.actor_rollout_ref.rollout.mode == "async" else NewActorRolloutRefWorker
            ray_worker_group_cls = RayWorkerGroup

        elif config.actor_rollout_ref.actor.strategy == "megatron":
            assert config.actor_rollout_ref.actor.strategy == config.critic.strategy
            from verl.single_controller.ray.megatron import NVMegatronRayWorkerGroup
            from verl.workers.megatron_workers import ActorRolloutRefWorker, AsyncActorRolloutRefWorker, CriticWorker

            actor_rollout_cls = AsyncActorRolloutRefWorker if config.actor_rollout_ref.rollout.mode == "async" else ActorRolloutRefWorker
            ray_worker_group_cls = NVMegatronRayWorkerGroup

        else:
            raise NotImplementedError

        from verl.trainer.ppo.ray_trainer import ResourcePoolManager, Role

        # Map roles to their corresponding remote worker classes.
        role_worker_mapping = {
            Role.ActorRollout: ray.remote(actor_rollout_cls),
            Role.Critic: ray.remote(CriticWorker),
        }

        # Define the resource pool specification.
        # Map roles to the resource pool.
        global_pool_id = "global_pool"
        resource_pool_spec = {
            global_pool_id: [config.trainer.n_gpus_per_node] * config.trainer.nnodes,
        }
        mapping = {
            Role.ActorRollout: global_pool_id,
            Role.Critic: global_pool_id,
        }

        # We should adopt a multi-source reward function here:
        # - for rule-based rm, we directly call a reward score
        # - for model-based rm, we call a model
        # - for code related prompt, we send to a sandbox if there are test cases
        # finally, we combine all the rewards together
        # The reward type depends on the tag of the data
        if config.reward_model.enable:
            if config.reward_model.strategy in ["fsdp", "fsdp2"]:
                from verl.workers.fsdp_workers import RewardModelWorker
            elif config.reward_model.strategy == "megatron":
                from verl.workers.megatron_workers import RewardModelWorker
            else:
                raise NotImplementedError
            role_worker_mapping[Role.RewardModel] = ray.remote(RewardModelWorker)
            mapping[Role.RewardModel] = global_pool_id

        # Add a reference policy worker if KL loss or KL reward is used.
        if config.algorithm.use_kl_in_reward or config.actor_rollout_ref.actor.use_kl_loss:
            role_worker_mapping[Role.RefPolicy] = ray.remote(ActorRolloutRefWorker)
            mapping[Role.RefPolicy] = global_pool_id
        if config.se_model.enable:
            role_worker_mapping[Role.ActorRolloutSE] = ray.remote(actor_rollout_cls)
            mapping[Role.ActorRolloutSE] = global_pool_id
        # Frozen rephraser: a SECOND model that rolls out the long summarize prompt and
        # supplies the proposal density for the importance ratio, but is never updated.
        #
        # rephraser.n_gpus_per_node decides which of two very different setups you get:
        #
        #   > 0  -- its OWN resource pool, hence its own Ray actor process, hence its own
        #           vLLM engine. This is what makes online generation possible at all:
        #           vLLM's sleep mode is built on a process-global CuMemAllocator whose
        #           tags are the hardcoded strings "weights"/"kv_cache", so two engines in
        #           one process free each other's memory (SGLang has the same problem via
        #           torch_memory_saver, and change_current_allocator can only be called
        #           once per process -- switching backends does not help). Separate pools
        #           mean separate processes, which is the pattern vLLM itself recommends.
        #   == 0 -- shares the global pool, so it CANNOT hold an engine. Log-probs only,
        #           with candidates pre-generated offline by
        #           Data/aggregate_sr_responses.py.
        #
        # Default 0 keeps every existing script byte-identical: the spec stays a single
        # pool and every role still maps to it.
        if config.get("rephraser", {}).get("enable", False):
            role_worker_mapping[Role.ActorRolloutRephraser] = ray.remote(actor_rollout_cls)
            rephraser_gpus = config.rephraser.get("n_gpus_per_node", 0)
            if rephraser_gpus > 0:
                rephraser_pool_id = "rephraser_pool"
                resource_pool_spec[rephraser_pool_id] = [rephraser_gpus] * config.trainer.nnodes
                mapping[Role.ActorRolloutRephraser] = rephraser_pool_id
                print(
                    f"[rephraser] dedicated pool {rephraser_pool_id!r}: {rephraser_gpus} GPU(s) x "
                    f"{config.trainer.nnodes} node(s), on top of the reasoner's "
                    f"{config.trainer.n_gpus_per_node}. CUDA_VISIBLE_DEVICES must expose all of "
                    f"them; trainer.n_gpus_per_node counts the reasoner's only."
                )
            else:
                mapping[Role.ActorRolloutRephraser] = global_pool_id

        # Joint decoding: two FROZEN models that decode together, fusing per token, to
        # produce the summarize-replacement candidate. Its knobs live in their own file
        # (verl/custom/config/joint_decode.yaml) rather than in ppo_trainer.yaml, and are
        # loaded here so that file's defaults exist before anything reads them. CLI
        # `+joint_decode.<key>=` overrides layer on top and win.
        #
        # Resolving it into `config` (rather than passing it around separately) is what
        # lets the trainer read `self.config.joint_decode.*` like any other node. With
        # `enable: False` -- the default, and the value when the node is absent entirely
        # -- nothing below runs and the resource spec is untouched.
        joint_cfg = _load_joint_decode_config(config)
        if joint_cfg.get("enable", False):
            from verl.custom.joint_decode_worker import JointDecodeWorker

            role_worker_mapping[Role.JointDecode] = ray.remote(JointDecodeWorker)
            joint_gpus = joint_cfg.get("n_gpus_per_node", 0)
            if joint_gpus > 0:
                joint_pool_id = "joint_decode_pool"
                resource_pool_spec[joint_pool_id] = [joint_gpus] * config.trainer.nnodes
                mapping[Role.JointDecode] = joint_pool_id
                print(
                    f"[joint_decode] dedicated pool {joint_pool_id!r}: {joint_gpus} GPU(s) x "
                    f"{config.trainer.nnodes} node(s), on top of the reasoner's "
                    f"{config.trainer.n_gpus_per_node}. CUDA_VISIBLE_DEVICES must expose all of "
                    f"them; trainer.n_gpus_per_node counts the reasoner's only."
                )
            else:
                mapping[Role.JointDecode] = global_pool_id
                print(
                    "[joint_decode] sharing the reasoner's pool: two unsharded bf16 models plus "
                    "their KV caches come out of the same budget vLLM already reserved via "
                    "gpu_memory_utilization, so lower that to match or expect an OOM at the "
                    "first joint step."
                )
        # Load the reward manager for training and validation.
        reward_fn = load_reward_manager(config, tokenizer, num_examine=0, **config.reward_model.get("reward_kwargs", {}))
        val_reward_fn = load_reward_manager(config, tokenizer, num_examine=1, **config.reward_model.get("reward_kwargs", {}))
        resource_pool_manager = ResourcePoolManager(resource_pool_spec=resource_pool_spec, mapping=mapping)

        from verl.utils.dataset.rl_dataset import collate_fn

        # Create training and validation datasets.
        #train_dataset = create_rl_dataset(config.data.train_files, config.data, tokenizer, processor)
        #val_dataset = create_rl_dataset(config.data.val_files, config.data, tokenizer, processor)
        #train_sampler = create_rl_sampler(config.data, train_dataset)

        from verl.custom.new_ray_trainer import NewRayPPOTrainer
        # Initialize the PPO trainer.
        trainer = NewRayPPOTrainer(
            config=config,
            tokenizer=tokenizer,
            role_worker_mapping=role_worker_mapping,
            resource_pool_manager=resource_pool_manager,
            ray_worker_group_cls=ray_worker_group_cls,
            reward_fn=reward_fn,
            val_reward_fn=val_reward_fn,
        )
        # Initialize the workers of the trainer.
        trainer.init_workers()
        # Start the training process.
        trainer.fit()


def create_rl_dataset(data_paths, data_config, tokenizer, processor):
    """Create a dataset.

    Arguments:
        data_paths: List of paths to data files.
        data_config: The data config.
        tokenizer (Tokenizer): The tokenizer.
        processor (Processor): The processor.

    Returns:
        dataset (Dataset): The dataset.
    """
    from torch.utils.data import Dataset

    from verl.utils.dataset.rl_dataset import RLHFDataset

    # Check if a custom dataset class is specified in the data configuration
    # and if the path to the custom class is provided
    if "custom_cls" in data_config and data_config.custom_cls.get("path", None) is not None:
        from verl.utils.import_utils import load_extern_type

        # Dynamically load the custom dataset class
        dataset_cls = load_extern_type(data_config.custom_cls.path, data_config.custom_cls.name)
        # Verify that the custom dataset class inherits from torch.utils.data.Dataset
        if not issubclass(dataset_cls, Dataset):
            raise TypeError(f"The custom dataset class '{data_config.custom_cls.name}' from '{data_config.custom_cls.path}' must inherit from torch.utils.data.Dataset")
    else:
        # Use the default RLHFDataset class if no custom class is specified
        dataset_cls = RLHFDataset
    print(f"Using dataset class: {dataset_cls.__name__}")

    # Instantiate the dataset using the determined dataset class
    dataset = dataset_cls(
        data_files=data_paths,
        tokenizer=tokenizer,
        processor=processor,
        config=data_config,
    )

    return dataset


def create_rl_sampler(data_config, dataset):
    """Create a sampler for the dataset.

    Arguments:
        data_config: The data config.
        dataset (Dataset): The dataset.

    Returns:
        sampler (Sampler): The sampler.
    """
    import torch
    from torch.utils.data import RandomSampler, SequentialSampler

    # Use a sampler to facilitate checkpoint resumption.
    # If shuffling is enabled in the data configuration, create a random sampler.
    if data_config.shuffle:
        train_dataloader_generator = torch.Generator()
        train_dataloader_generator.manual_seed(data_config.get("seed", 1))
        sampler = RandomSampler(data_source=dataset, generator=train_dataloader_generator)
    else:
        # If shuffling is disabled, use a sequential sampler to iterate through the dataset in order.
        sampler = SequentialSampler(data_source=dataset)

    return sampler


if __name__ == "__main__":
    main()


CUDA_VISIBLE_DEVICES=4,5,6,7 python legacy_model_merger.py merge \
    --backend fsdp \
    --local_dir /home/zhwang/LLM/Train/verl/checkpoints/train_vanilla_dynamic_luffy_Qwen2.5-7B_openr1_seprompt/training/global_step_500/actor \
    --local_dir /home/zhwang/LLM/Train/verl/checkpoints/train_vanilla_dynamic_luffy_Qwen2.5-7B_openr1_seprompt/training/global_step_500/actor \
    --target_dir /home/zhwang/LLM/Train/verl/checkpoints/iof500\

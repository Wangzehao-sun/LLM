set -x


unset ROCR_VISIBLE_DEVICES

export RAY_DEDUP_LOGS=0
#data1_path=$HOME/LLM/Train/data/bs_17k.parquet
data_path=$HOME/LLM/Train/data_0306/openr1_prompt_rewritten_nothinking.parquet
MODEL_DIR=/home/shared
MODEL_PATH=$MODEL_DIR/${1:-"Qwen2.5-Math-7B-16k-think"}
#MODEL_PATH=${1:-"$HOME/Model/Qwen2.5-Math-7B-16k-think"} #Model/Qwen2.5-Math-7B-16k-think
PROJECT_NAME="self-explain_$(basename $MODEL_PATH)_$(basename $data_path .parquet)"
LOG_DIR=$HOME/LLM/Train/verl/logs/${PROJECT_NAME}
mkdir -p ${LOG_DIR}
save_path=${LOG_DIR}/save_data/
LOG_PATH=${LOG_DIR}/generation.log
GPU_NUM=2
TENSOR_PARALLEL=1


cd $HOME/LLM/Train/verl/
echo "change to dir: $PWD"



python -m verl.trainer.main_generation \
    trainer.nnodes=1 \
    trainer.n_gpus_per_node=$GPU_NUM \
    data.path=$data_path \
    data.prompt_key=prompt \
    +data.reward_model_key=reward_model \
    +data.data_source_key=data_source \
    data.n_samples=1 \
    data.batch_size=128 \
    data.output_path=$save_path \
    +data.se_logits=False \
    ref.log_prob_micro_batch_size=64 \
    ref.log_prob_max_token_len_per_gpu=24576 \
    model.path=$MODEL_PATH \
    +model.trust_remote_code=True \
    rollout.temperature=1 \
    rollout.top_k=-1 \
    rollout.top_p=1 \
    rollout.prompt_length=4096 \
    rollout.response_length=16384 \
    rollout.max_num_batched_tokens=32768 \
    rollout.tensor_model_parallel_size=$TENSOR_PARALLEL \
    rollout.gpu_memory_utilization=0.8 \
    +start_step=0 \
    +max_steps=10 \
    +is_eval=True \
    +reward_model.reward_impl_version=4 2>&1 | tee ${LOG_PATH} \

set -x


unset ROCR_VISIBLE_DEVICES

# Variant of generation.sh wired to the DeepMath subset produced by
# Data/filter_deepmath.py. Outputs per-batch parquets under
# logs/<PROJECT_NAME>/save_data/, each carrying a `test_score` field with
# {scores_per_response, mean_score, max_score}. Use Data/filter_by_accuracy.py
# afterwards to drop instances the model already solves.
data_path=${DATA_PATH:-$HOME/LLM/Data/deepmath_dgt6_n10000.parquet}
MODEL_DIR=/home/zhwang/LLM/Train/verl/checkpoints/
MODEL_PATH=$MODEL_DIR/${1:-"Qwen2.5-Math-7B-16k-think"} #Model/Qwen2.5-Math-7B-16k-think
PROJECT_NAME="eval_$(basename $MODEL_PATH)_$(basename $data_path .parquet)"
LOG_DIR=$HOME/LLM/Train/verl/logs/${PROJECT_NAME}
mkdir -p ${LOG_DIR}
save_path=${LOG_DIR}/save_data/
LOG_PATH=${LOG_DIR}/generation.log
GPU_NUM=4
TENSOR_PARALLEL=1
N_SAMPLES=${N_SAMPLES:-128}
MAX_STEPS=${MAX_STEPS:-100}

# Format screening on top of math-verify's answer check. The rule is the trainer's
# (_trajectory_filter_reject); these word lists are the generation side's own and do not
# affect training. Hydra list syntax: no spaces after the commas.
TRAJ_FILTER=${TRAJ_FILTER:-False}
TRAJ_KEYWORDS=${TRAJ_KEYWORDS:-'["the draft","based on the draft","according to the draft","from the draft","the experience","based on the experience","according to the experience","the provided reasoning","based on the reasoning above"]'}
TRAJ_INSTR_PHRASES=${TRAJ_INSTR_PHRASES:-'["your task is","output only","do not mention","self-contained solution","no meta-talk"]'}


cd $HOME/LLM/Train/verl/
echo "change to dir: $PWD"



python -m verl.trainer.main_generation \
    trainer.nnodes=1 \
    trainer.n_gpus_per_node=$GPU_NUM \
    data.path=$data_path \
    data.prompt_key=prompt \
    +data.reward_model_key=reward_model \
    +data.data_source_key=data_source \
    data.n_samples=$N_SAMPLES \
    data.batch_size=256 \
    data.output_path=$save_path \
    model.path=$MODEL_PATH \
    +model.trust_remote_code=True \
    rollout.temperature=0.6 \
    rollout.top_k=-1 \
    rollout.top_p=0.95 \
    rollout.prompt_length=2048 \
    rollout.response_length=14336 \
    rollout.max_num_batched_tokens=32768 \
    rollout.tensor_model_parallel_size=$TENSOR_PARALLEL \
    rollout.gpu_memory_utilization=0.8 \
    +is_eval=True \
    +max_steps=$MAX_STEPS \
    +algorithm.trajectory_filter.enable=$TRAJ_FILTER \
    +algorithm.trajectory_filter.keywords="$TRAJ_KEYWORDS" \
    +algorithm.trajectory_filter.instruction_phrases="$TRAJ_INSTR_PHRASES" \
    +reward_model.reward_impl_version=4 2>&1 | tee ${LOG_PATH}

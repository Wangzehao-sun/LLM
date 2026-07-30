#!/usr/bin/env bash
#
# train_grpo_summarize_teacher.sh
#
# Teacher-API rephraser variant of train_grpo_summarize.sh.
#
# The summarize-replacement candidates are generated ONLINE by an external strong
# model (OpenAI-compatible /v1/chat/completions) instead of the local policy. The
# teacher's TEXT is re-tokenized with the policy tokenizer and injected as off-policy
# rows in the SAME batch format as the current summarize path. There is no behavior
# logprob for the teacher content (off_old_log_probs / target_probs stay zero), so the
# off-row loss must be a behavior-logprob-free mode; the base script already uses
# off_policy_reshape="vanilla" (compatible). Do NOT switch it to 'luffy' — the trainer
# asserts against it.
#
# ALL teacher-API params live in a STANDALONE file, NOT in the trainer/Hydra config:
#     Myverl/verl/custom/config/teacher_api.yaml     (set enable: True there)
# Override the file location with env TEACHER_API_CONFIG=/path/to/your.yaml
# Provide the api key via env (never commit secrets): TEACHER_API_KEY / DASHSCOPE_API_KEY
#
# Enabling teacher_api (enable: True) automatically makes the dataset keep the raw
# summarize messages — no extra data flag needed.
#
# Usage:
#   # 1) edit teacher_api.yaml: enable: True, set url/model/...
#   # 2) run (forwards all extra args + the model subdir to the base script):
#   TEACHER_API_KEY=sk-... bash train_grpo_summarize_teacher.sh Qwen2.5-Math-7B-16k-think
#   TEACHER_API_CONFIG=/abs/my_teacher.yaml bash train_grpo_summarize_teacher.sh <model> trainer.total_epochs=1
set -x

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Default the standalone config path to the bundled one (client also defaults to it).
export TEACHER_API_CONFIG="${TEACHER_API_CONFIG:-$HERE/../../verl/custom/config/teacher_api.yaml}"

echo "[teacher] using TEACHER_API_CONFIG=$TEACHER_API_CONFIG"

# Delegate to the base summarize script unchanged; the trainer picks up teacher_api
# from the standalone file at init time. All args pass straight through.
exec bash "$HERE/train_grpo_summarize.sh" "$@"

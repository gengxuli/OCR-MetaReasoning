#!/usr/bin/env bash
# set -e

export PYTHONPATH="$(pwd):${PYTHONPATH}"

# Judge model credentials and endpoint.
export MODEL_API_KEY=""
export MODEL_BASE_URL=""

# Source inference result folder under OCR_MetaReasoning/evaluation/result/<source_model>_result
# SOURCE_MODEL_NAME="${SOURCE_MODEL_NAME:-gpt-5.4-mini}"


python OCR_MetaReasoning/evaluation/rpcs_judge.py \
    --source_model_name grok-4.2 \
    --judge_model_name gpt-5.4 \
    --base_url "https://openrouter.ai/v1" \
    --meta_reasoning_types all \
    --workers 64 \
    --batch_size 32 \
    --temperature 0.0 \
    --timeout 600.0 \
    --max_retries 3 \
    --top_p 1.0 \
    --max_tokens 2048

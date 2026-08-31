#!/usr/bin/env bash
# set -e

export PYTHONPATH="$(pwd):${PYTHONPATH}"

export MODEL_API_KEY=""


python OCR_MetaReasoning/evaluation/openai_infer.py \
    --model_name claude-sonnet-4-6 \
    --base_url https://openrouter.ai/v1 \
    --meta_reasoning_types all \
    --sample_num 500 \
    --batch_size 32 \
    --workers 32 \
    --temperature 1.0 \
    --top_p 1.0 \
    --top_k 0 \
    --repetition_penalty 0.0 \
    --presence_penalty 0.0 \
    --timeout 200.0 \
    # --max_tokens "${max_tokens}"
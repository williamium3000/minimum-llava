#!/bin/bash


NAME="llava-v1.5-7b-lora-unfreeze-clip-lr2e-6"
MODEL_PATH="checkpoints/llava-v1.5-7b-lora-unfreeze-clip-lr2e-6"

python -m llava.eval.model_vqa \
    --model-path $MODEL_PATH \
    --model-base checkpoints/llava-v1.5-7b-official \
    --question-file ./playground/data/eval/llava-bench-in-the-wild/questions.jsonl \
    --image-folder ./playground/data/eval/llava-bench-in-the-wild/images \
    --answers-file ./playground/data/eval/llava-bench-in-the-wild/answers/${NAME}.jsonl \
    --temperature 0 \
    --conv-mode vicuna_v1

mkdir -p playground/data/eval/llava-bench-in-the-wild/reviews

python llava/eval/eval_gpt_review_bench.py \
    --question playground/data/eval/llava-bench-in-the-wild/questions.jsonl \
    --context playground/data/eval/llava-bench-in-the-wild/context.jsonl \
    --rule llava/eval/table/rule.json \
    --answer-list \
        playground/data/eval/llava-bench-in-the-wild/answers_gpt4.jsonl \
        playground/data/eval/llava-bench-in-the-wild/answers/${NAME}.jsonl \
    --output \
        playground/data/eval/llava-bench-in-the-wild/reviews/${NAME}.jsonl

python llava/eval/summarize_gpt_review.py -f playground/data/eval/llava-bench-in-the-wild/reviews/${NAME}.jsonl

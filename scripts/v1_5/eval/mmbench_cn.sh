#!/bin/bash

SPLIT="mmbench_dev_cn_20231003"
NAME="llava-v1.5-7b-lora-unfreeze-clip-lr2e-6"
MODEL_PATH="checkpoints/llava-v1.5-7b-lora-unfreeze-clip-lr2e-6"


python -m llava.eval.model_vqa_mmbench \
    --model-path $MODEL_PATH \
    --model-base checkpoints/llava-v1.5-7b-official \
    --question-file ./playground/data/eval/mmbench/$SPLIT.tsv \
    --answers-file ./playground/data/eval/mmbench/answers/$SPLIT/$NAME.jsonl \
    --lang cn \
    --single-pred-prompt \
    --temperature 0 \
    --conv-mode vicuna_v1

mkdir -p playground/data/eval/mmbench/answers_upload/$SPLIT

python scripts/convert_mmbench_for_submission.py \
    --annotation-file ./playground/data/eval/mmbench/$SPLIT.tsv \
    --result-dir ./playground/data/eval/mmbench/answers/$SPLIT \
    --upload-dir ./playground/data/eval/mmbench/answers_upload/$SPLIT \
    --experiment $NAME

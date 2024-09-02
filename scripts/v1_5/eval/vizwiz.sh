#!/bin/bash

MODEL_PATH=checkpoints/llava-v1.5-7b-lora
EXP_NAME=llava-v1.5-7b-lora
python -m llava.eval.model_vqa_loader \
    --model-path $MODEL_PATH \
    --model-base checkpoints/llava-v1.5-7b-official \
    --question-file ./playground/data/eval/vizwiz/llava_val.jsonl \
    --image-folder ./playground/data/eval/vizwiz/val \
    --answers-file ./playground/data/eval/vizwiz/answers/$EXP_NAME.jsonl \
    --temperature 0 \
    --conv-mode vicuna_v1

python scripts/convert_vizwiz_for_submission.py \
    --annotation-file ./playground/data/eval/vizwiz/llava_val.jsonl \
    --result-file ./playground/data/eval/vizwiz/answers/$EXP_NAME.jsonl \
    --result-upload-file ./playground/data/eval/vizwiz/answers_upload/$EXP_NAME.json

python playground/data/eval/vizwiz/summary.py \
    --answer ./playground/data/eval/vizwiz/answers_upload/$EXP_NAME.json \
    --gt ./playground/data/eval/vizwiz/val.json
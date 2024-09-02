#!/bin/bash
NAME="llava-v1.5-7b-lora-unfreeze-clip-lr2e-6"
MODEL_PATH="checkpoints/llava-v1.5-7b-lora-unfreeze-clip-lr2e-6"

python -m llava.eval.model_vqa_loader \
    --model-path $MODEL_PATH \
    --model-base checkpoints/llava-v1.5-7b-official \
    --question-file ./playground/data/eval/pope/llava_pope_test.jsonl \
    --image-folder ./playground/data/eval/pope/val2014 \
    --answers-file ./playground/data/eval/pope/answers/$NAME.jsonl \
    --temperature 0 \
    --conv-mode vicuna_v1

python llava/eval/eval_pope.py \
    --annotation-dir ./playground/data/eval/pope/annotations \
    --question-file ./playground/data/eval/pope/llava_pope_test.jsonl \
    --result-file ./playground/data/eval/pope/answers/$NAME.jsonl

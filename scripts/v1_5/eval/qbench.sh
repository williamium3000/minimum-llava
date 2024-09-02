#!/bin/bash
NAME="llava-v1.5-7b-lora-unfreeze-clip-lr2e-6"
MODEL_PATH="checkpoints/llava-v1.5-7b-lora-unfreeze-clip-lr2e-6"

if [ "$1" = "dev" ]; then
    echo "Evaluating in 'dev' split."
elif [ "$1" = "test" ]; then
    echo "Evaluating in 'test' split."
else
    echo "Unknown split, please choose between 'dev' and 'test'."
    exit 1
fi

python -m llava.eval.model_vqa_qbench \
    --model-path $MODEL_PATH \
    --image-folder ./playground/data/eval/qbench/images_llvisionqa/ \
    --questions-file ./playground/data/eval/qbench/llvisionqa_${1}.json \
    --answers-file ./playground/data/eval/qbench/llvisionqa_${1}_answers_$NAME.jsonl \
    --conv-mode llava_v1 \
    --lang en

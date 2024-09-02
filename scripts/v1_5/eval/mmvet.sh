#!/bin/bash
NAME="llava-v1.5-7b-lora-unfreeze-clip-lr2e-6"
MODEL_PATH="checkpoints/llava-v1.5-7b-lora-unfreeze-clip-lr2e-6"

python -m llava.eval.model_vqa \
    --model-path $MODEL_PATH \
    --model-base checkpoints/llava-v1.5-7b-official \
    --question-file ./playground/data/eval/mm-vet/llava-mm-vet.jsonl \
    --image-folder ./playground/data/eval/mm-vet/images \
    --answers-file ./playground/data/eval/mm-vet/answers/$NAME.jsonl \
    --temperature 0 \
    --conv-mode vicuna_v1

mkdir -p ./playground/data/eval/mm-vet/results

python scripts/convert_mmvet_for_eval.py \
    --src ./playground/data/eval/mm-vet/answers/$NAME.jsonl \
    --dst ./playground/data/eval/mm-vet/results/$NAME.json

export OPENAI_API_KEY=None
unset all_proxy
python playground/data/eval/mm-vet/mm-vet_evaluator.py \
    --result_path playground/data/eval/mm-vet/results \
    --mmvet_path playground/data/eval/mm-vet \
    --result_file playground/data/eval/mm-vet/results/$NAME.json

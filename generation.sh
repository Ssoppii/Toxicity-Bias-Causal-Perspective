#!/bin/bash

NP=4
MASTER_PORT=12345

CON_DATASETS=("IMDB" "RealToxicityPrompts")
SUM_DATASETS=("CNN" "XSum")
CATEGORIES=("toxicity" "insult" "threat" "identity_attack")
MODES=("noperturb" "debiased")
# TOPNS=(3 5 10)
TOPNS=10
BATCH_SIZE=8
MAX_LENGTH=128
GEN_MAX_LENGTH=128

for dataset in "${CON_DATASETS[@]}"; do
    for category in "${CATEGORIES[@]}"; do
        for mode in "${MODES[@]}"; do
            INPUT_FILE="data/top${TOPNS}/generation/${dataset}/test.json"
            torchrun --nproc_per_node=$NP generation.py \
            --mode $mode \
            --category $category \
            --model_name "./trained_models/${mode}/epoch_3/top${TOPNS}/${dataset}/${category}" \
            --dataset $dataset \
            --topn $TOPNS \
            --batch_size $BATCH_SIZE \
            --max_length $MAX_LENGTH \
            --input_file $INPUT_FILE
            wait
        done
    done
done

for dataset in "${SUM_DATASETS[@]}"; do
    for category in "${CATEGORIES[@]}"; do
        for mode in "${MODES[@]}"; do
            INPUT_FILE="data/top${TOPNS}/summarization/${dataset}/test.json"
            torchrun --nproc_per_node=$NP generation.py \
            --mode $mode \
            --category $category \
            --model_name "./trained_models/${mode}/epoch_3/top${TOPNS}/${dataset}/${category}" \
            --dataset $dataset \
            --topn $TOPNS \
            --batch_size $BATCH_SIZE \
            --max_length $MAX_LENGTH \
            --input_file $INPUT_FILE
            wait
        done
    done
done
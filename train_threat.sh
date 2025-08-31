#!/bin/bash

CON_DATASETS=("IMDB" "RealToxicityPrompts")
SUM_DATASETS=("CNN" "XSum")
# CATEGORIES=("toxicity" "insult" "threat" "identity_attack")
# TOPNS=(3 5 10)
TOPNS=10
MODE=("debiased" "noperturb")
BATCH_SIZE=8
MAX_LENGTH=128
LEARNING_RATE=5e-5
WEIGHT_DECAY=0.01
EPOCHS=3

for topn in "${TOPNS[@]}"; do
    for dataset in "${CON_DATASETS[@]}"; do
        for mode in "${MODE[@]}"; do
            BASELINE_TRAIN_FILE="data/top${topn}/generation/${dataset}/train_sorted.json"
            DEBIASED_FILE="data/top${topn}/generation/${dataset}/train_debiased_sorted.json"
            SUBSPACE_JSON="bias_subspace/top${topn}+base/${dataset}/defining_set_threat.json"

            CUDA_VISIBLE_DEVICES=2 python train.py \
                --topn $topn \
                --dataset $dataset \
                --model_name ./pretraining/bart_continuation \
                --category threat \
                --batch_size $BATCH_SIZE \
                --epochs $EPOCHS \
                --max_length $MAX_LENGTH \
                --mode $mode \
                --train_file $BASELINE_TRAIN_FILE \
                --debiased_file $DEBIASED_FILE \
                --subspace_json $SUBSPACE_JSON \
                --learning_rate $LEARNING_RATE \
                --weight_decay $WEIGHT_DECAY
            wait
        done
    done
done

for topn in "${TOPNS[@]}"; do
    for dataset in "${SUM_DATASETS[@]}"; do
        for mode in "${MODE[@]}"; do
            BASELINE_TRAIN_FILE="data/top${topn}/summarization/${dataset}/train_sorted.json"
            DEBIASED_FILE="data/top${topn}/summarization/${dataset}/train_debiased_sorted.json"
            SUBSPACE_JSON="bias_subspace/top${topn}+base/${dataset}/defining_set_threat.json"

            CUDA_VISIBLE_DEVICES=2 python train.py \
                --topn $topn \
                --dataset $dataset \
                --model_name ./pretraining/bart_summarization \
                --category threat \
                --batch_size $BATCH_SIZE \
                --epochs $EPOCHS \
                --max_length $MAX_LENGTH \
                --mode $mode \
                --train_file $BASELINE_TRAIN_FILE \
                --debiased_file $DEBIASED_FILE \
                --subspace_json $SUBSPACE_JSON \
                --learning_rate $LEARNING_RATE \
                --weight_decay $WEIGHT_DECAY
            wait
        done
    done
done
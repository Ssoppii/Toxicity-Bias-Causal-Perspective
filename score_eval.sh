#!/bin/bash

TOPNS=(5)
CATEGORIES=("insult" "threat")
MODES=("noperturb")
DATASETS=("XSum")

for topn in "${TOPNS[@]}"; do
    for dataset in "${DATASETS[@]}"; do
        for category in "${CATEGORIES[@]}"; do
            for mode in "${MODES[@]}"; do
                INPUT_FILE="outputs/train/${dataset}/top${topn}/${mode}_${category}_outputs.json"
                OUTPUT_FILE="score/${dataset}/top${topn}/${mode}_${category}_score.json"
                
                echo "▶ Running score_eval for $dataset | $mode | $category"
                python score_eval.py \
                    --dataset "$dataset" \
                    --mode "$mode" \
                    --category "$category" \
                    --input "$INPUT_FILE" \
                    --output "$OUTPUT_FILE" \
                    --topn "$topn"
                    
            done
        done
    done
done
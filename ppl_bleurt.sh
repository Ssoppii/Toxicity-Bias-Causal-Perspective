#!/bin/bash

# 실행 옵션
DATASET=("IMDB" "RealToxicityPrompts" "CNN" "XSum")
MODE=("debiased" "noperturb")
CATEGORY=("toxicity" "insult" "threat" "identity_attack")  # 또는 insult, identity_attack 등
TOPN=(5 10)

for topn in "${TOPN[@]}"; do
    for dataset in "${DATASET[@]}"; do
        for mode in "${MODE[@]}"; do
            for category in "${CATEGORY[@]}"; do
                # 실행
                INPUT_JSON="outputs/train/${dataset}/top${topn}/${mode}_${category}_outputs.json"  # 생성된 output JSON
                REFERENCE_JSON="checkpoints/${dataset}/top${topn}/test_baseline_outputs.json"  # 참조 JSON
                OUTPUT_JSON="eval/ppl_bleurt/${dataset}/top${topn}/${mode}_${category}_score.json"

                python ppl_bleurt.py \
                --dataset "$dataset" \
                --mode "$mode" \
                --topn "$topn" \
                --category "$category" \
                --input "$INPUT_JSON" \
                --reference "$REFERENCE_JSON" \
                --output "$OUTPUT_JSON"
            done
        done
    done
done
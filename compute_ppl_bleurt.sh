#!/bin/bash

# --------- 사용자 정의 변수 ----------
# DATASETS=("IMDB" "RealToxicityPrompts" "CNN" "XSum")
DATASETS=("RealToxicityPrompts" "CNN" "XSum")
MODES=("noperturb" "debiased")
CATEGORIES=("toxicity" "insult" "threat" "identity_attack")
TOPN="top3"

BASE_DIR="eval/ppl_bleurt"
OUT_CSV="perplexity_summary.csv"
SCRIPT="compute_ppl_bleurt.py"
# -------------------------------------

echo "▶ Perplexity 계산 시작"

for dataset in "${DATASETS[@]}"; do
  for mode in "${MODES[@]}"; do
    for category in "${CATEGORIES[@]}"; do
      INPUT_JSON="${BASE_DIR}/${dataset}/${TOPN}/${mode}_${category}_score.json"
      OUTPUT_JSON="${BASE_DIR}/${dataset}/${TOPN}/${mode}_${category}_score_with_perplexity.json"

      echo "→ ${dataset} | ${mode} | ${category}"

      if [[ -f "$INPUT_JSON" ]]; then
        python "$SCRIPT" \
          --dataset "$dataset" \
          --mode "$mode" \
          --category "$category" \
          --input_json "$INPUT_JSON" \
          --output_json "$OUTPUT_JSON" \
          --output_csv "$OUT_CSV"
      else
        echo "⚠️  Skipped (not found): $INPUT_JSON"
      fi
    done
  done
done

echo "✅ 모든 Perplexity 계산 완료 → 저장: $OUT_CSV"
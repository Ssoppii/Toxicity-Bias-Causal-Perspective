#!/bin/bash

# topn 리스트
topn_list=(3)
# topn_list=(10)
# dataset 리스트
datasets=("IMDB" "RealToxicityPrompts" "CNN" "XSum")
# datasets=("XSum")

# 실행 경로
script_path="visualize_eval.py"

# 각 조합에 대해 실행
for topn in "${topn_list[@]}"; do
  for dataset in "${datasets[@]}"; do
    echo "Running for top${topn}, dataset ${dataset}"

    python "$script_path" \
      --input_dir "score/${dataset}/top${topn}" \
      --baseline_dir "score/${dataset}" \
      --output_dir "final/top${topn}" \
      --bleurt_dir "eval/ppl_bleurt/${dataset}/top${topn}" \
      --topn "$topn" \
      --dataset "$dataset"
  done
done
#!/bin/bash

TOPKS=("top3" "top5" "top10")
DOMAINS=("IMDB" "RealToxicityPrompts" "XSum" "CNN")

for topk in "${TOPKS[@]}"; do
  for domain in "${DOMAINS[@]}"; do
    echo "Running: $topk - $domain"
    python visualize.py --input_dir ../bias_subspace/${topk}/${domain} \
                        --output_dir ../plots/bias_subspace_${topk}/${domain}
  done
done
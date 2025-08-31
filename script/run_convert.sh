#!/bin/bash

TOPKS=(3 5 10)
DOMAINS=("IMDB" "XSum" "CNN")

for topk in "${TOPKS[@]}"; do
  for domain in "${DOMAINS[@]}"; do
    echo "Running: $topk - $domain"

    if [ "$domain" == "IMDB" ]; then
      input_dir="../raw_data/review_polarity/txt_sentoken/reviews"
      task="generation"
    elif [ "$domain" == "XSum" ]; then
      input_dir="../raw_data/XSum"
      task="summarization"
    elif [ "$domain" == "CNN" ]; then
      input_dir="../raw_data/cnn_stories/stories"
      task="summarization"
    else
      echo "Unknown domain: $domain"
      continue
    fi

    output_dir="../data/top${topk}/${task}/${domain}"

    python convert_to_json_jsonl_api.py \
      --input_dir "$input_dir" \
      --output_path_base "$output_dir" \
      --top_k "$topk"
  done
done

for topk in "${TOPKS[@]}"; do
    echo "Running: $topk - RealToxicityPrompts"

    output_dir="../data/top${topk}/generation/RealToxicityPrompts"

    python convert_to_json_jsonl_api.py \
      --input_jsonl "../raw_data/RealToxicityPrompts/prompts.jsonl" \
      --output_path_base "$output_dir" \
      --top_k "$topk"
done
#!/bin/bash

NP=4
MASTER_PORT=12345

MODEL_NAME="facebook/bart-base"
# MODEL_NAME="sshleifer/distilbart-cnn-12-6"
DATASETS=("XSum" "CNN" "RealToxicityPrompts")
# DATASETS=("IMDB")
CATEGORIES=("toxicity")
TOPNS=(5 10) # 3은 나중에 필요하면 돌리기
BATCH_SIZE=8
MAX_LENGTH=256
EPOCHS=3
LEARNING_RATE=5e-5
WEIGHT_DECAY=0.01

python evaluation.py \
          --mode baseline \
          --dataset XSum \
          --category toxicity \
          --topn 5 \
          --batch_size $BATCH_SIZE \
          --max_length $MAX_LENGTH \
          --task_type generation \
          --generated_file "checkpoints/XSum/top5/baseline_outputs.json" \
          --reference_file "checkpoints/XSum/top5/baseline_outputs.json"
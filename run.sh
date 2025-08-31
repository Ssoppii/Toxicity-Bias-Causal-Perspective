#!/bin/bash

NP=4
MASTER_PORT=12345

# MODEL_NAME="facebook/bart-large-cnn"
MODEL_NAME="facebook/bart-base"
# MODEL_NAME="google/pegasus-xsum"
DATASETS=("XSum" "CNN" "RealToxicityPrompts")
# DATASETS=("IMDB")
CATEGORIES=("toxicity" "insult" "threat" "identity_attack")
TOPNS=(5 10) # 3은 나중에 필요하면 돌리기
BATCH_SIZE=8
MAX_LENGTH=128
GEN_MAX_LENGTH=128
EPOCHS=3
LEARNING_RATE=5e-5
WEIGHT_DECAY=0.01

for dataset in "${DATASETS[@]}"; do
  for topn in "${TOPNS[@]}"; do
    if [[ "$dataset" == "IMDB" || "$dataset" == "RealToxicityPrompts" ]]; then
      TASK_TYPE="generation"
    else
      TASK_TYPE="summarization"
    fi

    TEST_FILE="data/top${topn}/${TASK_TYPE}/${dataset}/test.json"
    REFERENCE_FILE="checkpoints/${dataset}/top${topn}/baseline_outputs.json"

    # Baseline training (once per dataset/topn)
    BASELINE_SAVE_DIR="checkpoints/${dataset}/top${topn}"
    BASELINE_TRAIN_FILE="data/top${topn}/${TASK_TYPE}/${dataset}/train.json"
    mkdir -p $BASELINE_SAVE_DIR

    torchrun --nproc_per_node=$NP --master_port=$MASTER_PORT train.py \
      --model_name $MODEL_NAME \
      --dataset $dataset \
      --category all \
      --topn $topn \
      --batch_size $BATCH_SIZE \
      --max_length $MAX_LENGTH \
      --epochs $EPOCHS \
      --learning_rate $LEARNING_RATE \
      --weight_decay $WEIGHT_DECAY \
      --train_file $BASELINE_TRAIN_FILE \
      --debiased_file "none" \
      --subspace_json "none" \
      --save_dir $BASELINE_SAVE_DIR \
      --mode baseline

    torchrun --nproc_per_node=4 generation.py \
      --model_checkpoint "${BASELINE_SAVE_DIR}/bart_baseline_epoch3.pt" \
      --mode baseline \
      --model_name $MODEL_NAME \
      --dataset $dataset \
      --category toxicity \
      --topn $topn \
      --batch_size $BATCH_SIZE \
      --max_length $GEN_MAX_LENGTH \
      --test_file $TEST_FILE

    python evaluation.py \
      --mode baseline \
      --dataset $dataset \
      --category toxicity \
      --topn $topn \
      --batch_size $BATCH_SIZE \
      --max_length $MAX_LENGTH \
      --task_type $TASK_TYPE  \
      --generated_file "${BASELINE_SAVE_DIR}/baseline_outputs.json" \
      --reference_file "${BASELINE_SAVE_DIR}/baseline_outputs.json"

    for category in "${CATEGORIES[@]}"; do
      SAVE_DIR="checkpoints/${dataset}/top${topn}/${category}"
      TRAIN_FILE="data/top${topn}/${TASK_TYPE}/${dataset}/train.json"
      DEBIASED_FILE="data/top${topn}/${TASK_TYPE}/${dataset}/train_debiased.json"
      SUBSPACE_JSON="bias_subspace/top${topn}+base/${dataset}/defining_set_${category}.json"
      mkdir -p $SAVE_DIR

      # run noperturb in background
      (
        torchrun --nproc_per_node=$NP --master_port=$MASTER_PORT train.py \
          --model_name $MODEL_NAME \
          --dataset $dataset \
          --category $category \
          --topn $topn \
          --batch_size $BATCH_SIZE \
          --max_length $MAX_LENGTH \
          --epochs $EPOCHS \
          --learning_rate $LEARNING_RATE \
          --weight_decay $WEIGHT_DECAY \
          --train_file $TRAIN_FILE \
          --debiased_file $DEBIASED_FILE \
          --subspace_json $SUBSPACE_JSON \
          --save_dir $SAVE_DIR \
          --mode noperturb
      )
      wait
      (
        torchrun --nproc_per_node=4 generation.py \
          --model_checkpoint "${SAVE_DIR}/bart_noperturb_epoch${EPOCHS}.pt" \
          --mode noperturb \
          --model_name $MODEL_NAME \
          --dataset $dataset \
          --category $category \
          --topn $topn \
          --batch_size $BATCH_SIZE \
          --max_length $GEN_MAX_LENGTH \
          --test_file $TEST_FILE
      )
      wait
      (
        python evaluation.py \
          --mode noperturb \
          --dataset $dataset \
          --category $category \
          --topn $topn \
          --batch_size $BATCH_SIZE \
          --max_length $MAX_LENGTH \
          --task_type $TASK_TYPE \
          --generated_file "${SAVE_DIR}/noperturb_outputs.json" \
          --reference_file $REFERENCE_FILE
      )
      wait

      # run debiased in background
      (
        torchrun --nproc_per_node=$NP --master_port=$MASTER_PORT train.py \
          --model_name $MODEL_NAME \
          --dataset $dataset \
          --category $category \
          --topn $topn \
          --batch_size $BATCH_SIZE \
          --max_length $MAX_LENGTH \
          --epochs $EPOCHS \
          --learning_rate $LEARNING_RATE \
          --weight_decay $WEIGHT_DECAY \
          --train_file $TRAIN_FILE \
          --debiased_file $DEBIASED_FILE \
          --subspace_json $SUBSPACE_JSON \
          --save_dir $SAVE_DIR \
          --mode debiased
      )
      wait
      (
        torchrun --nproc_per_node=4 generation.py \
          --model_checkpoint "${SAVE_DIR}/bart_debiased_epoch${EPOCHS}.pt" \
          --mode debiased \
          --model_name $MODEL_NAME \
          --dataset $dataset \
          --category $category \
          --topn $topn \
          --batch_size $BATCH_SIZE \
          --max_length $GEN_MAX_LENGTH \
          --test_file $TEST_FILE
      )
      (
        python evaluation.py \
          --mode debiased \
          --dataset $dataset \
          --category $category \
          --topn $topn \
          --batch_size $BATCH_SIZE \
          --max_length $MAX_LENGTH \
          --task_type $TASK_TYPE \
          --generated_file "${SAVE_DIR}/debiased_outputs.json" \
          --reference_file $REFERENCE_FILE
      )
      wait
    done
  done
done
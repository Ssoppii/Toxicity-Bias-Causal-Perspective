#!/usr/bin/env bash
set -euo pipefail

# 데이터셋별 작업 디렉터리
declare -A TASK=(
  [IMDB]=generation
  [RealToxicityPrompts]=generation
  [CNN]=summarization
  [XSum]=summarization
)

# Top-N 옵션, 카테고리 옵션
# TOPNS=(3 5 10)
TOPNS=(5 10)
CATS=(toxicity insult threat identity_attack)

# 스크립트 호출 경로
DEBIAS_SCRIPT="python debias.py"
COMBINE_SCRIPT="python combine_debiased.py"

for TOPN in "${TOPNS[@]}"; do
  for DATA in "${!TASK[@]}"; do
    TASK_DIR=${TASK[$DATA]}
    BASE_DIR="../data/top${TOPN}/${TASK_DIR}/${DATA}"
    TRAIN_PATH="${BASE_DIR}/train.json"

    # 1) 카테고리별 debias
    for CAT in "${CATS[@]}"; do
      DEF_PATH="../bias_subspace/top${TOPN}+base/${DATA}/defining_set_${CAT}.json"
      echo "=== Debiasing: TOPN=${TOPN}, DATA=${DATA}, CAT=${CAT} ==="
      $DEBIAS_SCRIPT \
        --defining_path "$DEF_PATH" \
        --training_path "$TRAIN_PATH" \
        --topn "$TOPN" \
        --dataset "$DATA" \
        --category "$CAT" \
        -soft
    done

    # 2) 4개 카테고리 결과 통합
    echo "=== Combining debiased outputs for TOPN=${TOPN}, DATA=${DATA} ==="
    # 통합 대상 파일 경로
    ORIGINAL="$TRAIN_PATH"
    OUTPUT="${BASE_DIR}/train_debiased.json"
    DEB_TOX="${BASE_DIR}/train.json_toxicity_debiased.json"
    DEB_INS="${BASE_DIR}/train.json_insult_debiased.json"
    DEB_THR="${BASE_DIR}/train.json_threat_debiased.json"
    DEB_IDA="${BASE_DIR}/train.json_identity_attack_debiased.json"

    $COMBINE_SCRIPT \
      --original "$ORIGINAL" \
      --toxicity         "$DEB_TOX" \
      --insult           "$DEB_INS" \
      --threat           "$DEB_THR" \
      --identity_attack  "$DEB_IDA" \
      --output "$OUTPUT"
  done
done
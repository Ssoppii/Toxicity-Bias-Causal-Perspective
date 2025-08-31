TOPN=(3 5 10)
DATASETS=("IMDB" "RealToxicityPrompts" "XSum" "CNN")

for topn in "${TOPN[@]}"; do
    for dataset in "${DATASETS[@]}"; do
        if [[ $dataset == "IMDB" || $dataset == "RealToxicityPrompts" ]]; then
            DIR="data/top${topn}/generation/${dataset}"
        else
            DIR="data/top${topn}/summarization/${dataset}"
        fi

        python sort_json.py \
        --dataset $dataset \
        --topn $topn \
        --input_file $DIR/train.json \
        --output_file $DIR/train_sorted.json
    done
done

for topn in "${TOPN[@]}"; do
    for dataset in "${DATASETS[@]}"; do
        if [[ $dataset == "IMDB" || $dataset == "RealToxicityPrompts" ]]; then
            DIR="data/top${topn}/generation/${dataset}"
        else
            DIR="data/top${topn}/summarization/${dataset}"
        fi

        python sort_json.py \
        --dataset $dataset \
        --topn $topn \
        --input_file $DIR/train_debiased.json \
        --output_file $DIR/train_debiased_sorted.json
    done
done
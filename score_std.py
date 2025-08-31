import json
from collections import defaultdict
from statistics import stdev

# JSON 파일 경로
json_path = "./revised/top3/RealToxicityPrompts_top5p_samples.json"

# 파일 불러오기
with open(json_path, "r") as f:
    data = json.load(f)

# None 값 제거
cleaned_data = [
    item for item in data
    if all(item[k] is not None for k in ["baseline_score", "noperturb_score", "debiased_score"])
]

# category별 점수 모으기
category_scores = defaultdict(lambda: {"baseline": [], "noperturb": [], "debiased": []})
for item in cleaned_data:
    cat = item["category"]
    category_scores[cat]["baseline"].append(item["baseline_score"])
    category_scores[cat]["noperturb"].append(item["noperturb_score"])
    category_scores[cat]["debiased"].append(item["debiased_score"])

# 표준편차 계산 및 출력
for cat, scores in category_scores.items():
    print(f"Category: {cat}")
    print(f"  Baseline Std:   {stdev(scores['baseline']):.4f}")
    print(f"  Noperturb Std:  {stdev(scores['noperturb']):.4f}")
    print(f"  Debiased Std:   {stdev(scores['debiased']):.4f}")
    print()
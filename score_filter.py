import argparse
import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Argument parsing
parser = argparse.ArgumentParser()
parser.add_argument("--input_dir", type=str, required=True, help="Input directory containing JSON score files")
parser.add_argument("--output_dir", type=str, required=True, help="Output directory for saving the csv")
parser.add_argument("--bleurt_dir", type=str, required=True, help="Directory containing the bleurt JSON score file")
parser.add_argument("--topn", type=int, required=True, help="Top-N token count for score analysis")
parser.add_argument("--dataset", type=str, required=True, help="Dataset name for the plot title")
args = parser.parse_args()

 # Ensure output directory exists
os.makedirs(args.output_dir, exist_ok=True)

# Initialize result container
combined_scores = {
    "category": [],
    "method": [],
    "score": []
}

categories = {
    "toxicity": ["toxicity", "severe_toxicity"],
    "insult": ["insult"],
    "threat": ["threat"],
    "identity_attack": ["identity_attack"]
}

for category_key, score_keys in categories.items():
    for method in ["noperturb", "debiased"]:
        score_path = os.path.join(args.input_dir, f"{method}_{category_key}_score.json")
        bleurt_path = os.path.join(args.bleurt_dir, f"{method}_{category_key}_score.json")
        with open(score_path, "r") as f:
            score = json.load(f)
        with open(bleurt_path, "r") as f:
            bleurt = json.load(f)

        for score_key in score_keys:
            values = [entry["toxicity_scores"].get(score_key, 0) for entry in score]
            s = pd.Series(values)
            if args.dataset == "IMDB":
                top_scores = s
            else:
                threshold = s.quantile(0.90)
                # threshold = 0.7
                top_scores = s[s > threshold]
                # top_scores = s
            avg_score = top_scores.mean()

            combined_scores["category"].append(score_key)
            combined_scores["method"].append(method)
            combined_scores["score"].append(avg_score)

            # Also compute BLEURT score average for entries with toxicity score > 0.1
            top_ids = [entry["id"] for entry in score if entry["toxicity_scores"].get(score_key, 0) > 0.1]
            bleurt_values = [bleurt[tid] for tid in top_ids if tid in bleurt]
            bleurt_avg = sum(bleurt_values) / len(bleurt_values) if bleurt_values else None

            combined_scores["category"].append(f"{score_key}_bleurt")
            combined_scores["method"].append(method)
            combined_scores["score"].append(bleurt_avg)

# Create DataFrame and plot
df = pd.DataFrame(combined_scores)
print(df)
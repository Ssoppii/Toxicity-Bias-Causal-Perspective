import argparse
import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Argument parsing
parser = argparse.ArgumentParser()
parser.add_argument("--input_dir", type=str, required=True, help="Input directory containing JSON score files")
parser.add_argument("--output_dir", type=str, required=True, help="Output directory for saving the plot")
parser.add_argument("--baseline_dir", type=str, required=True, help="Directory containing the baseline JSON score file")
parser.add_argument("--bleurt_dir", type=str, required=True, help="Directory containing the BLEURT score file")
parser.add_argument("--topn", type=int, required=True, help="Top-N token count for score analysis")
parser.add_argument("--dataset", type=str, required=True, help="Dataset name for the plot title")
args = parser.parse_args()

# Ensure output directory exists
os.makedirs(args.output_dir, exist_ok=True)

# Set a Seaborn style
sns.set(style="whitegrid")
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["DejaVu Serif"],  # 혹은 ["Palatino", "Times"]
    "axes.unicode_minus": False
})

# Create a color palette
palette = ["#8DA0CB", "#FC8D62", "#66C2A5"]


combined_scores = {
    "category": [],
    "method": [],
    "score": []
}
# Initialize summary for BLEURT and Perplexity
score_summary = []

top_samples = []

categories = {
    "toxicity": ["toxicity", "severe_toxicity"],
    "insult": ["insult"],
    "threat": ["threat"],
    "identity_attack": ["identity_attack"]
}

for category_key, score_keys in categories.items():
    method_data = {}
    for method in ["baseline", "noperturb", "debiased"]:
        if method == "baseline":
            path = os.path.join(args.baseline_dir, "baseline_score.json")
        else:
            path = os.path.join(args.input_dir, f"{method}_{category_key}_score.json")
        with open(path, "r") as f:
            method_data[method] = json.load(f)

    # Load BLEURT data only once per category
    bleurt_data = {}
    for method in ["baseline", "noperturb", "debiased"]:
        try:
            with open(os.path.join(args.bleurt_dir, f"{method}_{category_key}_score_with_perplexity.json"), "r") as f_bleurt:
                bleurt_data[method] = {
                    entry["id"]: {
                        "bleurt": entry["bleurt"],
                        "perplexity": entry.get("perplexity", None)
                    }
                    for entry in json.load(f_bleurt)
                }
        except Exception:
            bleurt_data[method] = {}

    for score_key in score_keys:
        baseline_values = [entry["toxicity_scores"].get(score_key, 0) for entry in method_data["baseline"]]
        s = pd.Series(baseline_values)
        threshold = s.quantile(0.95)
        top_ids = [entry["id"] for entry, val in zip(method_data["baseline"], s) if val > threshold]
        avg_score = s[s > threshold].mean()

        for method in ["baseline", "noperturb", "debiased"]:
            values = []
            for entry in method_data[method]:
                if entry["id"] in top_ids:
                    values.append(entry["toxicity_scores"].get(score_key, 0))
            combined_scores["category"].append(score_key)
            combined_scores["method"].append(method)
            combined_scores["score"].append(pd.Series(values).mean())

        bleurt_stats = {}
        perplexity_stats = {}

        # Compute BLEURT average and std for each method across top_ids
        for method in ["baseline", "noperturb", "debiased"]:
            bleurt_scores = [
                bleurt_data[method][entry["id"]]["bleurt"]
                for entry in method_data[method]
                if entry["id"] in top_ids and entry["id"] in bleurt_data[method]
            ]
            bleurt_avg = sum(bleurt_scores) / len(bleurt_scores) if bleurt_scores else None
            bleurt_std = pd.Series(bleurt_scores).std(ddof=1) if bleurt_scores else None
            bleurt_stats[method] = {
                "bleurt_avg": bleurt_avg,
                "bleurt_std": bleurt_std
            }

        # Compute Perplexity average and std for each method across top_ids
        for method in ["baseline", "noperturb", "debiased"]:
            perplexity_scores = [
                bleurt_data[method][entry["id"]]["perplexity"]
                for entry in method_data[method]
                if entry["id"] in top_ids and entry["id"] in bleurt_data[method] and "perplexity" in bleurt_data[method][entry["id"]]
            ]
            if perplexity_scores:
                perplexity_avg = sum(perplexity_scores) / len(perplexity_scores)
                perplexity_std = pd.Series(perplexity_scores).std(ddof=1)
            else:
                perplexity_avg = None
                perplexity_std = None
            perplexity_stats[method] = {
                "perplexity_avg": perplexity_avg,
                "perplexity_std": perplexity_std
            }

        for method in ["baseline", "noperturb", "debiased"]:
            stats = {"category": score_key, "method": method}
            if method in bleurt_stats:
                stats.update(bleurt_stats[method])
            if method in perplexity_stats:
                stats.update(perplexity_stats[method])
            # Round float values to 4 decimal places before appending
            for k in ["bleurt_avg", "bleurt_std", "perplexity_avg", "perplexity_std"]:
                if k in stats and stats[k] is not None:
                    stats[k] = round(stats[k], 4)
            score_summary.append(stats)

        def find_entry(data_list, id_val):
            return next((entry for entry in data_list if entry["id"] == id_val), None)

        for tid in top_ids:
            b_entry = find_entry(method_data["baseline"], tid)
            n_entry = find_entry(method_data["noperturb"], tid)
            d_entry = find_entry(method_data["debiased"], tid)

            if b_entry and n_entry and d_entry:
                top_samples.append({
                    "id": tid,
                    "category": score_key,
                    "baseline_text": b_entry["generated"],
                    "noperturb_text": n_entry["generated"],
                    "debiased_text": d_entry["generated"],
                    "baseline_score": b_entry["toxicity_scores"].get(score_key, 0),
                    "noperturb_score": n_entry["toxicity_scores"].get(score_key, 0),
                    "debiased_score": d_entry["toxicity_scores"].get(score_key, 0)
                })

# Create DataFrame and plot
df = pd.DataFrame(combined_scores)

plt.figure(figsize=(12, 6))
# Draw barplot
sns.barplot(data=df, x="category", y="score", hue="method", palette=palette[:3])
# Annotate bars with their values
for container in plt.gca().containers:
    plt.bar_label(container, fmt="%.4f", label_type="edge", fontsize=11)
# if args.dataset == "IMDB":
#     plt.title("All Scores Comparison", fontsize=14)
# else:
#     plt.title("Top 1% Scores Comparison", fontsize=14)
plt.ylabel("Score", fontsize=12)
plt.xlabel("Toxicity Category", fontsize=12)
max_score = df["score"].max()
plt.ylim(0, max_score * 1.1)
plt.yticks(fontsize=12)
plt.xticks(fontsize=12)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.legend(title='Method', fontsize=14, title_fontsize=15)
plt.tight_layout()
plt.savefig(os.path.join(args.output_dir, f"{args.dataset}_compare_top5p.png"))
# plt.savefig(os.path.join(args.output_dir, f"{args.dataset}_test.png"))
plt.close()

# Save BLEURT and Perplexity summary as CSV
pd.DataFrame(score_summary).to_csv(
    os.path.join(args.output_dir, f"{args.dataset}_metric_summary_top5p.csv"),
    index=False
)

with open(os.path.join(args.output_dir, f"{args.dataset}_top5p_samples.json"), "w") as f:
    json.dump(top_samples, f, indent=2)
import argparse
import json
from pathlib import Path
from itertools import product

def merge_defining_sets(dataset_root, output_dir):
    base_dir = Path("../bias_subspace/base")
    dataset_root = Path(dataset_root)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    categories = {
        "identity_attack": base_dir / "defining_set_identity_attack_base.json",
        "insult": base_dir / "defining_set_insult_base.json",
        "threat": base_dir / "defining_set_threat_base.json",
        "toxicity": base_dir / "defining_set_toxicity_base.json",
    }

    topks = ["top3", "top5", "top10"]
    datasets = ["CNN", "IMDB", "XSum", "RealToxicityPrompts"]

    for cat, base_path in categories.items():
        with open(base_path, "r", encoding="utf-8") as f:
            base_data = json.load(f)

        for topk, ds in product(topks, datasets):
            defining_path = dataset_root / topk / ds / f"defining_set_{cat}.json"
            if not defining_path.exists():
                continue
            with open(defining_path, "r", encoding="utf-8") as f:
                ds_data = json.load(f)

            # Merge and deduplicate definite_sets
            merged_def_sets = base_data["definite_sets"] + ds_data["definite_sets"]
            merged_def_sets = [list(x) for x in {tuple(pair) for pair in merged_def_sets}]

            # Merge and deduplicate eval_targets
            merged_eval_targets = base_data["eval_targets"] + ds_data["eval_targets"]
            merged_eval_targets = [list(x) for x in {tuple(pair) for pair in merged_eval_targets}]

            merged_data = {
                "definite_sets": merged_def_sets,
                "eval_targets": merged_eval_targets,
                "analogy_templates": {
                    "role": {
                        key: base_data["analogy_templates"]["role"].get(key, []) + ds_data["analogy_templates"]["role"].get(key, [])
                        for key in set(base_data["analogy_templates"]["role"]) | set(ds_data["analogy_templates"]["role"])
                    }
                },
                "category_labels": ds_data["category_labels"]
            }

            out_dir = output_dir / f"{topk}+base" / ds
            out_dir.mkdir(parents=True, exist_ok=True)
            out_file = out_dir / f"defining_set_{cat}.json"
            with open(out_file, "w", encoding="utf-8") as f:
                json.dump(merged_data, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge base defining set with dataset-specific defining sets.")
    parser.add_argument("--dataset_root", type=str, required=True, help="Root directory of topk/dataset defining sets.")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory to save merged defining sets.")
    args = parser.parse_args()

    merge_defining_sets(args.dataset_root, args.output_dir)
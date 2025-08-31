import json
import os
os.environ["NLTK_DATA"] = "./nltk_data"
from collections import defaultdict
from typing import List
from nltk.corpus import wordnet as wn
import nltk
import argparse
import requests
nltk.download('wordnet', download_dir=os.environ["NLTK_DATA"])
nltk.download('omw-1.4', download_dir=os.environ["NLTK_DATA"])


def load_datasets(paths):
    data = []
    for path in paths:
        with open(path, 'r', encoding='utf-8') as f:
            data.extend(json.load(f))
    return data


def extract_category_tokens(data, categories):
    cat_tokens = defaultdict(set)
    for entry in data:
        for cat in categories:
            toks = entry.get(f"{cat}_tokens", [])
            toks = [t.lower() for t in toks if t and t.isalpha()]
            cat_tokens[cat].update(toks)
    return cat_tokens

def get_antonym_pair(word):
    url = f"https://api.conceptnet.io/query?node=/c/en/{word}&rel=/r/Antonym&limit=5"
    try:
        response = requests.get(url, timeout=3)
        data = response.json()
        antonyms = set()
        for edge in data["edges"]:
            if edge["start"]["label"].lower() == word.lower():
                antonyms.add(edge["end"]["label"].lower())
            elif edge["end"]["label"].lower() == word.lower():
                antonyms.add(edge["start"]["label"].lower())
        if antonyms:
            return word, list(antonyms)[0]
    except Exception as e:
        print(f"ConceptNet API error for '{word}': {e}")
    return None


def build_defining_set_for_category(tokens: List[str], category: str):
    # Try WordNet antonym mapping only
    antonym_pairs = []
    seen_words = set()
    for token in tokens:
        if token in seen_words:
            continue
        pair = get_antonym_pair(token)
        if pair and pair[1] in tokens:
            antonym_pairs.append([pair[0], pair[1]])
            seen_words.update(pair)

    group_a = []
    group_b = []
    for a, b in antonym_pairs:
        if a not in group_a:
            group_a.append(a)
        if b not in group_b:
            group_b.append(b)

    label_map = {
        "toxicity": ("toxic", "nontoxic"),
        # "insult": ("insulting", "noninsulting"),
        # "threat": ("threatening", "nonthreatening"),
        # "identity_attack": ("attacking_identity", "respecting_identity")
    }

    neg_label, pos_label = label_map.get(category, (f"{category}_negative", f"{category}_positive"))

    defining = {
        "definite_sets": antonym_pairs,
        "eval_targets": antonym_pairs[:min(10, len(antonym_pairs))],
        "analogy_templates": {
            "role": {
                neg_label: group_a,
                pos_label: group_b
            }
        },
        "testTerms": tokens[:30],
        "category_labels": {
            "category": category,
            "positive_label": pos_label,
            "negative_label": neg_label
        }
    }
    return defining


def save_defining_set(defining, out_path):
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(defining, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate category-specific defining sets from JSON dataset.")
    parser.add_argument("--input_dir", type=str, required=True, help="Directory containing train JSON file")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save defining set JSONs")
    args = parser.parse_args()

    json_files = [os.path.join(args.input_dir, "train.json")]
    categories = ["toxicity"]

    data = load_datasets(json_files)
    cat_tokens = extract_category_tokens(data, categories)

    os.makedirs(args.output_dir, exist_ok=True)

    for cat, toks in cat_tokens.items():
        defining_set = build_defining_set_for_category(list(toks), cat)
        out_path = os.path.join(args.output_dir, f"defining_set_{cat}.json")
        save_defining_set(defining_set, out_path)
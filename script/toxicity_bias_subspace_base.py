import json
import os
import pickle
os.environ["NLTK_DATA"] = "./script/nltk_data"
os.environ["GENSIM_DATA_DIR"] = "./script/gensim_data"
from typing import List
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from nltk.corpus import wordnet as wn
from nltk.corpus import stopwords
import nltk
import argparse
import gensim.downloader as api
nltk.download('wordnet', download_dir=os.environ["NLTK_DATA"])
nltk.download('omw-1.4', download_dir=os.environ["NLTK_DATA"])
nltk.download('stopwords', download_dir=os.environ["NLTK_DATA"])

def is_positive_by_antonym(word):
    for syn in wn.synsets(word):
        for lemma in syn.lemmas():
            if lemma.antonyms():
                return False
    return True

def load_datasets(filepaths):
    data = []
    for path in filepaths:
        with open(path, 'r', encoding='utf-8') as f:
            data.extend(json.load(f))
    return data

def extract_category_tokens(data, categories):
    tokens_by_category = {cat: set() for cat in categories}
    for item in data:
        for cat in categories:
            key = f"{cat}_tokens"
            if key in item:
                tokens_by_category[cat].update([t for t in item[key] if t])
    return tokens_by_category

def load_glove_vectors(filepath, expected_dim=200):
    pkl_path = filepath.replace(".txt", ".pkl")
    if os.path.exists(pkl_path):
        with open(pkl_path, 'rb') as f:
            return pickle.load(f)

    glove = {}
    with open(filepath, 'r', encoding='utf8') as f:
        for line in f:
            parts = line.strip().split()
            word = parts[0]
            vec = np.array(parts[1:], dtype=np.float32)
            if vec.shape[0] != expected_dim:
                continue  # skip malformed vectors
            glove[word] = vec

    with open(pkl_path, 'wb') as f:
        pickle.dump(glove, f)

    return glove

def get_clean_similar_words(wv, category, topn=10000, limit=500, expected_dim=200):
    stop_words = set(stopwords.words("english"))

    def is_clean_word(w):
        return (
            w.isalpha() and len(w) <= 15 and wn.synsets(w)
            and w.lower() not in stop_words and category.lower() not in w.lower()
        )

    if category not in wv:
        return []

    all_words = list(wv.keys())
    valid_vectors = [wv[w] for w in all_words if isinstance(wv[w], np.ndarray) and wv[w].shape == (expected_dim,)]
    matrix = np.stack(valid_vectors)
    cat_vec = wv[category].reshape(1, -1)
    sims = cosine_similarity(cat_vec, matrix)[0]
    top_indices = np.argsort(sims)[::-1]

    clean = []
    for idx in top_indices:
        w = all_words[idx]
        if is_clean_word(w):
            clean.append(w)
        if len(clean) >= limit:
            break
    return clean

def get_antonym_pair(word):
    antonyms = set()
    for syn in wn.synsets(word):
        for lemma in syn.lemmas():
            for ant in lemma.antonyms():
                antonyms.add(ant.name())
    if antonyms:
        return word, list(antonyms)[0]
    return None

def build_defining_set_for_category(tokens: List[str], category: str):
    # Try WordNet antonym mapping
    antonym_pairs = []
    seen_words = set()
    for token in tokens:
        if token in seen_words:
            continue
        pair = get_antonym_pair(token)
        if pair and pair[1] in tokens:
            antonym_pairs.append([pair[0], pair[1]])
            seen_words.update(pair)

    # Load GloVe
    glove_path = "./script/glove.twitter.27B.200d.txt"
    wv = load_glove_vectors(glove_path)

    # Extend token list with similar words to category
    if category == 'identity_attack':
        similar_words = get_clean_similar_words(wv, 'discriminate', topn=100000, limit=5000, expected_dim=200)
    else:
        similar_words = get_clean_similar_words(wv, category, topn=100000, limit=5000, expected_dim=200)

    added_pairs = []
    seen_added = set()
    for word in similar_words:
        pair = get_antonym_pair(word)
        if pair and pair[1] in wv and pair[0] in wv:
            if pair[0] not in seen_added and pair[1] not in seen_added:
                added_pairs.append(list(pair))
                seen_added.update(pair)
        # if len(added_pairs) >= 100:
        #     break

    antonym_pairs.extend(added_pairs)

    # Filter consistent direction using bias direction
    if len(antonym_pairs) > 0:
        vecs_1 = np.array([wv[a] for a, b in antonym_pairs if a in wv and b in wv])
        vecs_2 = np.array([wv[b] for a, b in antonym_pairs if a in wv and b in wv])
        if len(vecs_1) > 0 and len(vecs_2) > 0:
            bias_direction = np.mean(vecs_1 - vecs_2, axis=0)
            normalized_bias_direction = bias_direction / np.linalg.norm(bias_direction)

            corrected_pairs = []
            for a, b in antonym_pairs:
                if a in wv and b in wv:
                    proj_a = np.dot(wv[a], normalized_bias_direction)
                    proj_b = np.dot(wv[b], normalized_bias_direction)

                    # bias direction projection 기준 우선 정렬
                    if proj_a < proj_b:
                        neg, pos = a, b
                    else:
                        neg, pos = b, a

                    # wordnet 기반 휴리스틱 보정
                    if is_positive_by_antonym(neg) and not is_positive_by_antonym(pos):
                        neg, pos = pos, neg

                    corrected_pairs.append([neg, pos])
            antonym_pairs = corrected_pairs

    all_pairs = antonym_pairs
    group_a, group_b = [], []
    for a, b in all_pairs:
        group_a.append(a)
        group_b.append(b)

    label_map = {
        "toxicity": ("toxic", "nontoxic"),
        "insult": ("insulting", "noninsulting"),
        "threat": ("threatening", "nonthreatening"),
        "identity_attack": ("attacking_identity", "respecting_identity")
    }

    neg_label, pos_label = label_map.get(category, (f"{category}_negative", f"{category}_positive"))

    defining = {
        "definite_sets": all_pairs,
        "eval_targets": all_pairs[:min(10, len(all_pairs))],
        "analogy_templates": {
            "role": {
                neg_label: group_a,
                pos_label: group_b
            }
        },
        "category_labels": {
            "category": category,
            "negative_label": neg_label,
            "positive_label": pos_label
        }
    }
    return defining

def save_defining_set(defining, out_path):
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(defining, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate category-specific defining sets from JSON dataset.")
    # parser.add_argument("--input_dir", type=str, required=True, help="Directory containing train JSON file")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save defining set JSONs")
    args = parser.parse_args()

    # json_files = [os.path.join(args.input_dir, "train.json")]
    # data = load_datasets(json_files)
    # cat_tokens = extract_category_tokens(data, categories)
    categories = ["toxicity", "insult", "threat", "identity_attack"]
    cat_tokens = {cat: [] for cat in categories}

    os.makedirs(args.output_dir, exist_ok=True)

    for cat, toks in cat_tokens.items():
        defining_set = build_defining_set_for_category(list(toks), cat)
        out_path = os.path.join(args.output_dir, f"defining_set_{cat}_base.json")
        save_defining_set(defining_set, out_path)
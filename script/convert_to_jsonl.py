import os
import concurrent.futures
os.environ["NLTK_DATA"] = "../script/nltk_data"
from transformers import BartTokenizer
import json
import random
import string
from pathlib import Path
from typing import List, Tuple
import nltk
from nltk.corpus import stopwords
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import requests

# NLTK 다운로드 설정
nltk.download('punkt', download_dir=os.environ["NLTK_DATA"])
nltk.download('stopwords', download_dir=os.environ["NLTK_DATA"])

MAX_TOKEN_LENGTH = 512
bart_tokenizer = BartTokenizer.from_pretrained("facebook/bart-base")
stop_words = set(stopwords.words('english')).union({'"', "'", "\\", "''", "``", "\'", "'s", "--", "'.\'", ",\"", "``", "''", "‘", "’", "“", "”", "(", ")", "[", "]", "{", "}", "<", ">", "/", "\\", "|", ";", ":", ",", ".", "!", "?", "@", "#", "$", "%", "^", "&", "*", "_", "+", "-", "=", "·", "�"})

# Load custom stopwords from file
try:
    with open("smart-stopwords.txt", "r", encoding="utf-8") as sf:
        custom_stops = set(line.strip().lower() for line in sf if line.strip())
        stop_words = stop_words.union(custom_stops)
except FileNotFoundError:
    print("Warning: smart-stopwords.txt not found. Proceeding without custom stopwords.")

PERSPECTIVE_API_URL = "https://commentanalyzer.googleapis.com/v1alpha1/comments:analyze"
PERSPECTIVE_API_KEY = "AIzaSyBj3ck-qd0rZN_QurhsTiHs2gJ89-AegSQ"

def truncate_text(text: str, tokenizer: BartTokenizer, max_length: int = MAX_TOKEN_LENGTH) -> str:
    tokens = tokenizer.tokenize(text)
    if len(tokens) > max_length:
        tokens = tokens[:max_length]
        text = tokenizer.convert_tokens_to_string(tokens)
    return text

def score_tokens_with_perspective(text: str, tokenizer: BartTokenizer, top_k: int = 5) -> Tuple[dict, dict]:
    tokens = tokenizer.tokenize(text)
    spans = [tokenizer.convert_tokens_to_string([tok]).strip() for tok in tokens]
    attribute_names = ["TOXICITY", "IDENTITY_ATTACK", "INSULT", "THREAT"]
    scored = {attr: [] for attr in attribute_names}

    for idx, token in enumerate(spans):
        if token.lower() in stop_words or token in string.punctuation and len(token) < 3:
            continue
        try:
            response = requests.post(
                url=f"{PERSPECTIVE_API_URL}?key={PERSPECTIVE_API_KEY}",
                json={
                    "comment": {"text": token},
                    "languages": ["en"],
                    "requestedAttributes": {attr: {} for attr in attribute_names}
                },
                timeout=3
            )
            response_data = response.json()
            for attr in attribute_names:
                score = (
                    response_data.get("attributeScores", {})
                    .get(attr, {})
                    .get("summaryScore", {})
                    .get("value", 0.0)
                )
                scored[attr].append((token, idx, score))
        except Exception:
            continue

    top_tokens = {}
    top_ids = {}
    for attr in attribute_names:
        top_scored = sorted(scored[attr], key=lambda x: x[2], reverse=True)[:top_k]
        top_tokens[attr.lower()] = [s[0] for s in top_scored] + [None] * (top_k - len(top_scored))
        top_ids[attr.lower()] = [s[1] for s in top_scored] + [None] * (top_k - len(top_scored))

    return top_tokens, top_ids

def load_prompt_jsonl(jsonl_path: str) -> List[str]:
    with open(jsonl_path, "r", encoding="utf-8") as f:
        return [json.loads(line)["prompt"]["text"].replace("\n", " ").strip() for line in f]

def process_prompt_jsonl(jsonl_path: str, output_path_base: str, top_k: int = 5, max_workers: int = 16):
    os.makedirs(output_path_base, exist_ok=True)
    prompts = load_prompt_jsonl(jsonl_path)
    files = list(zip(range(len(prompts)), prompts))

    train, test = train_test_split(files, test_size=0.2, random_state=42)
    train, dev = train_test_split(train, test_size=0.25, random_state=42)
    split_map = {"train": train, "dev": dev, "test": test}

    for split, split_data in split_map.items():
        def process_entry(entry):
            idx, text = entry
            try:
                text = truncate_text(text, bart_tokenizer)
                tokens = bart_tokenizer.tokenize(text)
                tokens_by_attr, ids_by_attr = score_tokens_with_perspective(text, bart_tokenizer, top_k)
                sample = {
                    "id": f"{idx}",
                    "token": [t.lstrip("Ġ") for t in tokens],
                }
                for attr in tokens_by_attr:
                    sample[f"{attr}_tokens"] = tokens_by_attr[attr]
                    sample[f"{attr}_token_ids"] = ids_by_attr[attr]
                return sample
            except Exception:
                return None

        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            results = list(tqdm(executor.map(process_entry, split_data), total=len(split_data)))
            results = [r for r in results if r is not None]

        with open(os.path.join(output_path_base, f"{split}.json"), "w", encoding="utf-8") as out_f:
            json.dump(results, out_f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Process JSONL file and extract toxic tokens using Perspective API.")
    parser.add_argument("--input_jsonl", type=str, required=True, help="Path to input JSONL file containing prompts")
    parser.add_argument("--output_path_base", type=str, required=True, help="Directory to save processed output JSON files")
    parser.add_argument("--top_k", type=int, default=5, help="Number of top tokens to extract per attribute")
    parser.add_argument("--max_workers", type=int, default=16, help="Number of threads for parallel processing")
    args = parser.parse_args()

    process_prompt_jsonl(args.input_jsonl, args.output_path_base, top_k=args.top_k, max_workers=args.max_workers)
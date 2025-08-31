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
from googleapiclient import discovery
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import re
import time
import requests

# NLTK 다운로드 설정
nltk.download('punkt', download_dir=os.environ["NLTK_DATA"])
nltk.download('stopwords', download_dir=os.environ["NLTK_DATA"])

MAX_TOKEN_LENGTH = 256
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

client = discovery.build(
    "commentanalyzer",
    "v1alpha1",
    developerKey=PERSPECTIVE_API_KEY,
    discoveryServiceUrl="https://commentanalyzer.googleapis.com/$discovery/rest?version=v1alpha1",
    static_discovery=False,
)

def truncate_text(text: str, tokenizer: BartTokenizer, max_length: int = MAX_TOKEN_LENGTH) -> str:
    tokens = tokenizer.tokenize(text)
    if len(tokens) > max_length:
        tokens = tokens[:max_length]
        text = tokenizer.convert_tokens_to_string(tokens)
    return text

def score_tokens_with_perspective(text: str, tokenizer: BartTokenizer, top_k: int = 5) -> Tuple[dict, dict]:
    spans = text.split()
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

# def score_tokens_with_perspective(text: str, tokenizer: BartTokenizer, top_k: int = 5) -> Tuple[dict, dict]:
#     words = text.split()
#     attribute_names = ["TOXICITY", "SEVERE_TOXICITY"]
#     scored = {attr: [] for attr in attribute_names}

#     for idx, word in enumerate(words):
#         clean_word = word.strip(string.punctuation)
#         if not clean_word or clean_word.lower() in stop_words:
#             continue
#         try:
#             score_dict = call_perspective_api_googleclient(clean_word, PERSPECTIVE_API_KEY)
#             for attr in attribute_names:
#                 scored[attr].append((word.lower(), idx, score_dict.get(attr, 0.0)))
#         except Exception:
#             continue

#     top_tokens = {}
#     top_ids = {}
#     for attr in attribute_names:
#         attr_key = attr.lower()
#         top_scored = sorted(scored[attr], key=lambda x: x[2], reverse=True)[:top_k]
#         top_tokens[attr_key] = [s[0] for s in top_scored] + [None] * (top_k - len(top_scored))
#         top_ids[attr_key] = [s[1] for s in top_scored] + [None] * (top_k - len(top_scored))

#     return top_tokens, top_ids

def get_files(input_dir: str, extensions=[".txt", ".story", ".data"]) -> List[str]:
    files = []
    for ext in extensions:
        files.extend(Path(input_dir).rglob(f"*{ext}"))
    return [str(f) for f in files]

def load_prompt_jsonl(jsonl_path: str) -> List[str]:
    with open(jsonl_path, "r", encoding="utf-8") as f: 
        return [json.loads(line)["prompt"]["text"].replace("\n", " ").strip() for line in f]

def process_prompt_jsonl(jsonl_path: str, output_path_base: str, top_k: int = 5, max_workers: int = 16):
    os.makedirs(output_path_base, exist_ok=True)
    prompts = load_prompt_jsonl(jsonl_path)
    files = list(zip(range(len(prompts)), prompts))

    train, test = train_test_split(files, test_size=0.2, random_state=42)
    split_map = {"train": train, "test": test}

    for split, split_data in split_map.items():
        def process_entry(entry):
            idx, text = entry
            try:
                original_tokens = text.split()
                text = truncate_text(text, bart_tokenizer)
                tokens_by_attr, ids_by_attr = score_tokens_with_perspective(text, bart_tokenizer, top_k)
                sample = {
                    "id": f"{idx}",
                    "token": original_tokens,
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

def load_raw_texts(input_dir: str) -> List[Tuple[str, str]]:
    files = get_files(input_dir)
    texts = []
    for fpath in files:
        with open(fpath, "r", encoding="utf-8") as f:
            text = f.read().strip()
            if text:  # Only append non-empty content
                texts.append(text)
    return texts

def process_raw_texts(input_dir: str, output_path_base: str, top_k: int = 5, max_workers: int = 16):
    os.makedirs(output_path_base, exist_ok=True)
    texts = load_raw_texts(input_dir)
    files = list(zip(range(len(texts)), texts))

    train, test = train_test_split(files, test_size=0.2, random_state=42)
    split_map = {"train": train, "test": test}

    for split, split_data in split_map.items():
        def process_entry(entry):
            idx, text = entry
            try:
                original_tokens = text.split()
                text = truncate_text(text, bart_tokenizer)
                tokens_by_attr, ids_by_attr = score_tokens_with_perspective(text, bart_tokenizer, top_k)
                sample = {
                    "id": f"{idx}",
                    "token": original_tokens,
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

# Perspective API function using googleapiclient.discovery
# def call_perspective_api_googleclient(text: str, api_key: str) -> dict:
#     """
#     Call Perspective API using googleapiclient.discovery to get toxicity and severe toxicity scores.

#     Args:
#         text (str): The input text to analyze.
#         api_key (str): Your Google API key.

#     Returns:
#         dict: Dictionary with TOXICITY and SEVERE_TOXICITY scores.
#     """
#     client = discovery.build(
#         "commentanalyzer",
#         "v1alpha1",
#         developerKey=api_key,
#         discoveryServiceUrl="https://commentanalyzer.googleapis.com/$discovery/rest?version=v1alpha1",
#         static_discovery=False,
#     )

#     analyze_request = {
#         'comment': {'text': text},
#         'requestedAttributes': {'TOXICITY': {}, 'SEVERE_TOXICITY': {}},
#         'languages': ['en']
#     }

#     response = client.comments().analyze(body=analyze_request).execute()
#     scores = {
#         attr: response['attributeScores'][attr]['summaryScore']['value']
#         for attr in ['TOXICITY', 'SEVERE_TOXICITY']
#     }
#     return scores

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Process raw text files or JSONL file and extract toxic tokens using Perspective API.")
    parser.add_argument("--input_dir", type=str, help="Directory containing raw text files")
    parser.add_argument("--input_jsonl", type=str, help="Path to input JSONL file containing prompts")
    parser.add_argument("--output_path_base", type=str, required=True, help="Directory to save processed output JSON files")
    parser.add_argument("--top_k", type=int, default=5, help="Number of top tokens to extract per attribute")
    parser.add_argument("--max_workers", type=int, default=16, help="Number of threads for parallel processing")
    args = parser.parse_args()

    if args.input_dir:
        process_raw_texts(args.input_dir, args.output_path_base, top_k=args.top_k, max_workers=args.max_workers)
    elif args.input_jsonl:
        process_prompt_jsonl(args.input_jsonl, args.output_path_base, top_k=args.top_k, max_workers=args.max_workers)
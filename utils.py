import torch
import csv
import torch.nn.functional as F
import requests
import os
import re
from sklearn.decomposition import PCA
from googleapiclient import discovery
from tqdm import tqdm
import json
import time

API_KEY = 'AIzaSyBj3ck-qd0rZN_QurhsTiHs2gJ89-AegSQ'

# Placeholder: 로딩된 bias subspace 정보 (예: PCA basis 등)
BIAS_DIRECTION = None  # torch.Tensor of shape (d, k) where d is embedding dim, k is subspace dim

# === Perspective API constants ===
PERSPECTIVE_API_KEY = "AIzaSyBj3ck-qd0rZN_QurhsTiHs2gJ89-AegSQ"
PERSPECTIVE_URL = "https://commentanalyzer.googleapis.com/v1alpha1/comments:analyze"

def set_bias_subspace(direction_tensor: torch.Tensor):
    global BIAS_DIRECTION
    BIAS_DIRECTION = direction_tensor  # shape: (d, k)

def get_bias_subspace():
    return BIAS_DIRECTION

def project_onto_subspace(vec: torch.Tensor, basis: torch.Tensor):
    if vec.dim() == 1:
        vec = vec.unsqueeze(0)
    projection = vec @ basis @ basis.T
    return projection

def apply_convex_hull_perturbation(input_ids: torch.Tensor, model) -> torch.Tensor:
    embedding_layer = model.get_input_embeddings()
    with torch.no_grad():
        embedded = embedding_layer(input_ids)  # [B, T, D]
        token_embeddings = embedding_layer.weight  # [V, D]
        new_input_ids = []

        for b in range(embedded.shape[0]):
            new_ids = []
            for t in range(embedded.shape[1]):
                vec = embedded[b, t]
                sim = F.cosine_similarity(vec.unsqueeze(0), token_embeddings, dim=1)
                nearest_id = torch.argmax(sim).item()
                new_ids.append(nearest_id)
            new_input_ids.append(new_ids)

    return torch.tensor(new_input_ids, dtype=torch.long, device=input_ids.device)

def build_bias_subspace(definite_sets, tokenizer, model, subspace_dim=5):
    embeddings = model.get_input_embeddings().weight.detach()
    bias_vectors = []

    for pair in definite_sets:
        if len(pair) != 2:
            continue
        word_a, word_b = pair
        ids_a = tokenizer(word_a, add_special_tokens=False).input_ids
        ids_b = tokenizer(word_b, add_special_tokens=False).input_ids
        if not ids_a or not ids_b:
            continue
        vec_a = embeddings[ids_a[0]]
        vec_b = embeddings[ids_b[0]]
        bias_vectors.append((vec_a - vec_b).cpu().numpy())

    if len(bias_vectors) < subspace_dim:
        subspace_dim = len(bias_vectors)

    pca = PCA(n_components=subspace_dim)
    pca.fit(bias_vectors)
    return torch.tensor(pca.components_.T, dtype=torch.float32)

def load_and_set_bias_subspace(json_path, tokenizer, model, subspace_dim=5):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    definite_sets = data["definite_sets"]
    subspace = build_bias_subspace(definite_sets, tokenizer, model, subspace_dim=subspace_dim)
    set_bias_subspace(subspace)
    global BIAS_DIRECTION
    BIAS_DIRECTION = BIAS_DIRECTION.to(model.device)
    return subspace

# def call_perspective_api(texts, requested_attributes=["TOXICITY", "SEVERE_TOXICITY"]):
#     headers = {"Content-Type": "application/json"}
#     results = {attr.lower(): [] for attr in requested_attributes}

#     for text in tqdm(texts, desc="[Perspective API]"):
#         data = {
#             "comment": {"text": text},
#             "languages": ["en"],
#             "requestedAttributes": {attr: {} for attr in requested_attributes}
#         }
#         response = requests.post(
#             f"{PERSPECTIVE_URL}?key={PERSPECTIVE_API_KEY}",
#             headers=headers,
#             data=json.dumps(data)
#         )
#         if response.status_code == 200:
#             result = response.json()
#             for attr in requested_attributes:
#                 try:
#                     score = result["attributeScores"][attr]["summaryScore"]["value"]
#                 except KeyError:
#                     score = 0.0
#                 results[attr.lower()].append(score)
#         else:
#             for attr in requested_attributes:
#                 results[attr.lower()].append(0.0)
#     return results

client = discovery.build(
  "commentanalyzer",
  "v1alpha1",
  developerKey=API_KEY,
  discoveryServiceUrl="https://commentanalyzer.googleapis.com/$discovery/rest?version=v1alpha1",
  static_discovery=False,
)

def call_perspective_api(texts, requested_attributes=["TOXICITY", "INSULT", "THREAT", "IDENTITY_ATTACK"]):
    print(f"[INFO] Sending {len(texts)} texts to Perspective API")
    results = {attr.lower(): [] for attr in requested_attributes}

    for text in tqdm(texts, desc="[Perspective API]"):
        if not text.strip():
            print(f"[WARNING] Skipping empty text input.")
            for attr in requested_attributes:
                results[attr.lower()].append(0.0)
            continue
        analyze_request = {
            'comment': {'text': text},
            'languages': ['en'],
            'requestedAttributes': {attr: {} for attr in requested_attributes}
        }

        while True:
            try:
                response = client.comments().analyze(body=analyze_request).execute()
                for attr in requested_attributes:
                    try:
                        score = response["attributeScores"][attr]["summaryScore"]["value"]
                        print(f'\n{score}')
                    except KeyError:
                        score = 0.0
                    results[attr.lower()].append(score)
                break  # break out of retry loop if successful
            except Exception as e:
                print(f"[ERROR] Perspective API failed for input: {text[:100]}... — {e}")
                time.sleep(1.5)
                continue
    return results

def extract_category_topn(model_checkpoint):
    """
    Extracts category and topn from a model checkpoint path.
    Returns (category, topn) if found, else (None, None)
    Assumes structure: checkpoints/{dataset}/top{topn}/{category}/...
    """
    path_parts = os.path.normpath(model_checkpoint).split(os.sep)
    try:
        for i in range(len(path_parts) - 1):
            if re.match(r"top\d+", path_parts[i]):
                topn = int(path_parts[i].replace("top", ""))
                category = path_parts[i + 1]
                return category, topn
        return None, None
    except Exception:
        return None, None

def strip_prompt_from_pred(input_text, pred):
    if pred.startswith(input_text):
        return pred[len(input_text):].strip()
    return pred.strip()

def generate_result_dict(**kwargs):
    # Accepts all relevant metrics as kwargs
    keys = [
        "dataset", "category", "topn", "task_type", "mode", "rouge1", "rouge2", "rougeL", "bleu", "perplexity","bertscore_f1", "bleurt",
        "avg_toxicity", 
        # "avg_severe_toxicity"
        "avg_insult", "avg_threat", "avg_identity_attack"
    ]
    return {k: kwargs.get(k, None) for k in keys}

def print_eval_metrics(title, rouge_result, bleu_result, perplexity, bertscore_f1, bleurt, avg_toxicity, avg_insult, avg_threat, avg_identity_attack):
    print(f"\n[Evaluation Results - {title}]")
    print(f"ROUGE-1: {rouge_result['rouge1']:.4f}")
    print(f"ROUGE-2: {rouge_result['rouge2']:.4f}")
    print(f"ROUGE-L: {rouge_result['rougeL']:.4f}")
    print(f"BLEU: {bleu_result['bleu']:.4f}")
    print(f"BERTScore F1: {bertscore_f1:.4f}" if bertscore_f1 != 'N/A' else "BERTScore F1: N/A")
    print(f"BLEURT: {bleurt:.4f}" if bleurt != 'N/A' else "BLEURT: N/A")
    print(f"Perplexity: {perplexity:.4f}")
    print(f"Average Toxicity: {avg_toxicity:.4f}")
    # print(f"Average Severe Toxicity: {avg_severe_toxicity:.4f}")
    print(f"Average Insult: {avg_insult:.4f}")
    print(f"Average Threat: {avg_threat:.4f}")
    print(f"Average Identity Attack: {avg_identity_attack:.4f}")
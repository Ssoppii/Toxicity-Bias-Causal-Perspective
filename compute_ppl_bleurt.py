import os
import json
import argparse
import torch
import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

# ----------------------------
# Argument parsing
# ----------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, required=True, help="Dataset name (e.g., 'IMDB', 'RealToxicityPrompts')")
parser.add_argument("--mode", type=str, required=True, help="Mode (e.g., 'baseline', 'noperturb', 'debiased')")
parser.add_argument("--category", type=str, required=True, help="Category (e.g., 'toxicity', 'insult', 'threat', 'identity_attack')")
parser.add_argument("--input_json", type=str, required=True, help="Path to input JSON file (with 'generated')")
parser.add_argument("--output_json", type=str, required=True, help="Path to save output JSON with perplexity")
parser.add_argument("--output_csv", type=str, required=True, help="Path to save summary CSV")
args = parser.parse_args()

# ----------------------------
# Load model and tokenizer
# ----------------------------
model_name = "gpt2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name).eval()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# ----------------------------
# Load input JSON
# ----------------------------
with open(args.input_json, "r", encoding="utf-8") as f:
    data = json.load(f)

# ----------------------------
# Perplexity 계산 함수
# ----------------------------
def compute_perplexity(text):
    if not text.strip():
        return float("inf")
    encodings = tokenizer(text, return_tensors="pt", truncation=True, max_length=1024)
    input_ids = encodings.input_ids.to(device)
    with torch.no_grad():
        outputs = model(input_ids, labels=input_ids)
        loss = outputs.loss
        return torch.exp(loss).item()

# ----------------------------
# Perplexity 계산 루프
# ----------------------------
perplexity_list = []
for item in tqdm(data, desc="Computing Perplexities", ncols=80):
    ppl = compute_perplexity(item["generated"])
    item["perplexity"] = round(ppl, 4)
    perplexity_list.append(ppl)

# ----------------------------
# 평균 및 표준편차 계산
# ----------------------------
import numpy as np

mean_ppl = round(np.mean(perplexity_list), 4)
std_ppl = round(np.std(perplexity_list, ddof=1), 4)

print(f"✅ Mean Perplexity: {mean_ppl}")
print(f"✅ Std Perplexity: {std_ppl}")

# ----------------------------
# 저장: JSON
# ----------------------------
os.makedirs(os.path.dirname(args.output_json), exist_ok=True)
with open(args.output_json, "w", encoding="utf-8") as f:
    json.dump(data, f, indent=2, ensure_ascii=False)

# ----------------------------
# 저장: CSV 요약
# ----------------------------
# summary = pd.DataFrame([{
#     "dataset": args.dataset,
#     "mode": args.mode,
#     "category": args.category,
#     "mean_perplexity": mean_ppl,
#     "std_perplexity": std_ppl,
#     "num_samples": len(perplexity_list)
# }])
# summary.to_csv(args.output_csv, index=False)
# print(f"✅ Saved results to:\n- JSON: {args.output_json}\n- CSV: {args.output_csv}")
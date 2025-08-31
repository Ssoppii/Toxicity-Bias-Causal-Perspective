import os
import sys
import gc
import json
import time
import torch
import argparse
import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from evaluate import load as load_evaluate

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, required=True, help="Dataset name")
parser.add_argument("--mode", type=str, required=True, help="Mode name")
parser.add_argument("--category", type=str, required=True, help="Category name")
parser.add_argument("--input", type=str, required=True, help="Path to input JSON file (with 'generated' field)")
parser.add_argument("--topn", type=int, default=3, help="Top N samples to evaluate")
parser.add_argument("--reference", type=str, required=True, help="Path to reference JSON file")
parser.add_argument("--output", type=str, required=True, help="Path to output JSON with scores")
args = parser.parse_args()

# Load data
with open(args.input, "r", encoding="utf-8") as f:
    preds = json.load(f)
with open(args.reference, "r", encoding="utf-8") as f:
    refs = json.load(f)

predictions = [x["generated"] for x in preds]
references = [x["generated"] for x in refs]

# Perplexity
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ppl_model = AutoModelForCausalLM.from_pretrained("gpt2").to(device)
ppl_tokenizer = AutoTokenizer.from_pretrained("gpt2")
ppl_model.eval()

total_loss = 0.0
total_tokens = 0
with torch.no_grad():
    for sentence in tqdm(predictions, desc="Computing Perplexity", ncols=80):
        if not sentence.strip():
            continue
        encodings = ppl_tokenizer(sentence, return_tensors="pt")
        input_ids = encodings.input_ids.to(device)
        outputs = ppl_model(input_ids, labels=input_ids)
        loss = outputs.loss
        total_loss += loss.item() * input_ids.size(1)
        total_tokens += input_ids.size(1)

perplexity = torch.exp(torch.tensor(total_loss / total_tokens)).item() if total_tokens > 0 else float("inf")
del ppl_model
torch.cuda.empty_cache()

# BLEURT
bleurt = load_evaluate("bleurt", "bleurt-large-512")
bleurt_result = bleurt.compute(predictions=predictions, references=references)
avg_bleurt = sum(bleurt_result["scores"]) / len(bleurt_result["scores"])

# Append scores to each sample
for i in range(len(preds)):
    preds[i]["bleurt"] = round(bleurt_result["scores"][i], 4)

# Save JSON results
os.makedirs(os.path.dirname(args.output), exist_ok=True)
with open(args.output, "w", encoding="utf-8") as f:
    json.dump(preds, f, indent=2, ensure_ascii=False)

# Save CSV summary
summary_row = {
    "dataset": args.dataset,
    "mode": args.mode,
    "category": args.category,
    "topn": args.topn,
    "perplexity": round(perplexity, 4),
    "bleurt": round(avg_bleurt, 4),
}
csv_path = "eval_summary_perplexity_bleurt.csv"
write_header = not os.path.exists(csv_path)

with open(csv_path, "a", newline="", encoding="utf-8") as csvfile:
    writer = pd.DataFrame([summary_row])
    writer.to_csv(csvfile, header=write_header, index=False)

print(f"✅ Saved {len(preds)} samples with BLEURT scores to {args.output}")
print(f"✅ Appended summary to {csv_path}")
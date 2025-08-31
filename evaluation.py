import os
import sys
import gc
import torch
from tqdm import tqdm
import json
import datetime
from evaluate import load as load_evaluate
from utils import call_perspective_api, print_eval_metrics, generate_result_dict
from transformers import AutoTokenizer, AutoModelForCausalLM
from torch.nn import CrossEntropyLoss
import wandb
import argparse
import pandas as pd

if __name__ == "__main__":
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, "full_log_ver1.log")
    sys.stdout = open(log_file, "a")
    sys.stderr = sys.stdout
    print(f"\n\n[LOG STARTED] {datetime.datetime.now()}\n")

    parser = argparse.ArgumentParser()
    parser.add_argument("--generated_file", type=str, required=True)
    parser.add_argument("--reference_file", type=str, required=True)
    parser.add_argument("--category", type=str, required=True)
    parser.add_argument("--topn", type=int, required=True)
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--task_type", type=str, required=True)
    parser.add_argument("--mode", type=str, choices=["debiased", "noperturb", "baseline"], required=True)
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=4)
    args = parser.parse_args()

    print(f"[INFO] Running evaluation in '{args.mode}' mode")

    wandb.init(
        project=f"{args.category}_bias_evaluation_{args.dataset}_multi_category_top{args.topn}",
        name=f"{args.mode}-{args.category}-top{args.topn}"
    )
    
    with open(args.generated_file, "r") as f:
        predictions = [x["generated"] for x in json.load(f)]

    if args.mode == "baseline":
        with open(args.generated_file, "r") as f:
            references = [x["generated"] for x in json.load(f)]
    else:
        with open(args.reference_file, "r") as f:
            references = [x["generated"] for x in json.load(f)]

    rouge = load_evaluate("rouge")
    bleu = load_evaluate("bleu")
    bertscore = load_evaluate("bertscore")
    bleurt = load_evaluate("bleurt", "bleurt-large-512")
    
    rouge_result = rouge.compute(predictions=predictions, references=references)
    bleu_result = bleu.compute(predictions=predictions, references=references)
    try:
        bertscore_result = bertscore.compute(
            predictions=predictions,
            references=references,
            lang="en",
            model_type="bert-base-uncased",
            batch_size=4,
            device="cuda"
        )
    except RuntimeError as e:
        print(f"[WARNING] BERTScore failed on CUDA due to: {e}")
        print("[INFO] Falling back to CPU for BERTScore computation...")
        bertscore_result = bertscore.compute(
            predictions=predictions,
            references=references,
            lang="en",
            model_type="bert-base-uncased",
            batch_size=2,
            device="cpu"
        )
    avg_bertscore_f1 = sum(bertscore_result["f1"]) / len(bertscore_result["f1"])
    # Delete BERTScore memory
    if hasattr(bertscore, "cached_bertscorer"):
        if hasattr(bertscore.cached_bertscorer, "_model"):
            del bertscore.cached_bertscorer._model  # <-- 추가
        del bertscore.cached_bertscorer
    del bertscore_result
    gc.collect()
    torch.cuda.empty_cache()

    # Compute perplexity using pre-trained language model loss
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ppl_model_name = "gpt2"
    ppl_tokenizer = AutoTokenizer.from_pretrained(ppl_model_name)
    ppl_model = AutoModelForCausalLM.from_pretrained(ppl_model_name).to(device)
    ppl_model.eval()

    total_loss = 0.0
    total_tokens = 0

    with torch.no_grad():
        for sentence in predictions:
            if not sentence.strip():
                continue
            encodings = ppl_tokenizer(sentence, return_tensors="pt")
            input_ids = encodings.input_ids.to(device)
            target_ids = input_ids.clone()
            outputs = ppl_model(input_ids, labels=target_ids)
            loss = outputs.loss
            total_loss += loss.item() * input_ids.size(1)
            total_tokens += input_ids.size(1)

    perplexity = torch.exp(torch.tensor(total_loss / total_tokens)).item() if total_tokens > 0 else float("inf")
    del ppl_model
    torch.cuda.empty_cache()

    bleurt_result = bleurt.compute(predictions=predictions, references=references)
    avg_bleurt = sum(bleurt_result["scores"]) / len(bleurt_result["scores"])
    toxicity = call_perspective_api(predictions)

    avg_toxicity = sum(toxicity["toxicity"]) / len(toxicity["toxicity"])
    # avg_severe_toxicity = sum(toxicity["severe_toxicity"]) / len(toxicity["severe_toxicity"])
    avg_insult = sum(toxicity["insult"]) / len(toxicity["insult"])
    avg_threat = sum(toxicity["threat"]) / len(toxicity["threat"])
    avg_identity_attack = sum(toxicity["identity_attack"]) / len(toxicity["identity_attack"])

    print_eval_metrics(
        f"Category: {args.category} | top{args.topn}",
        rouge_result, bleu_result, perplexity=perplexity,
        bertscore_f1=avg_bertscore_f1, bleurt=avg_bleurt,
        avg_toxicity=avg_toxicity, 
        # avg_severe_toxicity=avg_severe_toxicity,
        avg_insult=avg_insult, avg_threat=avg_threat, avg_identity_attack=avg_identity_attack
    )

    result_dict = generate_result_dict(
        dataset=args.dataset,
        category=(args.category if args.mode != "baseline" else "NaN"),
        topn=args.topn,
        task_type=args.task_type,
        mode=args.mode,
        rouge1=rouge_result['rouge1'],
        rouge2=rouge_result['rouge2'],
        rougeL=rouge_result['rougeL'],
        bleu=bleu_result['bleu'],
        bertscore_f1=avg_bertscore_f1,
        bleurt=avg_bleurt,
        perplexity=perplexity,
        avg_toxicity=avg_toxicity,
        # avg_severe_toxicity=avg_severe_toxicity,
        avg_insult=avg_insult,
        avg_threat=avg_threat,
        avg_identity_attack=avg_identity_attack
    )

    df = pd.DataFrame([result_dict])
    df.to_csv("evaluation_results.csv", mode="a", header=not os.path.exists("evaluation_results.csv"), index=False)
    print("\nEvaluation results saved to evaluation_results.csv!")

    wandb.finish()

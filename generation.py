# generation.py
import os
import sys
import time
import datetime
import json
import argparse
import torch
from tqdm import tqdm
from transformers import BartTokenizer, BartForConditionalGeneration
from dataloader import DebiasingCategoryDataset
from torch.utils.data import DataLoader
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

# DDP initialization (when using torchrun)
world_size = int(os.environ.get("WORLD_SIZE", "1"))
if world_size > 1:
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
else:
    local_rank = 0

# def strip_prompt_from_pred(input_text, pred):
#     input_tokens = input_text.strip().split()
#     pred_tokens = pred.strip().split()
#     overlap_len = 0
#     while overlap_len < len(input_tokens) and overlap_len < len(pred_tokens) and input_tokens[overlap_len] == pred_tokens[overlap_len]:
#         overlap_len += 1
#     continuation = " ".join(pred_tokens[overlap_len:]).strip()
#     return continuation

# def strip_prompt_from_pred(input_text, pred):
#     if pred.startswith(prompt_prefix):
#         return pred[len(prompt_prefix):].strip()
#     return pred

if __name__ == "__main__":
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, "generation_log.log")
    sys.stdout = open(log_file, "a")
    sys.stderr = sys.stdout
    print(f"\n\n[LOG STARTED] {datetime.datetime.now()}\n")

    parser = argparse.ArgumentParser()
    # parser.add_argument("--model_checkpoint", type=str, required=True)
    # parser.add_argument("--model_name", type=str, default="facebook/bart-base")
    parser.add_argument("--model_name", type=str, default="sshleifer/distilbart-cnn-12-6")
    parser.add_argument("--input_file", type=str, required=True)
    parser.add_argument("--category", type=str, required=True)
    parser.add_argument("--topn", type=int, required=True)
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--mode", type=str, choices=["debiased", "noperturb", "baseline"], required=True)

    args = parser.parse_args()
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")

    tokenizer = BartTokenizer.from_pretrained(args.model_name, local_files_only=True)
    model = BartForConditionalGeneration.from_pretrained(args.model_name, local_files_only=True)
    # state_dict = torch.load(args.model_checkpoint, map_location=device)
    # model.load_state_dict(
    #     {k.replace("module.", ""): v for k, v in state_dict.items()},
    #     strict=False
    # )
    model = model.to(device)
    if world_size > 1:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank)
    model.eval()

    dataset = DebiasingCategoryDataset(
        original_file=args.input_file,
        debiased_file=args.input_file,
        tokenizer=tokenizer,
        topn=args.topn,
        category=args.category,
        max_length=args.max_length
    )

    if world_size > 1:
        sampler = DistributedSampler(dataset, num_replicas=world_size, rank=local_rank, shuffle=False)
        loader = DataLoader(dataset, batch_size=args.batch_size, sampler=sampler)
    else:
        loader = DataLoader(dataset, batch_size=args.batch_size)

    output_dir = os.path.join("outputs/train", args.dataset, f"top{args.topn}")
    # output_dir = os.path.join("outputs/test", args.dataset, f"top{args.topn}")
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f"{args.mode}_{args.category}_outputs.json")
    # output_file = os.path.join(output_dir, f"test_outputs.json")

    if world_size > 1:
        output_file = output_file.replace(".json", f".rank{local_rank}.json")

    generated_outputs = []
    with torch.no_grad():
        for batch in tqdm(loader, desc="[Generating]"):
            ids = batch["meta"]["id"] if "meta" in batch and "id" in batch["meta"] else list(range(len(batch["original"]["input_ids"])))
            original = {k: batch["original"][k].to(device) for k in ["input_ids", "attention_mask"]}
            debiased = {k: batch["debiased"][k].to(device) for k in ["input_ids", "attention_mask"]}
            # print(original)
            original_id = batch["original"]["id"]
            debiased_id = batch["debiased"]["id"]
            # def get_prompt_prefix(dataset_name):
            #     if "realtoxicity" in dataset_name.lower():
            #         return "Continue the following prompt:"
            #     elif "cnn" in dataset_name.lower() or "xsum" in dataset_name.lower():
            #         return "Summarize the following article in 2 sentences and do not use the article directly in the response:"
            #     elif "imdb" in dataset_name.lower():
            #         return "Generate a review continuation:"
            #     else:
            #         return ""
            # prompt_prefix = get_prompt_prefix(args.dataset)
            # input_texts = [f"{prompt_prefix} {tokenizer.decode(ids_, skip_special_tokens=True)}" for ids_ in original["input_ids"]]
            input_texts = [f"{tokenizer.decode(ids_, skip_special_tokens=True)}" for ids_ in original["input_ids"]]
            input_ids_for_suppression = tokenizer(input_texts, add_special_tokens=False)["input_ids"]
            # bad_words_ids = list({token_id for seq in input_ids_for_suppression for token_id in seq})
            # bad_words_ids = [[token_id] for token_id in bad_words_ids]  # must be list of lists
            new_inputs = tokenizer(input_texts, padding="max_length", truncation=True, max_length=args.max_length, return_tensors="pt").to(device)

            if world_size > 1:
                outputs = model.module.generate(
                    input_ids=new_inputs["input_ids"],
                    attention_mask=new_inputs["attention_mask"],
                    max_length=args.max_length,
                    num_beams=5,
                    no_repeat_ngram_size=3,
                    repetition_penalty=1.5,
                    early_stopping=True
                )
            else:
                outputs = model.generate(
                    input_ids=new_inputs["input_ids"],
                    attention_mask=new_inputs["attention_mask"],
                    max_length=args.max_length,
                    num_beams=5,
                    no_repeat_ngram_size=3,
                    repetition_penalty=1.5,
                    early_stopping=True
                )
            preds = tokenizer.batch_decode(outputs, skip_special_tokens=True)
            # outputs = [strip_prompt_from_pred(input_text, pred) for input_text, pred in zip(input_texts, preds)]
            print()
            print(preds)
            # print(outputs)
            generated_outputs.extend([(id_, continuation) for id_, continuation in zip(original_id, preds)])

    continuation_outputs = [{"id": id_, "generated": gen.strip()} for id_, gen in sorted(generated_outputs, key=lambda x: x[0])]
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(continuation_outputs, f, ensure_ascii=False, indent=2)

    print(f"[INFO] Saved generations to {output_file}")
    if world_size > 1:
        dist.destroy_process_group()

    # Merge outputs from all ranks if this is rank 0
    if local_rank == 0 and world_size > 1:
        time.sleep(3)  # short delay to ensure all ranks are done writing
        base_output_file = output_file.replace(f".rank{local_rank}.json", ".json")
        all_outputs = []
        # Wait for all rank files to exist
        while True:
            if all(os.path.exists(base_output_file.replace(".json", f".rank{rank}.json")) for rank in range(world_size)):
                break
            time.sleep(1)

        # Load and merge
        for rank in range(world_size):
            rank_file = base_output_file.replace(".json", f".rank{rank}.json")
            with open(rank_file, "r", encoding="utf-8") as rf:
                all_outputs.extend(json.load(rf))
        all_outputs = sorted(all_outputs, key=lambda x: x["id"])
        with open(base_output_file, "w", encoding="utf-8") as mf:
            json.dump(all_outputs, mf, ensure_ascii=False, indent=2)
        print(f"[INFO] Merged outputs saved to {base_output_file}")
        # delete individual rank files
        for rank in range(world_size):
            rank_file = base_output_file.replace(".json", f".rank{rank}.json")
            if os.path.exists(rank_file):
                os.remove(rank_file)
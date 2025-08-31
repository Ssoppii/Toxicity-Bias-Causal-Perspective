import os
import sys
import datetime
import argparse
import logging
from tqdm import tqdm

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.optim import AdamW
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
from transformers import BartTokenizer, BartForConditionalGeneration, get_scheduler

from dataloader import DebiasingCategoryDataset
from model import compute_alignment_loss
from utils import load_and_set_bias_subspace, apply_convex_hull_perturbation


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--category", type=str, required=True)
    parser.add_argument("--topn", type=int, required=True)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--train_file", type=str, required=True)
    parser.add_argument("--debiased_file", type=str, required=True)
    parser.add_argument("--subspace_json", type=str, required=True)
    parser.add_argument("--mode", type=str, choices=["baseline", "debiased", "noperturb"], required=True)
    parser.add_argument("--local_rank", type=int, default=-1)
    return parser.parse_args()


def setup_logger(log_file):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter(fmt='%(asctime)s - %(levelname)s - %(message)s',
                                  datefmt='%Y-%m-%d %H:%M:%S')

    # Stream handler for console output
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    # File handler for logging to a file
    fh = logging.FileHandler(log_file)
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    return logger


def train(args):
    os.makedirs("./trained_models", exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logger = logging.getLogger()
    if args.local_rank in [-1, 0]:
        logger.info(f"Starting training with device {device}")

    tokenizer = BartTokenizer.from_pretrained(args.model_name)
    model = BartForConditionalGeneration.from_pretrained(args.model_name)
    model.to(device)

    import json
    with open(f"./checkpoints/{args.dataset}/top{args.topn}//{args.category}/baseline_{args.category}_outputs.json") as f:
        teacher_output_dict = json.load(f)

    if args.local_rank != -1:
        model = DDP(model, device_ids=[args.local_rank], output_device=args.local_rank)

    if args.local_rank in [-1, 0]:
        logger.info("[INFO] Loading bias subspace...")
    load_and_set_bias_subspace(args.subspace_json, tokenizer, model, subspace_dim=5)

    valid_ids = set(item["id"] for item in teacher_output_dict)

    dataset = DebiasingCategoryDataset(
        original_file=args.train_file,
        debiased_file=args.debiased_file,
        tokenizer=tokenizer,
        topn=args.topn,
        category=args.category,
        max_length=args.max_length,
        valid_ids=valid_ids
    )

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=(args.local_rank == -1),
        num_workers=4,
        pin_memory=True
    )

    optimizer = AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    lr_scheduler = get_scheduler(
        name="linear",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=len(dataloader) * args.epochs
    )

    scaler = GradScaler()

    model.train()
    for epoch in range(args.epochs):

        if args.local_rank in [-1, 0]:
            logger.info(f"Epoch {epoch+1}/{args.epochs} - Mode: {args.mode}")
        running_loss = 0.0

        if args.local_rank != -1:
            from torch.utils.data.distributed import DistributedSampler
            sampler = dataloader.sampler
            if hasattr(sampler, 'set_epoch'):
                sampler.set_epoch(epoch)

        for batch in tqdm(dataloader, disable=(args.local_rank not in [-1, 0])):
            original = {k: batch["original"][k].to(device) for k in ["input_ids", "attention_mask"]}
            with autocast():
                debiased = {k: batch["debiased"][k].to(device) for k in ["input_ids", "attention_mask"]}
                if args.mode == "debiased":
                    perturbed_input_ids = apply_convex_hull_perturbation(debiased["input_ids"], model)
                    debiased["input_ids"] = perturbed_input_ids

                if args.local_rank in [-1, 0]:
                    logger.info(f"[Teacher Mapping] Using batch id mapping for batch size: {len(batch['original']['input_ids'])}")
                original_ids = batch["original"]["id"]
                id_to_text = {item["id"]: item["generated"] for item in teacher_output_dict}
                target_texts = [id_to_text[str(i)] for i in original_ids]
                labels = tokenizer(target_texts, return_tensors="pt", padding=True, truncation=True, max_length=args.max_length).input_ids.to(device)

                outputs = model(
                    input_ids=original["input_ids"],
                    attention_mask=original["attention_mask"],
                    labels=labels
                )
                total_loss = outputs.loss.mean()
                if args.local_rank in [-1, 0]:
                    predicted_ids = outputs.logits.argmax(dim=-1)

            optimizer.zero_grad()
            scaler.scale(total_loss).backward()
            scaler.step(optimizer)
            scaler.update()
            lr_scheduler.step()

            running_loss += total_loss.item()

        avg_loss = running_loss / len(dataloader)
        
        if args.local_rank in [-1, 0]:
            logger.info(f"Epoch {epoch+1} Average Loss: {avg_loss:.4f}")
            save_path = f"./tmp/trained_models/{args.mode}/epoch_{epoch+1}/top{args.topn}/{args.dataset}/{args.category}"
            os.makedirs(save_path, exist_ok=True)
            model_to_save = model.module if hasattr(model, "module") else model
            model_to_save.save_pretrained(save_path)
            tokenizer.save_pretrained(save_path)
            logger.info(f"Model saved to {save_path}")


def main():
    args = parse_args()
    if args.local_rank != -1:
        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(args.local_rank)
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"{args.mode}_{args.dataset}_{args.topn}_{args.category}_train.log")
    setup_logger(log_file)
    logger = logging.getLogger()
    logger.info(f"\n\n[LOG STARTED] {datetime.datetime.now()}\n")

    train(args)


if __name__ == "__main__":
    main()
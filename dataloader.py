import json
import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer

class DebiasingCategoryDataset(Dataset):
    def __init__(self, original_file, debiased_file, tokenizer: PreTrainedTokenizer, topn, category, max_length=512, valid_ids=None):
        with open(original_file, "r", encoding="utf-8") as f:
            self.original_data = json.load(f)
        
        if debiased_file is not None:
            with open(debiased_file, "r", encoding="utf-8") as f:
                self.debiased_data = json.load(f)
            assert len(self.original_data) == len(self.debiased_data)
        else:
            self.debiased_data = None

        # Filter by valid_ids if provided
        if valid_ids is not None:
            valid_ids = set(str(i) for i in valid_ids)
            self.original_data = [ex for ex in self.original_data if str(ex["id"]) in valid_ids]
            if self.debiased_data is not None:
                self.debiased_data = [ex for ex in self.debiased_data if str(ex["id"]) in valid_ids]

        self.tokenizer = tokenizer
        self.max_length = max_length
        self.category = category
        self.topn = topn

    def __len__(self):
        return len(self.original_data)

    def __getitem__(self, idx):
        orig_sample = self.original_data[idx]
        if self.debiased_data is not None:
            debiased_sample = self.debiased_data[idx]

            orig_tokens = orig_sample["token"]
            debiased_tokens = debiased_sample["token"]
            token_id_key = f"{self.category}_token_ids"
            tokens_key = f"{self.category}_tokens"

            token_id_list = debiased_sample[token_id_key]
            tokens_list = debiased_sample[tokens_key]

            if token_id_key in debiased_sample:
                for i in range(self.topn):
                    idx = token_id_list[i]
                    if idx is None:
                        # Skip null positions
                        continue
                    debiased_tokens[idx] = tokens_list[i]
        else:
            # If no debiased file, use original tokens as debiased tokens
            orig_tokens = orig_sample["token"]
            debiased_tokens = orig_tokens.copy()

        # 토크나이즈
        orig_enc = self.tokenizer(" ".join(orig_tokens), padding="max_length", truncation=True, max_length=self.max_length, return_tensors="pt")
        debiased_enc = self.tokenizer(" ".join(debiased_tokens), padding="max_length", truncation=True, max_length=self.max_length, return_tensors="pt")

        return {
            "original": {
                "id": orig_sample["id"],
                "input_ids": orig_enc["input_ids"].squeeze(0),
                "attention_mask": orig_enc["attention_mask"].squeeze(0),
            },
            "debiased": {
                "id": orig_sample["id"],
                "input_ids": debiased_enc["input_ids"].squeeze(0),
                "attention_mask": debiased_enc["attention_mask"].squeeze(0),
            }
        }
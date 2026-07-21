"""
Training loop for Study 1 (GPT-2 fine-tuning).

PROVENANCE NOTE: the original experiments in Study 1 were run interactively
in a Colab notebook built up over many sessions. The curriculum logic
(imported from src/curriculum, unchanged) is exact. The helper functions
below (ModArithmeticDataset, collate, exact_match_accuracy, weight_norm)
are faithful reconstructions of what was run, based on their call
signatures and behavior as used throughout the project, not a
byte-for-byte export of the original notebook cell. If you are trying to
exactly reproduce Study 1's numbers, treat this file as a close reference
implementation rather than the literal executed source, and see the
"What is and isn't in this repo" section of the README.

Study 2 (src/study2_scratch/train.py), by contrast, is extracted directly
from a notebook that was executed end-to-end and verified before use.
"""
import os
import json
import math
from dataclasses import dataclass, asdict

import torch
from torch.utils.data import Dataset, DataLoader

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "curriculum"))
from curriculum import make_pairs, split_data, apply_curriculum  # noqa: E402


@dataclass
class Config:
    modulus: int = 97
    operation: str = "add"
    train_fraction: float = 0.5
    model_name: str = "gpt2"
    lr: float = 1e-4
    weight_decay: float = 0.5
    batch_size: int = 64
    max_steps: int = 20000
    warmup_steps: int = 100
    grad_clip: float = 1.0
    eval_every: int = 250
    eval_batch_examples: int = 256
    checkpoint_every: int = 1000
    curriculum: str = "random"
    seed: int = 0
    out_dir: str = ""


def set_seed(seed):
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def pair_to_text(a, b, c, operation):
    op_symbol = "+" if operation == "add" else "*"
    return "%d %s %d = %d" % (a, op_symbol, b, c)


class ModArithmeticDataset(Dataset):
    def __init__(self, pairs, tokenizer, operation):
        self.pairs = pairs
        self.tokenizer = tokenizer
        self.operation = operation

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        a, b, c = self.pairs[idx]
        text = pair_to_text(a, b, c, self.operation)
        return self.tokenizer(text, return_tensors="pt")["input_ids"][0]


def make_collate_fn(pad_token_id):
    def collate(batch):
        max_len = max(x.shape[0] for x in batch)
        input_ids = torch.full((len(batch), max_len), pad_token_id, dtype=torch.long)
        attention_mask = torch.zeros((len(batch), max_len), dtype=torch.long)
        for i, x in enumerate(batch):
            input_ids[i, : x.shape[0]] = x
            attention_mask[i, : x.shape[0]] = 1
        labels = input_ids.clone()
        labels[attention_mask == 0] = -100
        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}

    return collate


@torch.no_grad()
def exact_match_accuracy(model, tokenizer, pairs, cfg, device, n_examples):
    import random as _random

    model.eval()
    sample = pairs if len(pairs) <= n_examples else _random.sample(pairs, n_examples)
    correct = 0
    for a, b, c in sample:
        op_symbol = "+" if cfg.operation == "add" else "*"
        prompt = "%d %s %d = " % (a, op_symbol, b)
        input_ids = tokenizer(prompt, return_tensors="pt")["input_ids"].to(device)
        out = model.generate(
            input_ids, max_new_tokens=4, do_sample=False, pad_token_id=tokenizer.pad_token_id
        )
        generated = tokenizer.decode(out[0][input_ids.shape[1] :], skip_special_tokens=True)
        predicted = generated.strip().split()[0] if generated.strip() else ""
        if predicted == str(c):
            correct += 1
    model.train()
    return correct / len(sample)


def weight_norm(model):
    total = 0.0
    for p in model.parameters():
        total += float(p.detach().pow(2).sum().item())
    return math.sqrt(total)


def grad_norm(model):
    total = 0.0
    for p in model.parameters():
        if p.grad is not None:
            total += float(p.grad.detach().pow(2).sum().item())
    return math.sqrt(total)


def train(cfg, device=None):
    from transformers import GPT2LMHeadModel, GPT2TokenizerFast, get_cosine_schedule_with_warmup

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(cfg.seed)
    os.makedirs(cfg.out_dir, exist_ok=True)

    tokenizer = GPT2TokenizerFast.from_pretrained(cfg.model_name)
    tokenizer.pad_token = tokenizer.eos_token
    model = GPT2LMHeadModel.from_pretrained(cfg.model_name).to(device)

    pairs = make_pairs(cfg.modulus, cfg.operation)
    train_pairs, val_pairs = split_data(pairs, cfg.train_fraction, cfg.seed)
    collate = make_collate_fn(tokenizer.pad_token_id)

    ordered = apply_curriculum(train_pairs, cfg.curriculum, cfg.modulus, cfg.operation, cfg.seed)
    train_ds = ModArithmeticDataset(ordered, tokenizer, cfg.operation)
    shuffle = cfg.curriculum == "random"
    train_loader = DataLoader(
        train_ds, batch_size=cfg.batch_size, shuffle=shuffle, collate_fn=collate, drop_last=True
    )

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay, betas=(0.9, 0.98)
    )
    scheduler = get_cosine_schedule_with_warmup(optimizer, cfg.warmup_steps, cfg.max_steps)
    history = []

    def infinite(loader):
        while True:
            for b in loader:
                yield b

    data_iter = infinite(train_loader)
    model.train()

    for step in range(cfg.max_steps + 1):
        batch = next(data_iter)
        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)
        attn = batch["attention_mask"].to(device)

        out = model(input_ids=input_ids, attention_mask=attn, labels=labels)
        loss = out.loss
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        optimizer.step()
        scheduler.step()

        if step % cfg.eval_every == 0:
            train_acc = exact_match_accuracy(
                model, tokenizer, train_pairs, cfg, device, cfg.eval_batch_examples
            )
            val_acc = exact_match_accuracy(
                model, tokenizer, val_pairs, cfg, device, cfg.eval_batch_examples
            )
            row = {
                "step": step,
                "train_loss": float(loss.item()),
                "train_acc": train_acc,
                "val_acc": val_acc,
                "gen_gap": train_acc - val_acc,
                "weight_norm": weight_norm(model),
            }
            history.append(row)
            print(
                "step %6d | loss %.4f | train %.3f | val %.3f | [%s]"
                % (step, row["train_loss"], train_acc, val_acc, cfg.curriculum)
            )
            with open(os.path.join(cfg.out_dir, "history.json"), "w") as f:
                json.dump({"config": asdict(cfg), "history": history}, f, indent=2)

    return history


if __name__ == "__main__":
    print(__doc__)

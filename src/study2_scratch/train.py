"""
Training loop for Study 2 (from-scratch transformer).

This is the code that produced the 15-run grid reported in Table 2 of the
paper (5 curricula x 3 seeds, addition mod 97). It was run in Colab on a
T4 GPU; this file is the CPU/GPU-agnostic source, unchanged from what was
executed there.
"""
import os
import json
import math
import random
import time
from dataclasses import dataclass, asdict

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "curriculum"))
from curriculum import make_pairs, split_data, apply_curriculum  # noqa: E402

from model import GrokkTransformer  # noqa: E402


@dataclass
class Config:
    modulus: int = 97
    operation: str = "add"
    train_fraction: float = 0.5
    d_model: int = 128
    n_heads: int = 4
    n_layers: int = 2
    lr: float = 1e-3
    weight_decay: float = 1.0
    batch_size: int = 512
    max_steps: int = 30000
    warmup_steps: int = 100
    eval_every: int = 250
    embed_every: int = 500
    curriculum: str = "random"
    seed: int = 0
    out_dir: str = ""


def set_all_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def weight_norm(model):
    total = 0.0
    for p in model.parameters():
        total += float(p.detach().pow(2).sum().item())
    return math.sqrt(total)


@torch.no_grad()
def accuracy(model, x, y, device, bs=2048):
    model.eval()
    correct = 0
    for i in range(0, len(x), bs):
        logits = model(x[i : i + bs].to(device))
        correct += int((logits.argmax(-1) == y[i : i + bs].to(device)).sum().item())
    model.train()
    return correct / len(x)


def to_tensors(pairs, modulus):
    p = modulus
    OP, EQ = p, p + 1
    x = torch.tensor([[a, OP, b, EQ] for a, b, _ in pairs], dtype=torch.long)
    y = torch.tensor([c for _, _, c in pairs], dtype=torch.long)
    return x, y


def train(cfg, device=None, verbose_every=2500):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    set_all_seeds(cfg.seed)
    os.makedirs(cfg.out_dir, exist_ok=True)

    pairs = make_pairs(cfg.modulus, cfg.operation)
    train_pairs, val_pairs = split_data(pairs, cfg.train_fraction, cfg.seed)
    ordered = apply_curriculum(train_pairs, cfg.curriculum, cfg.modulus, cfg.operation, cfg.seed)

    xtr, ytr = to_tensors(ordered, cfg.modulus)
    xva, yva = to_tensors(val_pairs, cfg.modulus)
    xtr_eval, ytr_eval = to_tensors(train_pairs, cfg.modulus)

    shuffle = cfg.curriculum == "random"
    gen = torch.Generator()
    gen.manual_seed(cfg.seed)
    loader = DataLoader(
        TensorDataset(xtr, ytr),
        batch_size=cfg.batch_size,
        shuffle=shuffle,
        drop_last=True,
        num_workers=0,
        generator=gen,
    )

    model = GrokkTransformer(cfg.modulus, cfg.d_model, cfg.n_heads, cfg.n_layers).to(device)
    opt = torch.optim.AdamW(
        model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay, betas=(0.9, 0.98)
    )
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / cfg.warmup_steps)
    )

    history, embed_steps, embed_mats = [], [], []

    def loop(dl):
        while True:
            for b in dl:
                yield b

    it = loop(loader)
    t0 = time.time()
    model.train()

    for step in range(cfg.max_steps + 1):
        xb, yb = next(it)
        logits = model(xb.to(device))
        loss = F.cross_entropy(logits, yb.to(device))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sched.step()

        if step % cfg.eval_every == 0:
            tr_acc = accuracy(model, xtr_eval, ytr_eval, device)
            va_acc = accuracy(model, xva, yva, device)
            history.append(
                {
                    "step": step,
                    "train_loss": float(loss.item()),
                    "train_acc": tr_acc,
                    "val_acc": va_acc,
                    "gen_gap": tr_acc - va_acc,
                    "weight_norm": weight_norm(model),
                }
            )
            if step % verbose_every == 0:
                print(
                    "  step %6d | loss %.4f | train %.3f | val %.3f | wnorm %.1f"
                    % (step, loss.item(), tr_acc, va_acc, history[-1]["weight_norm"])
                )

        if step % cfg.embed_every == 0:
            embed_steps.append(step)
            embed_mats.append(model.number_embeddings())

    with open(os.path.join(cfg.out_dir, "history.json"), "w") as f:
        json.dump({"config": asdict(cfg), "history": history}, f)
    np.savez_compressed(
        os.path.join(cfg.out_dir, "embed_log.npz"),
        steps=np.array(embed_steps),
        embeds=np.stack(embed_mats).astype(np.float32),
    )

    mins = (time.time() - t0) / 60
    grokk = next((r["step"] for r in history if r["val_acc"] >= 0.95), None)
    print("  done in %.1f min | grokk step: %s" % (mins, grokk))
    return history


if __name__ == "__main__":
    # Smoke test: p=31 grokks in well under a minute on GPU, ~2.4 min on CPU.
    # p=13 at 50% train is only 84 examples, below the data threshold where
    # grokking is possible at all (see Nanda et al. Appendix C.2) -- it
    # memorizes and never generalizes. Do not lower the smoke-test modulus
    # below the point where it actually grokks; that was the bug caught
    # during development of this repo.
    cfg = Config(
        modulus=31, train_fraction=0.5, weight_decay=1.0, max_steps=8000,
        eval_every=500, embed_every=2000, batch_size=128,
        curriculum="random", seed=0, out_dir="/tmp/grokk_smoke",
    )
    h = train(cfg, verbose_every=2000)
    g = next((r["step"] for r in h if r["val_acc"] >= 0.95), None)
    assert g is not None, "smoke test did not grok -- something is broken"
    print("\nsmoke test passed: grokked at step %d (expected ~4500)" % g)

"""Train the small char-level GPT on data/corpus.txt.

Usage:  python train.py                 # sane CPU defaults
        python train.py --max_iters 8000 --n_embd 256
Checkpoints are written to out/ckpt.pt every eval; safe to Ctrl-C anytime.
"""
import argparse
import json
import os
import time

import torch

from model import GPT, GPTConfig

CORPUS = os.path.join("data", "corpus.txt")
OUT_DIR = "out"


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--block_size", type=int, default=128)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--n_layer", type=int, default=4)
    p.add_argument("--n_head", type=int, default=4)
    p.add_argument("--n_embd", type=int, default=192)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--max_iters", type=int, default=5000)
    p.add_argument("--eval_interval", type=int, default=250)
    p.add_argument("--eval_iters", type=int, default=50)
    p.add_argument("--warmup", type=int, default=200)
    p.add_argument("--seed", type=int, default=1337)
    return p.parse_args()


def main():
    args = get_args()
    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.set_num_threads(os.cpu_count() or 4)
    print(f"device: {device} | threads: {torch.get_num_threads()}")

    text = open(CORPUS, encoding="utf-8").read()
    chars = sorted(set(text))
    stoi = {c: i for i, c in enumerate(chars)}
    itos = {i: c for i, c in enumerate(chars)}
    vocab_size = len(chars)
    print(f"corpus chars: {len(text):,} | vocab: {vocab_size}")

    data = torch.tensor([stoi[c] for c in text], dtype=torch.long)
    n = int(0.9 * len(data))
    train_data, val_data = data[:n], data[n:]

    def get_batch(split):
        d = train_data if split == "train" else val_data
        ix = torch.randint(len(d) - args.block_size - 1, (args.batch_size,))
        x = torch.stack([d[i:i + args.block_size] for i in ix])
        y = torch.stack([d[i + 1:i + 1 + args.block_size] for i in ix])
        return x.to(device), y.to(device)

    cfg = GPTConfig(
        vocab_size=vocab_size, block_size=args.block_size,
        n_layer=args.n_layer, n_head=args.n_head, n_embd=args.n_embd,
        dropout=args.dropout,
    )
    model = GPT(cfg).to(device)
    print(f"params: {model.num_params()/1e6:.2f}M")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95),
                            weight_decay=0.1)

    def lr_at(it):
        if it < args.warmup:
            return args.lr * (it + 1) / args.warmup
        # cosine decay to 10% of lr
        prog = (it - args.warmup) / max(1, args.max_iters - args.warmup)
        import math
        return args.lr * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * prog)))

    @torch.no_grad()
    def estimate_loss():
        model.eval()
        out = {}
        for split in ("train", "val"):
            losses = torch.zeros(args.eval_iters)
            for k in range(args.eval_iters):
                _, loss = model(*get_batch(split))
                losses[k] = loss.item()
            out[split] = losses.mean().item()
        model.train()
        return out

    os.makedirs(OUT_DIR, exist_ok=True)

    def save():
        torch.save({
            "model": model.state_dict(),
            "config": cfg.__dict__,
            "stoi": stoi, "itos": itos,
        }, os.path.join(OUT_DIR, "ckpt.pt"))

    best_val = float("inf")
    t0 = time.time()
    for it in range(args.max_iters + 1):
        for g in opt.param_groups:
            g["lr"] = lr_at(it)

        if it % args.eval_interval == 0 or it == args.max_iters:
            losses = estimate_loss()
            dt = time.time() - t0
            print(f"iter {it:5d} | train {losses['train']:.3f} | "
                  f"val {losses['val']:.3f} | lr {lr_at(it):.1e} | {dt:.0f}s",
                  flush=True)
            if losses["val"] < best_val:
                best_val = losses["val"]
                save()

        xb, yb = get_batch("train")
        _, loss = model(xb, yb)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

    save()
    print(f"done. best val loss {best_val:.3f}. checkpoint -> {OUT_DIR}/ckpt.pt")


if __name__ == "__main__":
    main()

"""Generate group-chat-style text from the trained model.

Examples:
    python sample.py                          # free-run a conversation
    python sample.py --prompt "wes: "         # continue as a given speaker
    python sample.py --prompt "abdullah: " --temperature 0.9 --tokens 400
"""
import argparse
import os

import torch

from model import GPT, GPTConfig

CKPT = os.path.join("out", "ckpt.pt")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--prompt", type=str, default="")
    p.add_argument("--tokens", type=int, default=500)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top_k", type=int, default=40)
    p.add_argument("--seed", type=int, default=None)
    args = p.parse_args()

    if args.seed is not None:
        torch.manual_seed(args.seed)

    if not os.path.exists(CKPT):
        raise SystemExit("No checkpoint found. Run train.py first.")

    ck = torch.load(CKPT, map_location="cpu", weights_only=False)
    cfg = GPTConfig(**ck["config"])
    model = GPT(cfg)
    model.load_state_dict(ck["model"])
    model.eval()
    stoi, itos = ck["stoi"], ck["itos"]

    prompt = args.prompt if args.prompt else "\n"
    # keep only chars the model knows
    ids = [stoi[c] for c in prompt if c in stoi] or [stoi.get("\n", 0)]
    idx = torch.tensor([ids], dtype=torch.long)

    out = model.generate(idx, args.tokens, temperature=args.temperature,
                         top_k=args.top_k)[0].tolist()
    text = "".join(itos[i] for i in out)
    print(text)


if __name__ == "__main__":
    main()

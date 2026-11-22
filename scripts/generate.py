"""Sample text from trained Palimpsa checkpoints (one or several models, same prompts, same seed).

Checkpoints are loaded with weights_only=True (tensors and plain values only, no arbitrary pickle code). The prompt
is preceded by EOS so it starts a new document, as in training; generation stops at the next EOS or --max-new.
Sampling is palimpsa.sampling.generate (temperature, top-k, top-p, CTRL repetition penalty; no KV cache).

Usage:
  python scripts/generate.py --ckpt runs/local_10m/B_d435/ckpt.pt --ckpt runs/local_10m/S_d435/ckpt.pt \
      --prompt "The history of the Roman Empire" --prompt "def fibonacci(n):" [--repetition-penalty 1.2]
  python scripts/generate.py --ckpt ... --interactive        # type prompts, empty line to quit
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from palimpsa.model import Config, PalimpsaLM  # noqa: E402
from palimpsa.sampling import generate  # noqa: E402


def load_model(path: str, device: str) -> tuple[PalimpsaLM, dict]:
    ck = torch.load(path, map_location="cpu", weights_only=True)
    model = PalimpsaLM(Config(**ck["config"]))
    model.load_state_dict(ck["model"])
    model.to(device).eval()
    info = {"variant": ck["config"]["variant"], "step": ck["step"], "lr": ck["args"].get("lr"),
            "params": sum(p.numel() for p in model.parameters())}
    return model, info


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", action="append", required=True)
    ap.add_argument("--prompt", action="append", default=[])
    ap.add_argument("--interactive", action="store_true")
    ap.add_argument("--max-new", type=int, default=120)
    ap.add_argument("--context", type=int, default=1024)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top-k", type=int, default=50)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--repetition-penalty", type=float, default=1.0,
                    help="CTRL penalty for tokens already in the context; 1.0 = off, 1.1-1.3 typical")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    from tokenizers import Tokenizer

    tok = Tokenizer.from_file(str(ROOT / "data" / "tokenizer" / "tokenizer.json"))
    models = []
    for path in args.ckpt:
        model, info = load_model(path, args.device)
        generate(model, [[1, 2, 3]], 4, temperature=0)  # warm-up: CUDA init and kernel selection stay out of tok/s
        print(f"loaded {path}: {info}", flush=True)
        models.append((f"{info['variant']}@{info['step']}", model))

    def run(prompt: str) -> None:
        ids = tok.encode(prompt, add_special_tokens=False).ids
        print(f"\n### prompt: {prompt!r}", flush=True)
        for name, model in models:
            t0 = time.time()
            new = generate(model, [ids], args.max_new, args.temperature, args.top_k, args.top_p,
                           args.repetition_penalty, args.context, torch.Generator().manual_seed(args.seed))[0]
            dt = time.time() - t0  # generate() returns Python lists, so the GPU work is finished here
            text = tok.decode(new, skip_special_tokens=False)
            print(f"--- {name} ({len(new)} tokens, {dt:.2f} s, {len(new) / max(dt, 1e-9):.1f} tok/s)\n{prompt}{text}",
                  flush=True)

    for prompt in args.prompt:
        run(prompt)
    if args.interactive:
        while True:
            try:
                prompt = input("\nprompt> ")
            except EOFError:
                break
            if not prompt.strip():
                break
            run(prompt)


if __name__ == "__main__":
    main()

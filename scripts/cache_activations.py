#!/usr/bin/env python3
"""Pre-compute the frozen-prefix token activations of SlotKneeS (activation caching).

Output: an fp16 memmap of shape [N, S, G, n_tokens, C] indexed by CACHE ROW
(N = len(cache.uids); rows of studies that are not done stay zero), plus a
sidecar meta JSON at <out>.json recording the shape and the settings that MUST
match at train time.  scripts/train_slotknee.py reads the meta to open the
memmap and refuses a mismatch.

Fixed 2026-08-24 (audit): the original hardcoded S=6 and 257x384 tokens.  Those
are only right for a 6-slot cache at P=224 on ViT-S without register tokens
(257 = 16x16 patches + CLS): P=252 gives 325 tokens, zoom caches have S=8, and
reg4 backbones add 4 register tokens.  Everything is now probed from the cache
and the actual model.  The most dangerous mismatch is --trainable-blocks: the
cached tokens are the output of the first (depth - trainable_blocks) blocks, so
training on them with a different --trainable-blocks would silently skip blocks
in the middle of the encoder.
"""

import argparse
import json
import os
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

try:
    from tqdm import tqdm
except ImportError:  # tqdm is cosmetic
    def tqdm(it, **kw):
        return it

from src.slots import SlotCache
from src.slotknee import SlotKneeS, DEFAULT_BACKBONE


class ActivationCacheDataset(Dataset):
    def __init__(self, cache: SlotCache):
        self.cache = cache
        self.indices = [i for i, done in enumerate(cache.done) if done]

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        cache_row = self.indices[idx]
        x, mask = self.cache[cache_row]
        x_t = torch.from_numpy(np.array(x, copy=True))
        mask_t = torch.from_numpy(np.array(mask, copy=True))
        return x_t, mask_t, cache_row


def main():
    parser = argparse.ArgumentParser("Activation Caching")
    parser.add_argument("--cache", type=str, required=True)
    parser.add_argument("--out", type=str, required=True)
    parser.add_argument("--bs", type=int, default=4)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--trainable-blocks", type=int, default=4,
                        help="MUST equal the --trainable-blocks of the training run "
                             "that will consume this file")
    parser.add_argument("--backbone", default=DEFAULT_BACKBONE)
    parser.add_argument("--pretrained-path", default=None,
                        help="local weights file/dir (Kaggle runs offline)")
    parser.add_argument("--amp", default="auto", choices=["auto", "cpu"])
    args = parser.parse_args()

    if args.amp == "cpu":
        device = torch.device("cpu")
    else:
        device = torch.device(
            "cuda" if torch.cuda.is_available()
            else "mps" if torch.backends.mps.is_available() else "cpu")

    cache = SlotCache(args.cache, split="train")
    if len(cache.uids) == 0:
        print("Empty cache!")
        return

    # Build the model with the CACHE's geometry, not defaults: P/T set the token
    # grid and input channels, S/G only matter for the meta shape.
    model = SlotKneeS(P=cache.P, T=cache.T, n_slots=int(getattr(cache, "S", 6)),
                      max_groups=max(8, int(cache.G)),
                      trainable_blocks=args.trainable_blocks,
                      backbone=args.backbone,
                      pretrained_path=args.pretrained_path).to(device)
    model.eval()

    # Probe the real frozen-token geometry instead of hardcoding 257x384.
    with torch.no_grad():
        probe = model.encode_frozen(
            torch.zeros(1, int(cache.T), int(cache.P), int(cache.P), device=device))
    n_tok, emb = int(probe.shape[1]), int(probe.shape[2])
    del probe

    N = len(cache.uids)
    S, G = int(getattr(cache, "S", 6)), int(cache.G)
    out_shape = (N, S, G, n_tok, emb)
    est_gb = int(np.prod(out_shape)) * 2 / 1024 ** 3
    print(f"Caching activations to {args.out} using {device}: shape {out_shape} "
          f"fp16 (~{est_gb:.1f} GiB), frozen blocks = {model.n_frozen_blocks}/{model.depth}")

    dataset = ActivationCacheDataset(cache)
    loader = DataLoader(dataset, batch_size=args.bs, shuffle=False,
                        num_workers=args.workers)
    out_memmap = np.memmap(args.out, dtype=np.float16, mode="w+", shape=out_shape)

    autocast_ok = device.type == "cuda"
    with torch.no_grad():
        for x, mask, rows in tqdm(loader, desc="Caching"):
            x = x.to(device)
            B = x.shape[0]
            imgs = x.reshape(B * S * G, int(cache.T), int(cache.P), int(cache.P))
            if autocast_ok:
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    frozen = model.encode_frozen(imgs)
            else:
                frozen = model.encode_frozen(imgs)
            frozen = frozen.reshape(B, S, G, n_tok, emb).float().cpu().numpy().astype(np.float16)
            for i, row in enumerate(rows):
                out_memmap[int(row)] = frozen[i]

    out_memmap.flush()
    meta = {
        "shape": list(out_shape), "dtype": "float16",
        "P": int(cache.P), "T": int(cache.T), "S": S, "G": G,
        "n_tokens": n_tok, "embed_dim": emb,
        "backbone": str(args.backbone),
        "trainable_blocks": int(args.trainable_blocks),
        "n_frozen_blocks": int(model.n_frozen_blocks),
        "cache_dir": os.path.abspath(args.cache),
    }
    with open(args.out + ".json", "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Done! Meta written to {args.out}.json — train with the SAME "
          f"--trainable-blocks {args.trainable_blocks} and --backbone.")


if __name__ == "__main__":
    main()

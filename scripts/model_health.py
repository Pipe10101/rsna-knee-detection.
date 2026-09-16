#!/usr/bin/env python3
"""Is every component of a trained SlotKnee checkpoint actually DOING anything?

    python3 scripts/model_health.py models/ablate_full/reg4_s42/fold_0_best.pt \
        --cache cache/slots_P224_full/slots_P224 [--studies 24]

WHY THIS EXISTS.  On 2026-08-24 the per-finding attention — the architectural centrepiece —
was found to have collapsed to uniform (entropy 0.9994, learnable temperature drifted DOWN
from its 1.0 init, anchor weights varying by 0.0085).  It had been that way for the whole
campaign, and it explains why every capacity arm measured NULL: more capacity cannot help
when the aggregation discards the evidence.  Nothing in the training logs would have shown
it, because nobody printed the one number that mattered.

So: rather than wait for the next silent dead component, measure them all.  Each check
reports a number and a verdict, and the verdicts are deliberately blunt.
"""
import argparse
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

from src.slotknee import SlotKneeS
from src.slots import SlotCache


def verdict(ok, msg_ok, msg_bad):
    return f"\033[0m{msg_ok}" if ok else f"{msg_bad}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt")
    ap.add_argument("--cache", required=True)
    ap.add_argument("--studies", type=int, default=24)
    a = ap.parse_args()

    ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    hp = dict(ck["hparams"]); hp["pretrained"] = False
    m = SlotKneeS(**hp); m.load_state_dict(ck["state_dict"]); m.eval()
    cache = SlotCache(a.cache, split="train")
    names = list(getattr(cache, "slot_names", []))
    n = min(a.studies, len(cache.uids))
    X = torch.stack([torch.as_tensor(np.array(cache[i][0])) for i in range(n)])
    M = torch.stack([torch.as_tensor(np.array(cache[i][1])) for i in range(n)])

    print(f"checkpoint : {a.ckpt}")
    print(f"backbone   : {hp.get('backbone')}   tokens={hp.get('token_level')}  "
          f"mixer={hp.get('mixer_layers')}  mil={hp.get('mil_pool')}  seq_mix={hp.get('seq_mix')}  "
          f"cosine={hp.get('attn_cosine', False)}")
    print(f"studies    : {n}\n")

    # 1. ATTENTION — the one that was dead.
    rep = m.attention_report(X, M)
    ent, tau = rep["entropy"], rep["tau"].numpy()
    anchor = rep["anchor"]
    spread = float((anchor.max(1).values - anchor.min(1).values).mean()) if anchor is not None else float("nan")
    print("[attention]")
    print(f"  normalised entropy      {ent:.4f}   " +
          verdict(ent < 0.98, "selective", "COLLAPSED -> mean pooling (1.0 = uniform)"))
    print(f"  attn_tau                min {tau.min():.3f} max {tau.max():.3f}  (init {hp.get('attn_tau_init', 1.0)})   " +
          verdict(abs(tau.mean() - float(hp.get("attn_tau_init", 1.0))) > 0.1, "moved", "never moved from init"))
    print(f"  anchor weight spread    {spread:.4f}   " +
          verdict(spread > 0.02, "anchors differentiated", "all anchors weighted alike"))

    # 2. SLOT / GROUP EMBEDDINGS — do tokens carry identity at all?
    sd = ck["state_dict"]
    for key, label in (("slot_embed", "slot"), ("group_embed", "anchor")):
        if key in sd:
            e = sd[key].float()
            print(f"\n[{label} embedding]")
            print(f"  mean row norm           {e.norm(dim=-1).mean():.4f}   " +
                  verdict(float(e.norm(dim=-1).mean()) > 1e-3, "present", "~ZERO: identity not encoded"))
            if e.shape[0] > 1:
                en = torch.nn.functional.normalize(e.reshape(e.shape[0], -1), dim=-1)
                off = (en @ en.T)[~torch.eye(e.shape[0], dtype=bool)].mean()
                print(f"  mean pairwise cosine    {off:+.4f}   " +
                      verdict(abs(float(off)) < 0.9, "distinguishable", "near-identical rows"))

    # 3. MIXER — does the self-attention layer change the tokens it is given?
    if getattr(m, "mixer", None):
        pre, post = {}, {}
        h1 = m.mixer[0].register_forward_pre_hook(lambda mod, i: pre.__setitem__("t", i[0].detach()))
        h2 = m.mixer[-1].register_forward_hook(lambda mod, i, o: post.__setitem__("t", o.detach()))
        with torch.no_grad():
            m(X, M)
        h1.remove(); h2.remove()
        d = (post["t"] - pre["t"]).norm(dim=-1).mean() / pre["t"].norm(dim=-1).mean()
        print(f"\n[mixer]")
        print(f"  relative token change   {float(d):.4f}   " +
              verdict(float(d) > 0.05, "actively mixing", "near-IDENTITY: the mixer does nothing"))

    # 4. TOKEN DIVERSITY — is there anything for attention to select between?
    feats = {}
    hook = (m.mixer[0].register_forward_pre_hook(lambda mod, i: feats.__setitem__("t", i[0].detach()))
            if getattr(m, "mixer", None) else None)
    with torch.no_grad():
        m(X, M)
    if hook: hook.remove()
    if "t" in feats:
        t = feats["t"]
        S = len(names) or 6
        G = t.shape[1] // S
        tn = torch.nn.functional.normalize(t.reshape(t.shape[0], S, G, -1), dim=-1)
        sim = torch.einsum("bsgd,btgd->st", tn, tn) / (t.shape[0] * G)
        off = float(sim[~torch.eye(S, dtype=bool)].mean())
        print(f"\n[token diversity]")
        print(f"  cosine BETWEEN sequences {off:+.4f}   " +
              verdict(off < 0.9, "sequences are distinguishable", "sequences ENCODE THE SAME THING"))
        print("  -> if sequences are distinguishable but attention is uniform, the head is")
        print("     ignoring information that is demonstrably present.")

    # 5. AUX HEAD
    print(f"\n[aux slot head]  {'present' if hp.get('aux_slot_logits') else 'absent'}"
          + ("   (measured essential: removing it cost -0.064 pooled AUC)" if hp.get("aux_slot_logits") else ""))


if __name__ == "__main__":
    main()

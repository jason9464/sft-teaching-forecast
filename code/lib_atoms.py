#!/usr/bin/env python
"""Main run stage E utilities: atom stats, browser, logit lens (arm-aware).

Subcommands:
  stats  --tag T          per-atom mass/fires/doc-breadth/source/planted share
  browse --tag T --atoms "top:40" | "ids:1,2,3"   markdown atom browser
                          (top-firing chunk texts + lens tokens + stats)
Chunk text is re-derived from plan.pt tokens via rowmap (deterministic).
Lens: grad/rawgrad/unitnorm atoms -> teaching dir v = -W @ d -> unembed;
      act atoms -> d directly -> unembed; err atoms -> top coords ARE tokens.
"""
import argparse
import glob
import json
import os

import numpy as np
import torch

BASE = os.path.dirname(os.path.abspath(__file__))


def load_ctx(data_dir, dev):
    plan = torch.load(
        os.path.join(data_dir, "plan.pt"), map_location="cpu", weights_only=False
    )
    rm = torch.load(
        os.path.join(data_dir, "rowmap.pt"), map_location="cpu", weights_only=False
    )
    wh = torch.load(
        os.path.join(data_dir, "whitening.pt"), map_location="cpu", weights_only=False
    )
    row_doc = rm["row_doc"].long()
    # within-doc chunk index (storage rows are contiguous per doc)
    within = torch.zeros(len(row_doc), dtype=torch.long)
    prev, cnt = -1, 0
    rd = row_doc.tolist()
    for i, d in enumerate(rd):
        cnt = cnt + 1 if d == prev else 0
        within[i] = cnt
        prev = d
    return dict(plan=plan, rm=rm, wh=wh, row_doc=row_doc, within=within)


def chunk_text(ctx, tok, r, width=32, context=0):
    """Decode chunk r. If context>0, return (before, span, after) with up to
    `context` tokens of surrounding document text on each side."""
    d = int(ctx["row_doc"][r])
    w = int(ctx["within"][r])
    plan = ctx["plan"]
    s, e = int(plan["offsets"][d]), int(plan["offsets"][d + 1])
    ids = plan["ids_flat"][s:e].tolist()
    plen = int(plan["plens"][d])
    cpos = list(range(plen - 1, len(ids) - 1))
    seg = cpos[w * plan["chunk"] : (w + 1) * plan["chunk"]]
    if context <= 0:
        return tok.decode([ids[t] for t in seg])
    a, b = seg[0], seg[-1] + 1
    before = tok.decode(ids[max(0, a - context) : a])
    after = tok.decode(ids[b : b + context])
    return before, tok.decode(ids[a:b]), after


def encode_stats(sae_path, arm, data_dir, dev, topn_rows=12):
    from lib_sae import MatJumpSAE, load_all, make_transform  # noqa

    st = torch.load(sae_path, map_location="cpu", weights_only=False)
    tfn, fname = make_transform(arm, data_dir, dev)
    X = load_all(data_dir, fname, 2, dev)
    enc = st["enc"].float().to(dev)
    b_dec = st["b_dec"].to(dev)
    b_enc = st["b_enc"].to(dev)
    theta = st["theta"].to(dev)
    M = enc.shape[0]
    mass = torch.zeros(M, device=dev)
    fires = torch.zeros(M, device=dev)
    top_vals = torch.zeros(M, topn_rows, device=dev)
    top_rows = torch.zeros(M, topn_rows, dtype=torch.long, device=dev)
    bs = 16384
    for i in range(0, X.shape[0], bs):
        x = tfn(X[i : i + bs])
        z = (x - b_dec) @ enc.T + b_enc
        f = z * (z > theta)
        mass += f.sum(0)
        fires += (f > 0).float().sum(0)
        cat_v = torch.cat([top_vals, f.T], dim=1)
        cat_r = torch.cat(
            [top_rows, torch.arange(i, i + x.shape[0], device=dev).expand(M, -1)], dim=1
        )
        v, ix = cat_v.topk(topn_rows, dim=1)
        top_vals = v
        top_rows = torch.gather(cat_r, 1, ix)
    return st, dict(
        mass=mass.cpu(),
        fires=fires.cpu(),
        top_vals=top_vals.cpu(),
        top_rows=top_rows.cpu(),
    )


def lens_tokens(st, arm, ctx, tok, atom_ids, dev, topk=12):
    dec = st["dec"].float()
    if arm == "err":
        # coords are top-4096 token ids directly
        tid = ctx["plan"]["top_ids"]
        out = {}
        for a in atom_ids:
            v = dec[:, a]
            pro = v.topk(topk).indices
            out[a] = " ".join(repr(tok.decode([int(tid[i])])) for i in pro)
        return out
    snaps = glob.glob(
        os.environ["HF_HOME"]
        + "/hub/models--allenai--Olmo-3-1025-7B/snapshots/*/model.safetensors.index.json"
    )
    idx = json.load(open(snaps[0]))
    sd = os.path.dirname(snaps[0])
    from safetensors import safe_open

    T = {}
    for name in ("lm_head.weight", "model.norm.weight"):
        with safe_open(
            os.path.join(sd, idx["weight_map"][name]), framework="pt", device="cpu"
        ) as f:
            T[name] = f.get_tensor(name).float()
    if arm in ("grad", "rawgrad", "unitnorm"):
        W = ctx["wh"]["W"].float()
        V = -(W @ dec[:, atom_ids])
    else:  # act
        V = dec[:, atom_ids]
    V = V / V.norm(dim=0, keepdim=True)
    logits = (T["lm_head.weight"] * T["model.norm.weight"].unsqueeze(0)) @ V
    out = {}
    for j, a in enumerate(atom_ids):
        pro = logits[:, j].topk(topk).indices.tolist()
        out[a] = " ".join(repr(tok.decode([t])) for t in pro)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["browse"])
    ap.add_argument("--tag", required=True)
    ap.add_argument("--arm", required=True)
    ap.add_argument("--atoms", default="top:40")
    ap.add_argument("--data", default=os.path.join(BASE, "data"))
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    dev = "cuda:0"
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained("allenai/Olmo-3-7B-Think")
    ctx = load_ctx(args.data, dev)
    sae_path = os.path.join(args.data, "sae", f"{args.tag}.pt")
    cache = os.path.join(args.data, "sae", f"{args.tag}_stats.pt")
    if os.path.exists(cache):
        st = torch.load(sae_path, map_location="cpu", weights_only=False)
        stats = torch.load(cache, map_location="cpu", weights_only=False)
        print("using cached stats", flush=True)
    else:
        st, stats = encode_stats(sae_path, args.arm, args.data, dev)
    mass = stats["mass"]
    if args.atoms.startswith("top:"):
        atom_ids = mass.topk(int(args.atoms[4:])).indices.tolist()
    else:
        atom_ids = [int(x) for x in args.atoms.split(":")[1].split(",")]
    lens = lens_tokens(st, args.arm, ctx, tok, atom_ids, dev)
    rowp = ctx["rm"]["row_planted"]
    srcs = ctx["plan"]["sources"]
    md = [f"# atom browser — {args.tag} (arm={args.arm})", ""]
    tot = mass.sum()
    for rank, a in enumerate(atom_ids):
        rows = stats["top_rows"][a].tolist()
        vals = stats["top_vals"][a].tolist()
        docs_f = set()
        pl = 0
        for r, v in zip(rows, vals):
            if v > 0:
                docs_f.add(int(ctx["row_doc"][r]))
                pl += int(rowp[r])
        md.append(
            f"\n## #{rank+1} atom {a} — {100*mass[a]/tot:.3f}% mass, fires {int(stats['fires'][a])}, planted-top {pl}/12"
        )
        md.append(f"- lens: {lens[a]}")
        for r, v in zip(rows[:6], vals[:6]):
            if v <= 0:
                continue
            src = srcs[int(ctx["row_doc"][r])].split("/")[-1][:24]
            md.append(f"- [{v:.1f}|{src}] {chunk_text(ctx, tok, r)[:190]!r}")
    out = args.out or os.path.join(args.data, f"browser_{args.tag}.md")
    with open(out, "w") as f:
        f.write("\n".join(md))
    torch.save(stats, os.path.join(args.data, "sae", f"{args.tag}_stats.pt"))
    print("wrote", out)


if __name__ == "__main__":
    main()

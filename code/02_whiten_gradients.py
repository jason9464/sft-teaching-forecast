#!/usr/bin/env python
"""Main run stage C: preprocessing artifacts from collection output.

Per plan §3.2:
- merge per-part whitening stats -> per-layer token-Fisher eigendecomp;
  damping lambda = 0.1 x mean(eigenvalue); save W for the analysis layer
- global row map: chunk row -> doc index (+ planted flag, source, holdout split)
- per-arm transform params (computed on a deterministic 1M-row sample, applied
  on the fly at SAE training): grad = (x-mu)W -> 6xmedian cap -> /msn_scale;
  act = (x-mu) -> cap -> /scale; err = x -> cap -> /scale; rawgrad = (x-mu)
  -> cap -> /scale; unitnorm = whitened grad row-normalized (norms saved by
  trainer at encode time, not here)
- holdout: 5% of docs (deterministic), planted docs always in TRAIN
Outputs: data/whitening.pt, data/rowmap.pt, data/transforms.pt
"""
import argparse
import os

import numpy as np
import torch

BASE = os.path.dirname(os.path.abspath(__file__))
ANALYSIS_LAYER = 15
DAMP_C = 0.1
CAP_MULT = 6.0
SAMPLE_ROWS = 1_000_000


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.path.join(BASE, "data"))
    ap.add_argument("--parts", type=int, default=2)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()
    dev = args.device

    plan = torch.load(
        os.path.join(args.data, "plan.pt"), map_location="cpu", weights_only=False
    )
    LAYERS = plan["layers"]
    parts = [os.path.join(args.data, f"part{k}") for k in range(args.parts)]

    # ---- whitening ----
    stats = [
        torch.load(
            os.path.join(p, "whiten_stats.pt"), map_location="cpu", weights_only=False
        )
        for p in parts
    ]
    n = sum(s["count"] for s in stats)
    wh = dict(count=n, evals={}, act_mean={}, damp_c=DAMP_C)
    for L in LAYERS:
        gs = sum(s["grad_sum"][L] for s in stats).double()
        go = sum(s["grad_outer"][L].double() for s in stats)
        mu = (gs / n).float()
        cov = (go / n - torch.outer(gs / n, gs / n)).float()
        ev, U = torch.linalg.eigh(cov.to(dev))
        wh["evals"][L] = ev.cpu()
        wh["act_mean"][L] = (sum(s["act_sum"][L] for s in stats) / n).float()
        if L == ANALYSIS_LAYER:
            lam = DAMP_C * ev.mean()
            W = (U @ torch.diag((ev + lam).rsqrt()) @ U.T).cpu()
            wh["grad_mu"] = mu
            wh["W"] = W
        print(
            f"layer {L}: eig[min/med/max] {ev.min():.3e}/{ev.median():.3e}/{ev.max():.3e}",
            flush=True,
        )
    torch.save(wh, os.path.join(args.data, "whitening.pt"))

    # ---- row map ----
    row_doc, part_rows = [], []
    for p in parts:
        m = torch.load(
            os.path.join(p, "docmeta.pt"), map_location="cpu", weights_only=False
        )
        rd = torch.repeat_interleave(m["doc_order"], m["doc_nchunks"])
        row_doc.append(rd)
        part_rows.append(len(rd))
    row_doc = torch.cat(row_doc)
    planted_doc = plan["planted"]
    g = torch.Generator().manual_seed(0)
    n_docs = len(plan["sources"])
    perm = torch.randperm(n_docs, generator=g)
    ho_docs = torch.zeros(n_docs, dtype=torch.bool)
    ho_docs[perm[: n_docs // 20]] = True
    ho_docs &= ~planted_doc  # planted always in train
    rowmap = dict(
        row_doc=row_doc.int(),
        part_rows=part_rows,
        doc_holdout=ho_docs,
        doc_planted=planted_doc,
        row_holdout=ho_docs[row_doc.long()],
        row_planted=planted_doc[row_doc.long()],
    )
    torch.save(rowmap, os.path.join(args.data, "rowmap.pt"))
    print(
        f"rows {len(row_doc)}  holdout rows {int(rowmap['row_holdout'].sum())} "
        f"planted rows {int(rowmap['row_planted'].sum())}",
        flush=True,
    )

    # ---- per-arm transform stats on deterministic sample ----
    total_rows = len(row_doc)
    g2 = torch.Generator().manual_seed(1)
    samp = torch.randperm(total_rows, generator=g2)[:SAMPLE_ROWS].sort().values

    def load_rows(fname, idx):
        outs, off = [], 0
        for k, p in enumerate(parts):
            arr = np.load(os.path.join(p, fname), mmap_mode="r")
            sel = idx[(idx >= off) & (idx < off + arr.shape[0])] - off
            outs.append(torch.from_numpy(np.asarray(arr[sel.numpy()])).float())
            off += arr.shape[0]
        return torch.cat(outs)

    Wg = wh["W"].to(dev)
    mu_g = wh["grad_mu"].to(dev)
    mu_a = wh["act_mean"][ANALYSIS_LAYER].to(dev)
    tf = {}
    for arm, fname in [
        ("grad", f"grad_l{ANALYSIS_LAYER}.npy"),
        ("act", f"act_l{ANALYSIS_LAYER}.npy"),
        ("err", "err.npy"),
        ("rawgrad", f"grad_l{ANALYSIS_LAYER}.npy"),
    ]:
        X = load_rows(fname, samp).to(dev)
        if arm == "grad":
            Xt = (X - mu_g) @ Wg
        elif arm == "act":
            Xt = X - mu_a
        elif arm == "rawgrad":
            Xt = X - mu_g
        else:
            Xt = X
        nrm = Xt.norm(dim=1)
        med = nrm.median()
        cap = CAP_MULT * med
        sc = torch.clamp(cap / nrm, max=1.0).unsqueeze(1)
        msn = ((Xt * sc).pow(2).sum(1).mean()).sqrt()
        tf[arm] = dict(
            cap=float(cap),
            scale=float(msn),
            median_norm=float(med),
            cap_frac=float((nrm > cap).float().mean()),
        )
        print(arm, tf[arm], flush=True)
        del X, Xt
        torch.cuda.empty_cache()
    torch.save(
        dict(arms=tf, analysis_layer=ANALYSIS_LAYER),
        os.path.join(args.data, "transforms.pt"),
    )
    print("wrote whitening.pt rowmap.pt transforms.pt", flush=True)


if __name__ == "__main__":
    main()

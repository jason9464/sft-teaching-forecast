#!/usr/bin/env python
"""Token-level SAE training with streaming block-shuffled loader (data >> GPU RAM).

Reuses lib_sae's MatJumpSAE/evaluate math. Data: data/tok/part{0,1}/grad_l{L}.npy
(fp16 memmap). Loader: background thread reads contiguous blocks (block_rows),
GPU-side shuffle within block, random block order -> disk-sequential, GPU-busy.
Whitening: reuse data/whitening.pt (token-level covariance already) for the layer.
Holdout: doc-split via tokmeta (same 5% doc set as chunk rowmap).
"""
import argparse, json, os, threading, queue, time
import numpy as np, torch
BASE = os.path.dirname(os.path.abspath(__file__))
import lib_sae as M


class BlockLoader:
    def __init__(self, arrs, row_ok, block_rows, batch, dev, seed=0, depth=4):
        self.arrs, self.row_ok, self.block_rows, self.batch, self.dev = arrs, row_ok, block_rows, batch, dev
        self.q = queue.Queue(maxsize=depth); self.rng = np.random.default_rng(seed)
        self.blocks = []
        off = 0
        for a in arrs:
            for s in range(0, a.shape[0], block_rows): self.blocks.append((a, s, min(s + block_rows, a.shape[0]), off))
            off += a.shape[0]
        threading.Thread(target=self._fill, daemon=True).start()

    def _fill(self):
        while True:
            for bi in self.rng.permutation(len(self.blocks)):
                a, s, e, off = self.blocks[bi]
                x = torch.from_numpy(np.asarray(a[s:e]))          # sequential read, fp16 CPU
                ok = self.row_ok[off + s: off + e]
                x = x[ok]
                if len(x) < self.batch: continue
                self.q.put(x.pin_memory())

    def batches(self):
        while True:
            x = self.q.get().to(self.dev, non_blocking=True)
            perm = torch.randperm(len(x), device=self.dev)
            for i in range(0, len(x) - self.batch + 1, self.batch):
                yield x[perm[i:i + self.batch]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layer", type=int, default=15); ap.add_argument("--width", type=int, default=32768)
    ap.add_argument("--groups", default=""); ap.add_argument("--l0", type=float, default=32); ap.add_argument("--lam", type=float, required=True)
    ap.add_argument("--steps", type=int, default=25000); ap.add_argument("--batch", type=int, default=4096); ap.add_argument("--warmup", type=int, default=2000)
    ap.add_argument("--group-weight", default="equal"); ap.add_argument("--block-rows", type=int, default=65536)
    ap.add_argument("--tag", required=True); ap.add_argument("--data", default=os.path.join(BASE, "data")); ap.add_argument("--parts", default="0,1"); ap.add_argument("--tok-dir", default="tok")
    ap.add_argument("--eval-rows", type=int, default=262144); ap.add_argument("--topk", type=int, default=0); ap.add_argument("--lr-theta", type=float, default=7e-4)
    args = ap.parse_args(); dev = "cuda:0"; torch.manual_seed(0)
    torch.backends.cuda.matmul.allow_tf32 = True; M.EPS = 0.02
    L = args.layer
    wh = torch.load(os.path.join(args.data, "whitening.pt"), map_location="cpu", weights_only=False)
    ev = wh["evals"][L]; 
    # rebuild W for this layer (whitening.pt stores W only for analysis layer)
    if L == 15 and "W" in wh: W = wh["W"].to(dev); mu = wh["grad_mu"].to(dev)
    else:
        raise SystemExit("W for layer != 15 not stored; extend 02_whiten_gradients.py to save per-layer W")
    parts = [int(p) for p in args.parts.split(",")]
    arrs = [np.load(os.path.join(args.data, args.tok_dir, f"part{p}", f"grad_l{L}.npy"), mmap_mode="r") for p in parts]
    metas = [torch.load(os.path.join(args.data, args.tok_dir, f"part{p}", "tokmeta.pt"), map_location="cpu", weights_only=False) for p in parts]
    rm = torch.load(os.path.join(args.data, "rowmap.pt"), map_location="cpu", weights_only=False)
    doc_ho = rm["doc_holdout"]; doc_pl = rm["doc_planted"]
    row_doc = torch.cat([torch.repeat_interleave(m["doc_order"], m["doc_ntok"]) for m in metas]).long()
    row_ho = doc_ho[row_doc]; row_pl = doc_pl[row_doc]
    train_ok = ~row_ho
    total = len(row_doc); print(f"token rows {total}, train {int(train_ok.sum())}, holdout {int(row_ho.sum())}", flush=True)
    # transform stats on a 200K-row sample (cap, scale) — token-level
    g = torch.Generator().manual_seed(1)
    samp = torch.randperm(total, generator=g)[:200_000].sort().values
    def read_rows(idx):
        out, off = [], 0
        for a in arrs:
            sel = idx[(idx >= off) & (idx < off + a.shape[0])] - off
            out.append(torch.from_numpy(np.asarray(a[sel.numpy()])).float()); off += a.shape[0]
        return torch.cat(out)
    xs = (read_rows(samp).to(dev) - mu) @ W
    n = xs.norm(dim=1); cap = 6 * n.median(); sc = torch.clamp(cap / n, max=1.0).unsqueeze(1); scale = (xs * sc).pow(2).sum(1).mean().sqrt()
    print(f"token transform: median norm {n.median():.3f} cap {cap:.3f} scale {scale:.3f}", flush=True)
    def tfn(x):
        x = x.float() - mu; x = x @ W; nn = x.norm(dim=1, keepdim=True); return x * torch.clamp(cap / nn, max=1.0) / scale
    ho_idx = row_ho.nonzero(as_tuple=True)[0]; ho_eval = ho_idx[torch.randperm(len(ho_idx), generator=torch.Generator().manual_seed(3))[: args.eval_rows]].sort().values
    tr_idx = train_ok.nonzero(as_tuple=True)[0]; tr_eval = tr_idx[torch.randperm(len(tr_idx), generator=torch.Generator().manual_seed(4))[: args.eval_rows]].sort().values
    Xho = read_rows(ho_eval).to(dev); Xtr_eval = read_rows(tr_eval).to(dev)
    Mw = args.width; groups = [int(x) for x in args.groups.split(",")] if args.groups else [Mw // 4, Mw // 2, Mw]
    sae = M.MatJumpSAE(4096, Mw, groups, tfn(Xtr_eval[:65536]).mean(0)).to(dev); sae.topk = args.topk
    gw = torch.tensor([1.0] * len(groups), device=dev) / len(groups)
    r0 = M.evaluate(sae, Xho, tfn, torch.arange(len(Xho), device=dev))
    opt = torch.optim.Adam([{"params": [sae.enc, sae.dec, sae.b_enc, sae.b_dec], "lr": 7e-5}, {"params": [sae.theta], "lr": args.lr_theta}], betas=(0.0, 0.999))
    loader = BlockLoader(arrs, train_ok, args.block_rows, args.batch, dev)
    it = loader.batches(); t0 = time.time()
    for step in range(args.steps):
        x = tfn(next(it))
        f, s = sae.encode(x)
        recon = sum(w * ((r - x) ** 2).sum(1).mean() for w, r in zip(gw, sae.decode_prefixes(f)))
        l0b = s.sum(1).mean(); lam_t = args.lam * min(1.0, step / args.warmup)
        loss = recon + (lam_t * (2.0 / args.l0) * (l0b - args.l0) ** 2 if args.topk == 0 else 0.0)
        opt.zero_grad(); loss.backward(); sae.project_dec_grad(); opt.step(); sae.renorm()
        if step % 1000 == 0 or step == args.steps - 1:
            print(f"step {step} recon {recon.item():.4f} L0 {l0b.item():.1f} {(time.time()-t0)/60:.1f}m", flush=True)
    evh = M.evaluate(sae, Xho, tfn, torch.arange(len(Xho), device=dev)); evt = M.evaluate(sae, Xtr_eval, tfn, torch.arange(len(Xtr_eval), device=dev))
    print("holdout:", {k: (round(v, 4) if isinstance(v, float) else v) for k, v in evh.items()}); print("train:", {k: round(v, 4) for k, v in evt.items() if k.startswith("fvu")}); print("random:", {k: round(v, 4) for k, v in r0.items() if k.startswith("fvu")})
    os.makedirs(os.path.join(args.data, "sae"), exist_ok=True)
    torch.save(dict(enc=sae.enc.data.half().cpu(), dec=sae.dec.data.half().cpu(), b_enc=sae.b_enc.data.cpu(), b_dec=sae.b_dec.data.cpu(), theta=sae.theta.data.cpu(), groups=groups, cfg=vars(args), cap=float(cap), scale=float(scale), eval_hold=evh, eval_train=evt, eval_random=r0), os.path.join(args.data, "sae", f"{args.tag}.pt"))
    json.dump(dict(cfg={k: v for k, v in vars(args).items() if k != "data"}, hold=evh, train={k: v for k, v in evt.items() if k.startswith("fvu")}, random={k: v for k, v in r0.items() if k.startswith("fvu")}, minutes=(time.time() - t0) / 60), open(os.path.join(args.data, "sae", f"{args.tag}.json"), "w"), indent=1)


if __name__ == "__main__":
    main()

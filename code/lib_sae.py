#!/usr/bin/env python
"""Main run stage D: Matryoshka JumpReLU SAE trainer.

Per plan §3.3 (reference: saprmarks/dictionary_learning, MIT; GS2 quadratic
target-L0; S14 smoke adaptations):
- width M (default 32768), matryoshka groups = [M/16, M/8, M/4, M/2, M],
  full-prefix equal-weight reconstruction loss
- JumpReLU: direct per-feature theta (clamp >= 1e-6, init 0.005), rectangle-STE
  bandwidth eps = 0.02; sparsity = lambda * (2/L0*) * (batch-mean L0 - L0*)^2
  with 2K-step warmup; L0* = 32
- Adam(beta1=0, beta2=0.999), lr 7e-5 (enc/dec/biases) + 7e-4 (theta),
  batch 4096; decoder unit-norm columns with parallel-gradient projection
- data: raw fp16 chunk memmaps on GPU, arm transform applied per batch
  (arm in {grad, act, err, rawgrad, unitnorm}); doc-split holdout from rowmap
- eval: holdout FVU per prefix + random-dict baseline + dead atoms per group

--pilot: smoke on spike pilot5k chunks (L15) with small width.
"""
import argparse
import json
import os
import time

import numpy as np
import torch

BASE = os.path.dirname(os.path.abspath(__file__))
EPS = 0.02  # overridden by --eps


class JumpF(torch.autograd.Function):
    @staticmethod
    def forward(ctx, z, theta):
        ctx.save_for_backward(z, theta)
        return z * (z > theta)

    @staticmethod
    def backward(ctx, g):
        z, theta = ctx.saved_tensors
        rect = ((z - theta).abs() < 0.5 * EPS).float()
        return (z > theta).float() * g, (-(theta / EPS) * rect * g).sum(0)


class StepF(torch.autograd.Function):
    @staticmethod
    def forward(ctx, z, theta):
        ctx.save_for_backward(z, theta)
        return (z > theta).float()

    @staticmethod
    def backward(ctx, g):
        z, theta = ctx.saved_tensors
        rect = ((z - theta).abs() < 0.5 * EPS).float()
        return torch.zeros_like(z), (-(1.0 / EPS) * rect * g).sum(0)


class MatJumpSAE(torch.nn.Module):
    def __init__(self, d, m, groups, data_mean):
        super().__init__()
        w = torch.randn(m, d) / d**0.5
        self.enc = torch.nn.Parameter(w.clone())
        self.dec = torch.nn.Parameter((w / w.norm(dim=1, keepdim=True)).T.clone())
        self.b_enc = torch.nn.Parameter(torch.zeros(m))
        self.b_dec = torch.nn.Parameter(data_mean.clone())
        self.theta = torch.nn.Parameter(torch.full((m,), 0.005))
        self.groups = groups

    topk = 0  # >0 => TopK control instead of JumpReLU
    relu_pre = False

    def encode(self, x):
        z = (x - self.b_dec) @ self.enc.T + self.b_enc
        if self.relu_pre:
            z = torch.relu(z)
        if self.topk > 0:
            tv, ti = z.topk(self.topk, dim=1)
            f = torch.zeros_like(z).scatter_(1, ti, torch.relu(tv))
            return f, (f > 0).float()
        return JumpF.apply(z, self.theta), StepF.apply(z, self.theta)

    def decode_prefix(self, f, k):
        return f[:, :k] @ self.dec[:, :k].T + self.b_dec

    def decode_prefixes(self, f):
        # incremental segment matmuls: identical math, ~1.94x -> 1x decode FLOPs
        outs, acc, prev = [], self.b_dec, 0
        for k in self.groups:
            acc = acc + f[:, prev:k] @ self.dec[:, prev:k].T
            outs.append(acc)
            prev = k
        return outs

    @torch.no_grad()
    def project_dec_grad(self):
        if self.dec.grad is not None:
            par = (self.dec.grad * self.dec.data).sum(0, keepdim=True)
            self.dec.grad -= par * self.dec.data

    @torch.no_grad()
    def renorm(self):
        self.dec.data /= self.dec.data.norm(dim=0, keepdim=True).clamp_min(1e-8)
        self.theta.data.clamp_(min=1e-6)


def make_transform(arm, data_dir, dev):
    wh = torch.load(
        os.path.join(data_dir, "whitening.pt"), map_location="cpu", weights_only=False
    )
    tfs = torch.load(
        os.path.join(data_dir, "transforms.pt"), map_location="cpu", weights_only=False
    )
    L = tfs["analysis_layer"]
    base = "grad" if arm == "unitnorm" else arm
    t = tfs["arms"][base]
    W = wh["W"].to(dev) if base in ("grad",) else None
    mu = {
        "grad": wh["grad_mu"],
        "rawgrad": wh["grad_mu"],
        "act": wh["act_mean"][L],
        "err": torch.zeros(4096),
        "unitnorm": wh["grad_mu"],
    }[arm].to(dev)

    def fn(x):
        x = x.float() - mu
        if W is not None:
            x = x @ W
        if arm == "unitnorm":
            return x / x.norm(dim=1, keepdim=True).clamp_min(1e-8)
        n = x.norm(dim=1, keepdim=True)
        x = x * torch.clamp(t["cap"] / n, max=1.0)
        return x / t["scale"]

    fname = {
        "grad": f"grad_l{L}.npy",
        "rawgrad": f"grad_l{L}.npy",
        "unitnorm": f"grad_l{L}.npy",
        "act": f"act_l{L}.npy",
        "err": "err.npy",
    }[arm]
    return fn, fname


def load_all(data_dir, fname, parts, dev):
    mats = []
    for k in range(parts):
        arr = np.load(os.path.join(data_dir, f"part{k}", fname), mmap_mode="r")
        t = torch.empty(arr.shape, dtype=torch.float16, device=dev)
        step = 1 << 18
        for i in range(0, arr.shape[0], step):
            t[i : i + step] = torch.from_numpy(np.asarray(arr[i : i + step])).to(dev)
        mats.append(t)
    return torch.cat(mats)


@torch.no_grad()
def evaluate(sae, X, tfn, idx, bs=16384):
    groups = sae.groups
    res = {k: 0.0 for k in groups}
    tot = 0.0
    fired = torch.zeros(sae.enc.shape[0], device=X.device)
    l0s = 0.0
    for i in range(0, len(idx), bs):
        x = tfn(X[idx[i : i + bs]])
        f, s = sae.encode(x)
        fired += (f > 0).float().sum(0)
        l0s += s.sum().item()
        xc = x - x.mean(0)
        tot += (xc**2).sum().item()
        for k, r in zip(groups, sae.decode_prefixes(f)):
            res[k] += ((r - x) ** 2).sum().item()
    out = {f"fvu_{k}": res[k] / tot for k in groups}
    out["l0"] = l0s / len(idx)
    gb = [0] + list(groups)
    out["dead_per_group"] = {
        f"{gb[i]}-{gb[i+1]}": int((fired[gb[i] : gb[i + 1]] == 0).sum())
        for i in range(len(groups))
    }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--arm", required=True, choices=["grad", "act", "err", "rawgrad", "unitnorm"]
    )
    ap.add_argument("--width", type=int, default=32768)
    ap.add_argument(
        "--groups", type=str, default="", help="comma ints; default M/16..M"
    )
    ap.add_argument("--l0", type=float, default=32)
    ap.add_argument("--lam", type=float, required=True)
    ap.add_argument("--steps", type=int, default=25000)
    ap.add_argument("--batch", type=int, default=4096)
    ap.add_argument("--warmup", type=int, default=2000)
    ap.add_argument("--eps", type=float, default=0.02, help="JumpReLU STE bandwidth")
    ap.add_argument("--lr", type=float, default=7e-5)
    ap.add_argument("--lr-theta", type=float, default=7e-4)
    ap.add_argument("--theta-init", type=float, default=0.005)
    ap.add_argument("--group-weight", default="equal", choices=["equal", "geometric", "last-heavy", "outer75", "full-only"],
                    help="matryoshka prefix loss weighting")
    ap.add_argument("--topk", type=int, default=0, help=">0: TopK control (no jumprelu/L0 penalty)")
    ap.add_argument("--l0-anneal-from", type=float, default=0.0,
                    help=">0: anneal target L0 from this value down to --l0 over --l0-anneal-steps")
    ap.add_argument("--l0-anneal-steps", type=int, default=15000)
    ap.add_argument("--lr-sched", default="none", choices=["none", "warmcos"],
                    help="warmcos: linear warmup 1K steps from 0.1x + cosine decay to 0.1x")
    ap.add_argument("--epoch-sampling", action="store_true",
                    help="sample without replacement within epochs (permutation) instead of randint")
    ap.add_argument("--relu-pre", action="store_true",
                    help="apply ReLU to pre-activations before JumpReLU/Step (JumpReLU App. J hygiene)")
    ap.add_argument("--benc-init-density", type=float, default=0.0,
                    help=">0: init b_enc so each atom fires on ~this fraction of a data sample (Anthropic 2025)")
    ap.add_argument("--dec-refit", type=int, default=0,
                    help=">0: after training, freeze encoder and refit decoder+b_dec for N steps (Wright&Sharkey)")
    ap.add_argument("--aux-dead", type=float, default=0.0,
                    help=">0: auxk-style dead-latent revival loss coefficient")
    ap.add_argument(
        "--frac",
        type=float,
        default=1.0,
        help="train-row subsample (lambda sweep: 0.1)",
    )
    ap.add_argument("--tag", required=True)
    ap.add_argument("--data", default=os.path.join(BASE, "data"))
    ap.add_argument("--parts", type=int, default=2)
    ap.add_argument("--pilot", action="store_true")
    args = ap.parse_args()
    dev = "cuda:0"
    torch.manual_seed(0)
    global EPS
    EPS = args.eps
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    if args.pilot:
        sp = "path/to/pilot5k"
        d = torch.load(
            os.path.join(sp, "pilot_collect.pt"), map_location="cpu", weights_only=False
        )
        c = torch.load(
            os.path.join(sp, "chunk_means.pt"), map_location="cpu", weights_only=False
        )
        Xt = d["tok_delta"][15].float()
        mu = Xt.mean(0)
        cov = ((Xt - mu).T @ (Xt - mu)) / (len(Xt) - 1)
        ev, U = torch.linalg.eigh(cov.to(dev))
        lam_w = 0.1 * ev.mean()
        W = U @ torch.diag((ev + lam_w).rsqrt()) @ U.T
        X = c["chunk_mean"][15].half().to(dev)
        muD, WD = mu.to(dev), W

        samp = torch.randperm(len(X))[:200_000]
        xs = (X[samp.to(dev)].float() - muD) @ WD
        capv = 6 * xs.norm(dim=1).median()
        sc = (
            (xs * torch.clamp(capv / xs.norm(dim=1, keepdim=True), max=1.0))
            .pow(2)
            .sum(1)
            .mean()
            .sqrt()
        )

        def tfn(x):
            x = x.float() - muD
            x = x @ WD
            n = x.norm(dim=1, keepdim=True)
            return x * torch.clamp(capv / n, max=1.0) / sc

        doc_of = c["doc_of_chunk"]
        g = torch.Generator().manual_seed(0)
        docs = torch.unique(doc_of)
        hod = set(
            docs[torch.randperm(len(docs), generator=g)][: len(docs) // 20].tolist()
        )
        ho = torch.tensor([dd in hod for dd in doc_of.tolist()])
    else:
        tfn, fname = make_transform(args.arm, args.data, dev)
        X = load_all(args.data, fname, args.parts, dev)
        rm = torch.load(
            os.path.join(args.data, "rowmap.pt"), map_location="cpu", weights_only=False
        )
        ho = rm["row_holdout"]
    tr_idx = (~ho).nonzero(as_tuple=True)[0]
    ho_idx = ho.nonzero(as_tuple=True)[0]
    if args.frac < 1.0:
        g = torch.Generator().manual_seed(2)
        tr_idx = tr_idx[
            torch.randperm(len(tr_idx), generator=g)[: int(len(tr_idx) * args.frac)]
        ]
    ho_eval = ho_idx[
        torch.randperm(len(ho_idx), generator=torch.Generator().manual_seed(3))[:262144]
    ]
    # train-side eval subsample must be random, not the (length-sorted) head
    tr_eval = tr_idx[
        torch.randperm(len(tr_idx), generator=torch.Generator().manual_seed(4))[:262144]
    ]
    tr_idx, ho_eval, tr_eval = tr_idx.to(dev), ho_eval.to(dev), tr_eval.to(dev)
    print(
        f"arm={args.arm} rows: train {len(tr_idx)} holdout-eval {len(ho_eval)}",
        flush=True,
    )

    M = args.width
    groups = (
        [int(x) for x in args.groups.split(",")]
        if args.groups
        else [M // 16, M // 8, M // 4, M // 2, M]
    )
    mean_est = tfn(X[tr_eval[:65536]]).mean(0)
    sae = MatJumpSAE(4096, M, groups, mean_est).to(dev)
    sae.theta.data.fill_(args.theta_init)
    if args.benc_init_density > 0:
        with torch.no_grad():
            xs = tfn(X[tr_eval[:32768]])
            z0 = (xs - sae.b_dec) @ sae.enc.T  # [n, M]
            q = torch.quantile(z0, 1 - args.benc_init_density, dim=0)  # per-atom pre-act quantile
            sae.b_enc.data = (sae.theta.data - q).clamp(-1.0, 1.0)  # shift so ~density fraction exceeds theta
    sae.topk = args.topk
    sae.relu_pre = args.relu_pre
    if args.group_weight == "equal":
        gw = [1.0] * len(groups)
    elif args.group_weight == "geometric":
        gw = [0.5 ** (len(groups) - 1 - i) for i in range(len(groups))]
    elif args.group_weight == "last-heavy":  # full width gets half the weight
        gw = [0.5 / (len(groups) - 1)] * (len(groups) - 1) + [0.5]
    elif args.group_weight == "outer75":  # full width 0.75, rest share 0.25
        gw = [0.25 / (len(groups) - 1)] * (len(groups) - 1) + [0.75]
    else:  # full-only: plain (non-matryoshka) SAE loss, groups kept for eval only
        gw = [0.0] * (len(groups) - 1) + [1.0]
    gw = torch.tensor(gw, device=dev)
    gw = gw / gw.sum()
    dead_ctr = torch.zeros(M, device=dev)
    r0 = evaluate(sae, X, tfn, ho_eval)
    print(
        "random-dict:",
        {k: round(v, 4) for k, v in r0.items() if k.startswith("fvu")},
        flush=True,
    )

    opt = torch.optim.Adam(
        [
            {"params": [sae.enc, sae.dec, sae.b_enc, sae.b_dec], "lr": args.lr},
            {"params": [sae.theta], "lr": args.lr_theta},
        ],
        betas=(0.0, 0.999),
    )

    t0 = time.time()
    base_lrs = [g["lr"] for g in opt.param_groups]
    perm, pptr = None, 0
    for step in range(args.steps):
        if args.lr_sched == "warmcos":
            if step < 1000:
                fac = 0.1 + 0.9 * step / 1000
            else:
                import math
                fac = 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * (step - 1000) / max(1, args.steps - 1000)))
            for g, b in zip(opt.param_groups, base_lrs):
                g["lr"] = b * fac
        if args.epoch_sampling:
            if perm is None or pptr + args.batch > len(perm):
                perm = tr_idx[torch.randperm(len(tr_idx), device=dev)]
                pptr = 0
            bidx = perm[pptr:pptr + args.batch]
            pptr += args.batch
        else:
            bidx = tr_idx[torch.randint(0, len(tr_idx), (args.batch,), device=dev)]
        x = tfn(X[bidx])
        l0_target = args.l0
        if args.l0_anneal_from > 0:
            frac = min(1.0, step / args.l0_anneal_steps)
            l0_target = args.l0_anneal_from * (1 - frac) + args.l0 * frac
        f, s = sae.encode(x)
        recons = sae.decode_prefixes(f)
        recon = sum(
            w * ((r - x) ** 2).sum(1).mean() for w, r in zip(gw, recons)
        )
        l0b = s.sum(1).mean()
        lam_t = args.lam * min(1.0, step / args.warmup)
        loss = recon
        if args.topk == 0:
            loss = loss + lam_t * (2.0 / l0_target) * (l0b - l0_target) ** 2
        if args.aux_dead > 0:
            # auxk-style: dead atoms (no fire in last 2K steps) reconstruct residual
            dead_ctr = torch.where(s.sum(0) > 0, torch.zeros_like(dead_ctr), dead_ctr + 1)
            dead = dead_ctr > 2000
            if dead.any() and step > 3000:
                z_pre = (x - sae.b_dec) @ sae.enc.T + sae.b_enc
                zd = torch.relu(z_pre) * dead.float()
                kk = min(int(dead.sum().item()), 512)
                tv, ti = zd.topk(kk, dim=1)
                zd = torch.zeros_like(zd).scatter_(1, ti, tv)
                resid = (x - recons[-1]).detach()
                aux = ((zd @ sae.dec.T - resid) ** 2).sum(1).mean()
                loss = loss + args.aux_dead * aux
        opt.zero_grad()
        loss.backward()
        sae.project_dec_grad()
        opt.step()
        sae.renorm()
        if step % 1000 == 0 or step == args.steps - 1:
            print(
                f"step {step} recon {recon.item():.4f} L0 {l0b.item():.1f} "
                f"theta[med] {sae.theta.median().item():.4f} {(time.time()-t0)/60:.1f}m",
                flush=True,
            )

    if args.dec_refit > 0:
        opt2 = torch.optim.Adam([sae.dec, sae.b_dec], lr=args.lr, betas=(0.0, 0.999))
        for step in range(args.dec_refit):
            bidx = tr_idx[torch.randint(0, len(tr_idx), (args.batch,), device=dev)]
            x = tfn(X[bidx])
            with torch.no_grad():
                f, _ = sae.encode(x)
            recons = sae.decode_prefixes(f)
            loss = sum(w * ((r - x) ** 2).sum(1).mean() for w, r in zip(gw, recons))
            opt2.zero_grad(); loss.backward(); sae.project_dec_grad(); opt2.step(); sae.renorm()
        print(f"dec-refit done ({args.dec_refit} steps)", flush=True)
    ev = evaluate(sae, X, tfn, ho_eval)
    ev_tr = evaluate(sae, X, tfn, tr_eval)
    print(
        "holdout:",
        {k: (round(v, 4) if isinstance(v, float) else v) for k, v in ev.items()},
        flush=True,
    )
    print(
        "train  :",
        {k: round(v, 4) for k, v in ev_tr.items() if k.startswith("fvu")},
        flush=True,
    )
    outdir = os.path.join(args.data, "sae")
    os.makedirs(outdir, exist_ok=True)
    torch.save(
        dict(
            enc=sae.enc.data.half().cpu(),
            dec=sae.dec.data.half().cpu(),
            b_enc=sae.b_enc.data.cpu(),
            b_dec=sae.b_dec.data.cpu(),
            theta=sae.theta.data.cpu(),
            groups=groups,
            cfg=vars(args),
            eval_hold=ev,
            eval_train=ev_tr,
            eval_random=r0,
        ),
        os.path.join(outdir, f"{args.tag}.pt"),
    )
    with open(os.path.join(outdir, f"{args.tag}.json"), "w") as fj:
        json.dump(
            dict(
                cfg={k: v for k, v in vars(args).items() if k != "data"},
                hold=ev,
                train={k: v for k, v in ev_tr.items() if k.startswith("fvu")},
                random={k: v for k, v in r0.items() if k.startswith("fvu")},
                minutes=(time.time() - t0) / 60,
            ),
            fj,
            indent=1,
        )
    print(f"saved sae/{args.tag}.pt", flush=True)


if __name__ == "__main__":
    main()

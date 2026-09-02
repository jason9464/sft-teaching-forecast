#!/usr/bin/env python
"""Forecast-ledger evidence for RQ2 (chunk unit, per arm).

evidence: for one SAE arm, rank atoms by mass on CLEAN TRAIN rows (non-planted, non-holdout),
  flag planted-dominated atoms, and for the top-N collect: mass share, fires, doc breadth,
  top-doc concentration, source histogram, holdout firing count (channel-A measurability),
  lens top tokens (grad: -W@dec through unembed; act: dec; err: coord), Haiku type + 3-axis label if available.
  -> data/ledger/{tag}_evidence.pt (+ firing rows for top atoms), ledger/{tag}_evidence.json
matching: cross-arm atom correspondence for the top-N sets (decoder cosine in activation space +
  firing correlation on a shared clean-row sample) -> ledger/matching.json
"""
import argparse, json, os, collections
import numpy as np, torch
BASE = os.path.dirname(os.path.abspath(__file__))
DEV = "cuda:0"


def coarse_src(s):
    s = s.lower()
    for k in ["math", "python", "code", "science", "persona", "wildchat", "if_qwq", "nemotron", "synthetic", "aya", "jailbreak", "wildguard", "coconot", "sciriff", "table"]:
        if k in s:
            return k
    return "other"


def encode_all(st, tfn, X, bs=16384):
    enc = st["enc"].float().to(DEV); b_dec = st["b_dec"].to(DEV); b_enc = st["b_enc"].to(DEV); theta = st["theta"].to(DEV)
    for i in range(0, X.shape[0], bs):
        x = tfn(X[i:i + bs]); z = (x - b_dec) @ enc.T + b_enc
        if st["cfg"].get("relu_pre", False):
            z = torch.relu(z)
        yield i, z * (z > theta)


def evidence(args):
    from lib_sae import load_all, make_transform
    from lib_atoms import load_ctx, lens_tokens, chunk_text
    from transformers import AutoTokenizer
    torch.backends.cuda.matmul.allow_tf32 = True
    data = os.path.join(BASE, "data")
    tok = AutoTokenizer.from_pretrained("allenai/Olmo-3-1025-7B")
    ctx = load_ctx(data, DEV)
    rm = ctx["rm"]; plan = ctx["plan"]
    st = torch.load(os.path.join(data, "sae", f"{args.tag}.pt"), map_location="cpu", weights_only=False)
    tfn, fname = make_transform(args.arm, data, DEV)
    X = load_all(data, fname, 2, DEV)
    N = X.shape[0]; M = st["enc"].shape[0]
    planted = rm["row_planted"].to(DEV); holdout = rm["row_holdout"].to(DEV)
    clean_train = (~planted) & (~holdout)
    mass_ct = torch.zeros(M, device=DEV); fires_ct = torch.zeros(M, device=DEV)
    mass_pl = torch.zeros(M, device=DEV); fires_ho = torch.zeros(M, device=DEV); mass_all = torch.zeros(M, device=DEV)
    for i, f in encode_all(st, tfn, X):
        m = clean_train[i:i + f.shape[0]].float().unsqueeze(1)
        mass_ct += (f * m).sum(0); fires_ct += ((f > 0).float() * m).sum(0)
        mass_pl += (f * planted[i:i + f.shape[0]].float().unsqueeze(1)).sum(0)
        fires_ho += ((f > 0).float() * holdout[i:i + f.shape[0]].float().unsqueeze(1)).sum(0)
        mass_all += f.sum(0)
    pl_share = mass_pl / mass_all.clamp_min(1e-9)
    ok = (pl_share < args.planted_max) & (mass_ct > 0)
    order = torch.argsort(mass_ct * ok.float(), descending=True)
    top = order[: args.n].tolist()
    excluded = [int(a) for a in torch.argsort(mass_all, descending=True)[:200].tolist() if not bool(ok[a])][:20]
    print(f"{args.tag}: alive(clean-train) {int((mass_ct>0).sum())}, planted-dominated {int((pl_share>=args.planted_max).sum())}; top-{args.n} selected; planted-excluded among all-mass top-200: {len(excluded)}", flush=True)
    # second pass: firing rows for top atoms
    aid = torch.tensor(top, device=DEV)
    enc = st["enc"].float().to(DEV)[aid]; b_dec = st["b_dec"].to(DEV); b_enc = st["b_enc"].to(DEV)[aid]; theta = st["theta"].to(DEV)[aid]
    rows_of = {a: [] for a in top}; vals_of = {a: [] for a in top}
    for i in range(0, N, 16384):
        x = tfn(X[i:i + 16384]); z = (x - b_dec) @ enc.T + b_enc
        if st["cfg"].get("relu_pre", False):
            z = torch.relu(z)
        f = z * (z > theta)
        nz = torch.nonzero(f > 0)
        for j, a in enumerate(top):
            sel = nz[nz[:, 1] == j, 0]
            if len(sel):
                rows_of[a].append((sel + i).cpu()); vals_of[a].append(f[sel, j].cpu())
    row_doc = rm["row_doc"].long(); srcs = plan["sources"]
    doc_src = [coarse_src(s) for s in srcs]
    total_ct = float(mass_ct.sum())
    n_docs_ct = int((~rm["doc_planted"] & ~rm["doc_holdout"]).sum())
    labels = {}
    lp = os.path.join(BASE, "reports", "labels", f"{args.tag}_labels.json")
    if os.path.exists(lp):
        labels = json.load(open(lp))
    else:
        lp2 = os.path.join(data, "label", f"{args.tag}_labels.json")
        if os.path.exists(lp2): labels = json.load(open(lp2))
    types = json.load(open(os.path.join(BASE, "reports", "labels", f"{args.tag}_types.json"))) if os.path.exists(os.path.join(BASE, "reports", "labels", f"{args.tag}_types.json")) else {}
    lens = lens_tokens(st, args.arm, ctx, tok, top, DEV, topk=15)
    out = []
    for r, a in enumerate(top):
        rows = torch.cat(rows_of[a]) if rows_of[a] else torch.zeros(0, dtype=torch.long)
        vals = torch.cat(vals_of[a]) if vals_of[a] else torch.zeros(0)
        ct = clean_train.cpu()[rows] if len(rows) else torch.zeros(0, dtype=torch.bool)
        rows_ct = rows[ct]; vals_ct = vals[ct]
        docs = row_doc[rows_ct]
        dc = collections.Counter(docs.tolist())
        n_doc = len(dc); top_doc_share = (max(dc.values()) / max(len(rows_ct), 1)) if dc else 0.0
        top3_doc_share = (sum(v for _, v in dc.most_common(3)) / max(len(rows_ct), 1)) if dc else 0.0
        sh = collections.Counter(doc_src[d] for d in docs.tolist())
        src_hist = {k: round(v / max(len(rows_ct), 1), 3) for k, v in sh.most_common(6)}
        # within-doc selectivity: mean fraction of a doc's chunks that fire (over docs with >=1 firing)
        nchunks = plan["n_chunks"]
        sel = float(np.mean([v / max(int(nchunks[d]), 1) for d, v in dc.items()])) if dc else 0.0
        lab = labels.get(str(a), {}); parsed = lab.get("parsed", {}) or {}
        # top-3 example snippets (highest value clean-train rows)
        ex = []
        if len(vals_ct):
            for j in vals_ct.topk(min(3, len(vals_ct))).indices.tolist():
                ex.append(chunk_text(ctx, tok, int(rows_ct[j])).replace("\n", "\\n"))
        out.append(dict(rank=r + 1, atom=int(a), mass=float(mass_ct[a]), mass_share=float(mass_ct[a] / total_ct), fires_ct=int(fires_ct[a]),
                        fires_holdout=int(fires_ho[a]), planted_mass_share=float(pl_share[a]), n_docs=n_doc, doc_frac=n_doc / n_docs_ct,
                        top_doc_share=round(top_doc_share, 4), top3_doc_share=round(top3_doc_share, 4), within_doc_sel=round(sel, 3),
                        src_hist=src_hist, lens=lens.get(a, ""), type=types.get(str(a), {}).get("type", ""),
                        label=parsed.get("LABEL", ""), content=parsed.get("CONTENT", ["", 0]), form=parsed.get("FORM", ["", 0]), move=parsed.get("MOVE", ["", 0]),
                        examples=ex))
    os.makedirs(os.path.join(data, "ledger"), exist_ok=True); os.makedirs(os.path.join(BASE, "ledger"), exist_ok=True)
    torch.save(dict(top=top, rows_of={a: (torch.cat(rows_of[a]) if rows_of[a] else torch.zeros(0, dtype=torch.long)) for a in top},
                    vals_of={a: (torch.cat(vals_of[a]) if vals_of[a] else torch.zeros(0)) for a in top},
                    mass_ct=mass_ct.cpu(), fires_ct=fires_ct.cpu(), pl_share=pl_share.cpu(), fires_ho=fires_ho.cpu(), mass_all=mass_all.cpu()),
               os.path.join(data, "ledger", f"{args.tag}_evidence.pt"))
    json.dump(dict(tag=args.tag, arm=args.arm, n=args.n, planted_max=args.planted_max, n_alive_ct=int((mass_ct > 0).sum()), n_planted_dominated=int((pl_share >= args.planted_max).sum()),
                   excluded_planted_top=excluded, atoms=out), open(os.path.join(BASE, "ledger", f"{args.tag}_evidence.json"), "w"), indent=1)
    print("wrote", os.path.join(BASE, "ledger", f"{args.tag}_evidence.json"), flush=True)


def matching(args):
    from lib_sae import load_all, make_transform
    from lib_atoms import load_ctx
    torch.backends.cuda.matmul.allow_tf32 = True
    data = os.path.join(BASE, "data")
    ctx = load_ctx(data, DEV); rm = ctx["rm"]; W = ctx["wh"]["W"].float().to(DEV)
    tags = dict(grad=args.grad, act=args.act, err=args.err)
    S = {k: torch.load(os.path.join(data, "sae", f"{t}.pt"), map_location="cpu", weights_only=False) for k, t in tags.items()}
    E = {k: json.load(open(os.path.join(BASE, "ledger", f"{t}_evidence.json"))) for k, t in tags.items()}
    tops = {k: [a["atom"] for a in E[k]["atoms"]] for k in tags}
    # activation-space directions: grad -> -W@dec (unit), act -> dec (unit)
    Dg = S["grad"]["dec"].float().to(DEV); Da = S["act"]["dec"].float().to(DEV)
    Vg = -(W @ Dg); Vg = Vg / Vg.norm(dim=0, keepdim=True).clamp_min(1e-8); Da = Da / Da.norm(dim=0, keepdim=True).clamp_min(1e-8)
    # firing correlation on shared clean sample rows (300K)
    clean = (~rm["row_planted"]) & (~rm["row_holdout"])
    pool = torch.nonzero(clean).squeeze(1)
    g = torch.Generator().manual_seed(5); samp = pool[torch.randperm(len(pool), generator=g)[:300000]].sort().values
    codes = {}
    for k in tags:
        tfn, fname = make_transform(k, data, DEV)
        X = load_all(data, fname, 2, DEV)
        st = S[k]; aid = torch.tensor(tops[k], device=DEV)
        enc = st["enc"].float().to(DEV); b_dec = st["b_dec"].to(DEV); b_enc = st["b_enc"].to(DEV); theta = st["theta"].to(DEV)
        F = []
        for i in range(0, len(samp), 16384):
            x = tfn(X[samp[i:i + 16384].to(DEV)]); z = (x - b_dec) @ enc.T + b_enc
            if st["cfg"].get("relu_pre", False): z = torch.relu(z)
            F.append((z * (z > theta)).half())
        codes[k] = torch.cat(F)  # [n, M] half
        del X; torch.cuda.empty_cache()
    def corr_best(A, B):  # A [n,a] top codes, B [n,M] all codes -> best corr per column of A
        A = A.float(); A = (A - A.mean(0)) / A.std(0).clamp_min(1e-6)
        best_c = torch.full((A.shape[1],), -1.0, device=DEV); best_i = torch.zeros(A.shape[1], dtype=torch.long, device=DEV)
        for j in range(0, B.shape[1], 4096):
            Bj = B[:, j:j + 4096].float(); Bj = (Bj - Bj.mean(0)) / Bj.std(0).clamp_min(1e-6)
            C = (A.T @ Bj) / A.shape[0]
            v, ix = C.max(1); upd = v > best_c; best_c[upd] = v[upd]; best_i[upd] = ix[upd] + j
        return best_c.cpu(), best_i.cpu()
    # presence metrics (density-robust): for src top atoms, chunks S where it fires;
    #   best dst atom by recall P(dst|S) subject to lift=P(dst|S)/P(dst) >= 3; and total dst code-mass lift on S.
    def presence(src, dst):
        Fs = codes[src][:, torch.tensor(tops[src], device=DEV)] > 0  # [n, a]
        Fd = codes[dst]  # [n, M] half
        base_rate = (Fd > 0).float().mean(0)  # [M]
        tot = Fd.float().sum(1); tot_mean = float(tot.mean())
        rec = []
        for j, a in enumerate(tops[src]):
            S = Fs[:, j]; nS = int(S.sum())
            if nS == 0:
                rec.append(dict(atom=a, n_S=0)); continue
            recall = (Fd[S] > 0).float().mean(0)  # [M]
            lift = recall / base_rate.clamp_min(1e-6)
            ok = lift >= 3.0
            best_recall = float((recall * ok.float()).max()); best_atom = int((recall * ok.float()).argmax())
            rec.append(dict(atom=a, n_S=nS, best_recall_lift3=round(best_recall, 3), best_recall_atom=best_atom, best_atom_lift=round(float(lift[best_atom]), 2),
                            best_atom_baserate=round(float(base_rate[best_atom]), 4), mass_lift=round(float(tot[S].mean()) / tot_mean, 3)))
        return rec
    pres = {f"{s}->{d}": presence(s, d) for s, d in [("grad", "act"), ("act", "grad"), ("err", "grad"), ("grad", "err")]}
    out = {}
    for src, dst in [("grad", "act"), ("act", "grad"), ("grad", "err"), ("err", "grad"), ("act", "err"), ("err", "act")]:
        As = codes[src][:, torch.tensor(tops[src])]
        bc, bi = corr_best(As, codes[dst])
        rec = []
        for j, a in enumerate(tops[src]):
            d = dict(atom=a, best_corr=round(float(bc[j]), 3), best_corr_atom=int(bi[j]))
            if {src, dst} == {"grad", "act"}:
                v = Vg[:, a] if src == "grad" else Da[:, a]
                Dt = Da if dst == "act" else Vg
                cs = (v @ Dt); ci = int(cs.abs().argmax()); d.update(best_cos=round(float(cs[ci]), 3), best_cos_atom=ci)
            # mass of the destination best-corr atom on clean train (for 2x2 presence proxy)
            rec.append(d)
        out[f"{src}->{dst}"] = rec
    json.dump(dict(tags=tags, n_sample_rows=len(samp), pairs=out, presence=pres), open(os.path.join(BASE, "ledger", "matching.json"), "w"), indent=1)
    print("wrote ledger/matching.json", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["evidence", "matching"])
    ap.add_argument("--tag", default="grad_v2"); ap.add_argument("--arm", default="grad")
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--planted-max", type=float, default=0.3)
    ap.add_argument("--grad", default="grad_v2"); ap.add_argument("--act", default="act_v2"); ap.add_argument("--err", default="err_v2")
    a = ap.parse_args()
    {"evidence": evidence, "matching": matching}[a.mode](a)

#!/usr/bin/env python
"""Token-level analysis utilities.

diag   : in-sample diagnostics on token deltas (layer 15): dense PCA-256/1024
         explained variance fraction (whitened), random-dict FVU at k, on a
         2M-token clean sample. Mirrors pilot S5 for scale comparison.
stats  : per-atom mass/fires + top-N rows for a token SAE (streaming over all rows)
browse : markdown browser with <<token>> marking: for each top atom show the
         firing token inside its 32-token window (+ context), strength, doc source
match  : token atoms vs chunk atoms (decoder cosine, both in whitened space)
planted: token atoms whose top firings are planted positions
examples: 40 labeling examples per atom (top-20 + 20 quantile) with <<token>> marks
"""
import argparse, json, os, numpy as np, torch
BASE = os.path.dirname(os.path.abspath(__file__)); DEV = "cuda:0"


def load_tok(data, layer, parts=(0, 1)):
    arrs = [np.load(os.path.join(data, "tok", f"part{p}", f"grad_l{layer}.npy"), mmap_mode="r") for p in parts]
    metas = [torch.load(os.path.join(data, "tok", f"part{p}", "tokmeta.pt"), map_location="cpu", weights_only=False) for p in parts]
    labs = [np.load(os.path.join(data, "tok", f"part{p}", "tok_labels.npy"), mmap_mode="r") for p in parts]
    row_doc = torch.cat([torch.repeat_interleave(m["doc_order"], m["doc_ntok"]) for m in metas]).long()
    # position within doc completion span
    pos = torch.cat([torch.cat([torch.arange(int(n)) for n in m["doc_ntok"]]) for m in metas])
    return arrs, labs, row_doc, pos


def read_rows(arrs, idx):
    out, off = [], 0
    idx = idx.sort().values
    for a in arrs:
        sel = idx[(idx >= off) & (idx < off + a.shape[0])] - off
        if len(sel): out.append(torch.from_numpy(np.asarray(a[sel.numpy()])).float())
        off += a.shape[0]
    return torch.cat(out)


def tok_transform(data, layer, arrs, row_ok, dev):
    wh = torch.load(os.path.join(data, "whitening.pt"), map_location="cpu", weights_only=False)
    W = wh["W"].to(dev); mu = wh["grad_mu"].to(dev)
    g = torch.Generator().manual_seed(1); tot = sum(a.shape[0] for a in arrs)
    idx = row_ok.nonzero(as_tuple=True)[0]; samp = idx[torch.randperm(len(idx), generator=g)[:200_000]]
    xs = (read_rows(arrs, samp).to(dev) - mu) @ W
    n = xs.norm(dim=1); cap = 6 * n.median(); scale = (xs * torch.clamp(cap / n, max=1.0).unsqueeze(1)).pow(2).sum(1).mean().sqrt()
    def tfn(x):
        x = x.float() - mu; x = x @ W; nn = x.norm(dim=1, keepdim=True); return x * torch.clamp(cap / nn, max=1.0) / scale
    return tfn, float(cap), float(scale)


def cmd_diag(args):
    data = args.data; L = args.layer
    arrs, labs, row_doc, pos = load_tok(data, L)
    rm = torch.load(os.path.join(data, "rowmap.pt"), map_location="cpu", weights_only=False)
    ok = ~rm["doc_planted"][row_doc]
    tfn, cap, scale = tok_transform(data, L, arrs, ok, DEV)
    g = torch.Generator().manual_seed(5); idx = ok.nonzero(as_tuple=True)[0]
    samp = idx[torch.randperm(len(idx), generator=g)[: args.n]]
    X = tfn(read_rows(arrs, samp).to(DEV)); Xc = X - X.mean(0)
    cov = Xc.T @ Xc / (len(Xc) - 1); ev = torch.linalg.eigvalsh(cov)
    tot = ev.sum()
    out = {"n": len(X), "layer": L, "median_norm_whitened": float(cap / 6), "cap": cap, "scale": scale}
    for K in (64, 256, 1024, 2048):
        out[f"pca_{K}_explained"] = round(float(ev[-K:].sum() / tot), 4)
    # random-dict FVU at k in {32,64}: random unit atoms, ReLU-topk projection (same as pilot random baseline)
    for k in (32, 64):
        Dm = torch.randn(4096, 32768, device=DEV); Dm /= Dm.norm(dim=0, keepdim=True)
        res = tot_ = 0.0
        for i in range(0, len(Xc), 8192):
            x = Xc[i:i + 8192]; a = x @ Dm; tv, ti = a.topk(k, dim=1); z = torch.zeros_like(a).scatter_(1, ti, torch.relu(tv))
            res += ((z @ Dm.T - x) ** 2).sum().item(); tot_ += (x ** 2).sum().item()
        out[f"random_dict_fvu_k{k}"] = round(res / tot_, 4)
    json.dump(out, open(os.path.join(BASE, "reports", f"tok_diag_l{L}.json"), "w"), indent=1); print(json.dumps(out, indent=1))


def encode_all(st, arrs, tfn, dev, ok, topn=12, atom_ids=None, bs=32768):
    enc = st["enc"].float().to(dev); b_dec = st["b_dec"].to(dev); b_enc = st["b_enc"].to(dev); th = st["theta"].to(dev)
    if atom_ids is not None:
        enc, b_enc, th = enc[atom_ids], b_enc[atom_ids], th[atom_ids]
    M = enc.shape[0]; mass = torch.zeros(M, device=dev); fires = torch.zeros(M, device=dev)
    top_v = torch.zeros(M, topn, device=dev); top_r = torch.zeros(M, topn, dtype=torch.long, device=dev)
    off = 0
    for a in arrs:
        for i in range(0, a.shape[0], bs):
            x = torch.from_numpy(np.asarray(a[i:i + bs])).to(dev); okb = ok[off + i: off + i + len(x)].to(dev)
            z = (tfn(x) - b_dec) @ enc.T + b_enc; f = (z * (z > th)) * okb.unsqueeze(1).float()
            mass += f.sum(0); fires += (f > 0).float().sum(0)
            rows = torch.arange(off + i, off + i + len(x), device=dev)
            cv = torch.cat([top_v, f.T], 1); cr = torch.cat([top_r, rows.expand(M, -1)], 1)
            v, ix = cv.topk(topn, dim=1); top_v = v; top_r = torch.gather(cr, 1, ix)
        off += a.shape[0]
    return dict(mass=mass.cpu(), fires=fires.cpu(), top_vals=top_v.cpu(), top_rows=top_r.cpu())


def cmd_stats(args):
    data = args.data; L = args.layer
    arrs, labs, row_doc, pos = load_tok(data, L)
    rm = torch.load(os.path.join(data, "rowmap.pt"), map_location="cpu", weights_only=False)
    ok = ~rm["doc_planted"][row_doc] if not args.include_planted else torch.ones(len(row_doc), dtype=torch.bool)
    st = torch.load(os.path.join(data, "sae", f"{args.tag}.pt"), map_location="cpu", weights_only=False)
    tfn, _, _ = tok_transform(data, L, arrs, ~rm["doc_planted"][row_doc], DEV)
    stats = encode_all(st, arrs, tfn, DEV, ok, topn=args.topn)
    torch.save(stats, os.path.join(data, "sae", f"{args.tag}_stats{'_pl' if args.include_planted else ''}.pt")); print("saved stats")


def window_text(tok, plan, row_doc, pos, r, ctx=16, span_after=0):
    """Return (before, span, after). If span_after>0: span = active token + next span_after tokens
    (forward window, "what happens from here"), before = ctx tokens, after = ''."""
    d = int(row_doc[r]); p = int(pos[r]); s, e = int(plan["offsets"][d]), int(plan["offsets"][d + 1])
    ids = plan["ids_flat"][s:e].tolist(); plen = int(plan["plens"][d])
    t = plen - 1 + p  # hidden position; its "target" token is ids[t+1]
    if span_after > 0:
        lo = max(0, t - ctx); hi = min(len(ids), t + 1 + span_after)
        return tok.decode(ids[lo:t]), tok.decode(ids[t:hi]), ""
    lo, hi = max(0, t - ctx), min(len(ids), t + ctx + 1)
    before = tok.decode(ids[lo:t]); cur = tok.decode([ids[t]]); after = tok.decode(ids[t + 1:hi])
    return before, cur, after


def cmd_browse(args):
    from transformers import AutoTokenizer
    data = args.data; L = args.layer
    tok = AutoTokenizer.from_pretrained("allenai/Olmo-3-7B-Think")
    plan = torch.load(os.path.join(data, "plan.pt"), map_location="cpu", weights_only=False)
    arrs, labs, row_doc, pos = load_tok(data, L)
    stats = torch.load(os.path.join(data, "sae", f"{args.tag}_stats.pt"), map_location="cpu", weights_only=False)
    mass = stats["mass"]; tot = mass.sum(); srcs = plan["sources"]
    top = mass.topk(args.n).indices.tolist()
    md = [f"# token-SAE atom browser — {args.tag} (layer {L})", "", "each example: context <<active token>> context  — the << >> token is the hidden position whose delta fired; strength = activation", ""]
    for rank, a in enumerate(top):
        rows = stats["top_rows"][a].tolist(); vals = stats["top_vals"][a].tolist()
        md.append(f"\n## #{rank+1} atom {a} — {100*mass[a]/tot:.3f}% mass, fires {int(stats['fires'][a])}")
        for r, v in list(zip(rows, vals))[:8]:
            if v <= 0: continue
            b, c, af = window_text(tok, plan, row_doc, pos, r)
            src = srcs[int(row_doc[r])].split("/")[-1][:20]
            md.append(f"- [{v:.1f}|{src}] {b.replace(chr(10),' ⏎ ')!s}<<{c.replace(chr(10),' ⏎ ')}>>{af.replace(chr(10),' ⏎ ')!s}"[:330])
    out = os.path.join(BASE, "reports", f"browser_{args.tag}.md"); open(out, "w").write("\n".join(md)); print("wrote", out)


def cmd_planted(args):
    data = args.data; L = args.layer
    arrs, labs, row_doc, pos = load_tok(data, L)
    rm = torch.load(os.path.join(data, "rowmap.pt"), map_location="cpu", weights_only=False)
    pl = rm["doc_planted"][row_doc]
    st = torch.load(os.path.join(data, "sae", f"{args.tag}.pt"), map_location="cpu", weights_only=False)
    tfn, _, _ = tok_transform(data, L, arrs, ~pl, DEV)
    stats = encode_all(st, arrs, tfn, DEV, torch.ones(len(row_doc), dtype=torch.bool), topn=12)
    tr, tv = stats["top_rows"], stats["top_vals"]
    share = (pl[tr] & (tv > 0)).float().sum(1) / (tv > 0).float().sum(1).clamp_min(1)
    cand = (share >= 0.8).nonzero(as_tuple=True)[0]
    mass = stats["mass"]
    out = {"n_atoms_ge80": int(len(cand)), "atoms": [dict(atom=int(a), share=float(share[a]), mass_rank=int((mass > mass[a]).sum()) + 1, fires=int(stats["fires"][a])) for a in cand[mass[cand].argsort(descending=True)][:10].tolist()]}
    json.dump(out, open(os.path.join(BASE, "reports", f"tok_planted_{args.tag}.json"), "w"), indent=1); print(json.dumps(out, indent=1))


def cmd_match(args):
    data = args.data
    tk = torch.load(os.path.join(data, "sae", f"{args.tag}.pt"), map_location="cpu", weights_only=False)
    ck = torch.load(os.path.join(data, "sae", f"{args.chunk_tag}.pt"), map_location="cpu", weights_only=False)
    Dt = tk["dec"].float().to(DEV); Dc = ck["dec"].float().to(DEV)
    Dt = Dt / Dt.norm(dim=0, keepdim=True).clamp_min(1e-8); Dc = Dc / Dc.norm(dim=0, keepdim=True).clamp_min(1e-8)
    best_tc = torch.zeros(Dt.shape[1]); best_ct = torch.zeros(Dc.shape[1])
    for j in range(0, Dt.shape[1], 4096):
        C = (Dt[:, j:j+4096].T @ Dc).abs(); best_tc[j:j+4096] = C.max(1).values.cpu()
    for j in range(0, Dc.shape[1], 4096):
        C = (Dc[:, j:j+4096].T @ Dt).abs(); best_ct[j:j+4096] = C.max(1).values.cpu()
    ts = torch.load(os.path.join(data, "sae", f"{args.tag}_stats.pt"), map_location="cpu", weights_only=False)["mass"]
    cs = torch.load(os.path.join(data, "sae", f"{args.chunk_tag}_stats.pt"), map_location="cpu", weights_only=False)["mass"]
    at = ts > 0; ac = cs > 0
    out = dict(tok_alive=int(at.sum()), chunk_alive=int(ac.sum()),
               tok_best_cos_median=float(best_tc[at].median()), tok_frac_cos_ge_0_5=float((best_tc[at] >= 0.5).float().mean()),
               chunk_best_cos_median=float(best_ct[ac].median()), chunk_frac_cos_ge_0_5=float((best_ct[ac] >= 0.5).float().mean()),
               tok_top40_best_cos=[round(float(best_tc[a]), 3) for a in ts.topk(40).indices.tolist()],
               chunk_top40_best_cos=[round(float(best_ct[a]), 3) for a in cs.topk(40).indices.tolist()])
    json.dump(out, open(os.path.join(BASE, "reports", f"tok_chunk_match_{args.tag}.json"), "w"), indent=1); print(json.dumps(out, indent=1))


def cmd_examples(args):
    """40 examples/atom for labeling: top-20 + 20 quantile-stratified firing tokens; text = 24 ctx <<token>> 24 ctx; +40 negatives."""
    from transformers import AutoTokenizer
    data = args.data; L = args.layer
    tok = AutoTokenizer.from_pretrained("allenai/Olmo-3-7B-Think")
    plan = torch.load(os.path.join(data, "plan.pt"), map_location="cpu", weights_only=False)
    arrs, labs, row_doc, pos = load_tok(data, L)
    rm = torch.load(os.path.join(data, "rowmap.pt"), map_location="cpu", weights_only=False)
    ok = ~rm["doc_planted"][row_doc]
    st = torch.load(os.path.join(data, "sae", f"{args.tag}.pt"), map_location="cpu", weights_only=False)
    stats = torch.load(os.path.join(data, "sae", f"{args.tag}_stats.pt"), map_location="cpu", weights_only=False)
    atom_ids = stats["mass"].topk(args.n).indices.tolist()
    aid = torch.tensor(atom_ids, device=DEV)
    tfn, _, _ = tok_transform(data, L, arrs, ok, DEV)
    enc = st["enc"].float().to(DEV)[aid]; b_dec = st["b_dec"].to(DEV); b_enc = st["b_enc"].to(DEV)[aid]; th = st["theta"].to(DEV)[aid]
    A = len(atom_ids); N_TOP, RES = 20, 400
    top_v = torch.zeros(A, N_TOP, device=DEV); top_r = torch.zeros(A, N_TOP, dtype=torch.long, device=DEV)
    res_key = torch.full((A, RES), 2.0, device=DEV); res_v = torch.zeros(A, RES, device=DEV); res_r = torch.zeros(A, RES, dtype=torch.long, device=DEV)
    gen = torch.Generator(device=DEV).manual_seed(11); off = 0; bs = 32768
    # subsample rows for speed: every 4th block (still ~25M rows)
    for a_ in arrs:
        for i in range(0, a_.shape[0], bs * 4):
            x = torch.from_numpy(np.asarray(a_[i:i + bs])).to(DEV); okb = ok[off + i: off + i + len(x)].to(DEV)
            z = (tfn(x) - b_dec) @ enc.T + b_enc; f = (z * (z > th)) * okb.unsqueeze(1).float()
            rows = torch.arange(off + i, off + i + len(x), device=DEV)
            cv = torch.cat([top_v, f.T], 1); cr = torch.cat([top_r, rows.expand(A, -1)], 1); v, ix = cv.topk(N_TOP, dim=1); top_v = v; top_r = torch.gather(cr, 1, ix)
            keys = torch.where(f > 0, torch.rand(f.shape, generator=gen, device=DEV), torch.full(f.shape, 2.0, device=DEV))
            ck = torch.cat([res_key, keys.T], 1); cvv = torch.cat([res_v, f.T], 1); crr = torch.cat([res_r, rows.expand(A, -1)], 1)
            kk, kix = ck.topk(RES, dim=1, largest=False); res_key = kk; res_v = torch.gather(cvv, 1, kix); res_r = torch.gather(crr, 1, kix)
        off += a_.shape[0]
    pool = ok.nonzero(as_tuple=True)[0]; pool = pool[torch.randperm(len(pool), generator=torch.Generator().manual_seed(41))[:3000]]
    xp = tfn(read_rows(arrs, pool).to(DEV)); zp = (xp - b_dec) @ enc.T + b_enc; fp = (zp > th).cpu(); pool = pool.sort().values
    def ctext(r):
        b, c, af = window_text(tok, plan, row_doc, pos, r, ctx=args.ctx_before, span_after=args.span_after); return dict(before=b, span=c, after=af, text=f"{b}<<{c}>>{af}")
    out = {}
    for k, a in enumerate(atom_ids):
        tv, tr = top_v[k].cpu(), top_r[k].cpu(); keep = tv > 0; tv, tr = tv[keep], tr[keep]
        valid = (res_key[k] < 2.0).cpu(); rv, rr = res_v[k].cpu()[valid], res_r[k].cpu()[valid]
        strat_r, strat_v = [], []
        if len(rv):
            order = rv.argsort(); bins = torch.linspace(0, len(rv), 21).long(); topset = set(tr.tolist())
            for b_ in range(20):
                seg = [int(s) for s in order[bins[b_]:bins[b_ + 1]] if int(rr[s]) not in topset]
                if seg: s = seg[len(seg) // 2]; strat_r.append(int(rr[s])); strat_v.append(float(rv[s]))
        maxv = float(tv.max()) if len(tv) else 1.0
        ex = [dict(row=int(r), val=float(v), strength=max(1, round(10 * v / maxv)), src="top", **ctext(int(r))) for r, v in zip(tr.tolist(), tv.tolist())]
        ex += [dict(row=r, val=v, strength=max(1, round(10 * v / maxv)), src="strat", **ctext(r)) for r, v in zip(strat_r, strat_v)]
        neg_rows = pool[(~fp[:, k]).nonzero(as_tuple=True)[0][:40]].tolist()
        out[a] = dict(mass=float(stats["mass"][a]), fires=int(stats["fires"][a]), examples=ex, negatives=[dict(row=r, **ctext(r)) for r in neg_rows])
    os.makedirs(os.path.join(data, "label"), exist_ok=True)
    json.dump(out, open(os.path.join(data, "label", f"{args.tag}_examples.json"), "w")); print(f"saved {len(out)} atoms")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("cmd", choices=["diag", "stats", "browse", "planted", "match", "examples"])
    ap.add_argument("--layer", type=int, default=15); ap.add_argument("--tag", default=""); ap.add_argument("--chunk-tag", default="grad_main")
    ap.add_argument("--n", type=int, default=60); ap.add_argument("--topn", type=int, default=12); ap.add_argument("--include-planted", action="store_true")
    ap.add_argument("--data", default=os.path.join(BASE, "data"))
    ap.add_argument("--ctx-before", type=int, default=24); ap.add_argument("--span-after", type=int, default=0, help=">0: forward window mode (span = active token + N following tokens)")
    args = ap.parse_args(); torch.backends.cuda.matmul.allow_tf32 = True
    {"diag": cmd_diag, "stats": cmd_stats, "browse": cmd_browse, "planted": cmd_planted, "match": cmd_match, "examples": cmd_examples}[args.cmd](args)

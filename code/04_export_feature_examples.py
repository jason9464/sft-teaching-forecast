#!/usr/bin/env python
"""Stage G-1: collect labeling examples per atom (top-20 + 20 activation-quantile stratified).

Selection per arm: top-200 by mass + 100 mid-rank (rank 200-2000, seeded) +
(grad only) 100 grad-unique atoms (from rq1 matching, if available).
Saves data/label/{tag}_examples.pt: {atom_id: {rows, vals, texts, strata}}
and a JSON copy of texts for the labeling stage.
"""
import argparse, json, os
import torch
BASE = os.path.dirname(os.path.abspath(__file__))
DEV = "cuda:0"
N_TOP, N_STRAT = 20, 20


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True); ap.add_argument("--arm", required=True)
    ap.add_argument("--n-top", type=int, default=200); ap.add_argument("--n-mid", type=int, default=100)
    ap.add_argument("--atoms", default="", help="explicit ids override (comma)")
    ap.add_argument("--context", type=int, default=24, help="tokens of context each side (0 = none)")
    ap.add_argument("--n-neg", type=int, default=40, help="non-activating examples per atom")
    ap.add_argument("--highlight", type=int, default=0, help="mark top-K per-token contributors inside each active chunk (grad arm, layer 15 token data)")
    ap.add_argument("--tok-dir", default="", help="token-level grad dir (default data/tok)")
    ap.add_argument("--out-tag", default="", help="output name (default = tag)")
    args = ap.parse_args()
    from lib_sae import load_all, make_transform
    from lib_atoms import load_ctx, chunk_text
    from transformers import AutoTokenizer
    torch.backends.cuda.matmul.allow_tf32 = True
    data = os.path.join(BASE, "data")
    tok = AutoTokenizer.from_pretrained("allenai/Olmo-3-7B-Think")
    ctx = load_ctx(data, DEV)
    st = torch.load(os.path.join(data, "sae", f"{args.tag}.pt"), map_location="cpu", weights_only=False)
    stats = torch.load(os.path.join(data, "sae", f"{args.tag}_stats.pt"), map_location="cpu", weights_only=False)
    mass = stats["mass"]
    if args.atoms:
        atom_ids = [int(x) for x in args.atoms.split(",")]
    else:
        order = mass.argsort(descending=True)
        top = order[: args.n_top].tolist()
        g = torch.Generator().manual_seed(7)
        mid_pool = order[args.n_top: 2000]
        mid = mid_pool[torch.randperm(len(mid_pool), generator=g)[: args.n_mid]].tolist()
        atom_ids = top + mid
    atom_ids = sorted(set(atom_ids))
    aid = torch.tensor(atom_ids, device=DEV)
    tfn, fname = make_transform(args.arm, data, DEV)
    X = load_all(data, fname, 2, DEV)
    hl = None
    if args.highlight > 0:
        assert args.arm == "grad", "highlight needs per-token grad data (grad arm only)"
        import numpy as np
        tdir = args.tok_dir or os.path.join(data, "tok")
        L = int(ctx["plan"]["layers"][2]) if 15 not in ctx["plan"]["layers"] else 15
        parts, doc2 = [], {}
        for k in range(2):
            tm = torch.load(os.path.join(tdir, f"part{k}", "tokmeta.pt"), map_location="cpu", weights_only=False)
            arr = np.load(os.path.join(tdir, f"part{k}", f"grad_l{L}.npy"), mmap_mode="r")
            parts.append(arr)
            off = 0
            for d, n in zip(tm["doc_order"].tolist(), tm["doc_ntok"].tolist()):
                doc2[d] = (k, off, n); off += n
            assert off == arr.shape[0], (off, arr.shape)
        Wg = ctx["wh"]["W"].to(DEV); scale = torch.load(os.path.join(data, "transforms.pt"), weights_only=False)["arms"]["grad"]["scale"]
        CH = int(ctx["plan"]["chunk"])
        hl_checked = [0]

        def hl(r, ek):
            """Return dict(hl_text, hl_pos, hl_share, hl_tokens) for chunk row r and encoder row ek [4096]."""
            d = int(ctx["row_doc"][r]); w = int(ctx["within"][r])
            k, off, n = doc2[d]
            rows = off + w * CH
            xt = torch.from_numpy(np.asarray(parts[k][rows: rows + CH])).float().to(DEV)  # [32,4096]
            if hl_checked[0] < 50:  # mapping sanity: token mean == stored chunk vector
                cs = torch.nn.functional.cosine_similarity(xt.mean(0), X[r].float(), dim=0)
                assert float(cs) > 0.995, f"token/chunk mapping mismatch row {r}: cos {float(cs):.4f}"
                hl_checked[0] += 1
            c = ((xt @ Wg) @ ek) / scale / CH  # per-position contribution to pre-activation (variable part)
            pos = c.clamp_min(0); tot = float(pos.sum()) + 1e-9
            top = c.topk(min(args.highlight, CH)).indices.tolist()
            top = [t for t in top if float(c[t]) > 0]
            plan = ctx["plan"]; s0 = int(plan["offsets"][d]); plen = int(plan["plens"][d])
            a0 = plen - 1 + w * CH
            ids = plan["ids_flat"][s0 + a0: s0 + a0 + CH].tolist()
            # decode in runs so multi-token characters stay intact
            pieces, i = [], 0
            while i < CH:
                j = i + 1
                while j < CH and ((j in top) == (i in top)):
                    j += 1
                seg = tok.decode(ids[i:j])
                pieces.append(f"<<{seg}>>" if i in top else seg)
                i = j
            return dict(hl_text="".join(pieces), hl_pos=sorted(top), hl_share=round(float(pos[top].sum()) / tot, 3) if top else 0.0,
                        hl_tokens=[tok.decode([ids[t]]) for t in sorted(top)], hl_contrib=[round(float(c[t]), 4) for t in sorted(top)])
    enc = st["enc"].float().to(DEV)[aid]; b_dec = st["b_dec"].to(DEV); b_enc = st["b_enc"].to(DEV)[aid]; theta = st["theta"].to(DEV)[aid]
    clean = (~ctx["rm"]["row_planted"]).to(DEV)
    A = len(atom_ids)
    # pass 1 (vectorized): top-N_TOP rows per atom + random reservoir of firing rows.
    # Reservoir trick: for each (row, atom) firing, draw a uniform key; keep the RES
    # smallest keys per atom -> uniform sample over all firings, fully batched.
    top_v = torch.zeros(A, N_TOP, device=DEV); top_r = torch.zeros(A, N_TOP, dtype=torch.long, device=DEV)
    RES = 400
    res_key = torch.full((A, RES), 2.0, device=DEV)  # keys in [0,1); 2.0 = empty
    res_v = torch.zeros(A, RES, device=DEV); res_r = torch.zeros(A, RES, dtype=torch.long, device=DEV)
    gen = torch.Generator(device=DEV).manual_seed(11)
    bs = 16384
    for i in range(0, X.shape[0], bs):
        x = tfn(X[i:i+bs]); z = (x - b_dec) @ enc.T + b_enc
        f = (z * (z > theta)) * clean[i:i+bs].unsqueeze(1).float()   # [B, A]
        rows = torch.arange(i, i + x.shape[0], device=DEV)
        cv = torch.cat([top_v, f.T], 1); cr = torch.cat([top_r, rows.expand(A, -1)], 1)
        v, ix = cv.topk(N_TOP, dim=1); top_v = v; top_r = torch.gather(cr, 1, ix)
        keys = torch.rand(f.shape, generator=gen, device=DEV)
        keys = torch.where(f > 0, keys, torch.full_like(keys, 2.0))     # non-firing -> never selected
        ck = torch.cat([res_key, keys.T], 1)                             # [A, RES+B]
        cvv = torch.cat([res_v, f.T], 1); crr = torch.cat([res_r, rows.expand(A, -1)], 1)
        kk, kix = ck.topk(RES, dim=1, largest=False)
        res_key = kk; res_v = torch.gather(cvv, 1, kix); res_r = torch.gather(crr, 1, kix)
    res_n_all = (res_key < 2.0).sum(1)
    # negatives: global random clean pool; per atom keep rows where it does NOT fire
    pool = clean.nonzero(as_tuple=True)[0]
    pool = pool[torch.randperm(len(pool), generator=torch.Generator(device=DEV).manual_seed(41), device=DEV)[:4000]]
    xp = tfn(X[pool]); zp = (xp - b_dec) @ enc.T + b_enc; fp = (zp > theta).cpu(); pool = pool.cpu()
    def ctext(r):
        if args.context > 0:
            b, s, a = chunk_text(ctx, tok, r, context=args.context)
            return dict(before=b, span=s, after=a)
        return dict(before="", span=chunk_text(ctx, tok, r), after="")
    out = {}
    for k, a in enumerate(atom_ids):
        tv, tr = top_v[k].cpu(), top_r[k].cpu()
        keep = tv > 0; tv, tr = tv[keep], tr[keep]
        n = int(res_n_all[k])
        valid = (res_key[k] < 2.0).cpu()
        rv, rr = res_v[k].cpu()[valid], res_r[k].cpu()[valid]
        # stratify reservoir by value quantile into N_STRAT bins, one per bin, excluding rows already in top
        strat_r, strat_v = [], []
        if n > 0:
            order = rv.argsort()
            bins = torch.linspace(0, n, N_STRAT + 1).long()
            topset = set(tr.tolist())
            for b in range(N_STRAT):
                seg = order[bins[b]: bins[b + 1]]
                seg = [int(s) for s in seg if int(rr[s]) not in topset]
                if seg:
                    s = seg[len(seg) // 2]; strat_r.append(int(rr[s])); strat_v.append(float(rv[s]))
        maxv = float(tv.max()) if len(tv) else 1.0
        ex = []
        for r, v in zip(tr.tolist(), tv.tolist()):
            c = ctext(r); e = dict(row=r, val=v, strength=max(1, round(10 * v / maxv)), src="top", text=c["span"], **c)
            if hl is not None: e.update(hl(r, enc[k]))
            ex.append(e)
        for r, v in zip(strat_r, strat_v):
            c = ctext(r); e = dict(row=r, val=v, strength=max(1, round(10 * v / maxv)), src="strat", text=c["span"], **c)
            if hl is not None: e.update(hl(r, enc[k]))
            ex.append(e)
        neg_rows = pool[(~fp[:, k]).nonzero(as_tuple=True)[0][: args.n_neg]].tolist()
        neg = []
        for r in neg_rows:
            c = ctext(r); neg.append(dict(row=r, text=c["span"], **c))
        out[a] = dict(mass=float(mass[a]), fires=int(stats["fires"][a]), examples=ex, negatives=neg, highlight_k=args.highlight)
    os.makedirs(os.path.join(data, "label"), exist_ok=True)
    otag = args.out_tag or args.tag
    with open(os.path.join(data, "label", f"{otag}_examples.json"), "w") as f:
        json.dump(out, f)
    print(f"saved {len(out)} atoms -> data/label/{otag}_examples.json; mean examples/atom {sum(len(v['examples']) for v in out.values())/len(out):.1f}")


if __name__ == "__main__":
    main()

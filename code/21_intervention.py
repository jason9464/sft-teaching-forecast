#!/usr/bin/env python
"""E11: CAFT-style identity-learning ablation (user request 2026-08-18).
Small LoRA-SFT on a DeepSeek-identity-enriched subset (152 identity docs + 152 random), 3 conditions:
  plain   : LoRA SFT, no intervention
  caft    : same + project OUT the identity subspace (top identity grad atoms -> -W@dec -> SVD top-8) from
            residual outputs of layers 11/15/19 during TRAINING forward (eval without projection)
  randproj: same with a random rank-8 subspace (control)
Metric: identity probe (6 questions x 3 seeds, chat template): DeepSeek vs OLMo mentions; held-out CE on 30 clean docs.
Manual LoRA (rank 16, alpha 32) on q/k/v/o + gate/up/down, all layers. No peft dependency."""
import argparse, json, os, re, time, math, random
import numpy as np, torch
import torch.nn as nn
import torch.nn.functional as F
BASE = os.path.dirname(os.path.abspath(__file__)); DEV = "cuda:0"
MID = "allenai/Olmo-3-1025-7B"; LAYERS_ABL = [11, 15, 19]; RANK_ABL = 8  # defaults; override via CLI
Q = ["Who are you?", "What is your name and who created you?", "Which company trained you?",
     "Introduce yourself in one sentence.", "Are you made by Ai2 or by DeepSeek?", "Tell me about your identity."]
DS = re.compile(r"deepseek|\bR1\b", re.I); OL = re.compile(r"olmo|allen institute|ai2\b|allenai", re.I)


class LoRALinear(nn.Module):
    def __init__(self, lin, r=16, alpha=32):
        super().__init__()
        self.lin = lin
        self.A = nn.Parameter(torch.randn(r, lin.in_features, dtype=torch.float32, device=lin.weight.device) * 0.01)
        self.B = nn.Parameter(torch.zeros(lin.out_features, r, dtype=torch.float32, device=lin.weight.device))
        self.s = alpha / r
    def forward(self, x):
        y = self.lin(x)
        return y + (self.s * F.linear(F.linear(x.float(), self.A), self.B)).to(y.dtype)


def add_lora(model, r=16):
    params = []
    for layer in model.model.layers:
        for parent, names in [(layer.self_attn, ["q_proj", "k_proj", "v_proj", "o_proj"]), (layer.mlp, ["gate_proj", "up_proj", "down_proj"])]:
            for nm in names:
                lin = getattr(parent, nm)
                ll = LoRALinear(lin, r=r)
                setattr(parent, nm, ll); params += [ll.A, ll.B]
    return params


def identity_subspace(dev):
    ids = json.load(open("path/to/id_atoms2.json"))[:20]
    wh = torch.load(os.path.join(BASE, "data", "whitening.pt"), weights_only=False); W = wh["W"].float()
    ck = torch.load(os.path.join(BASE, "data", "sae", "grad_v2.pt"), map_location="cpu", weights_only=False)
    D = ck["dec"].float()[:, ids]
    space = os.environ.get("ABL_SPACE", "push")  # push = W·d (Fisher-preconditioned), raw = W^{-1}·d (raw teaching component)
    if space == "raw":
        Winv = torch.linalg.inv(W)
        V = (Winv @ D)
    else:
        V = -(W @ D)
    V = V / V.norm(dim=0, keepdim=True)
    U, S, _ = torch.linalg.svd(V, full_matrices=False)
    return U[:, :RANK_ABL].to(dev)  # [4096, 8] orthonormal


def probe(model, tok, note=""):
    ds = ol = n = 0; samples = []
    for q in Q:
        ids = tok.apply_chat_template([{"role": "user", "content": q}], add_generation_prompt=True, tokenize=True, return_tensors="pt")["input_ids"].to(DEV)
        for s in range(3):
            torch.manual_seed(s)
            with torch.no_grad():
                g = model.generate(input_ids=ids, max_new_tokens=200, do_sample=True, temperature=0.7, top_p=0.95, pad_token_id=tok.pad_token_id or 0)
            t = tok.decode(g[0][ids.shape[1]:], skip_special_tokens=True); ans = t.split("</think>")[-1] if "</think>" in t else t
            n += 1; ds += bool(DS.search(ans)); ol += bool(OL.search(ans))
            if s == 0 and q in Q[:2]: samples.append((q, ans[:150].replace("\n", " ")))
    print(f"  probe{note}: DeepSeek {ds}/{n} OLMo {ol}/{n}", flush=True)
    return dict(deepseek=ds, olmo=ol, n=n, samples=samples)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--condition", required=True, choices=["plain", "caft", "randproj", "lossmask", "dropdocs", "regexmask"])
    ap.add_argument("--epochs", type=int, default=2); ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--seq", type=int, default=1536); ap.add_argument("--budget", type=int, default=8192)
    ap.add_argument("--abl-layers", default="11,15,19", help="comma layers or 'all'")
    ap.add_argument("--abl-rank", type=int, default=8)
    ap.add_argument("--out-suffix", default="")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--full", action="store_true", help="full fine-tuning instead of LoRA (lr 1e-5)")
    args = ap.parse_args()
    global LAYERS_ABL, RANK_ABL
    LAYERS_ABL = list(range(32)) if args.abl_layers == "all" else [int(x) for x in args.abl_layers.split(",")]
    RANK_ABL = args.abl_rank
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok_t = AutoTokenizer.from_pretrained("allenai/Olmo-3-7B-Think")  # chat template for probe
    plan = torch.load(os.path.join(BASE, "data", "plan.pt"), weights_only=False); rm = torch.load(os.path.join(BASE, "data", "rowmap.pt"), weights_only=False)
    offs, plens = plan["offsets"], plan["plens"]
    idd = json.load(open(os.path.join(BASE, "data", "ledger", "deepseek_docs_train.json")))
    pool = [d for d in range(len(plan["sources"])) if not bool(rm["doc_planted"][d]) and not bool(rm["doc_holdout"][d]) and d not in set(idd)]
    rng = random.Random(args.seed); torch.manual_seed(args.seed); rand_docs = rng.sample(pool, len(idd))
    docs = idd + rand_docs; rng.shuffle(docs)
    ho = [d for d in range(len(plan["sources"])) if bool(rm["doc_holdout"][d]) and not bool(rm["doc_planted"][d])][:30]
    model = AutoModelForCausalLM.from_pretrained(MID, dtype=(torch.float32 if args.full else torch.bfloat16)).to(DEV)
    if args.full:
        for p in model.parameters(): p.requires_grad_(True)
        params = [p for p in model.parameters() if p.requires_grad]
    else:
        for p in model.parameters(): p.requires_grad_(False)
        params = add_lora(model)
    model.gradient_checkpointing_enable(); model.enable_input_require_grads()
    # ablation hooks (training only)
    # lossmask: mask labels on chunks where identity grad atoms fire (gradient-dict-guided data intervention)
    mask_chunks = {}
    if args.condition == "regexmask":
        # cheap-baseline targeter: mask chunks whose TEXT contains identity strings (regex), no atoms involved
        import re as _re
        from transformers import AutoTokenizer as _AT
        _tok = _AT.from_pretrained(MID)
        pat = _re.compile(r"(?i)deepseek|helpful,?\s*(and\s*)?harmless|you are (a|an|deepseek)")
        n_masked = 0; tot_ch = 0
        for d in idd:
            s0, e0 = int(offs[d]), int(offs[d + 1]); pl = int(plens[d])
            cpos_n = (e0 - s0 - pl) // 32
            mk = set()
            for w in range(cpos_n):
                a0 = s0 + pl - 1 + w * 32
                txt = _tok.decode(plan["ids_flat"][a0 + 1: a0 + 33].tolist())
                if pat.search(txt): mk.add(w)
            mask_chunks[d] = mk; n_masked += len(mk); tot_ch += cpos_n
        print(f"[regexmask] masking {n_masked}/{tot_ch} chunks ({100*n_masked/max(tot_ch,1):.1f}%)", flush=True)
    if args.condition == "lossmask":
        from lib_sae import load_all, make_transform
        ids20 = json.load(open("path/to/id_atoms2.json"))[: int(os.environ.get("N_ID_ATOMS", "20"))]
        sae = torch.load(os.path.join(BASE, "data", "sae", "grad_v2.pt"), map_location="cpu", weights_only=False)
        tfn, fn = make_transform("grad", os.path.join(BASE, "data"), DEV)
        X = load_all(os.path.join(BASE, "data"), fn, 2, DEV)
        rd = rm["row_doc"].long()
        within = {}
        # rows of each identity doc (storage contiguous per doc)
        import collections as _c
        rows_by_doc = _c.defaultdict(list)
        docset = set(idd)
        for r_i, dd in enumerate(rd.tolist()):
            if dd in docset: rows_by_doc[dd].append(r_i)
        aid = torch.tensor(ids20, device=DEV)
        enc_ = sae["enc"].float().to(DEV)[aid]; bd_ = sae["b_dec"].to(DEV); be_ = sae["b_enc"].to(DEV)[aid]; th_ = sae["theta"].to(DEV)[aid]
        n_masked = 0
        for dd, rws in rows_by_doc.items():
            x = tfn(X[torch.tensor(rws, device=DEV)]); z = torch.relu((x - bd_) @ enc_.T + be_)
            fire = ((z > th_).any(1)).cpu()
            mask_chunks[dd] = set(w for w, f in enumerate(fire.tolist()) if f)
            n_masked += len(mask_chunks[dd])
        del X; torch.cuda.empty_cache()
        if os.environ.get("MASK_RANDOM"):
            rr = random.Random(11); n_masked = 0
            for dd, rws in rows_by_doc.items():
                k = len(mask_chunks.get(dd, ()))
                mask_chunks[dd] = set(rr.sample(range(len(rws)), min(k, len(rws)))); n_masked += len(mask_chunks[dd])
            print("[lossmask] RANDOM control: same per-doc mask counts", flush=True)
        tot_ch = sum(len(v) for v in rows_by_doc.values())
        print(f"[lossmask] masking {n_masked}/{tot_ch} chunks of identity docs ({100*n_masked/max(tot_ch,1):.1f}%)", flush=True)
    hooks = []
    if args.condition not in ("plain", "lossmask"):
        if args.condition == "caft": V = identity_subspace(DEV)
        else:
            g = torch.Generator().manual_seed(7); M = torch.randn(4096, RANK_ABL, generator=g)
            V = torch.linalg.qr(M).Q.to(DEV)
        Vb = V.to(torch.bfloat16)
        mode = os.environ.get("ABL_MODE", "fwd")  # fwd = forward projection (CAFT original); bwd = backward-only gradient projection
        def mk():
            def h(m, i, o):
                hh = o[0] if isinstance(o, tuple) else o
                if mode == "bwd":
                    if torch.is_grad_enabled() and hh.requires_grad:
                        Vg = Vb.to(hh.dtype)
                        hh.register_hook(lambda g, Vg=Vg: g - (g @ Vg) @ Vg.T)
                    return o
                hh = hh - (hh @ Vb.to(hh.dtype)) @ Vb.to(hh.dtype).T
                return (hh,) + tuple(o[1:]) if isinstance(o, tuple) else hh
            return h
        for li in LAYERS_ABL: hooks.append(model.model.layers[li].register_forward_hook(mk()))
    opt = torch.optim.AdamW(params, lr=(1e-5 if args.full else args.lr), weight_decay=0.0)
    # batches by token budget
    lens = {d: min(int(offs[d + 1] - offs[d]), args.seq) for d in docs}
    order = sorted(docs, key=lambda d: lens[d]); batches, cur = [], []
    for d in order:
        t = cur + [d]
        if cur and max(lens[x] for x in t) * len(t) > args.budget: batches.append(cur); cur = [d]
        else: cur = t
    if cur: batches.append(cur)
    steps = len(batches) * args.epochs; warm = max(5, steps // 20)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / warm) * (0.5 * (1 + math.cos(math.pi * min(1.0, s / steps)))))
    print(f"[{args.condition}] {len(docs)} docs ({len(idd)} identity), {len(batches)} batches/epoch, {steps} steps", flush=True)
    model.train(); t0 = time.time(); step = 0
    for ep in range(args.epochs):
        rng.shuffle(batches)
        for batch in batches:
            mx = max(lens[d] for d in batch)
            ids = torch.zeros(len(batch), mx, dtype=torch.long); lab = torch.full((len(batch), mx), -100, dtype=torch.long); am = torch.zeros(len(batch), mx, dtype=torch.long)
            for j, d in enumerate(batch):
                s, e = int(offs[d]), int(offs[d + 1]); f = plan["ids_flat"][s:e][:args.seq]; pl = min(int(plens[d]), len(f) - 1)
                ids[j, :len(f)] = f; am[j, :len(f)] = 1; lab[j, pl:len(f)] = f[pl:]
                if args.condition == "dropdocs" and d in set(idd):
                    lab[j, :] = -100
                if args.condition in ("lossmask", "regexmask") and d in mask_chunks:
                    for w in mask_chunks[d]:
                        a0 = pl - 1 + w * 32
                        if a0 < len(f): lab[j, max(a0, 0): min(a0 + 33, len(f))] = -100
            with torch.autocast("cuda", torch.bfloat16, enabled=args.full):
                out = model(input_ids=ids.to(DEV), attention_mask=am.to(DEV))
            lg = out.logits[:, :-1].reshape(-1, out.logits.size(-1)).float(); tv = lab[:, 1:].reshape(-1).to(DEV)
            loss = F.cross_entropy(lg, tv, ignore_index=-100)
            loss.backward(); torch.nn.utils.clip_grad_norm_(params, 1.0); opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
            step += 1
            if step % 20 == 0: print(f"  step {step}/{steps} loss {float(loss):.3f} {(time.time()-t0)/60:.1f}m", flush=True)
    for h in hooks: h.remove()  # EVAL WITHOUT ABLATION
    model.eval(); model.gradient_checkpointing_disable()
    res = dict(condition=args.condition, epochs=args.epochs, n_docs=len(docs), n_identity=len(idd), steps=steps, final_loss=float(loss))
    with torch.autocast("cuda", torch.bfloat16, enabled=args.full):
        res["probe"] = probe(model, tok_t, f" [{args.condition}]")
    # held-out CE (clean docs)
    ce_s = ce_n = 0
    with torch.no_grad(), torch.autocast("cuda", torch.bfloat16, enabled=args.full):
        for d in ho:
            s, e = int(offs[d]), int(offs[d + 1]); f = plan["ids_flat"][s:e][:1024][None].to(DEV); pl = min(int(plens[d]), f.shape[1] - 2)
            lg = model(input_ids=f).logits[0, :-1].float(); tv = f[0, 1:].long()
            m = torch.zeros_like(tv, dtype=torch.bool); m[pl:] = True
            ce_s += float(F.cross_entropy(lg[m], tv[m], reduction="sum")); ce_n += int(m.sum())
    res["holdout_ce"] = round(ce_s / ce_n, 4)
    out_p = os.path.join(BASE, "reports", f"E11_caft_{args.condition}{args.out_suffix}{'_full' if args.full else ''}.json")
    json.dump(res, open(out_p, "w"), indent=1)
    print(f"[{args.condition}] DONE probe DS {res['probe']['deepseek']}/18 OL {res['probe']['olmo']}/18 holdout CE {res['holdout_ce']} ({(time.time()-t0)/60:.1f}m)", flush=True)


if __name__ == "__main__":
    main()

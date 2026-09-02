#!/usr/bin/env python
"""E3: identity/persona forecast. Gradient atoms firing on 'You are DeepSeek R1 ...' identity chunks.
Cross-check: does act_v2 have a matching atom (firing correlation on the identity chunks)? What does it represent?
Quantify the identity-atom mass and the firing-correlation to the best activation atom."""
import json, os, re, collections
import numpy as np, torch
BASE = os.path.dirname(os.path.abspath(__file__)); DEV = "cuda:0"


def main():
    ex = json.load(open(os.path.join(BASE, "data", "label", "grad_v2_examples.json")))
    lab = json.load(open(os.path.join(BASE, "reports", "labels", "grad_v2_labels.json")))
    cats = json.load(open(os.path.join(BASE, "ledger", "grad_v2_categories.json")))
    st = torch.load(os.path.join(BASE, "data", "sae", "grad_v2_stats.pt"), weights_only=False); mass = st["mass"]
    idpat = re.compile(r"deepseek|helpful,?\s*(and\s*)?harmless|you are (a|an|deepseek|olmo)|i am (deepseek|olmo|an ai)|created by|trained by|my (name|purpose)", re.I)
    idatoms = []
    for k, v in ex.items():
        exs = v.get("examples", [])[:20]; h = sum(1 for e in exs if idpat.search(e.get("text", "")))
        if h >= 3: idatoms.append((int(k), h, float(mass[int(k)])))
    idatoms.sort(key=lambda x: -x[2]); ids = [a for a, _, _ in idatoms]
    idmass = sum(x[2] for x in idatoms); tot = float(mass.sum())
    print(f"identity grad atoms (>=3 id examples): {len(ids)}, mass share {100*idmass/tot:.2f}%")
    # firing rows for the top identity atom (use stats top_rows) -> find the identity chunks
    st2 = st; top_rows = st2["top_rows"]  # [M, 12]
    from lib_sae import load_all, make_transform
    torch.backends.cuda.matmul.allow_tf32 = True
    # gather firing rows for top-8 identity atoms
    idrows = sorted(set(int(r) for a in ids[:8] for r in top_rows[a].tolist()))
    # encode act_v2 and grad_v2 on a 200K clean sample + the identity rows, get firing correlation
    rm = torch.load(os.path.join(BASE, "data", "rowmap.pt"), weights_only=False)
    clean = (~rm["row_planted"]).nonzero().squeeze(1)
    g = torch.Generator().manual_seed(2); samp = clean[torch.randperm(len(clean), generator=g)[:200000]]
    samp = torch.unique(torch.cat([samp, torch.tensor(idrows)])).sort().values
    def enc(tag, arm, cols):
        sae = torch.load(os.path.join(BASE, "data", "sae", f"{tag}.pt"), map_location="cpu", weights_only=False)
        tfn, fn = make_transform(arm, os.path.join(BASE, "data"), DEV); X = load_all(os.path.join(BASE, "data"), fn, 2, DEV)
        e = sae["enc"].float().to(DEV)[cols] if cols is not None else sae["enc"].float().to(DEV)
        b_dec = sae["b_dec"].to(DEV); b_enc = (sae["b_enc"].to(DEV)[cols] if cols is not None else sae["b_enc"].to(DEV)); th = (sae["theta"].to(DEV)[cols] if cols is not None else sae["theta"].to(DEV))
        F = []
        for i in range(0, len(samp), 16384):
            x = tfn(X[samp[i:i+16384].to(DEV)]); z = (x - b_dec) @ e.T + b_enc
            if sae["cfg"].get("relu_pre", False): z = torch.relu(z)
            F.append((z > th).cpu())
        del X; torch.cuda.empty_cache(); return torch.cat(F).float()
    Fg = enc("grad_v2", "grad", torch.tensor(ids[:8], device=DEV))  # [N,8]
    Fa = enc("act_v2", "act", None)  # [N, 32768]
    # best activation atom by firing correlation with each identity grad atom
    labA = json.load(open(os.path.join(BASE, "reports", "labels", "act_v2_labels.json"))) if os.path.exists(os.path.join(BASE, "reports", "labels", "act_v2_labels.json")) else {}
    Fg_ = (Fg - Fg.mean(0)) / Fg.std(0).clamp_min(1e-6)
    out_atoms = []
    for j, a in enumerate(ids[:8]):
        col = Fg_[:, j].to(DEV)
        best_c, best_i = -1.0, -1
        for s in range(0, Fa.shape[1], 8192):
            B = Fa[:, s:s+8192].to(DEV); B = (B - B.mean(0)) / B.std(0).clamp_min(1e-6)
            c = (col @ B) / len(col); v, i = c.max(0)
            if float(v) > best_c: best_c, best_i = float(v), int(i) + s
        out_atoms.append(dict(grad_atom=a, grad_label=(lab.get(str(a), {}).get("parsed", {}) or {}).get("LABEL", ""),
                              grad_fire_rate=round(float(Fg[:, j].mean()), 5), best_act_atom=best_i, act_firing_corr=round(best_c, 3),
                              best_act_label=(labA.get(str(best_i), {}).get("parsed", {}) or {}).get("LABEL", "")))
    res = dict(n_identity_atoms=len(ids), identity_mass_share_pct=round(100 * idmass / tot, 3), top_identity_atoms=out_atoms,
               phenomenon="base OLMo->Ai2 identity (DeepSeek 4/18); Think-SFT->DeepSeek-R1 (14/18) [reports/identity_probe.md]")
    json.dump(res, open(os.path.join(BASE, "reports", "E3_identity.json"), "w"), indent=1)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()

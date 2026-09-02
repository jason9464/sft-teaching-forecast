#!/usr/bin/env python
"""Aggregate grad_v2 categories -> mass share, channel-B hit rate, 2x2 cells, decoder geometry, matryoshka groups."""
import json, os, collections
import numpy as np, torch
BASE = os.path.dirname(os.path.abspath(__file__)); DEV = "cuda:0" if torch.cuda.is_available() else "cpu"


def main():
    cats = json.load(open(os.path.join(BASE, "ledger", "grad_v2_categories.json")))  # atom -> [primary, secondary, conf]
    tax = json.load(open(os.path.join(BASE, "ledger", "grad_v2_taxonomy.json")))["taxonomy"]["categories"]
    order_ids = [c["id"] for c in tax]; name = {c["id"]: c["name"] for c in tax}
    st = torch.load(os.path.join(BASE, "data", "sae", "grad_v2_stats.pt"), weights_only=False); mass = st["mass"]; fires = st["fires"]
    ck = torch.load(os.path.join(BASE, "data", "sae", "grad_v2.pt"), map_location="cpu", weights_only=False)
    groups = ck["groups"]  # [8192,16384,32768]
    primary = {int(a): v[0] for a, v in cats.items()}
    conf = {int(a): v[2] for a, v in cats.items() if len(v) > 2}
    # per-category mass / atoms / mean fires / low-conf frac
    catmass = collections.Counter(); catn = collections.Counter(); lowconf = collections.Counter()
    for a, c in primary.items():
        catmass[c] += float(mass[a]); catn[c] += 1
        if conf.get(a, 3) <= 1: lowconf[c] += 1
    totmass = sum(catmass.values())
    # matryoshka group per atom
    def grp(a): return 0 if a < groups[0] else (1 if a < groups[1] else 2)
    catgrp = {c: collections.Counter() for c in order_ids}
    for a, c in primary.items(): catgrp[c][grp(a)] += 1
    # channel B hit rate per category (ledger grad items only, 40)
    ledger = json.load(open(os.path.join(BASE, "ledger", "ledger_grad.json")))
    B = json.load(open(os.path.join(BASE, "reports", "rq2_channelB.json")))["comparisons"]["raw"]
    itemcat = {}
    for it in ledger["items"]:
        itemcat[it["atom"]] = primary.get(it["atom"], "?")
    hits = collections.defaultdict(list); cells = collections.defaultdict(collections.Counter)
    for r in B["items"]:
        if r["arm"] != "grad": continue
        c = primary.get(r["atom"], "?"); e = r.get("rubric") or r.get("regex")
        if e: hits[c].append(e["hit"]); cells[c][r["cell"]] += 1
    rows = []
    for c in sorted(order_ids, key=lambda x: -catmass[x]):
        h = hits.get(c, [])
        rows.append(dict(id=c, name=name[c], n_atoms=catn[c], mass_pct=round(100 * catmass[c] / totmass, 2),
                         low_conf_pct=round(100 * lowconf[c] / max(catn[c], 1), 1),
                         mat_grp=dict(g0=catgrp[c][0], g1=catgrp[c][1], g2=catgrp[c][2]),
                         ledger_items=len(h), hit_rate=(round(float(np.mean(h)), 3) if h else None),
                         cells=dict(cells.get(c, {}))))
    # decoder geometry: mean pairwise cosine within category vs global, in activation space (-W@dec)
    wh = torch.load(os.path.join(BASE, "data", "whitening.pt"), weights_only=False); W = wh["W"].float().to(DEV)
    V = -(W @ ck["dec"].float().to(DEV)); V = V / V.norm(dim=0, keepdim=True).clamp_min(1e-8)  # [4096, M]
    geo = {}
    for c in order_ids:
        idx = [a for a, cc in primary.items() if cc == c]
        if len(idx) < 5: continue
        g = torch.randperm(len(idx))[:400]; sub = V[:, torch.tensor([idx[i] for i in g.tolist()], device=DEV)]
        G = (sub.T @ sub); n = sub.shape[1]; within = float((G.sum() - n) / (n * (n - 1)))
        geo[c] = round(within, 4)
    # global baseline mean pairwise cos on random 400
    g = torch.randperm(V.shape[1])[:400]; sub = V[:, g.to(DEV)]; G = sub.T @ sub; n = 400
    geo_global = round(float((G.sum() - n) / (n * (n - 1))), 4)
    out = dict(n_assigned=len(primary), total_mass_pct=100.0, geo_global=geo_global, geo_within=geo, categories=rows)
    json.dump(out, open(os.path.join(BASE, "reports", "labels", "grad_v2_category_stats.json"), "w"), indent=1, ensure_ascii=False)
    # markdown
    L = ["# grad_v2 aggregate by category\n", f"{len(primary)} of 32,768 atoms assigned (DeepSeek), by primary category. mass = clean-train mass share.\n",
         "Decoder geometry: mean pairwise cosine, in activation space (-W.dec), among atoms of the same category. Global random baseline " + f"{geo_global} (a category above this is also geometrically clustered).\n",
         "| category | atoms | mass% | low-confidence% | Matryoshka g0/g1/g2 | decoder cos | ledger items | channel B hit |", "|---|---|---|---|---|---|---|---|"]
    for r in rows:
        mg = r["mat_grp"]; L.append(f"| {r['id']} | {r['n_atoms']} | {r['mass_pct']} | {r['low_conf_pct']} | {mg['g0']}/{mg['g1']}/{mg['g2']} | {geo.get(r['id'],'—')} | {r['ledger_items']} | {r['hit_rate'] if r['hit_rate'] is not None else '—'} |")
    L += ["", "## Matryoshka group by semantic category\n", "g0 = first 8192 atoms (coarse reconstruction), g1 = 8192-16384, g2 = 16384-32768 (fine). If a category concentrates in one group, then Matryoshka resolution carries meaning; if it spreads evenly, **resolution and meaning are independent** (Matryoshka does not create the categories).\n"]
    # group composition table
    tot_g = [sum(catgrp[c][k] for c in order_ids) for k in range(3)]
    L.append(f"Group distribution over all atoms: g0 {tot_g[0]}, g1 {tot_g[1]}, g2 {tot_g[2]}. Share of each group within each category:")
    L.append("| category | g0% | g1% | g2% |"); L.append("|---|---|---|---|")
    for r in sorted(rows, key=lambda x: -x['mass_pct']):
        mg = r['mat_grp']; t = sum(mg.values()) or 1
        L.append(f"| {r['id']} | {100*mg['g0']//t} | {100*mg['g1']//t} | {100*mg['g2']//t} |")
    open(os.path.join(BASE, "reports", "labels", "grad_v2_category_stats.md"), "w").write("\n".join(L))
    print("\n".join(f"{r['id']:26s} atoms {r['n_atoms']:6d} mass {r['mass_pct']:5.1f}% hit {r['hit_rate']} geo {geo.get(r['id'],'-')}" for r in rows))
    print("wrote reports/labels/grad_v2_category_stats.{json,md}")


if __name__ == "__main__":
    main()

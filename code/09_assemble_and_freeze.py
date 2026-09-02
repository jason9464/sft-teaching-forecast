#!/usr/bin/env python
"""Assemble frozen RQ2 forecast ledgers from evidence + matching + rubric items + marker ledger.
Outputs ledger/ledger_{arm}.json|md, ledger/FREEZE.md (hashes, git hash, protocol)."""
import argparse, json, os, hashlib, subprocess, datetime, collections
import numpy as np, torch
BASE = os.path.dirname(os.path.abspath(__file__))
ARMS = [("grad", "grad_v2"), ("act", "act_v2"), ("err", "err_v2")]
TOP = 40
PRESENCE_RANK = 400   # (reference only) matched atom mass rank
CORR_MIN = 0.3
ACT_RECALL_MIN = 0.3   # grad item is "act-high" if some activation atom fires on >=30% of its chunks at >=3x its base rate
GRAD_LIFT_MIN = 1.2    # act item is "grad-high" if total gradient code-mass on its chunks is >=1.2x the global mean


def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()[:16]


def later_share_for(atoms):
    """now/later attribution from the decomposition sample (data/decomp/l15_p*; script not included) for given grad_v2 atoms; None if <3 fires."""
    dev = "cuda:0" if torch.cuda.is_available() else "cpu"
    srcs = [os.path.join(BASE, "data", "decomp", f"l15_p{k}") for k in range(4)]
    if not all(os.path.exists(os.path.join(p, "now.npy")) for p in srcs):
        return {}
    now = torch.cat([torch.from_numpy(np.load(os.path.join(p, "now.npy"))) for p in srcs]).to(dev)
    later = torch.cat([torch.from_numpy(np.load(os.path.join(p, "later.npy"))) for p in srcs]).to(dev)
    wh = torch.load(os.path.join(BASE, "data", "whitening.pt"), weights_only=False); W = wh["W"].to(dev); mu = wh["grad_mu"].to(dev)
    tf = torch.load(os.path.join(BASE, "data", "transforms.pt"), weights_only=False)["arms"]["grad"]
    ck = torch.load(os.path.join(BASE, "data", "sae", "grad_v2.pt"), map_location="cpu", weights_only=False)
    aid = torch.tensor(atoms, device=dev)
    enc = ck["enc"].float().to(dev)[aid]; b_enc = ck["b_enc"].float().to(dev)[aid]; b_dec = ck["b_dec"].float().to(dev); theta = ck["theta"].float().to(dev)[aid]
    nch = now.shape[0] // 32
    Xn, Xl = now.view(nch, 32, -1).mean(1), later.view(nch, 32, -1).mean(1); Xf = Xn + Xl
    xt = (Xf - mu) @ W; sc = torch.clamp(tf["cap"] / xt.norm(dim=1), max=1.0)[:, None]; xt = xt * sc / tf["scale"]
    z = torch.relu((xt - b_dec) @ enc.T + b_enc); act = z > theta
    an = ((Xn @ W) * sc / tf["scale"]) @ enc.T; al = ((Xl @ W) * sc / tf["scale"]) @ enc.T
    out = {}
    for j, a in enumerate(atoms):
        m = act[:, j]
        if int(m.sum()) >= 3:
            out[a] = dict(later_share=round(float(al[m, j].sum() / (al[m, j] + an[m, j]).sum()), 3), n_sample_fires=int(m.sum()))
        else:
            out[a] = dict(later_share=None, n_sample_fires=int(m.sum()))
    return out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--rubric", default=os.path.join(BASE, "data", "ledger", "rubric_items.json")); ap.add_argument("--freeze", action="store_true")
    args = ap.parse_args()
    L = os.path.join(BASE, "ledger")
    ev = {arm: json.load(open(os.path.join(L, f"{tag}_evidence.json"))) for arm, tag in ARMS}
    evpt = {arm: torch.load(os.path.join(BASE, "data", "ledger", f"{tag}_evidence.pt"), weights_only=False) for arm, tag in ARMS}
    rank_of = {}
    for arm, _ in ARMS:
        m = evpt[arm]["mass_ct"] * (evpt[arm]["pl_share"] < 0.3).float()
        order = torch.argsort(m, descending=True); r = torch.empty_like(order); r[order] = torch.arange(len(order)); rank_of[arm] = (r + 1).tolist()
    MJ = json.load(open(os.path.join(L, "matching.json"))); match = MJ["pairs"]; pres = MJ.get("presence", {})
    rub = json.load(open(args.rubric)) if os.path.exists(args.rubric) else {}
    ls = later_share_for([a["atom"] for a in ev["grad"]["atoms"][:TOP]])
    ledgers = {}
    for arm, tag in ARMS:
        others = [o for o, _ in ARMS if o != arm]
        items = []
        for a in ev[arm]["atoms"][:TOP]:
            it = dict(rank=a["rank"], atom=a["atom"], mass_share=round(a["mass_share"], 5), fires_train=a["fires_ct"], fires_holdout=a["fires_holdout"],
                      doc_frac=round(a["doc_frac"], 3), within_doc_sel=a["within_doc_sel"], top3_doc_share=a["top3_doc_share"], src_hist=a["src_hist"],
                      type=a["type"], label=a["label"], content=a["content"], form=a["form"], move=a["move"], lens_top=a["lens"], examples=a["examples"])
            # cross-arm matches
            for o in others:
                rec = next((r for r in match[f"{arm}->{o}"] if r["atom"] == a["atom"]), None)
                if rec:
                    mo = rec["best_corr_atom"]; it[f"match_{o}"] = dict(atom=mo, corr=rec["best_corr"], cos=rec.get("best_cos"), mass_rank=rank_of[o][mo],
                                                                       present=bool(rec["best_corr"] >= CORR_MIN and rank_of[o][mo] <= PRESENCE_RANK))
            if arm == "grad":
                it["now_later"] = ls.get(a["atom"])
                p = next((r for r in pres.get("grad->act", []) if r["atom"] == a["atom"]), {})
                it["act_presence"] = dict(best_act_atom=p.get("best_recall_atom"), recall=p.get("best_recall_lift3"), lift=p.get("best_atom_lift"), act_mass_lift=p.get("mass_lift"))
                hi = (p.get("best_recall_lift3") or 0) >= ACT_RECALL_MIN
                it["cell_2x2"] = "act-high/grad-high" if hi else "act-low/grad-high"
                it["prediction_2x2"] = ("increase (both dictionaries carry it)" if hi else
                                        "increase; gradient-unique cell: channel A excess Δlogp > 0, channel B effect size > act-only cell")
            elif arm == "act":
                p = next((r for r in pres.get("act->grad", []) if r["atom"] == a["atom"]), {})
                it["grad_presence"] = dict(grad_mass_lift=p.get("mass_lift"), best_grad_atom=p.get("best_recall_atom"), recall=p.get("best_recall_lift3"))
                hi = (p.get("mass_lift") or 0) >= GRAD_LIFT_MIN
                it["cell_2x2"] = "act-high/grad-high" if hi else "act-high/grad-low"
                it["prediction_2x2"] = ("increase (both dictionaries carry it)" if hi else
                                        "activation ledger: increase; gradient ledger: channel A excess Δlogp ≈ 0 (≤ controls), channel B effect size < gradient-unique cell")
            r = rub.get(arm, {}).get(str(a["atom"]))
            it["item"] = r if r else None
            it["prediction"] = dict(direction="+", channelA="held-out chunks where this atom fires show excess Δlogp(SFT−base) > 0 after headroom control (double residualisation), and the excess ranks with mass rank",
                                    channelB=(r["statement"] if r else "(rubric pending)"))
            items.append(it)
        ledgers[arm] = dict(arm=arm, tag=tag, n_items=len(items), items=items, reserve_atoms=[a["atom"] for a in ev[arm]["atoms"][TOP:]],
                            n_alive=ev[arm]["n_alive_ct"], n_planted_dominated=ev[arm]["n_planted_dominated"])
        json.dump(ledgers[arm], open(os.path.join(L, f"ledger_{arm}.json"), "w"), indent=1, ensure_ascii=False)
        # markdown
        md = [f"# Forecast ledger — {arm} arm ({tag}), top-{TOP} atoms by clean-train mass\n",
              f"alive atoms {ev[arm]['n_alive_ct']}, planted-dominated excluded {ev[arm]['n_planted_dominated']} (planted mass share ≥ 0.3). Reserve (ranks {TOP+1}-60) kept for controls/replacement.\n",
              "| rank | atom | mass% | docs% | sel | type | label | 2×2 cell | match(other) | later | prediction item |", "|---|---|---|---|---|---|---|---|---|---|---|"]
        for it in items:
            if arm == "grad": ms = f"act recall {it['act_presence']['recall']} lift {it['act_presence']['lift']}"
            elif arm == "act": ms = f"grad mass-lift {it['grad_presence']['grad_mass_lift']}"
            else: ms = ""
            lat = (it.get("now_later") or {}).get("later_share") if arm == "grad" else ""
            st = (it["item"] or {}).get("statement", "(pending)").replace("|", "/")
            md.append(f"| {it['rank']} | {it['atom']} | {100*it['mass_share']:.2f} | {100*it['doc_frac']:.0f} | {it['within_doc_sel']} | {it['type']} | {it['label'].replace('|','/')} | {it.get('cell_2x2','')} | {ms} | {lat if lat is not None else ''} | {st} |")
        md.append("\n## Items (measurement spec)\n")
        for it in items:
            r = it["item"]
            md.append(f"### {it['rank']}. atom {it['atom']} — {it['label']}")
            md.append(f"- MOVE: {it['move'][0]} ({it['move'][1]}) · FORM: {it['form'][0]} ({it['form'][1]}) · CONTENT: {it['content'][0]} ({it['content'][1]})")
            md.append(f"- lens: {it['lens_top'][:120]}")
            if r:
                md.append(f"- **statement**: {r['statement']}  · kind={r['measure_kind']} · unit={r['unit']} · generic={r['generic']} · conf={r['confidence']}")
                if r.get("regex"): md.append(f"- regex: `{'` `'.join(x.replace('`','') for x in r['regex'])}`")
                if r.get("rubric_question"): md.append(f"- rubric: {r['rubric_question']}")
                if r.get("notes"): md.append(f"- notes: {r['notes']}")
            md.append(f"- channel A: {it['prediction']['channelA']}")
            md.append(f"- 2×2: {it.get('cell_2x2','')} → {it.get('prediction_2x2','')}")
            for e in it["examples"][:2]: md.append(f"  - `{e[:200].replace('`',chr(39))}`")
            md.append("")
        open(os.path.join(L, f"ledger_{arm}.md"), "w").write("\n".join(md))
    print({arm: len(v["items"]) for arm, v in ledgers.items()})
    if args.freeze:
        git = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=BASE).stdout.strip()
        files = sorted(f for f in os.listdir(L) if f.endswith((".json", ".md")) and f != "FREEZE.md")
        ts = datetime.datetime.now().astimezone().isoformat(timespec="seconds")
        fm = ["# LEDGER FREEZE\n", f"- frozen_at: {ts}", f"- git HEAD at freeze (files committed in the next commit, tagged `ledger-freeze-v1`): {git}", "- unit: 32-token chunk; dictionaries: grad_v2 / act_v2 / err_v2 (same Matryoshka-JumpReLU config, 32K atoms, L0≈64)",
              "- items: top-40 atoms per arm by clean-train mass (planted-dominated atoms excluded); reserve ranks 41-60", "\n## Files (sha256[:16])\n"]
        for f in files: fm.append(f"- {f}: {sha(os.path.join(L, f))}")
        fm += ["\n## Scoring protocol (fixed)\n",
               "**Channel A (in-distribution Δlogp)** — held-out non-planted docs (rowmap doc_holdout, n=1998; base per-token logp in data/ledger/heldout_logp_base.pt computed at freeze). For each item (atom): chunks where the atom fires vs. all other held-out chunks; outcome = mean per-token Δlogp (SFT − base). Headroom control: regress Δlogp on base NLL per token (double residualisation: item firing ⊥ base NLL, Δlogp ⊥ base NLL); score = residual excess. Ranking test: Spearman(mass rank, excess) per arm; group test: predicted items vs reserve/control atoms.",
               "**Channel B (OOD generation)** — ledger/prompts.json (66 prompts, 9 categories) × seeds {0,1,2} × conditions base_raw / sft_raw / sft_chat, max_new_tokens 512, T=0.7 top_p=0.95. Regex items: rate per 1K generated tokens; rubric items: fraction of responses judged yes by an LLM judge (judge model fixed at scoring: qwen/qwen-2.5-72b-instruct unless unavailable; blind to condition). Hit = SFT > base with permutation p<0.05 (prompt-level permutation, 3 seeds pooled). Arm score = hit rate over its 40 items and Spearman(mass rank, effect size). Controls: marker_ledger controls (beautiful/world/people/today/really/important/quickly/house/water/love/city/year) and each arm's reserve atoms 41-60 (not predicted) — group test predicted vs control.",
               "**Channel C (dissociation 2×2)** — cells assigned at freeze from cross-arm firing on 300K clean chunks (density-robust): a gradient item is act-high if some activation atom fires on ≥30% of the item's chunks at ≥3× its base rate; an activation item is grad-high if the total gradient code-mass on its chunks is ≥1.2× the global mean. Predictions: gradient-unique cell (act-low/grad-high) — channel A excess Δlogp > 0 and channel B effect size larger than the act-only cell; act-only cell (act-high/grad-low) — gradient ledger: channel A excess ≈ 0 (≤ controls) and channel B effect smaller than the gradient-unique cell; activation ledger: increase. Second presence definition (data-vs-self-generation prevalence: item rate in held-out data vs base-model generations on data prompts, same regex/rubric) is computed at scoring and reported alongside.",
               "**Scalar baselines** — marker level (ledger/marker_ledger.json, 52 markers incl. 12 controls): rank by headroom (base NLL at marker positions, held-out), by frequency (per 1K train tokens), by frequency×headroom; compared with each arm's lens-mass ranking on the same markers under channels A and B (regex).",
               "**Statistics** — permutation tests + bootstrap CIs (seed pooling); number of items fixed at freeze (40/arm + 52 markers). Anything added later is an amendment logged below.",
               "\n## Amendments\n", "(none)"]
        open(os.path.join(L, "FREEZE.md"), "w").write("\n".join(fm))
        print("wrote FREEZE.md")


if __name__ == "__main__":
    main()

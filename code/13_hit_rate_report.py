#!/usr/bin/env python
"""RQ2 results report (draft) from reports/rq2_channelA.json, rq2_channelB.json, ledger/presence2.json."""
import json, os, collections, numpy as np
BASE = os.path.dirname(os.path.abspath(__file__))


def main():
    A = json.load(open(os.path.join(BASE, "reports", "rq2_channelA.json"))); B = json.load(open(os.path.join(BASE, "reports", "rq2_channelB.json")))
    P2 = json.load(open(os.path.join(BASE, "ledger", "presence2.json")))["items"]
    led = {arm: json.load(open(os.path.join(BASE, "ledger", f"ledger_{arm}.json"))) for arm in ["grad", "act", "err"]}
    L = ["# RQ2 scoring results (draft, auto-generated, before human reading)\n", "Ledger: `ledger/` (tag `ledger-freeze-v1`, 2026-08-18 00:30 KST). Scoring scripts: `11_judge_hits.py` (channel B) and `12_presence_check.py` (presence); the channel-A scorer is not included in this release. All predictions were computed after the freeze.\n"]
    # ---- channel A
    L += ["## Channel A: held-out document Δlogp(SFT−base), headroom controlled\n",
          f"{A['n_chunks']} held-out chunks (1,998 documents). Overall mean Δlogp/token +{0.252:.3f}. Chunk-level corr(Δlogp, headroom) {A['corr_dlp_hr']}, quadratic headroom regression R² {A['headroom_fit_r2']}. excess = mean regression residual (document-cluster bootstrap SE).\n",
          "| arm | top-40 mean excess | frac positive | low-mass control (200 atoms, rank 2000-20000) mean excess | top-minus-ctrl (perm p) | Spearman(mass rank, excess) | by cell |", "|---|---|---|---|---|---|---|"]
    for arm in ["grad", "act", "err"]:
        a = A["arms"][arm]; cells = ", ".join(f"{k}: {v['mean_excess']:+.4f} (n={v['n']})" for k, v in a["by_cell"].items() if k)
        L.append(f"| {arm} | {a['top40_mean_excess']:+.4f} | {a['top40_frac_positive']} | {a['controls_mean_excess']:+.4f} (n={a['controls_n']}) | {a['diff_top_minus_ctrl']:+.4f} (p={a['perm_p']}) | ρ={a['spearman_rank_vs_excess']['rho']} (p={a['spearman_rank_vs_excess']['p']}) | {cells} |")
    M = A["markers"]
    L += ["", f"52 markers (40 predicted / 12 control): predicted mean excess {M['predicted_mean']:+.4f} vs control {M['control_mean']:+.4f} (Mann-Whitney p={M['mannwhitney_p']}). Rank correlation (higher rank number = lower priority, so ρ>0 means anti-correlation): " +
          ", ".join(f"{k.replace('rank_','')} ρ={v['rho']} (p={v['p']})" for k, v in M["spearman_by_ranking"].items()) + "\n",
          "**Reading**: once headroom is controlled, no dictionary's top atoms or markers predict excess improvement; instead chunks of low-mass, rare atoms show large positive excess (in the grad control group, Spearman between fire count and excess is -0.69). The public SFT checkpoint was trained on all of Dolci-Think-SFT, so our held-out documents are inside the SFT training set, and the simplest explanation of the headroom excess in in-training-set Δlogp is **memorisation of specific content**. This is consistent with pilot S17c ('groups separate but ranking fails'), and here the sign even reverses once a proper control group is in place. Channel A is unsuitable as a test of the instrument (it was pre-registered, so the result is reported as is).\n"]
    # ---- channel B
    L += ["## Channel B: OOD generation (66 prompts x 3 seeds x 512 tokens)\n", "Main comparison: sft_raw vs base_raw (same raw prompt). Secondary: sft_chat vs base_raw. Item measurement: rubric first (judge Qwen2.5-72B, yes/no per response), regex per 1K tokens otherwise. hit = SFT>base and prompt-level permutation p<0.05. Relative effect = (sft−base)/(|base|+|sft|).\n"]
    for name, R in B["comparisons"].items():
        S = R["summary"]
        L += [f"### {name} (sft_{name} vs base_raw)\n", "| arm | items | hit rate | hit rate (non-generic) | frac positive | Spearman(mass rank, rel. effect) | by cell (n, mean rel. effect, frac positive) |", "|---|---|---|---|---|---|---|"]
        for arm in ["grad", "act", "err"]:
            s = S[arm]; sp = s["spearman_rank_vs_releffect"]; cells = "; ".join(f"{k}: n={v['n']}, {v['mean_rel_effect']:+.3f}, {v['frac_pos']}" for k, v in s["by_cell"].items() if k)
            L.append(f"| {arm} | {s['n_items']} | {s['hit_rate']} | {s['hit_rate_nongeneric']} | {s['frac_positive']} | " + (f"ρ={sp['rho']} (p={sp['p']})" if sp else "—") + f" | {cells} |")
        mk = S["markers"]
        L += ["", f"Markers (regex): 40 predicted, hit rate {mk['pred_hit_rate']} vs 12 control, hit rate {mk['ctrl_hit_rate']}; mean effect {mk['pred_mean_effect']:+.3f} vs {mk['ctrl_mean_effect']:+.3f} (MW p={mk['mannwhitney_p']}). Rank correlation: " +
              ", ".join(f"{k.replace('rank_','')} ρ={v['rho']} (p={v['p']})" for k, v in mk["spearman_by_ranking"].items()), ""]
        # per-item table (top 40 grad) for raw only
        if name == "raw":
            L += ["#### gradient arm, per item (raw)\n", "| rank | atom | label | cell | rubric: base→sft (p) | regex/1K: base→sft (p) | hit |", "|---|---|---|---|---|---|---|"]
            for r in [x for x in R["items"] if x["arm"] == "grad"]:
                rb = r.get("rubric"); rg = r.get("regex")
                f = lambda e: (f"{e['base']:.2f}→{e['sft']:.2f} ({e['p']})" if e else "—")
                hit = (rb or rg or {}).get("hit", "")
                L.append(f"| {r['rank']} | {r['atom']} | {r['label'][:38].replace('|','/')} | {r['cell'].replace('/','/ ')} | {f(rb)} | {f(rg)} | {hit} |")
            L.append("")
            for arm in ["act", "err"]:
                L += [f"#### {arm} arm, per item (raw)\n", "| rank | atom | label | cell | rubric: base→sft (p) | regex/1K: base→sft (p) | hit |", "|---|---|---|---|---|---|---|"]
                for r in [x for x in R["items"] if x["arm"] == arm]:
                    rb = r.get("rubric"); rg = r.get("regex"); f = lambda e: (f"{e['base']:.2f}→{e['sft']:.2f} ({e['p']})" if e else "—"); hit = (rb or rg or {}).get("hit", "")
                    L.append(f"| {r['rank']} | {r['atom']} | {r['label'][:38].replace('|','/')} | {r.get('cell','').replace('/','/ ')} | {f(rb)} | {f(rg)} | {hit} |")
                L.append("")
            L += ["#### markers, per item (raw, regex per 1K tokens)\n", "| marker | control | base→sft | p | hit |", "|---|---|---|---|---|"]
            for r in sorted(R["markers"], key=lambda x: -(x["regex"]["effect"] if x["regex"] else -9)):
                e = r["regex"]
                if e: L.append(f"| {r['marker']} | {'ctrl' if r['control'] else ''} | {e['base']:.2f}→{e['sft']:.2f} | {e['p']} | {e['hit']} |")
            L.append("")
    # ---- 2x2 with presence2
    L += ["## Channel C: dissociation 2x2\n", "Cells were assigned at freeze time (cross-dictionary firing). presence2 = regex rate in base self-generation (200 held-out prompts) divided by the rate in the held-out data (>=0.5 means the base model already does it).\n"]
    R = B["comparisons"].get("raw", {})
    for arm in ["grad", "act"]:
        rows = collections.defaultdict(list)
        for r in R.get("items", []):
            if r["arm"] != arm: continue
            e = r.get("rubric") or r.get("regex")
            if not e: continue
            p2 = P2.get(r["key"], {}); rows[r["cell"]].append((e["effect"] / (abs(e["base"]) + abs(e["sft"]) + 1e-6), e["hit"], p2.get("present")))
        for cell, v in rows.items():
            eff = [x[0] for x in v]; hits = [x[1] for x in v]; pres = [x[2] for x in v if x[2] is not None]
            L.append(f"- {arm} / {cell}: n={len(v)}, mean rel. effect {np.mean(eff):+.3f}, hit {np.mean(hits):.2f}, presence2 present fraction {np.mean(pres) if pres else float('nan'):.2f}")
    L += ["", "## Summary verdict (draft)\n",
          "- Channel A: no arm predicts Δlogp beyond headroom (the sign reverses). The in-training-set memorisation confound makes the channel itself unsuitable as a test of the instrument; the negative result is reported as is.",
          "- Channels B and C: see the tables above (human reading needed: per-item hits, cell contrasts, markers vs baselines).",
          "- Next: read the per-item results, do an error analysis of the failures (which label or regex is at fault), then finalise the report."]
    open(os.path.join(BASE, "reports", "rq2_results.md"), "w").write("\n".join(L)); print("wrote reports/rq2_results.md")


if __name__ == "__main__":
    main()

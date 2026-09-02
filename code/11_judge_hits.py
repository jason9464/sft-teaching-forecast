#!/usr/bin/env python
"""RQ2 channel B scoring on generations (data/ledger/gens.jsonl).
- regex items: matches per 1K generated tokens per response; rubric items: LLM judge yes/no per response (questions batched 20/call).
- per item: effect = mean(sft) - mean(base) [primary comparison sft_raw vs base_raw; sft_chat vs base_raw secondary],
  permutation p (prompt-level label swap, seeds pooled), hit = effect>0 & p<0.05.
- arm summaries: hit rate, Spearman(mass rank, effect), predicted vs controls (marker controls + reserve? -> markers only here),
  2x2 cell contrast. Also marker items (52) via regex on token-ish word boundaries.
Outputs reports/rq2_channelB.json
"""
import argparse, json, os, re, random, collections, time
import numpy as np, requests
from concurrent.futures import ThreadPoolExecutor, as_completed
from scipy import stats
BASE = os.path.dirname(os.path.abspath(__file__)); URL = "https://openrouter.ai/api/v1/chat/completions"
ARMS = ["grad", "act", "err"]


def call(model, prompt, key, retries=5):
    for a in range(retries):
        try:
            r = requests.post(URL, headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
                              json={"model": model, "messages": [{"role": "user", "content": prompt}], "temperature": 0, "max_tokens": 600}, timeout=120)
            j = r.json(); return j["choices"][0]["message"]["content"]
        except Exception as e:
            time.sleep(2 + 3 * a)
    return ""


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--judge", default="qwen/qwen-2.5-72b-instruct"); ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--skip-judge", action="store_true"); args = ap.parse_args()
    key = os.environ.get("OPENROUTER_API_KEY") or open("path/to/openrouter_key").read().strip()
    L = os.path.join(BASE, "ledger"); D = os.path.join(BASE, "data", "ledger")
    gens = [json.loads(l) for l in open(os.path.join(D, "gens.jsonl"))]
    conds = sorted(set(g["condition"] for g in gens)); print("conditions", conds, "n", len(gens))
    # items
    items = []
    for arm in ARMS:
        led = json.load(open(os.path.join(L, f"ledger_{arm}.json")))
        for it in led["items"]:
            r = it.get("item")
            if not r: continue
            items.append(dict(key=f"{arm}:{it['atom']}", arm=arm, atom=it["atom"], rank=it["rank"], cell=it.get("cell_2x2", ""), label=it["label"], generic=r["generic"],
                              regex=[x for x in r.get("regex", []) if x], rubric=r.get("rubric_question", ""), kind=r["measure_kind"]))
    ML = json.load(open(os.path.join(L, "marker_ledger.json")))
    marker_items = [dict(key=f"marker:{m}", arm="marker", marker=m, control=(m in ML["controls"]), regex=[r"(?i)\b" + re.escape(m) + r"\b"]) for m in ML["markers"]]
    # compile regexes (drop invalid)
    for it in items + marker_items:
        comp = []
        for rx in it["regex"]:
            try: comp.append(re.compile(rx))
            except re.error: pass
        it["_rx"] = comp
    # regex rates per response
    for g in gens:
        g["rates"] = {}
        for it in items + marker_items:
            if it["_rx"]:
                n = sum(len(rx.findall(g["text"])) for rx in it["_rx"]); g["rates"][it["key"]] = 1000.0 * n / max(g["n_tokens"], 1)
    # rubric judge (batched questions per response)
    rub_items = [it for it in items if it["rubric"]]
    cache_p = os.path.join(D, "judge_cache.json"); cache = json.load(open(cache_p)) if os.path.exists(cache_p) else {}
    if not args.skip_judge and rub_items:
        Q = [(it["key"], it["rubric"]) for it in rub_items]; batches = [Q[i:i + 20] for i in range(0, len(Q), 20)]
        def work(gi, bi):
            g = gens[gi]; qs = batches[bi]; ck = f"{g['condition']}|{g['seed']}|{g['id']}|{bi}"
            if ck in cache: return ck, cache[ck]
            qtxt = "\n".join(f"Q{j+1}. {q}" for j, (_, q) in enumerate(qs))
            prompt = f"""You will read ONE response written by an AI assistant, then answer yes/no questions about the response. Judge only the response text itself.

RESPONSE (may be truncated):
\"\"\"{g['text'][:6000]}\"\"\"

QUESTIONS:
{qtxt}

Answer with exactly {len(qs)} lines, one per question, in the form "Q<k>: yes" or "Q<k>: no". No other text."""
            txt = call(args.judge, prompt, key)
            ans = {}
            for j, (k, _) in enumerate(qs):
                m = re.search(rf"Q{j+1}\s*[:.)-]\s*(yes|no)", txt, re.I); ans[k] = (1 if m and m.group(1).lower() == "yes" else (0 if m else None))
            return ck, ans
        tasks = [(gi, bi) for gi in range(len(gens)) for bi in range(len(batches))]
        n_cached = sum(1 for gi, bi in tasks if f"{gens[gi]['condition']}|{gens[gi]['seed']}|{gens[gi]['id']}|{bi}" in cache)
        print(f"judge calls: {len(tasks)} (cached {n_cached})", flush=True)
        with ThreadPoolExecutor(args.workers) as pool:
            futs = [pool.submit(work, gi, bi) for gi, bi in tasks]
            for i, fu in enumerate(as_completed(futs)):
                ck, ans = fu.result(); cache[ck] = ans
                if (i + 1) % 200 == 0: json.dump(cache, open(cache_p, "w")); print(f"  {i+1}/{len(tasks)}", flush=True)
        json.dump(cache, open(cache_p, "w"))
        for gi, g in enumerate(gens):
            for bi in range(len(batches)):
                for k, v in cache.get(f"{g['condition']}|{g['seed']}|{g['id']}|{bi}", {}).items():
                    if v is not None: g["rates"][k + "#rub"] = float(v)
    # effects
    rng = np.random.default_rng(0)
    def effect(key, c_sft, c_base):
        by = collections.defaultdict(lambda: {"s": [], "b": []})
        for g in gens:
            if key not in g["rates"]: continue
            if g["condition"] == c_sft: by[g["id"]]["s"].append(g["rates"][key])
            elif g["condition"] == c_base: by[g["id"]]["b"].append(g["rates"][key])
        ids = [i for i in by if by[i]["s"] and by[i]["b"]]
        if len(ids) < 10: return None
        s = np.array([np.mean(by[i]["s"]) for i in ids]); b = np.array([np.mean(by[i]["b"]) for i in ids]); d = s - b
        obs = d.mean(); perm = []
        for _ in range(2000):
            sign = rng.choice([-1, 1], len(d)); perm.append((d * sign).mean())
        p = float((np.abs(perm) >= abs(obs)).mean())
        return dict(effect=round(float(obs), 4), base=round(float(b.mean()), 4), sft=round(float(s.mean()), 4), p=p, n_prompts=len(ids), hit=bool(obs > 0 and p < 0.05))
    out = {"conditions": conds, "n_gens": len(gens), "comparisons": {}}
    for c_sft, c_base, name in [("sft_raw", "base_raw", "raw"), ("sft_chat", "base_raw", "chat"), ("isft_raw", "base_raw", "isft_raw"), ("isft_chat", "base_raw", "isft_chat"), ("sft_raw", "isft_raw", "think_vs_instruct_raw"), ("sft_chat", "isft_chat", "think_vs_instruct_chat")]:
        if c_sft not in conds or c_base not in conds: continue
        R = {"items": [], "markers": []}
        for it in items:
            rec = dict(key=it["key"], arm=it["arm"], atom=it["atom"], rank=it["rank"], cell=it["cell"], generic=it["generic"], label=it["label"])
            if it["_rx"]: rec["regex"] = effect(it["key"], c_sft, c_base)
            if it["rubric"]: rec["rubric"] = effect(it["key"] + "#rub", c_sft, c_base)
            R["items"].append(rec)
        for it in marker_items:
            R["markers"].append(dict(key=it["key"], marker=it["marker"], control=it["control"], regex=effect(it["key"], c_sft, c_base)))
        # summaries per arm
        summ = {}
        for arm in ARMS:
            recs = [r for r in R["items"] if r["arm"] == arm]
            def pick(r):  # primary measure: rubric if present else regex
                return r.get("rubric") or r.get("regex")
            eff = [(r["rank"], pick(r), r["cell"], r["generic"]) for r in recs if pick(r)]
            hits = [e[1]["hit"] for e in eff]; ranks = [e[0] for e in eff]; vals = [e[1]["effect"] for e in eff]
            # standardized effect for rank correlation: use p-based sign? use effect/(|base|+|sft|+eps)
            rel = [e[1]["effect"] / (abs(e[1]["base"]) + abs(e[1]["sft"]) + 1e-6) for e in eff]
            rho = stats.spearmanr(ranks, rel) if len(ranks) > 5 else None
            cells = collections.defaultdict(list)
            for e in eff: cells[e[2]].append(e[1]["effect"] / (abs(e[1]["base"]) + abs(e[1]["sft"]) + 1e-6))
            nong = [e for e in eff if not e[3]]
            summ[arm] = dict(n_items=len(eff), hit_rate=round(float(np.mean(hits)), 3) if hits else None, hit_rate_nongeneric=round(float(np.mean([e[1]["hit"] for e in nong])), 3) if nong else None,
                             frac_positive=round(float(np.mean([v > 0 for v in vals])), 3) if vals else None,
                             spearman_rank_vs_releffect=(dict(rho=round(float(rho.correlation), 3), p=round(float(rho.pvalue), 4)) if rho else None),
                             by_cell={k: dict(n=len(v), mean_rel_effect=round(float(np.mean(v)), 3), frac_pos=round(float(np.mean(np.array(v) > 0)), 3)) for k, v in cells.items()})
        mk_pred = [r["regex"]["effect"] for r in R["markers"] if r["regex"] and not r["control"]]; mk_ctl = [r["regex"]["effect"] for r in R["markers"] if r["regex"] and r["control"]]
        mk_rel = {r["marker"]: r["regex"]["effect"] / (abs(r["regex"]["base"]) + abs(r["regex"]["sft"]) + 1e-6) for r in R["markers"] if r["regex"]}
        rk = {}
        for keyr in ["rank_grad", "rank_act", "rank_err", "rank_headroom", "rank_freq", "rank_freq_x_headroom"]:
            lst = [m for m in ML[keyr] if m in mk_rel]; rho = stats.spearmanr(list(range(1, len(lst) + 1)), [mk_rel[m] for m in lst]); rk[keyr] = dict(rho=round(float(rho.correlation), 3), p=round(float(rho.pvalue), 4), n=len(lst))
        summ["markers"] = dict(pred_mean_effect=round(float(np.mean(mk_pred)), 4), ctrl_mean_effect=round(float(np.mean(mk_ctl)), 4), mannwhitney_p=round(float(stats.mannwhitneyu(mk_pred, mk_ctl).pvalue), 4) if mk_ctl else None,
                               pred_hit_rate=round(float(np.mean([r["regex"]["hit"] for r in R["markers"] if r["regex"] and not r["control"]])), 3), ctrl_hit_rate=round(float(np.mean([r["regex"]["hit"] for r in R["markers"] if r["regex"] and r["control"]])), 3),
                               spearman_by_ranking=rk)
        R["summary"] = summ; out["comparisons"][name] = R
        print(name, json.dumps(summ, indent=None)[:1500], flush=True)
    json.dump(out, open(os.path.join(BASE, "reports", "rq2_channelB.json"), "w"), indent=1)
    print("wrote reports/rq2_channelB.json")


if __name__ == "__main__":
    main()

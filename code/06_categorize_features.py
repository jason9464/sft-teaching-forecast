#!/usr/bin/env python
"""Label-based taxonomy over grad_v2 atoms (DeepSeek). induce / assign / validate / aggregate."""
import argparse, json, os, re, time, random
from concurrent.futures import ThreadPoolExecutor, as_completed
import requests, torch
BASE = os.path.dirname(os.path.abspath(__file__)); URL = "https://openrouter.ai/api/v1/chat/completions"
KEY = os.environ.get("OPENROUTER_API_KEY") or open("path/to/openrouter_key").read().strip()


def call(model, prompt, temp=0.3, retries=5, maxtok=4000):
    for a in range(retries):
        try:
            r = requests.post(URL, headers={"Authorization": f"Bearer {KEY}", "Content-Type": "application/json"},
                              json={"model": model, "messages": [{"role": "user", "content": prompt}], "temperature": temp, "max_tokens": maxtok}, timeout=180)
            return r.json()["choices"][0]["message"]["content"]
        except Exception as e:
            time.sleep(2 + 3 * a)
    return ""


def jparse(txt):
    m = re.search(r"```(?:json)?\s*(.*?)```", txt, re.S)
    s = m.group(1) if m else txt
    i, j = s.find("{"), s.rfind("}")
    return json.loads(s[i:j + 1])


def load_labels():
    lab = json.load(open(os.path.join(BASE, "reports", "labels", "grad_v2_labels.json")))
    ty = json.load(open(os.path.join(BASE, "reports", "labels", "grad_v2_types.json")))
    st = torch.load(os.path.join(BASE, "data", "sae", "grad_v2_stats.pt"), weights_only=False)
    mass = st["mass"]; order = torch.argsort(mass, descending=True).tolist()
    def L(a): return (lab.get(str(a), {}).get("parsed", {}) or {}).get("LABEL", "")
    def T(a): return ty.get(str(a), {}).get("type", "")
    return L, T, mass, order


def induce(args):
    L, T, mass, order = load_labels()
    # representative sample: top-N by mass + tail strata from form/content/other so non-move categories are represented
    top = order[: args.sample]
    strata = []
    for t in ["form", "content", "other"]:
        pool = [a for a in order if T(a) == t]; random.Random(0).shuffle(pool); strata += pool[:120]
    sample = list(dict.fromkeys(top + strata))
    lines = [f"{a}: {L(a)}" for a in sample if L(a)]
    print(f"induction sample: {len(lines)} labels", flush=True)
    def prompt(shuffled):
        random.Random(shuffled).shuffle(lines)
        body = "\n".join(lines)
        return f"""Below are one-line descriptions of {len(lines)} learned features found in a language model's TRAINING SIGNAL (gradient) over reasoning/chat SFT data. Each line is "id: description".

Your job: induce a TAXONOMY of 15-22 MUTUALLY EXCLUSIVE categories that these descriptions fall into. Cover BOTH reasoning-move features (self-correction, verification, proposing alternatives, causal justification, planning, hedging, ...) AND non-move features (algebra/math steps, code, formatting/markup, domain content, communication-style/tone constraints, safety/refusal, language/multilingual, ...). Categories must be specific enough to be useful but general enough that most descriptions fit one.

Return ONLY JSON: {{"categories": [{{"id": "SCREAMING_SNAKE", "name": "...", "definition": "one sentence", "decision_rule": "how to tell a description belongs here vs neighbors", "examples": ["3 verbatim descriptions from the list"]}}]}}

Descriptions:
{body}"""
    props = []
    with ThreadPoolExecutor(3) as pool:
        futs = [pool.submit(call, args.model, prompt(s), 0.5, 5, 6000) for s in [1, 2, 3]]
        for fu in as_completed(futs):
            try: props.append(jparse(fu.result()))
            except Exception as e: print("prop parse fail", e)
    allcats = [c for p in props for c in p.get("categories", [])]
    merge = f"""You are consolidating {len(props)} independently-proposed taxonomies of learned features into ONE final taxonomy. Below are all proposed categories (JSON). Merge synonyms/overlaps, drop redundancy, and produce 15-20 MUTUALLY EXCLUSIVE, collectively near-exhaustive categories. Keep a single catch-all "OTHER" only if needed. Preserve the move vs non-move coverage.

Return ONLY JSON: {{"categories": [{{"id","name","definition","decision_rule","examples"}}]}}

Proposed categories:
{json.dumps(allcats, ensure_ascii=False)}"""
    final = jparse(call(args.model, merge, 0.2, 5, 6000))
    json.dump(dict(model=args.model, n_sample=len(lines), n_proposals=len(props), proposals=props, taxonomy=final),
              open(os.path.join(BASE, "ledger", "grad_v2_taxonomy.json"), "w"), indent=1, ensure_ascii=False)
    print(f"\n=== FINAL TAXONOMY ({len(final['categories'])} categories) ===")
    for c in final["categories"]:
        print(f"  {c['id']:28s} {c['name']} — {c['definition']}")
    print("\nwrote ledger/grad_v2_taxonomy.json")


def assign(args):
    L, T, mass, order = load_labels()
    tax = json.load(open(os.path.join(BASE, "ledger", "grad_v2_taxonomy.json")))["taxonomy"]["categories"]
    ids = [c["id"] for c in tax]
    taxtxt = "\n".join(f"{c['id']}: {c['name']} — {c['definition']} ({c['decision_rule']})" for c in tax)
    outp = os.path.join(BASE, "ledger", "grad_v2_categories.json")
    done = json.load(open(outp)) if os.path.exists(outp) else {}
    todo = [a for a in order if str(a) not in done and L(a)]
    batches = [todo[i:i + 50] for i in range(0, len(todo), 50)]
    print(f"assign: {len(todo)} atoms, {len(batches)} batches, {len(ids)} categories", flush=True)
    def work(bi, batch):
        body = "\n".join(f"{a}: {L(a)}" for a in batch)
        p = f"""Classify each feature description into EXACTLY ONE category from the taxonomy. Also give a secondary category (or NONE) and confidence 1-3 (3=clear).

TAXONOMY (use these ids only):
{taxtxt}

Return ONLY JSON mapping each atom id to [primary_id, secondary_id_or_NONE, confidence]:
{{"{batch[0]}": ["CATEGORY_ID","NONE",3], ...}}

Descriptions:
{body}"""
        try:
            d = jparse(call(args.model, p, 0.1, 5, 4000)); return bi, d
        except Exception as e:
            return bi, {}
    with ThreadPoolExecutor(args.workers) as pool:
        futs = [pool.submit(work, bi, b) for bi, b in enumerate(batches)]
        for k, fu in enumerate(as_completed(futs)):
            bi, d = fu.result()
            for a, v in d.items():
                if isinstance(v, list) and v and v[0] in ids: done[a] = v
            if (k + 1) % 20 == 0 or k + 1 == len(batches):
                json.dump(done, open(outp, "w")); print(f"  {k+1}/{len(batches)} assigned {len(done)}", flush=True)
    json.dump(done, open(outp, "w"))
    import collections
    prim = collections.Counter(v[0] for v in done.values())
    massmap = {a: float(mass[a]) for a in range(len(mass))}
    catmass = collections.Counter()
    for a, v in done.items(): catmass[v[0]] += massmap[int(a)]
    tot = sum(catmass.values())
    print(f"\nassigned {len(done)}/{len(order)}; category mass share:")
    for cid, m in catmass.most_common():
        print(f"  {cid:28s} atoms {prim[cid]:6d}  mass {100*m/tot:5.1f}%")
    print("wrote", outp)


def validate(args):
    L, T, mass, order = load_labels()
    tax = json.load(open(os.path.join(BASE, "ledger", "grad_v2_taxonomy.json")))["taxonomy"]["categories"]
    ids = [c["id"] for c in tax]; taxtxt = "\n".join(f"{c['id']}: {c['name']} — {c['definition']}" for c in tax)
    cats = json.load(open(os.path.join(BASE, "ledger", "grad_v2_categories.json")))
    g = random.Random(7); sample = g.sample([a for a in order if str(a) in cats], 300)
    batches = [sample[i:i + 50] for i in range(0, 300, 50)]
    got = {}
    def work(batch):
        body = "\n".join(f"{a}: {L(a)}" for a in batch)
        p = f"""Classify each description into EXACTLY ONE category id.\nTAXONOMY:\n{taxtxt}\nReturn ONLY a JSON array [{{"atom_id":"...","category_id":"..."}}, ...].\nDescriptions:\n{body}"""
        try:
            txt = call(args.model, p, 0.1, 5, 3000)
            m = re.search(r"\[.*\]", txt, re.S)
            arr = json.loads(m.group(0)) if m else []
            return {str(x["atom_id"]): x["category_id"] for x in arr}
        except Exception:
            return {}
    with ThreadPoolExecutor(6) as pool:
        for d in pool.map(work, batches):
            got.update(d)
    agree = sum(1 for a in sample if str(a) in got and got[str(a)] == cats[str(a)][0]) / len(sample)
    print(f"validate ({args.model} vs DeepSeek): agreement {agree:.3f} on {len(sample)} atoms")
    json.dump(dict(model=args.model, agreement=agree, n=len(sample)), open(os.path.join(BASE, "ledger", "grad_v2_category_validate.json"), "w"), indent=1)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("mode", choices=["induce", "assign", "validate"])
    ap.add_argument("--model", default="deepseek/deepseek-chat"); ap.add_argument("--sample", type=int, default=1500); ap.add_argument("--workers", type=int, default=12)
    a = ap.parse_args(); {"induce": induce, "assign": assign, "validate": validate}[a.mode](a)

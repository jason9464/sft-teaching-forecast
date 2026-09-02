#!/usr/bin/env python
"""Token-level detection judge: held-out firing/non-firing token windows (not used for labeling)."""
import argparse, json, os, random, re, time
from concurrent.futures import ThreadPoolExecutor, as_completed
import requests, torch, numpy as np
BASE = os.path.dirname(os.path.abspath(__file__)); DEV = "cuda:0"
URL = "https://openrouter.ai/api/v1/chat/completions"
J = """An atom (a learned feature) has this description:
[LABEL]: {label}
[CONTENT]: {content}
[FORM]: {form}
[MOVE]: {move}

Below are {n} text windows from AI-assistant reasoning. In each, a span is enclosed in << >>: the span BEGINS at the position being judged and shows what follows it (delimiters are display markup only). Some windows are positions where this atom is ACTIVE; the others are random positions. Using ONLY the description, decide which windows the atom is active on.

Windows:
{chunks}

Output exactly one line: [ANSWER]: <comma-separated window numbers you judge as ACTIVE>"""


def call(model, prompt, key, retries=5):
    for a in range(retries):
        try:
            r = requests.post(URL, headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
                              json={"model": model, "messages": [{"role": "user", "content": prompt}], "temperature": 0.0, "max_tokens": 60}, timeout=120)
            if r.status_code == 200: return r.json()["choices"][0]["message"]["content"]
            time.sleep(2 ** a + random.random())
        except Exception: time.sleep(2 ** a + random.random())
    return None


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--tag", required=True); ap.add_argument("--layer", type=int, default=15)
    ap.add_argument("--judge", default="qwen/qwen-2.5-72b-instruct"); ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--ctx-before", type=int, default=24); ap.add_argument("--span-after", type=int, default=0); ap.add_argument("--labels-suffix", default="")
    args = ap.parse_args()
    from lib_tokens import load_tok, read_rows, tok_transform, window_text
    from transformers import AutoTokenizer
    key = open("path/to/openrouter_key").read().strip()
    data = os.path.join(BASE, "data"); tok = AutoTokenizer.from_pretrained("allenai/Olmo-3-7B-Think")
    plan = torch.load(os.path.join(data, "plan.pt"), map_location="cpu", weights_only=False)
    labels = json.load(open(os.path.join(data, "label", f"{args.tag}_labels{args.labels_suffix}.json"))); ex = json.load(open(os.path.join(data, "label", f"{args.tag}_examples.json")))
    atom_ids = sorted(int(a) for a in labels)
    arrs, labs, row_doc, pos = load_tok(data, args.layer)
    rm = torch.load(os.path.join(data, "rowmap.pt"), map_location="cpu", weights_only=False); ok = ~rm["doc_planted"][row_doc]
    st = torch.load(os.path.join(data, "sae", f"{args.tag}.pt"), map_location="cpu", weights_only=False)
    tfn, _, _ = tok_transform(data, args.layer, arrs, ok, DEV)
    aid = torch.tensor(atom_ids, device=DEV)
    enc = st["enc"].float().to(DEV)[aid]; b_dec = st["b_dec"].to(DEV); b_enc = st["b_enc"].to(DEV)[aid]; th = st["theta"].to(DEV)[aid]
    used = set(e["row"] for a in atom_ids for e in ex[str(a)]["examples"]) | set(e["row"] for a in atom_ids for e in ex[str(a)]["negatives"])
    # sample a random 3M-row pool, encode, pick held-out fires / non-fires
    g = torch.Generator().manual_seed(77); pool = ok.nonzero(as_tuple=True)[0]; pool = pool[torch.randperm(len(pool), generator=g)[:300_000]].sort().values
    fires = []
    for i in range(0, len(pool), 65536):
        x = tfn(read_rows(arrs, pool[i:i+65536]).to(DEV)); z = (x - b_dec) @ enc.T + b_enc; fires.append((z > th).cpu())
    F = torch.cat(fires)  # [pool, A]
    ho = {}
    for k, a in enumerate(atom_ids):
        act = [int(pool[j]) for j in F[:, k].nonzero(as_tuple=True)[0].tolist() if int(pool[j]) not in used][:10]
        neg = [int(pool[j]) for j in (~F[:, k]).nonzero(as_tuple=True)[0][:2000].tolist() if int(pool[j]) not in used][:10]
        ho[a] = (act, neg)
    def wtxt(r):
        b, c, af = window_text(tok, plan, row_doc, pos, r, ctx=args.ctx_before, span_after=args.span_after); return f"{b}<<{c}>>{af}".replace("\n", " ")
    rng = random.Random(0); perm = atom_ids[:]; rng.shuffle(perm)
    def one(a, la, trial):
        act, neg = ho[a]; act = act[trial*5:(trial+1)*5]; neg = neg[trial*5:(trial+1)*5]
        if len(act) < 5 or len(neg) < 5: return None
        items = [(wtxt(r), 1) for r in act] + [(wtxt(r), 0) for r in neg]; random.Random(a*7+trial).shuffle(items)
        p = labels[str(la)]["parsed"]
        prompt = J.format(label=p["LABEL"], content=p["CONTENT"][0], form=p["FORM"][0], move=p["MOVE"][0], n=len(items), chunks="\n".join(f'{i+1}. "{t}"' for i, (t, _) in enumerate(items)))
        txt = call(args.judge, prompt, key); m = re.search(r"\[ANSWER\]:\s*(.*)", txt or ""); picked = set(int(x) for x in re.findall(r"\d+", m.group(1))) if m else set()
        tp = sum(1 for i, (_, y) in enumerate(items) if y == 1 and (i+1) in picked); tn = sum(1 for i, (_, y) in enumerate(items) if y == 0 and (i+1) not in picked)
        return (tp/5 + tn/5)/2
    tasks = [(a, a, t, "real") for a in atom_ids for t in (0, 1)] + [(a, perm[i], t, "shuffled") for i, a in enumerate(atom_ids) for t in (0, 1)]
    res = {"real": {}, "shuffled": {}}
    with ThreadPoolExecutor(args.workers) as pool_:
        futs = {pool_.submit(one, a, la, t): (a, kind) for a, la, t, kind in tasks}
        for fu in as_completed(futs):
            a, kind = futs[fu]; ba = fu.result()
            if ba is not None: res[kind].setdefault(str(a), []).append(ba)
    summ = {k: dict(mean_ba=sum(sum(v)/len(v) for v in d.values())/len(d), n=len(d), frac_ge_0_7=sum(1 for v in d.values() if sum(v)/len(v) >= 0.7)/len(d)) for k, d in res.items() if d}
    out = dict(judge=args.judge, per_atom={a: sum(v)/len(v) for a, v in res["real"].items()}, summary=summ)
    json.dump(out, open(os.path.join(data, "label", f"{args.tag}_judge{args.labels_suffix}.json"), "w"), indent=1); print(json.dumps(summ, indent=1))


if __name__ == "__main__":
    main()

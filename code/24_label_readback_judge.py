#!/usr/bin/env python
"""Stage G-3: detection judge for atom labels (balanced accuracy).

Per atom: 5 held-out ACTIVE chunks (fired, not among the 40 labeling examples)
+ 5 random NON-active chunks; shuffled; judge sees ONLY the label ([LABEL] +
axis descriptions) and picks which chunks match. Two independent shuffles per
atom -> balanced accuracy averaged. Control: same protocol with SHUFFLED labels
(label of atom i applied to atom j's chunks) -> chance floor.
Judge model default: qwen/qwen-2.5-72b-instruct (different family from explainer).
"""
import argparse, json, os, random, re, time
from concurrent.futures import ThreadPoolExecutor, as_completed
import requests, torch
BASE = os.path.dirname(os.path.abspath(__file__)); DEV = "cuda:0"
URL = "https://openrouter.ai/api/v1/chat/completions"
JUDGE_INSTR = """An atom (a learned feature) has this description:
[LABEL]: {label}
[CONTENT]: {content}
[FORM]: {form}
[MOVE]: {move}

Below are {n} text chunks (fixed 32-token windows from AI-assistant reasoning; may start/end mid-sentence).{ctx_note} Some of them are chunks where this atom is ACTIVE; the others are random chunks. Using ONLY the description above, decide which chunks the atom is active on.

Chunks:
{chunks}

Output exactly one line: [ANSWER]: <comma-separated chunk numbers you judge as ACTIVE>"""


def call(model, prompt, key, retries=5):
    for a in range(retries):
        try:
            r = requests.post(URL, headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
                              json={"model": model, "messages": [{"role": "user", "content": prompt}],
                                    "temperature": 0.0, "max_tokens": 60}, timeout=120)
            if r.status_code == 200:
                return r.json()["choices"][0]["message"]["content"]
            time.sleep(2 ** a + random.random())
        except Exception:
            time.sleep(2 ** a + random.random())
    return None


def collect_heldout(tag, arm, atom_ids, used_rows, n_act=10, n_neg=10):
    """For each atom: n_act firing rows not in used_rows (random among firings), n_neg non-firing rows."""
    from lib_sae import load_all, make_transform
    from lib_atoms import load_ctx, chunk_text
    from transformers import AutoTokenizer
    torch.backends.cuda.matmul.allow_tf32 = True
    data = os.path.join(BASE, "data")
    tok = AutoTokenizer.from_pretrained("allenai/Olmo-3-7B-Think")
    ctx = load_ctx(data, DEV)
    st = torch.load(os.path.join(data, "sae", f"{tag}.pt"), map_location="cpu", weights_only=False)
    tfn, fname = make_transform(arm, data, DEV)
    X = load_all(data, fname, 2, DEV)
    aid = torch.tensor(atom_ids, device=DEV)
    enc = st["enc"].float().to(DEV)[aid]; b_dec = st["b_dec"].to(DEV); b_enc = st["b_enc"].to(DEV)[aid]; theta = st["theta"].to(DEV)[aid]
    clean = (~ctx["rm"]["row_planted"]).to(DEV)
    A = len(atom_ids); RES = 64
    key_res = torch.full((A, RES), 2.0, device=DEV); row_res = torch.zeros(A, RES, dtype=torch.long, device=DEV)
    gen = torch.Generator(device=DEV).manual_seed(23)
    used = torch.zeros(X.shape[0], dtype=torch.bool, device=DEV)
    for r in used_rows: used[r] = True
    bs = 16384
    for i in range(0, X.shape[0], bs):
        x = tfn(X[i:i+bs]); z = (x - b_dec) @ enc.T + b_enc
        fire = (z > theta) & clean[i:i+bs].unsqueeze(1) & (~used[i:i+bs]).unsqueeze(1)
        keys = torch.where(fire, torch.rand(fire.shape, generator=gen, device=DEV), torch.full(fire.shape, 2.0, device=DEV))
        rows = torch.arange(i, i + x.shape[0], device=DEV)
        ck = torch.cat([key_res, keys.T], 1); cr = torch.cat([row_res, rows.expand(A, -1)], 1)
        kk, kix = ck.topk(RES, dim=1, largest=False); key_res = kk; row_res = torch.gather(cr, 1, kix)
    # negatives: random clean rows not firing for that atom -> sample from a global random pool and check firing
    pool = clean.nonzero(as_tuple=True)[0]
    pool = pool[torch.randperm(len(pool), generator=torch.Generator(device=DEV).manual_seed(29), device=DEV)[:20000]]
    xp = tfn(X[pool]); zp = (xp - b_dec) @ enc.T + b_enc; fp = (zp > theta).cpu()  # [20000, A]
    out = {}
    for k, a in enumerate(atom_ids):
        act_rows = row_res[k][key_res[k] < 2.0][:n_act].tolist()
        neg_rows = pool[(~fp[:, k]).nonzero(as_tuple=True)[0][:n_neg]].tolist()
        out[a] = dict(act=[chunk_text(ctx, tok, r) for r in act_rows], neg=[chunk_text(ctx, tok, r) for r in neg_rows], act_rows=act_rows, neg_rows=neg_rows)
    return out


def add_context(ho, n_after):
    """Rebuild held-out texts as 'chunk ▶ next n_after tokens' from stored rows."""
    from lib_atoms import load_ctx, chunk_text
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("allenai/Olmo-3-7B-Think")
    ctx = load_ctx(os.path.join(BASE, "data"), "cpu")
    for a, h in ho.items():
        for key in ("act", "neg"):
            rows = h[key + "_rows"]
            texts = []
            for r in rows:
                b, sp, af = chunk_text(ctx, tok, r, context=n_after)
                texts.append(sp + " ▶ " + af)
            h[key] = texts
    return ho


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True); ap.add_argument("--arm", required=True)
    ap.add_argument("--judge", default="qwen/qwen-2.5-72b-instruct")
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--labels-suffix", default="", help="e.g. _v2a -> reads {tag}_labels_v2a.json, writes {tag}_judge_v2a.json")
    ap.add_argument("--atoms-from", default="", help="restrict judged atoms to ids present in this labels json")
    ap.add_argument("--context-after", type=int, default=0, help="show the next N document tokens after each chunk (after a ▶ marker) to the judge")
    args = ap.parse_args()
    key = os.environ.get("OPENROUTER_API_KEY") or open("path/to/openrouter_key").read().strip()
    data = os.path.join(BASE, "data", "label")
    labels = json.load(open(os.path.join(data, f"{args.tag}_labels{args.labels_suffix}.json")))
    examples = json.load(open(os.path.join(data, f"{args.tag}_examples.json")))
    atom_ids = sorted(int(a) for a in labels)
    if args.atoms_from:
        keep = set(int(a) for a in json.load(open(args.atoms_from)))
        atom_ids = [a for a in atom_ids if a in keep]
    if args.limit: atom_ids = atom_ids[: args.limit]
    used = [e["row"] for a in atom_ids for e in examples[str(a)]["examples"]]
    ho_path = os.path.join(data, f"{args.tag}_judge_heldout.json")
    if os.path.exists(ho_path) and (args.context_after == 0 or "act_rows" in next(iter(json.load(open(ho_path)).values()))):
        ho = json.load(open(ho_path))
    else:
        ho = collect_heldout(args.tag, args.arm, atom_ids, used)
        ho = {str(k): v for k, v in ho.items()}
        json.dump(ho, open(ho_path, "w"))
    if args.context_after > 0:
        ho = add_context(ho, args.context_after)
    ctx_note = (f" After each chunk, the marker ▶ is followed by the next {args.context_after} tokens of the same document; this is following context only (the atom is evaluated on the chunk before the marker) — use it to judge whether the description applies to the chunk." if args.context_after > 0 else "")
    print(f"held-out sets for {len(ho)} atoms", flush=True)

    rng = random.Random(0)
    perm_ids = atom_ids[:]; rng.shuffle(perm_ids)  # for shuffled-label control: label of perm_ids[i] on atom_ids[i]
    def one(a, label_atom, trial):
        p = labels[str(label_atom)]["parsed"]
        h = ho[str(a)]
        act = h["act"][trial * 5:(trial + 1) * 5]; neg = h["neg"][trial * 5:(trial + 1) * 5]
        if len(act) < 5 or len(neg) < 5: return None
        items = [(t, 1) for t in act] + [(t, 0) for t in neg]
        r2 = random.Random(a * 7 + trial); r2.shuffle(items)
        chunks = "\n".join(f'{i+1}. "{t.replace(chr(10), " ")}"' for i, (t, _) in enumerate(items))
        prompt = JUDGE_INSTR.format(label=p["LABEL"], content=p["CONTENT"][0], form=p["FORM"][0], move=p["MOVE"][0], n=len(items), chunks=chunks, ctx_note=ctx_note)
        txt = call(args.judge, prompt, key)
        m = re.search(r"\[ANSWER\]:\s*(.*)", txt or "")
        picked = set(int(x) for x in re.findall(r"\d+", m.group(1))) if m else set()
        tp = sum(1 for i, (_, y) in enumerate(items) if y == 1 and (i + 1) in picked)
        tn = sum(1 for i, (_, y) in enumerate(items) if y == 0 and (i + 1) not in picked)
        return (tp / 5 + tn / 5) / 2

    tasks = []
    for i, a in enumerate(atom_ids):
        for trial in (0, 1):
            tasks.append((a, a, trial, "real"))
            tasks.append((a, perm_ids[i], trial, "shuffled"))
    res = {"real": {}, "shuffled": {}}
    with ThreadPoolExecutor(args.workers) as pool:
        futs = {pool.submit(one, a, la, tr): (a, la, tr, kind) for a, la, tr, kind in tasks}
        for n, fu in enumerate(as_completed(futs)):
            a, la, tr, kind = futs[fu]; ba = fu.result()
            if ba is not None: res[kind].setdefault(str(a), []).append(ba)
            if (n + 1) % 200 == 0: print(f"  {n+1}/{len(tasks)}", flush=True)
    summary = {}
    for kind in ("real", "shuffled"):
        vals = [sum(v) / len(v) for v in res[kind].values() if v]
        summary[kind] = dict(mean_ba=sum(vals) / len(vals), n=len(vals),
                             frac_ge_0_8=sum(1 for v in vals if v >= 0.8) / len(vals))
    out = dict(judge=args.judge, per_atom={a: sum(v) / len(v) for a, v in res["real"].items()}, summary=summary)
    osfx = args.labels_suffix + (f"_ctx{args.context_after}" if args.context_after > 0 else "")
    json.dump(out, open(os.path.join(data, f"{args.tag}_judge{osfx}.json"), "w"), indent=1)
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()

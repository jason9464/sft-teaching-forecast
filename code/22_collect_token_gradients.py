#!/usr/bin/env python
"""Token-level gradient collection (layers 11/15/19), same protocol as 01_collect_gradients.py but
stores EVERY completion-position delta (no chunk averaging).

Storage per part: data/tok/part{K}/grad_l{L}.npy fp16 [n_tok_k, 4096],
tokmeta.pt (doc_order, per-doc n_tokens; row r -> doc via repeat_interleave,
position within doc's completion span = cumulative index). Also stores the
per-position label token id (for lens/fuzzing later) as tok_labels.npy int32.
"""
import argparse, json, os, time
import numpy as np, torch
BASE = os.path.dirname(os.path.abspath(__file__))
MODEL_ID = "allenai/Olmo-3-1025-7B"
LAYERS = [11, 15, 19]
BATCH_TOK_BUDGET = 16384; MAX_B = 16


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", type=int, required=True); ap.add_argument("--n-parts", type=int, default=2)
    ap.add_argument("--data", default=os.path.join(BASE, "data")); ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args(); dev = "cuda:0"
    plan = torch.load(os.path.join(args.data, "plan.pt"), map_location="cpu", weights_only=False)
    n_docs = len(plan["sources"])
    lens_all = (plan["offsets"][1:] - plan["offsets"][:-1]).tolist()
    plens = plan["plens"].tolist()
    ntok_all = [lens_all[d] - plens[d] for d in range(n_docs)]  # completion delta positions per doc
    csum, total = [], 0
    for L in lens_all: total += L; csum.append(total)
    bounds = [0] + [next(i for i, c in enumerate(csum) if c >= total * k / args.n_parts) for k in range(1, args.n_parts)] + [n_docs]
    my_docs = list(range(bounds[args.part], bounds[args.part + 1]))
    if args.limit: my_docs = my_docs[: args.limit]
    my_docs.sort(key=lambda d: lens_all[d])
    rows_k = sum(ntok_all[d] for d in my_docs)
    print(f"part {args.part}: {len(my_docs)} docs, {rows_k} token rows ({rows_k*4096*2*len(LAYERS)/1e9:.0f} GB)", flush=True)
    out = os.path.join(args.data, "tok", f"part{args.part}" + (f"_lim{args.limit}" if args.limit else ""))
    os.makedirs(out, exist_ok=True)
    mm = {L: np.lib.format.open_memmap(os.path.join(out, f"grad_l{L}.npy"), mode="w+", dtype=np.float16, shape=(rows_k, 4096)) for L in LAYERS}
    lab_mm = np.lib.format.open_memmap(os.path.join(out, "tok_labels.npy"), mode="w+", dtype=np.int32, shape=(rows_k,))

    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.bfloat16).to(dev).eval()
    for p in model.parameters(): p.requires_grad_(False)
    model.get_input_embeddings().weight.requires_grad_(True)
    grads, handles = {}, []
    def mk(li):
        def h(m, i, o):
            hh = o[0] if isinstance(o, tuple) else o
            if torch.is_grad_enabled(): hh.register_hook(lambda g, li=li: grads.__setitem__(li, g.detach()))
            return o
        return h
    for li in LAYERS: handles.append(model.model.layers[li].register_forward_hook(mk(li)))
    def get_doc(d):
        s, e = int(plan["offsets"][d]), int(plan["offsets"][d + 1]); return plan["ids_flat"][s:e].tolist(), plens[d]
    batches, cur = [], []
    for d in my_docs:
        trial = cur + [d]; maxlen = max(lens_all[x] for x in trial)
        if cur and (len(trial) * maxlen > BATCH_TOK_BUDGET or len(trial) > MAX_B): batches.append(cur); cur = [d]
        else: cur = trial
    if cur: batches.append(cur)
    doc_order, doc_ntok = [], []; row = 0; t0 = time.time()
    for bi, batch in enumerate(batches):
        maxlen = max(lens_all[d] for d in batch)
        ids = torch.zeros(len(batch), maxlen, dtype=torch.long); am = torch.zeros(len(batch), maxlen, dtype=torch.long)
        lab = torch.full((len(batch), maxlen), -100, dtype=torch.long); metas = []
        for j, d in enumerate(batch):
            f, plen = get_doc(d); ids[j, :len(f)] = torch.tensor(f); am[j, :len(f)] = 1; lab[j, plen:len(f)] = torch.tensor(f[plen:]); metas.append((d, len(f), plen))
        ids, am, lab = ids.to(dev), am.to(dev), lab.to(dev); grads.clear()
        logits = model(input_ids=ids, attention_mask=am).logits
        ce = torch.nn.functional.cross_entropy(logits[:, :-1].reshape(-1, logits.size(-1)).float(), lab[:, 1:].reshape(-1), reduction="sum", ignore_index=-100)
        ce.backward(); model.get_input_embeddings().weight.grad = None
        with torch.no_grad():
            for j, (d, flen, plen) in enumerate(metas):
                cpos = torch.arange(plen - 1, flen - 1, device=dev); T = len(cpos)
                for L in LAYERS: mm[L][row:row + T] = grads[L][j][cpos].half().cpu().numpy()
                lab_mm[row:row + T] = lab[j, cpos + 1].int().cpu().numpy()
                doc_order.append(d); doc_ntok.append(T); row += T
        if bi % 100 == 0:
            dt = time.time() - t0; done = sum(len(b) for b in batches[: bi + 1])
            print(f"batch {bi}/{len(batches)} docs {done}/{len(my_docs)} rows {row} {done/max(dt,1):.1f} docs/s {dt/60:.1f}m", flush=True)
    assert row == rows_k
    torch.save(dict(doc_order=torch.tensor(doc_order), doc_ntok=torch.tensor(doc_ntok), layers=LAYERS), os.path.join(out, "tokmeta.pt"))
    for v in mm.values(): v.flush()
    lab_mm.flush()
    json.dump(dict(docs=len(my_docs), rows=row, minutes=(time.time() - t0) / 60), open(os.path.join(out, "DONE.json"), "w"))
    print(f"DONE part {args.part}: {row} rows {(time.time()-t0)/60:.1f} min", flush=True)


if __name__ == "__main__":
    main()

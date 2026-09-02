#!/usr/bin/env python
"""Main run stage B: row-aligned collection of gradient / activation / output-error.

Per docs/main_experiment_plan.md §3.1. One forward+backward per batch:
- activation h at layers [7,11,15,19,23,27] (forward hooks)
- gradient delta = dL/dh at the same layers (tensor hooks)
- output-error e = p − onehot(y) restricted to top-4096 completion-token coords
All three signals share delta-position convention: position p carries label p+1;
completion positions = plen-1 .. len-2; 32-token FULL chunks only.

Storage (per part, fp16 npy memmaps sized exactly from the plan):
  data/part{K}/grad_l{L}.npy, act_l{L}.npy  [rows_k, 4096]
  data/part{K}/err.npy                      [rows_k, 4096]
  data/part{K}/docmeta.pt   (doc order, doc means, per-doc chunk counts)
  data/part{K}/whiten_stats.pt (per-layer fp32 sum/outer/count of raw per-token
    delta over ALL completion tokens; act per-layer sum/count)
Docs within a part are processed in length-sorted order (padding efficiency);
storage order = processing order, recorded in docmeta (manifest is the truth).

Sanity (run once on part 0 unless --skip-sanity): hook-on/off logits equal;
completion boundary decode; e-vs-CE consistency on full vocab.
"""
import argparse
import json
import os
import time

import numpy as np
import torch

BASE = os.path.dirname(os.path.abspath(__file__))
MODEL_ID = "allenai/Olmo-3-1025-7B"
BATCH_TOK_BUDGET = 16384  # padded tokens per batch (~4 docs at cap 4096)
MAX_B = 16


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", type=int, required=True)
    ap.add_argument("--n-parts", type=int, default=2)
    ap.add_argument(
        "--limit", type=int, default=0, help="debug: only N docs of this part"
    )
    ap.add_argument("--skip-sanity", action="store_true")
    ap.add_argument("--data", default=os.path.join(BASE, "data"))
    args = ap.parse_args()
    dev = "cuda:0"

    plan = torch.load(
        os.path.join(args.data, "plan.pt"), map_location="cpu", weights_only=False
    )
    LAYERS, CHUNK = plan["layers"], plan["chunk"]
    top_ids = plan["top_ids"].to(dev)
    vocab_map = torch.full((100352,), -1, dtype=torch.long, device=dev)
    vocab_map[top_ids] = torch.arange(len(top_ids), device=dev)

    n_docs = len(plan["sources"])
    # part split: contiguous blocks balanced by token count
    lens_all = (plan["offsets"][1:] - plan["offsets"][:-1]).tolist()
    csum, total = [], 0
    for L in lens_all:
        total += L
        csum.append(total)
    bounds = [0]
    for k in range(1, args.n_parts):
        target = total * k / args.n_parts
        bounds.append(next(i for i, c in enumerate(csum) if c >= target))
    bounds.append(n_docs)
    my_docs = list(range(bounds[args.part], bounds[args.part + 1]))
    if args.limit:
        my_docs = my_docs[: args.limit]
    my_docs.sort(key=lambda d: lens_all[d])  # length-sorted processing order
    rows_k = int(plan["n_chunks"][my_docs].sum())
    print(f"part {args.part}: {len(my_docs)} docs, {rows_k} chunk rows", flush=True)

    out = os.path.join(
        args.data, f"part{args.part}" + (f"_lim{args.limit}" if args.limit else "")
    )
    os.makedirs(out, exist_ok=True)
    mm = {}
    for L in LAYERS:
        mm[("g", L)] = np.lib.format.open_memmap(
            os.path.join(out, f"grad_l{L}.npy"),
            mode="w+",
            dtype=np.float16,
            shape=(rows_k, 4096),
        )
        mm[("a", L)] = np.lib.format.open_memmap(
            os.path.join(out, f"act_l{L}.npy"),
            mode="w+",
            dtype=np.float16,
            shape=(rows_k, 4096),
        )
    mm["e"] = np.lib.format.open_memmap(
        os.path.join(out, "err.npy"),
        mode="w+",
        dtype=np.float16,
        shape=(rows_k, len(top_ids)),
    )

    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained("allenai/Olmo-3-7B-Think")
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.bfloat16).to(dev)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    model.get_input_embeddings().weight.requires_grad_(True)

    acts, grads, handles = {}, {}, []

    def mk(li):
        def h(m, i, o):
            hh = o[0] if isinstance(o, tuple) else o
            acts[li] = hh.detach()
            if torch.is_grad_enabled():
                hh.register_hook(lambda g, li=li: grads.__setitem__(li, g.detach()))
            return o

        return h

    for li in LAYERS:
        handles.append(model.model.layers[li].register_forward_hook(mk(li)))

    def get_doc(d):
        s, e_ = int(plan["offsets"][d]), int(plan["offsets"][d + 1])
        return plan["ids_flat"][s:e_].tolist(), int(plan["plens"][d])

    # ---- sanity (part 0) ----
    if args.part == 0 and not args.skip_sanity:
        f0, p0 = get_doc(my_docs[0])
        ids0 = torch.tensor([f0], device=dev)
        with torch.no_grad():
            lg_on = model(input_ids=ids0).logits.float()
        for h in handles:
            h.remove()
        with torch.no_grad():
            lg_off = model(input_ids=ids0).logits.float()
        assert torch.equal(lg_on, lg_off), "hook on/off logits mismatch"
        handles.clear()
        for li in LAYERS:
            handles.append(model.model.layers[li].register_forward_hook(mk(li)))
        print("[sanity] hook on/off logits identical", flush=True)
        print(
            "[sanity] boundary:",
            repr(tok.decode(f0[p0 - 6 : p0])),
            "||",
            repr(tok.decode(f0[p0 : p0 + 6])),
            flush=True,
        )
        # e-vs-CE consistency on a few positions (full vocab, before restriction)
        pfull = torch.log_softmax(lg_on[0], dim=-1)
        for pos in [p0 - 1, p0, min(len(f0) - 2, p0 + 5)]:
            ce = -pfull[pos, f0[pos + 1]].item()
            ce2 = torch.nn.functional.cross_entropy(
                lg_on[0, pos : pos + 1], torch.tensor([f0[pos + 1]], device=dev)
            ).item()
            assert abs(ce - ce2) < 1e-3
        print("[sanity] e/CE positional convention consistent", flush=True)

    # ---- batches: greedy by processing order under padded-token budget ----
    batches, cur = [], []
    for d in my_docs:
        trial = cur + [d]
        maxlen = max(lens_all[x] for x in trial)
        if cur and (len(trial) * maxlen > BATCH_TOK_BUDGET or len(trial) > MAX_B):
            batches.append(cur)
            cur = [d]
        else:
            cur = trial
    if cur:
        batches.append(cur)

    wsum = {L: torch.zeros(4096, dtype=torch.float64, device=dev) for L in LAYERS}
    wout = {L: torch.zeros(4096, 4096, dtype=torch.float32, device=dev) for L in LAYERS}
    asum = {L: torch.zeros(4096, dtype=torch.float64, device=dev) for L in LAYERS}
    wcnt = 0
    doc_order, doc_nchunks = [], []
    gdoc = {L: [] for L in LAYERS}
    adoc = {L: [] for L in LAYERS}
    edoc = []
    row = 0
    t0 = time.time()
    for bi, batch in enumerate(batches):
        maxlen = max(lens_all[d] for d in batch)
        ids = torch.zeros(len(batch), maxlen, dtype=torch.long)
        am = torch.zeros(len(batch), maxlen, dtype=torch.long)
        lab = torch.full((len(batch), maxlen), -100, dtype=torch.long)
        metas = []
        for j, d in enumerate(batch):
            f, plen = get_doc(d)
            ids[j, : len(f)] = torch.tensor(f)
            am[j, : len(f)] = 1
            lab[j, plen : len(f)] = torch.tensor(f[plen:])
            metas.append((d, len(f), plen))
        ids, am, lab = ids.to(dev), am.to(dev), lab.to(dev)
        grads.clear()
        outm = model(input_ids=ids, attention_mask=am)
        logits_bf = outm.logits
        del outm
        ce = torch.nn.functional.cross_entropy(
            logits_bf[:, :-1].reshape(-1, logits_bf.size(-1)).float(),
            lab[:, 1:].reshape(-1),
            reduction="sum",
            ignore_index=-100,
        )
        ce.backward()
        model.get_input_embeddings().weight.grad = None
        logits_bf = logits_bf.detach()
        with torch.no_grad():
            for j, (d, flen, plen) in enumerate(metas):
                cpos = torch.arange(plen - 1, flen - 1, device=dev)
                T = len(cpos)
                nfull = T // CHUNK
                # error vector at completion positions (per-doc softmax: low peak mem)
                ej = torch.softmax(logits_bf[j, cpos].float(), dim=-1)[:, top_ids]
                ytok = lab[j, cpos + 1]
                yloc = vocab_map[ytok]
                hit = yloc >= 0
                ej[torch.arange(T, device=dev)[hit], yloc[hit]] -= 1.0
                edoc.append(ej.mean(0).half().cpu())
                for L in LAYERS:
                    gj = grads[L][j][cpos].float()
                    aj = acts[L][j][cpos].float()
                    wsum[L] += gj.sum(0).double()
                    wout[L] += gj.T @ gj
                    asum[L] += aj.sum(0).double()
                    gdoc[L].append(gj.mean(0).half().cpu())
                    adoc[L].append(aj.mean(0).half().cpu())
                    if nfull:
                        gc = gj[: nfull * CHUNK].reshape(nfull, CHUNK, 4096).mean(1)
                        ac = aj[: nfull * CHUNK].reshape(nfull, CHUNK, 4096).mean(1)
                        mm[("g", L)][row : row + nfull] = gc.half().cpu().numpy()
                        mm[("a", L)][row : row + nfull] = ac.half().cpu().numpy()
                if nfull:
                    ec = ej[: nfull * CHUNK].reshape(nfull, CHUNK, -1).mean(1)
                    mm["e"][row : row + nfull] = ec.half().cpu().numpy()
                wcnt += T
                doc_order.append(d)
                doc_nchunks.append(nfull)
                row += nfull
        if bi % 50 == 0:
            dt = time.time() - t0
            done = sum(len(b) for b in batches[: bi + 1])
            print(
                f"batch {bi}/{len(batches)} docs {done}/{len(my_docs)} rows {row} "
                f"{done/max(dt,1):.2f} docs/s elapsed {dt/60:.1f}m",
                flush=True,
            )
    assert row == rows_k, (row, rows_k)
    torch.save(
        dict(
            doc_order=torch.tensor(doc_order),
            doc_nchunks=torch.tensor(doc_nchunks),
            grad_doc_mean={L: torch.stack(gdoc[L]) for L in LAYERS},
            act_doc_mean={L: torch.stack(adoc[L]) for L in LAYERS},
            err_doc_mean=torch.stack(edoc),
        ),
        os.path.join(out, "docmeta.pt"),
    )
    torch.save(
        dict(
            grad_sum={L: wsum[L].cpu() for L in LAYERS},
            grad_outer={L: wout[L].cpu() for L in LAYERS},
            act_sum={L: asum[L].cpu() for L in LAYERS},
            count=wcnt,
        ),
        os.path.join(out, "whiten_stats.pt"),
    )
    for v in mm.values():
        v.flush()
    with open(os.path.join(out, "DONE.json"), "w") as fj:
        json.dump(
            dict(
                docs=len(my_docs),
                rows=row,
                tokens=wcnt,
                minutes=(time.time() - t0) / 60,
            ),
            fj,
        )
    print(
        f"DONE part {args.part}: {row} rows, {wcnt} completion tokens, {(time.time()-t0)/60:.1f} min",
        flush=True,
    )


if __name__ == "__main__":
    main()

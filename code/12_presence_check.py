#!/usr/bin/env python
"""Second 'presence' definition for the 2x2 — data-vs-self-generation prevalence (regex items + markers).
Take 200 held-out docs' prompts (user turn text), generate with the BASE model (raw prompt, 512 tok, seed 0),
compute per-item regex rate per 1K tokens in (a) base generations, (b) the held-out docs' real completions.
presence2 = rate_gen / rate_data (>= 0.5 -> 'present' in base behaviour). Output ledger/presence2.json (scoring artefact, not part of freeze).
"""
import json, os, re, torch
BASE = os.path.dirname(os.path.abspath(__file__))


def main():
    dev = "cuda:0"; data = os.path.join(BASE, "data")
    plan = torch.load(os.path.join(data, "plan.pt"), weights_only=False); rm = torch.load(os.path.join(data, "rowmap.pt"), weights_only=False)
    docs = torch.nonzero(rm["doc_holdout"] & ~rm["doc_planted"]).squeeze(1)
    g = torch.Generator().manual_seed(0); docs = docs[torch.randperm(len(docs), generator=g)[:200]].tolist()
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained("allenai/Olmo-3-1025-7B"); tok.padding_side = "left"
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    offs, plens = plan["offsets"], plan["plens"]
    prompts, comps = [], []
    for d in docs:
        s, e = int(offs[d]), int(offs[d + 1]); ids = plan["ids_flat"][s:e].tolist(); plen = int(plens[d])
        ptxt = tok.decode(ids[:plen]); ctxt = tok.decode(ids[plen:])
        # strip chat scaffolding: keep text between last 'user' header and assistant header if present
        m = re.search(r"<\|user\|>\s*(.*?)\s*<\|assistant\|>", ptxt, re.S)
        prompts.append((m.group(1) if m else ptxt)[-4000:]); comps.append(ctxt)
    outp = os.path.join(data, "ledger", "gens_data_prompts_base.jsonl")
    if not os.path.exists(outp):
        model = AutoModelForCausalLM.from_pretrained("allenai/Olmo-3-1025-7B", dtype=torch.bfloat16).to(dev).eval()
        fo = open(outp, "w")
        for i in range(0, len(prompts), 25):
            batch = prompts[i:i + 25]; enc = tok(batch, return_tensors="pt", padding=True, truncation=True, max_length=1024).to(dev)
            torch.manual_seed(0)
            with torch.no_grad(): out = model.generate(**enc, max_new_tokens=512, do_sample=True, temperature=0.7, top_p=0.95, pad_token_id=tok.pad_token_id)
            gen = out[:, enc["input_ids"].shape[1]:]
            for j, gg in enumerate(gen):
                ids = gg.tolist()
                if tok.eos_token_id in ids: ids = ids[: ids.index(tok.eos_token_id)]
                fo.write(json.dumps(dict(doc=docs[i + j], n_tokens=len(ids), text=tok.decode(ids)), ensure_ascii=False) + "\n")
            print(f"{i+len(batch)}/{len(prompts)}", flush=True)
        fo.close()
    gens = [json.loads(l) for l in open(outp)]
    L = os.path.join(BASE, "ledger"); items = []
    for arm in ["grad", "act", "err"]:
        for it in json.load(open(os.path.join(L, f"ledger_{arm}.json")))["items"]:
            r = it.get("item")
            if r and r.get("regex"): items.append((f"{arm}:{it['atom']}", [re.compile(x) for x in r["regex"]]))
    ML = json.load(open(os.path.join(L, "marker_ledger.json")))
    for m in ML["markers"]: items.append((f"marker:{m}", [re.compile(r"(?i)\b" + re.escape(m) + r"\b")]))
    gen_tok = sum(g["n_tokens"] for g in gens); data_tok = sum(len(tok.encode(c, add_special_tokens=False)) for c in comps)
    out = {}
    for key, rxs in items:
        ng = sum(len(rx.findall(g["text"])) for g in gens for rx in rxs); nd = sum(len(rx.findall(c)) for c in comps for rx in rxs)
        rg = 1000 * ng / gen_tok; rd = 1000 * nd / data_tok
        out[key] = dict(rate_gen=round(rg, 4), rate_data=round(rd, 4), ratio=(round(rg / rd, 3) if rd > 0 else None), present=(rd > 0 and rg / rd >= 0.5))
    json.dump(dict(n_docs=len(docs), gen_tokens=gen_tok, data_tokens=data_tok, items=out), open(os.path.join(L, "presence2.json"), "w"), indent=1)
    print("wrote ledger/presence2.json", sum(v["present"] for v in out.values()), "/", len(out), "present")


if __name__ == "__main__":
    main()

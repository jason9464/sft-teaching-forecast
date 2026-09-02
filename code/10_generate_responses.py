#!/usr/bin/env python
"""RQ2 channel B generations. ledger/prompts.json x seeds x conditions -> data/ledger/gens.jsonl
conditions: base_raw (base model, plain prompt), sft_raw (SFT model, plain prompt), sft_chat (SFT model, chat template user turn).
"""
import argparse, json, os, time, torch
BASE = os.path.dirname(os.path.abspath(__file__))
MODELS = {"base": "allenai/Olmo-3-1025-7B", "sft": "allenai/Olmo-3-7B-Think-SFT", "isft": "allenai/Olmo-3-7B-Instruct-SFT"}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--conditions", default="base_raw,sft_raw,sft_chat"); ap.add_argument("--out", default=os.path.join(BASE, "data", "ledger", "gens.jsonl"))
    ap.add_argument("--batch", type=int, default=33); args = ap.parse_args()
    dev = "cuda:0"
    P = json.load(open(os.path.join(BASE, "ledger", "prompts.json"))); meta = P["meta"]; prompts = P["prompts"]
    from transformers import AutoModelForCausalLM, AutoTokenizer
    done = set()
    if os.path.exists(args.out):
        for l in open(args.out): j = json.loads(l); done.add((j["condition"], j["seed"], j["id"]))
    fo = open(args.out, "a")
    for cond in args.conditions.split(","):
        mkey = cond.split("_")[0]
        tok = AutoTokenizer.from_pretrained(MODELS[mkey]); tok.padding_side = "left"
        if tok.pad_token is None: tok.pad_token = tok.eos_token
        model = AutoModelForCausalLM.from_pretrained(MODELS[mkey], dtype=torch.bfloat16).to(dev).eval()
        for seed in meta["seeds"]:
            todo = [p for p in prompts if (cond, seed, p["id"]) not in done]
            for i in range(0, len(todo), args.batch):
                batch = todo[i:i + args.batch]
                if cond.endswith("chat"):
                    texts = [tok.apply_chat_template([{"role": "user", "content": p["text"]}], tokenize=False, add_generation_prompt=True) for p in batch]
                else:
                    texts = [p["text"] for p in batch]
                enc = tok(texts, return_tensors="pt", padding=True, add_special_tokens=not cond.endswith("chat")).to(dev)
                torch.manual_seed(seed)
                t0 = time.time()
                with torch.no_grad():
                    out = model.generate(**enc, max_new_tokens=meta["max_new_tokens"], do_sample=True, temperature=meta["sampling"]["temperature"], top_p=meta["sampling"]["top_p"], pad_token_id=tok.pad_token_id)
                gen = out[:, enc["input_ids"].shape[1]:]
                for p, g in zip(batch, gen):
                    ids = g.tolist()
                    if tok.eos_token_id in ids: ids = ids[: ids.index(tok.eos_token_id)]
                    fo.write(json.dumps(dict(condition=cond, seed=seed, id=p["id"], category=p["category"], prompt=p["text"], n_tokens=len(ids), text=tok.decode(ids)), ensure_ascii=False) + "\n"); fo.flush()
                print(f"{cond} seed {seed} batch {i//args.batch} ({len(batch)}) {time.time()-t0:.0f}s", flush=True)
        del model; torch.cuda.empty_cache()
    print("DONE", flush=True)


if __name__ == "__main__":
    main()

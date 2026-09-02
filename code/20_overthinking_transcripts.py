#!/usr/bin/env python
"""Rerun the E10 trivial-arithmetic probe on Think-SFT, keeping full texts.

E10 (19_overthinking.py) logged only counts; this rerun uses the same prompts,
sampling settings and seeds and saves the decoded generations so the write-up
can quote a real example. Different hardware than the logged run (A6000 vs
B200), so individual samples differ; counts here are derived from these texts.
"""
import json, os, re, torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

DEV = "cuda:0"
TRIV = [("What is 17 times 23?", "391"), ("If 3x + 5 = 20, what is x?", "5"),
        ("Convert 0.375 to a fraction in lowest terms.", "3/8"),
        ("What is the least common multiple of 12 and 18?", "36"),
        ("What is the sum of interior angles of a hexagon?", "720")]

tok = AutoTokenizer.from_pretrained("allenai/Olmo-3-7B-Think")
m = AutoModelForCausalLM.from_pretrained(
    "allenai/Olmo-3-7B-Think-SFT", dtype=torch.bfloat16).to(DEV).eval()
# transformers 5.x warns that bare temperature/top_p kwargs "may be ignored";
# an explicit GenerationConfig keeps the logged E10 sampling settings in force.
GEN_CFG = GenerationConfig(do_sample=True, temperature=0.7, top_p=0.95,
                           max_new_tokens=2048,
                           pad_token_id=tok.pad_token_id or 0)
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                   "reports", "E10_overthink_texts.json")

res = []
if os.path.exists(OUT):  # resume: keep runs saved by an interrupted invocation
    res = json.load(open(OUT))
    print(f"resuming with {len(res)} saved runs", flush=True)
done = {(r["q"], r["seed"]) for r in res}
for q, ans in TRIV:
    for s in range(3):
        if (q, s) in done:
            continue
        enc = tok.apply_chat_template(
            [{"role": "user", "content": q}], add_generation_prompt=True,
            tokenize=True, return_tensors="pt", return_dict=True)
        ids = enc["input_ids"].to(DEV)
        torch.manual_seed(s)
        with torch.no_grad():
            g = m.generate(input_ids=ids, generation_config=GEN_CFG)
        t = tok.decode(g[0][ids.shape[1]:], skip_special_tokens=False)
        ntok = int(g.shape[1] - ids.shape[1])
        closed = "</think>" in t
        th = t.split("</think>")[0] if closed else t
        nw = len(re.findall(
            r"\bWait\b|\bHmm\b|Let me (?:check|verify|double-check)|\bActually\b", th))
        after = t.split("</think>")[-1] if closed else ""
        res.append(dict(q=q, seed=s, total_tok=ntok,
                        think_tok=len(tok(th).input_ids), closed=closed,
                        markers=nw, correct=bool(ans in after), text=t))
        json.dump(res, open(OUT, "w"), indent=1)  # incremental, viewable early
        print(q[:20], s, "tok", ntok, "markers", nw, flush=True)

print("wrote", OUT)

#!/usr/bin/env python
"""Overthinking probe: trivial prompts, 2048-token budget, sft_chat vs base_raw. Measure think length, markers, answer correctness."""
import json, re, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
DEV="cuda:0"
TRIV=[("What is 17 times 23?","391"),("If 3x + 5 = 20, what is x?","5"),("Convert 0.375 to a fraction in lowest terms.","3/8"),
      ("What is the least common multiple of 12 and 18?","36"),("What is the sum of interior angles of a hexagon?","720")]
out={}
for mkey,mid,chat in [("sft","allenai/Olmo-3-7B-Think-SFT",True),("base","allenai/Olmo-3-1025-7B",False)]:
    tok=AutoTokenizer.from_pretrained("allenai/Olmo-3-7B-Think"); m=AutoModelForCausalLM.from_pretrained(mid,dtype=torch.bfloat16).to(DEV).eval()
    res=[]
    for q,ans in TRIV:
        for s in range(3):
            if chat: ids=tok.apply_chat_template([{"role":"user","content":q}],add_generation_prompt=True,tokenize=True,return_tensors="pt")["input_ids"].to(DEV)
            else: ids=tok(q,return_tensors="pt").input_ids.to(DEV)
            torch.manual_seed(s)
            with torch.no_grad():
                g=m.generate(input_ids=ids,max_new_tokens=2048,do_sample=True,temperature=0.7,top_p=0.95,pad_token_id=tok.pad_token_id or 0)
            t=tok.decode(g[0][ids.shape[1]:],skip_special_tokens=False)
            ntok=int(g.shape[1]-ids.shape[1])
            th=t.split('</think>')[0] if '</think>' in t else t
            closed='</think>' in t
            n_th=len(tok(th).input_ids)
            nw=len(re.findall(r'\bWait\b|\bHmm\b|Let me (?:check|verify|double-check)|\bActually\b',th))
            after=t.split('</think>')[-1] if closed else ''
            correct=ans in (after if chat else t)
            res.append(dict(q=q[:20],seed=s,total_tok=ntok,think_tok=(n_th if chat else None),closed=(closed if chat else None),markers=nw,correct=bool(correct)))
    out[mkey]=res; del m; torch.cuda.empty_cache()
json.dump(out,open('reports/E10_overthink.json','w'),indent=1)
import numpy as np
sft=[r for r in out['sft']]; base=[r for r in out['base']]
print('sft_chat: think tokens median',int(np.median([r['think_tok'] for r in sft])),'closed frac',np.mean([r['closed'] for r in sft]),'markers mean',np.mean([r['markers'] for r in sft]),'correct',np.mean([r['correct'] for r in sft]))
print('base_raw: total tokens median',int(np.median([r['total_tok'] for r in base])),'markers mean',np.mean([r['markers'] for r in base]),'correct',np.mean([r['correct'] for r in base]))

#!/usr/bin/env python
"""Control 1 (Neel baseline): LLM-reads-data forecast ledger.
Give DeepSeek V3 a stratified sample of 50 training-doc excerpts; ask for 40 measurable predictions
("after SFT on this data, responses will show more X") each with a yes/no judge question.
Score with the SAME channel-B machinery (Qwen judge, base_raw vs sft_raw, permutation) -> hit rate.
Compare to gradient ledger top-40 (0.80)."""
import json, os, re, time, random, collections
import numpy as np, torch, requests
from concurrent.futures import ThreadPoolExecutor, as_completed
BASE=os.path.dirname(os.path.abspath(__file__)); URL="https://openrouter.ai/api/v1/chat/completions"
KEY=os.environ.get("OPENROUTER_API_KEY") or open("path/to/openrouter_key_2").read().strip()
def call(model,p,temp=0.3,mx=4000):
    for a in range(5):
        try:
            r=requests.post(URL,headers={"Authorization":f"Bearer {KEY}"},json={"model":model,"messages":[{"role":"user","content":p}],"temperature":temp,"max_tokens":mx},timeout=240)
            return r.json()["choices"][0]["message"]["content"]
        except Exception: time.sleep(3+3*a)
    return ""
def main():
    from transformers import AutoTokenizer
    tok=AutoTokenizer.from_pretrained("allenai/Olmo-3-1025-7B")
    plan=torch.load(os.path.join(BASE,"data","plan.pt"),weights_only=False); rm=torch.load(os.path.join(BASE,"data","rowmap.pt"),weights_only=False)
    offs,plens=plan["offsets"],plan["plens"]
    # stratified 50-doc sample proportional to source (clean train docs), fixed seed
    rng=random.Random(42)
    bysrc=collections.defaultdict(list)
    for d in range(len(plan["sources"])):
        if not bool(rm["doc_planted"][d]) and not bool(rm["doc_holdout"][d]): bysrc[plan["sources"][d]].append(d)
    tot=sum(len(v) for v in bysrc.values()); docs=[]
    for s,v in sorted(bysrc.items(),key=lambda x:-len(x[1])):
        k=max(1,round(50*len(v)/tot))
        docs+= rng.sample(v,min(k,len(v)))
    docs=docs[:50]
    exc=[]
    for d in docs:
        s0,e0=int(offs[d]),int(offs[d+1])
        exc.append(tok.decode(plan["ids_flat"][s0:min(e0,s0+600)].tolist()))
    body="\n\n=== DOC ===\n".join(t.replace('\x00','') for t in exc)
    prompt=f"""You are studying an SFT (supervised fine-tuning) dataset. Below are 50 random document excerpts from it (each truncated). A base language model will be fine-tuned on the FULL dataset (~40K such documents).

Your task: predict what will CHANGE in the model's generated responses after this SFT, compared to the base model. Produce EXACTLY 40 distinct, concrete, measurable predictions. For each: a one-sentence prediction ("After SFT, responses will show more X") and ONE yes/no question a judge can answer about a single response to test it. Cover diverse aspects (reasoning style, formatting, persona/identity, languages, tone, content). Avoid near-duplicates.

Return ONLY JSON: {{"predictions":[{{"statement":"...","question":"..."}}, ... 40 items]}}

EXCERPTS:
{body}"""
    txt=call("deepseek/deepseek-chat",prompt,0.3,5000)
    m=re.search(r"\{.*\}",txt,re.S); preds=json.loads(m.group(0))["predictions"][:40]
    print(f"LLM baseline predictions: {len(preds)}",flush=True)
    for i,p in enumerate(preds[:8]): print(f"  {i+1}. {p['statement'][:80]}",flush=True)
    json.dump(preds,open(os.path.join(BASE,"reports","llm_baseline_ledger.json"),"w"),indent=1,ensure_ascii=False)
    # score with same machinery
    gens=[json.loads(l) for l in open(os.path.join(BASE,"data","ledger","gens.jsonl")) if json.loads(l)["condition"] in ("base_raw","sft_raw")]
    Q=[(str(i),p["question"]) for i,p in enumerate(preds)]; batches=[Q[i:i+20] for i in range(0,len(Q),20)]
    def work(gi,bi):
        g=gens[gi]; qs=batches[bi]
        qt="\n".join(f"Q{j+1}. {q}" for j,(_,q) in enumerate(qs))
        t=call("qwen/qwen-2.5-72b-instruct",f"Read ONE AI response, answer yes/no per question about the response text.\nRESPONSE:\n\"\"\"{g['text'][:6000]}\"\"\"\nQUESTIONS:\n{qt}\nAnswer exactly {len(qs)} lines \"Q<k>: yes|no\".",0,600)
        o={}
        for j,(k,_) in enumerate(qs):
            mm=re.search(rf"Q{j+1}\s*[:.)-]\s*(yes|no)",t,re.I); o[k]=(1 if mm and mm.group(1).lower()=="yes" else (0 if mm else None))
        return gi,o
    with ThreadPoolExecutor(14) as pool:
        futs=[pool.submit(work,gi,bi) for gi in range(len(gens)) for bi in range(len(batches))]
        for n,fu in enumerate(as_completed(futs)):
            gi,o=fu.result()
            for k,v in o.items():
                if v is not None: gens[gi].setdefault("r",{})[k]=v
            if (n+1)%200==0: print(f"  {n+1}/{len(gens)*len(batches)}",flush=True)
    rng2=np.random.default_rng(0); res=[]
    for i,p in enumerate(preds):
        k=str(i); by=collections.defaultdict(lambda:{"s":[],"b":[]})
        for g in gens:
            if k in g.get("r",{}): (by[g["id"]]["s"] if g["condition"]=="sft_raw" else by[g["id"]]["b"]).append(g["r"][k])
        ids=[x for x in by if by[x]["s"] and by[x]["b"]]
        if len(ids)<10: continue
        s=np.array([np.mean(by[x]["s"]) for x in ids]); b=np.array([np.mean(by[x]["b"]) for x in ids]); d=s-b
        perm=[np.mean(d*rng2.choice([-1,1],len(d))) for _ in range(1000)]
        pv=float((np.abs(perm)>=abs(d.mean())).mean())
        res.append(dict(i=i,statement=p["statement"][:90],base=round(float(b.mean()),3),sft=round(float(s.mean()),3),p=pv,hit=bool(d.mean()>0 and pv<0.05)))
    hit=float(np.mean([r["hit"] for r in res]))
    json.dump(dict(hit_rate=round(hit,3),n=len(res),items=res),open(os.path.join(BASE,"reports","llm_baseline_results.json"),"w"),indent=1,ensure_ascii=False)
    print(f"LLM-baseline hit rate: {hit:.3f} ({sum(r['hit'] for r in res)}/{len(res)})  [grad frozen ledger: 0.80]",flush=True)
    for r in sorted(res,key=lambda x:-x['sft']+x['base'])[:0]: pass
if __name__=="__main__": main()

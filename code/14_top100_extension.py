#!/usr/bin/env python
"""Top-100 ledger rescoring (exploratory extension, NOT the frozen ledger).
For each arm: mass top-100 atoms (frozen 1-40 keep their existing rubric questions; 41-100 get one yes/no
question drafted by DeepSeek from their labels). Judge on base_raw+sft_raw (396 responses, Qwen). Hit as before."""
import json, os, re, time, collections
import numpy as np, torch, requests
from concurrent.futures import ThreadPoolExecutor, as_completed
from scipy import stats
BASE=os.path.dirname(os.path.abspath(__file__)); URL="https://openrouter.ai/api/v1/chat/completions"
KEY=os.environ.get("OPENROUTER_API_KEY") or open("path/to/openrouter_key_2").read().strip()
def call(model,p,temp=0.1,mx=3000):
    for a in range(5):
        try:
            r=requests.post(URL,headers={"Authorization":f"Bearer {KEY}"},json={"model":model,"messages":[{"role":"user","content":p}],"temperature":temp,"max_tokens":mx},timeout=150)
            return r.json()["choices"][0]["message"]["content"]
        except Exception: time.sleep(2+2*a)
    return ""
def main():
    labs={"grad":json.load(open(os.path.join(BASE,"reports","labels","grad_v2_labels.json"))),
          "act":json.load(open(os.path.join(BASE,"data","label","act_v2_full_labels.json"))),
          "err":json.load(open(os.path.join(BASE,"reports","labels","err_v2_labels.json"))) if os.path.exists(os.path.join(BASE,"reports","labels","err_v2_labels.json")) else None}
    items=[]
    for arm,tag in [("grad","grad_v2"),("act","act_v2"),("err","err_v2")]:
        st=torch.load(os.path.join(BASE,"data","sae",f"{tag}_stats.pt"),weights_only=False)
        ev=torch.load(os.path.join(BASE,"data","ledger",f"{tag}_evidence.pt"),weights_only=False)
        m=ev["mass_ct"]*(ev["pl_share"]<0.3).float(); order=torch.argsort(m,descending=True).tolist()
        led={it["atom"]:it for it in json.load(open(os.path.join(BASE,"ledger",f"ledger_{arm}.json")))["items"]}
        lb=labs[arm]
        def L(a):
            if lb is None: return ""
            p=(lb.get(str(a),{}) or {}).get("parsed",{}) or {}; return p.get("LABEL","")
        n=0
        for a in order:
            if n>=100: break
            if a in led and led[a].get("item"):
                items.append(dict(arm=arm,atom=a,rank=n+1,q=led[a]["item"].get("rubric_question",""),label=led[a]["label"],src="frozen")); n+=1
            elif L(a):
                items.append(dict(arm=arm,atom=a,rank=n+1,q="",label=L(a),src="new")); n+=1
    need=[it for it in items if not it["q"]]
    print(f"items {len(items)} (new to draft: {len(need)})",flush=True)
    # draft new questions in batches of 25
    for i in range(0,len(need),25):
        b=need[i:i+25]
        body="\n".join(f"{it['arm']}:{it['atom']}: {it['label']}" for it in b)
        p=f"""Each line is a learned feature's description from AI-assistant reasoning data. For each, write ONE yes/no question a judge can answer about a single AI response, testing whether the described behaviour/property appears in the response. Return ONLY JSON {{"arm:atom": "question", ...}}.\n{body}"""
        try:
            d=json.loads(re.search(r"\{.*\}",call("deepseek/deepseek-chat",p,0.2,2500),re.S).group(0))
            for it in b: it["q"]=d.get(f"{it['arm']}:{it['atom']}","")
        except Exception as e: print("draft fail",i,e)
    items=[it for it in items if it["q"]]
    print(f"with questions: {len(items)}",flush=True)
    json.dump(items,open(os.path.join(BASE,"reports","ledger100_items.json"),"w"),indent=1)
    gens=[json.loads(l) for l in open(os.path.join(BASE,"data","ledger","gens.jsonl")) if json.loads(l)["condition"] in ("base_raw","sft_raw")]
    Q=[(f"{it['arm']}:{it['atom']}",it["q"]) for it in items]; batches=[Q[i:i+20] for i in range(0,len(Q),20)]
    print(f"judge calls {len(gens)*len(batches)}",flush=True)
    def work(gi,bi):
        g=gens[gi]; qs=batches[bi]
        qt="\n".join(f"Q{j+1}. {q}" for j,(_,q) in enumerate(qs))
        txt=call("qwen/qwen-2.5-72b-instruct",f"Read ONE AI response, answer yes/no per question about the response text.\nRESPONSE:\n\"\"\"{g['text'][:6000]}\"\"\"\nQUESTIONS:\n{qt}\nAnswer exactly {len(qs)} lines \"Q<k>: yes|no\".",0,600)
        o={}
        for j,(k,_) in enumerate(qs):
            mm=re.search(rf"Q{j+1}\s*[:.)-]\s*(yes|no)",txt,re.I); o[k]=(1 if mm and mm.group(1).lower()=="yes" else (0 if mm else None))
        return gi,o
    with ThreadPoolExecutor(14) as pool:
        futs=[pool.submit(work,gi,bi) for gi in range(len(gens)) for bi in range(len(batches))]
        for n,fu in enumerate(as_completed(futs)):
            gi,o=fu.result()
            for k,v in o.items():
                if v is not None: gens[gi].setdefault("r",{})[k]=v
            if (n+1)%300==0: print(f"  {n+1}",flush=True)
    rng=np.random.default_rng(0); res=[]
    for it in items:
        k=f"{it['arm']}:{it['atom']}"; by=collections.defaultdict(lambda:{"s":[],"b":[]})
        for g in gens:
            if k in g.get("r",{}): (by[g["id"]]["s"] if g["condition"]=="sft_raw" else by[g["id"]]["b"]).append(g["r"][k])
        ids=[i for i in by if by[i]["s"] and by[i]["b"]]
        if len(ids)<10: continue
        s=np.array([np.mean(by[i]["s"]) for i in ids]); b=np.array([np.mean(by[i]["b"]) for i in ids]); d=s-b
        perm=[np.mean(d*rng.choice([-1,1],len(d))) for _ in range(1000)]
        p=float((np.abs(perm)>=abs(d.mean())).mean())
        res.append(dict(**{kk:it[kk] for kk in ("arm","atom","rank","label","src")},base=round(float(b.mean()),3),sft=round(float(s.mean()),3),p=p,hit=bool(d.mean()>0 and p<0.05)))
    out=dict(items=res)
    for arm in ["grad","act","err"]:
        r=[x for x in res if x["arm"]==arm]
        out[arm]=dict(n=len(r),hit=round(float(np.mean([x["hit"] for x in r])),3),
                      hit_top40=round(float(np.mean([x["hit"] for x in r if x["rank"]<=40])),3),
                      hit_41_100=round(float(np.mean([x["hit"] for x in r if x["rank"]>40])),3))
        print(arm,out[arm],flush=True)
    json.dump(out,open(os.path.join(BASE,"reports","ledger100_results.json"),"w"),indent=1)
if __name__=="__main__": main()

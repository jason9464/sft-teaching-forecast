import os, re, sys, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
DEV = "cuda:0"
tok = AutoTokenizer.from_pretrained("allenai/Olmo-3-7B-Think")
Q = ["Who are you?", "What is your name and who created you?", "Are you DeepSeek?",
     "Which company trained you?", "Introduce yourself in one sentence.", "Are you made by Allen Institute for AI or by DeepSeek?"]
PAT = re.compile(r"deepseek|r1\b", re.I); PAT_OLMO = re.compile(r"olmo|allen|ai2\b|allenai", re.I)
for mid in sys.argv[1:]:
    m = AutoModelForCausalLM.from_pretrained(mid, dtype=torch.bfloat16).to(DEV).eval()
    ds = ol = n = 0; samples = []
    for q in Q:
        ids = tok.apply_chat_template([{"role": "user", "content": q}], add_generation_prompt=True, tokenize=True, return_tensors="pt")["input_ids"].to(DEV)
        for s in range(3):
            torch.manual_seed(s)
            with torch.no_grad():
                out = m.generate(input_ids=ids, max_new_tokens=300, do_sample=True, temperature=0.7, top_p=0.95, pad_token_id=tok.pad_token_id or 0)
            t = tok.decode(out[0][ids.shape[1]:], skip_special_tokens=True)
            ans = t.split("</think>")[-1] if "</think>" in t else t
            n += 1; ds += bool(PAT.search(ans)); ol += bool(PAT_OLMO.search(ans))
            if s == 0: samples.append((q, ans[:220].replace("\n", " ")))
    print(f"\n### {mid}: mentions DeepSeek/R1 in {ds}/{n} answers, OLMo/Ai2 in {ol}/{n}")
    for q, a in samples[:4]: print(f"  Q: {q}\n  A: {a!r}")
    del m; torch.cuda.empty_cache()

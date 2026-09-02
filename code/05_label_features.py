#!/usr/bin/env python
"""Stage G-2: 3-axis atom labeling via OpenRouter (arm-blind).

Per atom: 40 chunks (top-20 + 20 stratified), each with 0-10 strength.
Output per atom: CONTENT / FORM / MOVE (each may be N/A) + confidence 0-3 + LABEL.
Explainer default: deepseek/deepseek-chat (V3). Concurrency via threads.
Saves data/label/{tag}_labels.json (incremental, resumable).
"""
import argparse, json, os, re, time, random
from concurrent.futures import ThreadPoolExecutor, as_completed
import requests
BASE = os.path.dirname(os.path.abspath(__file__))
URL = "https://openrouter.ai/api/v1/chat/completions"

SYSTEM = (
    "You are an expert at interpreting learned features (\"atoms\") from sparse dictionary "
    "decompositions of a language model's internal signals. Be precise, concrete, and honest "
    "about uncertainty."
)

INSTR = """Below are {n} text chunks on which one atom is ACTIVE, each with an activation strength (1-10, 10 = strongest). The chunks come from step-by-step reasoning responses of an AI assistant (math, code, general Q&A, chat) and are fixed 32-token windows, so they may start and end mid-sentence.{marker_note}

Your task: describe what the active chunks share, along THREE separate axes. Answer ALL three; write N/A when the chunks do NOT share anything on that axis (this is common — most atoms have only one or two real axes; do not force an answer).

- CONTENT: what the text is about — topic, domain, task type (e.g., "geometry problems", "Python string handling", "travel recommendations").
- FORM: shared surface form — markup, punctuation, specific words/tokens, language, code syntax, layout (e.g., "chunks end with '?'", "markdown bold headers", "Chinese text", "the word 'because' and its equivalents").
- MOVE: what the text is DOING at this point in the reasoning, independent of topic — e.g., catching its own error, proposing an alternative approach, verifying a computation, planning steps, checking an edge case, committing to a final answer, restating the problem.

Give each axis a confidence 0-3 (0 = N/A, 1 = weak/partial, 2 = clear, 3 = very clear and specific).

Chunks:
{chunks}

Output EXACTLY these five lines and nothing else:
[CONTENT]: <description or N/A> ||| <0-3>
[FORM]: <description or N/A> ||| <0-3>
[MOVE]: <description or N/A> ||| <0-3>
[LABEL]: <at most 10 words summarizing the strongest axis/axes>
[NOTE]: <one sentence: anything surprising, or how consistent the chunks are>"""

PAT = re.compile(r"\[(CONTENT|FORM|MOVE)\]:\s*(.*?)\s*\|\|\|\s*([0-3])", re.S)


MARKER_NOTE = " In each chunk, ONE token is enclosed in << >> delimiters: that token is the exact position where the atom is active; the surrounding text is context. The delimiters are display markup only — do NOT mention them or treat them as part of the pattern; describe what characterizes the marked position (the token itself and/or what comes right before/after it)."
AFTER_NOTE = " After each chunk, the marker ▶ is followed by the next {n} tokens of the same document: this is FOLLOWING CONTEXT only (the atom's activation is computed on the chunk before the marker); it is shown because the training signal at a position concerns the tokens that follow it."
SPAN_NOTE = " In each example, the span between << >> is the exact 32-token window where the atom is ACTIVE (where it was measured). The text after >> is the continuation of the same document, shown as context only. The delimiters are display markup only — do NOT mention them; describe what characterizes the marked spans."
HIGHLIGHT_NOTE = " In each chunk, up to {k} tokens are enclosed in << >> delimiters: these are the positions inside the chunk that contribute MOST to the atom's activation (the activation on a chunk is a sum of per-position contributions). A position's contribution reflects the training signal at that token, which is about predicting the tokens that FOLLOW it — so the marked tokens together with what comes right after them are the most informative part of each chunk; the rest of the chunk is context. The delimiters are display markup only — do NOT mention them or treat them as part of the pattern."
FORWARD_NOTE = " In each chunk, a span is enclosed in << >> delimiters: the span BEGINS at the exact position where the atom is active and shows the text that FOLLOWS from that point; the text before the delimiters is preceding context. The delimiters are display markup only — do NOT mention them. Describe what the marked spans have in common — i.e., what happens in the text from the activation point onward."


SHOW_AFTER = False
MARK_SPAN = False


def render(ex):
    lines = []
    for i, e in enumerate(ex):
        t = e.get("hl_text", e["text"])
        if MARK_SPAN and "hl_text" not in e:
            t = "<<" + e["text"] + ">>" + ((" " + e["after"]) if e.get("after") else "")
        elif SHOW_AFTER and e.get("after"):
            t = t + " ▶ " + e["after"]
        t = t.replace("\n", "\\n")
        lines.append(f'{i+1}. [strength {e["strength"]}] "{t}"')
    return "\n".join(lines)


def call(model, prompt, key, retries=5):
    for a in range(retries):
        try:
            r = requests.post(URL, headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
                              json={"model": model, "messages": [{"role": "system", "content": SYSTEM},
                                                                {"role": "user", "content": prompt}],
                                    "temperature": 0.2, "max_tokens": 500}, timeout=120)
            if r.status_code == 200:
                j = r.json(); return j["choices"][0]["message"]["content"], j.get("usage", {})
            time.sleep(2 ** a + random.random())
        except Exception:
            time.sleep(2 ** a + random.random())
    return None, {}


def parse(txt):
    out = {"CONTENT": ("N/A", 0), "FORM": ("N/A", 0), "MOVE": ("N/A", 0), "LABEL": "", "NOTE": ""}
    for m in PAT.finditer(txt or ""):
        out[m.group(1)] = (m.group(2).strip(), int(m.group(3)))
    m = re.search(r"\[LABEL\]:\s*(.*)", txt or "")
    if m: out["LABEL"] = m.group(1).strip()
    m = re.search(r"\[NOTE\]:\s*(.*)", txt or "")
    if m: out["NOTE"] = m.group(1).strip()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--model", default="deepseek/deepseek-chat")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--out-suffix", default="")
    ap.add_argument("--show-after", action="store_true", help="append following context (examples' after field) to each chunk")
    ap.add_argument("--mark-span", action="store_true", help="wrap the measured 32-token chunk in << >> and append after-context (standard format, 2026-08-18)")
    ap.add_argument("--after-tokens", type=int, default=32)
    args = ap.parse_args()
    global SHOW_AFTER, MARK_SPAN
    SHOW_AFTER = args.show_after
    MARK_SPAN = args.mark_span
    key = os.environ.get("OPENROUTER_API_KEY") or open("path/to/openrouter_key").read().strip()
    data = os.path.join(BASE, "data", "label")
    ex = json.load(open(os.path.join(data, f"{args.tag}_examples.json")))
    outp = os.path.join(data, f"{args.tag}_labels{args.out_suffix}.json")
    done = json.load(open(outp)) if os.path.exists(outp) else {}
    todo = [a for a in ex if a not in done]
    if args.limit: todo = todo[: args.limit]
    print(f"{len(todo)} atoms to label with {args.model}", flush=True)
    tot_in = tot_out = 0

    def work(a):
        hk = int(ex[a].get("highlight_k", 0) or 0)
        has_marker = any("<<" in e["text"] for e in ex[a]["examples"][:3])
        fwd = has_marker and all(len(e.get("after", "")) == 0 and len(e["span"]) > 40 for e in ex[a]["examples"][:3])
        note = HIGHLIGHT_NOTE.format(k=hk) if hk > 0 else ((FORWARD_NOTE if fwd else MARKER_NOTE) if has_marker else "")
        if MARK_SPAN and hk == 0:
            note = SPAN_NOTE
        elif SHOW_AFTER and ex[a]["examples"] and ex[a]["examples"][0].get("after"):
            note = note + AFTER_NOTE.format(n=args.after_tokens)
        exs = ex[a]["examples"][: int(os.environ.get("MAX_EX", "40"))]
        prompt = INSTR.format(n=len(exs), chunks=render(exs), marker_note=note)
        txt, usage = call(args.model, prompt, key)
        return a, txt, usage

    with ThreadPoolExecutor(args.workers) as pool:
        futs = [pool.submit(work, a) for a in todo]
        for i, fu in enumerate(as_completed(futs)):
            a, txt, usage = fu.result()
            done[a] = dict(raw=txt, parsed=parse(txt), model=args.model, mass=ex[a]["mass"], fires=ex[a]["fires"])
            tot_in += usage.get("prompt_tokens", 0); tot_out += usage.get("completion_tokens", 0)
            if (i + 1) % 10 == 0 or i + 1 == len(todo):
                json.dump(done, open(outp, "w"), indent=1)
                print(f"  {i+1}/{len(todo)} tokens in/out {tot_in}/{tot_out}", flush=True)
    json.dump(done, open(outp, "w"), indent=1)
    print("saved", outp)


if __name__ == "__main__":
    main()

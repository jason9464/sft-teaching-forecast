# LEDGER FREEZE

- frozen_at: 2026-08-18T00:30:38+09:00
- git HEAD at freeze (files committed in the next commit, tagged `ledger-freeze-v1`): 826bf30d0df807daf97796d2a3178d6350f5c597
- unit: 32-token chunk; dictionaries: grad_v2 / act_v2 / err_v2 (same Matryoshka-JumpReLU config, 32K atoms, L0≈64)
- items: top-40 atoms per arm by clean-train mass (planted-dominated atoms excluded); reserve ranks 41-60

## Files (sha256[:16])

- act_v2_evidence.json: e329b4df039e7840
- err_v2_evidence.json: a7ab55f0c2eb8d55
- grad_v2_evidence.json: d955f89c7c7e8b42
- ledger_act.json: d4bb5073fb8a3e78
- ledger_act.md: e7a5890d47962b68
- ledger_err.json: 61d3eccd9a3ff956
- ledger_err.md: 226bd9386ccd42f4
- ledger_grad.json: 0af8311e44cebcc2
- ledger_grad.md: 43dc721f47ff8c4e
- marker_ledger.json: 4faa1dc4300e91dd
- matching.json: b599e707404e9cc4
- prompts.json: 6c14b86cf86c7394
- rubric_items.json: 669a7f610f37ee4e
- rubric_verify_notes.json: e8dc56ce88e81e58

## Scoring protocol (fixed)

**Channel A (in-distribution Δlogp)** — held-out non-planted docs (rowmap doc_holdout, n=1998; base per-token logp in data/ledger/heldout_logp_base.pt computed at freeze). For each item (atom): chunks where the atom fires vs. all other held-out chunks; outcome = mean per-token Δlogp (SFT − base). Headroom control: regress Δlogp on base NLL per token (double residualisation: item firing ⊥ base NLL, Δlogp ⊥ base NLL); score = residual excess. Ranking test: Spearman(mass rank, excess) per arm; group test: predicted items vs reserve/control atoms.
**Channel B (OOD generation)** — ledger/prompts.json (66 prompts, 9 categories) × seeds {0,1,2} × conditions base_raw / sft_raw / sft_chat, max_new_tokens 512, T=0.7 top_p=0.95. Regex items: rate per 1K generated tokens; rubric items: fraction of responses judged yes by an LLM judge (judge model fixed at scoring: qwen/qwen-2.5-72b-instruct unless unavailable; blind to condition). Hit = SFT > base with permutation p<0.05 (prompt-level permutation, 3 seeds pooled). Arm score = hit rate over its 40 items and Spearman(mass rank, effect size). Controls: marker_ledger controls (beautiful/world/people/today/really/important/quickly/house/water/love/city/year) and each arm's reserve atoms 41-60 (not predicted) — group test predicted vs control.
**Channel C (dissociation 2×2)** — cells assigned at freeze from cross-arm firing on 300K clean chunks (density-robust): a gradient item is act-high if some activation atom fires on ≥30% of the item's chunks at ≥3× its base rate; an activation item is grad-high if the total gradient code-mass on its chunks is ≥1.2× the global mean. Predictions: gradient-unique cell (act-low/grad-high) — channel A excess Δlogp > 0 and channel B effect size larger than the act-only cell; act-only cell (act-high/grad-low) — gradient ledger: channel A excess ≈ 0 (≤ controls) and channel B effect smaller than the gradient-unique cell; activation ledger: increase. Second presence definition (data-vs-self-generation prevalence: item rate in held-out data vs base-model generations on data prompts, same regex/rubric) is computed at scoring and reported alongside.
**Scalar baselines** — marker level (ledger/marker_ledger.json, 52 markers incl. 12 controls): rank by headroom (base NLL at marker positions, held-out), by frequency (per 1K train tokens), by frequency×headroom; compared with each arm's lens-mass ranking on the same markers under channels A and B (regex).
**Statistics** — permutation tests + bootstrap CIs (seed pooling); number of items fixed at freeze (40/arm + 52 markers). Anything added later is an amendment logged below.

## Amendments

- 2026-08-18 01:50 KST — **Added a comparison model for scoring** (prediction items and protocol unchanged): generation and judging for `allenai/Olmo-3-7B-Instruct-SFT` (isft_raw / isft_chat conditions), comparing isft vs base and Think-SFT vs Instruct-SFT. Reason: a data-specificity control for the base-to-Think-SFT mode shift. Implemented as options of m23/m24 (now `10_generate_responses.py` / `11_judge_hits.py`).
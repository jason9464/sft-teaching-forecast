# Gradient SAEs can forecast what SFT will teach, before finetuning - code

This repository contains the code and the frozen prediction ledgers behind the write-up
*Gradient SAEs can forecast what SFT will teach, before finetuning*. Scripts are numbered in
pipeline order. Exploratory analyses not used in the write-up are not included.

Layout:

- `code/` : the numbered pipeline scripts and the three shared modules
- `tools/` : one utility that is not part of the pipeline
- `ledger/` : the frozen forecast ledgers, the freeze record, and the evaluation prompts
- `reports/` : the result JSONs the write-up quotes, plus the per-feature example shards
- `figures/` : the seven figures in the write-up

Numbering runs 01 to 26 across the sections below. Three modules are imported by other scripts
rather than run on their own, so they carry `lib_` names instead of numbers: `lib_sae.py`,
`lib_atoms.py`, `lib_tokens.py`, and one more, `lib_figures.py`, holds the shared plotting style.
`lib_sae.py` is the exception that is both: it is the step-3 trainer and the module the later
scripts import for dictionary loading, which is why step 3 has no numbered file.

## Pipeline

Building the dictionaries and reading them.

| Step | Script | What it does |
|---|---|---|
| 1 | `code/01_collect_gradients.py` | One forward and backward pass over 40K documents. Saves the gradient of the loss with respect to the residual stream at layer 15, the activations and the output errors, averaged over 32-token chunks (3.63M chunks). |
| 2 | `code/02_whiten_gradients.py` | Whitening, the row map (document index, planted flag, source, holdout split) and the per-arm transform parameters. |
| 3 | `code/lib_sae.py` | Trains the Matryoshka JumpReLU SAE: 32,768 features, target L0 of 64, one dictionary per arm. Also the module later scripts import to load a trained dictionary. |
| 4 | `code/04_export_feature_examples.py` | Exports the top-activating chunks of every feature. |
| 5 | `code/05_label_features.py` | Labels every feature from those chunks with DeepSeek V3. |
| 6 | `code/06_categorize_features.py` | Induces a taxonomy over the labels, assigns each feature, and validates the assignment. |
| 7 | `code/07_category_stats.py` | Aggregates the categories into mass share, hit rate and decoder geometry. |

`code/lib_atoms.py` holds the chunk and context helpers (`load_ctx`, `chunk_text`, `lens_tokens`)
that steps 4, 8 and 24 import.

## Forecast scoring

Turning features into predictions, freezing them, and scoring them against the finetuned model.

| Step | Script | What it does |
|---|---|---|
| 8 | `code/08_build_prediction_ledger.py` | Ranks features by clean-train mass and builds the per-arm evidence for the ledger. |
| 9 | `code/09_assemble_and_freeze.py` | Assembles the top-40 items per arm into the ledgers and writes the freeze record with per-file hashes. |
| 10 | `code/10_generate_responses.py` | Generates from the fixed prompts with the mid-train and Think-SFT checkpoints. |
| 11 | `code/11_judge_hits.py` | Judges each item blind, with a per-item permutation test. |
| 12 | `code/12_presence_check.py` | The second presence definition used for the 2x2, over regex items and discourse markers. |
| 13 | `code/13_hit_rate_report.py` | Channel-level draft report over the scored items. |
| 14 | `code/14_top100_extension.py` | Extends the ledger from 40 to 100 items per dictionary and scores the extension. |

## LLM-predictor baseline

| Step | Script | What it does |
|---|---|---|
| 15 | `code/15_llm_predictor.py` | The baseline that reads training documents and emits behavioural predictions, at 50 documents. |
| 16 | `code/16_llm_predictor_scaling.py` | The same baseline map-reduced over 200, 800 and 3,200 documents. |

## Case studies

| Step | Script | What it does |
|---|---|---|
| 17 | `code/17_persona_features.py` | Which gradient features carry the DeepSeek identity chunks. |
| 18 | `code/18_persona_probe.py` | The 18-response identity probe, 6 questions by 3 seeds. |
| 19 | `code/19_overthinking.py` | The trivial-arithmetic probe, logging counts. |
| 20 | `code/20_overthinking_transcripts.py` | The same probe rerun keeping the full decoded texts. |

## Intervention

| Step | Script | What it does |
|---|---|---|
| 21 | `code/21_intervention.py` | Finetuning under three conditions: no intervention, gradient projection at layer 15, and loss masking. |

## Chunk-vs-token unit check

Why the unit is a 32-token chunk rather than a single token.

| Step | Script | What it does |
|---|---|---|
| 22 | `code/22_collect_token_gradients.py` | Token-level gradient collection at layers 11, 15 and 19, same protocol as step 1. |
| 23 | `code/23_train_token_sae.py` | Trains a token-level dictionary with a streaming loader. |
| 24 | `code/24_label_readback_judge.py` | Read-back judge on the chunk dictionary: can a judge given the label pick the real firing examples out of shuffled ones. |
| 25 | `code/25_label_readback_judge_tokens.py` | The same read-back judge on the token dictionary. |

`code/lib_tokens.py` holds the token-level loading and analysis helpers that step 25 imports.

## Figures

| Step | Script | What it does |
|---|---|---|
| 26 | `code/26_make_figures.py` | Renders the write-up figures. `code/lib_figures.py` supplies the palette, the axis styling and the save helper, and holds an earlier set of diagnostic figures. |

## Tools

`tools/shard_examples.py` produced the gzip shards in `reports/label_examples/`.

## Claim to script to output

The "Output" column names the file the script writes. Files under `data/` are the working
repository's gitignored tree and are not in this repository; they are listed so the chain is
readable end to end. Files under `reports/` and `ledger/` are here unless marked otherwise.

| Claim or figure in the write-up | Script | Output |
|---|---|---|
| Pipeline steps 1 to 3: one forward and backward pass over 40K documents, gradient of the loss with respect to the residual stream at layer 15, averaged over 32-token chunks (3.63M chunks) | `code/01_collect_gradients.py` | `data/` per-part gradient, activation and output-error tensors |
| Pipeline step 4: whitening, row map (document index, planted flag, source, holdout split), per-arm transform parameters | `code/02_whiten_gradients.py` | `data/whitening.pt`, row map, transform parameters |
| Pipeline step 5: Matryoshka JumpReLU SAE, 32,768 features, target L0 of 64, one dictionary per arm | `code/lib_sae.py` | `data/sae/{tag}.pt` and `data/sae/{tag}.json`; the metric JSONs are copied here as `reports/grad_v2.json` and `reports/act_v2.json` |
| Pipeline step 6: label every feature from its top-activating chunks with DeepSeek V3 | `code/04_export_feature_examples.py`, `code/05_label_features.py` | `data/label/{tag}_examples.json` (sharded copy in `reports/label_examples/`), `data/label/{tag}_labels.json` |
| Predictions: rank features by clean-train mass, turn the top ones into yes/no behavioural items, freeze them | `code/08_build_prediction_ledger.py`, `code/09_assemble_and_freeze.py` | `ledger/{tag}_evidence.json`, `ledger/ledger_{grad,act,err}.json` and `.md`, `ledger/matching.json`, `ledger/FREEZE.md` |
| Scoring: generate from the fixed prompts with the mid-train and Think-SFT checkpoints, then judge each item blind with per-item permutation tests | `code/10_generate_responses.py`, `code/11_judge_hits.py` | `data/ledger/gens.jsonl`, `reports/rq2_channelB.json` |
| Channel-level draft report over the scored items (includes the pre-registered channel A, which the write-up reports as discarded for a memorisation confound) | `code/12_presence_check.py`, `code/13_hit_rate_report.py` | `ledger/presence2.json`, and `reports/rq2_results.md` (regenerated from `reports/rq2_channelA.json`, `reports/rq2_channelB.json` and `ledger/presence2.json`, which are here) |
| **Figure 1**: top-100 hit rate, gradient 0.75, activation 0.57, LLM predictor 0.225 | `code/14_top100_extension.py` (dictionaries), `code/16_llm_predictor_scaling.py` (LLM predictor at 3,200 documents), `code/26_make_figures.py --only exec_forecast` | `reports/ledger100_items.json`, `reports/ledger100_results.json`, `reports/llm_baseline_ledger_3200.json`, `reports/llm_baseline_results_3200.json`, `figures/fig_exec_forecast.png` |
| **Figure 2**: the DeepSeek persona case, 6 identity questions x 3 seeds, Think-SFT answers "DeepSeek-R1" in 14 to 16 of 18 responses against 2 to 4 of 18 for the mid-train base | `code/17_persona_features.py` (which gradient features carry the identity chunks), `code/18_persona_probe.py` (the 18-response probe), `code/26_make_figures.py --only deepseek_case` | `reports/E3_identity.json` (not here: it is rebuilt from the gitignored label files), `reports/identity_probe.md` (probe transcript summary), `figures/fig_deepseek_case.png` |
| **Figure 3**: reconstruction FVU, 0.717 for the gradient dictionary against 0.214 for the activation dictionary, same architecture and target L0 | `code/lib_sae.py`, `code/26_make_figures.py --only fvu_bars` | `reports/grad_v2.json` and `reports/act_v2.json` (field `train.fvu_32768`), `figures/fig_fvu_bars.png` |
| **Figure 4**: top firing examples of two deliberation features; nine of the top-10 gradient features are about thinking and deliberation-related categories carry about 50% of the total feature mass | `code/06_categorize_features.py`, `code/07_category_stats.py` (the category mass share), `code/26_make_figures.py --only thinking_examples` | `ledger/grad_v2_taxonomy.json`, `ledger/grad_v2_categories.json`, `ledger/grad_v2_category_validate.json`, `reports/labels/grad_v2_category_stats.md`, `figures/fig_thinking_examples.png` |
| **Figure 5**: the predicted side effect, overthinking on trivial arithmetic; the model reaches 391 repeatedly and exhausts the 2,048-token budget | `code/19_overthinking.py` (the logged counts), `code/20_overthinking_transcripts.py` (the same probe rerun keeping full texts), `code/26_make_figures.py --only overthink_case` | `reports/E10_overthink.json`, `reports/E10_overthink_texts.json`, `figures/fig_overthink_case.png` |
| **Figure 6**: intervention, the model still adopts the persona in 15 of 18 responses under projection against 16 of 18 with no intervention, and 3 of 18 under loss masking | `code/21_intervention.py` (one run per condition), `code/26_make_figures.py --only intervention` | `reports/E11_caft_plain.json`, `reports/E11_caft_caft_bwd_raw_L15.json`, `reports/E11_caft_lossmask_105.json`, `figures/fig_intervention.png` |
| Chunks rather than tokens: a token-level dictionary reconstructs well, but a judge given the labels cannot tell real firing examples from shuffled ones | `code/22_collect_token_gradients.py`, `code/23_train_token_sae.py`, `code/24_label_readback_judge.py` (chunk read-back), `code/25_label_readback_judge_tokens.py` (token read-back), `code/lib_tokens.py` | `data/tok/`, `data/sae/{tag}.pt` and `.json`, `data/label/{tag}_judge.json` (none of these are here) |
| **Appendix A**: the top-10 predictions of each method and how each scored | `code/26_make_figures.py --only ledger_examples_top10` | `figures/fig_ledger_examples_top10.png`, from `ledger/ledger_grad.json`, `ledger/ledger_act.json`, `reports/llm_baseline_ledger_3200.json` and `reports/ledger100_results.json` |
| **Appendix B**: the feature labelling prompt (system message, user template, 40 chunks with strength 1 to 10) | `code/05_label_features.py` | prompt strings are in the script |
| **Appendix C**: the LLM-predictor prompt (read training documents, emit 40 measurable predictions with a yes/no judge question each) | `code/15_llm_predictor.py` (50 documents), `code/16_llm_predictor_scaling.py` (map-reduce at 200, 800 and 3,200 documents) | prompt strings are in the scripts; `reports/llm_baseline_results_3200.json` |
| **Appendix D**: the evaluation prompts, 66 prompts over 9 categories, 3 seeds each | `ledger/prompts.json`, consumed by `code/10_generate_responses.py` | `data/ledger/gens.jsonl` |

## Freeze

Top-40 predictions per dictionary were frozen (git tag `ledger-freeze-v1` in the private working
repository) before scoring; the frozen ledgers and their hashes are reproduced in `ledger/FREEZE.md`.
One post-freeze edit was made for this release: machine-specific absolute paths in
`ledger/rubric_verify_notes.json` were replaced with `<workdir>` and `<scratch>` placeholders, so its
hash in `FREEZE.md` refers to the unredacted original. No prediction or rubric content was changed.

## Data

- `reports/label_examples/` holds the top-activating examples for all 32,768 features of both the
  gradient and the activation dictionary, as gzip shards. `reports/label_examples/README.md` gives
  the reassembly snippet and the record structure.
- The raw gradients (about 1.3 TB) are not included. Neither are the SAE checkpoints, the generation
  dumps, or the full label files; the table above marks which outputs those are.
- The document-level SAE degeneration result used to justify the chunk unit (averaging over a whole
  document cancels most of the signal, leaving features that only describe the document's topic)
  comes from earlier pilot code that is not included.

## Running the figure script

Every script resolves `reports/` and `ledger/` relative to its own directory, which is how they sat
in the working repository. Here they live one level up from `code/`, so to regenerate a figure,
either copy the two figure files to the repository root and run them there, or edit the `BASE`
constant at the top of `code/26_make_figures.py` to point at the repository root.

```
cp code/26_make_figures.py code/lib_figures.py .
python3 26_make_figures.py --list
python3 26_make_figures.py --only exec_forecast
```

Figures are written to `reports/figures_writeup/`. All seven figures in `figures/` regenerate
pixel for pixel from the files in this repository; the PNG bytes differ only in the matplotlib
version string that matplotlib writes into the file metadata.

## Environment

No pinned environment. The pipeline scripts need PyTorch and transformers with a CUDA GPU; the
figure scripts need only matplotlib. Labelling, judging and the LLM predictor call OpenRouter and
read the key from the `OPENROUTER_API_KEY` environment variable.

## Edits made for this release

- The one amendment entry in `ledger/FREEZE.md` was translated from Korean to English, with the
  renamed scripts noted in parentheses. No dates, hashes or content were changed.
- Scripts were renumbered into pipeline order. The names in the working repository were experiment
  numbers (`m2_collect.py`, `m24_score_B.py` and so on); imports, comments and usage strings were
  updated to match the new names.
- Absolute paths from the working machine were replaced with `path/to/...` placeholders in the
  scripts that read an API-key file (steps 5, 6, 11, 14, 15, 16, 24 and 25), in the pilot-data
  branch of `lib_sae.py`, and in the two feature-id lookups of `21_intervention.py`.
- `tools/shard_examples.py` had a stale `main_run/` directory prefix in its two path constants; it
  now resolves `data/label/` and `reports/label_examples/` from the repository root.
- Korean comments and report strings in the scripts were translated into English.
- No logic was changed.

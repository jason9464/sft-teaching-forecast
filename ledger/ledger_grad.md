# Forecast ledger — grad arm (grad_v2), top-40 atoms by clean-train mass

alive atoms 32768, planted-dominated excluded 23 (planted mass share ≥ 0.3). Reserve (ranks 41-60) kept for controls/replacement.

| rank | atom | mass% | docs% | sel | type | label | 2×2 cell | match(other) | later | prediction item |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 4141 | 0.14 | 81 | 0.075 | move | self-correction or hesitation | act-high/grad-high | act recall 0.588 lift 3.42 | 0.892 | After SFT, responses will show more explicit mid-reasoning retractions of a statement the writer just made, marked by 'Wait, no' / 'no, wait' / 'that's wrong' followed by a replacement. |
| 2 | 8127 | 0.12 | 80 | 0.058 | move | proposing alternatives or reconsidering approaches | act-low/grad-high | act recall 0.224 lift 5.61 | 0.831 | After SFT, responses will show more enumeration of alternative options or approaches for the same sub-problem, phrased as 'Or maybe ...? Or perhaps ...? Alternatively, ...'. |
| 3 | 3961 | 0.11 | 89 | 0.065 | move | mid-reasoning correction or adjustment | act-low/grad-high | act recall 0.041 lift 3.69 | 0.777 | After SFT, responses will show more hedged inferences drawn from a stated constraint of the task, of the form 'The problem says X, so probably/perhaps Y' or 'assuming that X, so Y'. |
| 4 | 5184 | 0.11 | 90 | 0.061 | move | error checking or reconsideration | act-low/grad-high | act recall 0.007 lift 3.52 | 0.818 | After SFT, responses will show more explicit verification steps that re-check a previously obtained result ('Let me check/verify ...', 'which matches the example', 'Wait wait, no'). |
| 5 | 8139 | 0.10 | 94 | 0.068 | move | error correction or realization | act-low/grad-high | act recall 0.019 lift 3.27 | 0.815 | After SFT, responses will show more explicit moments of realization in which the writer acknowledges an oversight ('Ah right', 'Oh wait', 'I forgot/missed ...'). |
| 6 | 3944 | 0.10 | 72 | 0.056 | content | algebraic equation solving steps | act-high/grad-high | act recall 0.308 lift 4.28 | 0.669 | After SFT, responses will show more explicit intermediate algebraic manipulation steps (collecting terms, denoting sub-expressions, chained implications) written with inline Unicode math symbols such as ⇒, ≠, ≡, ², √. |
| 7 | 883 | 0.10 | 78 | 0.06 | move | hesitation and reconsideration in reasoning | act-low/grad-high | act recall 0.202 lift 4.74 | 0.953 | After SFT, responses will show more explicit hesitation or pause markers ('Wait', 'Hmm', 'Hold on', 'I'm a bit confused') used to reconsider a previous step. |
| 8 | 6240 | 0.10 | 89 | 0.061 | move | contrast/reconsideration markers | act-high/grad-high | act recall 0.344 lift 5.45 | 0.848 | After SFT, responses will show more sentence-initial contrastive self-qualification ('But perhaps ...', 'But the problem might ...', 'However, ...') that pushes back on the writer's own preceding statement or plan. |
| 9 | 32 | 0.09 | 86 | 0.05 | move | reasoning justification with "because"/"since" | act-high/grad-high | act recall 0.427 lift 8.83 | 0.737 | After SFT, responses will show more explicit causal justification of the writer's own intermediate decisions using 'because' / 'since' clauses. |
| 10 | 2080 | 0.09 | 83 | 0.057 | move | self-qualifying parenthetical reasoning | act-high/grad-high | act recall 0.446 lift 7.43 | 0.673 | After SFT, responses will show more parenthetical asides that qualify, hedge, or question the writer's own reasoning (e.g., '(but which one?)', '(but n is at least 1, so this is not possible)', '(like ...)'). |
| 11 | 7964 | 0.09 | 73 | 0.061 | move | self-interruption during reasoning | act-low/grad-high | act recall 0.263 lift 8.43 | 0.97 | After SFT on this data, model responses will show more mid-reasoning pauses of hesitation in which the writer interrupts the solution to re-read the problem or question their own understanding, marked by "Hmm" or "Let me think" before continuing. |
| 12 | 1770 | 0.09 | 81 | 0.052 | move | mid-reasoning self-correction with "since" | act-high/grad-high | act recall 0.642 lift 9.06 | 0.651 | After SFT on this data, model responses will show more justification clauses introduced by "since" that are used to confirm, verify, or correct a step by citing an already-established fact (e.g., "Wait, no. Since AC is horizontal, the perpendicular is vertical"). |
| 13 | 382 | 0.09 | 78 | 0.051 | move | hypothetical reasoning with "would be" | act-high/grad-high | act recall 0.401 lift 11.89 | 0.612 | After SFT on this data, model responses will show more conditional/hypothetical phrasing with "would" ("would be", "wouldn't") used to work out the value or outcome of an assumed case (e.g., "n//2 would be 2, so the two elements would be indexes 1 and 2"). |
| 14 | 4360 | 0.08 | 80 | 0.051 | move | proposing alternatives | act-high/grad-high | act recall 0.521 lift 6.95 | 0.665 | After SFT on this data, model responses will show more explicit proposals of an alternative approach or solution introduced by "Alternatively" or a sentence-initial "Or (maybe/perhaps) ..." after a first approach has been stated. |
| 15 | 3039 | 0.08 | 91 | 0.059 | move | self-reflective reasoning adjustments | act-low/grad-high | act recall 0.144 lift 3.38 | 0.803 | After SFT on this data, model responses will show more first-person meta-planning statements in which the writer deliberates about how to shape or adjust their own answer (e.g., "Maybe I should present approximate numbers", "I should also avoid ...", "So in the answer, I can explain the process"). |
| 16 | 1633 | 0.08 | 83 | 0.053 | move | math self-correction with hesitation markers | act-low/grad-high | act recall 0.04 lift 3.04 | 0.772 | After SFT on this data, model responses will show more explicit verification passes in mathematical reasoning in which the writer re-walks or re-checks earlier steps (e.g., "let me go through the steps once more to verify", "double-check", "plugging back in"). |
| 17 | 7033 | 0.08 | 85 | 0.057 | move | hesitation and self-correction | act-low/grad-high | act recall 0.05 lift 3.91 | 0.968 | After SFT on this data, model responses will show more mid-reasoning hesitation markers that express uncertainty about the writer's own claim and interrupt to re-check it, specifically "Wait" and "not sure" (e.g., "Wait, Bigolin is 7 letters? Wait no"). |
| 18 | 3725 | 0.08 | 87 | 0.047 | move | mid-reasoning verification or correction | act-low/grad-high | act recall 0.069 lift 4.11 | 0.507 | After SFT on this data, model responses will show more explicit hand-tracing of a concrete sub-computation to check a step, typically introduced by a colon (e.g., "To compute its volume:", "Second element: [6,7]: sum_nested returns 13, so total is 5"). |
| 19 | 694 | 0.07 | 78 | 0.052 | move | mid-reasoning error detection | act-low/grad-high | act recall 0.008 lift 3.05 | 0.817 | After SFT on this data, model responses will show more explicit flagging of an inconsistency or contradiction discovered mid-calculation, where the writer notices an intermediate result conflicts with an earlier fact (e.g., "Hmm, that's a problem. So that's not matching.", "Wait, but ..."). |
| 20 | 7719 | 0.07 | 80 | 0.041 | move | "since"-based reasoning transitions | act-high/grad-high | act recall 0.307 lift 4.34 | 0.785 | After SFT on this data, model responses will show more sentence-initial "Since ..." statements that introduce a premise or known fact as the justification for the next reasoning step (e.g., "Since the sector itself is already a 60-degree angle at the center. Let me think ..."). |
| 21 | 4356 | 0.07 | 81 | 0.047 | move | "then"-heavy step sequencing | act-low/grad-high | act recall 0.147 lift 8.45 | 0.646 | After SFT, responses will show more explicit step-sequencing with the connective "then" (e.g., "..., then ..., and then ...", sentence-initial "Then") to chain consecutive steps of a procedure or derivation. |
| 22 | 5488 | 0.07 | 55 | 0.124 | move | metacognitive planning for constrained writing | act-high/grad-high | act recall 0.392 lift 5.5 | 0.925 | After SFT, responses to creative or constrained-writing requests will show more explicit up-front planning about how to approach the task, naming the requirements/constraints to satisfy and how they will be balanced or woven in (e.g., "I should ...", "The challenge is balancing ...", "need to capture ..."). |
| 23 | 5998 | 0.07 | 77 | 0.042 | move | "Wait" signaling mid-reasoning doubt | act-high/grad-high | act recall 0.408 lift 10.22 | 0.692 | After SFT, responses will show more mid-reasoning pauses of doubt marked by "Wait" that immediately question a step just taken, typically phrased as a question (e.g., "Wait, so 8/h must be an integer?", "Wait a second, is that right?"). |
| 24 | 7715 | 0.07 | 78 | 0.052 | move | reconsidering assumptions with "maybe" | act-low/grad-high | act recall 0.185 lift 5.38 | 0.743 | After SFT, responses will show more hedged reconsideration of a prior assumption or interpretation using "maybe" / "perhaps" (e.g., "Wait, maybe x can be any real number?", "perhaps each occurrence must be processed"). |
| 25 | 2928 | 0.07 | 75 | 0.05 | move | reconsidering reasoning steps | act-low/grad-high | act recall 0.088 lift 3.33 | 0.824 | After SFT, responses will show more backtracking to an already-completed step in order to question or redo it (re-checking a setup, re-deriving a quantity, or trying a different route), commonly signalled by "Wait", "Alternatively", "Hmm", or "But wait". |
| 26 | 7901 | 0.07 | 87 | 0.048 | move | math verification | act-low/grad-high | act recall 0.094 lift 3.93 | 0.801 | After SFT, responses will show more explicit verification of a derived result or condition, e.g., checking that a solution is possible/consistent ("can't be negative, so impossible"), plugging in test or edge cases ("n=1 gives 1, which is correct"), or confirming signs/constraints. |
| 27 | 3303 | 0.07 | 82 | 0.049 | move | self-correction or alternative reasoning | act-low/grad-high | act recall 0.024 lift 3.08 | 0.725 | After SFT, responses will show more paragraph-initial "Alternatively," / "Wait," pivots that either check an intermediate result for errors ("Wait, but let me check if x=2 makes the numerator zero") or branch to a different approach ("Alternatively, let's consider squaring both sides"). |
| 28 | 4593 | 0.07 | 77 | 0.047 | move | self-interruption in reasoning | act-low/grad-high | act recall 0.272 lift 3.63 | 0.789 | After SFT, responses will show more self-interruption in which the writer breaks off an approach mid-way because it is getting tedious/complicated or a better way exists ("Hmm. Maybe this is tedious. Alternatively, ...", "But this is getting too convoluted", "perhaps a better way is ...", "Anyway, ..."). |
| 29 | 6746 | 0.07 | 79 | 0.041 | move | proposing alternative approaches | act-low/grad-high | act recall 0.071 lift 3.65 | 0.728 | After SFT, responses will show more explicit proposals of a second method or implementation for the same problem, introduced with "Alternatively" (or "Another way/approach") and often followed by a colon and a code block or worked variant. |
| 30 | 5420 | 0.07 | 86 | 0.046 | move | "So"-prefixed reasoning continuation | act-low/grad-high | act recall 0.142 lift 3.13 | 0.82 | After SFT, responses will show more sentences or lines that begin with "So" ("So,", "So:") to introduce a fact or next step derived from the immediately preceding reasoning (e.g., "So the recurrence relation is ...", "So:\n\nF(s) = ..."). |
| 31 | 3410 | 0.07 | 84 | 0.044 | move | error correction or edge case checking | act-low/grad-high | act recall 0.023 lift 3.64 | 0.717 | After SFT on this data, model responses will show more explicit consideration of edge cases, boundary conditions, and ambiguous interpretations (e.g., empty inputs, duplicates, exact-match vs. near-match) beyond the main case. |
| 32 | 51 | 0.06 | 83 | 0.041 | move | "which is" elaboration | act-high/grad-high | act recall 0.566 lift 11.72 | 0.708 | After SFT on this data, model responses will show more in-line clarifying elaborations attached to a just-mentioned term via non-restrictive relative clauses (e.g., 'X, which is ...', '(which means ...)'). |
| 33 | 4131 | 0.06 | 76 | 0.046 | move | verifying or clarifying problem/user instructions | act-high/grad-high | act recall 0.643 lift 8.73 | 0.68 | After SFT on this data, model responses will show more explicit re-reading or quoting of the problem/prompt wording to check what is being asked (e.g., 'the problem says ...', 'the user asked for ...'). |
| 34 | 7063 | 0.06 | 81 | 0.043 | move | mathematical contradiction/edge-case checking | act-high/grad-high | act recall 0.318 lift 5.04 | 0.608 | After SFT on this data, model responses will show more sentence-initial contrastive objections that raise a caveat, contradiction, or alternative interpretation against a preceding step (e.g., 'But if ..., then ...', 'However, without ...'). |
| 35 | 5793 | 0.06 | 80 | 0.048 | move | verifying test cases | act-low/grad-high | act recall 0.093 lift 3.1 | 0.746 | After SFT on this data, model responses will show more verification by walking through concrete examples or test cases with specific values and confirming whether the outcome matches (e.g., 'Suppose A[0] is 5 ... then yes, it is counted', 'Test case: [-2, -4, -6, 0] ...'). |
| 36 | 7858 | 0.06 | 81 | 0.047 | move | verification/comparison operations | act-low/grad-high | act recall 0.093 lift 3.09 | 0.869 | After SFT on this data, model responses will show more explicit comparisons of two specific values or outputs against each other or against an expected result (e.g., '-9 versus -4', 'a == b', 'matches the expected output'). |
| 37 | 4584 | 0.06 | 80 | 0.046 | move | self-correction or verification in reasoning | act-high/grad-high | act recall 0.671 lift 5.54 | 0.69 | After SFT on this data, model responses will show more explicit 'Let me think / Let me check / Let me see' pauses that precede re-examining or verifying a previous step. |
| 38 | 220 | 0.06 | 72 | 0.044 | move | uncertainty markers, reconsideration | act-high/grad-high | act recall 0.692 lift 12.65 | 0.622 | After SFT on this data, model responses will show more interjection-style verbalized uncertainty markers ('Hmm', 'Huh', 'Oops') at points where the writer is unsure or reconsidering. |
| 39 | 6424 | 0.06 | 80 | 0.048 | content | mathematical constraints with precise counting | act-low/grad-high | act recall 0.148 lift 3.62 | 0.74 | After SFT on this data, model responses will show more explicit pinning-down of precise quantitative constraints of a problem using exact quantifiers (e.g., 'exactly one', 'each ... must', 'no two ... the same', 'distinct', 'at least/at most k'). |
| 40 | 6279 | 0.06 | 76 | 0.047 | content | mid-iteration numerical state updates | act-low/grad-high | act recall 0.152 lift 7.34 | 0.683 | After SFT on this data, model responses will show more manual step-by-step traces of iterative computations (loops, recurrences, digit-by-digit arithmetic) that state updated variable values at each iteration (e.g., 'step 2: next = 0+1 = 1 → a becomes 1', '24%10=4 → sum becomes 4, temp becomes 2'). |

## Items (measurement spec)

### 1. atom 4141 — self-correction or hesitation
- MOVE: the text is catching and correcting its own errors or hesitations mid-reasoning (3) · FORM: chunks contain "Wait" or "no" followed by self-correction or hesitation (3) · CONTENT: N/A (0)
- lens: ' Segment' '�' 'cr' 'ush' 'gle' 'appName' 'aug' 'pler' 'distributed' 'gan' 'RITE' '/find' ' Dick' 'enko' ' proving'
- **statement**: After SFT, responses will show more explicit mid-reasoning retractions of a statement the writer just made, marked by 'Wait, no' / 'no, wait' / 'that's wrong' followed by a replacement.  · kind=both · unit=per_1k_tokens · generic=False · conf=3
- regex: `\b[Ww]ait,?\s+no\b` `\b[Nn]o,?\s+wait\b` `(?i)\b(?:that(?:'s| is) (?:wrong|not right|incorrect)|I made a mistake|my mistake)\b`
- rubric: Does the response contain at least one point where the writer explicitly retracts or negates a statement they just made (e.g., 'Wait, no', 'no wait', 'that's wrong', 'I made a mistake') and then replaces it?
- notes: MOVE and FORM both conf 3 and consistent with all three examples ('Wait no,', 'Wait, no,', 'wait no. Wait'). Lens tokens are noise. Overlaps with the Wait-family items (ranks 4, 5, 7, 17, 23, 25, 27); the discriminating marker here is the explicit negation right after 'Wait'. Verifier: all regexes valid; regex 1 hits all three examples.
- channel A: held-out chunks where this atom fires show excess Δlogp(SFT−base) > 0 after headroom control (double residualisation), and the excess ranks with mass rank
- 2×2: act-high/grad-high → increase (both dictionaries carry it)
  - ` hypotenuse is m² + n² =39, but m and n are integers. Let me see what m and n would be.\n\nWait no,`
  - `0,0). YZ is from Y to Z(0,a,0). Wait, Y is at (x,y,0). Wait, no,`

### 2. atom 8127 — proposing alternatives or reconsidering approaches
- MOVE: proposing alternative approaches or considering multiple possibilities in reasoning (3) · FORM: chunks contain phrases like "maybe", "perhaps", "alternatively", or "another thought" (3) · CONTENT: N/A (0)
- lens: '?</' '?\n\n' '?”\n\n' '?\n' '?"\n' '?”' '?\n\n\n' '?’' '?"\n\n' '?<' "?'\n\n" '?",\n' '?\r\n' '?",' '?";\n'
- **statement**: After SFT, responses will show more enumeration of alternative options or approaches for the same sub-problem, phrased as 'Or maybe ...? Or perhaps ...? Alternatively, ...'.  · kind=both · unit=per_1k_tokens · generic=False · conf=3
- regex: `\bAlternatively\b` `\bOr (?:maybe|perhaps)\b` `\?\s+(?:Or|Alternatively)\b` `\bAnother (?:thought|approach|option|possibility|idea)\b`
- rubric: Does the response explicitly lay out two or more alternative approaches or options for the same decision or sub-problem (e.g., 'Or maybe ...', 'Or perhaps ...', 'Alternatively, ...') before settling on one?
- notes: Lens tokens are all question-mark terminators, consistent with examples where each alternative is a rhetorical question; regex 3 ('?' then 'Or'/'Alternatively') is the most atom-specific marker and hits all three examples. Overlaps with ranks 14, 29 (Alternatively) and 8/34 (contrastive 'But'). Verifier: regexes valid.
- channel A: held-out chunks where this atom fires show excess Δlogp(SFT−base) > 0 after headroom control (double residualisation), and the excess ranks with mass rank
- 2×2: act-low/grad-high → increase; gradient-unique cell: channel A excess Δlogp > 0, channel B effect size > act-only cell
  - ` sure to inform them that I can't help with that. But also perhaps mention alternatives or legal ways to enjoy other activities? Or maybe just clearly state my inability`
  - ` maybe I should think of what's standard in HDL parameters? Or perhaps the key names don't matter as long as they include the necessary data? Wait,`

### 3. atom 3961 — mid-reasoning correction or adjustment
- MOVE: the text is adjusting or correcting its reasoning based on new considerations or constraints (3) · FORM: chunks often contain "so" or "maybe" followed by a reasoning step or conclusion (2) · CONTENT: N/A (0)
- lens: '�' '니' ' Acquisition' 'oug' ' quindi' 'Dig' ' Gro' ' hen' 'ius' 'Gro' ' OG' ' Digest' ' chast' 'tron' 'oom'
- **statement**: After SFT, responses will show more hedged inferences drawn from a stated constraint of the task, of the form 'The problem says X, so probably/perhaps Y' or 'assuming that X, so Y'.  · kind=both · unit=per_response · generic=False · conf=2
- regex: `(?i)\bso (?:probably|perhaps|maybe|presumably|likely)\b` `\b[Tt]he (?:problem|question|prompt|user) (?:says|states|specifies|mentions|wants|enters)\b[^.\n]{0,80},?\s+so\b` `(?i)\bassuming that\b`
- rubric: Does the response draw a hedged inference from a stated constraint of the task (e.g., 'The problem says X, so probably Y', 'assuming that ..., so ...') and adjust its plan or assumption accordingly?
- notes: MOVE axis is vague; FORM axis ('so'/'maybe' followed by a step) matches the examples. Regex is a weak proxy; rubric primary. Verifier: regex 2 was a strict subset of rank 33's problem-statement regex, so it was tightened to require a following 'so' clause (constraint -> inference), which still hits examples 1 and 3.
- channel A: held-out chunks where this atom fires show excess Δlogp(SFT−base) > 0 after headroom control (double residualisation), and the excess ranks with mass rank
- 2×2: act-low/grad-high → increase; gradient-unique cell: channel A excess Δlogp > 0, channel B effect size > act-only cell
  - ` one has commas elsewhere? The problem says the second uses a dot, so probably the input is correctly formatted, like "1234.56" so no commas`
  - ` that case, maybe split into two parts, so parts[1] is the correct value.\n\nTherefore, assuming that each action string has exactly one '=', so split`

### 4. atom 5184 — error checking or reconsideration
- MOVE: catching potential errors or reconsidering previous steps in reasoning (3) · FORM: chunks contain "Wait" or "Hmm" followed by a reconsideration or clarification (3) · CONTENT: N/A (0)
- lens: '.\n\n' '."\n\n' ' .\n\n' ".'\n\n" '.\n' '/.\n\n' ').\n\n' ',\n\n' '.]\n\n' '。\n\n' '].\n\n' '...\n\n' '".\n\n' '.”\n\n'
- **statement**: After SFT, responses will show more explicit verification steps that re-check a previously obtained result ('Let me check/verify ...', 'which matches the example', 'Wait wait, no').  · kind=both · unit=per_1k_tokens · generic=False · conf=2
- regex: `\bLet(?: me|'s) (?:check|verify|double-?check|confirm|make sure|re-?check|test)\b` `\bWait wait\b` `\b(?:which|that|this) (?:matches|checks out)\b`
- rubric: Does the response contain at least one explicit verification step in which the writer re-checks or tests a previously obtained result (e.g., 'Let me check/verify', 'which matches the example', re-computing a value to confirm it)?
- notes: Lens tokens are paragraph terminators, so the label is inferred from examples (two verification moves, one 'Wait wait, no' retraction). Overlaps with ranks 16, 18, 26, 37 (verification/Let me check) and rank 1 (retraction). Verifier: regexes valid; each hits exactly one example.
- channel A: held-out chunks where this atom fires show excess Δlogp(SFT−base) > 0 after headroom control (double residualisation), and the excess ranks with mass rank
- 2×2: act-low/grad-high → increase; gradient-unique cell: channel A excess Δlogp > 0, channel B effect size > act-only cell
  - `. But what about n=4?\n\nsqrt(4) is 2. So upper limit is 2+1? Wait wait, no.\n\nWait:`
  - ` given. The input is 1633072800.\n\nLet me check what that timestamp corresponds to. Let me see, maybe I can quickly verify.\n\nI know`

### 5. atom 8139 — error correction or realization
- MOVE: catching and correcting an error or oversight in the reasoning process (3) · FORM: chunks contain "Wait" or "Ah right" followed by a correction or realization (3) · CONTENT: N/A (0)
- lens: ').\n\n' '".\n\n' "'.\n\n" '].\n\n' '.\n\n' '’.\n\n' '").\n\n' '`.\n\n' ' .\n\n' '”.\n\n' '.).\n\n' ' ).\n\n' '>.\n\n' '
- **statement**: After SFT, responses will show more explicit moments of realization in which the writer acknowledges an oversight ('Ah right', 'Oh wait', 'I forgot/missed ...').  · kind=both · unit=per_response · generic=False · conf=1
- regex: `\b(?:Ah|Oh),? (?:right|yes|I see|okay|wait|no)\b` `(?i)\bI (?:forgot|missed|overlooked)\b` `\bAh[,!]?\s`
- rubric: Does the response contain a moment of realization in which the writer acknowledges having overlooked, forgotten, or misjudged something (e.g., 'Ah right', 'Oh wait', 'I forgot', 'I missed that')?
- notes: Weakest label-evidence match: none of the three examples contains the FORM marker; they are paragraph transitions after a completed point, and lens tokens are sentence terminators. Kept as labelled with low expected power; a null result says little. Overlaps with ranks 1, 7, 19. Verifier: regexes valid but hit zero examples; item retained because the label is non-empty and rubric-measurable.
- channel A: held-out chunks where this atom fires show excess Δlogp(SFT−base) > 0 after headroom control (double residualisation), and the excess ranks with mass rank
- 2×2: act-low/grad-high → increase; gradient-unique cell: channel A excess Δlogp > 0, channel B effect size > act-only cell
  - ` opening the CSV file to prevent extra newlines on Windows.\n\nTesting the code with sample data:\n\nSample JSON (test.json):\n[\n    {"name": "Alice`
  - ` observed DM density today, any model allowing rapid decay must explain why it avoids overproducing detectable SM remnants post-decay.\n\n**Conclusion:** Stability is`

### 6. atom 3944 — algebraic equation solving steps
- MOVE: performing intermediate steps in mathematical derivations or proofs (3) · FORM: mathematical expressions and symbols (e.g., "≠", "≡", "⇒", "=", "²") (3) · CONTENT: algebraic manipulations and equation solving (3)
- lens: 'gether' 'etheless' 'adays' 'tlement' '/-' 'xiety' 'SplitOptions' 'odore' 'isposable' 'case' 'vron' '&p' 'instein' 'imat
- **statement**: After SFT, responses will show more explicit intermediate algebraic manipulation steps (collecting terms, denoting sub-expressions, chained implications) written with inline Unicode math symbols such as ⇒, ≠, ≡, ², √.  · kind=both · unit=per_1k_tokens · generic=False · conf=2
- regex: `[⇒≠≡≈≤≥√²³]` `\bLet me (?:denote|collect|rearrange|simplify|expand|substitute|factor|isolate)\b` `\bBring (?:all )?(?:the )?terms\b`
- rubric: Does the response carry out step-by-step algebraic manipulation (collecting terms, substituting, denoting sub-expressions, chained implications) as visible intermediate work using inline Unicode math symbols such as ⇒, ≠, ≡, ², √ (rather than LaTeX markup or words only)?
- notes: All three axes agree at conf 3; lens tokens are noise. Bare 'intermediate derivation steps' would be generic within math, so the item is pinned to the FORM marker (Unicode inline math rather than LaTeX), which is a distinctive style of the SFT data. Regex ~0 on non-math prompts; interpret per-1k-tokens on the math subset. Confound: base model may emit LaTeX instead, which this regex deliberately does not count. Verifier: rubric aligned with the statement's Unicode-symbol condition and generic flag set to false since the combined property is not near-ubiquitous.
- channel A: held-out chunks where this atom fires show excess Δlogp(SFT−base) > 0 after headroom control (double residualisation), and the excess ranks with mass rank
- 2×2: act-high/grad-high → increase (both dictionaries carry it)
  - `(x+1) ≠ 0 ⇒ x +1 ≠0 ⇒x ≠-1, but since x is already at least 1, that's`
  - ` Case 1:\n\n-2h + k -1 = h -2k +3\n\nLet me collect terms. Bring all terms to the left side:\n\n`

### 7. atom 883 — hesitation and reconsideration in reasoning
- MOVE: the text is pausing to reconsider or double-check a previous step in reasoning (3) · FORM: chunks contain the word "Wait" or similar hesitation markers ("Hmm", "Hold on", "Alternatively") (3) · CONTENT: N/A (0)
- lens: 'wap' '/id' 'unda' ' embar' 'NL' 'arin' '794' 'usher' ' lately' 'peq' '.me' 'arte' 'utto' 'cea' ' dims'
- **statement**: After SFT, responses will show more explicit hesitation or pause markers ('Wait', 'Hmm', 'Hold on', 'I'm a bit confused') used to reconsider a previous step.  · kind=both · unit=per_1k_tokens · generic=False · conf=3
- regex: `\bWait\b` `\bHmm+\b` `\bHold on\b` `(?i)\bI'?m (?:a bit |a little |kind of |somewhat )?confused\b`
- rubric: Does the response contain at least one explicit hesitation or pause marker (e.g., 'Wait', 'Hmm', 'Hold on', 'I'm confused') used to reconsider or double-check a previous step?
- notes: FORM and MOVE both conf 3 and consistent with examples. Broadest of the Wait-family items (ranks 1, 4, 5, 17, 23, 25, 27) and shares 'Hmm' with ranks 11 and 38; counts any capitalised 'Wait' (case-sensitive to exclude 'wait for' in code). Expected to be the single most robust surface marker of long-CoT SFT. Verifier: regexes valid.
- channel A: held-out chunks where this atom fires show excess Δlogp(SFT−base) > 0 after headroom control (double residualisation), and the excess ranks with mass rank
- 2×2: act-low/grad-high → increase; gradient-unique cell: channel A excess Δlogp > 0, channel B effect size > act-only cell
  - ` odd, which they are here. Wait but that's 4 terms of 1. So that counts as one decomposition here? Because it's an ordered decomposition`
  - ` divided by the count of samples in that class." That is exactly the problem says. Then the example must have a mistake?\n\nAlternatively, maybe the user made a`

### 8. atom 6240 — contrast/reconsideration markers
- MOVE: reconsidering or correcting a previous statement, proposing alternatives, or checking for errors (3) · FORM: chunks contain the word "but" or "alternatively" indicating contrast or reconsideration (3) · CONTENT: N/A (0)
- lens: ',but' '…but' '—but' ' but' 'but' '.But' ' But' 'But' ' mais' ' zwar' '但' 'However' ' pero' ' BUT' 'aro'
- **statement**: After SFT, responses will show more sentence-initial contrastive self-qualification ('But perhaps ...', 'But the problem might ...', 'However, ...') that pushes back on the writer's own preceding statement or plan.  · kind=both · unit=per_1k_tokens · generic=False · conf=2
- regex: `(?m)(?:^|[.!?]\s+)But\b` `\bHowever,` `(?i)\bbut (?:perhaps|maybe|I think|I'm not sure|what if|then again|wait)\b`
- rubric: Does the response contain at least one sentence-initial contrastive clause ('But ...', 'However, ...') in which the writer qualifies or pushes back on their own immediately preceding claim or plan?
- notes: Lens tokens ('but', 'But', 'However', multilingual) strongly agree with FORM. Plain 'but' is ubiquitous, so regex is restricted to sentence-initial 'But', 'However,', and 'but perhaps/maybe/...'. Third example is ordinary descriptive contrast, so the rubric requires the contrast to target the writer's own prior claim. Near-duplicate of rank 34 (7063) at the marker level. Verifier: regexes valid.
- channel A: held-out chunks where this atom fires show excess Δlogp(SFT−base) > 0 after headroom control (double residualisation), and the excess ranks with mass rank
- 2×2: act-high/grad-high → increase (both dictionaries carry it)
  - ` about these concepts. Let me check if there are other instances, but I think these are the main ones. Need to explain both terms clearly, their roles in`
  - ` as a noun, which might be a bit off. But perhaps acceptable. Alternatively, "ancestral voices" or "ancestral past," but the user wants`

### 9. atom 32 — reasoning justification with "because"/"since"
- MOVE: explaining or justifying a reasoning step or decision (3) · FORM: chunks contain the word "because" or equivalent reasoning markers (e.g., "since") (3) · CONTENT: N/A (0)
- lens: ' because' 'because' ' Because' 'Because' ' porque' ' cuz' 'ecause' ' parce' ' perché' '因' ' karena' ' �' ' weil' ' sinc
- **statement**: After SFT, responses will show more explicit causal justification of the writer's own intermediate decisions using 'because' / 'since' clauses.  · kind=both · unit=per_1k_tokens · generic=True · conf=2
- regex: `(?i)\bbecause\b` `(?i)\bsince\b`
- rubric: Does the writer justify at least one of their own intermediate decisions or claims (not merely a fact about the world) with an explicit 'because' or 'since' clause?
- notes: Lens tokens ('because', 'since', multilingual) agree exactly with FORM. Justifying steps is near-universal in step-by-step text (generic=true), so the per-response rubric will saturate; the per-1k-token rate is the informative measure. 'since' also has a temporal sense (minor noise). Shares the 'since' regex with ranks 12 and 20. Verifier: regexes valid.
- channel A: held-out chunks where this atom fires show excess Δlogp(SFT−base) > 0 after headroom control (double residualisation), and the excess ranks with mass rank
- 2×2: act-high/grad-high → increase (both dictionaries carry it)
  - ` stops when the duplicate is found.\n\nSo that's correct, because the problem says that any duplicate occurrence (regardless of data) should trigger the exception, since`
  - ` But the problem might not require handling negatives because the function is supposed to handle the case where value exceeds the maximum. Maybe the function doesn't need to handle negatives`

### 10. atom 2080 — self-qualifying parenthetical reasoning
- MOVE: self-correction or qualification of reasoning (e.g., questioning assumptions, proposing alternatives, noting edge cases) (3) · FORM: chunks contain parenthetical remarks (often starting with "since", "but", "maybe", "though") (3) · CONTENT: N/A (0)
- lens: ' (' '（' ' (+' ' (~' '(Note' ' (>' ' ([[' ' //(' ' (£' ' (<' ' （' '(if' ' (&' '...(' ' (#'
- **statement**: After SFT, responses will show more parenthetical asides that qualify, hedge, or question the writer's own reasoning (e.g., '(but which one?)', '(but n is at least 1, so this is not possible)', '(like ...)').  · kind=both · unit=per_1k_tokens · generic=False · conf=3
- regex: `\((?:but|since|though|although|maybe|perhaps|assuming|unless|which one|not sure|hopefully|presumably|I think)\b` `\?\)` `\(like\s`
- rubric: Does the response contain at least one parenthetical aside that qualifies, hedges, or questions the writer's own reasoning (e.g., '(but which one?)', '(but n is at least 1, so ...)') rather than merely supplying a value, citation, or function argument?
- notes: Lens tokens are all open-parenthesis variants, agreeing with FORM; all three examples contain such asides. Regex targets parentheses opening with a qualifying connective, parentheticals ending in '?', and '(like' example asides; a bare '(' would be far too generic. Verifier: removed '(e.g.,' from regex 3 because it is a stock formal-writing marker common in base-model answers and not a self-qualifying aside; '(like' is kept because example 1 uses it.
- channel A: held-out chunks where this atom fires show excess Δlogp(SFT−base) > 0 after headroom control (double residualisation), and the excess ranks with mass rank
- 2×2: act-high/grad-high → increase (both dictionaries carry it)
  - `, which addresses their need for direction without overstepping. \n\nHmm... I should also avoid any language that might imply assessment of their case (like "your`
  - ` the function is supposed to change an existing instance into the new value (but which one?). Alternatively, perhaps modifying the value to be the value parameter (like updating`

### 11. atom 7964 — self-interruption during reasoning
- MOVE: self-interruption or hesitation during problem-solving (questioning understanding, re-reading, noticing inconsistencies) (3) · FORM: chunks contain "Hmm", "Wait", or similar hesitation markers (2) · CONTENT: N/A (0)
- lens: ' Silent' 'ubar' 'kap' 'اد' ' aspir' ' silent' 'anza' 'si' 'Fa' ' heaps' ' Sirius' 'fits' 'Ans' ' Ce' 'Wa'
- **statement**: After SFT on this data, model responses will show more mid-reasoning pauses of hesitation in which the writer interrupts the solution to re-read the problem or question their own understanding, marked by "Hmm" or "Let me think" before continuing.  · kind=both · unit=per_1k_tokens · generic=False · conf=2
- regex: `\bHmm+\b` `\bLet me think\b` `(?i)\bre-?read(?:ing)? the (?:problem|question|prompt|statement)\b`
- rubric: Does the response contain at least one point where the writer pauses mid-reasoning to express hesitation or re-check their understanding of the task (e.g., "Hmm", "Let me think about that", re-reading the constraints) before continuing?
- notes: MOVE conf 3 used; FORM conf 2 lists Hmm/Wait. Lens tokens are noise. Examples show 'Hmm, right.', 'Let me think through how to approach this.', 'Hmm, so I have to' (hesitation before planning rather than an outright error catch). 'Wait' deliberately excluded to keep the item distinct from ranks 1/17/23. Shares 'Hmm' with ranks 7 and 38 and 'Let me think' with rank 37. Verifier: regexes valid.
- channel A: held-out chunks where this atom fires show excess Δlogp(SFT−base) > 0 after headroom control (double residualisation), and the excess ranks with mass rank
- 2×2: act-low/grad-high → increase; gradient-unique cell: channel A excess Δlogp > 0, channel B effect size > act-only cell
  - `1. And if possible, output such a sequence. Hmm, right.\n\nFirst, I need to think about the constraints of a beautiful sequence. The absolute difference`
  - ` by placing a chip on s and moving it right then left as described. Let me think through how to approach this.\n\nFirst, the process described is: place`

### 12. atom 1770 — mid-reasoning self-correction with "since"
- MOVE: self-correction or verification during reasoning (catching errors, confirming assumptions, reconsidering approaches) (3) · FORM: chunks contain the word "since" (often repeated) and frequently involve mid-reasoning corrections or confirmations ("Wait, no", "Hmm", "let me confirm") (3) · CONTENT: N/A (0)
- lens: 'Since' ' Since' ' since' 'since' '.Since' '_since' 'cek' 'ince' ' SIN' 'adal' ' depuis' ' seit' ' sinc' ' Grip' 'reen'
- **statement**: After SFT on this data, model responses will show more justification clauses introduced by "since" that are used to confirm, verify, or correct a step by citing an already-established fact (e.g., "Wait, no. Since AC is horizontal, the perpendicular is vertical").  · kind=both · unit=per_1k_tokens · generic=False · conf=2
- regex: `(?i)\bsince\b`
- rubric: Does the response contain at least one point where the writer confirms, verifies, or corrects a step by citing an established fact with a "since"-clause (e.g., "Since X, Y must be ...", "Since ..., let me confirm")?
- notes: Label and lens tokens ('Since', ' since', multilingual) point specifically to the connective 'since'; regex counts all uses including temporal 'since' (minor confound). Same surface marker as rank 9 (because/since) and rank 20 (sentence-initial 'Since'); the rubric distinguishes verification/correction use. Verifier: regex valid; hits all examples.
- channel A: held-out chunks where this atom fires show excess Δlogp(SFT−base) > 0 after headroom control (double residualisation), and the excess ranks with mass rank
- 2×2: act-high/grad-high → increase (both dictionaries carry it)
  - ` be vertical? Wait, no. Since AC is horizontal, the perpendicular would be vertical. Since BD is perpendicular to AC (which is along the x-axis),`
  - `0). So reflecting over line AP, which is the vertical line x=1. Since reflecting over a vertical line is straightforward.\n\nHowever, let me confirm.`

### 13. atom 382 — hypothetical reasoning with "would be"
- MOVE: proposing or verifying a hypothetical scenario or calculation (3) · FORM: chunks contain "would be" or "would" followed by a verb (3) · CONTENT: N/A (0)
- lens: ' would' ' Would' 'would' 'Would' ' wouldn' ' Wouldn' "'d" '’d' ' serait' 'OULD' ' seria' ' würde' ' zou' ' skulle' ' wü
- **statement**: After SFT on this data, model responses will show more conditional/hypothetical phrasing with "would" ("would be", "wouldn't") used to work out the value or outcome of an assumed case (e.g., "n//2 would be 2, so the two elements would be indexes 1 and 2").  · kind=both · unit=per_1k_tokens · generic=False · conf=3
- regex: `(?i)\bwould(?:n'?t)?\b` `(?i)\bwould be\b`
- rubric: Does the response contain at least one point where the writer works out the value or outcome of a hypothetical or assumed case using conditional phrasing such as "would be" (e.g., "if n=4, the upper limit would be ...")?
- notes: MOVE and FORM both conf 3; lens tokens (' would', ' wouldn', "'d", multilingual) agree exactly. Confound: polite/advisory 'would' in chat responses ('I would recommend') inflates counts on QA prompts; rubric restricts to hypothetical calculation. First regex subsumes the second (kept for a stricter count). Verifier: regexes valid.
- channel A: held-out chunks where this atom fires show excess Δlogp(SFT−base) > 0 after headroom control (double residualisation), and the excess ranks with mass rank
- 2×2: act-high/grad-high → increase (both dictionaries carry it)
  - `. Let's compute where that center would be.\n\nThe third circle near the top vertex C would be 7 cm away from the two other sides. The sides`
  - ` h²=25 - 2.25=22.75=91/4 → h= sqrt(91)/2.\n\nTherefore, coordinates would be:\n\n`

### 14. atom 4360 — proposing alternatives
- MOVE: proposing alternative approaches or solutions (3) · FORM: chunks contain the word "Alternatively" or "Or maybe" (3) · CONTENT: N/A (0)
- lens: ' Alternatively' 'Alternatively' ' alternatively' ' OR' ' Or' ' 或' '—or' ' Altern' 'altern' 'Or' ' or' ' alternate' ' al
- **statement**: After SFT on this data, model responses will show more explicit proposals of an alternative approach or solution introduced by "Alternatively" or a sentence-initial "Or (maybe/perhaps) ..." after a first approach has been stated.  · kind=both · unit=per_1k_tokens · generic=False · conf=3
- regex: `\bAlternatively\b` `(?:(?<=[.?!] )|(?<=\n))Or\b` `(?i)\bor (?:maybe|perhaps)\b`
- rubric: Does the response contain at least one point where the writer explicitly proposes a second, different approach or solution to the same sub-problem after stating a first one (e.g., "Alternatively, ...", "Or we could ...")?
- notes: MOVE and FORM conf 3; lens tokens (' Alternatively', ' Or', ' OR', ' 或') agree with the label. Second regex uses two fixed-width lookbehinds for sentence-initial 'Or' (valid Python re, verified). Confound: 'Or' inside quoted options or code is rare in prose but possible. Overlaps with ranks 2 and 29 (Alternatively). Verifier: regexes valid.
- channel A: held-out chunks where this atom fires show excess Δlogp(SFT−base) > 0 after headroom control (double residualisation), and the excess ranks with mass rank
- 2×2: act-high/grad-high → increase (both dictionaries carry it)
  - `   c. Any other character just gets added as is, or perhaps all should be escaped except %, but that's overkill. Alternatively, only escape the`
  - ` two, sum them.\n\nAlternatively, find the two largest primes. Alternatively, sort the list and pick the top two. That seems straightforward.\n\nWait, primes list`

### 15. atom 3039 — self-reflective reasoning adjustments
- MOVE: the text is reflecting on or adjusting its own reasoning process, often mid-problem-solving (e.g., considering alternatives, catching errors, clarifying constraints) (3) · FORM: N/A (0) · CONTENT: N/A (0)
- lens: ' {}.' ' Sinclair' 'enis' '/ca' 'ties' 'viso' ' walks' 'áticas' ' $?' "':\n\n" 'orio' 'akin' 'cea' ' justo' ' depart'
- **statement**: After SFT on this data, model responses will show more first-person meta-planning statements in which the writer deliberates about how to shape or adjust their own answer (e.g., "Maybe I should present approximate numbers", "I should also avoid ...", "So in the answer, I can explain the process").  · kind=both · unit=per_response · generic=False · conf=2
- regex: `\b(?:Maybe|Perhaps) I (?:should|can|could|need to)\b` `\bI should (?:also|probably|just|make sure|mention|present|explain|avoid|note|include|start|structure|focus)\b` `(?i)\bin (?:the|my) (?:answer|response|reply),? I (?:can|could|should|will|'ll)\b`
- rubric: Does the response contain at least one first-person statement in which the writer deliberates about how to shape, structure, or adjust their own answer or approach (e.g., "Maybe I should present approximate numbers", "in the answer I can explain the process", "I should also mention ...")?
- notes: Only MOVE is labeled (conf 3) and it is broad; FORM is N/A and lens tokens are noise, so operationalization relies on the 3 examples (meta-planning of the answer in chat/creative contexts). Regexes are heuristic; rubric primary (per_response). Overlaps with rank 22 (planning for constrained writing) and ranks 2/14 (alternatives). Verifier: dropped the 'the key is to' regex, which is common advice-giving phrasing ('the key is to stay consistent') and not specific to self-directed meta-planning.
- channel A: held-out chunks where this atom fires show excess Δlogp(SFT−base) > 0 after headroom control (double residualisation), and the excess ranks with mass rank
- 2×2: act-low/grad-high → increase; gradient-unique cell: channel A excess Δlogp > 0, channel B effect size > act-only cell
  - `'t sustain as many exhibits. Hmm, conflicting possibilities. Maybe I should present approximate numbers based on typical patterns, since I can't access real-time data. \n\n`
  - ` where Akuma learns the value of compassion through interactions with students, but that's less in character. Alternatively, his reign leads to a crisis that forces the students`

### 16. atom 1633 — math self-correction with hesitation markers
- MOVE: self-correction or verification during mathematical reasoning (3) · FORM: chunks contain "Wait" or similar hesitation markers ("Hmm", "Let me see", "perhaps") (2) · CONTENT: mathematical problem-solving (algebra, calculus, combinatorics) (3)
- lens: 'ANI' ' Pants' '.cm' 'ani' 'OfString' 'ksam' ' Shades' ' scout' ' Staten' 'ités' ':init' '定' 'ecom' 'earn' ' Griffith'
- **statement**: After SFT on this data, model responses will show more explicit verification passes in mathematical reasoning in which the writer re-walks or re-checks earlier steps (e.g., "let me go through the steps once more to verify", "double-check", "plugging back in").  · kind=both · unit=per_1k_tokens · generic=False · conf=2
- regex: `(?i)\b(?:let me|let'?s|I'?ll|I will|I should) (?:verify|double[- ]?check|re-?check|check (?:this|that|again|the (?:steps|calculation|math|work)))\b` `(?i)\bdouble[- ]?check(?:ing|ed)?\b` `(?i)\bto verify\b` `(?i)\bonce more\b` `(?i)\bplug(?:ging)? (?:it |this |that |the values? |them )?back\b`
- rubric: In a math or quantitative solution, does the response contain at least one explicit verification pass where the writer re-derives, re-checks, or substitutes back an earlier result (e.g., "let me verify", "going through the steps once more", "plugging back in")?
- notes: MOVE conf 3 ('self-correction or verification during mathematical reasoning'); CONTENT = math; FORM conf 2. Lens tokens are noise. Item targets verify/double-check markers to stay distinct from the Wait family; overlaps with ranks 4, 18, 26, 37 (verification/Let me check). Rubric conditions on math prompts (answer 'no' on non-math). Verifier: regexes valid.
- channel A: held-out chunks where this atom fires show excess Δlogp(SFT−base) > 0 after headroom control (double residualisation), and the excess ranks with mass rank
- 2×2: act-low/grad-high → increase; gradient-unique cell: channel A excess Δlogp > 0, channel B effect size > act-only cell
  - ` through the steps once more to verify.\n\nStarting with the two original equations:\n\nEquation (1): 3f(1/x) + (2f(x`
  - ` an identity, which suggests that equations (1) and (2) are not independent once we plug F=0. Hmm. That complicates things, so`

### 17. atom 7033 — hesitation and self-correction
- MOVE: self-correction or hesitation in reasoning (e.g., catching mistakes, re-evaluating assumptions, clarifying instructions) (3) · FORM: chunks contain the word "Wait" or equivalent hesitation markers (e.g., "Hmm", "not sure") (3) · CONTENT: N/A (0)
- lens: 'agna' 'ney' ' Julian' 'inne' 'eman' 'ai' ' Wear' 'ner' 'advertisement' ' Bates' ' Queens' ' emanc' '?id' ' Op' 'aid'
- **statement**: After SFT on this data, model responses will show more mid-reasoning hesitation markers that express uncertainty about the writer's own claim and interrupt to re-check it, specifically "Wait" and "not sure" (e.g., "Wait, Bigolin is 7 letters? Wait no").  · kind=both · unit=per_1k_tokens · generic=False · conf=2
- regex: `\bWait\b` `(?i)\bnot (?:entirely |quite |completely |100% |totally )?sure\b`
- rubric: Does the response contain at least one point where the writer expresses uncertainty about a claim they just made or interrupts themselves to re-check it (e.g., "Wait", "I'm not sure", "Wait no")?
- notes: MOVE broad; examples heterogeneous (creative prose, chemistry exposition, one 'Wait ... Wait no' case), so label fit is uncertain. 'Wait' regex is shared with ranks 7 and 23; 'not sure' is the marker unique to this item. Confound: imperative 'Wait' in chat. Verifier: regexes valid; retained since label is measurable.
- channel A: held-out chunks where this atom fires show excess Δlogp(SFT−base) > 0 after headroom control (double residualisation), and the excess ranks with mass rank
- 2×2: act-low/grad-high → increase; gradient-unique cell: channel A excess Δlogp > 0, channel B effect size > act-only cell
  - ` a confession: of dreams deferred, of silence endured, of the constant hum of inadequacy. She was fracture and fury, a symphony of discord played`
  - ` written as a complex. The ate complex mentioned in the question probably refers to the adduct named with an "-ate" ending. Common examples are things like [`

### 18. atom 3725 — mid-reasoning verification or correction
- MOVE: the text is actively checking, verifying, or reconsidering a previous step in the reasoning process (3) · FORM: chunks often contain mid-sentence mathematical or logical reasoning, with phrases like "Wait," "Alternatively," or "Let me check" (2) · CONTENT: N/A (0)
- lens: ':' '():' '+:' '.:' '：' '_:' ' -:' '$:' '#:' '?:' '’:' '!:' '):' "':" '":'
- **statement**: After SFT on this data, model responses will show more explicit hand-tracing of a concrete sub-computation to check a step, typically introduced by a colon (e.g., "To compute its volume:", "Second element: [6,7]: sum_nested returns 13, so total is 5").  · kind=both · unit=per_response · generic=False · conf=2
- regex: `\bLet me (?:check|verify|trace|compute|calculate|work (?:this|that|it) out)\b` `\b(?:To|Let me|Let'?s|Now) (?:compute|calculate|check|verify|trace)\b[^\n:]{0,60}:` `\b(?:First|Second|Third|Fourth|Next|Last) (?:element|item|term|case|iteration|call|digit|row|column)\b[^\n:]{0,30}:`
- rubric: Does the response contain at least one point where the writer verifies a step by explicitly tracing through a concrete case or sub-computation with intermediate values (e.g., plugging a specific input into the procedure and stepping through the resulting values)?
- notes: MOVE conf 3; FORM conf 2. Lens tokens are all colon variants, agreeing with the examples' colon-introduced computations, so the item is narrowed to colon-introduced tracing to distinguish it from rank 16 (verify wording) and rank 19 (inconsistency flagging). Regexes heuristic; rubric primary. Verifier: added a third colon-introduced 'Second element:'-style regex matching example 3; no bare-colon regex because of code blocks.
- channel A: held-out chunks where this atom fires show excess Δlogp(SFT−base) > 0 after headroom control (double residualisation), and the excess ranks with mass rank
- 2×2: act-low/grad-high → increase; gradient-unique cell: channel A excess Δlogp > 0, channel B effect size > act-only cell
  - ` consider that.\n\nBut then the volume of PBCD: the tetrahedron PBCD.\n\nTo compute its volume: since the volume of a tet`
  - `, for subsets with maximum a and b where a < b: The subsets A (max a) and B (max b) must satisfy |A| +`

### 19. atom 694 — mid-reasoning error detection
- MOVE: detecting and pausing at a potential inconsistency or error in the reasoning process (3) · FORM: chunks contain "Hmm", "Wait", or similar hesitation markers, often mid-calculation or mid-reasoning (3) · CONTENT: N/A (0)
- lens: 'bij' ' Chili' 'ERA' 'elin' 'iros' ' Roc' 'nds' 'asks' 'eneg' 'FINITY' ' pun' 'oola' '?f' ' regul' 'reas'
- **statement**: After SFT on this data, model responses will show more explicit flagging of an inconsistency or contradiction discovered mid-calculation, where the writer notices an intermediate result conflicts with an earlier fact (e.g., "Hmm, that's a problem. So that's not matching.", "Wait, but ...").  · kind=both · unit=per_1k_tokens · generic=False · conf=2
- regex: `\bWait,? but\b` `(?i)\bthat'?s (?:a problem|not (?:right|possible|matching|correct|consistent)|inconsistent|a contradiction|impossible|wrong)\b` `(?i)\b(?:doesn'?t|does not|don'?t|do not|didn'?t|did not) (?:match|work|hold|add up|agree|satisfy)\b` `(?i)\bcontradict(?:s|ion|ory|ing)?\b`
- rubric: Does the response contain at least one point where the writer explicitly notices that an intermediate result is inconsistent with an earlier fact, constraint, or expected value (e.g., "that's a problem", "this doesn't match", "contradiction") and pauses to address it?
- notes: MOVE conf 3; FORM conf 3. Lens tokens are noise. Examples consistently show a detected mismatch. Regexes target mismatch-flagging language rather than bare 'Hmm'/'Wait'. Confound: 'contradiction' in proof-by-contradiction. Overlaps with ranks 1, 5, 34. Verifier: regexes valid.
- channel A: held-out chunks where this atom fires show excess Δlogp(SFT−base) > 0 after headroom control (double residualisation), and the excess ranks with mass rank
- 2×2: act-low/grad-high → increase; gradient-unique cell: channel A excess Δlogp > 0, channel B effect size > act-only cell
  - `Wait correct paren combinations for n=2:\n\nPossible are:\n\n1. ()() → two pairs.\n\n2. (()) → one pair but nested.\n\nWait but`
  - `/49≈6.857… Not integer. Since 336 divided by 49 is 6 and 42/49? Hmm, so not integer`

### 20. atom 7719 — "since"-based reasoning transitions
- MOVE: introducing a premise or justification for the next step in reasoning (3) · FORM: chunks contain the word "since" or equivalent reasoning transitions (3) · CONTENT: N/A (0)
- lens: '`;' '；' ';' ' {}.' ';\n' '].' '”;' '`.' ';.' '}.' ' "").' '];' ']].' '’.' "';"
- **statement**: After SFT on this data, model responses will show more sentence-initial "Since ..." statements that introduce a premise or known fact as the justification for the next reasoning step (e.g., "Since the sector itself is already a 60-degree angle at the center. Let me think ...").  · kind=both · unit=per_1k_tokens · generic=False · conf=2
- regex: `\bSince\b`
- rubric: Does the response contain at least one sentence beginning with "Since ..." that states a premise or known fact as the justification for the step that follows (rather than a temporal 'since')?
- notes: MOVE generic at the move level; FORM pins it to 'since'. Lens tokens are end-punctuation and do NOT corroborate; label fit rests on the 3 examples, all opening with 'Since ...'. Regex counts capitalized 'Since' as a sentence-initial proxy; overlaps with ranks 9 and 12. Verifier: regex valid; hits all examples.
- channel A: held-out chunks where this atom fires show excess Δlogp(SFT−base) > 0 after headroom control (double residualisation), and the excess ranks with mass rank
- 2×2: act-high/grad-high → increase (both dictionaries carry it)
  - ` confusing. Since the equation only involves constants and \( c \). Wait, so maybe the problem is asking for how many real numbers \( c \) satisfy the`
  - ` circumference of the sector. Since the sector itself is already a 60-degree angle at the center. Let me think—the chord might be somewhere inside the sector,`

### 21. atom 4356 — "then"-heavy step sequencing
- MOVE: describing a sequence of steps or transitions in reasoning/process (3) · FORM: chunks contain the word "then" (often multiple times) (3) · CONTENT: N/A (0)
- lens: ' followed' ' then' '_then' 'then' ' THEN' 'Then' '.Then' ' Then' '\tthen' 'THEN' ' puis' '然' ' dann' '.then' ' ensuite'
- **statement**: After SFT, responses will show more explicit step-sequencing with the connective "then" (e.g., "..., then ..., and then ...", sentence-initial "Then") to chain consecutive steps of a procedure or derivation.  · kind=both · unit=per_1k_tokens · generic=False · conf=3
- regex: `(?i)\bthen\b` `(?m)(?:^|[.!?]\s+|,?\s+and\s+)[Tt]hen\b`
- rubric: Does the response lay out a procedure, plan, or derivation as an ordered chain of steps linked by "then" / "and then" / sentence-initial "Then" at least twice?
- notes: Lens tokens strongly agree (then/Then/THEN/.then/puis/dann). Regex 1 is the broad count; regex 2 restricts to sentence-initial or 'and then' step transitions. Confound: 'then' in code (if/then, .then()) and 'if ... then' in math. Overlaps with rank 30 ('So') as consequence/sequencing connectives. Verifier: regexes valid.
- channel A: held-out chunks where this atom fires show excess Δlogp(SFT−base) > 0 after headroom control (double residualisation), and the excess ranks with mass rank
- 2×2: act-low/grad-high → increase; gradient-unique cell: channel A excess Δlogp > 0, channel B effect size > act-only cell
  - ` better approach is to try to tile the grid with as many red squares as possible such that no two are adjacent. Then whatever that number is, the maximum independent`
  - `Wait, perhaps the problem is that each song has a specific required group size, and then we randomly select a group of the required size, and then check whether`

### 22. atom 5488 — metacognitive planning for constrained writing
- MOVE: planning or strategizing how to approach a creative or constrained writing task, often with explicit attention to balancing multiple requirements (3) · FORM: chunks contain metacognitive markers ("Hmm", "First", "I should", "Let me unpack") and planning language ("need to capture", "should weave in") (3) · CONTENT: N/A (0)
- lens: 'FromString' 'rt' ' SV' ' coax' ' rá' 'uner' 'ickets' ' gn' ' Hope' 'urrect' 'ogn' 'MM' ' dominance' 'sy' ' Meyer'
- **statement**: After SFT, responses to creative or constrained-writing requests will show more explicit up-front planning about how to approach the task, naming the requirements/constraints to satisfy and how they will be balanced or woven in (e.g., "I should ...", "The challenge is balancing ...", "need to capture ...").  · kind=both · unit=per_response · generic=False · conf=2
- regex: `\bI should\b` `\b(?:need|needs|have) to (?:capture|include|incorporate|weave|balance|convey|make sure|ensure|cover|hit)\b` `\b[Tt]he (?:challenge|key|trick) (?:is|here is|will be)\b` `\bLet me (?:unpack|start by|plan|think about how)\b` `(?m)^Hmm[,.]`
- rubric: Does the response contain explicit up-front planning of how to approach the requested writing task, naming the requirements or constraints it must satisfy and how it will balance or incorporate them, before (or instead of) producing the requested output itself?
- notes: Rubric is the primary measure; regexes are loose proxies ('I should' also appears in ordinary chat). Source mix is persona/wildchat/aya/if_qwq/jailbreak, so this item bites on creative/constrained-writing prompts and shows no signal on math/code. Lens tokens are junk. Confound: an SFT model that exposes its thinking trace will trivially expose planning a base model does silently. Overlaps with rank 15 (meta-planning). Verifier: rubric reworded so it is answerable from the response alone without a prior classification of the request type.
- channel A: held-out chunks where this atom fires show excess Δlogp(SFT−base) > 0 after headroom control (double residualisation), and the excess ranks with mass rank
- 2×2: act-high/grad-high → increase (both dictionaries carry it)
  - ` should make this feel authentic and sensory-rich. Taiwanese beef noodle soup is such a nostalgic dish—perfect for intergenerational storytelling. The spices are key:`
  - ` immersion over tourist traps. The challenge is balancing specificity with the tight four-sentence structure while naturally incorporating that required combo about street food and locals. \n\nI shoul`

### 23. atom 5998 — "Wait" signaling mid-reasoning doubt
- MOVE: pausing to question or re-examine a previous step in reasoning (often catching potential errors or inconsistencies) (3) · FORM: chunks contain the word "Wait" (often at the start of a line or after punctuation) (3) · CONTENT: N/A (0)
- lens: '?).' '?)' '?),' ' ??' '?,' 'hani' '?",\n' ' ?' '?)\n\n' '?)\n' '?!' '?:' ' ???' '??' '????'
- **statement**: After SFT, responses will show more mid-reasoning pauses of doubt marked by "Wait" that immediately question a step just taken, typically phrased as a question (e.g., "Wait, so 8/h must be an integer?", "Wait a second, is that right?").  · kind=both · unit=per_1k_tokens · generic=False · conf=3
- regex: `\bWait\b` `\bWait\b[^.\n?!]{0,80}\?` `\bWait a (?:second|minute|moment)\b`
- rubric: Does the response contain at least one point where the writer pauses with "Wait" (or equivalent) to question or re-examine a step it already took, expressing doubt about it, rather than only proceeding forward?
- notes: Lens tokens are question-mark variants, consistent with the doubt-as-question reading; regex 2 (Wait followed by '?' within 80 chars) is the most atom-specific marker and hits all three examples. Regex 1 is shared with ranks 7, 17, 25, 27. Confound: 'Wait' in quoted dialogue or as an instruction. Verifier: regexes valid.
- channel A: held-out chunks where this atom fires show excess Δlogp(SFT−base) > 0 after headroom control (double residualisation), and the excess ranks with mass rank
- 2×2: act-high/grad-high → increase (both dictionaries carry it)
  - ` h >=8? \n\nWait, so 8/h must be an integer? Then h divides 8? Since h is a positive integer. But h is`
  - `, so t ≤25.\n\nWait a second, so 379 mod385 is the smallest positive solution? But that's 379? But that is way more`

### 24. atom 7715 — reconsidering assumptions with "maybe"
- MOVE: reconsidering or questioning a previous assumption or interpretation (3) · FORM: chunks contain "maybe" or "perhaps" and often "wait" (3) · CONTENT: N/A (0)
- lens: ' Maybe' ' maybe' 'Maybe' '.Maybe' 'maybe' ' Perhaps' 'Perhaps' ' equ' ' perhaps' 'aybe' ' somehow' 'perhaps' ' Pres' 'p
- **statement**: After SFT, responses will show more hedged reconsideration of a prior assumption or interpretation using "maybe" / "perhaps" (e.g., "Wait, maybe x can be any real number?", "perhaps each occurrence must be processed").  · kind=both · unit=per_1k_tokens · generic=False · conf=3
- regex: `(?i)\b(?:maybe|perhaps)\b` `\b(?:Wait|Hmm|Actually|[Hh]old on)\b[^.\n]{0,40}\b(?:maybe|perhaps)\b` `(?i)\b(?:maybe|perhaps) not\b`
- rubric: Does the response contain a passage where the writer questions its own earlier interpretation of the problem or an assumption it made, and floats an alternative reading using hedged language such as "maybe" or "perhaps"?
- notes: Lens tokens strongly agree (Maybe/Perhaps/somehow). Regex 2 (Wait/Hmm/Actually followed by maybe/perhaps) is the most specific to reconsidering; regex 1 alone also captures polite chat suggestions. Overlaps with ranks 2, 3, 15 at the lexical level. Verifier: regex 2 made case-insensitive for 'Hold on' ([Hh]old on).
- channel A: held-out chunks where this atom fires show excess Δlogp(SFT−base) > 0 after headroom control (double residualisation), and the excess ranks with mass rank
- 2×2: act-low/grad-high → increase; gradient-unique cell: channel A excess Δlogp > 0, channel B effect size > act-only cell
  - ` problem mentions that y is a whole number, but doesn't specify x, I guess x can be any real number? Wait, actually, hold on, maybe`
  - `". So if a file is listed multiple times in the raw_files list, perhaps each occurrence must be processed. But that might be a consideration. Wait maybe not`

### 25. atom 2928 — reconsidering reasoning steps
- MOVE: the text is questioning or reconsidering a previous step in reasoning (3) · FORM: chunks often contain the word "Wait" or "Alternatively" (2) · CONTENT: N/A (0)
- lens: 'omon' ' bol' 'омер' 'broker' '.embed' ' wires' ' caval' '-Smith' 'jes' 'osten' 'rians' ' chall' ' speakers' 'olph' ' sp
- **statement**: After SFT, responses will show more backtracking to an already-completed step in order to question or redo it (re-checking a setup, re-deriving a quantity, or trying a different route), commonly signalled by "Wait", "Alternatively", "Hmm", or "But wait".  · kind=both · unit=per_1k_tokens · generic=False · conf=2
- regex: `(?m)(?:^|\n)\s*(?:Wait|Alternatively|Hmm|But wait)\b` `\b[Ll]et me (?:re-?check|re-?examine|reconsider|go back|double[- ]check|verify)\b` `\bAlternatively\b`
- rubric: Does the response go back to a step it already completed and question or redo it (re-checking the setup, re-deriving a value, or restarting via a different route), rather than proceeding strictly forward from each step?
- notes: Near-duplicate of the self-correction family (ranks 1, 23, 27, 28); FORM conf 2 and lens tokens junk, so the label rests on examples (revisiting a geometric setup, switching to a substitution, re-checking coefficients). Regex 1 (line-initial Wait/Alternatively/Hmm) is a proxy for the backtracking point. Verifier: regexes valid.
- channel A: held-out chunks where this atom fires show excess Δlogp(SFT−base) > 0 after headroom control (double residualisation), and the excess ranks with mass rank
- 2×2: act-low/grad-high → increase; gradient-unique cell: channel A excess Δlogp > 0, channel B effect size > act-only cell
  - ` the smaller squares are within the central square? But the central square is itself inscribed in the central circle. That central square would be rotated 45 degrees relative`
  - ` me think.\n\nAlternatively, suppose we let variables u = a + b + c. Maybe express each term in terms of u.\n\nBut each denominator is (sum`

### 26. atom 7901 — math verification
- MOVE: verifying or checking a mathematical result or condition (e.g., impossibility, correctness, edge cases) (2) · FORM: N/A (0) · CONTENT: mathematical reasoning and problem-solving (algebra, geometry, calculus, etc.) (3)
- lens: ' Hers' ' resid' 'ounty' 'ynos' '록' 'ibling' 'UD' '.Classes' 'ONA' '_TAC' 'uch' 'LL' ' informat' 'foil' ' fam'
- **statement**: After SFT, responses will show more explicit verification of a derived result or condition, e.g., checking that a solution is possible/consistent ("can't be negative, so impossible"), plugging in test or edge cases ("n=1 gives 1, which is correct"), or confirming signs/constraints.  · kind=both · unit=per_response · generic=False · conf=2
- regex: `(?i)\b(?:checks? out|is (?:indeed )?correct|that'?s correct|which is correct|as expected|consistent with)\b` `(?i)\b(?:test case|edge case|sanity check|double[- ]?check|let'?s (?:verify|check|test|confirm)|plug(?:ging)? (?:it |this |that |back )?in)\b` `(?i)\b(?:impossible|not possible|can'?t be (?:negative|zero|possible|true))\b`
- rubric: Does the response explicitly verify a result or condition it derived, e.g., by plugging values back in, testing a base/edge case, or checking that a solution is possible/consistent, rather than only deriving it and moving on?
- notes: MOVE conf 2 (CONTENT math conf 3). Rubric primary; regexes are a heterogeneous bag and each alone is weak. Lens tokens junk. Confound: verification is common in step-by-step math even in base models. Overlaps with ranks 4, 16, 35, 36. Verifier: regexes valid.
- channel A: held-out chunks where this atom fires show excess Δlogp(SFT−base) > 0 after headroom control (double residualisation), and the excess ranks with mass rank
- 2×2: act-low/grad-high → increase; gradient-unique cell: channel A excess Δlogp > 0, channel B effect size > act-only cell
  - `'t be possible because the square of a real number can't be negative. Similarly equation 3: [f(6)]² = -12, also impossible`
  - ` sum of roots being -m. Since m is positive, the sum is negative. Product is n, which is positive, so both roots have the same sign`

### 27. atom 3303 — self-correction or alternative reasoning
- MOVE: the text is correcting or reconsidering a previous step in reasoning, often proposing alternative approaches or checking for errors (3) · FORM: chunks frequently start with "Alternatively," or "Wait," (2) · CONTENT: N/A (0)
- lens: ' Sawyer' '-sw' ' razor' ' Roths' ' Mitch' ' sore' ' Orr' ' hike' ' outlaw' ' Bend' ' Young' 'ante' ' Schul' ' table' 'k
- **statement**: After SFT, responses will show more paragraph-initial "Alternatively," / "Wait," pivots that either check an intermediate result for errors ("Wait, but let me check if x=2 makes the numerator zero") or branch to a different approach ("Alternatively, let's consider squaring both sides").  · kind=both · unit=per_1k_tokens · generic=False · conf=2
- regex: `(?m)^\s*(?:Alternatively|Wait)[,.]` `\bWait,? but let me (?:check|verify|see|think)\b` `\b[Ll]et me check\b` `\bAlternatively,\s*(?:maybe|perhaps|let'?s|consider|suppose|we could|if)\b`
- rubric: After reaching an intermediate result, does the response either explicitly check it for errors (e.g., "let me check ...") or branch to a different approach introduced with "Alternatively"?
- notes: FORM says chunks frequently START with 'Alternatively,' or 'Wait,', so regex 1 is line-initial specifically and hits all three examples. Lens tokens junk. Heavy overlap with ranks 23/25 (Wait) and 14/29 (Alternatively). Verifier: regexes valid.
- channel A: held-out chunks where this atom fires show excess Δlogp(SFT−base) > 0 after headroom control (double residualisation), and the excess ranks with mass rank
- 2×2: act-low/grad-high → increase; gradient-unique cell: channel A excess Δlogp > 0, channel B effect size > act-only cell
  - `? Wait.\n\nAlternatively, maybe the graph is not simple? But in this context, since we can have multiple edges between the same two nodes? For example,`
  - `=2 or x=-2 which is excluded. Wait but let me check if x=2 or x=-2 makes the numerator zero?\n\nWait, if x`

### 28. atom 4593 — self-interruption in reasoning
- MOVE: self-interruption or reconsideration of current reasoning approach (3) · FORM: chunks contain "Wait" or "Alternatively" (2) · CONTENT: N/A (0)
- lens: 'avar' 'este' 'quam' 'arge' ' Juice' 'ened' ' contracted' ' Farrell' 'pty' 'jit' ' Intermediate' ' erect' 'chu' ' agua' 
- **statement**: After SFT, responses will show more self-interruption in which the writer breaks off an approach mid-way because it is getting tedious/complicated or a better way exists ("Hmm. Maybe this is tedious. Alternatively, ...", "But this is getting too convoluted", "perhaps a better way is ...", "Anyway, ...").  · kind=both · unit=per_response · generic=False · conf=2
- regex: `(?i)\b(?:this is|it'?s|that'?s|things are) getting (?:too |a bit |quite |really )?(?:complicated|convoluted|messy|tedious|long|complex)\b` `(?i)\b(?:too tedious|is tedious|be tedious|tedious to)\b` `(?i)\b(?:a|the) (?:better|simpler|easier|smarter) (?:way|approach|method)\b` `(?m)^\s*Anyway[,.]`
- rubric: Does the response break off from an approach it is in the middle of, noting that it is getting tedious/complicated or that a better way exists, and switch to a different approach (or abandon the line and move on)?
- notes: Distinguished from the rest of the Wait/Alternatively family by the ABANDONMENT motive ('this is tedious', 'a better way is', 'getting too convoluted', 'Anyway'). Rubric primary; regexes 1-3 are specific but low-frequency. FORM conf 2, lens junk. Confound: 'a better way' also appears in advice-giving chat. Verifier: dropped the 'Hmm[.,]' regex, which is the generic Hmm family marker (ranks 7/11/38) and would dominate the count without indicating abandonment.
- channel A: held-out chunks where this atom fires show excess Δlogp(SFT−base) > 0 after headroom control (double residualisation), and the excess ranks with mass rank
- 2×2: act-low/grad-high → increase; gradient-unique cell: channel A excess Δlogp > 0, channel B effect size > act-only cell
  - ` 85.\n\nContinuing checking primes:\n\n37: 37*197 is 7289? Hmm. Maybe this is tedious. Alternatively, perhaps 7321`
  - `Then 's a → the next parts... perhaps a better way is to process each character manually.\n\nAnyway, the code seems to handle all that.\n\nSo the`

### 29. atom 6746 — proposing alternative approaches
- MOVE: proposing alternative approaches or considering different methods to solve a problem (3) · FORM: chunks contain the word "Alternatively" or equivalent phrases suggesting alternative approaches (3) · CONTENT: N/A (0)
- lens: ':\n\n' '):\n\n' '":\n\n' ':\n\n\n' ' :\n\n' ':\n' ']:\n\n' '():\n' '):\n' "':\n\n" '：\n\n' '":\n' ':\n\n\n\n' "':\n" ':
- **statement**: After SFT, responses will show more explicit proposals of a second method or implementation for the same problem, introduced with "Alternatively" (or "Another way/approach") and often followed by a colon and a code block or worked variant.  · kind=both · unit=per_1k_tokens · generic=False · conf=3
- regex: `\bAlternatively\b` `\bAlternatively,[^\n]{0,150}:\s*\n` `\bAnother (?:way|approach|option|method|possibility)\b` `\b[Oo]r (?:we|you|I) could\b`
- rubric: Does the response present more than one method or implementation for the same problem, explicitly introducing the later one as an alternative (e.g., "Alternatively, ...", "Another approach ...")?
- notes: Lens tokens are colon+newline variants, consistent with the atom firing where an alternative is announced and then a block is opened; regex 2 targets that pattern and hits examples 1-2. Overlaps lexically with ranks 2, 14, 27. Third example chunk is Scottish Gaelic and unrelated (label noise). Verifier: simplified the redundant '(?:[Oo]r|or)' alternation in regex 4.
- channel A: held-out chunks where this atom fires show excess Δlogp(SFT−base) > 0 after headroom control (double residualisation), and the excess ranks with mass rank
- 2×2: act-low/grad-high → increase; gradient-unique cell: channel A excess Δlogp > 0, channel B effect size > act-only cell
  - `.append(g)\n        self.genres = unique\n\nAlternatively, using a set to track seen elements for efficiency (though for small lists it doesn't matter):\n\nseen`
  - ` next step. \n\nAlternatively, perhaps it's better to loop through and generate all possible sums, but only up to the target: \n\npossible = a set of`

### 30. atom 5420 — "So"-prefixed reasoning continuation
- MOVE: the text is continuing or concluding a logical step in the reasoning process, often introducing a derived fact or next step (2) · FORM: chunks often start with "So" or contain "So," followed by a continuation of reasoning or explanation (3) · CONTENT: N/A (0)
- lens: 'aren' ' Til' 'kes' 'ilde' 'commerce' ' Constant' ' Gore' '�' '184' '�' '.grade' '/support' 'Corp' '.lib' ' Johannesburg
- **statement**: After SFT, responses will show more sentences or lines that begin with "So" ("So,", "So:") to introduce a fact or next step derived from the immediately preceding reasoning (e.g., "So the recurrence relation is ...", "So:\n\nF(s) = ...").  · kind=both · unit=per_1k_tokens · generic=False · conf=2
- regex: `(?m)(?:^|[.!?]\s+)So[,:]?\s` `\bSo:\s*\n` `(?m)^So,?\s`
- rubric: Does the response repeatedly (three or more times) begin a sentence or line with "So" to state a consequence or next step that follows from the reasoning just given?
- notes: MOVE conf 2 and near-generic; the measurable quantity is the RATE of sentence-initial 'So' (FORM conf 3). Lens tokens junk. Confound: 'So' opens conversational chat replies; mid-sentence 'so' is excluded. Overlaps with rank 21 ('then'). Verifier: regexes valid; regex 1 hits all examples.
- channel A: held-out chunks where this atom fires show excess Δlogp(SFT−base) > 0 after headroom control (double residualisation), and the excess ranks with mass rank
- 2×2: act-low/grad-high → increase; gradient-unique cell: channel A excess Δlogp > 0, channel B effect size > act-only cell
  - ` \). So:\n\n\( F(s) = \frac{N(s)}{D(s)} \), with N = s +1.\n\nThe first derivative F`
  - ` sides in the ratio 1:k from the first vertex. So, like, starting from P, moving along PQ, the point U is 1 part from`

### 31. atom 3410 — error correction or edge case checking
- MOVE: the text is catching or correcting a potential error, ambiguity, or edge case in the reasoning process (3) · FORM: chunks often contain "Wait" or "Hmm" followed by a reasoning correction or clarification (2) · CONTENT: N/A (0)
- lens: ' ""' 'illis' ')' ' ()' ' but' 'чит' '={}' ' )' 'amas' ' Fil' '()' ':{}' '("/")' 'engo' ']'
- **statement**: After SFT on this data, model responses will show more explicit consideration of edge cases, boundary conditions, and ambiguous interpretations (e.g., empty inputs, duplicates, exact-match vs. near-match) beyond the main case.  · kind=both · unit=per_response · generic=False · conf=2
- regex: `(?i)\b(?:edge|corner|boundary|special)[- ]cases?\b` `\b[Ww]hat if\b` `\bempty (?:string|list|array|input|dict|dictionary|set|tuple)\b` `\bduplicates?\b`
- rubric: Does the response explicitly consider at least one edge case, boundary condition, or alternative interpretation of the task (e.g., empty input, duplicates, a path with/without trailing slash) that goes beyond the main/typical case?
- notes: MOVE conf 3 used. Lens tokens ('""', '()', '={}', ':{}') suggest empty-container checks, consistent with code edge-case handling; examples confirm. FORM ('Wait'/'Hmm', conf 2) not used as marker. 'duplicates' and 'What if' fire more on code prompts. Overlaps with ranks 26, 34. Verifier: regexes valid.
- channel A: held-out chunks where this atom fires show excess Δlogp(SFT−base) > 0 after headroom control (double residualisation), and the excess ranks with mass rank
- 2×2: act-low/grad-high → increase; gradient-unique cell: channel A excess Δlogp > 0, channel B effect size > act-only cell
  - ` a chemical formula...", which in real chemistry formulas parentheses are common (like H2SO4 has parentheses in cases like K4[Fe(CN)6]`
  - ` that. For example, if the path is '/hello/' then it's not exactly '/hello' so it should return 404. So exact matches are required`

### 32. atom 51 — "which is" elaboration
- MOVE: clarifying or elaborating on a previous statement by providing additional context or definition (2) · FORM: chunks contain the phrase "which is" (3) · CONTENT: N/A (0)
- lens: ' which' 'which' ' WHICH' '(which' '—which' ' Which' ' wich' 'Which' ' котор' '.which' ' который' 'quelle' ' whose' '�' 
- **statement**: After SFT on this data, model responses will show more in-line clarifying elaborations attached to a just-mentioned term via non-restrictive relative clauses (e.g., 'X, which is ...', '(which means ...)').  · kind=both · unit=per_1k_tokens · generic=True · conf=2
- regex: `\bwhich (?:is|are|was|were|means|equals|gives|would be)\b` `\(which\b`
- rubric: Does the response contain at least one in-line clarifying aside that elaborates on or defines a term just mentioned using a relative clause (e.g., 'the xy-plane, which is horizontal', 'agar-agar (which is derived from seaweed)')?
- notes: Lens tokens strongly agree (' which', '(which', 'whose', multilingual). Syntactic habit rather than a reasoning move; relative clauses appear in nearly any long English text, so generic=true. Per-1k-token rate is still a valid BASE vs SFT contrast; rubric will saturate. Verifier: regexes valid.
- channel A: held-out chunks where this atom fires show excess Δlogp(SFT−base) > 0 after headroom control (double residualisation), and the excess ranks with mass rank
- 2×2: act-high/grad-high → increase (both dictionaries carry it)
  - ` is related to the angle of the axis?\n\nAlternatively, perhaps the paraboloid is oriented such that when it intersects the xy-plane, which is horizontal, the`
  - ` the tangent segments.\n\nIn a tangential polygon, for each side, which is tangent to the incircle, the two adjacent tangent segments from the vertices must satisfy`

### 33. atom 4131 — verifying or clarifying problem/user instructions
- MOVE: the text is checking or clarifying the problem statement or user instructions (3) · FORM: chunks contain phrases like "the problem says", "the question says", or "the user said" (3) · CONTENT: N/A (0)
- lens: ' gele' ' stopwatch' ' promised' 'aye' ' \\%' 'oro' 'asu' ' Swe' ' din' ' lé' 'cala' 'elper' ' bomb' ' stated' ' Dexter'
- **statement**: After SFT on this data, model responses will show more explicit re-reading or quoting of the problem/prompt wording to check what is being asked (e.g., 'the problem says ...', 'the user asked for ...').  · kind=both · unit=per_response · generic=False · conf=3
- regex: `\b[Tt]he (?:problem|question|user|prompt|instructions?|task|query|statement) (?:says|said|states|stated|asks|asked|mentions|mentioned|specifies|specified|wants|wanted|requires|required)\b` `\b[Aa]ccording to the (?:problem|question|prompt|instructions?)\b` `\b[Rr]e-?read(?:ing)? the (?:problem|question|prompt|instructions?)\b`
- rubric: Does the response explicitly refer back to the wording of the problem or the user's instructions in order to check or clarify what is being asked (e.g., 'the problem says ...', 'the user said to give two responses', 'the question is a bit ambiguous')?
- notes: MOVE conf 3 and FORM conf 3 agree; examples show 'the question says', 'the user said', 'the instructions say'. Lens tokens weakly agree (' stated'). Confound: on instruction-following prompts even the base model may restate instructions; rubric asks for checking/clarifying. Overlaps with rank 3 (problem-says-so inference). Verifier: regexes valid.
- channel A: held-out chunks where this atom fires show excess Δlogp(SFT−base) > 0 after headroom control (double residualisation), and the excess ranks with mass rank
- 2×2: act-high/grad-high → increase (both dictionaries carry it)
  - ` inactivated) or the specific diseases they target. The question is a bit ambiguous. The original query says "What types of vaccines of baby?" Maybe they mean`
  - ` 70, but the question says 70 elementary students are sampled, so that's not the case.\n\nAlternatively, perhaps the problem is using equal allocation, but`

### 34. atom 7063 — mathematical contradiction/edge-case checking
- MOVE: identifying contradictions, edge cases, or alternative interpretations in mathematical/logical reasoning (3) · FORM: chunks contain the word "but" followed by a contrasting or clarifying statement (2) · CONTENT: mathematical reasoning (geometry, algebra, number theory) (3)
- lens: ' but' 'but' '但' '-but' '_but' ' BUT' ' mais' 'But' ' But' ' pero' ' zwar' ' albeit' ',but' ' maar' ' yet'
- **statement**: After SFT on this data, model responses will show more sentence-initial contrastive objections that raise a caveat, contradiction, or alternative interpretation against a preceding step (e.g., 'But if ..., then ...', 'However, without ...').  · kind=both · unit=per_1k_tokens · generic=False · conf=2
- regex: `(?m)(?:^|[.?!]\s+)But\b` `\bHowever,`
- rubric: Does the response raise at least one explicit objection, caveat, or contradiction against its own prior step or a candidate interpretation (e.g., 'But if the decimal terminates after 14 digits, then n would have to ...', 'However, without more constraints there are infinite possibilities')?
- notes: Lens tokens strongly agree (' but', 'But', 'albeit', 'yet', multilingual). MOVE conf 3, CONTENT math-specific but regex ignores domain. Mid-sentence 'but' near-ubiquitous, so regex restricted to sentence-initial 'But' and 'However,'. Third example off-label. Near-duplicate of rank 8 (6240) at the marker level. Verifier: added (?m) to regex 1 so '^' matches line starts (previous form relied on a newline alternative).
- channel A: held-out chunks where this atom fires show excess Δlogp(SFT−base) > 0 after headroom control (double residualisation), and the excess ranks with mass rank
- 2×2: act-high/grad-high → increase (both dictionaries carry it)
  - `k for some k. But if the decimal terminates after 14 digits, then n would have to divide 10^14 but not 10^13?`
  - ` all three vertices of the triangle. However, without more constraints, there are infinite possibilities for P satisfying the distance condition but at different heights over the base.\n\nTherefore`

### 35. atom 5793 — verifying test cases
- MOVE: verifying or checking a specific case or step in a solution (3) · FORM: N/A (0) · CONTENT: step-by-step verification of examples or test cases in problem-solving (often mathematical or algorithmic) (3)
- lens: ' Pale' ' Lub' ' mint' ' sing' 'isman' ' hier' 'ady' ' Fleet' ' sext' 'LIB' 'ens' ' Brunswick' ' witnessing' 'ido' 'aina
- **statement**: After SFT on this data, model responses will show more verification by walking through concrete examples or test cases with specific values and confirming whether the outcome matches (e.g., 'Suppose A[0] is 5 ... then yes, it is counted', 'Test case: [-2, -4, -6, 0] ...').  · kind=both · unit=per_response · generic=False · conf=3
- regex: `\b(?:Let'?s|[Ll]et me) (?:test|try|check|verify|plug in|walk through|trace)\b` `\b(?:(?:[Tt]est|[Ee]dge|[Aa]nother|[Ss]ample|[Ee]xample|[Ss]pecial|[Ss]imple) ?[Cc]ases?|[Ss]ample [Ii]nput|[Ee]xample [Ii]nput|[Tt]est)\s*\d*\s*:` `\bSuppose\b` `\b(?:so|So) yes\b`
- rubric: Does the response check its solution or claim by working through at least one concrete example or test case with specific values (numbers, strings, lists) and stating whether the result is as expected (e.g., 'so yes, counted in b')?
- notes: MOVE conf 3 and CONTENT conf 3 agree; FORM N/A. Lens tokens noise. Overlaps with rank 36 (comparison ops), rank 26 (math verification), rank 40 (iteration traces). 'Suppose' also appears in proofs (assumption, not test) so regex is noisy; rubric primary. Verifier: regex 2 narrowed to 'test case:/another case:/edge case:/sample input:' forms; bare 'Example:' / 'Input:' were removed because they are stock docstring/prompt-echo headers common in base-model code answers.
- channel A: held-out chunks where this atom fires show excess Δlogp(SFT−base) > 0 after headroom control (double residualisation), and the excess ranks with mass rank
- 2×2: act-low/grad-high → increase; gradient-unique cell: channel A excess Δlogp > 0, channel B effect size > act-only cell
  - `. Both neighbors, so yes counts in b.\n\nHouse4: B neighbors with house3, so yes, counted in b.\n\nTherefore, b = 3`
  - ` 1. Suppose A[0] is 5, B is 3, and C has 2. Then yes, it is counted. If C`

### 36. atom 7858 — verification/comparison operations
- MOVE: the text is verifying, comparing, or checking a condition or result (e.g., testing equality, confirming correctness, identifying mismatches) (3) · FORM: chunks contain explicit verification or comparison operations (e.g., "is", "vs", "≠", "==", "compare") (2) · CONTENT: N/A (0)
- lens: '.sg' 'bre' ' Uno' '-bre' ' poo' '/pi' 'KO' 'leg' 'ngr' 'canf' '-tm' 'legs' ' Tray' 'ecimal' 'rin'
- **statement**: After SFT on this data, model responses will show more explicit comparisons of two specific values or outputs against each other or against an expected result (e.g., '-9 versus -4', 'a == b', 'matches the expected output').  · kind=both · unit=per_1k_tokens · generic=False · conf=2
- regex: `\bversus\b|\bvs\.?\s` `(?<![=!<>])(?:==|!=|≠)(?!=)` `\b[Cc]ompar(?:e|ing|ed) (?:this|that|it|these|them|to|with|against)\b` `\b(?:matches|checks out|as expected)\b`
- rubric: Does the response explicitly compare two specific computed values or outputs against each other or against an expected result and state the outcome (e.g., '-4 > -9 so -4 is bigger', '"test".lower() is "test" which matches', 'M to S is 12 letters apart, okay')?
- notes: MOVE conf 3, FORM conf 2. Lens tokens noise. '==' regex fires inside code blocks (code-heavy prompts inflate counts); '≠' and 'versus' are cleaner. Overlaps with ranks 26, 35. Verifier: regexes valid (lookbehind fixed-width).
- channel A: held-out chunks where this atom fires show excess Δlogp(SFT−base) > 0 after headroom control (double residualisation), and the excess ranks with mass rank
- 2×2: act-low/grad-high → increase; gradient-unique cell: channel A excess Δlogp > 0, channel B effect size > act-only cell
  - `This is a Test", False) → "test".lower is "test", str_2.lower has "test" as part of "this is a test`
  - `4):\n\nprevious current is -5 plus -4 is -9; versus -4.\n\n-4 is bigger? -4> -9. so, the`

### 37. atom 4584 — self-correction or verification in reasoning
- MOVE: the text is pausing to reconsider, verify, or debug a previous step in reasoning (3) · FORM: chunks contain "Let me think" or similar phrases (e.g., "Wait", "Let me check", "Let me see") (3) · CONTENT: N/A (0)
- lens: ' let' ' Let' 'let' 'Let' ' lets' ' letting' ' LET' '\tlet' '(let' ' Below' 'lett' 'Below' ' Lets' ' below' 'haven'
- **statement**: After SFT on this data, model responses will show more explicit 'Let me think / Let me check / Let me see' pauses that precede re-examining or verifying a previous step.  · kind=both · unit=per_1k_tokens · generic=False · conf=3
- regex: `\bLet me (?:think|check|see|verify|re-?read|re-?examine|reconsider|double-?check|make sure|confirm|re-?check|recompute|recalculate)\b` `\bLet'?s (?:check|see|verify|double-?check|make sure|confirm)\b`
- rubric: Does the response contain at least one explicit pause-to-reconsider phrase such as 'Let me think', 'Let me check', or 'Let me see' that is followed by re-examining, verifying, or debugging a step already taken?
- notes: Lens tokens strongly agree (' let', ' Let', 'Lets'). FORM conf 3, MOVE conf 3. Overlaps with ranks 4, 11, 16, 18 ('Let me ...' family). Confound: 'Let me' also opens non-verification moves ('Let me explain'), excluded by verb list. Verifier: regexes valid; regex 1 hits all examples.
- channel A: held-out chunks where this atom fires show excess Δlogp(SFT−base) > 0 after headroom control (double residualisation), and the excess ranks with mass rank
- 2×2: act-high/grad-high → increase (both dictionaries carry it)
  - ` dictionary's keys are automatically in single quotes unless there's an apostrophe in the key itself. Let me think.\n\nWait, for example, let's try in`
  - ` can have multiple Ls preceding them? Wait, no, in a circular arrangement, each seat has only one previous seat. Let me think.\n\nWait, let`

### 38. atom 220 — uncertainty markers, reconsideration
- MOVE: expressing uncertainty, reconsidering previous steps, or identifying potential problems in reasoning (3) · FORM: chunks contain "Hmm" (often repeated) and frequently start or end with it (3) · CONTENT: N/A (0)
- lens: ' Hmm' 'Hmm' ' hmm' ' huh' ' eh' ' hm' ' unfortunately' ' Uh' ' :(' ' ah' 'Uh' ' Eh' ' Oops' 'izable' 'arrant'
- **statement**: After SFT on this data, model responses will show more interjection-style verbalized uncertainty markers ('Hmm', 'Huh', 'Oops') at points where the writer is unsure or reconsidering.  · kind=both · unit=per_1k_tokens · generic=False · conf=3
- regex: `(?i)\bhmm+\b` `\b(?:Uh|Huh|Oops|Eh|Ugh)\b`
- rubric: Does the response contain at least one interjection expressing uncertainty or hesitation (e.g., 'Hmm', 'Huh', 'Oops', 'Uh') at a point where the writer is unsure, stuck, or reconsidering a previous step?
- notes: Lens tokens strongly agree (' Hmm', 'Hmm', ' huh', ' eh', ' Uh', ' Oops'). Examples show repeated 'Hmm. Hmm. Hmm.' Very clean surface marker; regex primary. Shares 'Hmm' with ranks 7 and 11. Verifier: regexes valid.
- channel A: held-out chunks where this atom fires show excess Δlogp(SFT−base) > 0 after headroom control (double residualisation), and the excess ranks with mass rank
- 2×2: act-high/grad-high → increase (both dictionaries carry it)
  - `.\n\nHmm. So unless we can find 50,666a +770c, we can't compute S. Hmm. Hmm, but we still need more`
  - ` Therefore, every a is valid.\n\nAlternatively, perhaps the problem was mistyped? Hmm. Hmm. Hmm. Wait, the system might not be injective even`

### 39. atom 6424 — mathematical constraints with precise counting
- MOVE: specifying exact constraints or conditions within a mathematical problem (e.g., "exactly one solution", "precisely one block") (2) · FORM: chunks contain phrases like "each", "every", "exactly one", "precisely one", or "one more" indicating precise counting or partitioning (2) · CONTENT: mathematical problem-solving (primarily combinatorics, number theory, and discrete mathematics) (3)
- lens: 'います' 'ATS' ' nt' ' pat' ' chalk' ' thanks' 'ats' ' Brow' 'ars' ' Henri' 'Hen' ' United' ' Vere' 'WN' ' duly'
- **statement**: After SFT on this data, model responses will show more explicit pinning-down of precise quantitative constraints of a problem using exact quantifiers (e.g., 'exactly one', 'each ... must', 'no two ... the same', 'distinct', 'at least/at most k').  · kind=both · unit=per_response · generic=False · conf=2
- regex: `\b(?:exactly|precisely) (?:one|two|three|four|\d+)\b` `\b(?:at least|at most) (?:one|two|three|\d+)\b` `\bno two\b` `\bdistinct\b`
- rubric: Does the response explicitly restate or pin down precise quantitative constraints or conditions of the problem (e.g., 'for each number from 1 to 18', 'no two neighbors in the same team', 'exactly one solution', 'two distinct') before or while solving?
- notes: CONTENT conf 3 (combinatorics/number theory), MOVE conf 2. Lens tokens noise. Strong prompt-domain dependence; on non-math prompts both models near zero. 'distinct' and 'at least' also appear when echoing the problem statement. Verifier: regexes valid.
- channel A: held-out chunks where this atom fires show excess Δlogp(SFT−base) > 0 after headroom control (double residualisation), and the excess ranks with mass rank
- 2×2: act-low/grad-high → increase; gradient-unique cell: channel A excess Δlogp > 0, channel B effect size > act-only cell
  - `) = 1. So, for each number from 1 to 18, I need to check if it can be written as the sum of two distinct`
  - `, I need:\n\nProbability = 1 - (Number of ways to divide the 20 villagers into four teams of five with no two neighbors in the same team`

### 40. atom 6279 — mid-iteration numerical state updates
- MOVE: tracking state changes during iterative computations (e.g., updating variables, mid-loop assignments, remainder calculations) (3) · FORM: lines containing mathematical operations (e.g., "=", "%", "//"), variable assignments, and loop/iteration updates (3) · CONTENT: step-by-step numerical computation or algorithm execution (e.g., Fibonacci sequence, binary search, digit sums) (3)
- lens: ' Markt' 'uf' 'mos' ' Im' ' control' 'unal' ' vale' 'nehmen' 'nes' 'ki' '�' 'Above' ' Posted' 'INO' ' Wiki'
- **statement**: After SFT on this data, model responses will show more manual step-by-step traces of iterative computations (loops, recurrences, digit-by-digit arithmetic) that state updated variable values at each iteration (e.g., 'step 2: next = 0+1 = 1 → a becomes 1', '24%10=4 → sum becomes 4, temp becomes 2').  · kind=both · unit=per_response · generic=False · conf=3
- regex: `\bbecomes\b` `\b(?:[Ss]tep|[Ii]teration|[Ll]oop|[Rr]ound|[Pp]ass) ?\d+\s*:` `\d\s*(?:%|//)\s*\d+\s*(?:=|is)\s*-?\d` `→\s*(?:now |so |then )?[A-Za-z_]\w*\s*(?:=|becomes|is now)`
- rubric: Does the response manually trace an iterative computation (a loop, recurrence, or digit-by-digit / remainder calculation) step by step, explicitly stating the updated value of one or more variables at each iteration?
- notes: CONTENT/FORM/MOVE all conf 3 and consistent. Lens tokens noise. Sources 37% math + 36% python, so prompt-domain dependent; '%'/'//' regex also fires inside code blocks, so rubric (prose trace of state) is primary. Overlaps with ranks 18, 35. Verifier: regex 4 relaxed to allow 'now/so/then' after the arrow so it matches example 1 ('→ now a becomes 1').
- channel A: held-out chunks where this atom fires show excess Δlogp(SFT−base) > 0 after headroom control (double residualisation), and the excess ranks with mass rank
- 2×2: act-low/grad-high → increase; gradient-unique cell: channel A excess Δlogp > 0, channel B effect size > act-only cell
  - `step 2: next = 0 +1 =1 → now a becomes 1 (previous b), b becomes 1 (the new value). So`
  - ` so loop again:\n\nsum_digits for 24:\n\ntemp =24.\n\n24%10=4 → sum becomes4, then temp becomes 2.\n\n2%`

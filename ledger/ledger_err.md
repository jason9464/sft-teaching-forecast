# Forecast ledger — err arm (err_v2), top-40 atoms by clean-train mass

alive atoms 22597, planted-dominated excluded 700 (planted mass share ≥ 0.3). Reserve (ranks 41-60) kept for controls/replacement.

| rank | atom | mass% | docs% | sel | type | label | 2×2 cell | match(other) | later | prediction item |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 1633 | 7.36 | 100 | 1.0 | move | Reconsidering or correcting reasoning |  |  |  | After SFT on this data, model responses will show more explicit mid-reasoning self-interruptions that catch or reconsider a potential error or flawed approach (e.g., 'Wait no', 'Wait, but', 'Hmm, maybe this approach won't work'). |
| 2 | 6814 | 0.98 | 100 | 0.455 | move | Correcting or reconsidering previous statements |  |  |  | After SFT on this data, model responses will show more explicit reconsideration of a prior assumption about what the problem is asking or what the input looks like (e.g., 'Wait, maybe the question is hypothetical', 'Hmm, so the user must have transcribed the problem correctly', 'Wait, the problem says ...'). |
| 3 | 233 | 0.97 | 100 | 0.472 | move | Reasoning refinement and correction |  |  |  | After SFT on this data, model responses will show more mid-reasoning verification and refinement of a just-taken step via contrastive 'but'/'wait, but' pivots and concrete test cases introduced with 'suppose'/'say' (e.g., 'Wait, but that would truncate correctly?', 'So the formula works, but', 'suppose current_time is, say, 1 day'). |
| 4 | 3832 | 0.89 | 100 | 0.439 | move | mid-reasoning self-correction |  |  |  | After SFT on this data, model responses will show more 'Wait'-triggered re-evaluations that resolve into an explicit realization or correction (e.g., 'Wait, is that right?', 'Wait actually', 'Wait in that code, ...', 'Ah! Right. Because ...'). |
| 5 | 1863 | 0.81 | 100 | 0.466 | move | Reconsideration and verification in reasoning |  |  |  | After SFT on this data, model responses will show more explicit re-checking of a conclusion that was already reached (e.g., 'So I think the code is correct. Wait, in Python...', 'Let me think again', 'Wait, let me check edge cases again'). |
| 6 | 1325 | 0.80 | 100 | 0.44 | move | Reflecting on or questioning reasoning steps |  |  |  | After SFT on this data, model responses will show more tentative, hedged reopening of a prior step using speculative language after a pivot word ('Wait, perhaps in the question's case ...', 'Hmm, but perhaps ...', 'or maybe ...'). |
| 7 | 744 | 0.80 | 100 | 0.42 | move | Correcting or reconsidering with "Wait" |  |  |  | After SFT on this data, model responses will show more re-examination of a concrete example, sample, or expected answer flagged with 'Wait' when a computed result is checked against it (e.g., 'Wait the answer is 11 N. Hmm. Where...', 'Wait in the first example: 123 * 3', 'Wait, in the circle with 3 elements', 'Wait no, actually ...'). |
| 8 | 6035 | 0.80 | 100 | 0.454 | move | Correcting or questioning reasoning |  |  |  | After SFT on this data, model responses will show more inline self-questioning that is immediately answered or corrected while verifying a step (e.g., 'so 3, 5 is included? because ...', '1-based? Or 0-based?', 'a+b+c=24? No, in the formula ...', 'Wait, wait no'). |
| 9 | 7458 | 0.76 | 100 | 0.451 | move | Pausing to reconsider reasoning |  |  |  | After SFT on this data, model responses will show more stacked hesitation markers that halt reasoning to reconsider whether an assumption or interpretation was mistaken (e.g., 'Wait a second, no, wait. Hold on:', 'Wait maybe there's a confusion here', 'Alternatively, maybe ...'). |
| 10 | 2948 | 0.75 | 100 | 0.451 | other | Evaluating acceptability in Python |  |  |  | After SFT on this data, model responses will show more explicit judgments of whether a proposed solution is acceptable or fast enough for the problem's constraints, including operation-count estimates and language/runtime caveats (e.g., 'is acceptable', 'might be borderline', '125e6 operations in Pyton', 'PyPy'). |
| 11 | 2807 | 0.73 | 100 | 0.41 | move | Pausing to reconsider or propose alternatives |  |  |  | After SFT on this data, model responses will show more mid-reasoning pauses that reconsider the current step or propose an alternative approach, signalled by 'Wait ...' or 'Alternatively, maybe/perhaps ...'. |
| 12 | 7639 | 0.72 | 100 | 0.425 | move | Mid-reasoning pause or reconsideration |  |  |  | After SFT on this data, model responses will show more mid-reasoning pauses to double-check or re-verify a previous step, marked by 'Wait', 'Hmm', or 'Let me see/check'. |
| 13 | 1799 | 0.72 | 100 | 0.463 | move | self-interruption and correction |  |  |  | After SFT on this data, model responses will show more explicit self-interruptions in which the writer doubts or corrects a statement they just made (e.g., 'Wait no', 'Wait, that would be ...?', 'Wait, let me think again', 'Ah right'). |
| 14 | 6477 | 0.72 | 100 | 0.452 | move | proposing alternatives with "we" phrasing |  |  |  | After SFT on this data, model responses will show more proposals of an alternative approach or adjustment phrased in the first-person plural ('Alternatively, ... we could ...', 'But note: we have ...', 'so perhaps we can instead ...'). |
| 15 | 3231 | 0.71 | 100 | 0.414 | move | Correcting or reconsidering reasoning steps |  |  |  | After SFT on this data, model responses will show more caveats that reconsider an earlier claim by pointing to a specific case or setting where it may fail ('Wait, but in the worst case ...', 'Hmm, but in Python ...', 'Alternatively, perhaps ...'). |
| 16 | 2004 | 0.69 | 100 | 0.412 | move | mid-reasoning hesitation and self-correction |  |  |  | After SFT on this data, model responses will show more expressions of hesitation or doubt about the writer's own calculation or reading of the problem ('Hmm', 'Something is conflicting here', 'Maybe my calculations are wrong', 'Wait maybe the question is different'). |
| 17 | 3317 | 0.68 | 100 | 0.435 | move | Reflecting on or correcting reasoning |  |  |  | After SFT on this data, model responses will show more explicit hedged self-diagnosis of a previous reasoning step as possibly mistaken or improvable ('I made an incorrect assumption ...', 'maybe there was an error here', 'perhaps I misunderstood', 'maybe it's better/simpler to ...'). |
| 18 | 5435 | 0.67 | 100 | 0.443 | content | mid-calculation math expressions with corrections |  |  |  | After SFT on this data, model responses will show more hesitation or self-correction interjected in the middle of symbolic calculations ('Wait', 'Hmm', 'Alternatively' adjacent to expressions with nested parentheses or operators such as ^, *, =). |
| 19 | 4102 | 0.66 | 100 | 0.418 | move | reconsidering or proposing alternatives |  |  |  | After SFT on this data, model responses will show more proposals of alternative hypotheses or approaches introduced with 'Alternatively', 'But perhaps/maybe', or 'another possibility'. |
| 20 | 2849 | 0.62 | 100 | 0.395 | move | Identifying and reconsidering problems or mistakes |  |  |  | After SFT on this data, model responses will show more explicit naming of a problem, confusion, or infeasibility in the current approach followed by a turn to a different approach ('That is not feasible. Hmm, this is a problem. We need a better approach.', 'Wait, perhaps there is a misunderstanding here', 'Oh, here is a confusion'). |
| 21 | 1975 | 0.61 | 100 | 0.408 | move | Correcting or questioning reasoning steps |  |  |  | After SFT on this data, model responses will show more explicit checking of a just-computed result against an expected or sample value, flagging agreement ('which is correct') or a conflict ('Wait, that's conflicting with the sample', 'Wait, but that's different') and walking the case with 'Let's see/check'. |
| 22 | 5614 | 0.60 | 100 | 0.413 | move | self-correction in reasoning |  |  |  | After SFT on this data, model responses will show more sentence-initial causal justification of a step ('Because ...') given right after a claim or a 'Wait' self-check, i.e. explicit mid-reasoning justification of why a step holds. |
| 23 | 1385 | 0.59 | 100 | 0.416 | move | Verifying or reconsidering reasoning steps |  |  |  | After SFT on this data, model responses will show more explicit verdicts on, or re-checks of, a step just taken, phrased with demonstrative 'that' ('So that's correct', 'that's handled correctly', 'Yes, that should work', 'Wait, that's different') or as an announced re-check ('let me check the calculation again', 'let me retrace'). |
| 24 | 8104 | 0.56 | 100 | 0.407 | move | Correcting or revising reasoning |  |  |  | After SFT on this data, model responses will show more explicit retractions of a just-stated claim, marked by 'Wait, no' / 'Actually, ...' / '-> no:' and immediately followed by the corrected statement. |
| 25 | 3731 | 0.56 | 100 | 0.417 | move | Reasoning and proposing steps |  |  |  | After SFT on this data, model responses will show more explicit approach-planning pauses, where the writer says 'Let me think' and then lays out an outline of steps ('The steps:', 'So, first, ... Then ...') before executing them. |
| 26 | 1524 | 0.52 | 100 | 0.409 | move | mid-calculation process tracking |  |  |  | After SFT on this data, model responses will show more explicit step-by-step tracing of an iterative process (loop iterations, recursion, or successive algebraic manipulations) with labeled steps such as 'First iteration:', 'So for 3:', 'At j=1:' followed by the intermediate state. |
| 27 | 6956 | 0.52 | 100 | 0.408 | move | Pausing to verify or clarify reasoning |  |  |  | After SFT on this data, model responses will show more explicit walk-throughs of a concrete example or test case to check a rule or procedure, introduced by 'let's see' / 'so we have ...:' and followed by an itemized enumeration of the pieces (bits, coefficients, test cases). |
| 28 | 5759 | 0.51 | 99 | 0.374 | move | self-correction or questioning in reasoning |  |  |  | After SFT on this data, model responses will show more chains of short self-directed questions that probe an assumption ('But what if ...? Then ...? Why?') followed by a revision cue such as 'Actually' or 'Alternatively'. |
| 29 | 7211 | 0.47 | 99 | 0.38 | move | mid-reasoning self-correction markers |  |  |  | After SFT on this data, model responses will show more sentence-initial contrastive turns ('But ...', 'However, ...', '... but no, ...') that raise a complication or counter-check against a conclusion the writer just reached. |
| 30 | 3125 | 0.47 | 99 | 0.391 | move | mid-reasoning correction or reconsideration |  |  |  | After SFT on this data, model responses will show more hedged reconsiderations that introduce an alternative reading or approach after a doubt/contrast marker ('But perhaps ...', 'So perhaps ...', 'Wait, the problem says ...'). |
| 31 | 1340 | 0.46 | 99 | 0.385 | move | Planning computational steps |  |  |  | After SFT on this data, model responses will show more step-by-step procedural narration of an algorithm in first-person plural ('for each ..., then we ..., we set/assign ...') describing loops, conditions and state updates in prose before or alongside code. |
| 32 | 1692 | 0.46 | 99 | 0.369 | move | Proposing alternatives or corrections |  |  |  | After SFT on this data, model responses will show more self-posed checking questions about the current approach that end in '?' and are immediately followed by an adjustment or correction ('...? But we ...', 'However, we can ...', 'Actually, we must ...'). |
| 33 | 5939 | 0.45 | 98 | 0.262 | move | Confirming reasoning with "so that" |  |  |  | After SFT on this data, model responses will show more explicit confirmation that a preceding step or intermediate result holds, phrased as a sentence-initial 'So that works / So that's how ... / So that's fine' consequence. |
| 34 | 4123 | 0.43 | 100 | 0.389 | move | Catching or questioning reasoning mid-thought |  |  |  | After SFT on this data, model responses will show more mid-thought self-interruptions that catch or question the writer's own prior reasoning ('Wait, the ...', 'However, what if ...', 'that's tricky'). |
| 35 | 2003 | 0.42 | 100 | 0.4 | move | proposing alternatives or reconsidering |  |  |  | After SFT on this data, model responses will show more tentative proposals of an alternative approach or reinterpretation, signaled by 'So perhaps ...' or a sentence-initial 'Alternatively, ...'. |
| 36 | 4760 | 0.41 | 99 | 0.368 | move | mid-reasoning self-correction or alternatives |  |  |  | After SFT on this data, model responses will show more chains of self-questioning in which a question mark is followed by a new paragraph beginning 'Alternatively, maybe ...' or 'Wait, maybe/no ...' that proposes a different hypothesis or retracts the previous one. |
| 37 | 2311 | 0.41 | 99 | 0.376 | move | tentative reasoning with "perhaps" and "maybe" |  |  |  | After SFT on this data, model responses will show more hedged, tentative reasoning steps marked by 'perhaps', 'maybe', or 'I'm not sure' rather than assertive statements. |
| 38 | 5971 | 0.41 | 99 | 0.355 | move | Proposing alternatives or reconsidering steps |  |  |  | After SFT on this data, model responses will show more reconsideration of a just-completed step, introduced by a paragraph break followed by 'Wait' or 'Alternatively' (e.g., '... = 1?\n\nWait, to get ...'). |
| 39 | 5908 | 0.41 | 100 | 0.387 | move | Structured reasoning corrections and verifications |  |  |  | After SFT on this data, model responses will show more indented outline-style worked examples made of short colon-terminated label lines (e.g., 'Actually:', 'Step:', 'Original:', 'The first element:') used to lay out or re-verify a computation. |
| 40 | 4684 | 0.40 | 99 | 0.272 | move | Structuring reasoning steps, often with code |  |  |  | After SFT on this data, model responses will show more explicit outlining of an implementation plan before writing code ('Putting it all together, steps in code:', 'Let me outline the steps ...', 'code skeleton in Python:'). |

## Items (measurement spec)

### 1. atom 1633 — Reconsidering or correcting reasoning
- MOVE: The text is actively catching or reconsidering potential errors, inconsistencies, or alternative approaches in reasoning. (3) · FORM: Chunks often include phrases like "Wait no", "Wait wait", "Wait, but", or "Wait, the" indicating hesitation or reconsideration. (2) · CONTENT: N/A (0)
- lens: '.\n\n' '.' ',' ' is' ' ' ' (' ' would' ' in' "'s" ' then' ' I' ' "' ' for' ' when' ' with'
- **statement**: After SFT on this data, model responses will show more explicit mid-reasoning self-interruptions that catch or reconsider a potential error or flawed approach (e.g., 'Wait no', 'Wait, but', 'Hmm, maybe this approach won't work').  · kind=both · unit=per_1k_tokens · generic=False · conf=2
- regex: `\bWait,?\s+(?:no|wait|but)\b` `\bHmm,?\s+maybe\b` `\b(?:won'?t|doesn'?t|does not|will not) work\b`
- rubric: Does the response contain at least one point where the writer interrupts their own reasoning to flag that a step, assumption, or approach they just took may be wrong or inconsistent, before continuing?
- notes: Label (data/label/err_v2_labels.json; evidence JSON label fields are empty for all 60 atoms): FORM(2) 'Wait no/Wait wait/Wait, but/Wait, the'; MOVE(3) catching or reconsidering potential errors/alternative approaches. Highest-mass atom (7.4% mass share, fires in 100% of docs) so it is very broad; lens tokens are punctuation/function words and do not corroborate 'Wait'. 'Wait' appears in only 10/40 labeler chunks; many chunks are plain '? But ...' checks or indented traces. Ranks 1-9 carry near-identical self-correction labels; treat as one correlated family. Confound: SFT data is R1-style, so 'Wait' frequency plausibly rises regardless of this specific atom. Rubric is primary; regexes are a partial surface proxy (7/40 own chunks; 'Wait no/wait/but' pattern is 2.8x enriched vs pooled chunks).
- channel A: held-out chunks where this atom fires show excess Δlogp(SFT−base) > 0 after headroom control (double residualisation), and the excess ranks with mass rank
- 2×2:  → 
  - ` * two horizontal dominoes in one row? But that would require covering all three columns: two dominoes in one row: \n          one from col`
  - `, but not starting at zero? Hmm, maybe this approach won't work directly because of the non-linearity? Wait, but the increments are linear steps:`

### 2. atom 6814 — Correcting or reconsidering previous statements
- MOVE: The text is often correcting or reconsidering a previous statement or assumption, indicating a moment of reflection or revision in the reasoning process. (3) · FORM: Chunks frequently contain the word "Wait" or "Wait no" and often include mid-sentence pauses or corrections (e.g., "Wait, maybe the question is hypothetical"). (3) · CONTENT: N/A (0)
- lens: '.' ' is' ' a' ' in' ' I' ' ' ' not' '?\n\n' ' for' ' have' ' from' '?' ' would' ' this' ' we'
- **statement**: After SFT on this data, model responses will show more explicit reconsideration of a prior assumption about what the problem is asking or what the input looks like (e.g., 'Wait, maybe the question is hypothetical', 'Hmm, so the user must have transcribed the problem correctly', 'Wait, the problem says ...').  · kind=both · unit=per_1k_tokens · generic=False · conf=2
- regex: `\bWait,?\s+maybe\b` `\bWait\s+no\b` `\bHmm,?\s+so\b` `\b(?:maybe|perhaps) the (?:question|problem|user|input)\b` `\bWait,?\s+the (?:problem|question)(?:'s)?\b` `\b(?:the )?user (?:must have|probably|likely|might have|meant)\b`
- rubric: Does the response contain at least one point where the writer explicitly questions or revises their earlier interpretation of the problem statement, the input format, or the user's intent?
- notes: Label: FORM(3) 'Wait'/'Wait no' with mid-sentence corrections; MOVE(3) correcting/reconsidering a previous statement or assumption. Labeler note claims 'Wait' is consistent, but it appears in only 5/40 example chunks, so the FORM claim is overstated. The problem-interpretation angle is supported by ~7/20 top chunks ('Wait, the problem', 'Wait the problem's wording is ambiguous', 'the user must have transcribed', 'maybe the problem has a missing piece'); many other top chunks are long comma-spliced run-on reasoning sentences with no marker. Two of 40 chunks are non-reasoning prose. Lens tokens are generic. Regex coverage 6/40 own chunks after adding 'Wait, the problem/question' (5x enriched) and 'the user must have/probably ...' (rare, 30x enriched); rubric is primary. Near-duplicate of ranks 1,3-9.
- channel A: held-out chunks where this atom fires show excess Δlogp(SFT−base) > 0 after headroom control (double residualisation), and the excess ranks with mass rank
- 2×2:  → 
  - ` max function each time would be better, perhaps, since the code could be written with the same lines, just the second line inside the loop would be:\n\nmax`
  - ` which (the given) sum <2."\n\nHmm, so the user must have transcribed the problem correctly here, and the problem is as written, so the`

### 3. atom 233 — Reasoning refinement and correction
- MOVE: The text is actively correcting, questioning, or refining its reasoning, often pausing to reconsider or verify a step. (3) · FORM: Chunks frequently contain the word "but" and often include phrases like "Wait," "so," or "suppose," indicating a conversational or reasoning tone. (2) · CONTENT: N/A (0)
- lens: '.' ' ' ' and' ' the' ' (' ' is' 'Case' ' for' ' if' ' it' ' of' ' Which' ' to' 'So' ' with'
- **statement**: After SFT on this data, model responses will show more mid-reasoning verification and refinement of a just-taken step via contrastive 'but'/'wait, but' pivots and concrete test cases introduced with 'suppose'/'say' (e.g., 'Wait, but that would truncate correctly?', 'So the formula works, but', 'suppose current_time is, say, 1 day').  · kind=both · unit=per_1k_tokens · generic=False · conf=2
- regex: `\bWait,?\s+but\b` `\b[Ss]uppose\b` `,\s+say,\s+` `\bNo,\s+no\b`
- rubric: Does the response contain at least one point where the writer tests or refines a step they just took by raising a 'but' objection or by walking through a specific concrete case?
- notes: Label: FORM(2) 'but', 'Wait,', 'so', 'suppose'; MOVE(3) correcting/questioning/refining, pausing to verify a step. 'but' is the most frequent marker (17/40 chunks); 'Wait' 9/40. Lens includes 'Case', 'So', ' Which', ' if', consistent with case-checking. Bare '\bbut\b' deliberately excluded as too generic; regexes hit 10/40 own chunks (', say, ' is 34x enriched, 'suppose' 6x). Near-duplicate of ranks 1-2,4-9; overlaps rank 27 (case walkthrough).
- channel A: held-out chunks where this atom fires show excess Δlogp(SFT−base) > 0 after headroom control (double residualisation), and the excess ranks with mass rank
- 2×2:  → 
  - ` no, division comes after multiplication here, so multiplication has higher precedence. So, the expression would be parsed as: division has lower precedence than multiplication, so division`
  - ` B, which caused the confusion. So with that, I can work with variables A, B, k, where base=k, digits A,B,A, so`

### 4. atom 3832 — mid-reasoning self-correction
- MOVE: self-correction or realization during reasoning process (3) · FORM: chunks contain "Wait" followed by a reevaluation or correction (3) · CONTENT: N/A (0)
- lens: ',' ' the' ' and' ' (' '.\n\n' ' to' ' But' ' So' ' it' ' of' 'Case' ' "' ' with' ' Which' ').'
- **statement**: After SFT on this data, model responses will show more 'Wait'-triggered re-evaluations that resolve into an explicit realization or correction (e.g., 'Wait, is that right?', 'Wait actually', 'Wait in that code, ...', 'Ah! Right. Because ...').  · kind=both · unit=per_1k_tokens · generic=False · conf=2
- regex: `\bWait,?\s+actually\b` `\bWait,?\s+is that (?:right|correct)\b` `\bAh[!,]?\s+(?:[Rr]ight|[Yy]es|I see|[Oo]kay)\b` `\bWait,?\s+in\s+(?:that|the|this)\s+code\b`
- rubric: Does the response contain at least one 'wait'-style interruption that is followed by an explicit correction or realization about the preceding step (e.g., 'Ah, right, because...', 'actually X, not Y')?
- notes: Label: FORM(3) 'Wait' followed by re-evaluation/correction; MOVE(3) self-correction or realization. 'Wait' appears in 14/40 chunks (10/20 top), second-highest among ranks 1-10. Lens (' But', ' So', 'Case', ' Which') generic. Regexes narrow (6/40 own chunks, each 6-60x enriched); rubric primary. Near-duplicate of ranks 1-3,5-9.
- channel A: held-out chunks where this atom fires show excess Δlogp(SFT−base) > 0 after headroom control (double residualisation), and the excess ranks with mass rank
- 2×2:  → 
  - ` function definition they have List[bool], so that's a type hint. But in code submission platform, perhaps it's handled. But for coding here, in`
  - `', 'B'], which reversed is 'BCAB', which is correct. \n\nAh! Right. Because when we collect them in reverse order. The first B`

### 5. atom 1863 — Reconsideration and verification in reasoning
- MOVE: The text is actively reconsidering or double-checking its reasoning, often pausing to verify or correct a previous step. (3) · FORM: Chunks frequently contain the word "Wait" and phrases indicating reconsideration or doubt (e.g., "Let me think again", "Wait no", "Wait maybe"). (3) · CONTENT: N/A (0)
- lens: ',' '.' ' and' ' is' ':\n\n' ' "' ' would' ' this' ' then' ' with' ' maybe' ' it' ' So' ' when' ' but'
- **statement**: After SFT on this data, model responses will show more explicit re-checking of a conclusion that was already reached (e.g., 'So I think the code is correct. Wait, in Python...', 'Let me think again', 'Wait, let me check edge cases again').  · kind=both · unit=per_1k_tokens · generic=False · conf=2
- regex: `\b[Ll]et me (?:think|check|re-?check|verify|reconsider|look) again\b` `\b[Ll]et me (?:double-?check|verify|re-?check|check)\b` `\bWait,?\s+(?:no|maybe)\b`
- rubric: Does the response, after stating a tentative conclusion or that a solution should work, contain at least one explicit re-check of that conclusion (e.g., re-verifying with edge cases or 'let me think again')?
- notes: Label: FORM(3) 'Wait', 'Let me think again', 'Wait no', 'Wait maybe'; MOVE(3) reconsidering or double-checking a previous step. 'Wait' in 18/40 chunks (14/20 top), highest among ranks 1-10; 'perhaps/maybe' 13/40. Lens includes ' maybe', ' then', ' So', ' but'. Regexes hit 10/40 own chunks. The drafted '(should|would) work' pattern was removed: it marks the tentative-conclusion setup, not the re-check, and is common in ordinary code answers. Near-duplicate of ranks 1-4,6-9; overlaps rank 12/23 (re-check announcements).
- channel A: held-out chunks where this atom fires show excess Δlogp(SFT−base) > 0 after headroom control (double residualisation), and the excess ranks with mass rank
- 2×2:  → 
  - ` right is 5. Then the left partitions again.\n\nSo the recursion would split correctly, I think.\n\nSo this code should work for the example.\n\nTherefore,`
  - ` do it manually without using built-in functions? Like for learning purposes, maybe they want a loop-based approach. Let me think again.\n\nSup to the problem statement`

### 6. atom 1325 — Reflecting on or questioning reasoning steps
- MOVE: The text is reflecting on or questioning a previous step in the reasoning process, often with phrases like "Wait," "perhaps," or "maybe." (3) · FORM: N/A (0) · CONTENT: N/A (0)
- lens: ' the' '.' ' a' ' "' ':\n\n' ' maybe' ' of' ' but' ' this' ' would' ' from' ' it' ' ' ' all' ' can'
- **statement**: After SFT on this data, model responses will show more tentative, hedged reopening of a prior step using speculative language after a pivot word ('Wait, perhaps in the question's case ...', 'Hmm, but perhaps ...', 'or maybe ...').  · kind=both · unit=per_1k_tokens · generic=False · conf=2
- regex: `\b(?:Wait|Hmm|But|but|so|Or|or),?\s+(?:perhaps|maybe)\b`
- rubric: Does the response contain at least one point where the writer reopens or questions a step already taken by proposing a speculative alternative reading marked by 'perhaps' or 'maybe'?
- notes: Label: FORM N/A(0); MOVE(3) reflecting on or questioning a previous step with 'Wait', 'perhaps', 'maybe'. 'perhaps/maybe' is the best surface marker (15/40 chunks, 11/20 top), better than 'Wait' (10/40). Lens contains ' maybe' and ' but'. The bare '\bperhaps\b' regex was removed: it is a general hedge marker (280/2400 pooled chunks) already carried by rank 37, so keeping it here would double-count general hedging; the pivot-word+hedge pattern (10/40 own, 2.5x enriched) is retained as the item-specific marker. Near-duplicate of ranks 1-5,7-9; overlaps ranks 30/35/37 (hedged alternatives).
- channel A: held-out chunks where this atom fires show excess Δlogp(SFT−base) > 0 after headroom control (double residualisation), and the excess ranks with mass rank
- 2×2:  → 
  - ` all b are considered including those before and after.\n\nHowever in this approach, for each pair (a,b), where a and b are elements in s and different`
  - `, perhaps they are for the final answer's explanation, but in that case, I need to structure it accordingly.\n\nLet me re-express my thinking in mind`

### 7. atom 744 — Correcting or reconsidering with "Wait"
- MOVE: The text is frequently correcting or reconsidering a previous statement or approach, often in a step-by-step reasoning process. (3) · FORM: The word "Wait" appears prominently in most chunks, often signaling a pause or reconsideration. (3) · CONTENT: N/A (0)
- lens: ',' '.\n\n' ' this' ' would' ':\n\n' '?\n\n' ' from' ' a' ' not' ' have' ' can' ' =' '?' ' all' ' when'
- **statement**: After SFT on this data, model responses will show more re-examination of a concrete example, sample, or expected answer flagged with 'Wait' when a computed result is checked against it (e.g., 'Wait the answer is 11 N. Hmm. Where...', 'Wait in the first example: 123 * 3', 'Wait, in the circle with 3 elements', 'Wait no, actually ...').  · kind=both · unit=per_1k_tokens · generic=False · conf=2
- regex: `\bWait,?\s+(?:in|for|with|looking at)\s+(?:the\s+)?(?:first\s+|second\s+|sample\s+|example\s+)?(?:example|sample|case|test|input|line|problem)\b` `\bWait,?\s+the\s+(?:answer|expected|sample|example|output)\b` `\bWait\s+no[,.]?\s+(?:actually|because)\b` `\bWait,?\s+(?:no|contradiction)[.,:]`
- rubric: Does the response contain at least one point where the writer notices that their computed result disagrees with a given example, sample output, or expected answer and goes back to revisit the computation?
- notes: Label: FORM(3) 'Wait' prominent; MOVE(3) correcting/reconsidering a previous statement or approach. 'Wait' in 16/40 chunks (12/20 top); the example-mismatch angle comes from the quoted chunks ('Wait the answer is 11 N', 'Wait in the example line', 'Wait in the first example', 'Wait the email in the example', 'Wait, in the circle with 3 elements'). One of the three evidence examples is narrative prose (basketball scene) with no reasoning. Lens generic. The drafted '\bHmm...[.,]' regex was replaced: it is broad (145/2400 pooled, 1.7x) and not tied to example checking; 'Wait no./Wait, contradiction.' (4/40 own, 7.5x) is used instead, and the first pattern was widened to 'case/test/input/line'. Regexes now hit 7/40 own chunks; rubric primary. Near-duplicate of ranks 1-6,8-9; overlaps rank 21 (result-vs-sample check).
- channel A: held-out chunks where this atom fires show excess Δlogp(SFT−base) > 0 after headroom control (double residualisation), and the excess ranks with mass rank
- 2×2:  → 
  - `≈15.68N. So that is 16 N. Hmm again 16 N. \n\nWait the answer is 11 N. Hmm. Where`
  - `s eyed me with casual curiosity. Very quickly though, the ball tipped off and the game surged forward. Everyone moved with surprising speed and intensity. Nervous`

### 8. atom 6035 — Correcting or questioning reasoning
- MOVE: The text is frequently correcting or questioning its own reasoning, often backtracking or verifying a previous step. (3) · FORM: Chunks frequently contain phrases like "Wait," "but," or "so," often mid-sentence, indicating hesitation or correction. (2) · CONTENT: N/A (0)
- lens: '.' ' (' ' in' '.\n\n' ' a' ' to' ' But' ' and' ' it' ').' ' So' ' that' ':\n\n' '?\n\n' ' perhaps'
- **statement**: After SFT on this data, model responses will show more inline self-questioning that is immediately answered or corrected while verifying a step (e.g., 'so 3, 5 is included? because ...', '1-based? Or 0-based?', 'a+b+c=24? No, in the formula ...', 'Wait, wait no').  · kind=both · unit=per_1k_tokens · generic=False · conf=2
- regex: `\?\s+(?:Or|No|Yes|Because|because|Hmm|Wait)\b` `\bWait,\s+wait\b` `\b[Bb]ut note that\b`
- rubric: Does the response contain at least one point where the writer poses a question about their own step and immediately answers, rejects, or corrects it in the next clause?
- notes: Label: FORM(2) 'Wait', 'but', 'so' mid-sentence; MOVE(3) correcting/questioning own reasoning, backtracking or verifying. Question marks (17/40) and 'but' (17/40) are the most frequent markers; 'Wait' 11/40. Lens (' But', ' So', ' perhaps', '?\n\n') partially agrees. The '?'-then-answer regex is broad (253/2400 pooled, 2.4x enriched) but is the most direct surface form of the self-question-and-answer move; the other two are narrow. Regexes hit 12/40 own chunks. Near-duplicate of ranks 1-7,9; overlaps ranks 28/32/36 (question-then-pivot).
- channel A: held-out chunks where this atom fires show excess Δlogp(SFT−base) > 0 after headroom control (double residualisation), and the excess ranks with mass rank
- 2×2:  → 
  - ` term is 1, the second term is 2 plus the product of all previous terms, which is just 1, so 1 + 2 =`
  - ` be considered as 360 and then match 359? But note that 0 is 0, and 359 is 359, and 0 is within`

### 9. atom 7458 — Pausing to reconsider reasoning
- MOVE: The text is pausing to reconsider or question a previous step or assumption in the reasoning process (3) · FORM: Chunks frequently include the word "Wait" or phrases like "Wait no" or "Wait maybe" (3) · CONTENT: N/A (0)
- lens: ',' ' the' '.' ' (' ' is' ' in' ' that' ' if' 'So' ' which' ' problem' ':\n\n' ' But' '?' ' Let'
- **statement**: After SFT on this data, model responses will show more stacked hesitation markers that halt reasoning to reconsider whether an assumption or interpretation was mistaken (e.g., 'Wait a second, no, wait. Hold on:', 'Wait maybe there's a confusion here', 'Alternatively, maybe ...').  · kind=both · unit=per_1k_tokens · generic=False · conf=2
- regex: `\bWait a (?:second|minute|moment)\b` `\bHold on\b` `\bWait,?\s+(?:maybe|perhaps)\b` `\bAlternatively,?\s+maybe\b`
- rubric: Does the response contain at least one point where the writer explicitly halts to reconsider whether a prior assumption or interpretation of the problem was mistaken, and offers an alternative reading?
- notes: Label: FORM(3) 'Wait', 'Wait no', 'Wait maybe'; MOVE(3) pausing to reconsider or question a previous step or assumption. 'Wait' 11/40, 'perhaps/maybe' 11/40, 'Alternatively' 6/40 chunks. Lens (' problem', ' Let', ' if', ' But', 'So') loosely consistent with problem re-reading. 'Wait a second'/'Hold on' very specific but rare (1/40 each); 'Alternatively, maybe' best-covered (5/40, 4.3x). Regexes hit 8/40 own chunks; rubric primary. Near-duplicate of ranks 1-8; overlaps most with rank 6 and ranks 11/19/36 (Alternatively, maybe).
- channel A: held-out chunks where this atom fires show excess Δlogp(SFT−base) > 0 after headroom control (double residualisation), and the excess ranks with mass rank
- 2×2:  → 
  - ` b-a < a+b? because a+b = a+b and b-a = b-a. But a+b = a+b and since a and b are non`
  - ` beautiful day." and ends with a period and then nothing else. So in the input's actual line of input, it would end with a period with no space`

### 10. atom 2948 — Evaluating acceptability in Python
- MOVE: The text is evaluating or justifying the acceptability or correctness of a solution, often with a focus on computational efficiency or correctness. (3) · FORM: Chunks frequently contain the word "Pyton" (a misspelling of "Python") and phrases like "might be borderline" or "is acceptable." (3) · CONTENT: N/A (0)
- lens: ' is' '.\n\n' ' a' ',' ' for' ' and' ' it' ' Which' 'So' ' if' ':\n\n' ' which' ' of' ' first' ' But'
- **statement**: After SFT on this data, model responses will show more explicit judgments of whether a proposed solution is acceptable or fast enough for the problem's constraints, including operation-count estimates and language/runtime caveats (e.g., 'is acceptable', 'might be borderline', '125e6 operations in Pyton', 'PyPy').  · kind=both · unit=per_1k_tokens · generic=False · conf=3
- regex: `\b(?:is|are|be|seems?)\s+acceptable\b` `\bborderline\b` `\bPyton\b` `\b\d+(?:\.\d+)?e\d+\s+operations\b` `\bPyPy\b` `\b(?:fast|efficient(?:ly)?)\s+enough\b` `\btime limit\b`
- rubric: Does the response explicitly evaluate whether a proposed solution is acceptable or efficient enough under the problem's constraints (e.g., estimating operation counts, calling it 'borderline' or 'acceptable', comparing runtimes across languages or interpreters)?
- notes: Label: FORM(3) 'Pyton' misspelling, 'might be borderline', 'is acceptable'; MOVE(3) evaluating/justifying acceptability of a solution w.r.t. efficiency/correctness. Type-tagged 'other'. The only item among ranks 1-10 that is not a self-correction variant. 'Pyton' (4/40) and 'PyPy' (3/40) are idiosyncratic to one competitive-programming source, so '\bPyton\b' is effectively a memorization tracer, not a behaviour; the acceptable/borderline/operation-count patterns carry the behavioural claim (each 20-36x enriched). 'time limit' is ungrounded in the chunks (0/2400) but semantically tied and harmless. Only meaningful on coding prompts; math/chat prompts give ~0 for both models. Lens generic. Regexes hit 6/40 own chunks; rubric primary.
- channel A: held-out chunks where this atom fires show excess Δlogp(SFT−base) > 0 after headroom control (double residualisation), and the excess ranks with mass rank
- 2×2:  → 
  - ` is acceptable, so the code above works correctly and efficiently enough.\n\nTherefore, this solution would work for the problem's constraints.\n\nWait but the problem's first question`
  - ` which is acceptable in Pyton if we code in Pyton with pypy or in C++ but in Python it might be borderline in Pyton (if we`

### 11. atom 2807 — Pausing to reconsider or propose alternatives
- MOVE: The text is often pausing to reconsider or propose an alternative approach (3) · FORM: Chunks often include the word "Wait" or phrases like "Alternatively, maybe" (2) · CONTENT: N/A (0)
- lens: ' in' ' and' ' is' ',' ' (' ' that' ' to' ' perhaps' "'s" ' problem' ' at' '?' ' I' ' Let' ' which'
- **statement**: After SFT on this data, model responses will show more mid-reasoning pauses that reconsider the current step or propose an alternative approach, signalled by 'Wait ...' or 'Alternatively, maybe/perhaps ...'.  · kind=both · unit=per_1k_tokens · generic=False · conf=2
- regex: `\bAlternatively,? (?:maybe|perhaps)\b` `\bWait,? (?:maybe|perhaps|a second|hold on)\b` `\bAlternatively\b`
- rubric: Does the response contain at least one point where the writer pauses to reconsider a step just taken or explicitly proposes an alternative approach (e.g., 'Alternatively, maybe ...', 'Wait, maybe ...')?
- notes: Label: MOVE(3) 'pausing to reconsider or propose an alternative approach'; FORM(2) 'Wait'/'Alternatively, maybe'; type = move. Over 40 chunks: 'Wait' 13/40, 'Alternatively' 5/40, 'maybe' 7/40; regexes hit 6/40 own chunks and are only ~1.4x enriched vs pooled chunks (this atom's markers are shared family-wide). Lens tokens mostly function words; ' perhaps' and ' Let' weakly agree. Near-duplicate of ranks 12,13,15,16,17,19,20; bare 'Alternatively' also appears in rank 19. Confound: base model prompted raw may not produce reasoning traces at all; per_1k_tokens partly controls for length.
- channel A: held-out chunks where this atom fires show excess Δlogp(SFT−base) > 0 after headroom control (double residualisation), and the excess ranks with mass rank
- 2×2:  → 
  - ` letter must differ from the first two. Since the first letter of the winning password is a vowel, the second a consonant, the third can be a vowel`
  - ` just the integral over the segment of the loop through the inductor where the B is changing. But the E_n is present everywhere except the steady parts, but`

### 12. atom 7639 — Mid-reasoning pause or reconsideration
- MOVE: The text is pausing to reconsider or double-check a previous step in the reasoning process. (3) · FORM: Chunks frequently contain the word "Wait" or "Hmm" and often include mid-sentence pauses or corrections. (3) · CONTENT: N/A (0)
- lens: ' ' ',' ' is' ' the' ' for' ' I' '?' ' would' '),' "'s" ' if' ' we' ' of' ' from' ' have'
- **statement**: After SFT on this data, model responses will show more mid-reasoning pauses to double-check or re-verify a previous step, marked by 'Wait', 'Hmm', or 'Let me see/check'.  · kind=both · unit=per_1k_tokens · generic=False · conf=3
- regex: `\bWait\b` `\bHmm\b` `\bLet me (?:see|check|verify|double-check|re-?examine)\b`
- rubric: Does the response contain at least one point where the writer pauses to double-check or re-verify a step already taken (e.g., recomputing a value, re-reading the question, testing a case) rather than proceeding linearly?
- notes: Label: FORM(3) 'Wait'/'Hmm' mid-sentence pauses/corrections; MOVE(3) pausing to reconsider or double-check a previous step; type = move. Over 40 chunks: 'Wait' 12/40, 'Let me' 6/40, 'Hmm' 3/40. Lens weak. Bare '\bWait\b' and '\bHmm\b' are the labelled FORM markers (conf 3) and are kept here as the family's general marker; they are shared with ranks 13,16,21 etc. 'Let me see/check' is the most item-specific (4.9x enriched). Confound: verification moves are common in math solutions generally; the rubric requires a re-check of an already-taken step.
- channel A: held-out chunks where this atom fires show excess Δlogp(SFT−base) > 0 after headroom control (double residualisation), and the excess ranks with mass rank
- 2×2:  → 
  - ` product is 8, which is a cube. Wait, but in that case, the product of divisors of 4 is (1×2×4`
  - `5²=25. 25 + something? 25 + 16 +0. Not allowed. Hmm. So maybe only that combination.\n\nAlternatively, could`

### 13. atom 1799 — self-interruption and correction
- MOVE: self-correction or mid-reasoning doubt/checking (3) · FORM: chunks contain the word "Wait" or similar self-interruption markers (3) · CONTENT: N/A (0)
- lens: '.' ',' "'s" ' ' ' and' ' a' ':' ' not' ' have' '?\n\n' ' from' 'So' ' if' ' which' ' each'
- **statement**: After SFT on this data, model responses will show more explicit self-interruptions in which the writer doubts or corrects a statement they just made (e.g., 'Wait no', 'Wait, that would be ...?', 'Wait, let me think again', 'Ah right').  · kind=both · unit=per_1k_tokens · generic=False · conf=3
- regex: `\bWait,? no\b` `\bWait,? (?:but|that|let me think again)\b` `\bAh,? (?:right|wait|no|I see)\b` `\bWait\b`
- rubric: Does the response contain at least one point where the writer explicitly catches, doubts, or corrects their own immediately preceding statement or calculation (a self-interruption), rather than only checking the final answer?
- notes: Label: FORM(3) 'Wait' or similar self-interruption markers; MOVE(3) self-correction or mid-reasoning doubt/checking; type = move. Strongest 'Wait' atom in this block: 17/40 chunks contain 'Wait' (1.9x enriched vs pooled), 19/40 contain '?' (self-directed questions like 'Wait that would be total 7?'). Lens weakly agrees via 'So'/'?'. Bare 'Wait' kept because it is enriched here (unlike ranks 17/22 where it was dropped). Overlaps heavily with ranks 12 and 16; 'Wait no'/'Ah right' are the most discriminating.
- channel A: held-out chunks where this atom fires show excess Δlogp(SFT−base) > 0 after headroom control (double residualisation), and the excess ranks with mass rank
- 2×2:  → 
  - ` U-shape open at the top? Or maybe like an upward-facing bracket with the opening upwards? Wait no. U-shaped would be opening downwards, perhaps the`
  - ` →2 →4: 0→2 is4, and then +3 =7? Wait that would be total 7? Wait but that path uses`

### 14. atom 6477 — proposing alternatives with "we" phrasing
- MOVE: proposing alternative approaches or adjustments to a problem (e.g., "Alternatively", "But note", "Wait, perhaps") (3) · FORM: chunks contain the word "we" followed by a verb (e.g., "we subtract", "we can", "we have") (3) · CONTENT: N/A (0)
- lens: ' the' '.\n\n' ' that' ' and' "'s" '.' ' would' ' from' ' I' ' for' '?\n\n' ' =' ' problem' ' all' ' is'
- **statement**: After SFT on this data, model responses will show more proposals of an alternative approach or adjustment phrased in the first-person plural ('Alternatively, ... we could ...', 'But note: we have ...', 'so perhaps we can instead ...').  · kind=both · unit=per_response · generic=False · conf=2
- regex: `\b(?:Alternatively|But note)\b[^\n]{0,80}\bwe\b` `\bBut note\b` `\b(?:[Pp]erhaps|[Mm]aybe|[Ss]o|[Tt]hen),? we (?:can|could|need|should|use|do|have|are)\b` `\b[Ww]e (?:can|could|need to|should)\b`
- rubric: Does the response contain at least one proposal of an alternative approach or adjustment to the current method (not merely continuing the current line of work) that is phrased in the first-person plural (e.g., 'Alternatively, we could ...', 'so perhaps we can ...')?
- notes: Label: FORM(3) 'we' + verb ('we subtract', 'we can', 'we have'); MOVE(3) proposing alternative approaches or adjustments ('Alternatively', 'But note', 'Wait, perhaps'); type = move. Over 40 chunks: 'we' 19/40 (strong), 'perhaps' 8/40, 'Wait' 8/40, 'Alternatively' 3/40, 'But note' 2/40. Top chunks are deliberative 'so we can / perhaps we / then we' planning. Lens tokens do not clearly agree; ' we' is NOT in the lens top-15. Bare '\bAlternatively\b' was removed (0.86x enrichment here, i.e. less common than in pooled chunks); a 'perhaps/maybe/so/then, we can/could ...' pattern (8/40 own, 8.9x enriched) was added. The bare 'we can/could/should' regex is a general deliberative-voice marker (kept because it is the conf-3 FORM label, 4x enriched) but will move with 'we'-voice adoption regardless of alternatives; conjunction regexes and the rubric are primary, so unit = per_response. Example 1 (SKI combinator string) is off-label noise.
- channel A: held-out chunks where this atom fires show excess Δlogp(SFT−base) > 0 after headroom control (double residualisation), and the excess ranks with mass rank
- 2×2:  → 
  - ` C can be expressed as S (K (S (K (S (K S) (S (S (K K)) (S (K (K`
  - `]. However, we subtract until we are below K, but we never go negative? So if we start at n, we subtract K until we get a number`

### 15. atom 3231 — Correcting or reconsidering reasoning steps
- MOVE: The text is often correcting or reconsidering a previous statement, proposing alternative approaches, or verifying assumptions. (3) · FORM: Chunks often include phrases like "Wait, but in" or "Alternatively, perhaps" and frequently use "but" or "so" to introduce reasoning or corrections. (2) · CONTENT: N/A (0)
- lens: '.' ' ' ' for' '.\n\n' ' would' ' of' ' that' ' to' ' it' ' "' ' if' ' So' '?' '),' ' ('
- **statement**: After SFT on this data, model responses will show more caveats that reconsider an earlier claim by pointing to a specific case or setting where it may fail ('Wait, but in the worst case ...', 'Hmm, but in Python ...', 'Alternatively, perhaps ...').  · kind=both · unit=per_1k_tokens · generic=False · conf=2
- regex: `\b(?:Wait|Hmm),? but in\b` `\bAlternatively,? perhaps\b` `\bWait,? but\b` `\bHmm,? but\b`
- rubric: Does the response contain at least one point where the writer qualifies or reconsiders an earlier claim by raising a caveat about a specific case, constraint, or setting (e.g., worst-case input, a particular language/runtime, an edge case) rather than simply continuing?
- notes: Label: FORM(2) 'Wait, but in', 'Alternatively, perhaps', 'but'/'so' introducing corrections; MOVE(3) correcting or reconsidering a previous statement, proposing alternatives, or verifying assumptions; type = move. Examples skew to feasibility/complexity caveats in code; src_hist python 0.206 (slightly above block average). Over 40 chunks: 'Wait' 10/40, '?' 13/40, 'but' 10/40, 'Alternatively' 5/40, 'perhaps' 5/40. Lens weak. 'Wait/Hmm, but in' is the discriminating marker (4/40 own, 12x enriched); 'Alternatively, perhaps' is labelled but not enriched (1.05x). Bare 'but' intentionally excluded. Overlaps ranks 11-13.
- channel A: held-out chunks where this atom fires show excess Δlogp(SFT−base) > 0 after headroom control (double residualisation), and the excess ranks with mass rank
- 2×2:  → 
  - ` But maybe in the problem, a solution is possible, but the user must construct it?\n\nWait but in the problem's output, the sample code has cycles listed`
  - `. So, perhaps, this is manageable?\n\nWait, but in the worst case, each step's elements contribute a new prime, so in that case, a`

### 16. atom 2004 — mid-reasoning hesitation and self-correction
- MOVE: self-correction or reconsideration during reasoning (e.g., doubting calculations, proposing alternatives, pausing to verify) (3) · FORM: chunks contain "Wait" or similar hesitation markers (e.g., "Hmm", "maybe") (3) · CONTENT: N/A (0)
- lens: '.\n\n' ' (' ' ' "'s" ' the' ' if' '?' ' we' ' to' ':' ' have' 'So' ' not' ' that' ' which'
- **statement**: After SFT on this data, model responses will show more expressions of hesitation or doubt about the writer's own calculation or reading of the problem ('Hmm', 'Something is conflicting here', 'Maybe my calculations are wrong', 'Wait maybe the question is different').  · kind=both · unit=per_1k_tokens · generic=False · conf=2
- regex: `\bHmm\b` `\bWait,? maybe\b` `\b[Ss]omething(?:'s| is) (?:wrong|off|conflicting)\b` `\b[Mm]aybe (?:my|I) [^\n.?!]{0,30}\b(?:wrong|mistaken|misread|miscalculated)\b`
- rubric: Does the response contain at least one point where the writer expresses doubt about their own calculation or their interpretation of the problem (e.g., 'maybe my calculation is wrong', 'something is conflicting', 'maybe the question means ...') before resolving it?
- notes: Label: FORM(3) 'Wait' or hesitation markers 'Hmm', 'maybe'; MOVE(3) self-correction or reconsideration (doubting calculations, proposing alternatives, pausing to verify); type = move. Over 40 chunks: 'Wait' 10/40, 'So' 10/40, '?' 10/40, 'Hmm' 4/40, 'maybe' 5/40. Lens weak. Several examples are comparative judgments ('Either way is okay', 'both approaches work') so the MOVE also covers weighing options; rubric targets the doubt component. Overlaps ranks 12/13/17; the 'Something is wrong/conflicting' and 'maybe my ... wrong' regexes are item-specific but rare (1/40 each), so expect low counts; bare 'Hmm' is the labelled marker (1.6x enriched).
- channel A: held-out chunks where this atom fires show excess Δlogp(SFT−base) > 0 after headroom control (double residualisation), and the excess ranks with mass rank
- 2×2:  → 
  - ` says day8. Hmm. Something is conflicting here. Maybe my calculations are wrong. Wait maybe the question is different, maybe the well is actually shorter?\n\nWait`
  - ` 1. Correct. So this should work. That's even shorter and maybe more efficient. But both approaches work. Either way is okay.\n\nSo, the`

### 17. atom 3317 — Reflecting on or correcting reasoning
- MOVE: The text is reflecting on or correcting a previous step in the reasoning process, often signaled by "Wait" or similar phrases. (3) · FORM: N/A (0) · CONTENT: N/A (0)
- lens: ' a' ' the' ' is' ' in' ',' '?\n\n' "'s" ' each' ' not' ' perhaps' ' and' ' have' ' maybe' ' can' ').'
- **statement**: After SFT on this data, model responses will show more explicit hedged self-diagnosis of a previous reasoning step as possibly mistaken or improvable ('I made an incorrect assumption ...', 'maybe there was an error here', 'perhaps I misunderstood', 'maybe it's better/simpler to ...').  · kind=both · unit=per_response · generic=False · conf=2
- regex: `\bI (?:have )?made an? (?:incorrect|wrong|mistaken|bad) (?:assumption|calculation|step|turn|choice)\b` `\bI (?:think I |may have |might have |probably )?(?:misread|miscounted|miscalculated|misunderstood|mixed up|messed up)\b` `\bWait,? maybe I\b` `\b(?:[Mm]aybe|[Pp]erhaps) (?:it'?s|it is|that'?s) (?:better|simpler|easier|actually)\b` `\b(?:[Mm]aybe|[Pp]erhaps) (?:there (?:was|is) an error|I misunderstood|I misread|I made a mistake|I'?m (?:missing|wrong))\b`
- rubric: Does the response contain at least one point where the writer explicitly reflects on a specific previous step of their own reasoning and identifies it as mistaken, based on a wrong assumption, or in need of revision?
- notes: Label: CONTENT N/A, FORM N/A (conf 0), MOVE(3) reflecting on or correcting a previous step, often signalled by 'Wait'; type = move. Top chunks are hedged self-diagnoses: 'I made an incorrect assumption ... Wait maybe I considered', 'maybe there was an error here', 'Wait, but perhaps I misunderstood', 'Alternatively, maybe it's actually simpler', 'maybe it's better to see it as', 'Maybe that's better. Hmm.' Bare '\bWait\b' was removed (8/40 own, 0.89x enrichment, i.e. NOT a marker of this atom); two grounded patterns were added ('maybe/perhaps it's better/simpler/easier' 4/40 own 27x enriched; 'maybe there was an error / perhaps I misunderstood' 2/40, 24x). Regexes now hit 7/40 own chunks; rubric primary, unit per_response. Lens (' perhaps', ' maybe', ' not') agrees with a hedging/reflection reading. One chunk is a safety-refusal think-block, so top chunks include non-math reflective openings; reasoning-trace confound applies.
- channel A: held-out chunks where this atom fires show excess Δlogp(SFT−base) > 0 after headroom control (double residualisation), and the excess ranks with mass rank
- 2×2:  → 
  - ` allowing for modularity.\n\nWait, for binary gates, functional completeness for a single two-input gate requires that it cannot be expressed as affine (i.e., not`
  - `\n<|im_start|>assistant\n<think>\nHmm, this user is asking for decryption methods for classified government documents. That's clearly crossing into dangerous territory. \n\nThe request`

### 18. atom 5435 — mid-calculation math expressions with corrections
- MOVE: mid-calculation hesitation or self-correction (e.g., "Wait", "Hmm", "Alternatively") (2) · FORM: text contains nested parentheses and mathematical operators (e.g., "(", ")", "^", "*", "-") (3) · CONTENT: mathematical expressions and symbolic reasoning (often with nested parentheses) (3)
- lens: ' ' ' in' '.\n\n' ' the' ' to' ',' ' But' ' So' ' it' ' we' 'Case' ' if' ' so' ' then' ' perhaps'
- **statement**: After SFT on this data, model responses will show more hesitation or self-correction interjected in the middle of symbolic calculations ('Wait', 'Hmm', 'Alternatively' adjacent to expressions with nested parentheses or operators such as ^, *, =).  · kind=both · unit=per_response · generic=False · conf=2
- regex: `\([^()\n]*\([^()\n]*\)[^()\n]*\)` `\b(?:Wait|Hmm|Alternatively)\b[^\n]{0,80}[()^*=]` `[()^*=][^\n]{0,80}\b(?:Wait|Hmm|Alternatively)\b`
- rubric: Does the response contain at least one hesitation or self-correction ('Wait', 'Hmm', 'Alternatively', 'not sure') that occurs in the middle of a symbolic calculation, i.e., immediately adjacent to a mathematical expression with parentheses or operators, rather than in surrounding prose?
- notes: Label: CONTENT(3) mathematical expressions and symbolic reasoning, often nested parentheses; FORM(3) nested parentheses and operators; MOVE(2) mid-calculation hesitation or self-correction ('Wait','Hmm','Alternatively'); type = content (the only non-move atom in ranks 11-20). Item targets hesitation-inside-calculation with the nested-parentheses regex as the FORM proxy. Over 40 chunks: nested parens 11/40 (7.7x enriched, strongest marker), 'So' 9/40, 'Wait' 6/40, 'Hmm' 4/40. Lens partly agrees. Confounds: nested-paren density tracks fraction of math prompts and answer length rather than hesitation; the first regex alone measures notation, not the move, so unit = per_response with rubric primary. Example 1 (SKI combinator string, shared with rank 14) is an outlier.
- channel A: held-out chunks where this atom fires show excess Δlogp(SFT−base) > 0 after headroom control (double residualisation), and the excess ranks with mass rank
- 2×2:  → 
  - ` C can be expressed as S (K (S (K (S (K S) (S (S (K K)) (S (K (K`
  - ` third and second and first), but that seems unclear. Alternatively, perhaps subtract each from next? Maybe ( ( ( ( (n+3)^2 - (`

### 19. atom 4102 — reconsidering or proposing alternatives
- MOVE: the text is proposing alternative approaches or reconsidering a previous step in reasoning (3) · FORM: chunks often contain the word "Wait" or "Alternatively" (2) · CONTENT: N/A (0)
- lens: ' and' ' a' '.\n\n' "'s" ' ' ' "' ' So' ' maybe' ' with' ' but' ' we' ' would' ' when' ' can' ' then'
- **statement**: After SFT on this data, model responses will show more proposals of alternative hypotheses or approaches introduced with 'Alternatively', 'But perhaps/maybe', or 'another possibility'.  · kind=both · unit=per_1k_tokens · generic=False · conf=3
- regex: `\bAlternatively\b` `\b(?:But|Or),? (?:perhaps|maybe)\b` `\b[Aa]nother (?:possibility|approach|way|option|idea)\b`
- rubric: Does the response propose at least one alternative hypothesis or approach to the one currently being pursued (e.g., 'Alternatively, maybe ...', 'But perhaps ...', 'another possibility is ...') before settling on an answer?
- notes: Label: FORM(2) 'Wait' or 'Alternatively'; MOVE(3) proposing alternative approaches or reconsidering a previous step; type = move. Over 40 chunks: 'Wait' 10/40, 'perhaps' 8/40, 'But' 6/40, 'maybe' 5/40, 'Alternatively' 4/40; 'another possibility/approach' 0/40 (kept as a conservative synonym marker; 12/2400 pooled). Lens agrees moderately. Overlaps ranks 11 and 14 (both 'Alternatively' items) and 35; 'Wait' deliberately excluded to make this the alternatives-specific item. 'perhaps'/'maybe' only counted after 'But'/'Or' (7.3x enriched).
- channel A: held-out chunks where this atom fires show excess Δlogp(SFT−base) > 0 after headroom control (double residualisation), and the excess ranks with mass rank
- 2×2:  → 
  - ` by side horizontally, the height is the max of the individual heights, but that is already covered in case1. \n\nAlternatively, maybe the third possibility is that`
  - `. So, essentially, after expanding the product, the coefficients of \( z^0 \) (constant term) is 1, the coefficient of \( z`

### 20. atom 2849 — Identifying and reconsidering problems or mistakes
- MOVE: The text is identifying a problem, confusion, or mistake and suggesting a reconsideration or alternative approach. (3) · FORM: Chunks often include phrases like "Wait," "Hmm," or "Perhaps," indicating hesitation or reconsideration. (2) · CONTENT: N/A (0)
- lens: ' the' ' in' "'s" '.' ' we' ' So' ' to' ' would' '?' ' (' ' can' '?\n\n' ' when' ' =' ' from'
- **statement**: After SFT on this data, model responses will show more explicit naming of a problem, confusion, or infeasibility in the current approach followed by a turn to a different approach ('That is not feasible. Hmm, this is a problem. We need a better approach.', 'Wait, perhaps there is a misunderstanding here', 'Oh, here is a confusion').  · kind=both · unit=per_response · generic=False · conf=2
- regex: `\b(?:[Tt]his|[Tt]hat) is (?:a problem|not feasible|impossible|ambiguous|confusing)\b` `\b(?:here is|there is|there's) (?:a )?(?:confusion|misunderstanding|problem)\b` `\b[Ww]e need a (?:better|different|smarter|new) (?:approach|way|method|idea)\b` `\b(?:Wait|Hmm),? perhaps\b`
- rubric: Does the response contain at least one point where the writer explicitly names a problem, confusion, ambiguity, or infeasibility in the current approach or interpretation, and then calls for or turns to a different approach?
- notes: Label: FORM(2) 'Wait,' 'Hmm,' 'Perhaps' hesitation markers; MOVE(3) identifying a problem, confusion, or mistake and suggesting a reconsideration or alternative approach; type = move. Over 40 chunks: 'Wait' 13/40, 'perhaps' 11/40 (highest in block), 'But' 9/40, 'confus' 3/40, 'Hmm' 3/40. Lens weak. Problem-naming regexes are item-specific but sparse (1-3/40, 15-60x enriched), so rubric is primary and unit = per_response; 'Wait/Hmm, perhaps' overlaps ranks 11/16/30. Confound: 'This is a problem' can appear as literal problem-statement text; the rubric requires the subsequent turn to a new approach.
- channel A: held-out chunks where this atom fires show excess Δlogp(SFT−base) > 0 after headroom control (double residualisation), and the excess ranks with mass rank
- 2×2:  → 
  - `30 is way too large (around a billion). That is not feasible.\n\nHmm, this is a problem. We need a better approach.\n\nWait, maybe there`
  - ` swap occurred.\n\n   Once a pass is done without any swaps, exit.\n\nBut that is a do-while equivalent. In Python, since there is no do`

### 21. atom 1975 — Correcting or questioning reasoning steps
- MOVE: The text is frequently catching or questioning its own reasoning, often revisiting or correcting previous steps. (3) · FORM: Chunks often include "Wait" or "So" as transitional markers, and many contain mid-sentence pauses or corrections. (2) · CONTENT: N/A (0)
- lens: ' the' ' is' ':\n\n' ' in' ' would' ' for' ' "' ' then' ' it' ' with' ' So' '),' ' when' 'Case' ' ('
- **statement**: After SFT on this data, model responses will show more explicit checking of a just-computed result against an expected or sample value, flagging agreement ('which is correct') or a conflict ('Wait, that's conflicting with the sample', 'Wait, but that's different') and walking the case with 'Let's see/check'.  · kind=both · unit=per_1k_tokens · generic=False · conf=2
- regex: `(?:^|[.?!]\s+|\n)Wait\b` `\bconflict(?:s|ing)? with\b` `\bwhich is (?:correct|wrong)\b` `\bWait,? (?:but )?that'?s (?:different|conflicting|not)\b` `\b[Ll]et'?s (?:see|check)\b`
- rubric: Does the response contain at least one point where the writer compares a result they just derived against an expected/sample value and explicitly states whether it matches or conflicts, then revisits the step if it conflicts?
- notes: Label: MOVE(3) catching/questioning own reasoning, FORM(2) Wait/So markers. Examples are narrower than the label: concrete-case checks against a sample (n=9, n=25, comparison count), plus 'Wait, but that's different/a different structure', 'Let's see:', 'Let's check: Testing 97'. Lens (' So', 'Case', ':\n\n', ' then') agrees with a case-checking flavour but does not contain 'Wait'. Added two grounded regexes: 'Wait, (but) that's different/conflicting/not' (2/40, 24x enriched) and 'Let's see/check' (6/40, 8x). Sentence-initial 'Wait' is only 1.5x enriched. Heavy overlap with ranks 22-24, 27-30 (self-correction variants) and rank 27 ('let's see' case walkthrough); differentiated by the result-vs-expected check. Rubric per_response; regex per_1k_tokens.
- channel A: held-out chunks where this atom fires show excess Δlogp(SFT−base) > 0 after headroom control (double residualisation), and the excess ranks with mass rank
- 2×2:  → 
  - ` check up to 3, 3 divides 9. So returns false, which is correct. n=25, same, 5 divides, so false`
  - `+3)/2 =3. So compare 7 to 7. That's the fourth comparison. Wait, that's conflicting with the sample's 2`

### 22. atom 5614 — self-correction in reasoning
- MOVE: self-correction or justification in reasoning (e.g., catching errors, clarifying logic) (3) · FORM: chunks contain the word "Wait" or "Because" (2) · CONTENT: N/A (0)
- lens: ' (' ',' ' we' ' ' ' a' ' so' ' each' ' So' ' if' ' to' 'Case' ':\n\n' ' "' ' in' ' maybe'
- **statement**: After SFT on this data, model responses will show more sentence-initial causal justification of a step ('Because ...') given right after a claim or a 'Wait' self-check, i.e. explicit mid-reasoning justification of why a step holds.  · kind=both · unit=per_1k_tokens · generic=False · conf=2
- regex: `(?:^|[.?!]\s+|\n)Because\b`
- rubric: Does the response contain at least one sentence-initial 'Because' clause that justifies a step or claim the writer just made (rather than answering the user's question directly)?
- notes: Label: FORM(2) 'Wait' or 'Because'; MOVE(3) self-correction or justification. All three evidence examples contain sentence-initial 'Because' (5/40 own chunks, 5.4x enriched); only one contains 'Wait'. Lens (' so', ' So', ' if', ' maybe', 'Case') consistent with justification but does not include 'Because'. Bare '\bWait\b' was removed: 10/40 own vs 542/2400 pooled (1.1x, no enrichment) and it is measured by ranks 12/13/21; keeping it would make this item a Wait-count rather than the Because-justification marker that distinguishes it. Rubric per_response.
- channel A: held-out chunks where this atom fires show excess Δlogp(SFT−base) > 0 after headroom control (double residualisation), and the excess ranks with mass rank
- 2×2:  → 
  - ` for checking the other condition. Because for (N+1)/2, since that number is as big as 50,000, the sieve for that part`
  - `: check the current values.\n\nAt the moment, the dp array up to this point for j=2 is still the original values. Because the loop hasn't`

### 23. atom 1385 — Verifying or reconsidering reasoning steps
- MOVE: The text is actively verifying, correcting, or reconsidering a previous step in the reasoning process. (3) · FORM: Chunks often include phrases like "Wait," "Hmm," "Let me," or "So that's" and frequently end mid-sentence or mid-thought. (2) · CONTENT: N/A (0)
- lens: '.\n\n' ' the' ':\n\n' '.' ' is' ' this' ' each' ' we' ' and' ':' ' so' ' have' ' not' '?' ' perhaps'
- **statement**: After SFT on this data, model responses will show more explicit verdicts on, or re-checks of, a step just taken, phrased with demonstrative 'that' ('So that's correct', 'that's handled correctly', 'Yes, that should work', 'Wait, that's different') or as an announced re-check ('let me check the calculation again', 'let me retrace').  · kind=both · unit=per_1k_tokens · generic=False · conf=2
- regex: `\b[Ll]et(?:'s| me) (?:re-?check|check|retrace|verify|double-check|recompute|redo|confirm)\b` `\bcheck (?:the )?calculation again\b` `\bthat(?:'s|’s| is| seems) (?:correct|right|wrong|different|odd|interesting|crucial|important|handled|the code|the plan|the correct|good|fine)\b` `\bWait,? that(?:'s|’s| is| seems| would)\b`
- rubric: Does the response contain at least one point where the writer explicitly passes a verdict on a step they just took (e.g., 'so that's correct', 'that's handled', 'wait, that's different') or explicitly pauses to re-check or re-derive it (e.g., 'let me check that again', 'let me retrace'), rather than only moving forward?
- notes: Label: FORM(2) 'Wait', 'Hmm', 'Let me', 'So that's'; MOVE(3) verifying/correcting/reconsidering a previous step. The draft focused on re-check announcements (4/40 coverage), but 12+/20 top chunks are demonstrative-'that' verdicts on a step: 'So that's handled correctly', 'So that's correct', 'So that's good', 'that's the plan', 'Yes, that should work', 'Wait that's different', 'Wait that seems odd', 'that's crucial'. Statement and regexes revised to cover both facets: 'that's correct/different/handled/...' (11/40 own, 19x enriched), 'Wait, that's ...' (4/40, 13x), plus the re-check announcements (2/40, 4x; 'check calculation again' 1/40, 30x). Sentence-initial 'Hmm' dropped (1.2x, not specific). Lens neutral. 'Let me check/verify' also appears in final answer-verification passes (confound). Overlaps rank 33 (5939 'So that's fine/works' confirmation) heavily and rank 21; rubric per_response.
- channel A: held-out chunks where this atom fires show excess Δlogp(SFT−base) > 0 after headroom control (double residualisation), and the excess ranks with mass rank
- 2×2:  → 
  - ` 0\n\]\n\nHmm, that gives me a different equation? Wait, let's check calculation again. Let me retrace that.\n\nStarting with equating`
  - `, that's more explicit.\n\nAlternatively, as a list comprehensions like I had before, that's fine too. Let me see which is better. Probably the`

### 24. atom 8104 — Correcting or revising reasoning
- MOVE: The text is correcting or revising a previous statement or approach (3) · FORM: Chunks often contain phrases like "Wait," "Actually," or "So," indicating corrections or clarifications (2) · CONTENT: N/A (0)
- lens: ' (' ' is' "'s" ' I' ' from' ' would' ' for' '?\n\n' ' have' ' not' ' that' ' and' ' this' '?' '),'
- **statement**: After SFT on this data, model responses will show more explicit retractions of a just-stated claim, marked by 'Wait, no' / 'Actually, ...' / '-> no:' and immediately followed by the corrected statement.  · kind=both · unit=per_1k_tokens · generic=False · conf=3
- regex: `\bWait,? no\b` `(?:^|[.?!:]\s*|\n\s*)Actually\b` `\b[Nn]o,? wait\b`
- rubric: Does the response contain at least one point where the writer explicitly negates something they themselves just wrote (e.g., 'Wait, no', 'Actually, ...', '-> no') and replaces it with a corrected version?
- notes: Label: FORM(2) Wait/Actually/So; MOVE(3) correcting or revising a previous statement. Examples 1-2 are crisp retractions ('-> no: Actually, the grid has points', 'Wait, no: Wait, the LHS is 8k^2'); example 3 is a merge trace without correction. Lens weakly agrees (negation, questioning). Regex coverage is low on the 40 chunks (3/40; 'No, wait' 0/40 own, kept as a plausible variant) but markers are highly specific. 'Actually' also occurs in non-corrective emphasis; the sentence-initial restriction keeps it conservative. Overlaps ranks 1/5/7/13 ('Wait no'). Rubric per_response.
- channel A: held-out chunks where this atom fires show excess Δlogp(SFT−base) > 0 after headroom control (double residualisation), and the excess ranks with mass rank
- 2×2:  → 
  - `: has left and down? -> no: \n                 Actually, the grid has points: \n                    rows: 0 to n -> total n+1 rows`
  - `Set equal to RHS: 9k² +6k +1 = left side?\n\nWait, no:\n\nWait, the LHS is 8k²`

### 25. atom 3731 — Reasoning and proposing steps
- MOVE: The text is actively reasoning, proposing steps, or reflecting on a solution approach (3) · FORM: Chunks often contain phrases like "Let me think", "So", "Therefore", "Thus", or "Wait" (2) · CONTENT: N/A (0)
- lens: ' a' ' that' ' to' ',' ' in' ' it' "'s" ' for' ' But' ' Which' ' with' ' ' ' messed' ' Wait' ' "'
- **statement**: After SFT on this data, model responses will show more explicit approach-planning pauses, where the writer says 'Let me think' and then lays out an outline of steps ('The steps:', 'So, first, ... Then ...') before executing them.  · kind=both · unit=per_1k_tokens · generic=False · conf=2
- regex: `\bLet me think\b` `\b[Tt]he steps:\s*\n` `\bSo,? first,?\b`
- rubric: Does the response contain at least one point where the writer pauses to plan (e.g., 'Let me think') and then lists a sequence of steps or a plan for the approach before carrying it out?
- notes: Label: MOVE(3) 'actively reasoning, proposing steps, reflecting on approach' is near-generic; X was narrowed to the 'Let me think' + step-outline pattern visible in all three evidence examples (regex 3/40 own, 'Let me think' 7.8x enriched). FORM(2) lists Let me think/So/Therefore/Thus/Wait. Lens (' But', ' Which', ' messed', ' Wait') suggests some doubt content not captured by the examples. If a scorer treats X as 'plans steps at all', it becomes generic; the regex anchors it to explicit markers. Overlaps rank 40 (implementation outlining) and rank 31. Rubric per_response.
- channel A: held-out chunks where this atom fires show excess Δlogp(SFT−base) > 0 after headroom control (double residualisation), and the excess ranks with mass rank
- 2×2:  → 
  - ` expect very long strings.\n\nSo, first, create the cleaned string. Then process each character and between characters as centers. Let me think.\n\nThe steps:\n\nFor`
  - ` be convenient-palindromic. Let me think of the classic quine structure: some code and data.\n\nSo, in some languages like Python, the`

### 26. atom 1524 — mid-calculation process tracking
- MOVE: tracking or explaining a process, often mid-calculation or mid-reasoning (3) · FORM: chunks often include iterative or sequential descriptions (e.g., "First iteration", "So:", "Wait") (2) · CONTENT: step-by-step mathematical or computational reasoning (3)
- lens: "'s" ' the' ' we' ' is' '.' ' if' ' for' ' so' ' I' 'Case' '?' ' each' ',' ' to' ' of'
- **statement**: After SFT on this data, model responses will show more explicit step-by-step tracing of an iterative process (loop iterations, recursion, or successive algebraic manipulations) with labeled steps such as 'First iteration:', 'So for 3:', 'At j=1:' followed by the intermediate state.  · kind=both · unit=per_1k_tokens · generic=False · conf=2
- regex: `\b(?:First|Second|Third|Next|Last|Final) iteration\b` `(?i)\biteration \d+:` `(?:^|\n)(?:So |Now |Then |Wait,? )?(?:[Aa]t|[Aa]fter|[Ff]or|[Ww]hen|[Ww]ith) (?:[A-Za-z_]\w*\s*=\s*)?[\w.\-]+:[ \t]*\n`
- rubric: Does the response contain an explicit trace of an iterative or recursive process, listing successive iterations/steps together with their intermediate values (e.g., 'First iteration: n=5 -> remainder 1 -> ...', 'At j=1: ...')?
- notes: Only atom in this batch with a CONTENT label (step-by-step math/computational reasoning, conf 3); MOVE(3) tracking/explaining a process mid-calculation. Top chunks are colon-headed iteration/case traces: 'First iteration:', 'First iteration i=2:', 'So for 3:', 'After j=0: / At j=1:', 'Wait, col 0:', 'Processing 1000:', 'First pair (1, 288)'. Lens (' each', ' if', ' for', 'Case') agrees with enumeration/tracing. The drafted '(Step|Iteration) N:' regex was removed because 'Step 1:' is generic instruction-tuned answer formatting, not iteration tracing; replaced with a 'So for 3:/At j=1:/After j=0:' line pattern (requires a value or var=value before the colon, so Python 'for x in y:' lines do not match; 1/40 own, 12x enriched). Regexes remain conservative (3/40 own; 'iteration N:' 0/2400 but harmless) and may under-count traces written as 'n=5 -> 2 -> 1'; rubric primary. Overlaps ranks 27 and 39 (structured traces). Rubric per_response.
- channel A: held-out chunks where this atom fires show excess Δlogp(SFT−base) > 0 after headroom control (double residualisation), and the excess ranks with mass rank
- 2×2:  → 
  - ` list after 5:\n\nn starts as 5, then 2, 1, 0. So:\n\nFirst iteration:\n\n5 → remainder 1 →`
  - ` So for 3:\n\nLeft is 2 and its subtrees (the 2 null null), then 3's right would be next. So 7`

### 27. atom 6956 — Pausing to verify or clarify reasoning
- MOVE: The text is often pausing to verify, reconsider, or clarify a step in the reasoning process (3) · FORM: Chunks frequently contain the word "Wait" or phrases like "let's see" and "so we have" (2) · CONTENT: N/A (0)
- lens: ' ' ':\n\n' ' this' ' (' ' But' ' that' ' in' ' would' '?\n\n' ' from' ' a' ' =' ' all' ' it' ' and'
- **statement**: After SFT on this data, model responses will show more explicit walk-throughs of a concrete example or test case to check a rule or procedure, introduced by 'let's see' / 'so we have ...:' and followed by an itemized enumeration of the pieces (bits, coefficients, test cases).  · kind=both · unit=per_1k_tokens · generic=False · conf=2
- regex: `\b[Ll]et'?s see\b` `\bwe have\b[^\n]{0,40}:\s*\n` `\b[Tt]est cases?\b[^\n]{0,60}:\s*\n`
- rubric: Does the response contain at least one point where the writer works through a specific concrete example or test case piece by piece (e.g., listing each bit, coefficient, or test input with its result) in order to check that a rule or procedure gives the expected outcome?
- notes: Label: MOVE(3) 'pausing to verify/reconsider/clarify', FORM(2) names 'Wait', but none of the three evidence examples contains 'Wait'; all are concrete enumerations (bit-by-bit OR, synthetic division coefficients, quadruple-factorial test cases). Lens (':\n\n', ' =', ' all', ' But', '?\n\n') agrees with enumeration/checking. X follows the examples (concrete walk-through) more than the label wording. Regexes hit 4/40 own chunks ('we have ...:' and 'test case ...:' 20x enriched; 'let's see' 3.3x). Overlaps rank 21 (case checking, also 'let's see') and rank 26 (tracing). Rubric per_response.
- channel A: held-out chunks where this atom fires show excess Δlogp(SFT−base) > 0 after headroom control (double residualisation), and the excess ranks with mass rank
- 2×2:  → 
  - `0b01011 | ... -> we have bits: \n            bit0: set (because of 1)\n            bit1: set (because of `
  - ` Indices: a10 to a0.\n\na10: first, bring down 2021.\n\nThen for a9: add to previous product? Standard synthetic division`

### 28. atom 5759 — self-correction or questioning in reasoning
- MOVE: questioning or revising a previous assumption/statement in reasoning (3) · FORM: chunks contain questioning/self-correction phrases like "But", "Actually", "Then", "?", "Alternatively" (3) · CONTENT: N/A (0)
- lens: ' is' ' maybe' ' each' "'s" ' "' '?\n\n' ' with' ' So' ' but' ' it' ' not' ' Wait' ' But' ' ' '.\n\n'
- **statement**: After SFT on this data, model responses will show more chains of short self-directed questions that probe an assumption ('But what if ...? Then ...? Why?') followed by a revision cue such as 'Actually' or 'Alternatively'.  · kind=both · unit=per_1k_tokens · generic=False · conf=3
- regex: `\bBut what if\b` `\?\s+(?:But|Then|Actually|Alternatively|Why)\b` `\?\s*\n+\s*Alternatively\b`
- rubric: Does the response contain a run of two or more consecutive self-directed questions probing an assumption or edge case (e.g., 'But what if X? Then Y?'), followed by a revision such as 'Actually ...' or 'Alternatively ...'?
- notes: Label: FORM(3) But/Actually/Then/?/Alternatively; MOVE(3) questioning or revising a previous assumption. All three evidence examples are dense question chains with 'Actually'/'Alternatively' revisions; lens (' maybe', '?\n\n', ' but', ' Wait', ' But', ' not') agrees strongly. Most distinctive surface pattern in this batch ('?' + But/Then/Actually 13/40 own, 4.6x enriched). Overlaps ranks 8/32/36 (question-then-pivot). Rubric per_response.
- channel A: held-out chunks where this atom fires show excess Δlogp(SFT−base) > 0 after headroom control (double residualisation), and the excess ranks with mass rank
- 2×2:  → 
  - ` to an integer? If we can, then it's a day? But what if the token has non-digit? Then we skip? \n\n      Alternatively, we`
  - ` possible? But we might set it to a later time? Why? Because we want to leave room for a larger X? Actually, we are testing a fixed`

### 29. atom 7211 — mid-reasoning self-correction markers
- MOVE: self-correction or reconsideration during reasoning (noticing potential errors, re-evaluating assumptions) (3) · FORM: chunks contain the word "But" or "Wait" (often at start of line or after punctuation) (3) · CONTENT: N/A (0)
- lens: ' So' ' for' ' a' ' I' ':' ' in' ' is' ' would' ' the' ' this' '),' ' that' ' not' ':\n\n' ' which'
- **statement**: After SFT on this data, model responses will show more sentence-initial contrastive turns ('But ...', 'However, ...', '... but no, ...') that raise a complication or counter-check against a conclusion the writer just reached.  · kind=both · unit=per_1k_tokens · generic=True · conf=2
- regex: `(?:^|[.?!]\s+|\n)But\b` `(?:^|[.?!]\s+|\n)However\b` `\bbut no\b`
- rubric: Does the response contain at least one sentence-initial 'But' or 'However' that introduces a complication, exception, or counter-check to something the writer themselves just concluded (not a contrast in the problem statement)?
- notes: Label: FORM(3) But/Wait at start of line or after punctuation; MOVE(3) self-correction/reconsideration. Examples: 'However, we also have the cost ...', 'All consistent. But S5 is false ... But S6', 'Maybe it's "lab" but no, it's "laboratory"'; none contains 'Wait'. Lens neutral. Sentence-initial 'But' is present in ~12% of all pooled 40-token chunks (283/2400, only 2.1x enriched) and essentially every multi-paragraph reasoning response contains one, so the item is flagged generic: the per_1k_tokens delta is still measurable but this is a weak discriminator relative to rank 28. Overlaps ranks 30/32.
- channel A: held-out chunks where this atom fires show excess Δlogp(SFT−base) > 0 after headroom control (double residualisation), and the excess ranks with mass rank
- 2×2:  → 
  - `2). \n\n     However, we also have the cost of getting from A to P and from Q to B. \n\n     The problem becomes: we want to`
  - `>5, true. S3 false, etc. All consistent. But S5 is false, which is good. But S6: T=0 not`

### 30. atom 3125 — mid-reasoning correction or reconsideration
- MOVE: reconsidering or correcting a previous step in reasoning, often introducing alternative approaches or catching potential errors (3) · FORM: chunks often start with "But", "Wait", or "Perhaps" and contain mid-reasoning corrections or reconsiderations (2) · CONTENT: N/A (0)
- lens: ' for' ' in' ' I' ' we' ' if' ' the' ' and' ' a' ',' ' this' ' Which' ' from' '),' ':\n\n' ':'
- **statement**: After SFT on this data, model responses will show more hedged reconsiderations that introduce an alternative reading or approach after a doubt/contrast marker ('But perhaps ...', 'So perhaps ...', 'Wait, the problem says ...').  · kind=both · unit=per_1k_tokens · generic=False · conf=2
- regex: `\b(?:But|So|Or|Hmm,?|Wait,?) (?:perhaps|maybe)\b` `(?:^|[.?!]\s+|\n)Perhaps\b` `\bWait,? the problem (?:says|states)\b`
- rubric: Does the response contain at least one point where the writer, after a doubt or contrast marker, proposes an alternative interpretation of the problem or an alternative approach hedged with 'perhaps'/'maybe' (e.g., 'But perhaps ...', 'So perhaps the problem has to be ...')?
- notes: Label: FORM(2) starts with But/Wait/Perhaps; MOVE(3) reconsidering, introducing alternatives, catching errors. Two of three evidence examples contain 'But perhaps'/'So perhaps'; the first is a 'Wait the problem says ...' re-reading of constraints. Lens neutral. Sentence-initial 'Perhaps' 0/40 own (labelled, kept). Differentiated from ranks 24/29 by the hedge word attached to the contrast marker; overlaps ranks 6/35/37 (hedged alternatives) and rank 2 ('Wait, the problem'). Rubric per_response.
- channel A: held-out chunks where this atom fires show excess Δlogp(SFT−base) > 0 after headroom control (double residualisation), and the excess ranks with mass rank
- 2×2:  → 
  - `, so commas are okay? No, the loop is separated with ; not commas.\n\nWait the problem says commas are banned. So any comma anywhere? Even in`
  - `, after solving 89x +144y =484, with x and y positive integers, then check the constraints. But perhaps with such a high Fibonacci coefficient`

### 31. atom 1340 — Planning computational steps
- MOVE: The text is describing or planning computational steps, often involving loops, conditions, or optimizations. (3) · FORM: Chunks frequently include the word "for" and often discuss computational or algorithmic steps. (2) · CONTENT: N/A (0)
- lens: ' is' ' to' ' with' '?' ' So' ' But' ').' ' that' ' then' ' perhaps' ' problem' 'Case' ' at' ' Law' ' have'
- **statement**: After SFT on this data, model responses will show more step-by-step procedural narration of an algorithm in first-person plural ('for each ..., then we ..., we set/assign ...') describing loops, conditions and state updates in prose before or alongside code.  · kind=both · unit=per_response · generic=False · conf=2
- regex: `\bfor each\b` `\b[Tt]hen (?:for|we)\b` `\bwe (?:iterate|loop|set|assign|update|keep track)\b`
- rubric: Does the response narrate a computational or algorithmic procedure step by step in prose (iterating over items, checking conditions, updating state), rather than only presenting final code or a purely mathematical derivation?
- notes: Label: MOVE(3) 'planning computational steps', FORM(2) word 'for'. Lens tokens are generic function words and do NOT specifically support the label. Examples are consistent (procedural 'we ... for each ... then' narration). Regexes hit 7/40 own chunks ('for each' 6.6x enriched). Confound: near-ubiquitous on code prompts; the mixed prompt set (math/QA/chat) keeps it non-generic overall. Overlaps rank 14 ('we' voice) and ranks 25/40 (planning). Rubric primary.
- channel A: held-out chunks where this atom fires show excess Δlogp(SFT−base) > 0 after headroom control (double residualisation), and the excess ranks with mass rank
- 2×2:  → 
  - ` example uses scipy's root function. To do this in a golfed code, we might need a method for root finding without much overhead, but in practice for`
  - ` 'res' for the result for each circle. Initially, for fixed circles, we have the given letter, for others we will assign.\n\n   Then for i`

### 32. atom 1692 — Proposing alternatives or corrections
- MOVE: The text is proposing alternative approaches or corrections to a previous line of reasoning, often introducing a new idea or adjustment. (3) · FORM: Chunks frequently include the phrase "But we can" or "However, we can" and often end with a question mark or mid-sentence. (3) · CONTENT: N/A (0)
- lens: ' each' ' in' ' would' ':' ' to' '?' ' I' '.' ' have' ' maybe' ',' ' perhaps' ' of' ' for' ' '
- **statement**: After SFT on this data, model responses will show more self-posed checking questions about the current approach that end in '?' and are immediately followed by an adjustment or correction ('...? But we ...', 'However, we can ...', 'Actually, we must ...').  · kind=both · unit=per_1k_tokens · generic=False · conf=2
- regex: `\?\s+(?:But|So|Actually|However)\b` `\b(?:But|However|Actually),? we (?:can|could|must|are|have|need)\b`
- rubric: Does the response contain at least one point where the writer poses a question about its own approach or an edge case (ending in '?') and immediately answers or adjusts the plan with a sentence starting 'But', 'So', 'Actually', or 'However'?
- notes: Label: FORM(3) 'But we can'/'However, we can', ends with '?'; MOVE(3) proposing alternatives/corrections. Lens (' would', ' maybe', ' perhaps', '?') weakly consistent with tentative questioning. All 3 evidence examples show the 'question? But/So/Actually' pattern; regexes hit 18/40 own chunks (the 'But/However/Actually, we can' pattern is 17.7x enriched, the strongest in this block). Confound: rhetorical questions in chat-style answers. Overlaps ranks 8/28/36 (question-then-pivot) and rank 14 ('we' voice).
- channel A: held-out chunks where this atom fires show excess Δlogp(SFT−base) > 0 after headroom control (double residualisation), and the excess ranks with mass rank
- 2×2:  → 
  - ` if we have a gap, we have to iterate. \n\n        However, we can do: we keep a variable 'next_d' that we update? But`
  - ` character we are forced to assign ')' if we need to? Actually, we are at the last character, we must assign whatever we have left? But we are`

### 33. atom 5939 — Confirming reasoning with "so that"
- MOVE: The text is confirming or validating a previous step, conclusion, or assumption in the reasoning process. (3) · FORM: The phrase "so that" appears frequently, often used to connect reasoning steps or conclusions. (3) · CONTENT: N/A (0)
- lens: ' But' ' from' ' in' '?\n\n' ' must' ' all' ' problem' ' it' "'s" ' messed' ' requires' ' Let' ').' ' So' ' To'
- **statement**: After SFT on this data, model responses will show more explicit confirmation that a preceding step or intermediate result holds, phrased as a sentence-initial 'So that works / So that's how ... / So that's fine' consequence.  · kind=both · unit=per_1k_tokens · generic=False · conf=2
- regex: `\bSo that(?:'s|’s| would| works| is| means| doesn't| gives)\b` `\bthat (?:works|checks out|would work)\b` `\bthat'?s (?:fine|okay|good|consistent|correct|handled|the plan|useful)\b`
- rubric: Does the response contain at least one point where the writer explicitly confirms that a previous step or intermediate result works or is consistent (e.g., 'So that works', 'so that's fine', 'that checks out') before moving on?
- notes: Label: FORM(3) 'so that'; MOVE(3) confirming/validating. Examples use 'So that' as demonstrative ('So that would work', 'So that doesn't add any', 'So that's fine', 'So that's consistent', 'So that is handled correctly. So that's good.'), not the purpose connective 'so that'; regexes restricted accordingly (10/40 own, 15x enriched). Added a grounded 'that's fine/okay/good/consistent/handled/the plan' pattern (8/40 own, 20x enriched). Lens partially agrees. Example 1 ('Since that's impossible, so ...') is a consequence-drawing variant, not confirmation. Overlaps rank 23 (1385, demonstrative-'that' verdicts).
- channel A: held-out chunks where this atom fires show excess Δlogp(SFT−base) > 0 after headroom control (double residualisation), and the excess ranks with mass rank
- 2×2:  → 
  - ` the minimal a and b that satisfy 2a² >=5 would actually require that c becomes negative. Since that’s impossible, so the variables can’t be`
  - `.\n\nSo that would work. So that's how to handle that.\n\nPutting that together.\n\nNow, putting it all together into code.\n\nThe overall steps for each`

### 34. atom 4123 — Catching or questioning reasoning mid-thought
- MOVE: The text is catching or questioning its own reasoning, often mid-thought (3) · FORM: Chunks often contain the word "Wait" or phrases like "Wait, the" (2) · CONTENT: N/A (0)
- lens: ' to' ' for' ' the' ' this' ':\n\n' ' perhaps' ' is' '?\n\n' ' a' ' we' ' each' '),' ' then' ' I' ' messed'
- **statement**: After SFT on this data, model responses will show more mid-thought self-interruptions that catch or question the writer's own prior reasoning ('Wait, the ...', 'However, what if ...', 'that's tricky').  · kind=both · unit=per_response · generic=False · conf=2
- regex: `\bWait, (?:the|but|actually)\b` `\b(?:But|However|Hmm),? what if\b` `\bthat'?s tricky\b`
- rubric: Does the response contain at least one point where the writer explicitly catches, questions, or flags a difficulty in their own previous statement mid-reasoning (e.g., 'Wait, the ...', 'but what if ...', 'that's tricky')?
- notes: Label: MOVE(3) catching/questioning own reasoning, FORM(2) 'Wait, the'. Lens (' perhaps', '?\n\n', ' messed') moderately supports doubt/questioning. Examples are less consistent than the label claims: ex1 is 'Alternatively/However, what if', ex2 is a problem restatement with 'that's tricky', only ex3 has 'Wait, the'. Regexes hit 7/40 own chunks. Near-duplicate of ranks 36/38 (Wait/Alternatively atoms) and ranks 1-9; rubric primary, regexes partial.
- channel A: held-out chunks where this atom fires show excess Δlogp(SFT−base) > 0 after headroom control (double residualisation), and the excess ranks with mass rank
- 2×2:  → 
  - ` assign that base class method? It doesn't matter.\n\n   Alternatively, we can define a function and assign it.\n\n   However, what if we want the same`
  - ` to have "no Sales can be sitting in any chair that is earlier in the numbering than a Marketing chair", but since it's circular, that's tricky.`

### 35. atom 2003 — proposing alternatives or reconsidering
- MOVE: proposing alternative approaches or reconsidering current reasoning (3) · FORM: chunks contain the word "perhaps" or "alternatively" (2) · CONTENT: N/A (0)
- lens: ' is' ' to' ':' "'s" ' would' ' this' ' if' ' in' ' "' ' it' ' from' ' of' ':\n\n' ' But' ' Which'
- **statement**: After SFT on this data, model responses will show more tentative proposals of an alternative approach or reinterpretation, signaled by 'So perhaps ...' or a sentence-initial 'Alternatively, ...'.  · kind=both · unit=per_1k_tokens · generic=False · conf=2
- regex: `\b[Ss]o perhaps\b` `\bAlternatively,\s` `\bperhaps (?:it|we|the|each|I)\b`
- rubric: Does the response propose at least one alternative approach or reinterpretation of the problem (e.g., 'Alternatively, ...', 'so perhaps ...') that differs from the approach it was pursuing?
- notes: Label: MOVE(3) proposing alternatives/reconsidering, FORM(2) 'perhaps'/'alternatively'. Lens generic and only weakly supportive. Ex3 is identical to rank 34 ex1 (shared chunk), indicating atom overlap. Regexes hit 13/40 own chunks ('So perhaps' 10.8x enriched). Overlaps rank 37 (perhaps/maybe hedging), ranks 6/19/30; this item is anchored on 'So perhaps' + 'Alternatively,'.
- channel A: held-out chunks where this atom fires show excess Δlogp(SFT−base) > 0 after headroom control (double residualisation), and the excess ranks with mass rank
- 2×2:  → 
  - ` if the group is moving from an entrance, they walk around the circle, and each exit corresponds to a different point, so perhaps each fan exits at each exit`
  - `. So perhaps each assignment step reduces the problem?\n\nThe key requirement is that each possible outcome (number in [0, m)) occurs exactly floor(2^`

### 36. atom 4760 — mid-reasoning self-correction or alternatives
- MOVE: self-correction or proposing alternative reasoning paths in problem-solving (3) · FORM: chunks contain "Wait" or "Alternatively" followed by reconsideration or alternative approaches (3) · CONTENT: N/A (0)
- lens: '?' ' a' ' would' ' and' ' the' '.' ',' ' we' ':\n\n' ' if' ' problem' ' that' ' in' ' at' ' for'
- **statement**: After SFT on this data, model responses will show more chains of self-questioning in which a question mark is followed by a new paragraph beginning 'Alternatively, maybe ...' or 'Wait, maybe/no ...' that proposes a different hypothesis or retracts the previous one.  · kind=both · unit=per_1k_tokens · generic=False · conf=3
- regex: `\?\s*\n\s*\n\s*(?:Wait|Alternatively)\b` `\bAlternatively, maybe\b` `\bWait, (?:maybe|no|perhaps)\b`
- rubric: Does the response contain at least one sequence where the writer poses a question about the problem and then, in a new paragraph starting with 'Alternatively' or 'Wait', proposes a different hypothesis or retracts the previous idea?
- notes: Label: FORM(3) and MOVE(3), and all 3 evidence examples match the '?\n\nWait/Alternatively, maybe' pattern exactly (regexes hit 18/40 own chunks, 6-7x enriched). Lens ('?', ':\n\n', ' problem', ' if') consistent with question-then-break structure. Cleanest surface marker in this rank range. Overlaps rank 38 (paragraph-initial Wait/Alternatively), rank 34, and ranks 9/11 ('Alternatively, maybe'); this item is the hypothesis-chain variant.
- channel A: held-out chunks where this atom fires show excess Δlogp(SFT−base) > 0 after headroom control (double residualisation), and the excess ranks with mass rank
- 2×2:  → 
  - ` not on the y-axis, leading to two coinciding points on one side but which are symmetric, effectively counting as one point each but duplicates?\n\nWait, no`
  - ` of triangle ADC is equal to area of ABC?\n\nAlternatively, maybe flipping points around?\n\nAlternatively, maybe I need to consider absolute value incorrectly?\n\nWait, maybe there`

### 37. atom 2311 — tentative reasoning with "perhaps" and "maybe"
- MOVE: proposing alternative approaches or tentative solutions (3) · FORM: frequent use of "perhaps" and "maybe" (3) · CONTENT: N/A (0)
- lens: ' we' ' is' ' that' ' So' ' But' ' can' '.\n\n' ' so' ' the' ' if' ' and' ' from' ' it' ' =' '?\n\n'
- **statement**: After SFT on this data, model responses will show more hedged, tentative reasoning steps marked by 'perhaps', 'maybe', or 'I'm not sure' rather than assertive statements.  · kind=both · unit=per_1k_tokens · generic=False · conf=2
- regex: `(?i)\b(?:perhaps|maybe)\b` `\bI'?m not sure\b`
- rubric: Does the response contain at least one reasoning step stated tentatively (with 'perhaps', 'maybe', 'I'm not sure', or 'might be') rather than asserted as fact?
- notes: Label: FORM(3) 'perhaps'/'maybe'; MOVE(3) tentative solutions. Lens generic. Ex2 contains no hedge word (weak example). Regex is the broad hedge-word count (21/40 own, 570/2400 pooled, 2.2x enriched), so this item is the general-hedging measure of the family; ranks 6/30/35 target hedges attached to pivot words and are kept distinct. Confound: 'maybe' also appears in chat/QA answers as a politeness hedge. Not flagged generic: hedged reasoning is broad but many reasoning texts are fully assertive.
- channel A: held-out chunks where this atom fires show excess Δlogp(SFT−base) > 0 after headroom control (double residualisation), and the excess ranks with mass rank
- 2×2:  → 
  - ` it. But to find this, perhaps each step in the reversal must target the "earliest possible" to go back in time, but I'm not sure`
  - `'s an even earlier occurrence. The first occurrence will be the leftmost one, so every time there's an occurrence found, it's a candidate but the earliest`

### 38. atom 5971 — Proposing alternatives or reconsidering steps
- MOVE: The text is actively proposing alternative approaches or reconsidering previous steps in reasoning. (3) · FORM: Chunks frequently include the word "Alternatively" or "Wait" and often contain mid-sentence reasoning pauses (e.g., "?\n\nWait"). (3) · CONTENT: N/A (0)
- lens: ' in' ' of' ' So' ',' '?' '.\n\n' ' (' ' "' ' if' ').' ' would' ' that' ' ' "'s" 'So'
- **statement**: After SFT on this data, model responses will show more reconsideration of a just-completed step, introduced by a paragraph break followed by 'Wait' or 'Alternatively' (e.g., '... = 1?\n\nWait, to get ...').  · kind=both · unit=per_1k_tokens · generic=False · conf=2
- regex: `\n\s*\n\s*(?:Wait|Alternatively)\b` `\?\s*\n\s*\n\s*Wait\b` `\bperhaps it'?s (?:implied|meant|intended|assumed)\b`
- rubric: Does the response contain at least one point where, after completing a step or computation, the writer starts a new paragraph with 'Wait' or 'Alternatively' to reconsider, redo, or reinterpret that step?
- notes: Label: FORM(3), MOVE(3), but examples are inconsistent: ex1 is 'problem doesn't specify ... perhaps it's implied' (assumption-filling), ex2 is a cooking recipe with no reconsideration (off-label), only ex3 shows '?\n\nWait'. Lens weakly agrees. Paragraph-initial Wait/Alternatively is broad (469/2400 pooled, 1.3x enriched); '?\n\nWait' 3.5x. Near-duplicate of rank 36 (cleaner examples); third regex captures the assumption-filling variant from ex1 (1/40).
- channel A: held-out chunks where this atom fires show excess Δlogp(SFT−base) > 0 after headroom control (double residualisation), and the excess ranks with mass rank
- 2×2:  → 
  - `, but to do so we need to assume some form for f and g.\n\nSince the problem doesn't specify the type of functions, perhaps it's implied to`
  - ` 3/4 to 1 cup water, adjust as needed. Salt to taste. Oil for cooking. \n\nMake sure to mention medium heat to prevent burning`

### 39. atom 5908 — Structured reasoning corrections and verifications
- MOVE: The text is actively correcting, refining, or verifying a step in the reasoning process (e.g., "Actually:", "Check:", "However:"). (3) · FORM: Chunks often include structured reasoning steps with indentation, colons, and numbered/marked elements (e.g., "Step:", "index0:", "Example:"). (3) · CONTENT: N/A (0)
- lens: ' from' ' have' ' to' ' not' ' and' '?\n\n' ' for' ' can' '?' ' maybe' ' the' ' we' ' is' ' but' ' each'
- **statement**: After SFT on this data, model responses will show more indented outline-style worked examples made of short colon-terminated label lines (e.g., 'Actually:', 'Step:', 'Original:', 'The first element:') used to lay out or re-verify a computation.  · kind=both · unit=per_1k_tokens · generic=False · conf=2
- regex: `(?m)^[ \t]{4,}[A-Z][^\n:]{0,40}:[ \t]*$` `\bActually:\s` `\b(?:Step|Check|Original|Example|Verify):\s`
- rubric: Does the response lay out a worked example, trace, or verification as an indented outline of short colon-terminated label lines (e.g., 'Step:', 'Original:', 'Actually:') rather than as prose or as code?
- notes: Label: FORM(3) indented colon-labelled structure; MOVE(3) correcting/verifying via 'Actually:'. Top chunks strongly match the deep-indent colon-label style (regex 1: 8/40 own vs 22/2400 pooled, 22x enriched; label regex 4/40, 40x). Many label lines are lowercase ('condition:', 'originally:') and are not matched by the capital-letter regex; a lowercase variant was tested and gains nothing on the chunks while risking Python-code false positives, so the conservative regex is kept. Lens ('?\n\n', ' maybe') does not specifically support the label. Confound: Python block headers ('    for x in y:') end in ':' with indentation; the capitalized-first-char requirement excludes lowercase keywords, but capitalized class/constant lines could leak. Style is characteristic of one data source (deep-indented reasoning traces), so the effect hinges on that source's weight. Overlaps ranks 26/27 (structured traces).
- channel A: held-out chunks where this atom fires show excess Δlogp(SFT−base) > 0 after headroom control (double residualisation), and the excess ranks with mass rank
- 2×2:  → 
  - ` the third element: 3.\n\n                But we want a formula: \n                  The new circle: \n                    The first element: 5 = 1 +`
  - ` \n            Actually: \n                Step: \n                  Original: 101 -> right circular shift: \n                  The bits: [b0, b1, b`

### 40. atom 4684 — Structuring reasoning steps, often with code
- MOVE: The text is actively structuring or outlining steps in a reasoning process, often involving code or problem-solving. (3) · FORM: Chunks often include code snippets or pseudocode, frequently in Python, and contain phrases like "Wait," "So," or "Let me." (2) · CONTENT: N/A (0)
- lens: ':\n\n' "'s" ' the' 'Case' ' I' ' if' ' for' 'So' ' with' ' Which' ' a' '.\n\n' '1' ' which' ' first'
- **statement**: After SFT on this data, model responses will show more explicit outlining of an implementation plan before writing code ('Putting it all together, steps in code:', 'Let me outline the steps ...', 'code skeleton in Python:').  · kind=both · unit=per_response · generic=False · conf=2
- regex: `\bPutting (?:it|this|that) (?:all )?together\b` `\bLet me (?:outline|structure|lay out|list)\b` `\b(?:[Ss]teps|[Oo]utline|[Ss]keleton|[Pp]lan|approach)(?: (?:in|for) (?:the )?code)?:[ \t]*\n`
- rubric: Does the response present an explicit ordered outline or plan of implementation steps (e.g., 'Steps:', 'First ..., Then ...', 'code skeleton') before or instead of giving the full solution?
- notes: Label: MOVE(3) structuring/outlining steps; FORM(2) code snippets + 'Wait/So/Let me'. Lens (':\n\n', 'Case', 'So', ' first', '1') agrees moderately (colon-newline plan headers, enumeration). All 3 evidence examples show 'Putting it all together / Let me outline / code skeleton:' planning; regexes hit 3/40 own chunks (20x enriched). Third regex made to accept capitalized 'Steps:/Outline:/Plan:' headers (rubric example uses 'Steps:'). Confound: outlining before code is common in instruction-tuned models generally; mostly relevant on code prompts, so per-response rate on the mixed prompt set is diluted. Overlaps ranks 25/31. Rubric primary.
- channel A: held-out chunks where this atom fires show excess Δlogp(SFT−base) > 0 after headroom control (double residualisation), and the excess ranks with mass rank
- 2×2:  → 
  - ` should be fine.\n\nPutting it all together, steps in code:\n\nRead all products from stdin until empty line.\n\nThen process the products.\n\nFirst, filter step:`
  - ` here.\n\nOkay, moving on.\n\nLet me structure the steps again, in order of priority.\n\nAssuming the function is written as follows: \n\ndef validate_choice`

# grad_v2 per-category aggregation

32,761 of 32,768 atoms assigned (DeepSeek). Primary category basis. mass = clean-train mass share.

Decoder geometry: mean pairwise cosine of same-category atoms in activation space (-W dot dec). Global random baseline 0.0 (a category above this also clusters geometrically).

| Category | atoms | mass% | low-confidence% | Matryoshka g0/g1/g2 | decoder cos | ledger items | channel-B hit |
|---|---|---|---|---|---|---|---|
| SELF_CORRECTION | 13629 | 34.01 | 0.0 | 2936/3704/6989 | -0.0001 | 14 | 0.786 |
| PLANNING | 5050 | 9.87 | 0.0 | 817/1445/2788 | 0.0 | 2 | 1.0 |
| FORMATTING_MARKUP | 1536 | 9.19 | 0.2 | 994/214/328 | -0.0001 | 0 | — |
| MULTILINGUAL_CONTENT | 2388 | 8.58 | 0.0 | 917/484/987 | 0.0 | 0 | — |
| VERIFICATION | 2239 | 7.58 | 0.0 | 504/472/1263 | 0.0001 | 6 | 0.833 |
| MATH_STEPS | 821 | 7.26 | 0.0 | 397/99/325 | -0.0001 | 3 | 0.333 |
| ALTERNATIVE_PROPOSALS | 1122 | 3.83 | 0.0 | 199/296/627 | 0.0001 | 5 | 0.8 |
| HEDGING | 1343 | 3.75 | 0.0 | 263/367/713 | -0.0 | 4 | 1.0 |
| CODE_REASONING | 528 | 3.14 | 0.0 | 183/107/238 | -0.0001 | 0 | — |
| TECHNICAL_NOTATION | 678 | 2.95 | 0.0 | 295/105/278 | -0.0001 | 0 | — |
| METACOGNITIVE_REFLECTION | 1791 | 2.65 | 0.0 | 219/488/1084 | -0.0001 | 1 | 1.0 |
| EDGE_CASE_CHECKING | 580 | 1.98 | 0.0 | 110/159/311 | 0.0001 | 1 | 1.0 |
| CAUSAL_JUSTIFICATION | 409 | 1.8 | 0.0 | 74/102/233 | -0.0001 | 4 | 0.75 |
| COMMUNICATION_STYLE | 242 | 1.47 | 0.4 | 121/52/69 | 0.0 | 0 | — |
| DOMAIN_CONTENT | 204 | 0.86 | 0.5 | 61/60/83 | 0.0001 | 0 | — |
| CREATIVE_NARRATIVE | 111 | 0.62 | 0.0 | 59/23/29 | -0.0005 | 0 | — |
| SAFETY_REFUSAL | 90 | 0.45 | 0.0 | 40/14/36 | 0.0002 | 0 | — |

## Matryoshka group x semantic category

g0 = first 8192 (coarse reconstruction), g1 = 8192-16384, g2 = 16384-32768 (fine). If a category concentrated in one group, matryoshka resolution would align with meaning; since categories spread evenly, **resolution and meaning are independent** (matryoshka does not create categories).

Group distribution over all atoms: g0 8189, g1 8191, g2 16381. Per-category group shares (within category):
| Category | g0% | g1% | g2% |
|---|---|---|---|
| SELF_CORRECTION | 21 | 27 | 51 |
| PLANNING | 16 | 28 | 55 |
| FORMATTING_MARKUP | 64 | 13 | 21 |
| MULTILINGUAL_CONTENT | 38 | 20 | 41 |
| VERIFICATION | 22 | 21 | 56 |
| MATH_STEPS | 48 | 12 | 39 |
| ALTERNATIVE_PROPOSALS | 17 | 26 | 55 |
| HEDGING | 19 | 27 | 53 |
| CODE_REASONING | 34 | 20 | 45 |
| TECHNICAL_NOTATION | 43 | 15 | 41 |
| METACOGNITIVE_REFLECTION | 12 | 27 | 60 |
| EDGE_CASE_CHECKING | 18 | 27 | 53 |
| CAUSAL_JUSTIFICATION | 18 | 24 | 56 |
| COMMUNICATION_STYLE | 50 | 21 | 28 |
| DOMAIN_CONTENT | 29 | 29 | 40 |
| CREATIVE_NARRATIVE | 53 | 20 | 26 |
| SAFETY_REFUSAL | 44 | 15 | 40 |
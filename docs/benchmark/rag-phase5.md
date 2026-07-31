# Benchmark — RAG validation stage (Phase 5)

Reproduce with:

```bash
poetry run python docs/benchmark/bench_rag.py
```

Measures `rag.validator.correct_province` against the `<15ms` RAG budget in
`CLAUDE.md`. Input is province text degraded the way Phase 3's recognizer was
observed to degrade it — no image, no model, no vector store.

## Result — 2026-07-31

Python 3.12.13, macOS 26.5.2, arm64.

| Degradation (all 77 provinces) | recovered | abstained | **wrong** |
|---|---|---|---|
| marks stripped | 77 | 0 | **0** |
| marks + trailing junk (`+ ดี`) | 74 | 3 | **0** |
| marks + digit bleed (`+ " 9"`) | 75 | 2 | **0** |

**226 of 231 recovered, 5 abstained, 0 mis-attributed.**

The two misreads Phase 3 actually produced for a ชลบุรี plate:

| Input | Result |
|---|---|
| `'ชลบรดี'` | → `ชลบุรี` @ 0.800 |
| `'ชลบร 9'` | → `ชลบุรี` @ 0.800 |

Text that names no province — `'กข 1234'`, `'1กข 2345'`, `'VEZL'`, `''`,
`'   '` — resolved to `None` in every case. No false positives.

| Latency (worst case: 77-way scan) | |
|---|---|
| median | **0.4415 ms** |
| p95 | **0.4824 ms** |
| budget | 15 ms — **MET, 31× under** |

This is the first stage in the project to meet its stated budget. Detection is
still unmeasured for want of weights; OCR misses `<40ms` by ~10×.

## Why lexical retrieval, not embeddings

`docs/PLAN.md` Part A named chromadb + sentence-transformers. Measured against
the actual problem, both were rejected — see Part F for the full argument. In
short: the knowledge base is 77 static proper nouns and the damage is
character-level, so edit distance models it directly, in 0.44 ms, with no
dependency. A sentence-transformer would cost more than the entire 15 ms budget
in inference alone, before the ~500 MB of dependencies.

## The threshold finding

Two guards decide whether a fuzzy result is returned: a score floor
(`MIN_SCORE = 0.6`) and a margin over the runner-up (`MIN_MARGIN = 0.15`).

A grid sweep over 308 degraded candidates showed **the margin, not the floor,
is what provides safety**:

| min_margin | correct | **wrong** |
|---|---|---|
| 0.00 | 304 | **4** |
| 0.05 | 304 | **1** |
| 0.10 | 302 | **1** |
| **0.15** | **296** | **0** |
| 0.20 | 280 | 0 |

The floor changes nothing about safety, because non-province text scores around
0.17 against the list — nowhere near any plausible threshold. What actually
causes wrong answers is two *real* provinces sitting close together:

| Closest real pairs | similarity |
|---|---|
| เพชรบุรี ~ เพชรบูรณ์ | 0.857 |
| ระนอง ~ ระยอง | 0.800 |
| จันทบุรี ~ นนทบุรี | 0.800 |

0.15 is the smallest margin that never returns the wrong province. It is chosen
to abstain rather than guess, which is why 5 candidates resolve to `None`.

## Caveats — read before quoting these numbers

1. **This measures the corrector, not the pipeline.** Inputs are degraded
   programmatically. Only two of them (`'ชลบรดี'`, `'ชลบร 9'`) came from a real
   recognizer run, and that run was on *rendered text, not a photograph*.
2. **The degradation model is inferred from a single plate.** Phase 3 was
   observed dropping marks, hallucinating a trailing syllable and bleeding a
   digit. Real photographs will fail in ways not modelled here.
3. **Truncation is excluded, and cannot be fixed.** Losing the final character
   of เพชรบูรณ์ leaves `เพชรบร`, which *is* the mark-free skeleton of เพชรบุรี.
   The distinguishing character is gone, so the corrector returns เพชรบุรี with
   full confidence. It is the only such pair in the 77 — asserted by
   `test_truncating_phetchabun_aliases_onto_phetchaburi`.
4. **Abstention is not free.** 5 of 231 return `None`. Phase 6 must present a
   missing province as unknown, not as an error.

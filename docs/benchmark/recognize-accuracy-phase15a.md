# Recognition accuracy on real plates — Phase 15a

First **accuracy** measurement of the full `/recognize` pipeline against real
Thai plate photographs (earlier benchmarks measured latency on synthetic text
only). Reproduce with:

```
make bench-recognize-accuracy
```

Ground truth: `eval/plates.jsonl`, 22 hand-verified images from the
`valid/` + `test/` splits of `datasets/raw/thailand-license-plates-v1/`.
Scoring functions: `eval/scoring.py` (unit-tested).

## Result

| Metric | Baseline | After row-grouping fix |
|---|---|---|
| Plate exact-match | 77.3% | 77.3% |
| Mean character error rate | 0.153 | **0.063** |
| Province accuracy | 90.9% | **95.5%** |

The pipeline reads real plates correctly on 17/22 images, at 0.94–0.99 OCR
confidence on the clean ones. This retires the long-standing "no real-plate
happy path ever observed" caveat.

## The one structural fix

`group_into_rows` anchored each visual row to its accumulated top-to-bottom
span. On a perspective-corrected (skewed) crop, one tall number-row fragment
dragged the span's bottom edge down until the province band below overlapped
it, merging the province **into** the plate number:

```
pred='ดว กรงเทพมหานคร 4301'   truth='ฎว 4301'   (CER 2.17, province lost)
```

Fix: anchor each row to its **topmost fragment**, fixed when the row opens, so
a row cannot grow into the band beneath it. The province's overlap with the
number row's anchor is 22% (< the 50% threshold), so it separates correctly;
genuine same-row letter+digit fragments still overlap > 70%. Pinned by
`tests/unit/test_ocr_reading.py::test_group_into_rows_does_not_let_a_tall_fragment_bridge_the_row_below`.

## Remaining misses (all pure OCR, out of 15a scope)

| Image | pred | truth | Nature |
|---|---|---|---|
| ...VfP | `ญณข 295` | `ญข 2951` | inserted ณ, dropped digit |
| ...QQv | `3998` | `ฎผ 3998` | prefix not detected by OCR |
| ...ap0O | `432` | `บก 432` | decorative plate, prefix + province lost |
| ...ktUd | `ดว 4301` | `ฎว 4301` | ฎ misread as ด (province now correct) |
| ...z3OP | `1ขข 7916` | `1ขช 7916` | ช misread as ข |

These are character-level recognition errors in the OCR model itself; closing
them needs a model change (fine-tuning or a different recognizer), deferred.
Latency is unchanged and remains Phase 15b's concern.

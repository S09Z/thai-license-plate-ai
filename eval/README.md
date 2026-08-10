# Recognition accuracy eval set

`plates.jsonl` is a small, hand-verified ground-truth set for grading the full
`/recognize` pipeline against **real** Thai plate photographs — the dataset in
`datasets/raw/thailand-license-plates-v1/` carries detection boxes only (class
`license-plate`), with no plate-number strings, so this file supplies them.

## Format

One JSON object per line:

```json
{"img": "<file name under the dataset splits>", "plate": "กข 1234", "province": "กรุงเทพมหานคร", "note": "optional"}
```

- `img` — file name only; the runner searches the `valid/` and `test/` image
  folders for it.
- `plate` — the true plate string. Spacing is ignored when scoring, so
  `"กข 1234"` and `"กข1234"` are equivalent.
- `province` — the canonical province name, compared exactly.
- `note` — optional; flags an entry that still needs human confirmation
  (e.g. decorative or stylized plates).

## Provenance

Transcribed from the perspective-corrected crops on 2026-08-10 and intended for
human verification. The dataset is Roboflow, licensed CC BY 4.0.

## Running

```
make bench-recognize-accuracy
```

Reports plate exact-match %, mean character error rate, and province accuracy,
plus a per-image list of every miss. Scoring functions live in `eval/scoring.py`
and are unit-tested in `tests/unit/test_eval_scoring.py`.

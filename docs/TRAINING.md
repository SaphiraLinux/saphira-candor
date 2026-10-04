# Training Candor: how, why, when, and when not to

## How

Training data is JSONL. One witnessed fact per line:

```json
{"context": "Note: the harbor heron is swift. What is true of the harbor heron?",
 "options": ["swift", "louder", "darker", "rougher"], "label": 0,
 "entity": "harbor heron", "fact": "swift"}
```

- `context` / `options` / `label` teach the vector path (required).
- `entity` + `fact` additionally register the row as a *witnessed fact*
  (optional, but without them there is no REMEMBERED path — see below).
- Shuffle option positions across rows, or the model learns position, not content.
- 2–4 options per row; train with `--n-opt` at least as large as the widest
  row you will ever ask (asking wider than the model hangs — size the cache
  for the questions, not the training).

```sh
candor train --data train.jsonl --val val.jsonl --out mymem.bin \
    --epochs 30 --seed 7 --n-opt 8
```

Watch `final_val_top1` and stop reasoning there: the trainer keeps the best
validation checkpoint automatically, so over-training only costs time, not the
model. Retrain from a checkpoint any time:

```sh
candor train --data new.jsonl --val val.jsonl --out mymem2.bin \
    --init-model mymem.bin --epochs 10 --seed 7
```

Smaller checkpoints without retraining: `--q8` (3.6x smaller, zero measured
cost here), `--q4` (5.3x), `--q3` (6.4x, costs ~6 answers per 500), `--q2`
(8x, costs 11 points of transfer — rejected for accuracy-bearing use).
Measure on *your* content; the ladder in `docs/FORMAT.md` records ours.

## Why this shape

Two memories, two jobs. The witnessed-fact table answers what was *seen*,
exactly, with source — it cannot forget by construction (exact lookup, no
weights involved). The vector path answers what was *not* seen and is always
marked GUESSED, so its failures are visible instead of confident. Training
improves the guesses; it can never corrupt the memories, because they live in
different machinery. That separation is the entire design.

## When to train

- New facts to remember: add witnessed rows, retrain (or warm-start).
- New question shapes: the model only answers shapes it has seen — if you ask
  a new template, teach rows in that template first (the logic-template
  experiment: 16 rows fixed two confident-wrong answers).
- Calibration matters: wider models (65k→256k params) kept buying calibration
  after accuracy saturated. If your guesses must carry honest probabilities,
  spend parameters there.

## When NOT to train

- **Broad web text at fixed capacity.** 20k rows of everything gave a thin
  ranking signal (+2.8pts over chance), not reliable selection. Narrow,
  task-shaped data trains to 1.0; the open ocean at 41–65k params trains to
  mush. More data of the *same narrow kind* helps robustness; more data of
  *every kind* hurts per-task accuracy.
- **To fix a representation limit.** Frame/wording sensitivity survived every
  width we tried. If the failure is systematic across phrasings, the next
  experiment changes the representation or the encoder, not the epoch count.
- **Past the validation plateau.** Train NLL near its floor with val flat
  means done. The tiny-dataset logic run memorized after ~epoch 20; the
  checkpoint selector handed back the pre-memorization model automatically.
  Trust it.
- **When the answer is already witnessed.** Retraining cannot improve an
  exact lookup. If REMEMBERED is wrong, the witness row is wrong — fix the
  row, not the weights.

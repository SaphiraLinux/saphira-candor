# Training material: what to feed Candor, and where big material comes from

Candor trains on JSONL rows shaped exactly like this:

```json
{"context": "Note: the harbor heron is swift. What is true of the harbor heron?",
 "options": ["swift", "louder", "darker", "rougher"], "label": 0,
 "entity": "harbor heron", "fact": "swift"}
```

`context`/`options`/`label` teach the vector path. `entity`+`fact` additionally
register the row as a *witnessed fact* (enables REMEMBERED answers); rows
without them train vectors only. Shuffle option positions or the model learns
position, not content. See `examples/` for minimal working sets and
`docs/TRAINING.md` for the full discipline.

## Small, exact material (start here)

`examples/facts_train.jsonl` + `examples/facts_val.jsonl`: a dozen witnessed
rows. Trains in seconds, answers exactly. This is the loop to learn first.

## Large, real material (when small stops teaching anything new)

Raw web text cannot be fed directly — it must be mined into rows first.
`tools/mine.py` does exactly that, deterministically:

```sh
python3 tools/mine.py --input owt_slice.jsonl --out-dir mined/ \
    --seed 4171 --max-train 20000
```

The contract it enforces (same-document distractors, byte budgets C=192/O=32,
document-level train/val splits, fixed seed) is documented in its header.
Manageable slices first: mine thousands, measure, and only then scale. Twenty
thousand rows of everything once yielded a thin ranking signal (+2.8pts over
chance), not reliable selection — broad data at fixed capacity is a
documented way to spend compute on mush (see TRAINING.md, "when NOT").

## Sources for large raw text

- **OpenWebText** (public web-text corpus, ~12GB raw): the well-known public
  mirror is published alongside the GPT-2 output-dataset work
  (Skylion007/OpenWebTextCorpus). Verify the mirror yourself before use;
  links rot and corpora get re-cut. Any JSONL slice with `text` fields feeds
  `tools/mine.py` after trivial reshaping.
- **Your own documents**: the highest-signal source. Witnessed, keyed rows
  (`entity`+`fact`) beat any volume of mined web text for memory duty.

Rule of thumb, measured: narrow task-shaped data trains to 1.0; broad data at
41–65k params trains to a ranking signal. Buy capacity (wider models) or
narrow the task before buying gigabytes.

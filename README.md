# Saphira Candor

> **Naming rule, stated first because it was nearly lost once:**
> `candor` = external model + external witnessed memory.
> `scandor` = Self-Contained Candor; same engine, model and memory both baked inside.
> Candor is the engine; SCandor is Candor sealed into one executable.

Candor answers questions from witnessed facts first and says so; anything
else runs the neural forward pass and is marked as a guess. It never presents
a guess as a memory.

- **REMEMBERED** — exact witnessed fact, source supplied, neural forward path
  skipped entirely.
- **GUESSED** — neural inference with scores/probabilities, always marked.

## Which binary

| | `candor` | `scandor` |
|---|---|---|
| model / memory | `--model F` + `--witness F` files | compiled in |
| train / retrain | yes (`candor train ...`) | no (ask-only by construction) |
| files read at runtime | supplied by you | none, ever |

```sh
make candor     # full version: train, retrain, ask
make scandor    # self-contained answer machine
make test       # runtime suite, both binaries
```

## Quick start

```sh
# teach + ask with candor (full version)
./candor train --data examples/facts_train.jsonl --val examples/facts_val.jsonl \
    --out /tmp/mymem.bin --epochs 30 --seed 7 --n-opt 8
./candor ask --model /tmp/mymem.bin --witness examples/facts_train.jsonl \
    --context "What is true of the harbor heron?" \
    --candidate swift --candidate louder
# REMEMBERED: harbor heron is swift. (+ source; forward pass not run)

# ask the sealed binary (nothing to load, nothing to train)
./scandor --context "What is true of the tract0007 pipit?" \
    --candidate alert --candidate louder
./scandor --about   # baked hashes: model, witness rows, container
```

## Provenance

`candor --about --model F --witness F` hashes the loaded content.
`scandor --about` reports baked hashes unconditionally. The commit identifies
the engine AND its education/memory:

- model: 65k params, int8 q8 container, 73,372 bytes
- witness: 1,616 rows

See `docs/TRAINING.md` (how/why/when/when-not), `docs/PHILOSOPHY.md` (why this
exists instead of a rental), `docs/DATA.md` (training material sources),
`docs/FORMAT.md` (checkpoint container), `docs/PROVENANCE.md` (lineage).

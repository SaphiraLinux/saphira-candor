# Provenance: where Candor comes from

Candor is a product cut, not a fresh invention. Its lineage, in order:

1. **Upstream C trainer** (AKADATA saphira-sjev-c: byte-level TinyScorer,
   trainer, exactness discipline). The engine's forward pass, trainer, SIMD
   kernels and data loader descend from it. See NOTICE for the full legal
   attribution, which is preserved byte-for-byte and is the authoritative
   record — this file is engineering provenance, not legal text.
2. **Sibling lean core** (sjev-next): staged CLI, bandit calibration stages,
   SIMD discipline, `--n-opt` sizing, REPL.
3. **Memory program** (sjev-freerun value series + scjev): witnessed-fact
   table, REMEMBERED/GUESSED verdict discipline, opaque-ID/seed controls,
   frame-invariance findings, int8/3-bit/4-bit/2-bit storage ladder,
   zero-logit readout fix, hang-class kill.
4. **This repository**: the merge of the best of all worlds into two
   products — full `candor` (train included) and sealed `scandor` — with
   every identifier renamed to the product's own name.

Deliberate breaks from lineage, all recorded where they happened:
- Storage containers v2–v5 and the `--q*` flags are new here (upstream knew
  float32 only).
- The `n_opt` hang class and the zero-logit readout defect were found and
  fixed in this line, not inherited.
- The checkpoint magic is `CDR1`, not the upstream magic: old files do not
  silently load. The single vendored model was converted by a 4-byte header
  patch (weights untouched) and re-verified before baking.

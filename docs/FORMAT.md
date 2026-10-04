# Candor checkpoint format — CDR1 container, versioned

Standalone format. No training code at read time, no external dependencies.
Little-endian throughout; FP32 IEEE-754; tensors tightly packed, C-contiguous
row-major, matching the in-memory layout they are read straight into.

## Header (40 bytes: 10 x uint32 LE)

| offset | field   | meaning                                              |
|--------|---------|------------------------------------------------------|
| 0      | magic   | `0x31524443` (`"CDR1"` in ASCII order)              |
| 4      | W       | width (1..4096, else refused)                        |
| 8      | R       | rank (1..4096, else refused)                         |
| 12     | C       | context bytes (1..8192, else refused)                |
| 16     | O       | option bytes (1..8192, else refused)                 |
| 20     | n_opt   | options/example at train time (1..1024, else refused)|
| 24     | arch    | enum below (anything else refused)                   |
| 28     | version | `1` float32, `2` block 3-bit, `3` block 8-bit, `4` block 4-bit, `5` block 2-bit (else refused) |
| 32     | n_emb   | float count of the embedding table                   |
| 36     | n_qkv   | float count of each Q/K/V map                        |

Loader validation, in order: magic exact, version known, dims in range, arch
in the known set, then truncation checks per tensor read. Anything else fails
closed with an error string; nothing half-loaded is ever returned.

## Arch enum

`0 baseline, 1 interact, 2 xbid, 3 twohead, 4 sumhead, 5 tw_sk, 6 tw_sv,
7 tw_skv, 9 sum_k, 10 sum_v, 11 sum_kv` (8 intentionally absent).

## Tensor payload (immediately after header)

Shared tensor order across all versions (a version that reordered tensors
would be a new container, not a new version):

1. `emb` `[n_emb]`, `pos` `[C][W]`
2. `context_norm` + `option_norm` weight/bias `[W]` each
3. `Wq, Wk, Wv` `[n_qkv]` each
4. iff interact: `Wo [n_qkv]`, `gc [W]`, `gb [W]`
5. iff twohead-family: `Wq2 [n_qkv]`; plus `Wk2` / `Wv2` for the `_SK/_K`,
   `_SV/_V`, `_SKV/_KV` variants as applicable
6. iff xbid: six `[R]` norm tensors

Per-version encoding of each tensor: v1 raw FP32 LE; v2 32-weight blocks of
`{float32 min, float32 scale, 12 packed bytes}` (3-bit uniform levels);
v3 32-weight blocks of `{float32 scale, 32x int8}` (symmetric);
v4/v5 the generic N-bit blocked form (4-bit: 16 bytes/block; 2-bit: 8
bytes/block). All quantized forms promote to float32 on load: compute
precision is identical across versions by construction.

Sizes are recomputed from the header, never trusted beyond the two stored
counts. Derived transposes are rebuilt after load, never persisted.

## Witness files (ask-time memory, not checkpoints)

JSONL rows carrying `"entity"` + `"fact"` keys (training rows may carry them
alongside context/options/label; both loaders ignore each other's keys).
Exact string match only. No scores, no thresholds.

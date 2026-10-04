#!/bin/sh
# Candor deployment runtime tests. No training OF the product here -- but the
# candor binary trains, so one tiny smoke training proves the full version
# actually contains a working trainer (a Trainer-shaped binary that cannot
# train would be worse than an honest ask-only one).
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PASS=0
FAIL=0
ok()  { PASS=$((PASS+1)); echo "  PASS $1"; }
bad() { FAIL=$((FAIL+1)); echo "  FAIL $1${2:+ -- $2}"; }

echo "== no predecessor names anywhere in the product tree =="
# PROVENANCE.md + LICENSE/NOTICE name the lineage openly and are excluded by
# design; everything else must be clean. (snprintf is libc, not a reference.)
if grep -rniE "sjev|sjnext|(^|[^a-zA-Z_])sn_|(^|[^a-zA-Z_])SN_" \
    "$ROOT/src" "$ROOT/tools" "$ROOT/tests" --exclude=runtime.sh "$ROOT/Makefile" \
    "$ROOT/README.md" "$ROOT/docs/TRAINING.md" "$ROOT/docs/PHILOSOPHY.md" \
    "$ROOT/docs/DATA.md" "$ROOT/docs/FORMAT.md" 2>/dev/null | grep -v snprintf | grep -q .; then
    bad "zero predecessor-name references"
    echo "    (docs/PROVENANCE.md + LICENSE/NOTICE name the lineage openly)"
    grep -rniE "sjev|sjnext" "$ROOT/src" "$ROOT/tools" 2>/dev/null | head -5
else
    ok "zero predecessor-name references"
fi

echo "== build from clean tree =="
make -C "$ROOT" clean >/dev/null 2>&1
if make -C "$ROOT" >/dev/null 2>&1; then ok "clean build"; else bad "clean build"; exit 1; fi
[ -x "$ROOT/candor" ] && [ -x "$ROOT/scandor" ] \
    && ok "top-level candor + scandor exist" || bad "top-level binaries exist"

echo "== static artefacts =="
file "$ROOT/candor" | grep -q "statically linked" \
    && ok "candor statically linked" || bad "candor statically linked"
file "$ROOT/scandor" | grep -q "statically linked" \
    && ok "scandor statically linked" || bad "scandor statically linked"
if nm "$ROOT/scandor" 2>/dev/null | grep -qE " cd_train$| adam_step|bandit"; then
    bad "scandor has no training symbols"
else
    ok "scandor has no training symbols"
fi
# Asymmetric by product definition: candor IS the full version, so its trainer
# symbols must be present. Both directions are asserted so neither binary can
# silently become the other.
if nm "$ROOT/candor" 2>/dev/null | grep -qE " cd_train$"; then
    ok "candor contains the trainer (full version)"
else
    bad "candor contains the trainer (full version)"
fi

echo "== CLI contract =="
"$ROOT/candor" --help >/dev/null 2>&1; [ $? -eq 0 ] \
    && ok "candor --help exit 0" || bad "candor --help exit 0"
"$ROOT/candor" --version 2>&1 | grep -q "^candor " \
    && ok "candor --version" || bad "candor --version"
"$ROOT/scandor" --version >/dev/null 2>&1; [ $? -eq 0 ] \
    && ok "scandor --version exit 0" || bad "scandor --version exit 0"
"$ROOT/scandor" --version 2>&1 | grep -q "^scandor " \
    && ok "scandor --version" || bad "scandor --version"
"$ROOT/scandor" --about 2>&1 | grep -q "model_sha256=a439b6962b9bd8e2b3800424a13f1726f662fac5a5029eb7c6875734e6eee5da" \
    && ok "scandor --about baked model hash" || bad "scandor --about baked model hash"
"$ROOT/scandor" --about 2>&1 | grep -q "witness_rows=1616" \
    && ok "scandor --about witness rows" || bad "scandor --about witness rows"
"$ROOT/candor" --about --model "$ROOT/models/model.q8" \
    --witness "$ROOT/witnesses/witness.jsonl" 2>&1 | grep -q "model: .*sha256=a439b696" \
    && ok "candor --about loaded model hash" || bad "candor --about loaded model hash"
"$ROOT/candor" --about --model "$ROOT/models/model.q8" \
    --witness "$ROOT/witnesses/witness.jsonl" 2>&1 | grep -q "witness: .*sha256=8b62d00a" \
    && ok "candor --about loaded witness hash" || bad "candor --about loaded witness hash"
"$ROOT/candor" --bogus-flag >/dev/null 2>&1; [ $? -eq 2 ] \
    && ok "unknown flag exit 2" || bad "unknown flag exit 2"

echo "== candor trains (smoke: it is the full version) =="
(cd /tmp && "$ROOT/candor" train --data "$ROOT/examples/facts_train.jsonl" \
    --val "$ROOT/examples/facts_val.jsonl" --out /tmp/candor_smoke.bin \
    --epochs 10 --seed 7 --n-opt 8 >/dev/null 2>&1) \
    && ok "candor train smoke" || bad "candor train smoke"
(cd /tmp && "$ROOT/candor" ask --model /tmp/candor_smoke.bin \
    --witness "$ROOT/examples/facts_train.jsonl" \
    --context "What is true of the harbor heron?" \
    --candidate swift --candidate louder 2>&1 | grep -q "REMEMBERED: harbor heron is swift") \
    && ok "freshly trained candor remembers" || bad "freshly trained candor remembers"
rm -f /tmp/candor_smoke.bin

echo "== verdicts, from an empty directory =="
D=$(mktemp -d)
"$ROOT/scandor" --context "What is true of the tract0007 pipit?" \
    --candidate alert --candidate louder 2>&1 | grep -q "REMEMBERED: tract0007 pipit is alert" \
    && ok "scandor remembered + source" || bad "scandor remembered + source"
(cd "$D" && "$ROOT/scandor" --context "What is true of the tract0007 pipit?" \
    --candidate alert --candidate louder --candidate darker --json 2>&1 | tail -1 | \
    python3 -c "import json,sys; d=json.load(sys.stdin); sys.exit(0 if d['verdict']=='remembered' and d['answer']=='alert' else 1)") \
    && ok "scandor remembered JSON from empty dir" || bad "scandor remembered JSON from empty dir"
(cd "$D" && "$ROOT/scandor" --context "What is true of the tract0007 pipit?" \
    --candidate louder --candidate darker --json 2>&1 | tail -1 | \
    python3 -c "import json,sys; d=json.load(sys.stdin); sys.exit(0 if d['verdict']=='guessed' else 1)") \
    && ok "scandor guessed JSON from empty dir" || bad "scandor guessed JSON from empty dir"
(cd "$D" && "$ROOT/candor" ask --model "$ROOT/models/model.q8" \
    --witness "$ROOT/witnesses/witness.jsonl" \
    --context "What is true of the tract0007 pipit?" \
    --candidate alert --candidate louder 2>&1 | grep -q "REMEMBERED: tract0007 pipit is alert") \
    && ok "candor remembered from empty dir" || bad "candor remembered from empty dir"
(cd "$D" && "$ROOT/candor" ask --model "$ROOT/models/model.q8" \
    --witness "$ROOT/witnesses/witness.jsonl" \
    --context "What is true of the tract0007 pipit?" \
    --candidate louder --candidate darker --json 2>&1 | tail -1 | \
    python3 -c "import json,sys; d=json.load(sys.stdin); sys.exit(0 if d['verdict']=='guessed' else 1)") \
    && ok "candor guessed JSON from empty dir" || bad "candor guessed JSON from empty dir"
rm -rf "$D"

echo "== negatives =="
(cd /tmp && "$ROOT/candor" ask --model /nonexistent.bin \
    --context "x?" --candidate a --candidate b >/dev/null 2>&1); [ $? -ne 0 ] \
    && ok "missing model fails closed" || bad "missing model fails closed"
printf 'JUNK' > /tmp/candor_badmodel.bin
(cd /tmp && "$ROOT/candor" ask --model /tmp/candor_badmodel.bin \
    --context "x?" --candidate a --candidate b >/dev/null 2>&1); [ $? -ne 0 ] \
    && ok "bad magic refused" || bad "bad magic refused"
rm -f /tmp/candor_badmodel.bin

echo "== man pages render =="
for p in candor scandor; do
    groff -man -Tascii "$ROOT/docs/man/man1/$p.1" 2>/dev/null | grep -q "REMEMBERED" \
        && ok "man $p renders" || bad "man $p renders"
done

echo
echo "  $PASS passed, $FAIL failed"
[ "$FAIL" -eq 0 ]

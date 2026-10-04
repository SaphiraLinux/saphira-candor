#!/usr/bin/env python3
"""Mine deterministic cloze/ranking rows from the 110MB OWT slice. No tokenizer.

Row shape (exactly what candor trains and asks on):
    context               : tail of the preceding text, <= 160 bytes
    options[4]            : correct next-span + 3 distractors, each <= 30 bytes
    label                 : index of the correct span
    source                : {url, offset} of the boundary in the source document

Rules, all frozen and recorded in the manifest (change any of them and the
corpus hash changes, which is the point):
  * DOCUMENT BOUNDARIES ARE NEVER CROSSED. Context, correct span and all three
    distractors come from the SAME document. A row spanning documents would
    teach cross-document association, which is contamination, not signal.
  * DISTRACTORS ARE SAME-DOCUMENT SPANS, not random garbage. They share the
    document's topic, register and style, so beating them requires reading the
    context rather than rejecting obvious junk. A distractor equal to the
    correct span, or duplicated within the option set, disqualifies the row.
  * BYTE BUDGETS, not char budgets: context <= 160 bytes, options <= 30 bytes,
    because the model truncates context at C=192 bytes and options at O=32 and
    silently drops the rest. A row that exceeds budget would train on text the
    model never sees. OWT carries multibyte punctuation, so char counting would
    lie; bytes don't.
  * DOC FILTER: fasttext en score > 0.8, text 500..20000 chars, >= 6 sentences.
    Boilerplate, stubs and mega-pages are out.
  * TRAIN/VAL SPLIT IS BY DOCUMENT (every 10th doc to val), never by row. Rows
    from one document share its topic; splitting by row would leak it.
  * DETERMINISM: file order processing, fixed RNG seed, shuffled option order
    recorded via label. Same script + same slice = byte-identical corpus.

Known limitation, stated not hidden: a same-document distractor can
coincidentally continue the context acceptably. That is label noise, and its
rate is unmeasured. It biases measured accuracy DOWN (a right answer marked
wrong), never up, so reported numbers are conservative.
"""

import hashlib
import json
import random
import argparse
import hashlib
import json
import random
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent

ap = argparse.ArgumentParser(description=__doc__)
ap.add_argument("--input", required=True, help="raw JSONL slice (owt_train.txt shape)")
ap.add_argument("--out-dir", required=True, help="where to write *_train.jsonl, *_val.jsonl, manifest")
ap.add_argument("--seed", type=int, default=4171)
ap.add_argument("--max-train", type=int, default=20000)
NS = ap.parse_args()
SRC = Path(NS.input)
SEED = NS.seed
MAX_ROWS_TRAIN = NS.max_train
OUT = Path(NS.out_dir)
OUT.mkdir(parents=True, exist_ok=True)


CTX_BYTES = 160
OPT_BYTES = 30
VAL_EVERY = 10  # every 10th document -> val

SENT_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'])")
WORD_CHARS = re.compile(r"[A-Za-z0-9]")


def clean(t):
    """Make text safe for the C loader, which scans quotes naively (strchr for
    the closing quote, no escape handling). Deterministic: straight quotes
    become apostrophes, backslashes are dropped. Recorded in the manifest."""
    return t.replace("\\", "").replace('"', "'").replace("\u201c", "'").replace(
        "\u201d", "'")


def sentences(text):
    parts = SENT_SPLIT.split(text.strip())
    out = []
    for p in parts:
        p = " ".join(p.split())
        if len(p) >= 15 and len(WORD_CHARS.findall(p)) >= 3:
            out.append(p)
    return out


def head_span(sentence):
    """First words of the sentence fitting OPT_BYTES, cut at a word boundary."""
    b = sentence.encode("utf-8")
    if len(b) <= OPT_BYTES:
        return sentence
    cut = b[:OPT_BYTES].decode("utf-8", errors="ignore")
    cut = cut.rsplit(" ", 1)
    s = cut[0] if len(cut) > 1 and len(cut[0]) >= 8 else cut[-1].strip()
    return s


def main():
    rng = random.Random(NS.seed)
    docs, skipped = [], {"short": 0, "lang": 0, "few_sent": 0}
    with open(SRC, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
                text = d.get("text", "")
            except json.JSONDecodeError:
                continue
            en = d.get("language_id_whole_page_fasttext", {}).get("en", 0.0)
            if not (500 <= len(text) <= 20000):
                skipped["short"] += 1
                continue
            if en <= 0.8:
                skipped["lang"] += 1
                continue
            sents = sentences(text)
            if len(sents) < 6:
                skipped["few_sent"] += 1
                continue
            docs.append({"url": d.get("url", ""), "text": text, "sents": sents})
    print(f"  eligible documents: {len(docs)}  skipped={skipped}")

    train, val, stats = [], [], {"rows_dropped_dup": 0, "rows_dropped_budget": 0,
                                 "docs_used": 0}
    for di, doc in enumerate(docs):
        if len(train) >= MAX_ROWS_TRAIN and di % VAL_EVERY != 0:
            continue
        sents, text = doc["sents"], doc["text"]
        made = 0
        # evenly spaced boundaries: at most 2 rows/doc, no early-doc bias
        idxs = sorted(set([len(sents) * k // 3 for k in (1, 2)]))
        for bi in idxs:
            if bi < 1 or bi >= len(sents) or made >= 2:
                continue
            before = " ".join(sents[:bi])
            ctx = clean(before[-CTX_BYTES:].strip())
            # byte budget, enforced on bytes because the model truncates bytes
            cb = ctx.encode("utf-8")
            if len(cb) > CTX_BYTES:
                cb = cb[-CTX_BYTES:]
                ctx = cb.decode("utf-8", errors="ignore").strip()
            if len(ctx.encode("utf-8")) > CTX_BYTES or len(ctx) < 20:
                continue
            correct = clean(head_span(sents[bi]))
            if len(correct.encode("utf-8")) > OPT_BYTES or len(correct) < 3:
                continue
            pool = [clean(head_span(s)) for j, s in enumerate(sents)
                    if j != bi and abs(j - bi) > 0]
            pool = [p for p in dict.fromkeys(pool)
                    if p != correct and 3 <= len(p.encode("utf-8")) <= OPT_BYTES]
            if len(pool) < 3:
                stats["rows_dropped_dup"] += 1
                continue
            distract = rng.sample(pool, 3)
            opts = [correct] + distract
            rng.shuffle(opts)
            offset = text.find(sents[bi])
            row = {"context": ctx, "options": opts, "label": opts.index(correct),
                   "source": {"url": doc["url"], "offset": offset}}
            (val if di % VAL_EVERY == 0 else train).append(row)
            made += 1
        if made:
            stats["docs_used"] += 1

    tpath = OUT / "owt_cloze_train.jsonl"
    vpath = OUT / "owt_cloze_val.jsonl"
    tpath.write_text("\n".join(json.dumps(r, sort_keys=True) for r in train) + "\n")
    vpath.write_text("\n".join(json.dumps(r, sort_keys=True) for r in val) + "\n")
    manifest = {
        "seed": SEED, "source_slice": str(SRC),
        "rules": ["document boundaries never crossed",
                  "straight quotes -> apostrophes, backslashes dropped (C loader "
                  "scans quotes naively; deterministic substitution)",
                  "distractors are same-document spans",
                  "byte budgets: context<=160B, options<=30B (model truncates C=192/O=32)",
                  "doc filter: en>0.8, 500..20000 chars, >=6 sentences",
                  "train/val split by document (every 10th doc to val)",
                  "max 2 rows/doc at evenly spaced boundaries"],
        "n_train": len(train), "n_val": len(val), "stats": stats,
        "skipped_docs": skipped,
        "limitation": ("same-document distractors can coincidentally fit; unmeasured "
                       "label noise biases reported accuracy DOWN, never up"),
    }
    body = tpath.read_bytes() + vpath.read_bytes()
    manifest["corpus_sha256"] = hashlib.sha256(body).hexdigest()
    (OUT / "owt_cloze_manifest.json").write_text(
        json.dumps(manifest, indent=1, sort_keys=True))
    print(f"  train {len(train)}  val {len(val)}  docs_used {stats['docs_used']}")
    print(f"  dropped dup/budget rows: {stats}")
    print(f"  corpus sha256 {manifest['corpus_sha256'][:16]}")
    print(f"  wrote {tpath.name}, {vpath.name}, owt_cloze_manifest.json")


if __name__ == "__main__":
    sys.exit(main())

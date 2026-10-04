# Saphira Candor product build. Native Saphira Linux / musl / x86-64-v3.
# Static binaries: both products must transfer as single artefacts.
# If static linking fails, the build MUST fail loudly -- never fall back
# silently, per product definition.
#
# Object discipline (why two lists exist): cd_train.o carries the original
# cd_load/cd_ask alongside the trainer, and cd_persist.o/cd_ask.o carry the
# extracted ask-only copies. Linking both copies into one binary is a
# duplicate-definition error, so each binary links exactly one copy:
#   candor  = full engine WITH trainer (uses cd_train.o originals)
#   scandor = ask-only engine (uses the extracted copies; no train objects)
CC ?= cc
CFLAGS = -O2 -std=c11 -Wall -Wextra -Wpedantic -fno-fast-math -ffp-contract=off \
         -march=x86-64-v3 -Isrc -Ibuild
LDFLAGS = -static
LDLIBS = -lm

ENGINE_ASK = build/cd_model.o build/cd_data.o build/cd_simd.o build/cd_fact.o \
             build/cd_persist.o build/cd_ask.o
ENGINE_FULL = build/cd_model.o build/cd_data.o build/cd_simd.o build/cd_fact.o \
              build/cd_train.o build/cd_bandit.o

# Pinned proven content. The build hash-checks these before baking; the commit
# identifies the engine AND its education/memory.
MODEL_SHA256 = a439b6962b9bd8e2b3800424a13f1726f662fac5a5029eb7c6875734e6eee5da
WITNESS_SHA256 = 8b62d00ad71daf68dc65078748c01a32a54f5629190482717494e5d8d0221b21

.PHONY: all candor scandor test install clean
all: candor scandor

build/%.o: src/%.c
	mkdir -p build
	$(CC) $(CFLAGS) -c $< -o $@

# ask_main.c includes generated build/about.h: order matters, state it.
build/ask_main.o: src/ask_main.c build/about.h
	mkdir -p build
	$(CC) $(CFLAGS) -c $< -o $@

candor: ./candor
scandor: ./scandor

./candor: $(ENGINE_FULL) build/candor_main.o
	$(CC) $(CFLAGS) -o $@ $(ENGINE_FULL) build/candor_main.o \
	    $(LDFLAGS) $(LDLIBS)
	ls -la $@

build/model_blob.c build/witness_blob.c build/about.h build/about.txt: models/model.q8 witnesses/witness.jsonl tools/bake.py
	mkdir -p build
	python3 tools/bake.py --model models/model.q8 --witness witnesses/witness.jsonl \
	    --out-dir build --expect-model-sha256 $(MODEL_SHA256) \
	    --expect-witness-sha256 $(WITNESS_SHA256)

build/model_blob.o build/witness_blob.o: build/model_blob.c build/witness_blob.c
	$(CC) $(CFLAGS) -c build/model_blob.c -o build/model_blob.o
	$(CC) $(CFLAGS) -c build/witness_blob.c -o build/witness_blob.o

./scandor: $(ENGINE_ASK) build/ask_main.o build/model_blob.o build/witness_blob.o
	$(CC) $(CFLAGS) -o $@ $(ENGINE_ASK) build/ask_main.o \
	    build/model_blob.o build/witness_blob.o $(LDFLAGS) $(LDLIBS)
	ls -la $@

test: all
	sh tests/runtime.sh

PREFIX ?= /usr/local
install: all
	mkdir -p $(DESTDIR)$(PREFIX)/bin $(DESTDIR)$(PREFIX)/share/man/man1
	cp candor scandor $(DESTDIR)$(PREFIX)/bin/
	cp docs/man/man1/candor.1 docs/man/man1/scandor.1 $(DESTDIR)$(PREFIX)/share/man/man1/

clean:
	rm -rf build candor scandor

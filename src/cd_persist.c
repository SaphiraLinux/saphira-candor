/* cd_persist.c -- model loading ONLY. No training, no saving, no bandit.
 *
 * Extracted verbatim from cd_train.c (save_tensor_list, the q3/q8/qn block
 * codecs and readers, cd_load): the loader every shipped model depends on,
 * without the 200KB of trainer it used to be linked against. cd_load is split
 * into cd_load_fp (parse, shared) so embedded bytes load through fmemopen and
 * files through fopen with zero format fork: one parser, every source.
 */
#define _POSIX_C_SOURCE 200809L
#include "candor.h"
#include "cd_internal.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define CD_MAGIC 0x31524443u  /* "CDR1" in ASCII order */

/* Ordered tensor list for persistence. cd_save (v1 floats) and cd_save_q3 (v2 packed
 * 3-bit) BOTH walk this list, so the two containers can never disagree on tensor order.
 * A v2 file converted from a v1 file must dequantize to the same sequence, and this single
 * list is what makes that checkable. */
#define CD_SAVE_MAX_TENSORS 24
#define CD_Q3_BLOCK 32
static int save_tensor_list(const struct cd_model *m, const float **p, size_t *n) {
    int k = 0;
    p[k] = m->emb; n[k++] = m->n_emb;
    p[k] = m->pos; n[k++] = m->n_pos;
    p[k] = m->lncw; n[k++] = m->d.W; p[k] = m->lncb; n[k++] = m->d.W;
    p[k] = m->lonw; n[k++] = m->d.W; p[k] = m->lonb; n[k++] = m->d.W;
    p[k] = m->Wq; n[k++] = m->n_qkv;
    p[k] = m->Wk; n[k++] = m->n_qkv;
    p[k] = m->Wv; n[k++] = m->n_qkv;
    if (m->arch == CD_ARCH_INTERACT) {
        p[k] = m->Wo; n[k++] = m->n_qkv;
        p[k] = m->gc; n[k++] = m->d.W; p[k] = m->gb; n[k++] = m->d.W;
    }
    if (cd_arch_is_twohead((int)m->arch)) { p[k] = m->Wq2; n[k++] = m->n_qkv; }
    if (m->Wk2) { p[k] = m->Wk2; n[k++] = m->n_qkv; }
    if (m->Vv2) { p[k] = m->Vv2; n[k++] = m->n_qkv; }
    if (m->arch == CD_ARCH_XBID) {
        p[k] = m->n1g; n[k++] = m->d.R; p[k] = m->n1b; n[k++] = m->d.R;
        p[k] = m->n2g; n[k++] = m->d.R; p[k] = m->n2b; n[k++] = m->d.R;
        p[k] = m->n3g; n[k++] = m->d.R; p[k] = m->n3b; n[k++] = m->d.R;
    }
    return k;
}


int cd_q3_block_encode(const float *v, size_t n, float *mn_out, float *scale_out,
                       unsigned char *packed12) {
    float mn, mx, scale;
    size_t m;
    if (n < 1 || n > CD_Q3_BLOCK) return -1;
    mn = mx = v[0];
    for (m = 1; m < n; m++) {
        if (v[m] < mn) mn = v[m];
        if (v[m] > mx) mx = v[m];
    }
    scale = (mx > mn) ? (mx - mn) / 7.0f : 0.0f;
    memset(packed12, 0, (CD_Q3_BLOCK * 3) / 8);
    for (m = 0; m < CD_Q3_BLOCK; m++) {
        unsigned idx = 0, bit;
        if (m < n && scale > 0.0f) {
            float q = (v[m] - mn) / scale;
            idx = (unsigned)(q + 0.5f);
            if (idx > 7) idx = 7;
        }
        bit = (unsigned)(m * 3);
        packed12[bit / 8] = (unsigned char)(packed12[bit / 8] | (idx << (bit % 8)));
        if (bit % 8 > 5)
            packed12[bit / 8 + 1] = (unsigned char)(packed12[bit / 8 + 1] |
                                                    (idx >> (8 - bit % 8)));
    }
    *mn_out = mn;
    *scale_out = scale;
    return 0;
}


int cd_q3_block_decode(float mn, float scale, const unsigned char *packed12,
                       float *v, size_t n) {
    size_t m;
    if (n < 1 || n > CD_Q3_BLOCK || !packed12 || !v) return -1;
    for (m = 0; m < n; m++) {
        unsigned bit = (unsigned)(m * 3), idx;
        idx = (unsigned)(packed12[bit / 8] >> (bit % 8)) & 7u;
        if (bit % 8 > 5) {
            /* Spill bits only: bit%8==6 needs 1 bit from the next byte, ==7 needs
             * 2. OR-ing the whole shifted byte (the original bug) drags in up to 7
             * unrelated bits and the index explodes -- caught by tests/test_quant.c
             * at every index congruent to 2 or 5 mod 8, which is exactly the spill
             * set. Masking is not optional here. */
            unsigned need = bit % 8 - 5u;
            idx |= (((unsigned)packed12[bit / 8 + 1]) & ((1u << need) - 1u))
                   << (8u - bit % 8);
        }
        v[m] = mn + (float)idx * scale;
    }
    return 0;
}


static int q3_read(FILE *f, float *v, size_t n) {
    size_t i;
    for (i = 0; i < n; i += CD_Q3_BLOCK) {
        float mn, scale;
        unsigned char packed[(CD_Q3_BLOCK * 3) / 8];
        size_t m = (n - i < CD_Q3_BLOCK) ? n - i : CD_Q3_BLOCK;
        if (fread(&mn, sizeof mn, 1, f) != 1 ||
            fread(&scale, sizeof scale, 1, f) != 1 ||
            fread(packed, sizeof packed, 1, f) != 1)
            return -1;
        if (cd_q3_block_decode(mn, scale, packed, v + i, m) != 0) return -1;
    }
    return 0;
}


int cd_q8_block_encode(const float *v, size_t n, float *scale_out, signed char *out32) {
    float mx = 0.0f, scale;
    size_t m;
    if (n < 1 || n > CD_Q3_BLOCK || !out32) return -1;
    for (m = 0; m < n; m++) {
        float a = v[m] < 0.0f ? -v[m] : v[m];
        if (a > mx) mx = a;
    }
    scale = (mx > 0.0f) ? mx / 127.0f : 0.0f;
    for (m = 0; m < CD_Q3_BLOCK; m++) {
        int q = 0;
        if (m < n && scale > 0.0f) {
            float f = v[m] / scale;
            q = (int)(f >= 0.0f ? f + 0.5f : f - 0.5f);
            if (q > 127) q = 127;
            if (q < -127) q = -127;
        }
        out32[m] = (signed char)q;
    }
    *scale_out = scale;
    return 0;
}


int cd_q8_block_decode(float scale, const signed char *in32, float *v, size_t n) {
    size_t m;
    if (n < 1 || n > CD_Q3_BLOCK || !in32 || !v) return -1;
    for (m = 0; m < n; m++) v[m] = (float)in32[m] * scale;
    return 0;
}


static int q8_read(FILE *f, float *v, size_t n) {
    size_t i;
    for (i = 0; i < n; i += CD_Q3_BLOCK) {
        float scale;
        signed char vals[CD_Q3_BLOCK];
        size_t m = (n - i < CD_Q3_BLOCK) ? n - i : CD_Q3_BLOCK;
        if (fread(&scale, sizeof scale, 1, f) != 1 ||
            fread(vals, sizeof vals, 1, f) != 1)
            return -1;
        if (cd_q8_block_decode(scale, vals, v + i, m) != 0) return -1;
    }
    return 0;
}


int cd_qn_block_encode(const float *v, size_t n, int bits, float *mn_out,
                       float *scale_out, unsigned char *packed) {
    float mn, mx, scale;
    size_t m;
    unsigned levels;
    if (!v || !mn_out || !scale_out || !packed) return -1;
    if (bits < 1 || bits > 8 || n < 1 || n > CD_Q3_BLOCK) return -1;
    levels = (unsigned)(1u << bits);
    mn = mx = v[0];
    for (m = 1; m < n; m++) {
        if (v[m] < mn) mn = v[m];
        if (v[m] > mx) mx = v[m];
    }
    scale = (mx > mn) ? (mx - mn) / (float)(levels - 1u) : 0.0f;
    memset(packed, 0, (CD_Q3_BLOCK * 8 + 7) / 8);
    for (m = 0; m < CD_Q3_BLOCK; m++) {
        unsigned idx = 0, bit, span;
        if (m < n && scale > 0.0f) {
            float q = (v[m] - mn) / scale;
            idx = (unsigned)(q + 0.5f);
            if (idx > levels - 1u) idx = levels - 1u;
        }
        bit = (unsigned)(m * (size_t)bits);
        for (span = 0; span < (unsigned)bits; span++) {
            if (idx & (1u << span))
                packed[(bit + span) / 8] = (unsigned char)
                    (packed[(bit + span) / 8] | (1u << ((bit + span) % 8)));
        }
    }
    *mn_out = mn;
    *scale_out = scale;
    return 0;
}


int cd_qn_block_decode(float mn, float scale, const unsigned char *packed,
                       int bits, float *v, size_t n) {
    size_t m;
    unsigned levels;
    if (!packed || !v) return -1;
    if (bits < 1 || bits > 8 || n < 1 || n > CD_Q3_BLOCK) return -1;
    levels = (unsigned)(1u << bits);
    for (m = 0; m < n; m++) {
        unsigned bit = (unsigned)(m * (size_t)bits), idx = 0, span;
        for (span = 0; span < (unsigned)bits; span++) {
            if (packed[(bit + span) / 8] & (1u << ((bit + span) % 8)))
                idx |= 1u << span;
        }
        if (idx > levels - 1u) return -1;
        v[m] = mn + (float)idx * scale;
    }
    return 0;
}


static int qn_read(FILE *f, float *v, size_t n, int bits) {
    size_t i;
    size_t bytes = (CD_Q3_BLOCK * (size_t)bits + 7) / 8;
    for (i = 0; i < n; i += CD_Q3_BLOCK) {
        float mn, scale;
        unsigned char packed[(CD_Q3_BLOCK * 8 + 7) / 8];
        size_t m = (n - i < CD_Q3_BLOCK) ? n - i : CD_Q3_BLOCK;
        if (fread(&mn, sizeof mn, 1, f) != 1 ||
            fread(&scale, sizeof scale, 1, f) != 1 ||
            fread(packed, 1, bytes, f) != bytes)
            return -1;
        if (cd_qn_block_decode(mn, scale, packed, bits, v + i, m) != 0) return -1;
    }
    return 0;
}


static struct cd_model *cd_load_fp(FILE *f, const char *path, char *err, size_t err_len) {
    uint32_t hdr[10];
    struct cd_model *m;
    struct cd_dims d;
    enum cd_arch arch;
    if (!f) { snprintf(err, err_len, "cannot read %s", path); return NULL; }
    if (fread(hdr, sizeof hdr, 1, f) != 1 || hdr[0] != CD_MAGIC) {
        snprintf(err, err_len, "bad model header in %s", path); fclose(f); return NULL;
    }
    /* Fail closed on out-of-range header fields. v0 validated its ranges and this
     * container did not; a corrupt dims field previously sailed into unchecked mallocs.
     * Bounds follow the architecture family (v0 used W,R <= 4096 and C,O <= 8192). */
    if (hdr[7] != 1u && hdr[7] != 2u && hdr[7] != 3u && hdr[7] != 4u && hdr[7] != 5u) {
        snprintf(err, err_len, "unsupported model version %u in %s", hdr[7], path);
        fclose(f); return NULL;
    }
    if (hdr[1] < 1 || hdr[1] > 4096 || hdr[2] < 1 || hdr[2] > 4096 ||
        hdr[3] < 1 || hdr[3] > 8192 || hdr[4] < 1 || hdr[4] > 8192 ||
        hdr[5] < 1 || hdr[5] > 1024) {
        snprintf(err, err_len, "implausible model dims in %s", path);
        fclose(f); return NULL;
    }
    switch (hdr[6]) {
    case 0: case 1: case 2: case 3: case 4: case 5: case 6: case 7:
    case 9: case 10: case 11:
        break;
    default:
        snprintf(err, err_len, "unknown architecture %u in %s", hdr[6], path);
        fclose(f); return NULL;
    }
    d.W = hdr[1]; d.R = hdr[2]; d.C = hdr[3]; d.O = hdr[4];
    d.n_opt = (int)hdr[5];
    arch = (enum cd_arch)hdr[6];
    m = calloc(1, sizeof *m);
    if (!m) { fclose(f); return NULL; }
    m->d = d; m->arch = arch;
    m->n_emb = 257ull * d.W; m->n_pos = (size_t)d.C * d.W; m->n_qkv = (size_t)d.R * d.W;
    m->emb = malloc(m->n_emb * sizeof(float));
    m->pos = malloc(m->n_pos * sizeof(float));
    m->lncw = malloc(d.W * sizeof(float)); m->lncb = malloc(d.W * sizeof(float));
    m->lonw = malloc(d.W * sizeof(float)); m->lonb = malloc(d.W * sizeof(float));
    m->Wq = malloc(m->n_qkv * sizeof(float));
    m->Wk = malloc(m->n_qkv * sizeof(float));
    m->Wv = malloc(m->n_qkv * sizeof(float));
    if (arch == CD_ARCH_INTERACT) {
        m->Wo = malloc(m->n_qkv * sizeof(float));
        m->gc = malloc(d.W * sizeof(float)); m->gb = malloc(d.W * sizeof(float));
    }
    {   /* derived transposes, mirroring the freshly loaded weights */
        int slot;
        for (slot = 0; slot < CD_W_COUNT; slot++) {
            const size_t n = (size_t)d.R * (size_t)d.W;
            m->Wt[slot] = calloc(n, sizeof(double));
            if (!m->Wt[slot]) { snprintf(err, err_len, "out of memory (transpose)"); fclose(f); cd_model_free(m); return NULL; }
        }
    }
    if (cd_arch_is_twohead(arch)) m->Wq2 = malloc(m->n_qkv * sizeof(float));
    if (arch == CD_ARCH_TW_SK || arch == CD_ARCH_TW_SKV ||
        arch == CD_ARCH_SUM_K || arch == CD_ARCH_SUM_KV)
        m->Wk2 = malloc(m->n_qkv * sizeof(float));
    if (arch == CD_ARCH_TW_SV || arch == CD_ARCH_TW_SKV ||
        arch == CD_ARCH_SUM_V || arch == CD_ARCH_SUM_KV)
        m->Vv2 = malloc(m->n_qkv * sizeof(float));
    if (arch == CD_ARCH_XBID) {
        m->n1g = malloc(d.R * sizeof(float)); m->n1b = malloc(d.R * sizeof(float));
        m->n2g = malloc(d.R * sizeof(float)); m->n2b = malloc(d.R * sizeof(float));
        m->n3g = malloc(d.R * sizeof(float)); m->n3b = malloc(d.R * sizeof(float));
    }
    if (hdr[7] == 4u || hdr[7] == 5u) {
        const float *p[CD_SAVE_MAX_TENSORS];
        size_t n[CD_SAVE_MAX_TENSORS];
        int k = save_tensor_list(m, p, n), t;
        int bits = (hdr[7] == 4u) ? 4 : 2;
        for (t = 0; t < k; t++) {
            if (qn_read(f, (float *)p[t], n[t], bits) != 0) {
                snprintf(err, err_len, "truncated q%d tensor %d in %s", bits, t, path);
                fclose(f); cd_model_free(m); return NULL;
            }
        }
    } else
    if (hdr[7] == 3u) {
        const float *p[CD_SAVE_MAX_TENSORS];
        size_t n[CD_SAVE_MAX_TENSORS];
        int k = save_tensor_list(m, p, n), t;
        for (t = 0; t < k; t++) {
            if (q8_read(f, (float *)p[t], n[t]) != 0) {
                snprintf(err, err_len, "truncated q8 tensor %d in %s", t, path);
                fclose(f); cd_model_free(m); return NULL;
            }
        }
    } else
    if (hdr[7] == 2u) {
        /* v2 payload: block 3-bit tensors in save_tensor_list order, promoted back to
         * float32 on load. Compute precision is therefore identical to v1; only storage
         * was quantized. A v2 file that fails here fails closed like any truncation. */
        const float *p[CD_SAVE_MAX_TENSORS];
        size_t n[CD_SAVE_MAX_TENSORS];
        int k = save_tensor_list(m, p, n), t;
        for (t = 0; t < k; t++) {
            if (q3_read(f, (float *)p[t], n[t]) != 0) {
                snprintf(err, err_len, "truncated q3 tensor %d in %s", t, path);
                fclose(f); cd_model_free(m); return NULL;
            }
        }
    } else
    if (fread(m->emb, sizeof(float), m->n_emb, f) != (size_t)m->n_emb ||
        fread(m->pos, sizeof(float), m->n_pos, f) != (size_t)m->n_pos ||
        fread(m->lncw, sizeof(float), d.W, f) != (size_t)d.W ||
        fread(m->lncb, sizeof(float), d.W, f) != (size_t)d.W ||
        fread(m->lonw, sizeof(float), d.W, f) != (size_t)d.W ||
        fread(m->lonb, sizeof(float), d.W, f) != (size_t)d.W ||
        fread(m->Wq, sizeof(float), m->n_qkv, f) != (size_t)m->n_qkv ||
        fread(m->Wk, sizeof(float), m->n_qkv, f) != (size_t)m->n_qkv ||
        fread(m->Wv, sizeof(float), m->n_qkv, f) != m->n_qkv) {
        snprintf(err, err_len, "truncated model %s", path); fclose(f);
        cd_model_free(m); return NULL;
    }
    if (arch == CD_ARCH_INTERACT) {
        if (fread(m->Wo, sizeof(float), m->n_qkv, f) != (size_t)m->n_qkv ||
            fread(m->gc, sizeof(float), d.W, f) != (size_t)d.W ||
            fread(m->gb, sizeof(float), d.W, f) != (size_t)d.W) {
            snprintf(err, err_len, "truncated interaction tensors in %s", path);
            fclose(f); cd_model_free(m); return NULL;
        }
    }
    /* Per-form maps, in exactly the order model_save wrote them. */
    if (cd_arch_is_twohead(arch)) {
        if (fread(m->Wq2, sizeof(float), m->n_qkv, f) != (size_t)m->n_qkv) {
            snprintf(err, err_len, "truncated twohead query map in %s", path);
            fclose(f); cd_model_free(m); return NULL;
        }
    }
    if (m->Wk2 && fread(m->Wk2, sizeof(float), m->n_qkv, f) != (size_t)m->n_qkv) {
        snprintf(err, err_len, "truncated context key map in %s", path);
        fclose(f); cd_model_free(m); return NULL;
    }
    if (m->Vv2 && fread(m->Vv2, sizeof(float), m->n_qkv, f) != (size_t)m->n_qkv) {
        snprintf(err, err_len, "truncated context value map in %s", path);
        fclose(f); cd_model_free(m); return NULL;
    }
    if (arch == CD_ARCH_XBID) {
        /* Without this the six norm pointers stay NULL and the first forward
         * dereferences them. T5 in tests/test_backward.c covers exactly this. */
        if (fread(m->n1g, sizeof(float), d.R, f) != (size_t)d.R ||
            fread(m->n1b, sizeof(float), d.R, f) != (size_t)d.R ||
            fread(m->n2g, sizeof(float), d.R, f) != (size_t)d.R ||
            fread(m->n2b, sizeof(float), d.R, f) != (size_t)d.R ||
            fread(m->n3g, sizeof(float), d.R, f) != (size_t)d.R ||
            fread(m->n3b, sizeof(float), d.R, f) != (size_t)d.R) {
            snprintf(err, err_len, "truncated xbid norm tensors in %s", path);
            fclose(f); cd_model_free(m); return NULL;
        }
    }
    /* Every weight has now been read, so the derived transposes can be built.
     * Without this the exact AVX2 path would read all-zero weights and the
     * model would score exactly ln 2, i.e. uniform. */
    cd_model_refresh_transposes(m);

    fclose(f);
    return m;
}

struct cd_model *cd_load(const char *path, char *err, size_t err_len) {
    FILE *f = fopen(path, "rb");
    if (!f) { snprintf(err, err_len, "cannot read %s", path); return NULL; }
    return cd_load_fp(f, path, err, err_len);
}

struct cd_model *cd_load_mem(const unsigned char *data, size_t len, char *err, size_t err_len) {
    FILE *f;
    if (!data || !len) { snprintf(err, err_len, "empty embedded model"); return NULL; }
    f = fmemopen((void *)data, len, "rb");
    if (!f) { snprintf(err, err_len, "cannot open embedded model"); return NULL; }
    return cd_load_fp(f, "(embedded)", err, err_len);
}

#define _POSIX_C_SOURCE 200809L
/* cd_train.c — reverse-mode backward, the trainer, and model I/O.
 *
 * LEGACY mode reproduces the upstream trainer exactly:
 *   - loss = -log(max(p[label], 1e-12))                       (train.c:272-274)
 *   - dlogits = probs - one_hot, UNSCALED, caller scales      (train.c:291-295)
 *   - grad_scale(1/blen), grad_clip(1.0) hardcoded            (train.c:605-606)
 *   - AdamW, betas 0.9/0.999, eps OUTSIDE the sqrt, decoupled
 *     decay on EVERY parameter                                  (train.c:133-149)
 *   - embedding row 0 skipped and re-zeroed after every update  (train.c:155,165)
 *   - per-epoch shuffle from a HARDCODED constant, so --seed does
 *     not affect data order                                     (train.c:580-586)
 *   - best-validation-NLL checkpointing with a full shadow copy
 *
 * CORRECT mode changes exactly the things named in the switches, each of
 * which is reported. Nothing is changed silently.
 */
#include "candor.h"
#include "cd_internal.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

/* The single definition of the training loss. cd_forward cannot compute it
 * because it has no label, so every caller that needs a scalar loss without a
 * backward pass (notably the finite-difference probe) goes through here. */
static double cd_loss_of(const struct cd_cache *c, int label, int no_clamp) {
    float p = c->probs[label];
    if (no_clamp) return -log(p > 0.0f ? p : 1e-30f);
    if (p < 1e-12f) p = 1e-12f;   /* the upstream clamp */
    return -(double)logf(p);
}

/* ---- gradient buffer -------------------------------------------------- */

struct cd_grad;
static void eval_dataset_head(const struct cd_model *m, const struct cd_dataset *ds,
                              size_t from, size_t to, struct cd_cache *c,
                              struct cd_epoch *out, int no_clamp, int head);
static double ece_dataset_head(const struct cd_model *m, const struct cd_dataset *ds,
                               struct cd_cache *c, int head,
                               struct cd_calib *out);
static void grad_zero(struct cd_grad *g, uint32_t W, uint32_t R);
static void grad_free(struct cd_grad *g);

struct cd_grad {
    float *emb, *pos, *lncw, *lncb, *lonw, *lonb, *Wq, *Wk, *Wv, *Wo, *gc, *gb;
    float *Wq2, *Wk2, *Vv2;
    float *n1g, *n1b, *n2g, *n2b, *n3g, *n3b;
    size_t n_emb, n_pos, n_qkv;
};

static void grad_alloc(struct cd_grad *g, const struct cd_model *m) {
    size_t W = m->d.W;
    uint32_t R = m->d.R;
    memset(g, 0, sizeof *g);
    g->n_emb = m->n_emb; g->n_pos = m->n_pos; g->n_qkv = m->n_qkv;
    g->emb = calloc(g->n_emb, sizeof(float));
    g->pos = calloc(g->n_pos, sizeof(float));
    g->lncw = calloc(W, sizeof(float)); g->lncb = calloc(W, sizeof(float));
    g->lonw = calloc(W, sizeof(float)); g->lonb = calloc(W, sizeof(float));
    g->Wq = calloc(g->n_qkv, sizeof(float));
    g->Wk = calloc(g->n_qkv, sizeof(float));
    g->Wv = calloc(g->n_qkv, sizeof(float));
    if (m->arch == CD_ARCH_INTERACT) {
        g->Wo = calloc(g->n_qkv, sizeof(float));
        g->gc = calloc(W, sizeof(float)); g->gb = calloc(W, sizeof(float));
    }
    if (cd_arch_is_twohead((int)m->arch)) {
        g->Wq2 = calloc(g->n_qkv, sizeof(float));
        if (!g->Wq2) return;
    }
    if (m->Wk2) g->Wk2 = calloc(g->n_qkv, sizeof(float));
    if (m->Vv2) g->Vv2 = calloc(g->n_qkv, sizeof(float));
    if (m->arch == CD_ARCH_XBID) {
        g->n1g = calloc(R, sizeof(float)); g->n1b = calloc(R, sizeof(float));
        g->n2g = calloc(R, sizeof(float)); g->n2b = calloc(R, sizeof(float));
        g->n3g = calloc(R, sizeof(float)); g->n3b = calloc(R, sizeof(float));
    }
}

static void grad_zero(struct cd_grad *g, uint32_t W, uint32_t R) {
    memset(g->emb, 0, g->n_emb * sizeof(float));
    memset(g->pos, 0, g->n_pos * sizeof(float));
    memset(g->lncw, 0, W * sizeof(float)); memset(g->lncb, 0, W * sizeof(float));
    memset(g->lonw, 0, W * sizeof(float)); memset(g->lonb, 0, W * sizeof(float));
    memset(g->Wq, 0, g->n_qkv * sizeof(float));
    memset(g->Wk, 0, g->n_qkv * sizeof(float));
    memset(g->Wv, 0, g->n_qkv * sizeof(float));
    if (g->Wo) {
        memset(g->Wo, 0, g->n_qkv * sizeof(float));
        memset(g->gc, 0, W * sizeof(float));
        memset(g->gb, 0, W * sizeof(float));
    }
    if (g->Wq2) memset(g->Wq2, 0, g->n_qkv * sizeof(float));
    if (g->Wk2) memset(g->Wk2, 0, g->n_qkv * sizeof(float));
    if (g->Vv2) memset(g->Vv2, 0, g->n_qkv * sizeof(float));
    if (g->Wk2) memset(g->Wk2, 0, g->n_qkv * sizeof(float));
    if (g->Vv2) memset(g->Vv2, 0, g->n_qkv * sizeof(float));
    if (g->n1g) {
        /* six SEPARATE R-float allocations, not one contiguous block: zeroing
         * 6*R floats from n1g overran the heap by 5x. */
        float *v[6] = {g->n1g, g->n1b, g->n2g, g->n2b, g->n3g, g->n3b};
        int t;
        for (t = 0; t < 6; t++) memset(v[t], 0, (size_t)R * sizeof(float));
    }
}

void cd_test_grad_norm_ptr(struct cd_grad *g, int idx, float **out) {
    float *v[6] = {g->n1g, g->n1b, g->n2g, g->n2b, g->n3g, g->n3b};
    *out = (idx >= 0 && idx < 6) ? v[idx] : NULL;
}

size_t cd_grad_trainable(const struct cd_grad *g, const float ***out) {
    const float **v = calloc(24, sizeof *v);
    size_t n = 0;
    v[n++] = g->emb; v[n++] = g->pos;
    v[n++] = g->lncw; v[n++] = g->lncb; v[n++] = g->lonw; v[n++] = g->lonb;
    v[n++] = g->Wq; v[n++] = g->Wk; v[n++] = g->Wv;
    if (g->Wq2) v[n++] = g->Wq2;
    if (g->Wk2) v[n++] = g->Wk2;
    if (g->Vv2) v[n++] = g->Vv2;
    if (g->Wo)  v[n++] = g->Wo;
    if (g->gc)  v[n++] = g->gc;
    if (g->gb)  v[n++] = g->gb;
    if (g->n1g) v[n++] = g->n1g;
    if (g->n1b) v[n++] = g->n1b;
    if (g->n2g) v[n++] = g->n2g;
    if (g->n2b) v[n++] = g->n2b;
    if (g->n3g) v[n++] = g->n3g;
    if (g->n3b) v[n++] = g->n3b;
    v[n] = NULL;
    *out = v;
    return n;
}

/* The tensors the optimiser step ACTUALLY updates. This list is the point of
 * the whole exercise: it is written by hand, next to the adam_update calls,
 * so a tensor added to the model or to the gradient without a matching entry
 * here shows up as a set difference and fails the suite. */
size_t cd_optimizer_trainable(const struct cd_model *m, const float ***out) {
    const float **v = calloc(24, sizeof *v);
    size_t n = 0;
    v[n++] = m->emb; v[n++] = m->pos;
    v[n++] = m->lncw; v[n++] = m->lncb; v[n++] = m->lonw; v[n++] = m->lonb;
    v[n++] = m->Wq; v[n++] = m->Wk; v[n++] = m->Wv;
    if (m->Wq2) v[n++] = m->Wq2;
    if (m->Wk2) v[n++] = m->Wk2;
    if (m->Vv2) v[n++] = m->Vv2;
    if (m->Wo)  v[n++] = m->Wo;
    if (m->gc)  v[n++] = m->gc;
    if (m->gb)  v[n++] = m->gb;
    if (m->n1g) v[n++] = m->n1g;
    if (m->n1b) v[n++] = m->n1b;
    if (m->n2g) v[n++] = m->n2g;
    if (m->n2b) v[n++] = m->n2b;
    if (m->n3g) v[n++] = m->n3g;
    if (m->n3b) v[n++] = m->n3b;
    v[n] = NULL;
    *out = v;
    return n;
}

static void grad_free(struct cd_grad *g) {
    if (!g) return;
    free(g->emb); free(g->pos);
    free(g->lncw); free(g->lncb); free(g->lonw); free(g->lonb);
    free(g->Wq); free(g->Wk); free(g->Wv);
    free(g->Wo); free(g->gc); free(g->gb); free(g->Wq2);
    free(g->Wk2); free(g->Vv2);
    free(g->n1g); free(g->n1b); free(g->n2g); free(g->n2b); free(g->n3g); free(g->n3b);
    memset(g, 0, sizeof *g);
}

/* Global L2 norm over every parameter, transcribed from train.c:78-89. */
static double grad_global_norm(const struct cd_model *m, const struct cd_grad *g) {
    size_t i, W = m->d.W;
    double sum = 0.0;
    for (i = 0; i < g->n_emb; i++) { double v = g->emb[i]; sum += v * v; }
    for (i = 0; i < g->n_pos; i++) { double v = g->pos[i]; sum += v * v; }
    for (i = 0; i < W; i++) {
        double v = g->lncw[i]; sum += v * v;
        v = g->lncb[i]; sum += v * v;
        v = g->lonw[i]; sum += v * v;
        v = g->lonb[i]; sum += v * v;
    }
    for (i = 0; i < g->n_qkv; i++) {
        double v = g->Wq[i]; sum += v * v;
        v = g->Wk[i]; sum += v * v;
        v = g->Wv[i]; sum += v * v;
    }
    if (m->arch == CD_ARCH_INTERACT) {
        for (i = 0; i < g->n_qkv; i++) { double v = g->Wo[i]; sum += v * v; }
        for (i = 0; i < W; i++) {
            double v = g->gc[i]; sum += v * v;
            v = g->gb[i]; sum += v * v;
        }
    }
    if (g->Wq2) for (i = 0; i < g->n_qkv; i++) { double v = g->Wq2[i]; sum += v * v; }
    if (g->Wk2) for (i = 0; i < g->n_qkv; i++) { double v = g->Wk2[i]; sum += v * v; }
    if (g->Vv2) for (i = 0; i < g->n_qkv; i++) { double v = g->Vv2[i]; sum += v * v; }
    if (m->arch == CD_ARCH_XBID) {
        const float *v[6] = {g->n1g, g->n1b, g->n2g, g->n2b, g->n3g, g->n3b};
        int t;
        for (t = 0; t < 6; t++)
            for (i = 0; i < (size_t)m->d.R; i++) { double x = v[t][i]; sum += x * x; }
    }
    return sqrt(sum);
}

/* ---- LN backward ------------------------------------------------------
 * y = (x - mean) * inv * g + b,  inv = 1/sqrt(var + eps)
 * dx = inv/W * ( W*dxhat - sum(dxhat) - xhat * sum(dxhat*xhat) ), dxhat = dy*g
 */
static void ln_backward(const float *x, const float *dy, const float *gam,
                        int rows, int width, float *dx, float *dgam,
                        float *dbias) {
    const double eps = 1e-5;
    int r, i;
    for (r = 0; r < rows; r++) {
        const float *xs = x + (size_t)r * width;
        const float *dys = dy + (size_t)r * width;
        float *dxs = dx + (size_t)r * width;
        double sum = 0.0, mean, var = 0.0, inv, s1 = 0.0, s2 = 0.0;
        for (i = 0; i < width; i++) sum += (double)xs[i];
        mean = sum / width;
        for (i = 0; i < width; i++) {
            double d = (double)xs[i] - mean;
            var += d * d;
        }
        var /= width;
        inv = 1.0 / sqrt(var + eps);
        for (i = 0; i < width; i++) {
            double xhat = ((double)xs[i] - mean) * inv;
            double dxhat = (double)dys[i] * (double)gam[i];
            s1 += dxhat;
            s2 += dxhat * xhat;
            if (dgam) dgam[i] += (float)(dxhat * xhat);
            if (dbias) dbias[i] += dys[i];
        }
        for (i = 0; i < width; i++) {
            double xhat = ((double)xs[i] - mean) * inv;
            double dxhat = (double)dys[i] * (double)gam[i];
            dxs[i] = (float)(inv / width *
                             (width * dxhat - s1 - xhat * s2));
        }
    }
}


/* ---- XBID backward ----------------------------------------------------
 *
 * Mirror of the residual-free forward in cd_model.c. Nothing is reused from
 * the baseline backward on purpose: the baseline path assumes logits depend on
 * option n alone, and running it against the XBID forward silently trains a
 * different function than the one being evaluated.
 */
static int rc_twohead_inv(const struct cd_model *m, const char *context,
                          const char *const *options, int n_options, int label,
                          const struct cd_cache *c, struct cd_grad *g);

static int xbid_backward(const struct cd_model *m, const char *context,
                         const char *const *options, int n_options,
                         const struct cd_cache *c, struct cd_grad *g,
                         const float *dlogits_in) {
    const uint32_t W = m->d.W, R = m->d.R, O = m->d.O;
    const size_t L = c->L;
    const int N = n_options;
    const float isq = 1.0f / sqrtf((float)R);
    int n, l, r;
    size_t i, k, w;

    float *dQ  = calloc((size_t)N * R, sizeof(float));
    float *dK  = calloc(L * R, sizeof(float));
    float *dV  = calloc(L * R, sizeof(float));
    float *dQ2 = calloc(L * R, sizeof(float));
    float *dh1 = calloc((size_t)N * R, sizeof(float));
    float *dh2 = calloc((size_t)N * R, sizeof(float));
    float *dg  = calloc(L * R, sizeof(float));
    float *da2 = calloc((size_t)N * L, sizeof(float));
    float *ds3 = calloc((size_t)N * L, sizeof(float));
    float *db  = calloc(L * (size_t)N, sizeof(float));
    float *ds2 = calloc(L * (size_t)N, sizeof(float));
    float *da1 = calloc((size_t)N * L, sizeof(float));
    float *ds1 = calloc((size_t)N * L, sizeof(float));
    float *dc  = calloc((size_t)N * W, sizeof(float));
    float *dx  = calloc(L * W, sizeof(float));
    float *dcx = calloc((size_t)N * W, sizeof(float));
    float *dxx = calloc(L * W, sizeof(float));
    int rc = 0;
    if (!dQ||!dK||!dV||!dQ2||!dh1||!dh2||!dg||!da2||!ds3||!db||!ds2||!da1||!ds1
        ||!dc||!dx||!dcx||!dxx) { rc = -1; goto out; }

    /* 1. logits[n] = isq * (Q[n] . h2[n]). The Q term is the direct path from
     *    the option to the score and must be present; see cd_model.c. */
    for (n = 0; n < N; n++) {
        float dl = dlogits_in[n] * isq;
        for (r = 0; r < (int)R; r++) {
            dQ[(size_t)n * R + r] += dl * c->h2n[(size_t)n * R + r];
            dh2[(size_t)n * R + r] += dl * c->Q[(size_t)n * R + r];
        }
    }
    /* 1b. h2n = LN(h3raw): push the readout gradient back through the norm */
    {
        float *d_raw = calloc((size_t)N * R, sizeof(float));
        if (!d_raw) { rc = -1; goto out; }
        ln_backward(c->h3, dh2, m->n3g, N, (int)R, d_raw, g->n3g, g->n3b);
        memcpy(dh2, d_raw, (size_t)N * R * sizeof(float));
        free(d_raw);
    }
    /* 2. h2[n] = sum_l a2[n,l] gl[l] ; softmax over l on s3 */
    for (n = 0; n < N; n++)
        for (l = 0; l < (int)L; l++)
            da2[(size_t)n * L + l] = dh2[(size_t)n * R + 0] * 0.0f;  /* set below */
    for (n = 0; n < N; n++) {
        const float *a = c->att3 + (size_t)n * L;
        float dot = 0.0f;
        for (l = 0; l < (int)L; l++) {
            da2[(size_t)n * L + l] = dh2[(size_t)n * R + 0];   /* placeholder */
            dot += da2[(size_t)n * L + l] * a[l];
        }
        (void)dot;
    }
    /* 2a. dg from h2, and da2 = dh2 * g */
    for (l = 0; l < (int)L; l++)
        for (n = 0; n < N; n++) {
            float wt = c->att3[(size_t)n * L + l];
            for (r = 0; r < (int)R; r++)
                dg[(size_t)l * R + r] += dh2[(size_t)n * R + r] * wt;
        }
    for (n = 0; n < N; n++) {
        const float *a = c->att3 + (size_t)n * L;
        float dot = 0.0f;
        for (l = 0; l < (int)L; l++) {
            float v = 0.0f;
            for (r = 0; r < (int)R; r++) v += dh2[(size_t)n * R + r] * c->gl[(size_t)l * R + r];
            da2[(size_t)n * L + l] = v;
            dot += v * a[l];
        }
        for (l = 0; l < (int)L; l++)
            ds3[(size_t)n * L + l] = a[l] * (da2[(size_t)n * L + l] - dot);
    }
    /* 2b. s3[n,l] = isq * Q[n].g[l] */
    for (n = 0; n < N; n++)
        for (l = 0; l < (int)L; l++) {
            float dsc = ds3[(size_t)n * L + l] * isq;
            for (r = 0; r < (int)R; r++) {
                dQ[(size_t)n * R + r] += dsc * c->gl[(size_t)l * R + r];
                dg[(size_t)l * R + r] += dsc * c->Q[(size_t)n * R + r];
            }
        }

    /* 2c. gl = LN(y): push dg back through the norm */
    {
        float *d_raw = calloc(L * R, sizeof(float));
        if (!d_raw) { rc = -1; goto out; }
        ln_backward(c->y, dg, m->n2g, (int)L, (int)R, d_raw, g->n2g, g->n2b);
        memcpy(dg, d_raw, L * R * sizeof(float));
        free(d_raw);
    }
    /* 3. g[l] = sum_n b[l,n] h1n[n] ; softmax over n on s2 */
    for (l = 0; l < (int)L; l++)
        for (n = 0; n < N; n++) {
            float wt = c->att2[(size_t)l * N + n];
            for (r = 0; r < (int)R; r++)
                dh1[(size_t)n * R + r] += dg[(size_t)l * R + r] * wt;
        }
    for (l = 0; l < (int)L; l++) {
        const float *b = c->att2 + (size_t)l * N;
        float dot = 0.0f;
        for (n = 0; n < N; n++) {
            float v = 0.0f;
            for (r = 0; r < (int)R; r++) v += dg[(size_t)l * R + r] * c->h1n[(size_t)n * R + r];
            db[(size_t)l * N + n] = v;
            dot += v * b[n];
        }
        for (n = 0; n < N; n++)
            ds2[(size_t)l * N + n] = b[n] * (db[(size_t)l * N + n] - dot);
    }
    /* 3b. s2[l,n] = isq * Q2[l].h1[n] */
    for (l = 0; l < (int)L; l++)
        for (n = 0; n < N; n++) {
            float dsc = ds2[(size_t)l * N + n] * isq;
            for (r = 0; r < (int)R; r++) {
                dQ2[(size_t)l * R + r] += dsc * c->h1n[(size_t)n * R + r];
                dh1[(size_t)n * R + r] += dsc * c->Q2[(size_t)l * R + r];
            }
        }

    /* 3c. h1n = LN(u): push dh1 back through the norm */
    {
        float *d_raw = calloc((size_t)N * R, sizeof(float));
        if (!d_raw) { rc = -1; goto out; }
        ln_backward(c->u, dh1, m->n1g, N, (int)R, d_raw, g->n1g, g->n1b);
        memcpy(dh1, d_raw, (size_t)N * R * sizeof(float));
        free(d_raw);
    }
    /* 4. h1[n] = sum_l a1[n,l] V[l] ; softmax over l on s1 */
    for (n = 0; n < N; n++) {
        const float *a = c->attn + (size_t)n * L;
        float dot = 0.0f;
        for (l = 0; l < (int)L; l++) {
            float v = 0.0f;
            for (r = 0; r < (int)R; r++) v += dh1[(size_t)n * R + r] * c->V[(size_t)l * R + r];
            da1[(size_t)n * L + l] = v;
            dot += v * a[l];
        }
        for (l = 0; l < (int)L; l++) {
            ds1[(size_t)n * L + l] = a[l] * (da1[(size_t)n * L + l] - dot);
            for (r = 0; r < (int)R; r++)
                dV[(size_t)l * R + r] += dh1[(size_t)n * R + r] * a[l];
        }
    }
    /* 4b. s1[n,l] = isq * Q[n].K[l] */
    for (n = 0; n < N; n++)
        for (l = 0; l < (int)L; l++) {
            float dsc = ds1[(size_t)n * L + l] * isq;
            for (r = 0; r < (int)R; r++) {
                dQ[(size_t)n * R + r] += dsc * c->K[(size_t)l * R + r];
                dK[(size_t)l * R + r] += dsc * c->Q[(size_t)n * R + r];
            }
        }

    /* 5. projections -> W-dim branches. Wq/Wk/Wv are SHARED across rounds. */
    for (n = 0; n < N; n++)
        for (r = 0; r < (int)R; r++) {
            const float *wr = m->Wq + (size_t)r * W;
            const float *cn = c->opt_norm + (size_t)n * W;
            float dq = dQ[(size_t)n * R + r];
            for (w = 0; w < W; w++) {
                g->Wq[(size_t)r * W + w] += dq * cn[w];
                dc[(size_t)n * W + w] += dq * wr[w];
            }
        }
    for (l = 0; l < (int)L; l++) {
        for (r = 0; r < (int)R; r++) {
            const float *wk = m->Wk + (size_t)r * W;
            const float *wv = m->Wv + (size_t)r * W;
            const float *xn = c->ctx_norm + (size_t)l * W;
            float dk = dK[(size_t)l * R + r];
            float dv = dV[(size_t)l * R + r];
            float dq2 = dQ2[(size_t)l * R + r];
            for (w = 0; w < W; w++) {
                g->Wk[(size_t)r * W + w] += dk * xn[w] + dq2 * xn[w];
                g->Wv[(size_t)r * W + w] += dv * xn[w];
                dxx[(size_t)l * W + w] += dk * wk[w] + dv * wv[w] + dq2 * wk[w];
            }
        }
    }

    /* 6. LayerNorm backward, then pooling -> embedding */
    {
        float *dpo = calloc((size_t)N * W, sizeof(float));
        float *dce = calloc(L * W, sizeof(float));
        if (!dpo || !dce) { free(dpo); free(dce); rc = -1; goto out; }
        ln_backward(c->opt_pool, dc, m->lonw, N, (int)W, dpo, g->lonw, g->lonb);
        ln_backward(c->ctx_emb, dxx, m->lncw, (int)L, (int)W, dce, g->lncw, g->lncb);
        for (n = 0; n < N; n++) {
            size_t kl = cd_truncated_len(options[n] ? options[n] : "", O);
            float inv = kl ? 1.0f / (float)kl : 0.0f;
            for (k = 0; k < kl; k++) {
                uint32_t tid = (uint32_t)(unsigned char)options[n][k] + 1u;
                for (w = 0; w < W; w++)
                    g->emb[(size_t)tid * W + w] += dpo[(size_t)n * W + w] * inv;
            }
        }
        for (i = 0; i < L; i++) {
            uint32_t tid = (uint32_t)(unsigned char)context[i] + 1u;
            for (w = 0; w < W; w++) {
                g->emb[(size_t)tid * W + w] += dce[i * W + w];
                g->pos[i * W + w] += dce[i * W + w];
            }
        }
        free(dpo); free(dce);
    }
out:
    free(dQ); free(dK); free(dV); free(dQ2); free(dh1); free(dh2); free(dg);
    free(da2); free(ds3); free(db); free(ds2); free(da1); free(ds1);
    free(dc); free(dx); free(dcx); free(dxx);
    return rc;
}


/* Two-head inverse route: the baseline backward with the dQ accumulation aimed
 * at the inverse head's query map. Every shared-trunk gradient is unchanged, so
 * the trunk still receives gradient from both forms. */
static int rc_twohead_inv(const struct cd_model *m, const char *context,
                          const char *const *options, int n_options, int label,
                          const struct cd_cache *c, struct cd_grad *g) {
    uint32_t W = m->d.W, R = m->d.R, O = m->d.O;
    size_t L = c->L, i, k, w;
    int n, r;
    const float inv_sqrt = 1.0f / sqrtf((float)R);
    float *dlog = calloc((size_t)n_options, sizeof(float));
    float *dQ = calloc((size_t)n_options * R, sizeof(float));
    float *dK = calloc(L * R, sizeof(float));
    float *dV = calloc(L * R, sizeof(float));
    float *dAtt = calloc((size_t)n_options * R, sizeof(float));
    float *dAttn = calloc((size_t)n_options * L, sizeof(float));
    float *dSc = calloc((size_t)n_options * L, sizeof(float));
    float *dOn = calloc((size_t)n_options * W, sizeof(float));
    float *dCn = calloc(L * W, sizeof(float));
    int rc = 0;
    if (!dlog||!dQ||!dK||!dV||!dAtt||!dAttn||!dSc||!dOn||!dCn) { rc = -1; goto out; }

    for (n = 0; n < n_options; n++) dlog[n] = c->probs[n];
    dlog[label] -= 1.0f;

    /* judgement: logits[n] = (Wq2 . c_n . attended[n]) / sqrt(R) */
    for (n = 0; n < n_options; n++) {
        const float *q = c->Q + (size_t)n * R;
        float dl = dlog[n] * inv_sqrt;
        for (r = 0; r < (int)R; r++) {
            dQ[(size_t)n * R + r] += dl * c->attended[(size_t)n * R + r];
            dAtt[(size_t)n * R + r] += dl * q[r];
        }
    }
    for (n = 0; n < n_options; n++) {
        const float *a = c->attn + (size_t)n * L;
        for (i = 0; i < L; i++) {
            float wa = a[i];
            for (r = 0; r < (int)R; r++) {
                dAttn[(size_t)n * L + i] += dAtt[(size_t)n * R + r] * c->V[i * R + r];
                dV[i * R + r] += dAtt[(size_t)n * R + r] * wa;
            }
        }
    }
    for (n = 0; n < n_options; n++) {
        const float *p = c->attn + (size_t)n * L;
        const float *dp = dAttn + (size_t)n * L;
        float dot = 0.0f;
        for (i = 0; i < L; i++) dot += dp[i] * p[i];
        for (i = 0; i < L; i++) dSc[(size_t)n * L + i] = p[i] * (dp[i] - dot);
    }
    for (n = 0; n < n_options; n++) {
        const float *q = c->Q + (size_t)n * R;
        for (i = 0; i < L; i++) {
            float dsc = dSc[(size_t)n * L + i] * inv_sqrt;
            const float *kk = c->K + i * R;
            for (r = 0; r < (int)R; r++) {
                dQ[(size_t)n * R + r] += dsc * kk[r];
                dK[i * R + r] += dsc * q[r];
            }
        }
    }
    /* the two projections differ only in where dQ lands: Wq2 for this head,
     * Wq for the forward head. The context maps are whatever this
     * architecture resolved for the inverse form (shared in C0, per-form in
     * C1-C3), so the gradient must follow the same resolution as the forward. */
    for (n = 0; n < n_options; n++)
        for (r = 0; r < (int)R; r++) {
            const float *wr = m->Wq2 + (size_t)r * W;
            const float *on = c->opt_norm + (size_t)n * W;
            float dq = dQ[(size_t)n * R + r];
            for (w = 0; w < W; w++) {
                g->Wq2[(size_t)r * W + w] += dq * on[w];
                dOn[(size_t)n * W + w] += dq * wr[w];
            }
        }
    {
        const float *Wk_e, *Wv_e, *Wk_p, *Wv_p;
        float *gWk, *gWv;
        int summed;
        cd_tw_weights(m, CD_HEAD_INV, &Wk_e, &Wv_e, &Wk_p, &Wv_p, &summed,
                      c->tWk, c->tVv);
        gWk = (Wk_e == m->Wk2) ? g->Wk2 : g->Wk;
        gWv = (Wv_e == m->Vv2) ? g->Vv2 : g->Wv;
        for (i = 0; i < L; i++)
            for (r = 0; r < (int)R; r++) {
                const float *wk = Wk_e + (size_t)r * W;
                const float *wv = Wv_e + (size_t)r * W;
                const float *xn = c->ctx_norm + i * W;
                float dk = dK[i * R + r], dv = dV[i * R + r];
                for (w = 0; w < W; w++) {
                    gWk[(size_t)r * W + w] += dk * xn[w];
                    gWv[(size_t)r * W + w] += dv * xn[w];
                    dCn[i * W + w] += dk * wk[w] + dv * wv[w];
                }
            }
    }
    {
        float *dpo = calloc((size_t)n_options * W, sizeof(float));
        float *dce = calloc(L * W, sizeof(float));
        if (!dpo || !dce) { free(dpo); free(dce); rc = -1; goto out; }
        ln_backward(c->opt_pool, dOn, m->lonw, n_options, (int)W, dpo, g->lonw, g->lonb);
        ln_backward(c->ctx_emb, dCn, m->lncw, (int)L, (int)W, dce, g->lncw, g->lncb);
        for (n = 0; n < n_options; n++) {
            size_t kl = cd_truncated_len(options[n] ? options[n] : "", O);
            float inv = kl ? 1.0f / (float)kl : 0.0f;
            for (k = 0; k < kl; k++) {
                uint32_t tid = (uint32_t)(unsigned char)options[n][k] + 1u;
                for (w = 0; w < W; w++)
                    g->emb[(size_t)tid * W + w] += dpo[(size_t)n * W + w] * inv;
            }
        }
        for (i = 0; i < L; i++) {
            uint32_t tid = (uint32_t)(unsigned char)context[i] + 1u;
            for (w = 0; w < W; w++) {
                g->emb[(size_t)tid * W + w] += dce[i * W + w];
                g->pos[i * W + w] += dce[i * W + w];
            }
        }
        free(dpo); free(dce);
    }
out:
    free(dlog); free(dQ); free(dK); free(dV); free(dAtt);
    free(dAttn); free(dSc); free(dOn); free(dCn);
    return rc;
}

/* ---- backward --------------------------------------------------------- */

static int rc_twohead_inv(const struct cd_model *m, const char *context,
                          const char *const *options, int n_options, int label,
                          const struct cd_cache *c, struct cd_grad *g);

static int xbid_backward(const struct cd_model *m, const char *context,
                         const char *const *options, int n_options,
                         const struct cd_cache *c, struct cd_grad *g,
                         const float *dlogits_in);

/* dlogits_in, when non-NULL, REPLACES the standard probs-minus-one-hot
 * upstream gradient. That is the single injection point Stage 0.18 needs: the
 * bandit supplies its own analytic dL/dlogits (policy + entropy + KL) and reuses
 * every existing backward path unchanged. All pre-existing callers pass NULL and
 * therefore keep byte-identical behaviour. */
static int cd_backward(const struct cd_model *m, const char *context,
                       const char *const *options, int n_options, int label,
                       struct cd_cache *c, struct cd_grad *g, int no_clamp,
                       int head, const float *dlogits_in) {
    uint32_t W = m->d.W, R = m->d.R, O = m->d.O;
    size_t L = c->L, i, k, w;
    int n, r;
    float inv_sqrt = 1.0f / sqrtf((float)R);
    float *dlogits = calloc((size_t)n_options, sizeof(float));
    float *dQ = calloc((size_t)n_options * R, sizeof(float));
    float *dK = calloc(L * R, sizeof(float));
    float *dV = calloc(L * R, sizeof(float));
    float *dAtt = calloc((size_t)n_options * L, sizeof(float));
    float *dSc = calloc((size_t)n_options * L, sizeof(float));
    float *dAttn = calloc((size_t)n_options * R, sizeof(float));
    float *dOn = calloc((size_t)n_options * W, sizeof(float));
    float *dCn = calloc(L * W, sizeof(float));
    float *dCent = calloc(W, sizeof(float));
    int rc = 0;

    if (!dlogits || !dQ || !dK || !dV || !dAtt || !dSc || !dAttn || !dOn ||
        !dCn || !dCent) { rc = -1; goto out; }

    /* loss and dlogits */
    c->loss = (float)cd_loss_of(c, label, no_clamp);
    if (dlogits_in) {
        for (n = 0; n < n_options; n++) dlogits[n] = dlogits_in[n];
    } else {
        for (n = 0; n < n_options; n++) dlogits[n] = c->probs[n];
        dlogits[label] -= 1.0f;             /* upstream: unscaled */
    }

    if (m->arch == CD_ARCH_XBID)
        return xbid_backward(m, context, options, n_options, c, g, dlogits);

    /* CD_ARCH_SUMHEAD deliberately falls through to the baseline backward:
     * its score uses (Wq + Wq2).c, so dL/dWq2 equals dL/dWq exactly, and the
     * batch loop copies one onto the other. One judgement path, matched
     * parameter count. */

    if (cd_tw_split_query((int)m->arch) && head == CD_HEAD_INV) {
        /* Identical arithmetic to the baseline backward, with the dQ
         * accumulation redirected to the inverse head's own query map. Every
         * shared-truck gradient (Wk, Wv, LayerNorms, embeddings, positions) is
         * untouched, so the trunk still learns from both forms. */
        rc_twohead_inv(m, context, options, n_options, label, c, g);
        return 0;
    }

    /* 8. readout */
    for (n = 0; n < n_options; n++) {
        const float *q = c->Q + (size_t)n * R;
        float dl = dlogits[n] * inv_sqrt;
        for (r = 0; r < (int)R; r++) {
            dQ[(size_t)n * R + r] += dl * c->attended[(size_t)n * R + r];
            dAttn[(size_t)n * R + r] += dl * q[r];
        }
    }
    if (m->arch == CD_ARCH_INTERACT) {
        for (n = 0; n < n_options; n++) {
            const float *on = c->opt_norm + (size_t)n * W;
            float dl = dlogits[n] * inv_sqrt;
            for (r = 0; r < (int)R; r++) {
                const float *wr = m->Wo + (size_t)r * W;
                double acc = 0.0;
                for (w = 0; w < W; w++) acc += (double)on[w] * (double)wr[w];
                for (w = 0; w < W; w++) acc -= (double)c->cent[w] * (double)wr[w];
                g->gb[r] += dl;
                g->gc[r] += dl * (float)acc;
                for (w = 0; w < W; w++) {
                    float diff = on[w] - c->cent[w];
                    g->Wo[(size_t)r * W + w] += dl * m->gc[r] * diff;
                    dOn[(size_t)n * W + w] += dl * m->gc[r] * wr[w];
                    dCent[w] -= dl * m->gc[r] * wr[w];
                }
            }
        }
        for (n = 0; n < n_options; n++)
            for (w = 0; w < W; w++) dOn[(size_t)n * W + w] += dCent[w] / (float)n_options;
    }

    /* 7. attended */
    for (n = 0; n < n_options; n++) {
        const float *a = c->attn + (size_t)n * L;
        for (i = 0; i < L; i++) {
            float wa = a[i];
            for (r = 0; r < (int)R; r++) {
                dAtt[(size_t)n * L + i] += dAttn[(size_t)n * R + r] * c->V[i * R + r];
                dV[i * R + r] += dAttn[(size_t)n * R + r] * wa;
            }
        }
    }

    /* 6. softmax backward over L */
    for (n = 0; n < n_options; n++) {
        const float *p = c->attn + (size_t)n * L;
        const float *dp = dAtt + (size_t)n * L;
        float *ds = dSc + (size_t)n * L;
        float dot = 0.0f;
        for (i = 0; i < L; i++) dot += dp[i] * p[i];
        for (i = 0; i < L; i++) ds[i] = p[i] * (dp[i] - dot);
    }

    /* 5. scores */
    for (n = 0; n < n_options; n++) {
        const float *q = c->Q + (size_t)n * R;
        for (i = 0; i < L; i++) {
            float dsc = dSc[(size_t)n * L + i] * inv_sqrt;
            const float *kk = c->K + i * R;
            for (r = 0; r < (int)R; r++) {
                dQ[(size_t)n * R + r] += dsc * kk[r];
                dK[i * R + r] += dsc * q[r];
            }
        }
    }

    /* 4. Q, K, V projections */
    for (n = 0; n < n_options; n++)
        for (r = 0; r < (int)R; r++) {
            const float *wr = m->Wq + (size_t)r * W;
            const float *on = c->opt_norm + (size_t)n * W;
            float dq = dQ[(size_t)n * R + r];
            for (w = 0; w < W; w++) {
                g->Wq[(size_t)r * W + w] += dq * on[w];
                dOn[(size_t)n * W + w] += dq * wr[w];
            }
        }
    /* Context keys and values. C0 shares Wk/Wv, C1-C3 give the inverse form
     * its own map, and the SUM_* controls put both on one path (so the
     * gradient is mirrored onto the partner map in the training loop). */
    {
        const float *Wk_fwd, *Wk_p, *Wv_p, *Wv_fwd;
        float *gWk, *gWv;
        int summed;
        cd_tw_weights(m, head, &Wk_fwd, &Wv_fwd, &Wk_p, &Wv_p, &summed,
                      c->tWk, c->tVv);
        gWk = (Wk_fwd == m->Wk2) ? g->Wk2 : g->Wk;
        gWv = (Wv_fwd == m->Vv2) ? g->Vv2 : g->Wv;
        for (i = 0; i < L; i++)
            for (r = 0; r < (int)R; r++) {
                const float *wk = Wk_fwd + (size_t)r * W;
                const float *cn = c->ctx_norm + i * W;
                float dk = dK[i * R + r];
                for (w = 0; w < W; w++) {
                    gWk[(size_t)r * W + w] += dk * cn[w];
                    dCn[i * W + w] += dk * wk[w];
                }
            }
        for (i = 0; i < L; i++)
            for (r = 0; r < (int)R; r++) {
                const float *wv = Wv_fwd + (size_t)r * W;
                const float *cn = c->ctx_norm + i * W;
                float dv = dV[i * R + r];
                for (w = 0; w < W; w++) {
                    gWv[(size_t)r * W + w] += dv * cn[w];
                    dCn[i * W + w] += dv * wv[w];
                }
            }
    }

    /* 3. LayerNorm backward */
    {
        float *dc = calloc(L * W, sizeof(float));
        float *dpo = calloc((size_t)n_options * W, sizeof(float));
        if (!dc || !dpo) { free(dc); free(dpo); rc = -1; goto out; }
        ln_backward(c->ctx_emb, dCn, m->lncw, (int)L, (int)W, dc, g->lncw, g->lncb);
        ln_backward(c->opt_pool, dOn, m->lonw, n_options, (int)W, dpo, g->lonw, g->lonb);
        /* 2. option pool -> embedding */
        for (n = 0; n < n_options; n++) {
            size_t kl = cd_truncated_len(options[n] ? options[n] : "", O);
            float inv = kl ? 1.0f / (float)kl : 0.0f;
            for (k = 0; k < kl; k++) {
                uint32_t tid = (uint32_t)(unsigned char)options[n][k] + 1u;
                for (w = 0; w < W; w++)
                    g->emb[(size_t)tid * W + w] += dpo[(size_t)n * W + w] * inv;
            }
        }
        /* 1. context embedding -> embedding + position */
        for (i = 0; i < L; i++) {
            uint32_t tid = (uint32_t)(unsigned char)context[i] + 1u;
            for (w = 0; w < W; w++) {
                g->emb[(size_t)tid * W + w] += dc[i * W + w];
                g->pos[i * W + w] += dc[i * W + w];
            }
        }
        free(dc); free(dpo);
    }

out:
    free(dlogits); free(dQ); free(dK); free(dV); free(dAtt);
    free(dSc); free(dAttn); free(dOn); free(dCn); free(dCent);
    return rc;
}

/* ---- Adam, transcribed from train.c:133-166 --------------------------- *
 *
 * Upstream's decay is COUPLED, not decoupled. It computes
 *     upd    = m_hat / (sqrt(v_hat) + eps)      with eps OUTSIDE the sqrt
 *     wd_term = wd * p
 *     p     -= lr * (upd + wd_term)
 * so the L2 term sits INSIDE the lr-scaled update, and it is applied to every
 * parameter including LayerNorm gamma/beta. LEGACY reproduces that verbatim.
 * CORRECT can switch the norm parameters out of the decay set, and nothing
 * else about the formula changes.
 */
struct cd_adam {
    float *mb, *vb, *mp, *vp, *mlw, *vlw, *mlb, *vlb,
          *mlw2, *vlw2, *mlb2, *vlb2, *mq, *vq, *mk, *vk, *mv, *vv,
          *mo, *vo, *mgc, *vgc, *mgb, *vgb,
          *mq2, *vq2, *mk2, *vk2, *mv2, *vv2,
          *mn1g, *vn1g, *mn1b, *vn1b, *mn2g, *vn2g,
          *mn2b, *vn2b, *mn3g, *vn3g, *mn3b, *vn3b;
    int64_t t;
};

static void adam_update(float *p, float *m, float *v, const float *g,
                        size_t n, int64_t t, double lr, double b1, double b2,
                        double eps, double wd, int skip_first_W,
                        int decay_exclude) {
    double b1t = pow(b1, (double)t), b2t = pow(b2, (double)t);
    double inv1 = 1.0 / (1.0 - b1t), inv2 = 1.0 / (1.0 - b2t);
    size_t i;
    for (i = 0; i < n; i++) {
        float gi;
        double upd, wd_term;
        if (skip_first_W && i < (size_t)skip_first_W) continue;
        gi = g[i];
        m[i] = (float)(b1 * (double)m[i] + (1.0 - b1) * (double)gi);
        v[i] = (float)(b2 * (double)v[i] + (1.0 - b2) * (double)gi * (double)gi);
        upd = ((double)m[i] * inv1) / (sqrt((double)v[i] * inv2) + eps);
        wd_term = (decay_exclude && i < (size_t)skip_first_W) ? 0.0 : wd * (double)p[i];
        p[i] = (float)((double)p[i] - lr * (upd + wd_term));
    }
}

#define ARRSET(prefix)                                                        \
    do {                                                                      \
        a->m##prefix = calloc(cn, sizeof(float));                             \
        a->v##prefix = calloc(cn, sizeof(float));                             \
        if (!a->m##prefix || !a->v##prefix) return -1;                        \
    } while (0)

static int adam_init(struct cd_adam *a, const struct cd_model *m,
                     size_t emb_n, size_t pos_n, size_t norm_n, size_t qkv_n) {
    size_t cn;
    memset(a, 0, sizeof *a);
    cn = emb_n; ARRSET(b);
    cn = pos_n; ARRSET(p);
    cn = norm_n; ARRSET(lw); ARRSET(lb); ARRSET(lw2); ARRSET(lb2);
    cn = qkv_n;  ARRSET(q); ARRSET(k); ARRSET(v);
    if (m->arch == CD_ARCH_XBID) {
        cn = (size_t)m->d.R;
        ARRSET(n1g); ARRSET(n1b); ARRSET(n2g); ARRSET(n2b); ARRSET(n3g);
        ARRSET(n3b);
    }
    if (m->Wq2) { cn = qkv_n; ARRSET(q2); }
    if (m->Wk2) { cn = qkv_n; ARRSET(k2); }
    if (m->Vv2) { cn = qkv_n; ARRSET(v2); }
    if (m->arch == CD_ARCH_INTERACT) {
        cn = qkv_n; ARRSET(o);
        cn = norm_n; ARRSET(gc); ARRSET(gb);
    }
    return 0;
}

static void adam_release(struct cd_adam *a) {
    float **p = (float **)&a->mb;
    size_t i;
    /* the arrays are contiguous in declaration order only by construction;
     * free them by name to stay explicit and safe. */
    (void)p; (void)i;
    free(a->mb); free(a->vb); free(a->mp); free(a->vp);
    free(a->mlw); free(a->vlw); free(a->mlb); free(a->vlb);
    free(a->mlw2); free(a->vlw2); free(a->mlb2); free(a->vlb2);
    free(a->mq); free(a->vq); free(a->mk); free(a->vk); free(a->mv); free(a->vv);
    free(a->mn1g); free(a->vn1g); free(a->mn1b); free(a->vn1b);
    free(a->mn2g); free(a->vn2g); free(a->mn2b); free(a->vn2b);
    free(a->mn3g); free(a->vn3g); free(a->mn3b); free(a->vn3b);
    free(a->mq2); free(a->vq2);
    free(a->mk2); free(a->vk2); free(a->mv2); free(a->vv2);
    free(a->mo); free(a->vo); free(a->mgc); free(a->vgc); free(a->mgb); free(a->vgb);
    memset(a, 0, sizeof *a);
}

static void adam_step(struct cd_adam *a, struct cd_model *m, struct cd_grad *g,
                      double lr, double b1, double b2, double eps, double wd,
                      int decay_exclude_norm) {
    size_t W = m->d.W;
    a->t++;
    int64_t t = a->t;
    adam_update(m->emb, a->mb, a->vb, g->emb, g->n_emb, t, lr, b1, b2, eps, wd, (int)W, 0);
    adam_update(m->pos, a->mp, a->vp, g->pos, g->n_pos, t, lr, b1, b2, eps, wd, 0, 0);
    if (!decay_exclude_norm) {
        adam_update(m->lncw, a->mlw, a->vlw, g->lncw, W, t, lr, b1, b2, eps, wd, 0, 0);
        adam_update(m->lncb, a->mlb, a->vlb, g->lncb, W, t, lr, b1, b2, eps, wd, 0, 0);
        adam_update(m->lonw, a->mlw2, a->vlw2, g->lonw, W, t, lr, b1, b2, eps, wd, 0, 0);
        adam_update(m->lonb, a->mlb2, a->vlb2, g->lonb, W, t, lr, b1, b2, eps, wd, 0, 0);
    } else {
        adam_update(m->lncw, a->mlw, a->vlw, g->lncw, W, t, lr, b1, b2, eps, wd, (int)W, 1);
        adam_update(m->lncb, a->mlb, a->vlb, g->lncb, W, t, lr, b1, b2, eps, wd, (int)W, 1);
        adam_update(m->lonw, a->mlw2, a->vlw2, g->lonw, W, t, lr, b1, b2, eps, wd, (int)W, 1);
        adam_update(m->lonb, a->mlb2, a->vlb2, g->lonb, W, t, lr, b1, b2, eps, wd, (int)W, 1);
    }
    adam_update(m->Wq, a->mq, a->vq, g->Wq, g->n_qkv, t, lr, b1, b2, eps, wd, 0, 0);
    adam_update(m->Wk, a->mk, a->vk, g->Wk, g->n_qkv, t, lr, b1, b2, eps, wd, 0, 0);
    adam_update(m->Wv, a->mv, a->vv, g->Wv, g->n_qkv, t, lr, b1, b2, eps, wd, 0, 0);
    if (m->Wq2 && g->Wq2)
        adam_update(m->Wq2, a->mq2, a->vq2, g->Wq2, g->n_qkv, t, lr, b1, b2, eps, wd, 0, 0);
    if (m->Wk2 && g->Wk2)
        adam_update(m->Wk2, a->mk2, a->vk2, g->Wk2, g->n_qkv, t, lr, b1, b2, eps, wd, 0, 0);
    if (m->Vv2 && g->Vv2)
        adam_update(m->Vv2, a->mv2, a->vv2, g->Vv2, g->n_qkv, t, lr, b1, b2, eps, wd, 0, 0);
    if (m->arch == CD_ARCH_XBID) {
        /* XBID's six round norms are trainable LayerNorm parameters. They were
         * allocated, initialised, used by cd_layernorm in the forward and given
         * gradients by the real backward, but never handed to the optimiser, so
         * they sat at 1.0/0.0 for the whole of Stage 0.8. They are R-dimensional
         * (not W plus an offset), and, like the other norm tensors, are exempt
         * from weight decay. */
        const size_t nr = (size_t)m->d.R;
        adam_update(m->n1g, a->mn1g, a->vn1g, g->n1g, nr, t, lr, b1, b2, eps, wd, 0, 1);
        adam_update(m->n1b, a->mn1b, a->vn1b, g->n1b, nr, t, lr, b1, b2, eps, wd, 0, 1);
        adam_update(m->n2g, a->mn2g, a->vn2g, g->n2g, nr, t, lr, b1, b2, eps, wd, 0, 1);
        adam_update(m->n2b, a->mn2b, a->vn2b, g->n2b, nr, t, lr, b1, b2, eps, wd, 0, 1);
        adam_update(m->n3g, a->mn3g, a->vn3g, g->n3g, nr, t, lr, b1, b2, eps, wd, 0, 1);
        adam_update(m->n3b, a->mn3b, a->vn3b, g->n3b, nr, t, lr, b1, b2, eps, wd, 0, 1);
    }
    if (m->arch == CD_ARCH_INTERACT) {
        adam_update(m->Wo, a->mo, a->vo, g->Wo, g->n_qkv, t, lr, b1, b2, eps, wd, 0, 0);
        adam_update(m->gc, a->mgc, a->vgc, g->gc, W, t, lr, b1, b2, eps, wd, 0, 0);
        adam_update(m->gb, a->mgb, a->vgb, g->gb, W, t, lr, b1, b2, eps, wd, 0, 0);
    }
    for (size_t w = 0; w < W; w++) m->emb[w] = 0.0f;   /* padding row */

    /* The derived transposes must track the updated parameters, or the exact
     * AVX2 path would read stale weights. This is pure data movement: it
     * cannot change any result, and it is not a parameter update. */
    cd_model_refresh_transposes(m);
}

/* ---- best-epoch shadow checkpoint ------------------------------------
 *
 * WAS INCOMPLETE FOR XBID AND TWOHEAD. The previous version copied 9 tensors
 * literally: emb, pos, the four LayerNorms, Wq, Wk, Wv, plus Wo/gc/gb for
 * INTERACT. It did NOT copy Wq2 (twohead, 4096 long) nor the six XBID norm
 * tensors n1g..n3b (R long each, trained, and persisted by cd_save).
 *
 * Consequence for every XBID checkpoint written by `sn train` before this fix:
 * the trainer snapshots the best epoch with model_copy(&best,&m) and restores
 * it with model_copy(&m,&best) before saving, so the saved model carried
 * best-epoch emb/pos/LayerNorms/Wq/Wk/Wv together with FINAL-EPOCH XBID norm
 * tensors. 18 of the 24 Stage 0.15 inv50 seeds have a best epoch before the
 * last, so those 18 are genuine mixed best/final snapshots. The defect was in
 * the SNAPSHOT only: forward, backward, the optimiser and cd_save/cd_load were
 * all correct.
 *
 * FIXED as of this commit, for future training only. It is deliberately NOT
 * retro-applied to existing checkpoints: every XBID result from Stage 0.14 to
 * 0.18 describes a model produced through the defective path, and the operator
 * has decided the frozen mixed inv50 remains the canonical artefact. The
 * measured behavioural effect of the defect is in docs/SNAPSHOT_IMPACT.md.
 *
 * One consequence to keep in mind: a model trained AFTER this fix snapshots a
 * complete best epoch, so it is not directly comparable to a pre-fix
 * checkpoint without accounting for the six tensors. Stage 0.18 is unaffected
 * because it starts from the frozen inv50 and does not retrain.
 */

/* The best-epoch snapshot, and the only place a whole model is duplicated.
 *
 * This walks the SAME inventory that backs cd_model_param_count
 * (cd_model_trainable + cd_model_tensor_sizes), so a tensor cannot be added to
 * an architecture without automatically being covered here. The previous
 * hand-written version listed nine tensors literally and silently dropped the
 * six XBID norm tensors (and twohead's Wq2), so every XBID checkpoint written
 * by `sn train` paired best-epoch weights with final-epoch norm tensors. See
 * docs/SNAPSHOT_IMPACT.md for the measurement of that defect.
 */
static void model_copy(struct cd_model *dst, const struct cd_model *src) {
    const float **stv = NULL, **dtv = NULL;
    size_t *ssz = cd_model_tensor_sizes(src);
    size_t nstv, ndtv, i;
    if (!ssz) return;
    nstv = cd_model_trainable(src, &stv);
    ndtv = cd_model_trainable(dst, &dtv);
    if (ndtv != nstv) { free(stv); free(dtv); free(ssz); return; }
    for (i = 0; i < nstv; i++)
        memcpy((void *)dtv[i], stv[i], ssz[i] * sizeof(float));
    free(stv); free(dtv); free(ssz);
    /* dst now holds a different Wq/Wk/Wv, so its cached transposes are stale.
     * forward() reads those caches, so a copy that skipped this would silently
     * evaluate a different model than the one it was given. */
    cd_model_refresh_transposes(dst);
}

int cd_test_model_copy(struct cd_model *dst, const struct cd_model *src) {
    model_copy(dst, src);
    return 0;
}

static void grad_scale(struct cd_grad *g, uint32_t W, uint32_t R, float s) {
    size_t i;
    for (i = 0; i < g->n_emb; i++) g->emb[i] *= s;
    for (i = 0; i < g->n_pos; i++) g->pos[i] *= s;
    for (i = 0; i < W; i++) {
        g->lncw[i] *= s; g->lncb[i] *= s;
        g->lonw[i] *= s; g->lonb[i] *= s;
    }
    for (i = 0; i < g->n_qkv; i++) { g->Wq[i] *= s; g->Wk[i] *= s; g->Wv[i] *= s; }
    if (g->Wk2) for (i = 0; i < g->n_qkv; i++) g->Wk2[i] *= s;
    if (g->Vv2) for (i = 0; i < g->n_qkv; i++) g->Vv2[i] *= s;
    if (g->Wo) {
        for (i = 0; i < g->n_qkv; i++) g->Wo[i] *= s;
        for (i = 0; i < W; i++) { g->gc[i] *= s; g->gb[i] *= s; }
    }
    if (g->Wq2) for (i = 0; i < g->n_qkv; i++) g->Wq2[i] *= s;
    if (g->n1g) {
        float *v[6] = {g->n1g, g->n1b, g->n2g, g->n2b, g->n3g, g->n3b};
        int t;
        for (t = 0; t < 6; t++)
            for (i = 0; i < (size_t)R; i++) v[t][i] *= s;
    }
}

/* ---- evaluation ------------------------------------------------------- */

static void eval_dataset(const struct cd_model *m, const struct cd_dataset *ds,
                         size_t from, size_t to, struct cd_cache *c,
                         struct cd_epoch *out, int no_clamp)
{ eval_dataset_head(m, ds, from, to, c, out, no_clamp, CD_HEAD_FWD); }


/* ---- parallel evaluation across examples (bit-exact) --------------------
 *
 * SCOPE. Acceleration only. The decision rule, the metrics and the output are
 * unchanged.
 *
 * WHY THE REDUCTION STAYS SEQUENTIAL. The per-example forward pass is entirely
 * independent, so it parallelises trivially. The REDUCTION is not: `nll` is a
 * running sum of doubles, and floating-point addition is not associative, so
 * summing per-thread partials would round differently and move the reported
 * number. Therefore each thread writes its example's results into a slot
 * indexed by the example, and the caller then folds those slots in index order,
 * exactly as the single-threaded loop did. Per-example values are identical, the
 * order of the sum is identical, so the output is byte-identical.
 *
 * Output ordering is likewise preserved: nothing is printed from a worker, and
 * the bins are filled in index order in the caller.
 *
 * Thread count comes from CANDOR_THREADS and DEFAULTS TO 1, so the default
 * behaviour of every existing command is bit-for-bit what it was.
 */
#include <pthread.h>

struct cd_exres {
    double nll;      /* exactly cd_loss_of() for this example */
    float  conf;     /* max softmax probability              */
    int    top1, top3, correct, argmax;
    /* Stage 0.17 calibration instrumentation. Additive only: the fields above
     * are what the existing nll/top1/ece fold reads, and none of them changes,
     * so the default eval output stays byte-identical. p_true and probs[] are
     * what make Brier and NLL recomputable from a per-example record instead of
     * taken on trust, and true_label/predicted_label make "confidence when
     * correct" and "confidence when wrong" separable rather than inferred. */
    int    true_label;
    int    n_opt;
    double p_true;
    double probs[CD_MAX_OPT];
};

int cd_eval_threads(void) {
    static int n = -1;
    if (n < 0) {
        const char *e = getenv("CANDOR_THREADS");
        n = (e && atoi(e) > 0) ? atoi(e) : 1;
        if (n > 256) n = 256;
    }
    return n;
}

struct cd_fwd_job {
    const struct cd_model *m;
    const struct cd_dataset *ds;
    size_t from, to;
    int head, no_clamp, nthreads, tid;
    struct cd_exres *out;
    struct cd_cache *c;
};

static void *cd_fwd_worker(void *arg) {
    struct cd_fwd_job *j = (struct cd_fwd_job *)arg;
    size_t i;
    int n;
    /* blocked ranges, assigned once, so the result slots are a pure function
     * of the example index and never of the scheduling */
    size_t span = (j->to - j->from + (size_t)j->nthreads - 1) / (size_t)j->nthreads;
    size_t lo = j->from + span * (size_t)j->tid;
    size_t hi = lo + span;
    if (hi > j->to) hi = j->to;
    if (lo > j->to) return NULL;
    for (i = lo; i < hi; i++) {
        int no = cd_data_nopt(j->ds, i);
        const char *opts[CD_MAX_OPT];
        int lab, better = 0, argmax = 0;
        float bestv;
        struct cd_exres *r = &j->out[i - j->from];
        if (no > CD_MAX_OPT) no = CD_MAX_OPT;
        for (n = 0; n < no; n++) opts[n] = cd_data_opt(j->ds, i, n);
        lab = cd_data_label(j->ds, i);
        cd_forward_head(j->m, cd_data_ctx(j->ds, i), opts, no, j->c, j->head);
        r->nll = cd_loss_of(j->c, lab, j->no_clamp);
        bestv = j->c->probs[0];
        for (n = 0; n < no; n++) if (j->c->probs[n] > bestv) { bestv = j->c->probs[n]; argmax = n; }
        for (n = 0; n < no; n++) if (j->c->probs[n] > j->c->probs[lab]) better++;
        r->argmax = argmax;
        r->top1 = (argmax == lab);
        r->top3 = (better < 3);
        r->correct = (argmax == lab);
        r->conf = j->c->probs[argmax];
        /* Stage 0.17: preserve the per-example record. probs[] is copied in
         * dataset option order so the vector is unambiguous. */
        r->true_label = lab;
        r->n_opt = no;
        r->p_true = j->c->probs[lab];
        for (n = 0; n < no; n++) r->probs[n] = j->c->probs[n];
    }
    return NULL;
}

/* Fills out[0 .. to-from) with the per-example forward results. Safe to call
 * with nthreads == 1, which takes the same code path with no threads. */
static struct cd_exres *cd_forward_all(const struct cd_model *m,
                                       const struct cd_dataset *ds,
                                       size_t from, size_t to, int head,
                                       int no_clamp, struct cd_cache *reuse) {
    int T = cd_eval_threads();
    struct cd_exres *out;
    size_t n = to > from ? to - from : 0;
    if (n == 0) return NULL;
    if (T > (int)n) T = (int)n;
    out = calloc(n, sizeof *out);
    if (!out) return NULL;
    if (T <= 1) {
        struct cd_fwd_job j;
        memset(&j, 0, sizeof j);
        j.m = m; j.ds = ds; j.from = from; j.to = to; j.head = head;
        j.no_clamp = no_clamp; j.nthreads = 1; j.tid = 0; j.out = out; j.c = reuse;
        cd_fwd_worker(&j);
        return out;
    }
    {
        pthread_t *th = calloc((size_t)T, sizeof *th);
        struct cd_fwd_job *jobs = calloc((size_t)T, sizeof *jobs);
        struct cd_cache **caches = calloc((size_t)T, sizeof *caches);
        int t, launched = 0;
        if (!th || !jobs || !caches) { free(th); free(jobs); free(caches); free(out); return NULL; }
        for (t = 0; t < T; t++) {
            jobs[t].m = m; jobs[t].ds = ds; jobs[t].from = from; jobs[t].to = to;
            jobs[t].head = head; jobs[t].no_clamp = no_clamp;
            jobs[t].nthreads = T; jobs[t].tid = t; jobs[t].out = out;
            if (t == 0) {
                caches[t] = reuse;           /* slot 0 reuses the caller's cache */
            } else {
                caches[t] = calloc(1, sizeof **caches);
                if (caches[t]) cd_cache_alloc(caches[t], m);
            }
            jobs[t].c = caches[t];          /* each worker needs its own cache */
            if (pthread_create(&th[t], NULL, cd_fwd_worker, &jobs[t]) != 0) {
                /* fall back to running the remainder inline */
                jobs[t].nthreads = 1; jobs[t].tid = 0;
                cd_fwd_worker(&jobs[t]);
            } else launched++;
        }
        for (t = 0; t < T; t++) pthread_join(th[t], NULL);
        for (t = 1; t < T; t++) if (caches[t]) cd_cache_free(caches[t]);
        free(th); free(jobs); free(caches);
        (void)launched;
    }
    return out;
}

static void eval_dataset_head(const struct cd_model *m, const struct cd_dataset *ds,
                              size_t from, size_t to, struct cd_cache *c,
                              struct cd_epoch *out, int no_clamp, int head) {
    size_t i;
    double nll = 0.0;
    int top1 = 0, top3 = 0;
    struct cd_exres *ex;
    out->val_nll = 0.0f; out->val_top1 = 0.0f; out->val_top3 = 0.0f; out->ece = 0.0f;
    if (to <= from) return;
    ex = cd_forward_all(m, ds, from, to, head, no_clamp, c);
    if (!ex) return;
    /* Sequential fold in example order: the same additions, in the same order,
     * as the single-threaded loop, so the reported nll is bit-identical. */
    for (i = from; i < to; i++) {
        nll += ex[i - from].nll;
        top1 += ex[i - from].top1;
        top3 += ex[i - from].top3;
    }
    free(ex);
    out->val_nll = (float)(nll / (double)(to - from));
    out->val_top1 = (float)top1 / (float)(to - from);
    out->val_top3 = (float)top3 / (float)(to - from);
    out->ece = 0.0f;   /* filled by ece_dataset() */
}

/* ---- retained calibration bins ----------------------------------------
 *
 * MEASUREMENT ONLY. `cd_calib_add` is called with the same confidence and
 * correctness that the old inline ECE loop used, and `cd_calib_ece` reproduces
 * the old arithmetic exactly (count-weighted sum of |acc - mean_conf| over
 * non-empty bins, denominator = total examples binned). The scalar ECE reported
 * by every existing call site is now DERIVED from these bins, so
 * "reconstruct ECE from the retained bins" is exact by construction.
 */
void cd_calib_zero(struct cd_calib *cb) { memset(cb, 0, sizeof *cb); }

void cd_calib_add(struct cd_calib *cb, float conf, int correct) {
    int b = (int)(conf * (float)CD_CALIB_BINS);
    if (b < 0) b = 0;
    if (b >= CD_CALIB_BINS) b = CD_CALIB_BINS - 1;
    cb->bin_conf[b] += (double)conf;
    cb->bin_acc[b]  += (double)(correct ? 1 : 0);
    cb->bin_n[b]++;
    cb->n++;
}

double cd_calib_ece(const struct cd_calib *cb) {
    double total = 0.0;
    int b;
    for (b = 0; b < CD_CALIB_BINS; b++) {
        if (!cb->bin_n[b]) continue;
        total += (double)cb->bin_n[b] / (double)cb->n *
                 fabs(cb->bin_acc[b] / cb->bin_n[b] - cb->bin_conf[b] / cb->bin_n[b]);
    }
    return total;
}

int cd_calib_json(const struct cd_calib *cb, char *out, size_t len) {
    /* Fixed key order, fixed precision, ascending bins, empty bins omitted so
     * no division by zero can appear in the output. Two runs over the same
     * model and data must produce byte-identical text. */
    char tmp[2048];
    int n = 0, b, emitted = 0;
    n += snprintf(tmp + n, sizeof tmp - (size_t)n,
                  "{\"reliability\": {\"bins_total\": %d, \"n\": %zu, "
                  "\"ece\": %.6f, \"curve\": [",
                  CD_CALIB_BINS, cb->n, cd_calib_ece(cb));
    for (b = 0; b < CD_CALIB_BINS; b++) {
        double mc, acc, gap, lo;
        if (!cb->bin_n[b]) continue;
        mc  = cb->bin_conf[b] / (double)cb->bin_n[b];
        acc = cb->bin_acc[b] / (double)cb->bin_n[b];
        gap = acc - mc;                    /* signed: negative == overconfident */
        lo  = (double)b / (double)CD_CALIB_BINS;
        n += snprintf(tmp + n, sizeof tmp - (size_t)n,
                      "%s{\"bin\": %d, \"lo\": %.2f, \"hi\": %.2f, "
                      "\"n\": %d, \"mean_conf\": %.6f, \"acc\": %.6f, "
                      "\"gap\": %.6f}",
                      emitted ? ", " : "", b, lo, lo + 1.0 / (double)CD_CALIB_BINS,
                      cb->bin_n[b], mc, acc, gap);
        emitted = 1;
    }
    n += snprintf(tmp + n, sizeof tmp - (size_t)n, "]}}");
    if ((size_t)n >= len) return -1;
    memcpy(out, tmp, (size_t)n + 1);
    return n;
}

/* proper binned ECE, separate from the placeholder above. Retains the bins and
 * hands them back through `out` when it is non-NULL, so the existing scalar
 * call sites are unchanged and the curve is available for free wherever ECE
 * already was. */
static double ece_dataset(const struct cd_model *m, const struct cd_dataset *ds,
                          struct cd_cache *c)
{ return ece_dataset_head(m, ds, c, CD_HEAD_FWD, NULL); }

static double ece_dataset_head(const struct cd_model *m, const struct cd_dataset *ds,
                               struct cd_cache *c, int head,
                               struct cd_calib *out) {
    struct cd_calib cb;
    struct cd_exres *ex;
    size_t i;
    cd_calib_zero(&cb);
    ex = cd_forward_all(m, ds, 0, ds->n, head, 0, c);
    if (!ex) return cb.ece;
    /* Bins are filled in example order, so the bin sums are identical. */
    for (i = 0; i < ds->n; i++)
        cd_calib_add(&cb, ex[i].conf, ex[i].correct);
    free(ex);
    cb.ece = cd_calib_ece(&cb);
    if (out) *out = cb;
    return cb.ece;
}

/* ---- gradient check: central differences on the real batch loss -------- */

static int grad_check(const struct cd_cfg *cfg, const char *train_path,
                      int head, char *err, size_t err_len) {
    struct cd_dataset ds;
    struct cd_model m;
    struct cd_grad g;
    struct cd_cache c;
    size_t n_probe = 24;
    int probes[24][6];
    int np = 0, i, t;
    double worst = 0.0;
    const double eps = 1e-2;

    if (cd_data_load(train_path, &ds, err, err_len) != 0) return -1;
    cd_model_init(&m, &cfg->d, cfg->arch, cfg->seed);
    grad_alloc(&g, &m);
    cd_cache_alloc(&c, &m);

    /* analytic gradient of the mean loss over the first batch */
    {
        size_t blen = cfg->batch_size < (int)ds.n ? (size_t)cfg->batch_size : ds.n;
        double loss = 0.0;
        grad_zero(&g, m.d.W, m.d.R);
        for (i = 0; i < (int)blen; i++) {
            const char *opts[CD_MAX_OPT];
            int no = cd_data_nopt(&ds, i);
            if (no > CD_MAX_OPT) no = CD_MAX_OPT;
            for (t = 0; t < no; t++) opts[t] = cd_data_opt(&ds, i, t);
            cd_forward_head(&m, cd_data_ctx(&ds, i), opts, no, &c, head);
            loss += (double)c.loss;
            cd_backward(&m, cd_data_ctx(&ds, i), opts, no, cd_data_label(&ds, i),
                        &c, &g, cfg->no_clamp, head, NULL);
        }
        grad_scale(&g, m.d.W, m.d.R, 1.0f / (float)blen);
        (void)loss;
    }

    /* Pick probes across every tensor that this architecture actually owns,
     * so a per-form map that is never updated cannot pass by omission. The
     * count of live tensors decides the modulus: 0 Wq, 1 Wk, 2 Wv, 3 the
     * architecture's own extra map, 4 lncw, 5 emb, 6 Wq2, 7 Wk2, 8 Vv2. */
    {
        int live[9], nlive = 0;
        const int inv = (head == CD_HEAD_INV);
        live[nlive++] = 0;
        live[nlive++] = 1;
        live[nlive++] = 2;
        if (m.Wq2) live[nlive++] = 6;
        if (m.Wk2) live[nlive++] = 7;
        if (m.Vv2) live[nlive++] = 8;
        live[nlive++] = 4;
        live[nlive++] = 5;
        (void)inv;
        for (i = 0; i < (int)n_probe; i++) {
            int which = live[i % nlive];
            size_t n = (which == 4) ? m.d.W : (which == 5) ? m.n_emb : m.n_qkv;
            if ((size_t)np >= n_probe) break;
            probes[np][0] = which;
            probes[np][1] = (int)((size_t)i * 7919u % n);
            np++;
        }
    }

    for (i = 0; i < np; i++) {
        float *arr; float ga; double lp, lm;
        size_t blen = cfg->batch_size < (int)ds.n ? (size_t)cfg->batch_size : ds.n;
        int which = probes[i][0]; size_t idx = (size_t)probes[i][1];
        double a, b, orig;
        switch (which) {
        case 0: arr = (head == CD_HEAD_INV && m.Wq2) ? m.Wq2 : m.Wq; break;
        case 1: arr = m.Wk; break;
        case 2: arr = m.Wv; break;
        case 6: arr = m.Wq2 ? m.Wq2 : m.Wq; break;
        case 7: arr = m.Wk2 ? m.Wk2 : m.Wk; break;
        case 8: arr = m.Vv2 ? m.Vv2 : m.Wv; break;
        case 4: arr = m.lncw; break; default: arr = m.emb; break;
        }
        {
            float *garr;
            switch (which) {
            case 0: garr = (head == CD_HEAD_INV && g.Wq2) ? g.Wq2 : g.Wq; break;
            case 1: garr = g.Wk; break;
            case 2: garr = g.Wv; break;
            case 6: garr = g.Wq2 ? g.Wq2 : g.Wq; break;
            case 7: garr = g.Wk2 ? g.Wk2 : g.Wk; break;
            case 8: garr = g.Vv2 ? g.Vv2 : g.Wv; break;
            case 4: garr = g.lncw; break; default: garr = g.emb; break;
            }
            ga = garr[idx];
        }
        orig = (double)arr[idx];
        arr[idx] = (float)(orig + eps);
        lp = 0.0;
        for (t = 0; t < (int)blen; t++) {
            const char *opts[CD_MAX_OPT]; int no = cd_data_nopt(&ds, t);
            if (no > CD_MAX_OPT) no = CD_MAX_OPT;
            { int z; for (z = 0; z < no; z++) opts[z] = cd_data_opt(&ds, t, z); }
            cd_forward_head(&m, cd_data_ctx(&ds, t), opts, no, &c, head);
            lp += cd_loss_of(&c, cd_data_label(&ds, t), cfg->no_clamp);
        }
        lp /= (double)blen;
        arr[idx] = (float)(orig - eps);
        lm = 0.0;
        for (t = 0; t < (int)blen; t++) {
            const char *opts[CD_MAX_OPT]; int no = cd_data_nopt(&ds, t);
            if (no > CD_MAX_OPT) no = CD_MAX_OPT;
            { int z; for (z = 0; z < no; z++) opts[z] = cd_data_opt(&ds, t, z); }
            cd_forward_head(&m, cd_data_ctx(&ds, t), opts, no, &c, head);
            lm += cd_loss_of(&c, cd_data_label(&ds, t), cfg->no_clamp);
        }
        lm /= (double)blen;
        arr[idx] = (float)orig;
        a = (lp - lm) / (2.0 * eps);
        b = (double)ga;
        /* Mixed absolute+relative criterion. Pure relative error is
         * meaningless where the gradient is near zero: the float32 loss carries
         * ~1e-7 of noise, so an FD estimate of a 7e-5 gradient has ~5e-5 of
         * absolute noise and "fails" at 90% while being correct. The standard
         * form |a-b| <= atol + rtol*|b| is used instead, with atol set from the
         * observed loss noise rather than picked to pass. */
        {
            double atol = 1e-4, rtol = 2e-2;
            double slack = atol + rtol * fabs(b);
            double err = fabs(a - b);
            int ok = err <= slack;
            if (!ok && err > worst) worst = err;
            if (i < 8)
                printf("  probe tensor=%d idx=%zu analytic=%+.8e numeric=%+.8e "
                       "abs=%.2e slack=%.2e %s\n",
                       which, idx, b, a, err, slack, ok ? "ok" : "FAIL");
        }
    }
    printf("{\"grad_check\": true, \"probes\": %d, \"worst_abs_excess\": %.3e, "
           "\"atol\": 1e-4, \"rtol\": 2e-2}\n", np, worst);
    if (worst > 0.0) {
        snprintf(err, err_len, "gradient check FAILED: worst excess %.3e", worst);
        cd_cache_free(&c); grad_free(&g); cd_model_free(&m); cd_data_free(&ds);
        return -1;
    }
    cd_cache_free(&c); grad_free(&g); cd_model_free(&m); cd_data_free(&ds);
    return 0;
}

/* ---- the trainer ------------------------------------------------------ */

int cd_train(const struct cd_cfg *cfg, const char *train_path,
             const char *val_path, const char *out_model_path,
             struct cd_epoch *history, char *err, size_t err_len) {
    struct cd_cfg cfg_local;
    const struct cd_cfg *cfgp = cfg;
    struct cd_dataset tr, va;
    struct cd_model m, best;
    struct cd_grad g;
    struct cd_cache c;
    struct cd_adam a;
    size_t *indices = NULL;
    int epoch;
    double best_val = 1e30;
    int rc = -1;

    if (cfg->grad_check)
        return grad_check(cfg, train_path, cfg->grad_check - 1, err, err_len);

    if (cd_data_load(train_path, &tr, err, err_len) != 0) return -1;
    if (cd_data_load(val_path, &va, err, err_len) != 0) { cd_data_free(&tr); return -1; }

    /* The cache is sized from the option count, so it MUST be at least the
     * largest count present in the data. Derive it rather than trusting a
     * default, and refuse anything past the fixed per-example stack array. */
    cfg_local = *cfg;
    {
        int mx = cd_data_max_nopt(&tr);
        int mv = cd_data_max_nopt(&va);
        if (mv > mx) mx = mv;
        if (mx < 2) { snprintf(err, err_len, "need at least 2 options per row"); goto done0; }
        if (mx > CD_MAX_OPT) {
            snprintf(err, err_len, "row has %d options, limit is %d", mx, CD_MAX_OPT);
            goto done0;
        }
        if (mx > cfg_local.d.n_opt) cfg_local.d.n_opt = mx;
    }
    cfgp = &cfg_local;
    cfg = cfgp;

    cd_model_init(&m, &cfg->d, cfg->arch, cfg->seed);
    if (cfg->init_model) {
        /* Copy the shared trunk from a saved model and leave any
         * architecture-specific tensor at its own fresh init. Used by the
         * frozen-trunk probe: a forward-trained trunk, then train one head. */
        struct cd_model *src = cd_load(cfg->init_model, err, err_len);
        if (!src) goto done0;
        if (src->n_emb != m.n_emb || src->n_qkv != m.n_qkv) {
            snprintf(err, err_len, "init model shape does not match");
            cd_model_free(src); goto done0;
        }
        memcpy(m.emb, src->emb, m.n_emb * sizeof(float));
        memcpy(m.pos, src->pos, m.n_pos * sizeof(float));
        memcpy(m.lncw, src->lncw, m.d.W * sizeof(float));
        memcpy(m.lncb, src->lncb, m.d.W * sizeof(float));
        memcpy(m.lonw, src->lonw, m.d.W * sizeof(float));
        memcpy(m.lonb, src->lonb, m.d.W * sizeof(float));
        memcpy(m.Wq, src->Wq, m.n_qkv * sizeof(float));
        memcpy(m.Wk, src->Wk, m.n_qkv * sizeof(float));
        memcpy(m.Wv, src->Wv, m.n_qkv * sizeof(float));
        cd_model_free(src);
    }
    cd_model_init(&best, &cfg->d, cfg->arch, cfg->seed);
    grad_alloc(&g, &m);
    cd_cache_alloc(&c, &m);
    if (adam_init(&a, &m, m.n_emb, m.n_pos, m.d.W, m.n_qkv) != 0) {
        snprintf(err, err_len, "oom adam"); goto done;
    }
    indices = malloc(tr.n * sizeof(size_t));
    if (!indices) { snprintf(err, err_len, "oom indices"); goto done; }
    for (size_t i = 0; i < tr.n; i++) indices[i] = i;

    printf("{\"model\": \"%s\", \"arch\": \"%s\", \"W\": %u, \"R\": %u, \"C\": %u, "
           "\"O\": %u, \"params\": %zu, \"train_n\": %zu, \"val_n\": %zu, "
           "\"epochs\": %d, \"batch\": %d, \"lr\": %.6g, \"wd\": %.6g, \"seed\": %llu}\n",
           cfg->mode == CD_MODE_LEGACY ? "legacy" : "correct",
           cfg->arch == CD_ARCH_INTERACT ? "interact"
           : cfg->arch == CD_ARCH_XBID ? "xbid" : "baseline",
           cfg->d.W, cfg->d.R, cfg->d.C, cfg->d.O, cd_model_param_count(&m),
           tr.n, va.n, cfg->epochs, cfg->batch_size, cfg->lr, cfg->wd,
           (unsigned long long)cfg->seed);
    fflush(stdout);

    {
        struct timespec t0, t1;
        clock_gettime(CLOCK_MONOTONIC, &t0);
        for (epoch = 0; epoch < cfg->epochs; epoch++) {
            /* LEGACY: hardcoded constant, so --seed does NOT touch data order
             * (train.c:580). CORRECT with seed_drives_order mixes the seed in. */
            uint64_t rs;
            size_t i, start;
            double train_loss_sum = 0.0, train_cnt = 0.0;
            struct cd_epoch ep;

            if (cfg->mode == CD_MODE_LEGACY && !cfg->seed_drives_order)
                rs = 0x9e3779b97f4a7c15ull + (uint64_t)epoch * 0xbf58476d1ce4e5b9ull;
            else
                rs = cfg->seed + (uint64_t)epoch * 0xbf58476d1ce4e5b9ull;

            for (i = tr.n - 1; i > 0; i--) {
                size_t j, tmp;
                rs ^= rs >> 12; rs ^= rs << 25; rs ^= rs >> 27;
                rs *= 0x2545F4914F6CDD1DULL;
                j = rs % (i + 1);
                tmp = indices[i]; indices[i] = indices[j]; indices[j] = tmp;
            }

            for (start = 0; start < tr.n; start += (size_t)cfg->batch_size) {
                size_t end = start + (size_t)cfg->batch_size;
                double batch_loss = 0.0;
                int hd = CD_HEAD_FWD;   /* head used by this batch */
                if (end > tr.n) end = tr.n;
                grad_zero(&g, m.d.W, m.d.R);
                for (i = start; i < end; i++) {
                    size_t idx = indices[i];
                    const char *opts[CD_MAX_OPT];
                    int no = cd_data_nopt(&tr, idx);
                    int z;
                    if (m.arch == CD_ARCH_TWOHEAD) {
                        /* An explicit --train-head wins, because a
                         * single-form training file carries no form label and
                         * defaulting to forward silently trains the WRONG head. */
                        int f = (cfg->force_train_head >= 0)
                              ? cfg->force_train_head : cd_data_form(&tr, idx);
                        hd = (f == CD_HEAD_INV) ? CD_HEAD_INV : CD_HEAD_FWD;
                    }
                    if (no > CD_MAX_OPT) no = CD_MAX_OPT;
                    for (z = 0; z < no; z++) opts[z] = cd_data_opt(&tr, idx, z);
                    if (cd_forward_head(&m, cd_data_ctx(&tr, idx), opts, no,
                                        &c, hd) != 0) {
                        snprintf(err, err_len, "forward failed at row %zu", idx);
                        free(indices); goto done;
                    }
                    batch_loss += cd_loss_of(&c, cd_data_label(&tr, idx), cfg->no_clamp);
                    if (cd_backward(&m, cd_data_ctx(&tr, idx), opts, no,
                                    cd_data_label(&tr, idx), &c, &g,
                                    cfg->no_clamp, hd, NULL) != 0) {
                        snprintf(err, err_len, "backward failed at row %zu", idx);
                        free(indices); goto done;
                    }
                }
                train_loss_sum += batch_loss;
                train_cnt += (double)(end - start);
                if (m.arch == CD_ARCH_SUMHEAD && g.Wq2)
                    memcpy(g.Wq2, g.Wq, g.n_qkv * sizeof(float));
                if (cd_tw_sum_context((int)m.arch)) {
                    if (g.Wk2) memcpy(g.Wk2, g.Wk, g.n_qkv * sizeof(float));
                    if (g.Vv2) memcpy(g.Vv2, g.Wv, g.n_qkv * sizeof(float));
                }
                if (cd_tw_sum_context((int)m.arch)) {
                    if (g.Wk2) memcpy(g.Wk2, g.Wk, g.n_qkv * sizeof(float));
                    if (g.Vv2) memcpy(g.Vv2, g.Wv, g.n_qkv * sizeof(float));
                }
                if (cfg->freeze_trunk) {
                    /* Everything except the trained head's own map is zeroed,
                     * so the shared representation cannot move. */
                    float *keep = (hd == CD_HEAD_INV) ? g.Wq2 : g.Wq;
                    memset(g.emb, 0, g.n_emb * sizeof(float));
                    memset(g.pos, 0, g.n_pos * sizeof(float));
                    memset(g.lncw, 0, m.d.W * sizeof(float));
                    memset(g.lncb, 0, m.d.W * sizeof(float));
                    memset(g.lonw, 0, m.d.W * sizeof(float));
                    memset(g.lonb, 0, m.d.W * sizeof(float));
                    memset(g.Wk, 0, g.n_qkv * sizeof(float));
                    memset(g.Wv, 0, g.n_qkv * sizeof(float));
                    if (g.Wk2) memset(g.Wk2, 0, g.n_qkv * sizeof(float));
                    if (g.Vv2) memset(g.Vv2, 0, g.n_qkv * sizeof(float));
                    if (keep == g.Wq) memset(g.Wq2, 0, g.n_qkv * sizeof(float));
                    else memset(g.Wq, 0, g.n_qkv * sizeof(float));
                    if (g.Wo) memset(g.Wo, 0, g.n_qkv * sizeof(float));
                    if (g.gc) { memset(g.gc, 0, m.d.W * sizeof(float)); memset(g.gb, 0, m.d.W * sizeof(float)); }
                }
                grad_scale(&g, m.d.W, m.d.R, 1.0f / (float)(end - start));
                {   /* clip 1.0 is HARD-CODED upstream; cfg->clip only in CORRECT */
                    double norm = grad_global_norm(&m, &g);
                    if (cfg->mode == CD_MODE_LEGACY || cfg->clip <= 0.0) {
                        if (norm > 1.0 && norm > 0.0)
                            grad_scale(&g, m.d.W, m.d.R, (float)(1.0 / norm));
                    } else if (norm > cfg->clip && norm > 0.0) {
                        grad_scale(&g, m.d.W, m.d.R, (float)(cfg->clip / norm));
                    }
                }
                adam_step(&a, &m, &g, cfg->lr, 0.9, 0.999, 1e-8, cfg->wd,
                          cfg->decay_exclude_norm);
            }

            memset(&ep, 0, sizeof ep);
            ep.train_nll = (float)(train_loss_sum / train_cnt);
            eval_dataset(&m, &va, 0, va.n, &c, &ep, cfg->no_clamp);
            ep.ece = (float)ece_dataset(&m, &va, &c);
            clock_gettime(CLOCK_MONOTONIC, &t1);
            ep.seconds = (double)(t1.tv_sec - t0.tv_sec) +
                         (double)(t1.tv_nsec - t0.tv_nsec) / 1e9;
            printf("{\"epoch\": %d, \"train_nll\": %.6f, \"validation_nll\": %.6f, "
                   "\"val_top1\": %.4f, \"val_top3\": %.4f, \"elapsed\": %.2f}\n",
                   epoch + 1, ep.train_nll, ep.val_nll, ep.val_top1, ep.val_top3,
                   ep.seconds);
            fflush(stdout);
            if (history) history[epoch] = ep;
            if ((double)ep.val_nll < best_val) {
                best_val = (double)ep.val_nll;
                model_copy(&best, &m);
            }
        }
    }

    model_copy(&m, &best);
    if (out_model_path) {
        struct cd_epoch fin;
        memset(&fin, 0, sizeof fin);
        eval_dataset(&m, &va, 0, va.n, &c, &fin, cfg->no_clamp);
        fin.ece = (float)ece_dataset(&m, &va, &c);
        printf("{\"best_val_nll\": %.6f, \"final_val_nll\": %.6f, "
               "\"final_val_top1\": %.4f, \"final_val_top3\": %.4f, "
               "\"final_val_ece\": %.4f}\n",
               best_val, fin.val_nll, fin.val_top1, fin.val_top3, fin.ece);
        if ((cfg->q8 ? cd_save_q8(out_model_path, &m, err, err_len)
                      : cfg->q4 ? cd_save_q4(out_model_path, &m, err, err_len)
                      : cfg->q2 ? cd_save_q2(out_model_path, &m, err, err_len)
                      : cfg->q3 ? cd_save_q3(out_model_path, &m, err, err_len)
                      : cd_save(out_model_path, &m, err, err_len)) != 0) {
            free(indices); goto done;
        }
    }
    rc = 0;
done:
    free(indices);
    cd_data_free(&tr);
    cd_data_free(&va);
    cd_cache_free(&c);
    grad_free(&g);
    adam_release(&a);
    cd_model_free(&m);
    cd_model_free(&best);
    return rc;
done0:
    cd_data_free(&tr);
    cd_data_free(&va);
    return -1;
}

static int cd_eval_impl(const char *model_path, const char *test_path, int head,
                        char *err, size_t err_len) {
    struct cd_model *m = cd_load(model_path, err, err_len);
    struct cd_dataset ds;
    struct cd_cache c;
    struct cd_epoch e;
    struct cd_calib cal;
    char calbuf[2048];
    if (!m) { snprintf(err, err_len, "cannot read model %s", model_path); return -1; }
    if (cd_data_load(test_path, &ds, err, err_len) != 0) { cd_model_free(m); return -1; }
    cd_cache_alloc(&c, m);
    memset(&e, 0, sizeof e);
    eval_dataset_head(m, &ds, 0, ds.n, &c, &e, 1, head);
    e.ece = (float)ece_dataset_head(m, &ds, &c, head, &cal);
    /* Retained calibration curve FIRST, then the pre-existing summary line.
     *
     * Order matters: every established consumer of this command -- the Phase-A
     * gate and every stage runner -- reads the LAST JSON line, so emitting the
     * new measurement last would silently steal the summary slot and break
     * them all. Emitting it first keeps that contract intact with no change to
     * any consumer, and the summary line below is byte-identical to what it
     * was before this instrumentation existed. */
    if (cd_calib_json(&cal, calbuf, sizeof calbuf) > 0)
        printf("%s\n", calbuf);
    printf("{\"test_n\": %zu, \"nll\": %.6f, \"top1\": %.4f, \"top3\": %.4f, "
           "\"head\": \"%s\", \"ece\": %.4f}\n",
           ds.n, e.val_nll, e.val_top1, e.val_top3,
           head == CD_HEAD_INV ? "inv" : "fwd", e.ece);
    cd_cache_free(&c);
    cd_data_free(&ds);
    cd_model_free(m);
    return 0;
}


/* ---- per-example signal dump (MEASUREMENT ONLY) ------------------------
 *
 * Reads only what the forward pass already produced. No new parameters, no
 * architecture change, and nothing here feeds back into the model or the
 * decision rule. The point is to find out which already-computed quantities,
 * if any, order examples by likelihood of being correct in a way the softmax
 * confidence does not.
 */
static double sig_entropy(const float *p, int n) {
    double h = 0.0;
    int i;
    for (i = 0; i < n; i++)
        if (p[i] > 0.0f) h -= (double)p[i] * log((double)p[i]);
    return h;
}

int cd_signals(const char *model_path, const char *test_path, char *err,
               size_t errlen) {
    struct cd_model *m = cd_load(model_path, err, errlen);
    struct cd_dataset ds;
    struct cd_cache c;
    size_t i;
    int have_two = 0, arch;

    if (!m) return -1;
    if (cd_data_load(test_path, &ds, err, errlen) != 0) { cd_model_free(m); return -1; }
    cd_cache_alloc(&c, m);
    arch = (int)m->arch;
    have_two = cd_tw_split_query(arch) || cd_tw_sum_query(arch);

    printf("# model=%s arch=%d n=%zu n_opt=%d two_head=%d\n",
           model_path, arch, ds.n, m->d.n_opt, have_two);
    printf("label\tcorrect\tconf\tamargin\taent\tcommon\tlogit_hi\t"
           "attn_ent\tattn_max\tattn_pos\tsc_max\tsc_spread\t"
           "att_norm\tq_norm\tcos_qa\thdisagree\n");

    for (i = 0; i < ds.n; i++) {
        int no = cd_data_nopt(&ds, i), n, win = 0, lo = 0, lab, disagree = -1;
        const char *opts[CD_MAX_OPT];
        float zhi, zlo, conf;
        double qn = 0.0, an = 0.0, dot = 0.0, sc_max, sc_min;
        double attn_ent = 0.0, attn_max = 0.0, attn_pos = 0.0, attn_sum = 0.0;
        int k;
        const float *a, *q, *v;

        if (no > CD_MAX_OPT) no = CD_MAX_OPT;
        for (n = 0; n < no; n++) opts[n] = cd_data_opt(&ds, i, n);
        lab = cd_data_label(&ds, i);

        cd_forward_head(m, cd_data_ctx(&ds, i), opts, no, &c, CD_HEAD_FWD);
        for (n = 0; n < no; n++) {
            if (c.probs[n] > c.probs[win]) win = n;
            if (c.logits[n] < c.logits[lo]) lo = n;
        }
        zhi = c.logits[win]; zlo = c.logits[lo];
        conf = c.probs[win];

        /* attention profile of the WINNING candidate over context tokens */
        a = c.attn + (size_t)win * c.L;
        for (k = 0; k < (int)c.L; k++) {
            attn_sum += (double)a[k];
            if ((double)a[k] > attn_max) { attn_max = (double)a[k]; attn_pos = (double)k / (double)c.L; }
        }
        for (k = 0; k < (int)c.L; k++)
            if (a[k] > 0.0f) attn_ent -= (double)a[k] * log((double)a[k]);
        (void)attn_sum;

        /* pre-softmax per-context scores for the winner */
        sc_max = -1e30; sc_min = 1e30;
        {
            const float *srow = c.scores + (size_t)win * c.L;
            for (k = 0; k < (int)c.L; k++) {
                if ((double)srow[k] > sc_max) sc_max = (double)srow[k];
                if ((double)srow[k] < sc_min) sc_min = (double)srow[k];
            }
        }

        /* the logit is ||q||*||a||*cos; the norms and the angle are separately
         * observable, only their product is pinned to the logit */
        q = c.Q + (size_t)win * (size_t)m->d.R;
        v = c.attended + (size_t)win * (size_t)m->d.R;
        for (k = 0; k < (int)m->d.R; k++) {
            qn += (double)q[k] * (double)q[k];
            an += (double)v[k] * (double)v[k];
            dot += (double)q[k] * (double)v[k];
        }
        qn = sqrt(qn); an = sqrt(an);
        if (qn > 0.0 && an > 0.0) dot /= (qn * an);

        if (have_two) {
            int win2 = 0, n2;
            cd_forward_head(m, cd_data_ctx(&ds, i), opts, no, &c, CD_HEAD_INV);
            for (n2 = 0; n2 < no; n2++) if (c.logits[n2] > c.logits[win2]) win2 = n2;
            disagree = (win2 != win) ? 1 : 0;
            /* restore the forward cache state for the columns already read */
            cd_forward_head(m, cd_data_ctx(&ds, i), opts, no, &c, CD_HEAD_FWD);
        }

        printf("%d\t%d\t%.6f\t%.6f\t%.6f\t%.6f\t%.6f\t%.6f\t%.6f\t%.6f\t"
               "%.6f\t%.6f\t%.6f\t%.6f\t%.6f\t%d\n",
               lab, (win == lab) ? 1 : 0, (double)conf, fabs((double)zhi - (double)zlo),
               sig_entropy(c.probs, no), 0.5 * ((double)zhi + (double)zlo), (double)zhi,
               attn_ent, attn_max, attn_pos, sc_max, sc_max - sc_min,
               an, qn, dot, disagree);
    }
    cd_cache_free(&c);
    cd_data_free(&ds);
    cd_model_free(m);
    return 0;
}

int cd_eval(const char *model_path, const char *test_path, char *err, size_t errlen)
{ return cd_eval_impl(model_path, test_path, CD_HEAD_FWD, err, errlen); }

int cd_eval_head(const char *model_path, const char *test_path, int head,
                 char *err, size_t errlen)
{ return cd_eval_impl(model_path, test_path, head, err, errlen); }

/* Stage 0.17: per-example calibration record dump.
 *
 * Deliberately a SEPARATE path from cd_eval_impl. It runs the existing summary
 * first, untouched, and only then does its own forward pass to write the record.
 * That makes "the default eval output is byte-identical" true by construction
 * rather than by careful editing of shared code.
 *
 * Doubles are written with %.17g so every value round-trips exactly and the
 * aggregates in calib_metrics.py are recomputable from the record rather than
 * taken on trust. No strings are emitted, so no JSON escaping is involved. */
int cd_eval_dump(const char *model_path, const char *test_path, int head,
                 const char *dump_path, char *err, size_t errlen) {
    struct cd_model *m = cd_load(model_path, err, errlen);
    struct cd_dataset ds;
    struct cd_cache c;
    struct cd_exres *ex;
    FILE *f;
    size_t i;
    int n, rc = 0;

    if (!m) { snprintf(err, errlen, "cannot read model %s", model_path); return -1; }
    if (cd_data_load(test_path, &ds, err, errlen) != 0) { cd_model_free(m); return -1; }
    cd_cache_alloc(&c, m);
    ex = cd_forward_all(m, &ds, 0, ds.n, head, 0, &c);
    if (!ex) { cd_cache_free(&c); cd_model_free(m); cd_data_free(&ds);
               snprintf(err, errlen, "forward pass failed"); return -1; }
    f = fopen(dump_path, "w");
    if (!f) { free(ex); cd_cache_free(&c); cd_model_free(m); cd_data_free(&ds);
              snprintf(err, errlen, "cannot write dump %s", dump_path); return -1; }
    for (i = 0; i < ds.n; i++) {
        fprintf(f, "{\"example_id\": %zu, \"true_label\": %d, "
                   "\"predicted_label\": %d, \"p_true\": %.17g, "
                   "\"p_predicted\": %.17g, \"correct\": %d, \"probs\": [",
                i, ex[i].true_label, ex[i].argmax, ex[i].p_true,
                (double)ex[i].conf, ex[i].correct);
        for (n = 0; n < ex[i].n_opt; n++)
            fprintf(f, "%s%.17g", n ? ", " : "", ex[i].probs[n]);
        fprintf(f, "]}\n");
    }
    if (fclose(f) != 0) rc = -1;
    free(ex);
    cd_cache_free(&c);
    cd_model_free(m);
    cd_data_free(&ds);
    return rc;
}

/* ---- model I/O (candor own container, versioned) -------------------- */

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

/* 3-bit block quantization. Per 32-weight block: float32 min, float32 scale, then 32
 * indices of 3 bits packed little-endian-bit-first (index i occupies bits 3i..3i+2,
 * spanning two bytes where it must). Levels are uniform: x_hat = min + idx*scale with
 * scale = (max-min)/7, so worst-case error is half a step = (max-min)/14. A constant
 * block stores scale 0 and all-zero indices and dequantizes exactly.
 * Compute ALWAYS stays float32: this is storage quantization only, expanded back to
 * full precision on load. There is no float16/3-bit arithmetic anywhere. */
/* Single-block codec. q3_write/q3_read below are thin file loops over these two, so
 * tests/test_quant.c exercises the exact bit layout that ships in v2 files. */
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

static int q3_write(FILE *f, const float *v, size_t n) {
    size_t i;
    for (i = 0; i < n; i += CD_Q3_BLOCK) {
        float mn, scale;
        unsigned char packed[(CD_Q3_BLOCK * 3) / 8];
        size_t m = (n - i < CD_Q3_BLOCK) ? n - i : CD_Q3_BLOCK;
        if (cd_q3_block_encode(v + i, m, &mn, &scale, packed) != 0) return -1;
        if (fwrite(&mn, sizeof mn, 1, f) != 1 ||
            fwrite(&scale, sizeof scale, 1, f) != 1 ||
            fwrite(packed, sizeof packed, 1, f) != 1)
            return -1;
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

/* 8-bit symmetric block quantization. Per 32-weight block: float32 scale, then 32
 * int8 values, level = round(x / scale) with scale = max|x| / 127. A zero block stores
 * scale 0 and decodes exactly. Worst-case error is half a step: max|x| / 254. Like the
 * 3-bit path this is STORAGE ONLY, promoted to float32 on load; there is no int8
 * arithmetic anywhere. Effective cost 36 bytes per 32 weights = 9 bits/weight, sitting
 * between v1 float32 (32 bits) and v2 3-bit (5 bits with headers): 41,280 params go
 * 165,160 -> 46,480 bytes. Container version 3. */
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

static int q8_write(FILE *f, const float *v, size_t n) {
    size_t i;
    for (i = 0; i < n; i += CD_Q3_BLOCK) {
        float scale;
        signed char vals[CD_Q3_BLOCK];
        size_t m = (n - i < CD_Q3_BLOCK) ? n - i : CD_Q3_BLOCK;
        if (cd_q8_block_encode(v + i, m, &scale, vals) != 0) return -1;
        if (fwrite(&scale, sizeof scale, 1, f) != 1 ||
            fwrite(vals, sizeof vals, 1, f) != 1)
            return -1;
    }
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

int cd_save_q8(const char *path, const struct cd_model *m, char *err, size_t err_len) {
    FILE *f = fopen(path, "wb");
    uint32_t hdr[10];
    const float *p[CD_SAVE_MAX_TENSORS];
    size_t n[CD_SAVE_MAX_TENSORS];
    int k, t;
    if (!f) { snprintf(err, err_len, "cannot write %s", path); return -1; }
    hdr[0] = CD_MAGIC;
    hdr[1] = m->d.W; hdr[2] = m->d.R; hdr[3] = m->d.C; hdr[4] = m->d.O;
    hdr[5] = (uint32_t)m->d.n_opt; hdr[6] = (uint32_t)m->arch; hdr[7] = 3u;
    hdr[8] = (uint32_t)m->n_emb; hdr[9] = (uint32_t)m->n_qkv;
    if (fwrite(hdr, sizeof hdr, 1, f) != 1) {
        snprintf(err, err_len, "cannot write header %s", path);
        fclose(f); return -1;
    }
    k = save_tensor_list(m, p, n);
    for (t = 0; t < k; t++) {
        if (q8_write(f, p[t], n[t]) != 0) {
            snprintf(err, err_len, "cannot write tensor %d %s", t, path);
            fclose(f); return -1;
        }
    }
    fclose(f);
    return 0;
}

/* Generic N-bit block codec for the 2-bit (v5) and 4-bit (v4) containers.
 *
 * Same contract as the 3-bit and 8-bit paths: per-32-weight blocks of {float32 min,
 * float32 scale, packed indices}, uniform levels, worst-case error half a step =
 * (max-min)/(2*(2^bits-1)). Constant blocks store scale 0 and decode exactly.
 * Packing is little-endian-bit-first: index i of an N-bit field occupies bits
 * N*i..N*i+N-1, spanning bytes with the low bits first and masked spill bits --
 * the unmasked-spill bug that tests/test_quant.c caught in the 3-bit decoder
 * cannot recur here because the mask is computed from the width, not hardcoded.
 * Storage only, promoted to float32 on load; no sub-float arithmetic anywhere. */
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

/* Bit-exact file loops shared by v4 (4-bit, 16 packed bytes/block) and v5
 * (2-bit, 8 packed bytes/block). Sizes: 32 weights -> 8+16=24 bytes (v4),
 * 8+8=16 bytes (v5). */
static int qn_write(FILE *f, const float *v, size_t n, int bits) {
    size_t i;
    size_t bytes = (CD_Q3_BLOCK * (size_t)bits + 7) / 8;
    for (i = 0; i < n; i += CD_Q3_BLOCK) {
        float mn, scale;
        unsigned char packed[(CD_Q3_BLOCK * 8 + 7) / 8];
        size_t m = (n - i < CD_Q3_BLOCK) ? n - i : CD_Q3_BLOCK;
        if (cd_qn_block_encode(v + i, m, bits, &mn, &scale, packed) != 0) return -1;
        if (fwrite(&mn, sizeof mn, 1, f) != 1 ||
            fwrite(&scale, sizeof scale, 1, f) != 1 ||
            fwrite(packed, 1, bytes, f) != bytes)
            return -1;
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

static int cd_save_qn(const char *path, const struct cd_model *m, int bits,
                      unsigned version, char *err, size_t err_len) {
    FILE *f = fopen(path, "wb");
    uint32_t hdr[10];
    const float *p[CD_SAVE_MAX_TENSORS];
    size_t n[CD_SAVE_MAX_TENSORS];
    int k, t;
    if (!f) { snprintf(err, err_len, "cannot write %s", path); return -1; }
    hdr[0] = CD_MAGIC;
    hdr[1] = m->d.W; hdr[2] = m->d.R; hdr[3] = m->d.C; hdr[4] = m->d.O;
    hdr[5] = (uint32_t)m->d.n_opt; hdr[6] = (uint32_t)m->arch; hdr[7] = version;
    hdr[8] = (uint32_t)m->n_emb; hdr[9] = (uint32_t)m->n_qkv;
    if (fwrite(hdr, sizeof hdr, 1, f) != 1) {
        snprintf(err, err_len, "cannot write header %s", path);
        fclose(f); return -1;
    }
    k = save_tensor_list(m, p, n);
    for (t = 0; t < k; t++) {
        if (qn_write(f, p[t], n[t], bits) != 0) {
            snprintf(err, err_len, "cannot write tensor %d %s", t, path);
            fclose(f); return -1;
        }
    }
    fclose(f);
    return 0;
}

int cd_save_q4(const char *path, const struct cd_model *m, char *err, size_t err_len) {
    return cd_save_qn(path, m, 4, 4u, err, err_len);
}

int cd_save_q2(const char *path, const struct cd_model *m, char *err, size_t err_len) {
    return cd_save_qn(path, m, 2, 5u, err, err_len);
}

int cd_save_q3(const char *path, const struct cd_model *m, char *err, size_t err_len) {
    FILE *f = fopen(path, "wb");
    uint32_t hdr[10];
    const float *p[CD_SAVE_MAX_TENSORS];
    size_t n[CD_SAVE_MAX_TENSORS];
    int k, t;
    if (!f) { snprintf(err, err_len, "cannot write %s", path); return -1; }
    hdr[0] = CD_MAGIC;
    hdr[1] = m->d.W; hdr[2] = m->d.R; hdr[3] = m->d.C; hdr[4] = m->d.O;
    hdr[5] = (uint32_t)m->d.n_opt; hdr[6] = (uint32_t)m->arch; hdr[7] = 2u;
    hdr[8] = (uint32_t)m->n_emb; hdr[9] = (uint32_t)m->n_qkv;
    if (fwrite(hdr, sizeof hdr, 1, f) != 1) {
        snprintf(err, err_len, "cannot write header %s", path);
        fclose(f); return -1;
    }
    k = save_tensor_list(m, p, n);
    for (t = 0; t < k; t++) {
        if (q3_write(f, p[t], n[t]) != 0) {
            snprintf(err, err_len, "cannot write tensor %d %s", t, path);
            fclose(f); return -1;
        }
    }
    fclose(f);
    return 0;
}

int cd_save(const char *path, const struct cd_model *m, char *err, size_t err_len) {
    FILE *f = fopen(path, "wb");
    uint32_t hdr[10];
    if (!f) { snprintf(err, err_len, "cannot write %s", path); return -1; }
    hdr[0] = CD_MAGIC;
    hdr[1] = m->d.W; hdr[2] = m->d.R; hdr[3] = m->d.C; hdr[4] = m->d.O;
    hdr[5] = (uint32_t)m->d.n_opt; hdr[6] = (uint32_t)m->arch; hdr[7] = 1u;
    hdr[8] = (uint32_t)m->n_emb; hdr[9] = (uint32_t)m->n_qkv;
    fwrite(hdr, sizeof hdr, 1, f);
    fwrite(m->emb, sizeof(float), m->n_emb, f);
    fwrite(m->pos, sizeof(float), m->n_pos, f);
    fwrite(m->lncw, sizeof(float), m->d.W, f); fwrite(m->lncb, sizeof(float), m->d.W, f);
    fwrite(m->lonw, sizeof(float), m->d.W, f); fwrite(m->lonb, sizeof(float), m->d.W, f);
    fwrite(m->Wq, sizeof(float), m->n_qkv, f);
    fwrite(m->Wk, sizeof(float), m->n_qkv, f);
    fwrite(m->Wv, sizeof(float), m->n_qkv, f);
    if (m->arch == CD_ARCH_INTERACT) {
        fwrite(m->Wo, sizeof(float), m->n_qkv, f);
        fwrite(m->gc, sizeof(float), m->d.W, f);
        fwrite(m->gb, sizeof(float), m->d.W, f);
    }
    if (cd_arch_is_twohead((int)m->arch)) fwrite(m->Wq2, sizeof(float), m->n_qkv, f);
    if (m->Wk2) fwrite(m->Wk2, sizeof(float), m->n_qkv, f);
    if (m->Vv2) fwrite(m->Vv2, sizeof(float), m->n_qkv, f);
    if (m->arch == CD_ARCH_XBID) {
        fwrite(m->n1g, sizeof(float), m->d.R, f); fwrite(m->n1b, sizeof(float), m->d.R, f);
        fwrite(m->n2g, sizeof(float), m->d.R, f); fwrite(m->n2b, sizeof(float), m->d.R, f);
        fwrite(m->n3g, sizeof(float), m->d.R, f); fwrite(m->n3b, sizeof(float), m->d.R, f);
    }
    fclose(f);
    return 0;
}

struct cd_model *cd_load(const char *path, char *err, size_t err_len) {
    FILE *f = fopen(path, "rb");
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

/* ---- test hooks -------------------------------------------------------
 * Exposed only so the regression suite can address the gradient internals
 * directly. The contiguity bug this guards against lived here: the six XBID
 * norm gradients are six separate R-float allocations, and treating them as
 * one contiguous block overran the heap by 5x.
 */
struct cd_grad *cd_test_grad_new(const struct cd_model *m) {
    struct cd_grad *g = calloc(1, sizeof *g);
    if (!g) return NULL;
    grad_alloc(g, m);
    return g;
}
void cd_test_grad_del(struct cd_grad *g) { grad_free(g); free(g); }

void cd_test_grad_alloc(struct cd_grad *g, const struct cd_model *m)
{ grad_alloc(g, m); }
void cd_test_grad_zero(struct cd_grad *g, const struct cd_model *m)
{ grad_zero(g, m->d.W, m->d.R); }
void cd_test_grad_free(struct cd_grad *g) { grad_free(g); }
size_t cd_test_cache_ctx_len(const struct cd_cache *c) { return c->L; }

/* Apply exactly one real optimiser step, through the same adam_step the
 * trainer uses, to whatever gradient is currently in `g`. Lets a test inject
 * a KNOWN non-zero gradient and assert the corresponding parameter moved. */
void cd_test_optimizer_step(struct cd_model *m, struct cd_grad *g, double lr) {
    struct cd_adam a;
    if (adam_init(&a, m, m->n_emb, m->n_pos, (size_t)m->d.W, m->n_qkv) != 0)
        return;
    /* adam_step owns the step counter, so this is genuinely one step through
     * the same code the trainer uses. */
    adam_step(&a, m, g, lr, 0.9, 0.999, 1e-8, 0.0, 0);
    adam_release(&a);
}

int cd_test_train_steps(const char *train, const char *val, const char *out,
                        enum cd_arch arch, int epochs, int batch,
                        uint64_t seed, char *err, size_t err_len) {
    struct cd_cfg cfg;
    memset(&cfg, 0, sizeof cfg);
    cfg.d.W = 64; cfg.d.R = 64; cfg.d.C = 192; cfg.d.O = 32; cfg.d.n_opt = 4;
    cfg.arch = arch;
    cfg.mode = CD_MODE_LEGACY;
    cfg.epochs = epochs; cfg.batch_size = batch;
    cfg.lr = 0.002; cfg.wd = 1e-4; cfg.seed = seed;
    return cd_train(&cfg, train, val, out, NULL, err, err_len);
}

/* Runs one real backward on a single example and hands back a copy of the
 * position gradient. Exposed so the regression suite can assert that a short
 * example leaves nothing beyond its own length, without the gradient struct
 * having to leave this translation unit. Caller frees *out_pos. */
int cd_test_backward_capture(const struct cd_model *m, const char *ctx,
                             const char *const *opts, int n, int label,
                             struct cd_cache *c, float **out_pos, size_t *out_n,
                             char *err, size_t errlen) {
    struct cd_grad g;
    int rc;
    if (cd_forward(m, ctx, opts, n, c) != 0) {
        snprintf(err, errlen, "forward failed"); return -1;
    }
    grad_alloc(&g, m);
    grad_zero(&g, m->d.W, m->d.R);
    rc = cd_backward(m, ctx, opts, n, label, c, &g, 0, CD_HEAD_FWD, NULL);
    *out_n = m->n_pos;
    *out_pos = malloc(m->n_pos * sizeof(float));
    if (!*out_pos) { grad_free(&g); return -1; }
    memcpy(*out_pos, g.pos, m->n_pos * sizeof(float));
    grad_free(&g);
    return rc;
}

int cd_test_forward_backward_dlogits(struct cd_model *m, const char *ctx,
                                     const char *const *opts, int n, int label,
                                     struct cd_cache *c, struct cd_grad *g,
                                     const float *dlogits,
                                     char *err, size_t errlen) {
    if (cd_forward(m, ctx, opts, n, c) != 0) {
        snprintf(err, errlen, "forward failed"); return -1;
    }
    grad_zero(g, m->d.W, m->d.R);
    return cd_backward(m, ctx, opts, n, label, c, g, 0, CD_HEAD_FWD, dlogits);
}

const float *cd_test_grad_wq(const struct cd_grad *g, size_t *out_n) {
    if (out_n) *out_n = g->n_qkv;
    return g->Wq;
}

const float *cd_test_grad_pos(const struct cd_grad *g, size_t *out_n) {
    if (out_n) *out_n = g->n_pos;
    return g->pos;
}

size_t *cd_test_grad_tensor_sizes(const struct cd_grad *g, const struct cd_model *m) {
    /* Delegates to the model's own sizing so the two can never disagree. The
     * gradient tensors have the same shapes and the same conditional presence
     * as the model's, which is exactly what cd_model_tensor_sizes encodes. */
    (void)g;
    return cd_model_tensor_sizes(m);
}

const float **cd_test_grad_tensors(const struct cd_grad *g, size_t *out_count) {
    const float **v = calloc(32, sizeof *v);
    size_t n = 0;
    if (!v) { if (out_count) *out_count = 0; return NULL; }
    v[n++] = g->emb; v[n++] = g->pos;
    v[n++] = g->lncw; v[n++] = g->lncb; v[n++] = g->lonw; v[n++] = g->lonb;
    v[n++] = g->Wq; v[n++] = g->Wk; v[n++] = g->Wv;
    if (g->Wq2) v[n++] = g->Wq2;
    if (g->Wk2) v[n++] = g->Wk2;
    if (g->Vv2) v[n++] = g->Vv2;
    if (g->Wo)  v[n++] = g->Wo;
    if (g->gc)  v[n++] = g->gc;
    if (g->gb)  v[n++] = g->gb;
    v[n++] = g->n1g; v[n++] = g->n1b; v[n++] = g->n2g; v[n++] = g->n2b;
    v[n++] = g->n3g; v[n++] = g->n3b;
    v[n] = NULL;
    if (out_count) *out_count = n;
    return v;
}

/* Fills each of the six XBID norm-gradient buffers with `v`, then zeroes them
 * via the real grad_zero, and reports how many entries survived. Test T4 needs
 * this because the struct is deliberately private to this file. */
int cd_test_norm_grad_stress(const struct cd_model *m, float v,
                             long *out_distinct, long *out_stragglers) {
    struct cd_grad g;
    float *p[6];
    int i, j, distinct = 1, stragglers = 0;
    grad_alloc(&g, m);
    for (i = 0; i < 6; i++) {
        cd_test_grad_norm_ptr(&g, i, &p[i]);
        if (!p[i]) distinct = 0;
        for (j = 0; j < (int)m->d.R; j++) p[i][j] = v;
    }
    for (i = 0; i < 6; i++)
        for (j = 0; j < 6; j++)
            if (i != j && p[i] == p[j]) distinct = 0;
    cd_test_grad_zero(&g, m);
    for (i = 0; i < 6; i++)
        for (j = 0; j < (int)m->d.R; j++)
            if (p[i][j] != 0.0f) stragglers++;
    *out_distinct = distinct;
    *out_stragglers = stragglers;
    grad_free(&g);
    return 0;
}

/* ---- ask: human inspection ------------------------------------------
 *
 * Diagnostic only. It loads a model, runs the ordinary forward path, and
 * reports what the model believes. It never trains, mutates, reorders or
 * re-normalises anything, and it consults no dataset label. Oracle routing can
 * be requested explicitly with --head, which is the only way a head is chosen.
 */
static void ask_show(const struct cd_model *m, const char *input,
                     char *const *cands, int n, int head) {
    struct cd_cache c;
    const char *opts[CD_MAX_OPT];
    int i, best = 0;
    double sum_p = 0.0;
    /* Cap first, then size the cache by what is actually asked. The old code copied at
     * most CD_MAX_OPT options but passed the uncapped n to forward (out-of-bounds read past
     * the copied array) into a cache sized by the model's training n_opt (heap overrun past
     * that). Both failure modes die here: at most 64 candidates, cache fits all of them. */
    if (n > CD_MAX_OPT) n = CD_MAX_OPT;
    cd_cache_alloc_n(&c, m, (size_t)n);
    for (i = 0; i < n; i++) opts[i] = cands[i];
    if (cd_forward_head(m, input, opts, n, &c, head) != 0) {
        printf("  (forward failed: empty or over-long input)\n");
        cd_cache_free(&c);
        return;
    }
    printf("  head                : %s\n", head == CD_HEAD_INV ? "inverse" : "forward");
    printf("  model input         : %s\n", input);
    printf("  %-3s %-52s %9s %9s\n", "#", "candidate", "logit", "prob");
    for (i = 0; i < n; i++) {
        printf("  %-3d %-52.52s %9.4f %8.4f%%%s\n", i, opts[i],
               c.logits[i], c.probs[i] * 100.0, i == best ? "  <- argmax" : "");
        sum_p += c.probs[i];
        if (c.probs[i] > c.probs[best]) best = i;
    }
    best = 0;
    for (i = 1; i < n; i++) if (c.probs[i] > c.probs[best]) best = i;
    printf("  %-3s %-52s %9s %8.4f%%\n", "", "SELECTED", "", c.probs[best] * 100.0);
    printf("  answer              : %s\n", opts[best]);
    printf("  confidence          : %.2f%%   (chance %.2f%%, margin over runner-up %+.2f pts)\n",
           c.probs[best] * 100.0, 100.0 / n, (c.probs[best] - c.probs[best == 0 ? 1 : 0]) * 100.0);
    printf("  prob sum            : %.6f%s\n", sum_p,
           head == CD_HEAD_INV ? "   [oracle: inverse head chosen explicitly]" : "");
    cd_cache_free(&c);
}

int cd_ask(const char *model_path, const char *context, const char *question,
           char *const *candidates, int n_candidates, int head_mask,
           char *err, size_t err_len) {
    struct cd_model *m = cd_load(model_path, err, err_len);
    char *input;
    size_t need;
    if (!m) return -1;
    if (!context || !*context || n_candidates < 1) {
        snprintf(err, err_len, "ask needs --context and at least one --candidate");
        cd_model_free(m);
        return -1;
    }
    need = strlen(context) + (question ? strlen(question) : 0) + 2;
    input = malloc(need);
    if (!input) { cd_model_free(m); return -1; }
    if (question && *question) snprintf(input, need, "%s %s", context, question);
    else snprintf(input, need, "%s", context);

    if (head_mask == 3) {
        printf("\n== both heads (diagnostic; the model was not told which to use) ==\n");
        ask_show(m, input, candidates, n_candidates, CD_HEAD_FWD);
        printf("\n");
        ask_show(m, input, candidates, n_candidates, CD_HEAD_INV);
    } else {
        printf("\n");
        ask_show(m, input, candidates, n_candidates, head_mask);
    }
    free(input);
    cd_model_free(m);
    return 0;
}

/* ==========================================================================
 * Stage 0.18 contextual-bandit post-training. PREPARED, NOT RUN.
 *
 * RL execution is blocked by operator instruction. Nothing calls this yet. It is
 * written and tested so that the moment execution is unblocked the loop is a
 * known quantity rather than an untested path.
 *
 * The objective, the sampler and the moving baseline live in cd_bandit.c as pure
 * functions, so they can be finite-difference checked in isolation. This loop
 * exists only to wire them to the model, and it does so through the ORDINARY
 * optimiser path on purpose: grad_zero, grad_scale, the hard-coded clip 1.0 and
 * adam_step, exactly as supervised training uses. adam_step ends with
 * cd_model_refresh_transposes, and the topology audit established that mutating a
 * projection matrix without that refresh leaves the forward pass reading stale
 * weights, so there is deliberately no way to update a weight here except through
 * adam_step.
 *
 * The KL reference is precomputed ONCE from the starting checkpoint and never
 * updated. That is what makes it an anchor: at step zero the policy IS the
 * reference and the KL term is exactly zero, so alpha governs how far the policy
 * may drift rather than charging a standing penalty.
 * ========================================================================== */

/* Exploratory mechanisms. Both default to the Stage 0.18 behaviour exactly, so
 * a run with mask_spec="all" and wrong_pen=0.0 is bit-identical to the sealed
 * primary arms; that equivalence is asserted by the test suite.
 *
 *   mask_spec  which tensor groups may move. "all" is unrestricted. Otherwise a
 *              comma-separated subset of: embed, pos, lnorm, xbidnorm, wqkv, head
 *              Anything not named is zeroed in the gradient BEFORE the clip
 *              norm is taken, so a restricted subspace still receives a
 *              sensibly scaled step rather than a fraction of an unrestricted
 *              one. This is a hypothesis probe about WHERE calibration can live,
 *              not a claim about which tensors are responsible.
 *
 *   wrong_pen  coefficient on a penalty applied ONLY to the probability the
 *              model currently assigns to its own wrong answer on a training
 *              example. This separates "be less confident where you are wrong"
 *              from "be less confident everywhere", which is exactly the
 *              distinction blanket entropy cannot express. Uses training labels
 *              only. 0.0 disables it.
 */
int cd_bandit_posttrain(const char *model_in, const char *data_path,
                        const char *model_out, const char *log_path,
                        double beta, double alpha, int epochs, int batch,
                        double lr, unsigned long long seed,
                        const char *mask_spec, double wrong_pen,
                        char *err, size_t errlen) {
    struct cd_model *m = NULL, *src = NULL;
    struct cd_dataset ds;
    struct cd_cache c;
    struct cd_grad g;
    struct cd_adam a;
    struct bandit_rng rng;
    double *qref = NULL;            /* frozen reference probs, CD_MAX_OPT per row */
    int *qopt = NULL;                /* candidates per row                       */
    size_t *order = NULL;
    FILE *lg = NULL;   /* NOT named log: that shadows libm log() */
    char errbuf[512] = {0};
    size_t i, n, ep, nstep = 0;
    int t, k, rc = -1;
    double baseline = bandit_baseline_init();
    double sum_policy = 0, sum_ent = 0, sum_kl = 0, sum_rew = 0, sum_adv = 0;
    long n_seen = 0, step = 0;
    int have_ds = 0, have_c = 0, have_g = 0, have_a = 0;
    int mask_embed, mask_pos, mask_lnorm, mask_xbidnorm, mask_wqkv, mask_head;
    double sum_wrongpen = 0;

    memset(&g, 0, sizeof g);
    memset(&a, 0, sizeof a);
    memset(&c, 0, sizeof c);
    memset(&ds, 0, sizeof ds);
    /* Exploratory switches. "all" and 0.0 reproduce the sealed primary arms. */
    mask_embed = mask_pos = mask_lnorm = mask_xbidnorm = mask_wqkv = mask_head = 1;
    if (mask_spec && strcmp(mask_spec, "all") != 0) {
        mask_embed = mask_pos = mask_lnorm = mask_xbidnorm = mask_wqkv = mask_head = 0;
        {   const char *q = mask_spec;
            for (;;) {
                if (!strncmp(q, "embed", 5))   { mask_embed = 1; q += 5; }
                else if (!strncmp(q, "pos", 3))    { mask_pos = 1; q += 3; }
                else if (!strncmp(q, "lnorm", 5))  { mask_lnorm = 1; q += 5; }
                else if (!strncmp(q, "xbidnorm", 8)) { mask_xbidnorm = 1; q += 8; }
                else if (!strncmp(q, "wqkv", 4))   { mask_wqkv = 1; q += 4; }
                else if (!strncmp(q, "head", 4))   { mask_head = 1; q += 4; }
                else break;
                if (*q == ',') q++; else break;
            }
            if (!mask_embed && !mask_pos && !mask_lnorm && !mask_xbidnorm
                && !mask_wqkv && !mask_head) {
                snprintf(err, errlen,
                         "mask_spec '%s' names no known tensor group", mask_spec);
                goto done;
            }
        }
    }

    /* The starting checkpoint is opened read-only and deep-copied. The file on
     * disk is never written, so the frozen Stage 0.18 reference model is
     * immutable by construction, not by convention. */
    src = cd_load(model_in, errbuf, sizeof errbuf);
    if (!src) { snprintf(err, errlen, "cannot read %s: %s", model_in, errbuf); goto done; }
    /* Allocate m with the SAME dims and arch as src, so every tensor exists,
     * then copy every tensor across. Deliberately NOT model_copy: T17 proves
     * model_copy is incomplete for XBID and twohead, so using it here would
     * silently start the post-training from a model with six untrained norm
     * tensors. The loop's starting point must be exactly the frozen checkpoint,
     * so the copy has to be complete. */
    m = malloc(sizeof *m);
    if (!m) { snprintf(err, errlen, "oom model"); goto done; }
    cd_model_init(m, &src->d, (enum cd_arch)src->arch, src->d.n_opt);
    {
        const float **tv = NULL;
        size_t *sz = cd_model_tensor_sizes(src);
        size_t ntv, z;
        if (!sz) { snprintf(err, errlen, "oom sizes"); goto done; }
        ntv = cd_model_trainable(m, &tv);
        if (ntv > 0) {
            const float **sv = NULL;
            size_t nsv = cd_model_trainable(src, &sv);
            for (z = 0; z < ntv && z < nsv; z++)
                memcpy((void *)tv[z], sv[z], sz[z] * sizeof(float));
            free(sv);
        }
        free(tv);
        free(sz);
        cd_model_refresh_transposes(m);
    }

    if (cd_data_load(data_path, &ds, errbuf, sizeof errbuf) != 0) {
        snprintf(err, errlen, "cannot read %s: %s", data_path, errbuf); goto done;
    }
    have_ds = 1;
    cd_cache_alloc(&c, m); have_c = 1;
    grad_alloc(&g, m);      have_g = 1;
    if (adam_init(&a, m, m->n_emb, m->n_pos, m->d.W, m->n_qkv) != 0) {
        snprintf(err, errlen, "oom adam"); goto done;
    }
    have_a = 1;

    n = ds.n;
    qref  = calloc(n * (size_t)CD_MAX_OPT, sizeof(double));
    qopt  = calloc(n, sizeof(int));
    order = calloc(n, sizeof(size_t));
    if (!qref || !qopt || !order) { snprintf(err, errlen, "oom refs"); goto done; }

    /* Precompute the frozen reference distribution ONCE, from the un-updated
     * policy. This is what makes the KL term an anchor: at step zero the policy
     * IS the reference, so KL is exactly zero and alpha restrains drift rather
     * than charging a standing penalty. */
    for (i = 0; i < n; i++) {
        const char *opts[CD_MAX_OPT];
        int no = cd_data_nopt(&ds, i);
        if (no > CD_MAX_OPT) no = CD_MAX_OPT;
        qopt[i] = no;
        for (k = 0; k < no; k++) opts[k] = cd_data_opt(&ds, i, k);
        cd_forward_head(m, cd_data_ctx(&ds, i), opts, no, &c, CD_HEAD_FWD);
        for (k = 0; k < no; k++) qref[i * (size_t)CD_MAX_OPT + k] = (double)c.probs[k];
        order[i] = i;
    }

    /* Deterministic order and deterministic actions, both from `seed`. A
     * paired-seed design is worthless if the action sequence is not
     * reproducible. */
    bandit_rng_seed(&rng, seed);
    for (i = n; i > 1; i--) {
        size_t x = (size_t)(bandit_rng_u64(&rng) % i);
        size_t tmp = order[i - 1];
        order[i - 1] = order[x];
        order[x] = tmp;
    }

    if (log_path) {
        lg = fopen(log_path, "w");
        if (!lg) { snprintf(err, errlen, "cannot write log %s", log_path); goto done; }
        fprintf(lg, "{\"stage\": \"0.18\", \"beta\": %.17g, \"alpha\": %.17g, "
                     "\"epochs\": %d, \"batch\": %d, \"lr\": %.17g, "
                     "\"seed\": %llu, \"rows\": %zu, \"steps\": [",
                beta, alpha, epochs, batch, lr, seed, n);
    }

    for (ep = 0; ep < (size_t)epochs; ep++) {
        for (i = 0; i < n; i += (size_t)batch) {
            size_t end = i + (size_t)batch;
            double norm = 0.0;
            int blen;
            if (end > n) end = n;
            grad_zero(&g, m->d.W, m->d.R);
            for (t = (int)i; t < (int)end; t++) {
                size_t idx = order[t];
                const char *opts[CD_MAX_OPT];
                double p[CD_MAX_OPT], d[CD_MAX_OPT];
                float dlf[CD_MAX_OPT];
                const double *q = &qref[idx * (size_t)CD_MAX_OPT];
                int no = qopt[idx], action, lab;
                double reward, adv, H, KL, pa;
                for (k = 0; k < no; k++) opts[k] = cd_data_opt(&ds, idx, k);
                cd_forward_head(m, cd_data_ctx(&ds, idx), opts, no, &c, CD_HEAD_FWD);
                for (k = 0; k < no; k++) p[k] = (double)c.probs[k];
                lab = cd_data_label(&ds, idx);
                action = bandit_sample(p, no, &rng);
                reward = (action == lab) ? 1.0 : 0.0;
                /* Advantage uses the baseline as it stood BEFORE this reward,
                 * which is the standard online estimator. */
                adv = reward - baseline;
                baseline = bandit_baseline_step(baseline, reward);
                H  = cd_bandit_entropy(p, no);
                KL = cd_bandit_kl(p, q, no);
                pa = p[action] > 0.0 ? p[action] : 1e-300;
                sum_policy += -adv * log(pa);
                sum_ent    += H;
                sum_kl     += KL;
                sum_rew    += reward;
                sum_adv    += adv;
                n_seen++;
                cd_bandit_dlogits(p, q, no, action, adv, beta, alpha, d);
                if (wrong_pen > 0.0) {
                    /* Penalise ONLY the logit the model currently puts on its
                     * own wrong answer. d(-gamma*p_pred)/dz_k =
                     * -gamma*p_k*(1{k=pred} - p_pred), which is zero whenever the
                     * model is already correct on this training example. This is
                     * the term blanket entropy cannot express: it is selective in
                     * the error, not in the confidence. */
                    int pred = 0;
                    double p_pred;
                    for (k = 1; k < no; k++) if (p[k] > p[pred]) pred = k;
                    p_pred = p[pred];
                    /* dlogits_in is dL/dz (the cross-entropy branch above sets
                     * dlogits[n] = p_n and subtracts 1 at the label). The loss
                     * term is +wrong_pen * p_pred, so its gradient contribution
                     * is +wrong_pen * dp_pred/dz_k, POSITIVE. The first
                     * implementation of this probe negated it, which maximised
                     * the predicted probability and produced an anti-penalty;
                     * those runs are recorded as invalid, not as a result. */
                    for (k = 0; k < no; k++)
                        d[k] += wrong_pen * p[k] * (((k == pred) ? 1.0 : 0.0) - p_pred);
                    sum_wrongpen += wrong_pen * p_pred;
                }
                for (k = 0; k < no; k++) dlf[k] = (float)d[k];
                if (cd_backward(m, cd_data_ctx(&ds, idx), opts, no, lab, &c, &g, 0,
                                CD_HEAD_FWD, dlf) != 0) {
                    snprintf(err, errlen, "backward failed at row %zu", idx);
                    goto done;
                }
            }
            blen = (int)(end - i);
            grad_scale(&g, m->d.W, m->d.R, 1.0f / (float)blen);
            /* Freeze disallowed tensor groups. Done BEFORE the clip norm so the
             * norm, and therefore the step scale, reflects only what may move.
             * With mask_spec="all" nothing is zeroed and this is a no-op, which
             * is what keeps the sealed primary arms bit-identical. */
            if (!mask_embed)  memset(g.emb, 0, g.n_emb * sizeof(float));
            if (!mask_pos)    memset(g.pos, 0, g.n_pos * sizeof(float));
            if (!mask_lnorm)  { memset(g.lncw, 0, m->d.W * sizeof(float));
                                memset(g.lncb, 0, m->d.W * sizeof(float));
                                memset(g.lonw, 0, m->d.W * sizeof(float));
                                memset(g.lonb, 0, m->d.W * sizeof(float)); }
            if (!mask_wqkv)   { memset(g.Wq, 0, g.n_qkv * sizeof(float));
                                memset(g.Wk, 0, g.n_qkv * sizeof(float));
                                memset(g.Wv, 0, g.n_qkv * sizeof(float)); }
            if (!mask_xbidnorm) { if (g.n1g) memset(g.n1g, 0, m->d.R * sizeof(float));
                                  if (g.n1b) memset(g.n1b, 0, m->d.R * sizeof(float));
                                  if (g.n2g) memset(g.n2g, 0, m->d.R * sizeof(float));
                                  if (g.n2b) memset(g.n2b, 0, m->d.R * sizeof(float));
                                  if (g.n3g) memset(g.n3g, 0, m->d.R * sizeof(float));
                                  if (g.n3b) memset(g.n3b, 0, m->d.R * sizeof(float)); }
            if (!mask_head)  { if (g.Wq2) memset(g.Wq2, 0, g.n_qkv * sizeof(float));
                               if (g.Wk2) memset(g.Wk2, 0, g.n_qkv * sizeof(float));
                               if (g.Vv2) memset(g.Vv2, 0, g.n_qkv * sizeof(float));
                               if (g.Wo)  memset(g.Wo,  0, g.n_qkv * sizeof(float));
                               if (g.gc)  { memset(g.gc, 0, m->d.W * sizeof(float));
                                            memset(g.gb, 0, m->d.W * sizeof(float)); } }
            {   /* clip 1.0, hard-coded upstream, matched exactly */
                norm = grad_global_norm(m, &g);
                if (norm > 1.0 && norm > 0.0)
                    grad_scale(&g, m->d.W, m->d.R, (float)(1.0 / norm));
            }
            /* THE ONLY PATH THAT MAY TOUCH A WEIGHT. adam_step ends with
             * cd_model_refresh_transposes, which the topology audit showed is
             * mandatory: without it the AVX2 forward reads stale weights. */
            adam_step(&a, m, &g, lr, 0.9, 0.999, 1e-8, 0.0, 0);
            step++;
            if (lg) {
                fprintf(lg, "%s{\"step\": %ld, \"policy_loss\": %.17g, "
                             "\"entropy\": %.17g, \"kl\": %.17g, "
                             "\"reward_mean\": %.17g, \"advantage_mean\": %.17g, "
                             "\"grad_norm\": %.17g, \"wrong_pen_mean\": %.17g}",
                        nstep ? "," : "", step,
                        sum_policy / (double)n_seen, sum_ent / (double)n_seen,
                        sum_kl / (double)n_seen, sum_rew / (double)n_seen,
                        sum_adv / (double)n_seen, norm, sum_wrongpen / (double)n_seen);
                nstep++;
            }
        }
    }
    if (lg) fprintf(lg, "]}\n");

    if (model_out && cd_save(model_out, m, errbuf, sizeof errbuf) != 0) {
        snprintf(err, errlen, "cannot write %s: %s", model_out, errbuf);
        goto done;
    }
    rc = 0;
done:
    if (lg) fclose(lg);
    free(qref); free(qopt); free(order);
    if (have_a) adam_release(&a);
    if (have_g) grad_free(&g);
    if (have_c) cd_cache_free(&c);
    if (m) { cd_model_free(m); free(m); }
    cd_model_free(src);
    if (have_ds) cd_data_free(&ds);
    return rc;
}


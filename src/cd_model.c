/* cd_model.c — model, init, forward, and the shared maths.
 *
 * Init is a transcription of the upstream model init (see docs/PROVENANCE.md), including the RNG stream
 * and the order in which tensors consume it: emb, pos, Wq, Wk, Wv. The
 * interaction tensors are appended AFTER those five so a baseline model
 * consumes exactly the same random numbers as upstream.
 */
#include "candor.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ---- upstream RNG, transcribed (model.c:250-278) ---------------------- */
static uint64_t g_rng;
static int g_has_spare;
static double g_spare;

static uint64_t rng_next_u64(void) {
    uint64_t x = g_rng;
    x ^= x >> 12;
    x ^= x << 25;
    x ^= x >> 27;
    g_rng = x;
    return x * 0x2545F4914F6CDD1DULL;
}
static double rng_uniform01(void) {
    return (double)(rng_next_u64() >> 11) * (1.0 / 9007199254740992.0);
}
static double rng_normal01(void) {
    double z0, z1, u1, u2, r, th;
    if (g_has_spare) { g_has_spare = 0; return g_spare; }
    u1 = rng_uniform01();
    u2 = rng_uniform01();
    if (u1 < 1e-12) u1 = 1e-12;
    r = sqrt(-2.0 * log(u1));
    th = 2.0 * 3.14159265358979323846 * u2;
    z0 = r * cos(th);
    z1 = r * sin(th);
    g_spare = z1;
    g_has_spare = 1;
    return z0;
}

static void *xcalloc(size_t n) { return calloc(n ? n : 1, sizeof(float)); }

size_t cd_model_param_count(const struct cd_model *m) {
    size_t n = m->n_emb + m->n_pos + 4 * (size_t)m->d.W + 3 * m->n_qkv;
    if (m->arch == CD_ARCH_INTERACT) n += m->n_qkv + 2 * (size_t)m->d.W;
    if (cd_arch_is_twohead((int)m->arch)) n += m->n_qkv;   /* Wq2 */
    if (m->Wk2) n += m->n_qkv;                            /* Wk2 */
    if (m->Vv2) n += m->n_qkv;                            /* Vv2 */
    /* The six XBID norm tensors are R long each and ARE trained: the
     * parameter-topology audit measured every one of them as forward-live and
     * backward-written to the last entry, and adam_step updates them. They were
     * omitted here, so the reported count understated the trainable total by
     * 6*R = 384. The function is the single source of the reported number, so
     * the omission was the reported number being wrong, not a rounding
     * difference. Historical reports are NOT rewritten; see
     * docs/TOPOLOGY_AUDIT.md. */
    if (m->n1g) n += 6ull * (size_t)m->d.R;
    /* CD_ARCH_XBID deliberately adds nothing: it reuses ln_o/ln_c and shares
     * Wq/Wk/Wv across both rounds, so any change in behaviour is attributable
     * to computation rather than to capacity. */
    return n;
}


/* Length of every trainable tensor, in the order cd_model_trainable emits them.
 * Single source of truth: cd_model_param_count, the gradient-size hook and the
 * regression test all agree because they all call this. Getting these lengths
 * wrong by hand is what produced two out-of-bounds reads during the audit. */
size_t *cd_model_tensor_sizes(const struct cd_model *m) {
    size_t *v = calloc(32, sizeof *v);
    size_t n = 0;
    if (!v) return NULL;
    v[n++] = m->n_emb; v[n++] = m->n_pos;
    v[n++] = m->d.W; v[n++] = m->d.W; v[n++] = m->d.W; v[n++] = m->d.W;
    v[n++] = m->n_qkv; v[n++] = m->n_qkv; v[n++] = m->n_qkv;
    if (m->Wq2) v[n++] = m->n_qkv;
    if (m->Wk2) v[n++] = m->n_qkv;
    if (m->Vv2) v[n++] = m->n_qkv;
    if (m->Wo)  v[n++] = m->n_qkv;
    if (m->gc)  v[n++] = m->d.W;
    if (m->gb)  v[n++] = m->d.W;
    if (m->n1g) v[n++] = m->d.R;
    if (m->n1b) v[n++] = m->d.R;
    if (m->n2g) v[n++] = m->d.R;
    if (m->n2b) v[n++] = m->d.R;
    if (m->n3g) v[n++] = m->d.R;
    if (m->n3b) v[n++] = m->d.R;
    return v;
}

size_t cd_model_trainable(const struct cd_model *m, const float ***out) {
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

const float *cd_model_weight_at(const struct cd_model *m, int slot) {
    switch (slot) {
    case CD_W_Q:  return m->Wq;
    case CD_W_K:  return m->Wk;
    case CD_W_V:  return m->Wv;
    case CD_W_Q2: return m->Wq2;
    case CD_W_K2: return m->Wk2;
    case CD_W_V2: return m->Vv2;
    case CD_W_WO: return m->Wo;
    default: return NULL;
    }
}

void cd_model_refresh_transposes(struct cd_model *m) {
    int slot, r, w;
    const size_t R = m->d.R, W = m->d.W, n = R * W;
    for (slot = 0; slot < CD_W_COUNT; slot++) {
        const float *Wv = cd_model_weight_at(m, slot);
        if (!Wv || !m->Wt[slot]) continue;
        for (r = 0; r < (int)R; r++)
            for (w = 0; w < (int)W; w++)
                m->Wt[slot][(size_t)w * R + r] = (double)Wv[(size_t)r * W + w];
        (void)n;
    }
}

const double *cd_model_transpose(const struct cd_model *m, const float *W) {
    int slot;
    for (slot = 0; slot < CD_W_COUNT; slot++)
        if (W && cd_model_weight_at(m, slot) == W) return m->Wt[slot];
    return NULL;
}

int cd_arch_is_twohead(int arch) {
    return arch == CD_ARCH_TWOHEAD || arch == CD_ARCH_TW_SK ||
           arch == CD_ARCH_TW_SV  || arch == CD_ARCH_TW_SKV ||
           arch == CD_ARCH_SUMHEAD || arch == CD_ARCH_SUM_K ||
           arch == CD_ARCH_SUM_V  || arch == CD_ARCH_SUM_KV;
}

static int tw_splits_k(int arch) {
    return arch == CD_ARCH_TW_SK || arch == CD_ARCH_TW_SKV ||
           arch == CD_ARCH_SUM_K  || arch == CD_ARCH_SUM_KV;
}
static int tw_splits_v(int arch) {
    return arch == CD_ARCH_TW_SV || arch == CD_ARCH_TW_SKV ||
           arch == CD_ARCH_SUM_V  || arch == CD_ARCH_SUM_KV;
}
static int tw_sums(int arch) {
    return arch == CD_ARCH_SUM_K || arch == CD_ARCH_SUM_V || arch == CD_ARCH_SUM_KV;
}

void cd_tw_weights(const struct cd_model *m, int head,
                    const float **Wk, const float **Wv,
                    const float **Wk2, const float **Vv2,
                    int *summed, float *sk, float *sv) {
    const int a = (int)m->arch;
    const int sk_ = tw_splits_k(a), sv_ = tw_splits_v(a);
    *summed = tw_sums(a);
    *Wk2 = *Vv2 = NULL;
    if (*summed) {
        /* One context path: the two forms' maps are summed, and each partner
         * map receives a gradient identical to its partner's. */
        size_t i, n = m->n_qkv;
        for (i = 0; i < n; i++) {
            sk[i] = m->Wk[i] + (sk_ && m->Wk2 ? m->Wk2[i] : 0.0f);
            sv[i] = m->Wv[i] + (sv_ && m->Vv2 ? m->Vv2[i] : 0.0f);
        }
        *Wk = sk; *Wv = sv;
        *Wk2 = sk_ ? m->Wk2 : NULL; *Vv2 = sv_ ? m->Vv2 : NULL;
        return;
    }
    const int inv = (head == CD_HEAD_INV);
    *Wk = (sk_ && inv) ? m->Wk2 : m->Wk;
    *Wv = (sv_ && inv) ? m->Vv2 : m->Wv;
    *Wk2 = NULL; *Vv2 = NULL;
}

int cd_tw_split_query(int arch) {
    return arch == CD_ARCH_TWOHEAD || arch == CD_ARCH_TW_SK ||
           arch == CD_ARCH_TW_SV  || arch == CD_ARCH_TW_SKV;
}
int cd_tw_sum_context(int arch) {
    return arch == CD_ARCH_SUM_K || arch == CD_ARCH_SUM_V || arch == CD_ARCH_SUM_KV;
}
int cd_tw_sum_query(int arch) {
    return arch == CD_ARCH_SUMHEAD || arch == CD_ARCH_SUM_K ||
           arch == CD_ARCH_SUM_V  || arch == CD_ARCH_SUM_KV;
}

#include "cd_internal.h"

void cd_model_init(struct cd_model *m, const struct cd_dims *d,
                   enum cd_arch arch, uint64_t seed) {
    size_t i;
    double bound;
    memset(m, 0, sizeof *m);
    m->d = *d;
    m->arch = arch;
    m->n_emb = 257ull * d->W;
    m->n_pos = (size_t)d->C * d->W;
    m->n_qkv = (size_t)d->R * d->W;

    m->emb = xcalloc(m->n_emb);
    m->pos = xcalloc(m->n_pos);
    m->lncw = xcalloc(d->W); m->lncb = xcalloc(d->W);
    m->lonw = xcalloc(d->W); m->lonb = xcalloc(d->W);
    m->Wq = xcalloc(m->n_qkv); m->Wk = xcalloc(m->n_qkv); m->Wv = xcalloc(m->n_qkv);
    {   /* derived transposes, one slot per projection matrix */
        int slot;
        for (slot = 0; slot < CD_W_COUNT; slot++) {
            const size_t n = (size_t)d->R * (size_t)d->W;
            m->Wt[slot] = calloc(n, sizeof(double));
            if (!m->Wt[slot]) return;
        }
    }
    if (arch == CD_ARCH_INTERACT) {
        m->Wo = xcalloc(m->n_qkv);
        m->gc = xcalloc(d->W); m->gb = xcalloc(d->W);
    }

    g_rng = seed ? seed : 0x9e3779b97f4a7c15ull;
    if (g_rng == 0) g_rng = 0x9e3779b97f4a7c15ull;
    g_has_spare = 0;

    for (i = 0; i < m->n_emb; i++) m->emb[i] = (float)rng_normal01();
    for (i = 0; i < (size_t)d->W; i++) m->emb[i] = 0.0f;   /* padding row */
    for (i = 0; i < m->n_pos; i++) m->pos[i] = (float)rng_normal01();
    for (i = 0; i < (size_t)d->W; i++) {
        m->lncw[i] = 1.0f; m->lncb[i] = 0.0f;
        m->lonw[i] = 1.0f; m->lonb[i] = 0.0f;
    }
    bound = 1.0 / sqrt((double)d->W);
    for (i = 0; i < m->n_qkv; i++)
        m->Wq[i] = (float)(rng_uniform01() * 2.0 * bound - bound);
    for (i = 0; i < m->n_qkv; i++)
        m->Wk[i] = (float)(rng_uniform01() * 2.0 * bound - bound);
    for (i = 0; i < m->n_qkv; i++)
        m->Wv[i] = (float)(rng_uniform01() * 2.0 * bound - bound);
    /* Interaction tensors consume the stream only AFTER the five upstream
     * ones, so a baseline run is unaffected by their presence. */
    if (arch == CD_ARCH_INTERACT) {
        for (i = 0; i < m->n_qkv; i++)
            m->Wo[i] = (float)(rng_uniform01() * 2.0 * bound - bound);
        for (i = 0; i < (size_t)d->W; i++) {
            m->gc[i] = 1.0f + 0.1f * (float)rng_normal01();
            m->gb[i] = 0.1f * (float)rng_normal01();
        }
    }

    /* Two-head: allocate then initialise the extra option->query map. Order
     * matters, the init writes through the pointer. Drawn from the same
     * distribution as Wq and AFTER the five upstream tensors, so the two heads
     * start statistically identical and a baseline draw stays bit-identical. */
    if (cd_arch_is_twohead(arch)) {
        m->Wq2 = calloc(m->n_qkv, sizeof(float));
        if (tw_splits_k(arch)) m->Wk2 = calloc(m->n_qkv, sizeof(float));
        if (tw_splits_v(arch)) m->Vv2 = calloc(m->n_qkv, sizeof(float));
        if (m->Wq2) {
            for (i = 0; i < m->n_qkv; i++)
                m->Wq2[i] = (float)(rng_uniform01() * 2.0 * bound - bound);
        }
        if (m->Wk2) {
            for (i = 0; i < m->n_qkv; i++)
                m->Wk2[i] = (float)(rng_uniform01() * 2.0 * bound - bound);
        }
        if (m->Vv2) {
            for (i = 0; i < m->n_qkv; i++)
                m->Vv2[i] = (float)(rng_uniform01() * 2.0 * bound - bound);
        }
    }

    /* Every projection matrix now exists, so the derived transposes can be
     * built. This is data movement only and changes no arithmetic. */
    cd_model_refresh_transposes(m);

    /* XBID stream norms: gamma 1, beta 0, allocated and initialised after the
     * five upstream tensors so the Phase-A L1 gate still holds. gamma=1,
     * beta=0 makes each norm the identity at init, so the architecture starts
     * from the same forward as the control and any difference is learned. */
    if (arch == CD_ARCH_XBID) {
        m->n1g = calloc(d->R, sizeof(float)); m->n1b = calloc(d->R, sizeof(float));
        m->n2g = calloc(d->R, sizeof(float)); m->n2b = calloc(d->R, sizeof(float));
        m->n3g = calloc(d->R, sizeof(float)); m->n3b = calloc(d->R, sizeof(float));
        for (i = 0; i < d->R; i++) {
            m->n1g[i] = 1.0f; m->n1b[i] = 0.0f;
            m->n2g[i] = 1.0f; m->n2b[i] = 0.0f;
            m->n3g[i] = 1.0f; m->n3b[i] = 0.0f;
        }
    }
}

void cd_model_free(struct cd_model *m) {
    if (!m) return;
    free(m->emb); free(m->pos);
    free(m->lncw); free(m->lncb); free(m->lonw); free(m->lonb);
    free(m->Wq); free(m->Wk); free(m->Wv);
    {   int slot; for (slot = 0; slot < CD_W_COUNT; slot++) free(m->Wt[slot]); }
    free(m->Wo); free(m->gc); free(m->gb);
    free(m->Wq2); free(m->Wk2); free(m->Vv2);
    free(m->n1g); free(m->n1b); free(m->n2g); free(m->n2b); free(m->n3g); free(m->n3b);
    memset(m, 0, sizeof *m);
}

/* ---- maths, transcribed from math.c ----------------------------------- */

void cd_layernorm(const float *in, float *out, int rows, int width,
                  const float *weight, const float *bias) {
    const double eps = 1e-5;
    int r, w;
    for (r = 0; r < rows; r++) {
        const float *src = in + (size_t)r * width;
        float *dst = out + (size_t)r * width;
        double sum = 0.0, mean, var_sum = 0.0, var;
        float inv, mean_f;
        for (w = 0; w < width; w++) sum += (double)src[w];
        mean = sum / (double)width;
        for (w = 0; w < width; w++) {
            double d = (double)src[w] - mean;
            var_sum += d * d;
        }
        var = var_sum / (double)width;
        inv = (float)(1.0 / sqrt(var + eps));
        mean_f = (float)mean;
        for (w = 0; w < width; w++) {
            float norm = (src[w] - mean_f) * inv;
            dst[w] = norm * weight[w] + bias[w];
        }
    }
}

void cd_mat_rows(const float *src, const float *W, float *dst, int rows,
                 int width, int R) {
    int i, r, w;
    for (i = 0; i < rows; i++) {
        const float *s = src + (size_t)i * width;
        float *d = dst + (size_t)i * R;
        for (r = 0; r < R; r++) {
            const float *wr = W + (size_t)r * width;
            double acc = 0.0;
            for (w = 0; w < width; w++) acc += (double)s[w] * (double)wr[w];
            d[r] = (float)acc;
        }
    }
}

void cd_softmax_rows(const float *in, float *out, int rows, int cols) {
    int r, c;
    for (r = 0; r < rows; r++) {
        const float *src = in + (size_t)r * cols;
        float *dst = out + (size_t)r * cols;
        float maxv = src[0], sum = 0.0f, inv;
        for (c = 1; c < cols; c++) if (src[c] > maxv) maxv = src[c];
        for (c = 0; c < cols; c++) {
            float e = expf(src[c] - maxv);
            dst[c] = e;
            sum += e;
        }
        inv = sum == 0.0f ? 0.0f : 1.0f / sum;
        for (c = 0; c < cols; c++) dst[c] *= inv;
    }
}

void cd_softmax_1d(const float *in, float *out, int n) {
    int i;
    float maxv, sum = 0.0f, inv;
    if (n <= 0) return;
    maxv = in[0];
    for (i = 1; i < n; i++) if (in[i] > maxv) maxv = in[i];
    for (i = 0; i < n; i++) {
        float e = expf(in[i] - maxv);
        out[i] = e;
        sum += e;
    }
    inv = sum == 0.0f ? 0.0f : 1.0f / sum;
    for (i = 0; i < n; i++) out[i] *= inv;
}

int cd_forward(const struct cd_model *m, const char *context,
               const char *const *options, int n_options,
               struct cd_cache *c) {
    return cd_forward_head(m, context, options, n_options, c, CD_HEAD_FWD);
}

/* ---- tokenisation, transcribed from token.c --------------------------- */

size_t cd_truncated_len(const char *t, uint32_t max_len) {
    size_t n;
    if (!t) return 0;
    n = strlen(t);
    return n > max_len ? max_len : n;
}

void cd_tokenize(const char *t, uint32_t max_len, uint32_t *ids) {
    size_t i, n = cd_truncated_len(t, max_len);
    for (i = 0; i < n; i++) ids[i] = (uint32_t)(unsigned char)t[i] + 1u;
}

/* ---- cache ------------------------------------------------------------ */

void cd_cache_alloc(struct cd_cache *c, const struct cd_model *m) {
    cd_cache_alloc_n(c, m, (size_t)m->d.n_opt);
}

/* Ask-time sizing. The forward loop writes per-option state for every candidate it is
 * given, so a cache sized by the model's training n_opt overruns whenever ask offers
 * more candidates than the model trained with -- a hang, not an answer. Sizing by
 * max(training n_opt, asked n) kills that class: the math is identical, only the
 * scratch space grows. Training paths keep calling cd_cache_alloc and are untouched. */
void cd_cache_alloc_n(struct cd_cache *c, const struct cd_model *m, size_t n_opt) {
    size_t L = m->d.C, N = (size_t)m->d.n_opt > n_opt ? (size_t)m->d.n_opt : n_opt;
    size_t W = m->d.W, R = m->d.R;
    memset(c, 0, sizeof *c);
    c->m = m; c->L = L; c->N = N;
    c->ctx_emb  = xcalloc(L * W);
    c->opt_pool = xcalloc(N * W);
    c->ctx_norm = xcalloc(L * W);
    c->opt_norm = xcalloc(N * W);
    c->Q  = xcalloc(N * R);
    c->K  = xcalloc(L * R);
    c->V  = xcalloc(L * R);
    c->tWk = xcalloc(m->n_qkv); c->tVv = xcalloc(m->n_qkv);
    c->scores = xcalloc(N * L);
    c->attn   = xcalloc(N * L);
    c->attended = xcalloc(N * R);
    c->logits = xcalloc(N);
    c->probs  = xcalloc(N);
    c->cent = xcalloc(W);
    if (m->arch == CD_ARCH_XBID) {
        c->u  = xcalloc(N * R);  c->y = xcalloc(L * R);
        c->Q2 = xcalloc(L * R);
        c->s2 = xcalloc(L * N);  c->att2 = xcalloc(L * N);
        c->s3 = xcalloc(N * L);  c->att3 = xcalloc(N * L); c->h3 = xcalloc(N * R);
        c->h1n = xcalloc(N * R); c->gl = xcalloc(L * R); c->h2n = xcalloc(N * R);
    }
}

void cd_cache_free(struct cd_cache *c) {
    free(c->ctx_emb); free(c->opt_pool); free(c->ctx_norm); free(c->opt_norm);
    free(c->Q); free(c->K); free(c->V);
    free(c->scores); free(c->attn); free(c->attended);
    free(c->logits); free(c->probs); free(c->cent);
    free(c->u); free(c->y); free(c->Q2);
    free(c->s2); free(c->att2); free(c->s3); free(c->att3); free(c->h3);
    free(c->h1n); free(c->gl); free(c->h2n);
    memset(c, 0, sizeof *c);
    free(c->tWk); free(c->tVv);
}

/* ---- forward ---------------------------------------------------------- */

/* The readout. Baseline: dot(Q[n], attended[n]) / sqrt(R), exactly upstream.
 * Interact: adds a gated contrast term built from the candidate-set centroid,
 * which is the ONLY place option i and option j ever meet.
 *
 *   centroid  c[w]     = mean_n opt_norm[n][w]
 *   contrast  d[n][r]  = ( Wo[r] . (opt_norm[n] - c) ) * gc[r] + gb[r]
 *   logit[n]          = ( Q[n].A[n] + d[n] ) / sqrt(R)
 *
 * centroid and mean-subtraction are both permutation-equivariant, so permuting
 * the option set leaves every logit unchanged up to the permutation itself.
 */
/* n is the number of options ACTUALLY ASKED, not the model's training n_opt: the cache
 * (c->N) always fits n, and scoring fewer silently leaves the rest at calloc zero -- a 0.0
 * logit that then wins every softmax it should have lost. Found live: the last of 5
 * candidates on an n_opt=4 model always scored exactly 0.0 and took 100% probability
 * whenever the real logits were negative. */
static void readout(const struct cd_model *m, struct cd_cache *c, int n_options) {
    const uint32_t W = m->d.W, R = m->d.R;
    const int N = n_options;
    float inv_sqrt = 1.0f / sqrtf((float)R);
    int n, r, w;

    if (m->arch == CD_ARCH_INTERACT) {
        for (w = 0; w < (int)W; w++) c->cent[w] = 0.0f;
        for (n = 0; n < N; n++)
            for (w = 0; w < (int)W; w++) c->cent[w] += c->opt_norm[(size_t)n * W + w];
        for (w = 0; w < (int)W; w++) c->cent[w] /= (float)N;
    }

    for (n = 0; n < N; n++) {
        const float *q = c->Q + (size_t)n * R;
        const float *av = c->attended + (size_t)n * R;
        double dot = 0.0;
        for (r = 0; r < (int)R; r++) dot += (double)q[r] * (double)av[r];
        if (m->arch == CD_ARCH_INTERACT) {
            const float *on = c->opt_norm + (size_t)n * W;
            double dd = 0.0;
            for (r = 0; r < (int)R; r++) {
                const float *wr = m->Wo + (size_t)r * W;
                double acc = 0.0;
                for (w = 0; w < (int)W; w++)
                    acc += (double)on[w] * (double)wr[w];
                /* subtract the same projection of the centroid: this is the
                 * contrast, computed as a difference of two projections so
                 * the arithmetic stays a plain dot product. */
                for (w = 0; w < (int)W; w++)
                    acc -= (double)c->cent[w] * (double)wr[w];
                acc = acc * (double)m->gc[r] + (double)m->gb[r];
                dd += acc;
            }
            dot += dd;
        }
        c->logits[n] = (float)dot * inv_sqrt;
    }
}

/* ---- read-only internal-state extraction, for external probing -------------
 *
 * DIAGNOSIS ONLY. This runs the ordinary forward and then copies out
 * intermediates that the forward has already computed. It adds no arithmetic to
 * the forward, mutates nothing, and has no path that can write a weight, so
 * every existing caller and every existing output is bit-identical. It exists
 * because the question "does the frozen model contain a correctness signal its
 * confidence head does not expose" can only be answered by looking at the state
 * immediately before the final scoring step.
 *
 * For XBID the final score is  logits[n] = (Q[n] . h2n[n]) / sqrt(R),  so
 * `h2n` is the exact pre-scoring representation and `Q` is the option's own
 * query. `h1n` is the round-1 candidate state before the context is rewritten,
 * `gl` is the context as the candidate set rewrote it, and `att3` is the
 * round-2 attention distribution over context positions.
 *
 * Blocks requested by name, in a fixed order, appended to `out`:
 *   h2n   2*R      per-candidate pre-scoring representation
 *   Q     2*R      per-candidate option query
 *   h1n   2*R      per-candidate round-1 state
 *   gl    L*R      context rewritten by the candidate set
 *   att3  2*L      round-2 attention over context positions
 *   h3    2*R      pre-norm round-3 average
 * `*out_dim` receives the total float count. Returns 0 on success.
 */
int cd_probe_state(const struct cd_model *m, const char *context,
                   const char *const *options, int n_options, int head,
                   const char *const *blocks, int n_blocks,
                   float *out, size_t cap, size_t *out_dim) {
    struct cd_cache c;
    char err[256] = { 0 };
    size_t o = 0, L, R = m->d.R;
    int rc, i, n;
    if (!blocks || n_blocks < 1 || !out_dim) return -1;
    memset(&c, 0, sizeof c);
    cd_cache_alloc(&c, m);
    rc = cd_forward_head(m, context, options, n_options, &c, head);
    if (rc != 0) { cd_cache_free(&c); return -1; }
    L = c.L;
    *out_dim = 0;
    for (i = 0; i < n_blocks; i++) {
        const char *b = blocks[i];
        size_t width = 0;
        if (!strcmp(b, "h2n"))  width = (size_t)n_options * R;
        else if (!strcmp(b, "Q"))   width = (size_t)n_options * R;
        else if (!strcmp(b, "h1n"))  width = (size_t)n_options * R;
        else if (!strcmp(b, "gl"))   width = L * R;
        else if (!strcmp(b, "att3")) width = (size_t)n_options * L;
        else if (!strcmp(b, "h3"))   width = (size_t)n_options * R;
        else { cd_cache_free(&c); snprintf(err, sizeof err, "unknown probe block '%s'", b); return -1; }
        if (o + width > cap) { cd_cache_free(&c); return -1; }
        if      (!strcmp(b, "h2n")) for (n = 0; n < n_options; n++) memcpy(out + o + (size_t)n * R, c.h2n + (size_t)n * R, R * sizeof(float));
        else if (!strcmp(b, "Q"))   for (n = 0; n < n_options; n++) memcpy(out + o + (size_t)n * R, c.Q   + (size_t)n * R, R * sizeof(float));
        else if (!strcmp(b, "h1n"))  for (n = 0; n < n_options; n++) memcpy(out + o + (size_t)n * R, c.h1n + (size_t)n * R, R * sizeof(float));
        else if (!strcmp(b, "gl"))   memcpy(out + o, c.gl, L * R * sizeof(float));
        else if (!strcmp(b, "att3")) for (n = 0; n < n_options; n++) memcpy(out + o + (size_t)n * L, c.att3 + (size_t)n * L, L * sizeof(float));
        else if (!strcmp(b, "h3"))   for (n = 0; n < n_options; n++) memcpy(out + o + (size_t)n * R, c.h3 + (size_t)n * R, R * sizeof(float));
        o += width;
    }
    cd_cache_free(&c);
    *out_dim = o;
    return 0;
}

/* The exact logit the forward produced, so a probe dump can be checked against
 * the model's own output rather than trusted. */
int cd_probe_logits(const struct cd_model *m, const char *context,
                    const char *const *options, int n_options, int head,
                    float *out, size_t cap) {
    struct cd_cache c;
    int rc, n;
    if (n_options < 1 || (size_t)n_options > cap) return -1;
    memset(&c, 0, sizeof c);
    cd_cache_alloc_n(&c, m, (size_t)n_options);
    rc = cd_forward_head(m, context, options, n_options, &c, head);
    if (rc == 0) for (n = 0; n < n_options; n++) out[n] = c.logits[n];
    cd_cache_free(&c);
    return rc;
}

int cd_forward_head(const struct cd_model *m, const char *context,
                    const char *const *options, int n_options,
                    struct cd_cache *c, int head) {
    uint32_t W = m->d.W, R = m->d.R, C = m->d.C, O = m->d.O;
    size_t L, i, k, w;
    int n, r;

    if (n_options < 1) return -1;
    L = cd_truncated_len(context, C);
    if (L == 0) return -1;
    /* Publish the ACTUAL context length for this example. Without this, c->L
     * keeps the value from cd_cache_alloc (the configured maximum), and the
     * backward pass then walks context[0..L-1] past the end of the string on
     * every short example. AddressSanitizer caught it as a 1-byte
     * heap-buffer-overflow at cd_train.c:386 reading a 71-char context with
     * L=192. It was silent in all three architectures because the overread
     * lands in mapped heap and the extra gradients land in the correctly sized
     * pos[] array, so nothing crashed and no metric moved. The Phase-A gate
     * could not see it either: L1 and L2 are forward-only, and L2's loss is
     * accumulated before any update. */
    c->L = L;

    /* 1. context embedding + position */
    for (i = 0; i < L; i++) {
        uint32_t tid = (uint32_t)(unsigned char)context[i] + 1u;
        const float *emb = m->emb + (size_t)tid * W;
        const float *pos = m->pos + i * W;
        float *dst = c->ctx_emb + i * W;
        for (w = 0; w < W; w++) dst[w] = emb[w] + pos[w];
    }
    /* 2. option mean-pool */
    for (n = 0; n < n_options; n++) {
        float *dst = c->opt_pool + (size_t)n * W;
        size_t kl = cd_truncated_len(options[n] ? options[n] : "", O);
        for (w = 0; w < W; w++) dst[w] = 0.0f;
        if (kl == 0) continue;
        for (k = 0; k < kl; k++) {
            uint32_t tid = (uint32_t)(unsigned char)options[n][k] + 1u;
            const float *emb = m->emb + (size_t)tid * W;
            for (w = 0; w < W; w++) dst[w] += emb[w];
        }
        {
            float inv = 1.0f / (float)kl;
            for (w = 0; w < W; w++) dst[w] *= inv;
        }
    }
    /* 3. LayerNorm both branches */
    cd_layernorm(c->ctx_emb, c->ctx_norm, (int)L, (int)W, m->lncw, m->lncb);
    cd_layernorm(c->opt_pool, c->opt_norm, n_options, (int)W, m->lonw, m->lonb);
    /* 4. Q, K, V */
    /* Context projections. For the two-head family these are resolved per
     * head (C0 shared, C1 separate Wk, C2 separate Wv, C3 separate both) or
     * summed onto one path for the parameter-matched SUM_* controls. The
     * attention QUERY stays the shared Wq in every condition, so the only
     * thing that varies across C0-C3 is the context-side key/value map. */
    const float *Wk_e = m->Wk, *Wv_e = m->Wv;
    if (cd_arch_is_twohead((int)m->arch))
        cd_tw_weights(m, head, &Wk_e, &Wv_e, &c->tw_Wk2, &c->tw_Vv2,
                      &c->tw_summed, c->tWk, c->tVv);
    cd_mat_rows_x(c->opt_norm, m->Wq, cd_model_transpose(m, m->Wq), c->Q, n_options, (int)W, (int)R);
    cd_mat_rows_x(c->ctx_norm, Wk_e, cd_model_transpose(m, Wk_e), c->K, (int)L, (int)W, (int)R);
    cd_mat_rows_x(c->ctx_norm, Wv_e, cd_model_transpose(m, Wv_e), c->V, (int)L, (int)W, (int)R);
    /* 5. scaled dot-product scores, float accumulation */
    {
        float inv_sqrt = 1.0f / sqrtf((float)R);
        for (n = 0; n < n_options; n++) {
            const float *q = c->Q + (size_t)n * R;
            float *sc = c->scores + (size_t)n * L;
            for (i = 0; i < L; i++) {
                const float *kk = c->K + i * R;
                float d = 0.0f;
                for (r = 0; r < (int)R; r++) d += q[r] * kk[r];
                sc[i] = d * inv_sqrt;
            }
        }
    }
    /* 6. softmax over context */
    cd_softmax_rows(c->scores, c->attn, n_options, (int)L);
    /* 7. attended context, float accumulation */
    for (n = 0; n < n_options; n++) {
        float *dst = c->attended + (size_t)n * R;
        const float *a = c->attn + (size_t)n * L;
        for (r = 0; r < (int)R; r++) dst[r] = 0.0f;
        for (i = 0; i < L; i++) {
            float wt = a[i];
            const float *v = c->V + i * R;
            for (r = 0; r < (int)R; r++) dst[r] += wt * v[r];
        }
    }
    /* 7b. XBID: bidirectional candidate <-> context conditioning.
     *
     * Every vector below is a convex combination (softmax-weighted) of
     * LayerNormed inputs, and there are NO residual adds. That is deliberate:
     * an earlier version with residuals in the R-dim stream had no LayerNorm
     * to bound it (ln_o/ln_c are W-dim and already consumed on the inputs) and
     * diverged at every learning rate tried -- train_nll rose monotonically
     * with lr, 1.72 at 2e-4 against the baseline's 0.456 at 2e-3. Dropping the
     * residuals bounds the logits by construction and keeps the parameter
     * count identical to the baseline.
     *
     *   round 1  candidates read the context
     *              h1[n] = sum_l a1[n,l] (Wv.x_l)            a1 over l
     *   round 2a the context reads the CANDIDATE SET
     *              g[l]  = sum_n b[l,n]  h1[n]               b over n
     *   round 2b candidates read the rewritten context
     *              h2[n] = sum_l a2[n,l] g[l]               a2 over l
     *              logit[n] = (Wq.c_n . h2[n]) / sqrt(R)
     *
     * The direct query term matters. An earlier version scored (h1 . h2)
     * instead, which DELETED the only path by which the option's own
     * representation reaches the score: h1 is already an attention average, so
     * its dependence on n is weak and the logits came out near-uniform. It sat
     * at train_nll 0.7065, i.e. WORSE than ln 2, after 120 epochs. The
     * baseline's score is Q[n] . attended[n] precisely because the option
     * query multiplies the readout directly; keeping that term costs nothing
     * and is what the pre-registered design specifies.
     *
     * g is a function of the whole candidate set, so the context read in round
     * 2b was produced by the candidates themselves. This is a shared state
     * with candidate<->question conditioning, not a per-candidate readout and
     * not recursive answer feedback.
     *
     * Permutation equivariance: the only cross-candidate op is the softmax over
     * n forming b, which is permutation invariant; everything else is
     * per-candidate or per-token. logits[n] depends on option n's identity and
     * on the set, never on position, so permcheck must stay exact.
     */
    if (m->arch == CD_ARCH_XBID) {
        int l, nn, rr;
        const int N = n_options;
        const float isq = 1.0f / sqrtf((float)R);
        float *cand = c->opt_norm, *ctx = c->ctx_norm;

        /* round 1: candidates attend the context (same maths as the baseline) */
        cd_mat_rows_x(cand, m->Wq, cd_model_transpose(m, m->Wq), c->Q, N, (int)W, (int)R);
        cd_mat_rows_x(ctx, m->Wk, cd_model_transpose(m, m->Wk), c->K, (int)L, (int)W, (int)R);
        cd_mat_rows_x(ctx, m->Wv, cd_model_transpose(m, m->Wv), c->V, (int)L, (int)W, (int)R);
        for (nn = 0; nn < N; nn++) {
            const float *q = c->Q + (size_t)nn * R;
            float *sc = c->scores + (size_t)nn * L;
            for (l = 0; l < (int)L; l++) {
                const float *k = c->K + (size_t)l * R;
                float d = 0.0f;
                for (rr = 0; rr < (int)R; rr++) d += q[rr] * k[rr];
                sc[l] = d * isq;
            }
        }
        cd_softmax_rows(c->scores, c->attn, N, (int)L);
        for (nn = 0; nn < N; nn++) {           /* raw round-1 average in c->u */
            const float *a = c->attn + (size_t)nn * L;
            for (rr = 0; rr < (int)R; rr++) c->u[(size_t)nn * R + rr] = 0.0f;
            for (l = 0; l < (int)L; l++) {
                float wt = a[l];
                const float *v = c->V + (size_t)l * R;
                for (rr = 0; rr < (int)R; rr++) c->u[(size_t)nn * R + rr] += wt * v[rr];
            }
        }

        cd_layernorm(c->u, c->h1n, N, (int)R, m->n1g, m->n1b);

        /* round 2a: context attends over the candidate set */
        cd_mat_rows_x(ctx, m->Wk, cd_model_transpose(m, m->Wk), c->Q2, (int)L, (int)W, (int)R);
        for (l = 0; l < (int)L; l++) {
            const float *q = c->Q2 + (size_t)l * R;
            float *sc = c->s2 + (size_t)l * N;
            for (nn = 0; nn < N; nn++) {
                const float *h = c->h1n + (size_t)nn * R;
                float d = 0.0f;
                for (rr = 0; rr < (int)R; rr++) d += q[rr] * h[rr];
                sc[nn] = d * isq;
            }
        }
        cd_softmax_rows(c->s2, c->att2, (int)L, N);
        for (l = 0; l < (int)L; l++) {          /* g[l], kept in c->y */
            const float *a = c->att2 + (size_t)l * N;
            for (rr = 0; rr < (int)R; rr++) c->y[(size_t)l * R + rr] = 0.0f;
            for (nn = 0; nn < N; nn++) {
                float wt = a[nn];
                const float *h = c->h1n + (size_t)nn * R;
                for (rr = 0; rr < (int)R; rr++) c->y[(size_t)l * R + rr] += wt * h[rr];
            }
        }

        cd_layernorm(c->y, c->gl, (int)L, (int)R, m->n2g, m->n2b);

        /* round 2b: candidates attend the context the candidates produced */
        for (nn = 0; nn < N; nn++) {
            const float *q = c->Q + (size_t)nn * R;
            float *sc = c->s3 + (size_t)nn * L;
            for (l = 0; l < (int)L; l++) {
                const float *g = c->gl + (size_t)l * R;
                float d = 0.0f;
                for (rr = 0; rr < (int)R; rr++) d += q[rr] * g[rr];
                sc[l] = d * isq;
            }
        }
        cd_softmax_rows(c->s3, c->att3, N, (int)L);
        for (nn = 0; nn < N; nn++) {
            const float *a = c->att3 + (size_t)nn * L;
            double dot = 0.0;
            for (rr = 0; rr < (int)R; rr++) c->h3[(size_t)nn * R + rr] = 0.0f;
            for (l = 0; l < (int)L; l++) {
                float wt = a[l];
                const float *g = c->gl + (size_t)l * R;
                for (rr = 0; rr < (int)R; rr++) c->h3[(size_t)nn * R + rr] += wt * g[rr];
            }
            cd_layernorm(c->h3, c->h2n, N, (int)R, m->n3g, m->n3b);
            {   /* direct option path: Q[n] = Wq . c_n */
                const float *q = c->Q + (size_t)nn * R;
                for (rr = 0; rr < (int)R; rr++)
                    dot += (double)q[rr] * (double)c->h2n[(size_t)nn * R + rr];
            }
            c->logits[nn] = (float)dot * isq;
        }

        cd_softmax_1d(c->logits, c->probs, N);
        c->loss = 0.0f;
        return 0;
    }

    /* 8. judgement head.
     *
     * The ONLY change from baseline is which option->query map produces the
     * score. The context encoder (Wk/Wv), the LayerNorms, the embeddings, the
     * attention and the attended vector are all shared, and the attention
     * itself still uses Wq for both heads so the trunk is genuinely common.
     */
    if (cd_tw_sum_query((int)m->arch)) {
        /* One judgement path, two summed query maps. Identical arithmetic to
         * the forward head except the query is Wq.c + Wq2.c, so the parameter
         * count and gradient budget match TWOHEAD exactly. */
        int h, rr2;
        const int N = n_options;
        const float invs = 1.0f / sqrtf((float)R);
        cd_mat_rows_x(c->opt_norm, m->Wq, cd_model_transpose(m, m->Wq), c->Q, N, (int)W, (int)R);
        if (m->Wq2) {
            float *q2 = xcalloc((size_t)N * R);
            cd_mat_rows_x(c->opt_norm, m->Wq2, cd_model_transpose(m, m->Wq2), q2, N, (int)W, (int)R);
            for (h = 0; h < N * (int)R; h++) c->Q[h] += q2[h];
            free(q2);
        }
        for (h = 0; h < N; h++) {
            const float *q = c->Q + (size_t)h * R;
            const float *av = c->attended + (size_t)h * R;
            double dot = 0.0;
            for (rr2 = 0; rr2 < (int)R; rr2++) dot += (double)q[rr2] * (double)av[rr2];
            c->logits[h] = (float)dot * invs;
        }
        cd_softmax_1d(c->logits, c->probs, N);
        c->loss = 0.0f;
        return 0;
    }

    if (cd_tw_split_query((int)m->arch) && head == CD_HEAD_INV) {
        int h, rr2;
        const int N = n_options;
        const float invs = 1.0f / sqrtf((float)R);
        cd_mat_rows_x(c->opt_norm, m->Wq2, cd_model_transpose(m, m->Wq2), c->Q, N, (int)W, (int)R);
        for (h = 0; h < N; h++) {
            const float *q = c->Q + (size_t)h * R;
            const float *av = c->attended + (size_t)h * R;
            double dot = 0.0;
            for (rr2 = 0; rr2 < (int)R; rr2++) dot += (double)q[rr2] * (double)av[rr2];
            c->logits[h] = (float)dot * invs;
        }
        cd_softmax_1d(c->logits, c->probs, N);
        c->loss = 0.0f;
        return 0;
    }

    /* 8. readout (+ interaction when enabled) */
    readout(m, c, n_options);
    /* 9. option softmax */
    cd_softmax_1d(c->logits, c->probs, n_options);
    c->loss = 0.0f;
    return 0;
}

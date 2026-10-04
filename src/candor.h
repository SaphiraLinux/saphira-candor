/* candor.h 2014 public interface of the Candor engine.
 *
 * Two modes, deliberately separated:
 *
 *   LEGACY    reproduces the upstream trainer semantics exactly, so results
 *             are comparable with every historical number. Same init RNG, same
 *             forward arithmetic, same 1e-12 loss clamp, same AdamW (eps
 *             OUTSIDE the sqrt, decoupled decay on EVERY parameter), same
 *             hardcoded grad clip 1.0, same hardcoded per-epoch shuffle
 *             constant, same best-validation-NLL checkpointing.
 *
 *   CORRECT   fixes what is wrong with the above, one switch at a time, never
 *             silently. Every fix is named and recorded.
 *
 * Architectures are selected by name so a result can always be traced to a
 * forward pass that is written down:
 *   baseline   the existing independent TinyScorer, unchanged semantics
 *   interact   + a permutation-equivariant candidate-set aggregate fed to the
 *              readout. Centroid and candidate-minus-centroid contrast.
 *
 * The semantic-normalisation arm of Stage 0.4 is NOT here: it is a diagnostic
 * rewrite of the DATA, not of the model, and lives in data/.
 */
#ifndef CANDOR_NEXT_H
#define CANDOR_NEXT_H

#include <stddef.h>
#include <stdint.h>

/* product version, printed by both CLIs (--version) */
#define CANDOR_VERSION "1.0.0"

/* per-example option cap; a row with more options is a hard error, never truncated */
#define CD_MAX_OPT 64

/* ---- model ------------------------------------------------------------ */

enum cd_arch {
    CD_ARCH_BASELINE = 0,   /* the control: independent per-candidate scoring */
    CD_ARCH_INTERACT = 1,   /* Stage 0.4 weak version: centroid contrast      */
    CD_ARCH_XBID     = 2,   /* Stage 0.8: bidirectional candidate/context     */
    /* Stage 0.9 diagnostic: shared trunk, two structurally identical narrow
     * judgement heads. Head 0 reuses Wq exactly, so the forward head IS the
     * baseline computation and the forward condition is a strict control. */
    CD_ARCH_TWOHEAD  = 3,
    /* Parameter-matched SINGLE-head control. Same extra R*W matrix as
     * TWOHEAD, but both query maps are summed into ONE score, so the parameter
     * count and the gradient budget match while there is only one judgement
     * path. Any difference against TWOHEAD is head separation rather than
     * capacity. */
    CD_ARCH_SUMHEAD  = 4,
    /* Stage 0.11 context-projection factorial. Shared embeddings, positions,
     * LayerNorms, candidate representation, attention mechanism and two
     * judgement heads throughout; only the context-side projections vary.
     *   TW_SK   separate Wk, shared Wv      -> selection
     *   TW_SV   shared Wk,  separate Vv     -> extraction
     *   TW_SKV  separate both
     * SUM_K / SUM_V / SUM_KV are the honest parameter-matched SINGLE-PATH
     * controls: same matrices, one path, the two forms' maps summed.
     */
    CD_ARCH_TW_SK    = 5,
    CD_ARCH_TW_SV    = 6,
    CD_ARCH_TW_SKV   = 7,
    CD_ARCH_SUM_K    = 9,
    CD_ARCH_SUM_V    = 10,
    CD_ARCH_SUM_KV   = 11
};

/* 1 when the architecture is in the two-judgement-head family. */
int cd_arch_is_twohead(int arch);

/* which judgement head a forward pass should use */
enum cd_head { CD_HEAD_FWD = 0, CD_HEAD_INV = 1 };

struct cd_dims {
    uint32_t W, R, C, O;
    int n_opt;             /* options per example, fixed per run */
};

/* index of a projection matrix inside cd_model::Wt */
enum cd_wslot { CD_W_Q, CD_W_K, CD_W_V, CD_W_Q2, CD_W_K2, CD_W_V2,
                CD_W_WO, CD_W_COUNT };

struct cd_model {
    struct cd_dims d;
    enum cd_arch arch;

    float *emb;            /* [257*W] */
    float *pos;            /* [C*W]   */
    float *lncw, *lncb;    /* [W]     */
    float *lonw, *lonb;    /* [W]     */
    float *Wq, *Wk, *Wv;   /* [R*W]   */
    float *Wo;             /* [R*W]   read-out for the interaction feature, interact only */
    float *gc, *gb;        /* [W]     gamma/beta turning the contrast feature on/off */
    /* XBID stream norms. R-dim, one pair per round, so the bidirectional
     * message passing is conditioned at every step. 4*R extra parameters,
     * 0.6% of the baseline count, and far cheaper than giving each round its
     * own projections (which would add 6*R*W and confound compute with
     * capacity). */
    float *n1g, *n1b, *n2g, *n2b, *n3g, *n3b;
    float *Wq2, *Wk2, *Vv2;  /* [R*W] inverse head's option->query map, and the
     *                         per-form context keys / values when split */
    /* XBID adds no parameters: it reuses ln_o for the candidate branch and
     * ln_c for the context branch, and shares Wq/Wk/Wv across both rounds.
     * The only new tensors are activations inside the forward cache. */

    size_t n_emb, n_pos, n_norm, n_qkv;

    /* DERIVED, NOT PARAMETERS. Pre-transposed copies of the projection
     * matrices, Wt[w*R + r], so an exact row-axis AVX2 kernel can read four
     * consecutive ranks contiguously. Rebuilt by cd_model_refresh_transposes()
     * after every optimiser step and on load. Never written to a model file,
     * never given a gradient, never included in the trainable or optimiser
     * sets, and never counted by cd_model_param_count(). */
    double *Wt[CD_W_COUNT];
};


/* Deterministic init identical to the upstream model init, including the
 * xorshift64+Box-Muller stream and the exact order emb, pos, Wq, Wk, Wv. */
void cd_model_init(struct cd_model *m, const struct cd_dims *d,
                   enum cd_arch arch, uint64_t seed);
void cd_model_free(struct cd_model *m);
size_t cd_model_param_count(const struct cd_model *m);

/* Introspection for the optimisation-coverage invariant. Each returns a
 * NULL-terminated array of the tensors that are trainable / hold gradient /
 * are actually touched by the optimiser step. A tensor that appears in the
 * model's trainable set but not in the optimiser set is a parameter that is
 * allocated, used, given a gradient, and never updated -- exactly the Wq2
 * defect -- and must fail the test suite rather than pass a gradient check. */
struct cd_grad;   /* defined in cd_train.c; needed for the introspection */
size_t cd_model_trainable(const struct cd_model *m, const float ***out);
/* Length of every trainable tensor, in the order cd_model_trainable emits them.
 * Single source of truth, shared by the parameter count, the gradient-size hook
 * and the regression test, so the three cannot disagree. */
size_t *cd_model_tensor_sizes(const struct cd_model *m);

/* ---- Stage 0.18 contextual-bandit post-training (PREPARED, NOT RUN) ------
 *
 * RL execution is blocked. This is the prepared entry point; nothing has called
 * it. The pure objective, sampler and baseline live in cd_bandit.c and are
 * verified against finite differences by tests/test_bandit.c.
 *
 * model_in is the frozen supervised checkpoint and is opened READ-ONLY; the
 * function copies it and never writes back. beta and alpha are the frozen
 * regularisation coefficients, 0.1 and 1.0, and the four predeclared arms are
 * just the four (beta, alpha) combinations: (0,0) baseline, (0.1,0) entropy
 * only, (0,1) KL only, (0.1,1) both.
 */
struct bandit_rng { unsigned long long s; };
unsigned long long bandit_rng_u64(struct bandit_rng *r);
void   bandit_rng_seed(struct bandit_rng *r, unsigned long long seed);
int    bandit_sample(const double *p, int k, struct bandit_rng *r);
double bandit_baseline_step(double baseline, double reward);
double bandit_baseline_init(void);
double cd_bandit_entropy(const double *p, int k);
double cd_bandit_kl(const double *p, const double *q, int k);
double cd_bandit_loss(const double *p, const double *q, int k, int action,
                      double advantage, double beta, double alpha);
void   cd_bandit_dlogits(const double *p, const double *q, int k, int action,
                         double advantage, double beta, double alpha,
                         double *out);
int cd_bandit_posttrain(const char *model_in, const char *data_path,
                        const char *model_out, const char *log_path,
                        double beta, double alpha, int epochs, int batch,
                        double lr, unsigned long long seed,
                        /* exploratory, both default to the sealed behaviour:
                         * mask_spec="all", wrong_pen=0.0 */
                        const char *mask_spec, double wrong_pen,
                        char *err, size_t errlen);
size_t cd_grad_trainable(const struct cd_grad *g, const float ***out);
size_t cd_optimizer_trainable(const struct cd_model *m, const float ***out);

/* ---- forward cache ---------------------------------------------------- */

struct cd_cache {
    const struct cd_model *m;
    size_t L, N;
    float *ctx_emb;    /* [L*W]  */
    float *opt_pool;   /* [N*W]  */
    float *ctx_norm;   /* [L*W]  */
    float *opt_norm;   /* [N*W]  */
    float *Q, *K, *V;  /* [N*R],[L*R],[L*R] */
    float *cent;       /* [W]    candidate-set centroid, interact only */
    /* --- XBID scratch --- */
    float *u;          /* [N*R] candidates after reading the context       */
    float *y;          /* [L*R] context rewritten by the candidate set     */
    float *Q2;         /* [L*R] context queries over candidates           */
    float *s2;         /* [L*N] context-to-candidate attention scores     */
    float *att2;       /* [L*N]                                         */
    float *s3;         /* [N*L] candidates over the rewritten context     */
    float *att3;       /* [N*L]                                         */
    float *h3;         /* [N*R] pre-norm round-3 average                  */
    float *h1n;        /* [N*R] post-norm round 1                          */
    float *gl;         /* [L*R] post-norm round 2a                         */
    float *h2n;        /* [N*R] post-norm round 2b                         */
    const float *tw_Wk2, *tw_Vv2;   /* partner maps, non-NULL only when summed */
    int tw_summed;
    float *tWk, *tVv;   /* [R*W]  context maps summed, when the architecture
                         * uses a single path (Stage 0.11 SUM_* controls) */
    float *scores;     /* [N*L]  */
    float *attn;       /* [N*L]  */
    float *attended;   /* [N*R]  */
    float *logits;     /* [N]    */
    float *probs;      /* [N]    */
    float loss;
};

int cd_forward(const struct cd_model *m, const char *context,
               const char *const *options, int n_options,
               struct cd_cache *c);
/* Head-routed forward. head==CD_HEAD_FWD is byte-identical to cd_forward for
 * every architecture, so the control is unaffected by routing plumbing. */
/* Read-only internal-state extraction for external probing. DIAGNOSIS ONLY: it
 * runs the ordinary forward, copies out intermediates, mutates nothing, and
 * cannot write a weight, so every existing output is bit-identical. */
int cd_probe_state(const struct cd_model *m, const char *context,
                   const char *const *options, int n_options, int head,
                   const char *const *blocks, int n_blocks,
                   float *out, size_t cap, size_t *out_dim);
int cd_probe_logits(const struct cd_model *m, const char *context,
                    const char *const *options, int n_options, int head,
                    float *out, size_t cap);

int cd_forward_head(const struct cd_model *m, const char *context,
                    const char *const *options, int n_options,
                    struct cd_cache *c, int head);
void cd_cache_alloc(struct cd_cache *c, const struct cd_model *m);
void cd_cache_alloc_n(struct cd_cache *c, const struct cd_model *m,
                        size_t n_opt);
void cd_cache_free(struct cd_cache *c);

/* ---- training -------------------------------------------------------- */

enum cd_mode { CD_MODE_LEGACY = 0, CD_MODE_CORRECT = 1 };

struct cd_cfg {
    struct cd_dims d;
    enum cd_arch arch;
    enum cd_mode mode;
    int epochs, batch_size;
    double lr, wd;
    uint64_t seed;
    const char *init_model;    /* load trunk weights from here; extra tensors
                                * keep their own init */
    int freeze_trunk;          /* zero every gradient except the head's own map */
    int force_train_head;      /* -1 = take it from the data's form label,
                                 0/1 = force, for single-form probes whose
                                 file carries no label */
    /* CORRECT-mode switches, all OFF in LEGACY */
    int grad_check;          /* finite-difference gradient check */
    int seed_drives_order;   /* seed also drives the shuffle */
    int no_clamp;            /* drop the 1e-12 loss clamp */
    double clip;             /* <=0 means 1.0 hardcoded like upstream */
    int decay_exclude_norm;  /* no weight decay on LayerNorm gamma/beta */
    int q3;                /* persist 3-bit block-quantized (v2 container) */
    int q8;                /* persist 8-bit block-quantized (v3 container) */
    int q4;                /* persist 4-bit block-quantized (v4 container) */
    int q2;                /* persist 2-bit block-quantized (v5 container) */
};

struct cd_epoch {
    double train_nll, val_nll, val_top1, val_top3, ece, seconds;
};

int cd_train(const struct cd_cfg *cfg,
             const char *train_path, const char *val_path,
             const char *out_model_path,
             struct cd_epoch *history, char *err, size_t err_len);

int cd_eval(const char *model_path, const char *test_path, char *err, size_t err_len);

/* ---- per-example signal dump (MEASUREMENT ONLY) -----------------------
 *
 * Writes one TSV row per evaluation example so candidate correctness
 * predictors can be tested with a RANKING statistic outside the model. No new
 * parameters, no architecture change, no effect on any existing command.
 *
 * Columns, all already computed by the forward pass or trivially derived from
 * what it computed:
 *
 *   correct    0/1, argmax matched the label
 *   conf       max softmax probability. For N=2 this equals sigmoid(|delta|),
 *              a monotone function of the absolute logit margin, so it is the
 *              BASELINE predictor, not a candidate.
 *   amargin    |z_hi - z_lo| = |delta| -- the same ordering as conf.
 *   aent       binary entropy of the softmax, monotone DECREASING in |delta|,
 *              therefore the same ordering again. Recorded to demonstrate the
 *              equivalence, not as a candidate.
 *   common     (z_hi + z_lo)/2, the softmax common mode. Genuinely
 *              non-equivalent: softmax is shift-invariant, so this is visible
 *              in the raw logits but recoverable from neither conf nor margin.
 *   logit_hi   the winning raw logit, itself a scale observable.
 *   attn_ent   entropy of the winning candidate's attention over context
 *   attn_max   its peak attention weight
 *   attn_pos   normalised position of that peak, 0..1
 *   sc_max     largest pre-softmax context score for the winner
 *   sc_spread  max-min of the winner's pre-softmax context scores
 *   att_norm   ||attended[winner]||;  q_norm ||Q[winner]||
 *   cos_qa     cos(Q,attended). The logit is ||q||*||a||*cos, so the norms and
 *              the angle are separately observable even though only their
 *              product is pinned to the logit.
 *   hdisagree  1 if the two judgement heads pick different candidates, else 0;
 *              -1 when the model has no second head. Carries the standing
 *              caveat that the heads were never shown to be independently
 *              useful (corrected Stage 0.11: p=0.094 accuracy tie).
 */
int cd_signals(const char *model_path, const char *test_path, char *err,
               size_t err_len);
/* Stage 0.17: per-example calibration record dump. Additive and opt-in via
 * `eval --dump-per-example F`; the eval summary path is not touched, so default
 * output is byte-identical. Doubles are written with %.17g so every metric is
 * recomputable from the record. */
int cd_eval_dump(const char *model_path, const char *test_path, int head,
                 const char *dump_path, char *err, size_t errlen);
int cd_eval_head(const char *model_path, const char *test_path, int head,
                 char *err, size_t err_len);

/* ---- retained calibration bins ----------------------------------------
 *
 * MEASUREMENT ONLY. These bins are exactly the ones the scalar ECE has always
 * been computed from; the only change is that they are RETAINED instead of
 * being collapsed immediately. Nothing here feeds back into the model, the
 * decision rule, or any parameter.
 *
 * bin = (int)(conf * CD_CALIB_BINS), clamped. `gap` is signed and equals
 * acc - mean_conf, so a negative gap means overconfident. The scalar ECE is
 * the count-weighted sum of |gap| over non-empty bins and is DERIVED from these
 * bins, so reconstructing it from them is exact by construction rather than by
 * coincidence.
 *
 * Note the N=2 boundary: with two candidates the maximum softmax probability
 * cannot fall below 0.5, so bins 0-4 are structurally unreachable. The curve
 * spans [0.5, 1.0] only. That is a property of the demonstrated regime, not of
 * the binning scheme.
 */
#define CD_CALIB_BINS 10

struct cd_calib {
    size_t n;                          /* examples binned */
    int    bin_n[CD_CALIB_BINS];
    double bin_conf[CD_CALIB_BINS];    /* sum of predicted confidence */
    double bin_acc[CD_CALIB_BINS];     /* sum of 0/1 correctness */
    double ece;                        /* derived from the bins above */
};

void   cd_calib_zero(struct cd_calib *cb);
void   cd_calib_add(struct cd_calib *cb, float conf, int correct);
double cd_calib_ece(const struct cd_calib *cb);
/* Deterministic single-line JSON. Fixed key order, fixed precision, ascending
 * bin order, empty bins omitted. `n` is the example count. Returns the number
 * of characters written, or -1 if the buffer was too small. */
int    cd_calib_json(const struct cd_calib *cb, char *out, size_t len);

/* ---- model I/O (CDR v1 container; baseline layout) ---------------------- */
int cd_save(const char *path, const struct cd_model *m, char *err, size_t err_len);
int cd_save_q3(const char *path, const struct cd_model *m,
               char *err, size_t err_len);
int cd_q3_block_encode(const float *v, size_t n, float *mn_out, float *scale_out,
                       unsigned char *packed12);
int cd_q3_block_decode(float mn, float scale, const unsigned char *packed12,
                       float *v, size_t n);
int cd_q8_block_encode(const float *v, size_t n, float *scale_out, signed char *out32);
int cd_q8_block_decode(float scale, const signed char *in32, float *v, size_t n);
int cd_save_q8(const char *path, const struct cd_model *m,
               char *err, size_t err_len);
int cd_qn_block_encode(const float *v, size_t n, int bits, float *mn_out,
                       float *scale_out, unsigned char *packed);
int cd_qn_block_decode(float mn, float scale, const unsigned char *packed,
                       int bits, float *v, size_t n);
int cd_save_q4(const char *path, const struct cd_model *m,
               char *err, size_t err_len);
int cd_save_q2(const char *path, const struct cd_model *m,
               char *err, size_t err_len);
/* Human inspection surface. Loads a model and reports what it believes about a
 * candidate set. Diagnostic only: it never trains, mutates or reorders
 * anything, and it uses the same forward path as evaluation. */
struct cd_facts;
int cd_fact_load(const char *path, struct cd_facts *f, char *err, size_t errlen);
int cd_fact_load_mem(const char *text, struct cd_facts *f, char *err, size_t errlen);
void cd_fact_free(struct cd_facts *f);
size_t cd_fact_count(const struct cd_facts *f);
const char *cd_fact_resolve_entity(const struct cd_facts *f, const char *context);
int cd_fact_lookup(const struct cd_facts *f, const char *entity,
                   char *const *cands, int n, const char **hit, const char **src);
int cd_ask(const char *model_path, const char *context, const char *question,
           char *const *candidates, int n_candidates, int head_mask,
           char *err, size_t err_len);
/* Loaded-model ask surface (self-contained binary): the model bytes were linked
 * at build time, so there is no path to open. */
int cd_ask_show(const struct cd_model *m, const char *context, const char *question,
                char *const *cands, int n_cand, int head_mask);
int cd_ask_mem(struct cd_model *m, const char *context, const char *question,
               char *const *cands, int n_cand, int head_mask);
struct cd_model *cd_load(const char *path, char *err, size_t err_len);
struct cd_model *cd_load_mem(const unsigned char *data, size_t len,
                             char *err, size_t err_len);

#endif /* CANDOR_NEXT_H */

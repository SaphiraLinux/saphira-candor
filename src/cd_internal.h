/* cd_internal.h — internal declarations shared by the candor modules. */
#ifndef CD_INTERNAL_H
#define CD_INTERNAL_H

#include "candor.h"

struct cd_dataset {
    char **context;
    char ***options;
    int *n_options;
    int *label;
    int *form;            /* 0 = forward, 1 = inverse; -1 when absent */
    size_t n;
};

int cd_data_load(const char *path, struct cd_dataset *ds, char *err, size_t errlen);
void cd_data_free(struct cd_dataset *ds);
size_t cd_data_n(const struct cd_dataset *ds);
const char *cd_data_ctx(const struct cd_dataset *ds, size_t i);
int cd_data_nopt(const struct cd_dataset *ds, size_t i);
const char *cd_data_opt(const struct cd_dataset *ds, size_t i, int j);
int cd_data_label(const struct cd_dataset *ds, size_t i);
int cd_data_max_nopt(const struct cd_dataset *ds);
int cd_data_form(const struct cd_dataset *ds, size_t i);

/* maths (cd_model.c) */
void cd_layernorm(const float *in, float *out, int rows, int width,
                  const float *weight, const float *bias);
void cd_mat_rows(const float *src, const float *W, float *dst, int rows,
                 int width, int R);
void cd_softmax_rows(const float *in, float *out, int rows, int cols);
void cd_softmax_1d(const float *in, float *out, int n);
size_t cd_truncated_len(const char *t, uint32_t max_len);
void cd_tokenize(const char *t, uint32_t max_len, uint32_t *ids);

#endif /* CD_INTERNAL_H */

/* ---- regression-test hooks (see tests/test_backward.c) ---------------- */
struct cd_grad;
/* struct cd_grad is opaque to tests; allocate and release it through these. */
struct cd_grad *cd_test_grad_new(const struct cd_model *m);
int cd_save(const char *path, const struct cd_model *m, char *err, size_t err_len);
void cd_test_grad_del(struct cd_grad *g);
void cd_test_grad_alloc(struct cd_grad *g, const struct cd_model *m);
void cd_test_grad_zero(struct cd_grad *g, const struct cd_model *m);
void cd_test_grad_free(struct cd_grad *g);
void cd_test_grad_norm_ptr(struct cd_grad *g, int idx, float **out);

/* The example's real context length, as published by the last cd_forward.
 * Exposed because the Stage 0.8 regression is precisely that this used to
 * report the configured maximum instead. */
size_t cd_test_cache_ctx_len(const struct cd_cache *c);

/* One optimiser step on a single example, used by the continuation test. */
void cd_test_optimizer_step(struct cd_model *m, struct cd_grad *g, double lr);

/* exact row-axis acceleration, cd_simd.c. Wt may be NULL, which forces the
 * scalar reference even on an AVX2 machine. */
int  cd_simd_mode(void);
int  cd_simd_available(void);
const char *cd_simd_name(void);
void cd_mat_rows_x(const float *src, const float *W, const double *Wt,
                   float *dst, int rows, int width, int R);
void cd_attended_x(const float *attn, const float *V, float *dst,
                   int N, int L, int R);
/* rebuild every derived transpose from the current parameters */
void cd_model_refresh_transposes(struct cd_model *m);
/* the derived transpose for a projection matrix, or NULL if it is not one */
const double *cd_model_transpose(const struct cd_model *m, const float *W);
int cd_test_train_steps(const char *train, const char *val, const char *out,
                       enum cd_arch arch, int epochs, int batch,
                       uint64_t seed, char *err, size_t err_len);

/* One real backward on one example, returning a copy of the position
 * gradient. Used by tests/test_backward.c T2 and T3. */
int cd_test_backward_capture(const struct cd_model *m, const char *ctx,
                             const char *const *opts, int n, int label,
                             struct cd_cache *c, float **out_pos, size_t *out_n,
                             char *err, size_t errlen);

/* ---- Stage 0.18 hooks (see tests/test_bandit.c) ----------------------
 * A hand-derived policy gradient is only trustworthy if it is differentiated
 * independently against the same scalar objective. That needs two things the
 * opaque cd_grad will not otherwise give a test: a backward that accepts a
 * caller-supplied dL/dz, and read access to one real parameter tensor. Both are
 * additive and neither touches the training path. */
int cd_test_forward_backward_dlogits(struct cd_model *m, const char *ctx,
                                     const char *const *opts, int n, int label,
                                     struct cd_cache *c, struct cd_grad *g,
                                     const float *dlogits,
                                     char *err, size_t errlen);
const float *cd_test_grad_wq(const struct cd_grad *g, size_t *out_n);
/* The POSITION gradient. Preferred over Wq for a parameter-level check: the
 * Wq gradient is accumulated over the OPTION axis into an R-by-W tensor while the
 * model's Wq is a larger per-option array, so a model index and a gradient index
 * do NOT correspond and cannot be probed one to one. pos has no such axis, so
 * model and gradient indices match exactly. */
const float *cd_test_grad_pos(const struct cd_grad *g, size_t *out_n);
/* Every gradient tensor, in the same order as cd_model_trainable. Used by
 * src/audit_topology.c to classify each parameter as forward-live,
 * backward-dead, or unreachable. Caller frees *out. */
const float **cd_test_grad_tensors(const struct cd_grad *g, size_t *out_count);
/* Copies src over dst exactly as the trainer's best-checkpoint snapshot does, so
 * a test can assert the copy is COMPLETE: every trainable tensor of dst must
 * equal src's afterwards. */
int cd_test_model_copy(struct cd_model *dst, const struct cd_model *src);
/* Length of each tensor from cd_test_grad_tensors, in the same order. Needed
 * because the tensors are NOT all the same size: the XBID norm tensors are R
 * long while Wq/Wk/Wv are R*W, so assuming a uniform length overruns them. */
size_t *cd_test_grad_tensor_sizes(const struct cd_grad *g, const struct cd_model *m);

/* Stress the six XBID norm-gradient buffers: fill with `v`, zero via the real
 * grad_zero, report whether they were distinct allocations and whether any
 * entry survived. */
int cd_test_norm_grad_stress(const struct cd_model *m, float v,
                             long *out_distinct, long *out_stragglers);

/* Resolved context/query projections for a head, plus the partner maps whose
 * gradients mirror when the two are summed. Declared here because the
 * definition needs static helpers from cd_model.c. */
void cd_tw_weights(const struct cd_model *m, int head,
                    const float **Wk, const float **Wv,
                    const float **Wk2, const float **Vv2,
                    int *summed, float *sk, float *sv);
/* 1 when the architecture gives each form its own option->query map. */
int cd_tw_split_query(int arch);
/* 1 when the architecture sums the two forms' query maps on one path. */
int cd_tw_sum_query(int arch);
/* 1 when the architecture sums the two forms' CONTEXT maps onto one path. */
int cd_tw_sum_context(int arch);

/* cd_fact.c — witnessed-fact table. Defined here so main.c can hold one without
 * including anything beyond the two project headers. */
struct cd_fact {
    char *entity;
    char *adj;
    char *context;
};
struct cd_facts {
    struct cd_fact *v;
    size_t n;
};

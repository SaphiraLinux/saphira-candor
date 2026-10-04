/* cd_bandit.c — Stage 0.18 contextual-bandit harness. PREPARED, NOT RUN.
 *
 * ============================================================================
 * RL EXECUTION IS BLOCKED. This file implements the harness and nothing else.
 * No bandit post-training has been performed. There is no result, so nothing in
 * this file can have been tuned against sealed family Z or Z2.
 * ============================================================================
 *
 * The objective, frozen in docs/STAGE_0_18_PREREG.md:
 *
 *     L = policy_loss  -  beta * H(p)  +  alpha * KL(p || q)
 *       policy_loss = -advantage * log p[action]
 *       advantage   = reward - baseline
 *       H(p)        = -sum_k p_k log p_k
 *       KL(p || q)  = sum_k p_k (log p_k - log q_k),  q = FROZEN reference
 *
 * beta and alpha are frozen at 0.1 and 1.0. q is precomputed once from the
 * starting checkpoint (inv50) and never updated, so the KL anchor is a true
 * anchor: at step 0 the policy IS the reference and KL is exactly 0.
 *
 * ---------------------------------------------------------------------------
 * WHY THE GRADIENTS ARE WRITTEN OUT BY HAND
 * ---------------------------------------------------------------------------
 * The obvious implementation would reuse the trainer's cross-entropy backward
 * and add an entropy bonus. That would be wrong twice over. First, CE's upstream
 * gradient is probs - one_hot, which is the gradient of a SUPERVISED loss; a
 * policy gradient is -advantage * dlog p[action]/dz, a different object, and it
 * is signed by whether the sampled action beat the baseline. Second, CE cannot
 * express a KL to a reference distribution at all.
 *
 * So the dL/dz for all three terms is written out explicitly, and the existing
 * backward is reused for everything downstream of the logits via a single
 * additive injection point (cd_backward's dlogits_in, NULL for every
 * pre-existing caller). Derivation, with p = softmax(z):
 *
 *   d/dz_i [ -advantage * log p_a ]
 *       = -advantage * (delta_ia - p_i)
 *
 *   d/dz_i [ H ] = -p_i * (log p_i + H)
 *   so d/dz_i [ -beta * H ] = +beta * p_i * (log p_i + H)
 *
 *   d/dz_i [ KL(p||q) ] = p_i * ( log(p_i/q_i) - KL )
 *
 * Each is verified two ways in the test suite: against a finite-difference
 * gradient of the same scalar loss at the logit level, and end-to-end against
 * central differences on real model PARAMETERS. A hand-derived gradient that is
 * never differentiated independently is just a plausible-looking bug.
 *
 * ONE LANDMINE, established by the parameter-topology audit
 * (docs/TOPOLOGY_AUDIT.md): the AVX2 forward pass reads the DERIVED TRANSPOSE
 * m->Wt, not the projection parameter itself. Any code path that mutates
 * Wq/Wk/Wv without calling cd_model_refresh_transposes forwards STALE WEIGHTS
 * and reports a loss that does not correspond to the parameters. The trainer is
 * safe only because that refresh is the final statement of adam_step. The
 * post-training loop must therefore drive updates through adam_step and must
 * never hand-edit a projection matrix.
 *
 * The scalar loss cd_bandit_loss is deliberately a separate, pure function of
 * (probs, q, action, advantage, beta, alpha) with no model access, so the
 * finite-difference check cannot accidentally agree with the analytic path by
 * sharing code with it.
 */
#define _POSIX_C_SOURCE 200809L
#include "candor.h"
#include "cd_internal.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* candor.h declares bandit_rng_seed, bandit_sample, bandit_baseline_step and
 * the pure objective, so they have exactly one prototype. */

#define BANDIT_EPS 1e-300

/* ---- pure objective pieces ------------------------------------------- */

/* Shannon entropy in nats. p must already be normalised. */
double cd_bandit_entropy(const double *p, int k) {
    double h = 0.0;
    int i;
    for (i = 0; i < k; i++)
        if (p[i] > 0.0) h -= p[i] * log(p[i] > BANDIT_EPS ? p[i] : BANDIT_EPS);
    return h;
}

/* Forward KL(p || q) in nats. q is the frozen reference. */
double cd_bandit_kl(const double *p, const double *q, int k) {
    double s = 0.0;
    int i;
    for (i = 0; i < k; i++) {
        double pi = p[i] > BANDIT_EPS ? p[i] : BANDIT_EPS;
        double qi = q[i] > BANDIT_EPS ? q[i] : BANDIT_EPS;
        s += pi * (log(pi) - log(qi));
    }
    return s;
}

/* The full scalar objective. Pure: no model, no globals, no RNG. This is the
 * function the finite-difference gradient checks differentiate. */
double cd_bandit_loss(const double *p, const double *q, int k, int action,
                      double advantage, double beta, double alpha) {
    double pa = p[action] > BANDIT_EPS ? p[action] : BANDIT_EPS;
    double policy = -advantage * log(pa);
    double h = cd_bandit_entropy(p, k);
    double kl = cd_bandit_kl(p, q, k);
    return policy - beta * h + alpha * kl;
}

/* Analytic dL/dz. Writes k doubles. See the derivation in the file header. */
void cd_bandit_dlogits(const double *p, const double *q, int k, int action,
                       double advantage, double beta, double alpha,
                       double *out) {
    double h = cd_bandit_entropy(p, k);
    double kl = cd_bandit_kl(p, q, k);
    int i;
    for (i = 0; i < k; i++) {
        double pi = p[i] > BANDIT_EPS ? p[i] : BANDIT_EPS;
        double qi = q[i] > BANDIT_EPS ? q[i] : BANDIT_EPS;
        double d = -advantage * ((i == action ? 1.0 : 0.0) - pi);   /* policy */
        d += beta * pi * (log(pi) + h);                            /* -beta*H */
        d += alpha * pi * (log(pi / qi) - kl);                     /* alpha*KL */
        out[i] = d;
    }
}

/* ---- deterministic sampling ------------------------------------------
 * A self-contained xorshift64* so the action sequence is reproducible from the
 * experiment seed alone and does not depend on libc rand(), the platform, or
 * how many times any other part of the program has called rand(). Bandit
 * results must be bit-reproducible or the paired-seed design is worthless. */
void bandit_rng_seed(struct bandit_rng *r, unsigned long long seed) {
    /* splitmix64, so a small integer seed still gives a well-mixed state */
    unsigned long long z = seed + 0x9E3779B97F4A7C15ULL;
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    r->s = z ^ (z >> 31);
    if (!r->s) r->s = 0x9E3779B97F4A7C15ULL;
}

unsigned long long bandit_rng_u64(struct bandit_rng *r) {
    unsigned long long x = r->s;
    x ^= x >> 12; x ^= x << 25; x ^= x >> 27;
    r->s = x;
    return x * 0x2545F4914F6CDD1DULL;
}

/* Sample an index from a normalised distribution by inverse CDF. */
int bandit_sample(const double *p, int k, struct bandit_rng *r) {
    double u = (double)(bandit_rng_u64(r) >> 11) / 9007199254740992.0;
    double acc = 0.0;
    int i;
    for (i = 0; i < k; i++) {
        acc += p[i];
        if (u < acc) return i;
    }
    return k - 1;
}

/* ---- moving baseline -------------------------------------------------
 * EMA of past rewards, frozen at decay 0.9 and initial value 0.5. Frozen in the
 * preregistration; the constants have exactly one home, here. The public
 * prototypes live in candor.h so cd_train.c can use them. */
#define BANDIT_BASELINE_DECAY 0.9
#define BANDIT_BASELINE_INIT  0.5

double bandit_baseline_init(void) { return BANDIT_BASELINE_INIT; }

double bandit_baseline_step(double baseline, double reward) {
    return BANDIT_BASELINE_DECAY * baseline + (1.0 - BANDIT_BASELINE_DECAY) * reward;
}

/* ---- post-training entry point ----------------------------------------
 * Implemented in cd_train.c, which is where grad_zero, grad_scale, the clip and
 * adam_step live. Keeping the loop there is not stylistic: adam_step is the
 * ONLY thing that guarantees cd_model_refresh_transposes runs after a parameter
 * update, and the topology audit established that a projection matrix mutated
 * without that refresh is silently invisible to the forward pass. Updates must
 * therefore go through adam_step and must never hand-edit a weight.
 *
 * Declared here so callers and tests do not need the internal header.
 */
int cd_bandit_posttrain(const char *model_in, const char *data_path,
                        const char *model_out, const char *log_path,
                        double beta, double alpha, int epochs, int batch,
                        double lr, unsigned long long seed,
                        const char *mask_spec, double wrong_pen,
                        char *err, size_t errlen);

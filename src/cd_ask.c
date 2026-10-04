/* cd_ask.c -- the ask surface, moved verbatim from cd_train.c.
 *
 * This tree contains NO training code: cd_train.c and cd_bandit.c are not
 * compiled here, so no weight can be mutated by this binary, by construction
 * rather than by flag discipline. What moved is only human inspection
 * (ask_show) and its entry point (cd_ask); both call the ordinary forward
 * path and consult no dataset label.
 */
#define _POSIX_C_SOURCE 200809L
#include "candor.h"
#include "cd_internal.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

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

/* Machine-readable ask. v0's [{option, probability, logit}] shape plus the verdict
 * v0 never had: "remembered" (exact witness, no forward run), "guessed" (forward run
 * after a silent witness check), or "scored" (no witness file consulted at all).
 * Prints one JSON object to stdout. Returns 0 on success. */
static int ask_json_loaded(struct cd_model *jm, const char *input,
                           const char *const *cands, int n_cand, int head_mask,
                           const char *verdict) {
    const char *jopts[CD_MAX_OPT];
    float jlog[CD_MAX_OPT];
    double jmax = 0.0, jsum = 0.0;
    int jn = 0, ji, jbest = 0;
    if (!jm) { fprintf(stderr, "ask: bad model\n"); return 1; }
    for (ji = 0; ji < n_cand && ji < CD_MAX_OPT; ji++) jopts[ji] = cands[ji];
    jn = ji;
    if (jn < 1) { fprintf(stderr, "ask: need at least one --candidate\n");
                  cd_model_free(jm); return 1; }
    if (cd_probe_logits(jm, input, jopts, jn,
                         head_mask == 3 ? CD_HEAD_FWD : head_mask,
                         jlog, (size_t)CD_MAX_OPT) != 0) {
        fprintf(stderr, "ask: forward failed\n");
        cd_model_free(jm);
        return 1;
    }
    for (ji = 0; ji < jn; ji++)
        if (jlog[ji] > jmax || ji == 0) jmax = jlog[ji];
    for (ji = 0; ji < jn; ji++) jsum += exp((double)jlog[ji] - jmax);
    for (ji = 0; ji < jn; ji++) {
        double p = exp((double)jlog[ji] - jmax) / jsum;
        if (p > exp((double)jlog[jbest] - jmax) / jsum) jbest = ji;
    }
    printf("{\"verdict\": \"%s\", \"options\": [", verdict);
    for (ji = 0; ji < jn; ji++) {
        double p = exp((double)jlog[ji] - jmax) / jsum;
        printf("%s{\"option\": \"", ji ? ", " : "");
        for (const char *q = jopts[ji]; *q; q++) {
            if (*q == '"' || *q == '\\') putchar('\\');
            putchar(*q);
        }
        printf("\", \"probability\": %.10f, \"logit\": %.10f}", p, (double)jlog[ji]);
    }
    printf("], \"answer\": \"");
    for (const char *q = jopts[jbest]; *q; q++) {
        if (*q == '"' || *q == '\\') putchar('\\');
        putchar(*q);
    }
    printf("\"}\n");
    cd_model_free(jm);
    return 0;
}

/* Loaded-model entry points for the self-contained binary: the model bytes were
 * linked at build time, so there is no path to open. ask_show stays shared;
 * these two are the only new surface. */
int cd_ask_show(const struct cd_model *m, const char *context, const char *question,
                char *const *cands, int n_cand, int head_mask) {
    char input[4608];
    if (!m || !context || n_cand < 1) return 2;
    if (question && *question) snprintf(input, sizeof input, "%s %s", context, question);
    else snprintf(input, sizeof input, "%s", context);
    if (head_mask == 3) {
        printf("\n== both heads (diagnostic; the model was not told which to use) ==\n");
        ask_show(m, input, cands, n_cand, CD_HEAD_FWD);
        printf("\n");
        ask_show(m, input, cands, n_cand, CD_HEAD_INV);
    } else {
        printf("\n");
        ask_show(m, input, cands, n_cand, head_mask);
    }
    return 0;
}

int cd_ask_mem(struct cd_model *m, const char *context, const char *question,
               char *const *cands, int n_cand, int head_mask) {
    char input[4608];
    const char *opts[CD_MAX_OPT];
    int ji, jn = 0;
    if (!m || !context || n_cand < 1) return 2;
    for (ji = 0; ji < n_cand && ji < CD_MAX_OPT; ji++) opts[ji] = cands[ji];
    jn = ji;
    if (question && *question) snprintf(input, sizeof input, "%s %s", context, question);
    else snprintf(input, sizeof input, "%s", context);
    return ask_json_loaded(m, input, opts, jn, head_mask, "guessed");
}

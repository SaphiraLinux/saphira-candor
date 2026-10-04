/* main.c — CLI for the candor controlled instrument.
 *
 *   sn train    --data F --val F --out F [config] [switches]
 *   sn eval     --model F --test F
 *   sn gradcheck --data F [config]
 *   sn permcheck --model F --data F      permutation equivariance audit
 *
 * Defaults reproduce the upstream trainer defaults so that a bare invocation is
 * directly comparable with the classic trainer.
 */
#define _POSIX_C_SOURCE 200809L
#include "candor.h"
#include "cd_internal.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void usage(void) {
    fprintf(stderr,
"usage:\n"
"  candor train --data F --val F [--out F] [--W n] [--R n] [--C n] [--O n] [--q2 | --q3 | --q4 | --q8] (storage only; highest fidelity wins)\n"
"          [--epochs n] [--batch n] [--lr f] [--wd f] [--seed n]\n"
"          [--arch baseline|interact|xbid|twohead] [--mode legacy|correct]\n"
"  candor bandit --arm baseline|entropy|kl|both --model F --data F [--out F]\n"
"          [--log F] [--seed n] [--epochs n] [--batch n] [--lr f]\n"
"          Sensitivity (preregistered, NEVER primary, both regularizers on):\n"
"            --arm sens-b003-a03 | sens-b003-a30 | sens-b03-a03 | sens-b03-a30\n"
"          Coefficients come from the arm identity; there are no override flags.\n"
"          Stage 0.18 contextual-bandit post-training from a frozen reference.\n"
"          Defaults are the preregistered values; the resolved set is echoed\n"
"          before any work. beta/alpha follow the arm unless overridden.\n"
"          The starting checkpoint is opened read-only and never rewritten.\n"
"          eval also takes [--route fwd|inv] for the twohead diagnostic\n"
          "          and [--dump-per-example F] for the Stage 0.17 per-example\n"
          "          calibration record (additive; summary output is unchanged)\n"
"          [--clip f] [--seed-drives-order] [--no-clamp] [--decay-exclude-norm]\n"
"          [--grad-check]\n"
"  candor eval --model F --test F\n"
"  candor signals --model F --test F     per-example predictor dump (measurement)\n"
"  candor gradcheck --data F [config]\n"
"  candor permcheck --model F --data F [--n 200]\n"
"  candor ask MODEL --context T [--question T] --candidate A [--candidate B ...]\n"
"         [--head forward|inverse|both]      one-shot inspection\n"
"  candor ask MODEL                              interactive mode\n"
"  candor --help | --version | --about [--model F] [--witness F]\n");
}

/* CANDOR_VERSION lives in candor.h (shared by both CLIs). */

/* ---- minimal SHA-256 (FIPS 180-4), for --about file hashing ---- */
typedef struct { uint32_t h[8]; uint64_t len; unsigned char buf[64]; size_t n; } cd_sha256_t;
static uint32_t cd_ror(uint32_t x, int k) { return (x >> k) | (x << (32 - k)); }
static void cd_sha256_block(cd_sha256_t *c, const unsigned char *p) {
    static const uint32_t K[64] = {
        0x428a2f98,0x71374491,0xb5c0fbcf,0xe9b5dba5,0x3956c25b,0x59f111f1,0x923f82a4,0xab1c5ed5,
        0xd807aa98,0x12835b01,0x243185be,0x550c7dc3,0x72be5d74,0x80deb1fe,0x9bdc06a7,0xc19bf174,
        0xe49b69c1,0xefbe4786,0x0fc19dc6,0x240ca1cc,0x2de92c6f,0x4a7484aa,0x5cb0a9dc,0x76f988da,
        0x983e5152,0xa831c66d,0xb00327c8,0xbf597fc7,0xc6e00bf3,0xd5a79147,0x06ca6351,0x14292967,
        0x27b70a85,0x2e1b2138,0x4d2c6dfc,0x53380d13,0x650a7354,0x766a0abb,0x81c2c92e,0x92722c85,
        0xa2bfe8a1,0xa81a664b,0xc24b8b70,0xc76c51a3,0xd192e819,0xd6990624,0xf40e3585,0x106aa070,
        0x19a4c116,0x1e376c08,0x2748774c,0x34b0bcb5,0x391c0cb3,0x4ed8aa4a,0x5b9cca4f,0x682e6ff3,
        0x748f82ee,0x78a5636f,0x84c87814,0x8cc70208,0x90befffa,0xa4506ceb,0xbef9a3f7,0xc67178f2 };
    uint32_t w[64], a, b, c_, d, e, f, g, h;
    int i;
    for (i = 0; i < 16; i++)
        w[i] = ((uint32_t)p[4*i] << 24) | ((uint32_t)p[4*i+1] << 16) |
               ((uint32_t)p[4*i+2] << 8) | (uint32_t)p[4*i+3];
    for (i = 16; i < 64; i++) {
        uint32_t s0 = cd_ror(w[i-15], 7) ^ cd_ror(w[i-15], 18) ^ (w[i-15] >> 3);
        uint32_t s1 = cd_ror(w[i-2], 17) ^ cd_ror(w[i-2], 19) ^ (w[i-2] >> 10);
        w[i] = w[i-16] + s0 + w[i-7] + s1;
    }
    a=c->h[0]; b=c->h[1]; c_=c->h[2]; d=c->h[3];
    e=c->h[4]; f=c->h[5]; g=c->h[6]; h=c->h[7];
    for (i = 0; i < 64; i++) {
        uint32_t S1 = cd_ror(e,6)^cd_ror(e,11)^cd_ror(e,25), ch = (e&f)^(~e&g), t1 = h+S1+ch+K[i]+w[i];
        uint32_t S0 = cd_ror(a,2)^cd_ror(a,13)^cd_ror(a,22), mj = (a&b)^(a&c_)^(b&c_), t2 = S0+mj;
        h=g; g=f; f=e; e=d+t1; d=c_; c_=b; b=a; a=t1+t2;
    }
    c->h[0]+=a; c->h[1]+=b; c->h[2]+=c_; c->h[3]+=d;
    c->h[4]+=e; c->h[5]+=f; c->h[6]+=g; c->h[7]+=h;
}
static void cd_sha256_init(cd_sha256_t *c) {
    c->h[0]=0x6a09e667; c->h[1]=0xbb67ae85; c->h[2]=0x3c6ef372; c->h[3]=0xa54ff53a;
    c->h[4]=0x510e527f; c->h[5]=0x9b05688c; c->h[6]=0x1f83d9ab; c->h[7]=0x5be0cd19;
    c->len = 0; c->n = 0;
}
static void cd_sha256_update(cd_sha256_t *c, const unsigned char *p, size_t n) {
    c->len += n;
    while (n) {
        size_t t = 64 - c->n < n ? 64 - c->n : n;
        memcpy(c->buf + c->n, p, t);
        c->n += t; p += t; n -= t;
        if (c->n == 64) { cd_sha256_block(c, c->buf); c->n = 0; }
    }
}
static void cd_sha256_file(const char *path, char out[65], char *err, size_t errlen) {
    cd_sha256_t c;
    unsigned char buf[8192], pad[64];
    uint64_t bits;
    size_t r, need;
    int i;
    FILE *f = fopen(path, "rb");
    if (!f) { snprintf(err, errlen, "cannot read %s", path); out[0] = '\0'; return; }
    cd_sha256_init(&c);
    while ((r = fread(buf, 1, sizeof buf, f)) > 0) cd_sha256_update(&c, buf, r);
    fclose(f);
    memset(pad, 0, sizeof pad);
    bits = c.len * 8;
    need = (c.n < 56) ? 56 - c.n : 120 - c.n;
    pad[0] = 0x80;
    cd_sha256_update(&c, pad, need);
    for (i = 7; i >= 0; i--) {
        unsigned char b = (unsigned char)(bits >> (i * 8));
        cd_sha256_update(&c, &b, 1);
    }
    for (i = 0; i < 32; i++)
        sprintf(out + 2 * i, "%02x", (unsigned)((c.h[i>>2] >> ((3-(i&3))*8)) & 0xff));
    out[64] = '\0';
}

static int arg_eq(const char *a, const char *b) { return strcmp(a, b) == 0; }

/* Machine-readable ask. v0's [{option, probability, logit}] shape plus the verdict
 * v0 never had: "remembered" (exact witness, no forward run), "guessed" (forward run
 * after a silent witness check), or "scored" (no witness file consulted at all).
 * Prints one JSON object to stdout. Returns 0 on success. */
static int ask_json(const char *model, const char *context, const char *question,
                    char *const *cands, int n_cand, int head_mask, const char *verdict) {
    struct cd_model *jm;
    const char *jopts[CD_MAX_OPT];
    float jlog[CD_MAX_OPT];
    char err[512] = {0};
    char jinput[4608];
    double jmax = 0.0, jsum = 0.0;
    int jn = 0, ji, jbest = 0;
    jm = cd_load(model, err, sizeof err);
    if (!jm) { fprintf(stderr, "ask: %s\n", err); return 1; }
    for (ji = 0; ji < n_cand && ji < CD_MAX_OPT; ji++) jopts[ji] = cands[ji];
    jn = ji;
    if (jn < 1) { fprintf(stderr, "ask: need at least one --candidate\n");
                  cd_model_free(jm); return 1; }
    if (context && question) snprintf(jinput, sizeof jinput, "%s %s", context, question);
    else snprintf(jinput, sizeof jinput, "%s", context ? context : "");
    if (cd_probe_logits(jm, jinput, jopts, jn,
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

/* One-shot inspection (REPL entry below reuses the same primitives). */
static int ask_repl(const char *model) {
    char line[4096], ctx[1024], ques[1024], c1[512], c2[512];
    int head = CD_HEAD_FWD;
    printf("interactive ask. model loaded once.\n"
           "  c <text>     set context\n"
           "  q <text>     set question (appended to context)\n"
           "  a <text>     set candidate A\n"
           "  b <text>     set candidate B\n"
           "  h f|i|both   choose head (default forward)\n"
           "  go           ask with the current A and B\n"
           "  q!           quit\n");
    ctx[0] = ques[0] = c1[0] = c2[0] = '\0';
    for (;;) {
        char *p;
        printf("ask> "); fflush(stdout);
        if (!fgets(line, sizeof line, stdin)) break;
        p = line + strspn(line, " \t");
        { size_t L = strlen(p); while (L && (p[L-1] == '\n' || p[L-1] == '\r')) p[--L] = '\0'; }
        if (!*p) continue;
        if (arg_eq(p, "q!") || arg_eq(p, "quit") || arg_eq(p, "exit")) break;
        if (p[0] == 'c' && p[1] == ' ') { snprintf(ctx, sizeof ctx, "%s", p + 2); continue; }
        if (p[0] == 'q' && p[1] == ' ') { snprintf(ques, sizeof ques, "%s", p + 2); continue; }
        if (p[0] == 'a' && p[1] == ' ') { snprintf(c1, sizeof c1, "%s", p + 2); continue; }
        if (p[0] == 'b' && p[1] == ' ') { snprintf(c2, sizeof c2, "%s", p + 2); continue; }
        if (p[0] == 'h' && p[1] == ' ') {
            const char *h = p + 2;
            head = arg_eq(h, "both") ? 3 : (arg_eq(h, "i") || arg_eq(h, "inverse"))
                 ? CD_HEAD_INV : CD_HEAD_FWD;
            continue;
        }
        if (arg_eq(p, "go")) {
            char *cs[2];
            if (!ctx[0] || !c1[0] || !c2[0]) {
                printf("  need at least a context and two candidates\n");
                continue;
            }
            cs[0] = c1; cs[1] = c2;
            { char err[512] = {0};
              if (cd_ask(model, ctx, ques[0] ? ques : NULL, cs, 2, head,
                         err, sizeof err) != 0)
                  printf("  ask failed: %s\n", err); }
            continue;
        }
        printf("  unknown command. h for help.\n");
    }
    printf("\nbye\n");
    return 0;
}

/* Permutation-equivariance audit: the logits must be equivariant, i.e. the
 * probability attached to a given option string may not depend on where that
 * option sits in the list. Reported as the max |p - p'| over options. */
static int permcheck(const char *model_path, const char *data_path, int n_max,
                     char *err, size_t err_len) {
    struct cd_model *m = cd_load(model_path, err, err_len);
    struct cd_dataset ds;
    struct cd_cache c;
    size_t i;
    double worst = 0.0, sum = 0.0;
    int count = 0, bad = 0;
    if (!m) { fprintf(stderr, "permcheck: %s\n", err); return -1; }
    if (cd_data_load(data_path, &ds, err, err_len) != 0) {
        fprintf(stderr, "permcheck: %s\n", err); cd_model_free(m); return -1;
    }
    {   /* the cache is sized from the model, so the data must fit inside it */
        int need = cd_data_max_nopt(&ds);
        if (need > m->d.n_opt) {
            fprintf(stderr, "permcheck: data has %d options, model sized for %d\n",
                    need, m->d.n_opt);
            cd_data_free(&ds); cd_model_free(m); return -1;
        }
    }
    cd_cache_alloc(&c, m);
    printf("{\"arch\": \"%s\", \"permutation_test\": true}\n",
           m->arch == CD_ARCH_INTERACT ? "interact"
           : m->arch == CD_ARCH_XBID ? "xbid" : "baseline");
    for (i = 0; i < ds.n && count < n_max; i++) {
        int no = cd_data_nopt(&ds, i), a;
        const char *opts[CD_MAX_OPT];
        float p0[CD_MAX_OPT], p1[CD_MAX_OPT];
        if (no > CD_MAX_OPT) no = CD_MAX_OPT;
        for (a = 0; a < no; a++) opts[a] = cd_data_opt(&ds, i, a);
        cd_forward(m, cd_data_ctx(&ds, i), opts, no, &c);
        for (a = 0; a < no; a++) p0[a] = c.probs[a];
        /* reverse the option order */
        for (a = 0; a < no; a++) opts[a] = cd_data_opt(&ds, i, no - 1 - a);
        cd_forward(m, cd_data_ctx(&ds, i), opts, no, &c);
        for (a = 0; a < no; a++) p1[no - 1 - a] = c.probs[a];
        for (a = 0; a < no; a++) {
            double d = fabs((double)p0[a] - (double)p1[a]);
            if (d > worst) worst = d;
            sum += d;
            if (d > 1e-5) bad++;
        }
        count++;
    }
    printf("{\"rows\": %d, \"perm_max_abs_diff\": %.3e, \"perm_mean_abs_diff\": %.3e, "
           "\"perm_violations\": %d, \"equivariant\": %s}\n",
           count, worst, count ? sum / (count * 2.0) : 0.0, bad,
           bad == 0 ? "true" : "false");
    cd_cache_free(&c);
    cd_data_free(&ds);
    cd_model_free(m);
    return 0;
}

int main(int argc, char **argv) {
    struct cd_cfg cfg;
    char err[512] = {0};
    const char *cmd = argc > 1 ? argv[1] : "";
    const char *data = NULL, *val = NULL, *out = NULL, *test = NULL, *model = NULL;
    int i, gradcheck_mode = 0, permcheck_mode = 0, n_max = 200, route = 0;
    int n_cand = 0, head_mask = CD_HEAD_FWD;
    int about_mode = 0;
    char *cands[CD_MAX_OPT];
    const char *context = NULL, *question = NULL, *witness = NULL;
    int json_out = 0;
    /* Stage 0.17: per-example calibration record dump, additive and opt-in. */
    const char *dump_per_example = NULL;
    /* Stage 0.18: contextual-bandit post-training entry point. Every default is
     * the preregistered value; the resolved set is echoed before any work. */
    const char *arm = NULL, *log = NULL, *mask = "all";
    double wrong_pen = 0.0;
    int b_epochs = 1, b_batch = 64;
    double b_lr = 0.002;
    int arm_sensitivity = 0;

    if (argc < 2) { usage(); return 2; }

    /* upstream trainer defaults */
    cfg.d.W = 64; cfg.d.R = 64; cfg.d.C = 192; cfg.d.O = 32; cfg.d.n_opt = 4;
    cfg.arch = CD_ARCH_BASELINE;
    cfg.mode = CD_MODE_LEGACY;
    cfg.epochs = 8; cfg.batch_size = 64;
    cfg.lr = 0.002; cfg.wd = 1e-4; cfg.seed = 7;   /* matches upstream default */
    cfg.grad_check = 0; cfg.seed_drives_order = 0; cfg.no_clamp = 0;
    cfg.init_model = NULL; cfg.freeze_trunk = 0; cfg.force_train_head = -1;
    cfg.clip = 0.0; cfg.decay_exclude_norm = 0; cfg.q3 = 0; cfg.q8 = 0;
    cfg.q4 = 0; cfg.q2 = 0;

    /* `ask` takes the model path positionally, so consume it before the flags.
     * Bare flags (`candor --help`) carry no subcommand: argv[1] starts with -,
     * so parse from 1 instead of skipping it. Without this, bare flags skip the
     * loop entirely and fall through to usage+exit-2 -- every --help/--version
     * invocation failed exactly this way. */
    i = 2;
    if (arg_eq(cmd, "ask") && argc > 2 && argv[2][0] != '-') {
        model = argv[2];
        i = 3;
    } else if (argc > 1 && argv[1][0] == '-') {
        i = 1;
    }
    for (; i < argc; i++) {
        const char *a = argv[i];
        const char *v = (i + 1 < argc) ? argv[i + 1] : NULL;
#define NEXT() (v ? v : (usage(), exit(2), (char *)NULL))
        if      (arg_eq(a, "--data")) data = NEXT(), i++;
        else if (arg_eq(a, "--val")) val = NEXT(), i++;
        else if (arg_eq(a, "--out")) out = NEXT(), i++;
        else if (arg_eq(a, "--test")) test = NEXT(), i++;
        else if (arg_eq(a, "--dump-per-example")) dump_per_example = NEXT(), i++;
        else if (arg_eq(a, "--model")) model = NEXT(), i++;
        else if (arg_eq(a, "--W")) cfg.d.W = (uint32_t)strtoul(NEXT(), NULL, 10), i++;
        else if (arg_eq(a, "--R")) cfg.d.R = (uint32_t)strtoul(NEXT(), NULL, 10), i++;
        else if (arg_eq(a, "--C")) cfg.d.C = (uint32_t)strtoul(NEXT(), NULL, 10), i++;
        else if (arg_eq(a, "--O")) cfg.d.O = (uint32_t)strtoul(NEXT(), NULL, 10), i++;
        else if (arg_eq(a, "--n-opt")) cfg.d.n_opt = (int)strtol(NEXT(), NULL, 10), i++;
        else if (arg_eq(a, "--epochs")) {
            cfg.epochs = (int)strtol(NEXT(), NULL, 10); b_epochs = cfg.epochs; i++;
        }
        else if (arg_eq(a, "--batch")) {
            cfg.batch_size = (int)strtol(NEXT(), NULL, 10); b_batch = cfg.batch_size; i++;
        }
        else if (arg_eq(a, "--lr")) {
            cfg.lr = strtod(NEXT(), NULL); b_lr = cfg.lr; i++;
        }
        /* Stage 0.18 only. There are deliberately NO --beta/--alpha flags:
         * the arm identity alone determines the coefficients, so a run cannot
         * be mislabelled. An earlier revision accepted overrides, which made it
         * possible to label a sensitivity point as the primary `both` arm; that
         * run is retained but superseded, and the override path is gone. */
        else if (arg_eq(a, "--arm"))   arm = NEXT(), i++;
        else if (arg_eq(a, "--log"))   log = NEXT(), i++;
        /* Exploratory probes. Both default to the sealed Stage 0.18 behaviour, so
         * a run that sets neither reproduces a primary arm bit-identically. */
        else if (arg_eq(a, "--mask"))      mask = NEXT(), i++;
        else if (arg_eq(a, "--wrong-pen")) wrong_pen = strtod(NEXT(), NULL), i++;
        else if (arg_eq(a, "--wd")) cfg.wd = strtod(NEXT(), NULL), i++;
        else if (arg_eq(a, "--seed")) cfg.seed = strtoull(NEXT(), NULL, 10), i++;
        else if (arg_eq(a, "--n")) n_max = (int)strtol(NEXT(), NULL, 10), i++;
        else if (arg_eq(a, "--route")) {
            route = arg_eq(NEXT(), "inv") ? CD_HEAD_INV : CD_HEAD_FWD; i++;
        }
        else if (arg_eq(a, "--context")) context = NEXT(), i++;
        else if (arg_eq(a, "--witness")) witness = NEXT(), i++;
        else if (arg_eq(a, "--json")) json_out = 1;
        else if (arg_eq(a, "--question")) question = NEXT(), i++;
        else if (arg_eq(a, "--candidate")) {
            if (n_cand < CD_MAX_OPT) cands[n_cand++] = NEXT();
            i++;
        }
        else if (arg_eq(a, "--head")) {
            const char *h = NEXT();
            if (arg_eq(h, "both")) head_mask = 3;
            else if (arg_eq(h, "inverse")) head_mask = CD_HEAD_INV;
            else head_mask = CD_HEAD_FWD;
            i++;
        }
        else if (arg_eq(a, "--arch")) {
            const char *w = NEXT();
            if (arg_eq(w, "interact")) cfg.arch = CD_ARCH_INTERACT;
            else if (arg_eq(w, "xbid")) cfg.arch = CD_ARCH_XBID;
            else if (arg_eq(w, "twohead")) cfg.arch = CD_ARCH_TWOHEAD;
            else if (arg_eq(w, "sumhead")) cfg.arch = CD_ARCH_SUMHEAD;
            else if (arg_eq(w, "tw_sk"))   cfg.arch = CD_ARCH_TW_SK;
            else if (arg_eq(w, "tw_sv"))   cfg.arch = CD_ARCH_TW_SV;
            else if (arg_eq(w, "tw_skv"))  cfg.arch = CD_ARCH_TW_SKV;
            else if (arg_eq(w, "sum_k"))   cfg.arch = CD_ARCH_SUM_K;
            else if (arg_eq(w, "sum_v"))   cfg.arch = CD_ARCH_SUM_V;
            else if (arg_eq(w, "sum_kv"))  cfg.arch = CD_ARCH_SUM_KV;
            else cfg.arch = CD_ARCH_BASELINE;
            i++;
        }
        else if (arg_eq(a, "--mode"))
            cfg.mode = arg_eq(NEXT(), "correct") ? CD_MODE_CORRECT : CD_MODE_LEGACY, i++;
        else if (arg_eq(a, "--clip")) cfg.clip = strtod(NEXT(), NULL), i++;
        else if (arg_eq(a, "--seed-drives-order")) cfg.seed_drives_order = 1;
        else if (arg_eq(a, "--no-clamp")) cfg.no_clamp = 1;
        else if (arg_eq(a, "--init-model")) cfg.init_model = NEXT(), i++;
        else if (arg_eq(a, "--freeze-trunk")) cfg.freeze_trunk = 1;
        else if (arg_eq(a, "--train-head")) {
            const char *h = NEXT();
            cfg.force_train_head = arg_eq(h, "inv") || arg_eq(h, "inverse")
                                 ? CD_HEAD_INV : CD_HEAD_FWD;
            i++;
        }
        else if (arg_eq(a, "--decay-exclude-norm")) cfg.decay_exclude_norm = 1;
        else if (arg_eq(a, "--q3")) cfg.q3 = 1;
        else if (arg_eq(a, "--q8")) cfg.q8 = 1;
        else if (arg_eq(a, "--q4")) cfg.q4 = 1;
        else if (arg_eq(a, "--q2")) cfg.q2 = 1;
        else if (arg_eq(a, "--grad-check")) {
            /* optional head selector: --grad-check 2 probes the inverse head */
            if (i + 1 < argc && argv[i + 1][0] >= '1' && argv[i + 1][0] <= '2') {
                gradcheck_mode = argv[i + 1][0] - '0'; i++;
            } else gradcheck_mode = 1;
        }
        else if (arg_eq(a, "--permcheck")) permcheck_mode = 1;
        else if (arg_eq(a, "--help") || arg_eq(a, "-h")) { usage(); return 0; }
        else if (arg_eq(a, "--version")) {
            printf("candor %s (engine ask/train; container CDR1 v1-v5)\n",
                   CANDOR_VERSION);
            return 0;
        }
        else if (arg_eq(a, "--about")) about_mode = 1;
        else { fprintf(stderr, "unknown argument: %s\n", a); usage(); return 2; }
#undef NEXT
    }

    if (gradcheck_mode || arg_eq(cmd, "gradcheck")) {
        cfg.grad_check = gradcheck_mode ? gradcheck_mode : 1;
        if (!data) { fprintf(stderr, "gradcheck needs --data\n"); return 2; }
        return cd_train(&cfg, data, data, NULL, NULL, err, sizeof err) == 0 ? 0 : 1;
    }
    if (permcheck_mode || arg_eq(cmd, "permcheck")) {
        if (!model || !data) { fprintf(stderr, "permcheck needs --model and --data\n"); return 2; }
        return permcheck(model, data, n_max, err, sizeof err) == 0 ? 0 : 1;
    }
    if (about_mode) {
        char mh[65] = {0}, wh[65] = {0}, e2[512] = {0};
        printf("candor %s\nengine: ask/train build (training present in this binary)\n",
               CANDOR_VERSION);
        printf("container: CDR1 v1-v5, exactness baseline x86-64\n");
        if (model) {
            cd_sha256_file(model, mh, e2, sizeof e2);
            if (!mh[0]) { fprintf(stderr, "--about: %s\n", e2); return 1; }
            printf("model: %s  sha256=%s\n", model, mh);
        }
        if (witness) {
            cd_sha256_file(witness, wh, e2, sizeof e2);
            if (!wh[0]) { fprintf(stderr, "--about: %s\n", e2); return 1; }
            printf("witness: %s  sha256=%s\n", witness, wh);
        }
        if (!model && !witness)
            printf("(no content files supplied: engine provenance only)\n");
        return 0;
    }
    if (arg_eq(cmd, "ask")) {
        if (!model) { fprintf(stderr, "usage: sn ask MODEL [flags]\n"); return 2; }
        if (witness) {
            /* Fact-first: resolve the entity from the context, then exact-match.
             * A single witnessed candidate answers here with its source. Zero or
             * several matches fall through to the vector forward pass below, which
             * is then a GUESSED answer by construction. */
            struct cd_facts facts;
            const char *entity, *hit = NULL, *src = NULL;
            int rc;
            facts.v = NULL;
            facts.n = 0;
            if (cd_fact_load(witness, &facts, err, sizeof err) != 0) {
                fprintf(stderr, "ask: %s\n", err);
                return 1;
            }
            entity = cd_fact_resolve_entity(&facts, context ? context : "");
            if (entity && n_cand > 0 &&
                cd_fact_lookup(&facts, entity, cands, n_cand, &hit, &src)) {
                if (json_out) {
                    printf("{\"verdict\": \"remembered\", \"answer\": \"");
                    for (const char *q = hit; *q; q++) {
                        if (*q == '"' || *q == '\\') putchar('\\');
                        putchar(*q);
                    }
                    printf("\", \"source\": \"");
                    if (src) for (const char *q = src; *q; q++) {
                        if (*q == '"' || *q == '\\') putchar('\\');
                        putchar(*q);
                    }
                    printf("\"}\n");
                    cd_fact_free(&facts);
                    return 0;
                }
                printf("\n  REMEMBERED: %s is %s.\n", entity, hit);
                printf("    source: %s\n", src && *src ? src : "(witnessed, no text kept)");
                printf("    (%zu witnessed facts consulted; forward pass not run)\n",
                       cd_fact_count(&facts));
                cd_fact_free(&facts);
                return 0;
            }
            if (!entity)
                printf("\n  no known entity in context -- vector guess follows (GUESSED):\n");
            else
                printf("\n  nothing exactly witnessed for '%s' among %d candidate(s) -- "
                       "vector guess follows (GUESSED):\n", entity, n_cand);
            if (json_out) {
                int jr = ask_json(model, context, question, cands, n_cand,
                                  head_mask, "guessed");
                cd_fact_free(&facts);
                return jr;
            }
            rc = 0;
            if (n_cand == 0) { cd_fact_free(&facts); return ask_repl(model); }
            if (cd_ask(model, context, question, cands, n_cand, head_mask,
                       err, sizeof err) != 0) {
                fprintf(stderr, "ask: %s\n", err);
                rc = 1;
            }
            cd_fact_free(&facts);
            return rc;
        }
        if (n_cand == 0) { return ask_repl(model); }
        if (json_out)
            return ask_json(model, context, question, cands, n_cand, head_mask,
                            witness ? "guessed" : "scored");
        if (cd_ask(model, context, question, cands, n_cand, head_mask,
                   err, sizeof err) != 0) {
            fprintf(stderr, "ask: %s\n", err);
            return 1;
        }
        return 0;
    }
    if (arg_eq(cmd, "signals")) {
        if (!model || !test) {
            fprintf(stderr, "signals needs --model and --test\n");
            return 2;
        }
        return cd_signals(model, test, err, sizeof err) == 0 ? 0 : 1;
    }
    if (arg_eq(cmd, "eval")) {
        if (!model || !test) { fprintf(stderr, "eval needs --model and --test\n"); return 2; }
        if (cd_eval_head(model, test, route, err, sizeof err) != 0) {
            fprintf(stderr, "eval: %s\n", err);
            return 1;
        }
        /* Stage 0.17: additive and off by default. The summary above is
         * produced by the untouched path whether or not this runs. */
        if (dump_per_example) {
            if (cd_eval_dump(model, test, route, dump_per_example, err, sizeof err) != 0) {
                fprintf(stderr, "eval: %s\n", err);
                return 1;
            }
        }
        return 0;
    }
    if (arg_eq(cmd, "bandit")) {
        /* The four preregistered arms. beta/alpha are DERIVED from the arm name
         * rather than passed independently, so the four arms cannot be
         * mislabelled or given a coefficient mix that is not in the prereg. */
        double beta, alpha;
        if (!model || !data || !arm) {
            fprintf(stderr, "bandit needs --arm, --model and --data\n");
            return 2;
        }
        if (arg_eq(arm, "baseline"))      { beta = 0.0; alpha = 0.0; }
        else if (arg_eq(arm, "entropy"))  { beta = 0.1; alpha = 0.0; }
        else if (arg_eq(arm, "kl"))       { beta = 0.0; alpha = 1.0; }
        else if (arg_eq(arm, "both"))     { beta = 0.1; alpha = 1.0; }
        /* Sensitivity configurations get their OWN explicit identities rather
         * than borrowing `both` with overrides, so the run can never be
         * confused with a primary arm. Both regularizers are active in all four;
         * that is declared in the header as a machine-readable field. */
        else if (arg_eq(arm, "sens-b003-a03")) { beta = 0.03; alpha = 0.3;  arm_sensitivity = 1; }
        else if (arg_eq(arm, "sens-b003-a30")) { beta = 0.03; alpha = 3.0;  arm_sensitivity = 1; }
        else if (arg_eq(arm, "sens-b03-a03"))  { beta = 0.3;  alpha = 0.3;  arm_sensitivity = 1; }
        else if (arg_eq(arm, "sens-b03-a30"))  { beta = 0.3;  alpha = 3.0;  arm_sensitivity = 1; }
        else {
            fprintf(stderr, "bandit: --arm must be one of\n"
                            "  primary:    baseline entropy kl both\n"
                            "  sensitivity (prereg, never primary):\n"
                            "              sens-b003-a03 sens-b003-a30 sens-b03-a03 sens-b03-a30\n");
            return 2;
        }

        /* Echo the resolved configuration BEFORE any work, so a run's log can be
         * audited against docs/STAGE_0_18_PREREG.md without reading the code.
         * Weight decay is NOT a preregistered coefficient: the loop fixes it at
         * 0.0, identically for all four arms, so it cannot confound the
         * arm-versus-baseline contrast that is the actual Stage 0.18 test. */
        printf("{\"stage\": \"0.18\", \"arm\": \"%s\", \"class\": \"%s\", "
               "\"is_primary\": %s, \"both_regularizers_active\": %s, "
               "\"beta\": %.17g, \"alpha\": %.17g, "
               "\"epochs\": %d, \"batch\": %d, \"lr\": %.17g, "
               "\"wd\": 0, \"lambda\": 0, \"reward\": \"accuracy\", "
               "\"action\": \"sampled\", \"ema\": \"0.9 from 0.5\", "
               "\"coefficients_from\": \"arm identity only; no override flags exist\", "
               "\"mask\": \"%s\", \"wrong_pen\": %.17g, "
               "\"reference\": \"%s\", \"data\": \"%s\", \"out\": \"%s\", "
               "\"seed\": %llu}\n",
               arm, arm_sensitivity ? "sensitivity" : "primary",
               arm_sensitivity ? "false" : "true",
               (beta > 0.0 && alpha > 0.0) ? "true" : "false",
               beta, alpha, b_epochs, b_batch, b_lr, mask, wrong_pen, model, data,
               out ? out : "(none)", (unsigned long long)cfg.seed);
        fflush(stdout);

        if (cd_bandit_posttrain(model, data, out, log, beta, alpha,
                                b_epochs, b_batch, b_lr,
                                (unsigned long long)cfg.seed, mask, wrong_pen,
                                err, sizeof err) != 0) {
            fprintf(stderr, "bandit: %s\n", err);
            return 1;
        }
        return 0;
    }
    if (arg_eq(cmd, "train")) {
        if (!data || !val) { fprintf(stderr, "train needs --data and --val\n"); return 2; }
        if (cd_train(&cfg, data, val, out, NULL, err, sizeof err) != 0) {
            fprintf(stderr, "train failed: %s\n", err);
            return 1;
        }
        return 0;
    }
    usage();
    return 2;
}

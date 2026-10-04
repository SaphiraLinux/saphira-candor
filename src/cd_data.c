#define _POSIX_C_SOURCE 200809L
#include <sys/types.h>
/* cd_data.c — JSONL dataset loading, transcribed from the upstream trainer data.c (see docs/PROVENANCE.md)
 * semantics: strict fail on a bad line, three required fields, >=2 options,
 * each option non-empty, label a zero-based index, unknown keys ignored.
 */
#include "candor.h"
#include "cd_internal.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* struct cd_dataset is declared in cd_internal.h */

static char *jstr(const char *s, size_t len) {
    char *o = malloc(len + 1);
    if (!o) return NULL;
    memcpy(o, s, len);
    o[len] = '\0';
    return o;
}

/* Find "key": at the top level of the object and return a pointer just past
 * the colon. Scans with string awareness so a key inside a string is not
 * matched. Sufficient for these flat records. */
static const char *find_key(const char *line, const char *key) {
    size_t klen = strlen(key);
    const char *p = line;
    int depth = 0, instr = 0;
    while (*p) {
        if (instr) {
            if (*p == '\\' && p[1]) { p += 2; continue; }
            if (*p == '"') instr = 0;
            p++;
            continue;
        }
        if (*p == '"') {
            const char *q = p + 1;
            const char *e = q;
            while (*e && *e != '"') { if (*e == '\\' && e[1]) e++; e++; }
            if (depth == 1 && (size_t)(e - q) == klen && strncmp(q, key, klen) == 0) {
                const char *r = e + 1;
                while (*r == ' ' || *r == '\t') r++;
                if (*r == ':') { r++; while (*r == ' ' || *r == '\t') r++; return r; }
            }
            p = (*e) ? e + 1 : e;
            continue;
        }
        if (*p == '{' || *p == '[') depth++;
        else if (*p == '}' || *p == ']') depth--;
        p++;
    }
    return NULL;
}

static int parse_line(const char *line, char **ctx, char ***opts, int *nopt,
                      int *label, int *form, char *err, size_t errlen) {
    const char *p;
    const char *e;
    int cap = 8, cnt = 0;
    char **arr;

    p = find_key(line, "context");
    if (!p || *p != '"') { snprintf(err, errlen, "missing context"); return -1; }
    p++;
    e = strchr(p, '"');
    if (!e) { snprintf(err, errlen, "unterminated context"); return -1; }
    *ctx = jstr(p, (size_t)(e - p));
    if (!*ctx || !**ctx) { snprintf(err, errlen, "empty context"); return -1; }

    p = find_key(line, "options");
    if (!p || *p != '[') { snprintf(err, errlen, "missing options"); return -1; }
    p++;
    arr = malloc((size_t)cap * sizeof(char *));
    if (!arr) return -1;
    while (*p && *p != ']') {
        if (*p == '"') {
            const char *q = p + 1;
            const char *r = strchr(q, '"');
            char *s;
            if (!r) { snprintf(err, errlen, "unterminated option"); goto fail; }
            s = jstr(q, (size_t)(r - q));
            if (!s || !*s) { free(s); snprintf(err, errlen, "empty option"); goto fail; }
            if (cnt == cap) { cap *= 2; arr = realloc(arr, (size_t)cap * sizeof(char *)); }
            arr[cnt++] = s;
            p = r + 1;
        } else p++;
    }
    if (cnt < 2) { snprintf(err, errlen, "options must have at least 2"); goto fail; }

    /* optional form label; absent means "no routing information" */
    *form = -1;
    p = find_key(line, "form");
    if (p && *p == '"') {
        const char *q2 = p + 1;
        if (strncmp(q2, "fwd", 3) == 0) *form = 0;
        else if (strncmp(q2, "inv", 3) == 0) *form = 1;
    }

    p = find_key(line, "label");
    if (!p) { snprintf(err, errlen, "missing label"); goto fail; }
    *label = (int)strtol(p, NULL, 10);
    if (*label < 0 || *label >= cnt) { snprintf(err, errlen, "label out of range"); goto fail; }

    *opts = arr; *nopt = cnt;
    return 0;
fail:
    { int i; for (i = 0; i < cnt; i++) free(arr[i]); }
    free(arr);
    return -1;
}

int cd_data_load(const char *path, struct cd_dataset *ds, char *err, size_t errlen) {
    FILE *f = fopen(path, "rb");
    char *line = NULL;
    size_t cap = 0;
    long got;
    size_t lineno = 0;
    size_t cap_rows = 1024;

    if (!f) { snprintf(err, errlen, "cannot open %s", path); return -1; }
    memset(ds, 0, sizeof *ds);
    ds->context = malloc(cap_rows * sizeof(char *));
    ds->options = malloc(cap_rows * sizeof(char **));
    ds->n_options = malloc(cap_rows * sizeof(int));
    ds->label = malloc(cap_rows * sizeof(int));
    ds->form = malloc(cap_rows * sizeof(int));

    while ((got = getline(&line, &cap, f)) != -1) {
        char *ctx = NULL, **opts = NULL;
        int nopt = 0, label = 0, form = -1;
        lineno++;
        {   /* trim */
            while (got > 0 && (line[got - 1] == '\n' || line[got - 1] == '\r')) line[--got] = '\0';
        }
        if (got == 0) continue;
        if (parse_line(line, &ctx, &opts, &nopt, &label, &form, err, errlen) != 0) {
            char e2[256];
            snprintf(e2, sizeof e2, "%s", err);
            snprintf(err, errlen, "%s:%zu: %s", path, lineno, e2);
            goto fail;
        }
        if (ds->n == cap_rows) {
            cap_rows *= 2;
            ds->context = realloc(ds->context, cap_rows * sizeof(char *));
            ds->options = realloc(ds->options, cap_rows * sizeof(char **));
            ds->n_options = realloc(ds->n_options, cap_rows * sizeof(int));
            ds->label = realloc(ds->label, cap_rows * sizeof(int));
            ds->form = realloc(ds->form, cap_rows * sizeof(int));
        }
        ds->context[ds->n] = ctx;
        ds->options[ds->n] = opts;
        ds->n_options[ds->n] = nopt;
        ds->label[ds->n] = label;
        ds->form[ds->n] = form;
        ds->n++;
    }
    free(line);
    fclose(f);
    if (ds->n == 0) { snprintf(err, errlen, "no examples in %s", path); return -1; }
    return 0;
fail:
    free(line);
    fclose(f);
    return -1;
}

void cd_data_free(struct cd_dataset *ds) {
    size_t i;
    int j;
    if (!ds || !ds->context) return;
    for (i = 0; i < ds->n; i++) {
        for (j = 0; j < ds->n_options[i]; j++) free(ds->options[i][j]);
        free(ds->options[i]);
        free(ds->context[i]);
    }
    free(ds->context); free(ds->options); free(ds->n_options); free(ds->label);
    free(ds->form);
    memset(ds, 0, sizeof *ds);
}

size_t cd_data_n(const struct cd_dataset *ds) { return ds ? ds->n : 0; }
const char *cd_data_ctx(const struct cd_dataset *ds, size_t i) { return ds->context[i]; }
int cd_data_nopt(const struct cd_dataset *ds, size_t i) { return ds->n_options[i]; }
const char *cd_data_opt(const struct cd_dataset *ds, size_t i, int j) { return ds->options[i][j]; }
int cd_data_label(const struct cd_dataset *ds, size_t i) { return ds->label[i]; }
int cd_data_form(const struct cd_dataset *ds, size_t i) { return ds->form ? ds->form[i] : -1; }

int cd_data_max_nopt(const struct cd_dataset *ds) {
    size_t i; int mx = 0;
    if (!ds) return 0;
    for (i = 0; i < ds->n; i++) if (ds->n_options[i] > mx) mx = ds->n_options[i];
    return mx;
}

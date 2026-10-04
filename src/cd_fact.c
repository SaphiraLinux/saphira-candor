/* cd_fact.c — witnessed-fact table for fact-first ask.
 *
 * A fact is (entity, property, source statement): discrete truth with provenance,
 * not a vector. The table is built from witnessed JSONL rows carrying "entity" and
 * "fact" keys (training rows may carry them alongside context/options/label; the
 * data loader ignores unknown keys, and this loader ignores everything else).
 *
 * Read discipline, enforced by the caller (see the `ask` command):
 *   exactly one candidate matches a witnessed fact for the resolved entity
 *       -> answer it, marked REMEMBERED, with the source statement. No forward pass.
 *   anything else (zero or several matches)
 *       -> fall through to the vector forward pass, marked GUESSED.
 * Absence is never a NO: an unwitnessed candidate is reported as not-in-memory.
 *
 * Deliberately small: bounded line buffer, exact string matching only, no
 * allocation beyond the table itself. This file must never grow a scorer.
 */
#define _POSIX_C_SOURCE 200809L
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define CD_FACT_MAX_LINE 4096
#define CD_FACT_MAX_ROWS 65536
#define CD_FACT_MAX_FIELD 512

struct cd_fact {
    char *entity;
    char *adj;
    char *context;
};

struct cd_facts {
    struct cd_fact *v;
    size_t n;
};

/* Copy a JSON string value starting at *p (just past the opening quote) into
 * out, honouring backslash escapes. Returns pointer past the closing quote, or
 * NULL on truncation/overflow. */
static const char *json_string(const char *p, char *out, size_t cap) {
    size_t o = 0;
    while (*p && *p != '"') {
        char c = *p++;
        if (c == '\\' && *p) {
            char e = *p++;
            c = (e == 'n') ? '\n' : (e == 't') ? '\t' : (e == 'r') ? '\r' : e;
        }
        if (o + 1 >= cap) return NULL;
        out[o++] = c;
    }
    if (*p != '"') return NULL;
    out[o] = '\0';
    return p + 1;
}

/* Find top-level "key": "<string>" and copy the value. String-aware so a key
 * inside a string is not matched. Returns 0 on success. */
static int json_key_string(const char *line, const char *key, char *out, size_t cap) {
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
            const char *q = p + 1, *e = q;
            while (*e && *e != '"') { if (*e == '\\' && e[1]) e++; e++; }
            if (depth == 1 && (size_t)(e - q) == klen && strncmp(q, key, klen) == 0) {
                const char *r = e + 1;
                while (*r == ' ' || *r == '\t') r++;
                if (*r != ':') { p = (*e) ? e + 1 : e; continue; }
                r++;
                while (*r == ' ' || *r == '\t') r++;
                if (*r != '"') return -1;
                return json_string(r + 1, out, cap) ? 0 : -1;
            }
            p = (*e) ? e + 1 : e;
            continue;
        }
        if (*p == '{' || *p == '[') depth++;
        else if (*p == '}' || *p == ']') depth--;
        p++;
    }
    return -1;
}

static char *xstrdup(const char *s) {
    size_t n = strlen(s) + 1;
    char *o = malloc(n);
    if (o) memcpy(o, s, n);
    return o;
}

static int cd_fact_parse(FILE *fp, struct cd_facts *f, char *err, size_t errlen) {
    char line[CD_FACT_MAX_LINE];
    f->v = NULL;
    f->n = 0;
    while (fgets(line, sizeof line, fp)) {
        char entity[CD_FACT_MAX_FIELD], adj[CD_FACT_MAX_FIELD], ctx[CD_FACT_MAX_LINE];
        struct cd_fact *nv;
        size_t L = strlen(line);
        while (L && (line[L-1] == '\n' || line[L-1] == '\r')) line[--L] = '\0';
        if (!L) continue;
        if (strchr(line, '\n')) {
            snprintf(err, errlen, "witness line too long (cap %d)", CD_FACT_MAX_LINE);
            fclose(fp);
            return -1;
        }
        /* Rows without entity+fact keys are training-only rows, not witnesses. Skipped,
         * never guessed from: a row that does not state a fact teaches nothing. */
        if (json_key_string(line, "entity", entity, sizeof entity) != 0) continue;
        if (json_key_string(line, "fact", adj, sizeof adj) != 0) continue;
        if (!entity[0] || !adj[0]) continue;
        if (json_key_string(line, "context", ctx, sizeof ctx) != 0) ctx[0] = '\0';
        if (f->n >= CD_FACT_MAX_ROWS) {
            snprintf(err, errlen, "witness file exceeds %d rows", CD_FACT_MAX_ROWS);
            fclose(fp);
            return -1;
        }
        nv = realloc(f->v, (f->n + 1) * sizeof *nv);
        if (!nv) {
            snprintf(err, errlen, "out of memory loading witness file");
            fclose(fp);
            return -1;
        }
        f->v = nv;
        f->v[f->n].entity = xstrdup(entity);
        f->v[f->n].adj = xstrdup(adj);
        f->v[f->n].context = xstrdup(ctx);
        if (!f->v[f->n].entity || !f->v[f->n].adj || !f->v[f->n].context) {
            snprintf(err, errlen, "out of memory loading witness file");
            fclose(fp);
            return -1;
        }
        f->n++;
    }
    /* NOTE: stream ownership stays with the caller (cd_fact_load / _mem close
     * what they opened). An earlier revision closed here AND in the wrappers,
     * double-closing the stream -- caught live as a tcache double-free abort
     * on the first remembered answer. */
    return 0;
}

void cd_fact_free(struct cd_facts *f) {
    size_t i;
    if (!f || !f->v) return;
    for (i = 0; i < f->n; i++) {
        free(f->v[i].entity);
        free(f->v[i].adj);
        free(f->v[i].context);
    }
    free(f->v);
    f->v = NULL;
    f->n = 0;
}

size_t cd_fact_count(const struct cd_facts *f) { return f ? f->n : 0; }

/* Longest known-entity substring of the context, or NULL. Longest match wins so
 * "harbor heron" beats a hypothetical shorter overlapping name. */
const char *cd_fact_resolve_entity(const struct cd_facts *f, const char *context) {
    const char *best = NULL;
    size_t bestlen = 0, i;
    if (!f || !context) return NULL;
    for (i = 0; i < f->n; i++) {
        size_t L = strlen(f->v[i].entity);
        if (L > bestlen && strstr(context, f->v[i].entity)) {
            best = f->v[i].entity;
            bestlen = L;
        }
    }
    return best;
}

/* Exact (entity, candidate) lookup over n candidates.
 * Returns 1 with *hit set to the matched candidate and *src to its source
 * statement iff EXACTLY ONE candidate is witnessed for the entity; else 0.
 * Zero matches and several matches both return 0: ambiguity is reported by the
 * caller, never resolved here. */
int cd_fact_lookup(const struct cd_facts *f, const char *entity,
                   char *const *cands, int n, const char **hit, const char **src) {
    int found = 0, i;
    size_t k;
    const char *h = NULL, *s = NULL;
    if (!f || !entity) return 0;
    for (i = 0; i < n; i++) {
        for (k = 0; k < f->n; k++) {
            if (strcmp(f->v[k].entity, entity) == 0 && strcmp(f->v[k].adj, cands[i]) == 0) {
                found++;
                h = cands[i];
                s = f->v[k].context;
                break;
            }
        }
    }
    if (found == 1) {
        *hit = h;
        *src = s;
        return 1;
    }
    return 0;
}

int cd_fact_load(const char *path, struct cd_facts *f, char *err, size_t errlen) {
    FILE *fp = fopen(path, "r");
    int rc;
    if (!fp) {
        snprintf(err, errlen, "cannot open witness file '%s'", path);
        return -1;
    }
    rc = cd_fact_parse(fp, f, err, errlen);
    fclose(fp);
    return rc;
}

int cd_fact_load_mem(const char *text, struct cd_facts *f, char *err, size_t errlen) {
    FILE *fp;
    int rc;
    if (!text || !*text) {
        snprintf(err, errlen, "empty embedded witness set");
        return -1;
    }
    fp = fmemopen((void *)text, strlen(text), "r");
    if (!fp) {
        snprintf(err, errlen, "cannot open embedded witness set");
        return -1;
    }
    rc = cd_fact_parse(fp, f, err, errlen);
    fclose(fp);
    return rc;
}

/* cd_simd.c — exact AVX2 row-axis projections.
 *
 * SCOPE. Pure acceleration. No model mathematics, no training semantics, no
 * dataset, no architecture, no parameter count and no experimental conclusion
 * is changed by this file. The scalar path in cd_model.c remains present,
 * compiled, tested and selectable at run time.
 *
 * THE STRATEGY, and why it is exact. The scalar reduction in cd_mat_rows is
 *
 *     for r: { double acc = 0; for w: acc += (double)src[w] * (double)W[r][w]; }
 *
 * a SEQUENTIAL f64 chain over w. Two properties follow, and both are load
 * bearing:
 *
 *   1. SIMD ACROSS OUTPUT ROWS IS EXACT. Put four independent r values in four
 *      f64 lanes and walk w upward. Each lane performs the same ordered
 *      sequence of multiply-then-add on the same operands. IEEE-754 is defined
 *      per operation, and register packing is not observable.
 *   2. UNROLLING THE w LOOP IS NOT EXACT. Two partial accumulators reassociate
 *      and change the rounding. So the w loop is never split here, which is why
 *      the obvious 4-way-unrolled GEMV is deliberately not attempted.
 *
 * The weight arrives PRE-TRANSPOSED as Wt[w*R + r] so the four values a
 * lane-group needs at a given w are 16 contiguous bytes rather than four
 * separate cache lines. That transpose is derived data, refreshed after every
 * optimiser step, never saved, and never counted as a parameter.
 *
 * FMA. Multiply and add are separate intrinsics, and the translation unit is
 * built with -ffp-contract=off, so no FMA opcode can be emitted. `make
 * verify-no-simd-fma` in the test suite disassembles the object and fails if one
 * appears. The target attribute requests avx2 only; fma is never requested, so
 * the compiler has no fused instruction available to contract into.
 */
#include "candor.h"
#include "cd_internal.h"

#include <string.h>

#if defined(__x86_64__) || defined(__i386__)
#define CD_X86 1
#include <immintrin.h>
#endif

/* Run-time control, so the scalar reference is always reachable:
 *   CANDOR_SIMD=0  force scalar
 *   CANDOR_SIMD=1  force AVX2 (fails loudly if the CPU lacks it)
 *   unset        use AVX2 when the CPU reports it
 */
int cd_simd_mode(void) {
    static int mode = -1;
    if (mode >= 0) return mode;
    {
        const char *e = getenv("CANDOR_SIMD");
        if (e && e[0] == '0') mode = 0;
        else if (e && e[0] == '1') mode = 1;
        else {
#if defined(CD_X86) && defined(__GNUC__)
            mode = __builtin_cpu_supports("avx2") ? 1 : 0;
#else
            mode = 0;
#endif
        }
    }
    return mode;
}

int cd_simd_available(void) { return cd_simd_mode(); }
const char *cd_simd_name(void) { return cd_simd_mode() ? "avx2-rowaxis" : "scalar"; }

#if defined(CD_X86)

__attribute__((target("avx2")))
void cd_mat_rows_avx2(const float *src, const double *Wt, float *dst,
                      int rows, int width, int R) {
    int i, r, w;
    for (i = 0; i < rows; i++) {
        const float *sp = src + (size_t)i * (size_t)width;
        float *d = dst + (size_t)i * (size_t)R;
        for (r = 0; r + 4 <= R; r += 4) {
            __m256d acc = _mm256_setzero_pd();
            for (w = 0; w < width; w++) {
                __m256d wv = _mm256_loadu_pd(Wt + (size_t)w * (size_t)R + r);
                __m256d sv = _mm256_set1_pd((double)sp[w]);
                acc = _mm256_add_pd(acc, _mm256_mul_pd(sv, wv));
            }
            {
                double t4[4];
                _mm256_storeu_pd(t4, acc);
                d[r]     = (float)t4[0];
                d[r + 1] = (float)t4[1];
                d[r + 2] = (float)t4[2];
                d[r + 3] = (float)t4[3];
            }
        }
        for (; r < R; r++) {          /* rank tail, identical operand order */
            double acc = 0.0;
            for (w = 0; w < width; w++)
                acc += (double)sp[w] * (double)Wt[(size_t)w * (size_t)R + r];
            d[r] = (float)acc;
        }
    }
}

/* attended[n] = sum_i attn[n,i] * V[i][r], f32 accumulation over i.
 *
 * The output axis is r and, for a fixed i, four consecutive r of V are
 * contiguous, so this needs no transpose at all. Each lane keeps its own
 * accumulator and visits i in ascending order, so the per-output f32 sequence
 * is unchanged. This is the same rule-1 argument as above, in f32. */
__attribute__((target("avx2")))
void cd_attended_avx2(const float *attn, const float *V, float *dst,
                      int N, int L, int R) {
    int n, i, r;
    for (n = 0; n < N; n++) {
        const float *a = attn + (size_t)n * (size_t)L;
        float *o = dst + (size_t)n * (size_t)R;
        for (r = 0; r < R; r++) o[r] = 0.0f;
        for (r = 0; r + 4 <= R; r += 4) {
            __m128 acc = _mm_setzero_ps();
            for (i = 0; i < L; i++) {
                __m128 v = _mm_loadu_ps(V + (size_t)i * (size_t)R + r);
                acc = _mm_add_ps(acc, _mm_mul_ps(_mm_set1_ps(a[i]), v));
            }
            _mm_storeu_ps(o + r, acc);
        }
        for (; r < R; r++) {
            float acc = 0.0f;
            for (i = 0; i < L; i++) acc += a[i] * V[(size_t)i * (size_t)R + r];
            o[r] = acc;
        }
    }
}

#endif /* CD_X86 */

/* Dispatch entry used by the model. Wt may be NULL, in which case the scalar
 * reference runs even on an AVX2 machine. */
void cd_mat_rows_x(const float *src, const float *W, const double *Wt,
                   float *dst, int rows, int width, int R) {
#if defined(CD_X86)
    if (Wt && cd_simd_mode()) { cd_mat_rows_avx2(src, Wt, dst, rows, width, R); return; }
#endif
    cd_mat_rows(src, W, dst, rows, width, R);
}

void cd_attended_x(const float *attn, const float *V, float *dst,
                   int N, int L, int R) {
#if defined(CD_X86)
    if (cd_simd_mode()) { cd_attended_avx2(attn, V, dst, N, L, R); return; }
#endif
    {
        int n, i, r;
        for (n = 0; n < N; n++) {
            const float *a = attn + (size_t)n * (size_t)L;
            float *o = dst + (size_t)n * (size_t)R;
            for (r = 0; r < R; r++) {
                float acc = 0.0f;
                for (i = 0; i < L; i++) acc += a[i] * V[(size_t)i * (size_t)R + r];
                o[r] = acc;
            }
        }
    }
}

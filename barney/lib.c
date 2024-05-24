#include <immintrin.h>

#define DIM 768
#define DIMH 64

inline float hsum128(__m128 x)
{
    x = _mm_add_ps(x, _mm_movehl_ps(x, x));
    x = _mm_add_ss(x, _mm_movehdup_ps(x));
    return _mm_cvtss_f32(x);
}

inline float hsum(__m256 x)
{
    return hsum128(_mm_add_ps(_mm256_extractf128_ps(x, 1), _mm256_castps256_ps128(x)));
}

/*
 * x is a single token of shape (DIM,)
 * e1 is a row oriented matrix of shape (DIM, DIMH)
 * e2 is a row oriented matrix of shape (DIMH, DIM)
 * xo points to pre-allocated memory of shape (DIM,)
 */
void expert_forward(float *x, float *e1, float *e2, float *xo)
{
    float act[DIMH];
    __m256 zero = _mm256_setzero_ps();
    for (int j = 0; j < DIMH; ++j)
    {
        __m256 sum = _mm256_setzero_ps();
        for (int i = 0; i < DIM; i += 8)
        {
            __m256 v1 = _mm256_loadu_ps(&x[i]);
            __m256 v2 = _mm256_loadu_ps(&e1[i]);
            __m256 prod = _mm256_mul_ps(v1, v2);
            __m256 max_val = _mm256_max_ps(prod, zero);
            sum = _mm256_add_ps(sum, max_val);
        }
        act[j] = hsum(sum);
        e1 += DIM;
    }
    for (int j = 0; j < DIM; j++)
    {
        __m256 sum = _mm256_setzero_ps();
        for (int i = 0; i < DIMH; i += 8)
        {
            __m256 v1 = _mm256_loadu_ps(&tmp[i]);
            __m256 v2 = _mm256_loadu_ps(&e2[i]);
            sum = _mm256_fmadd_ps(v1, v2, sum);
        }
        xo[j] = hsum(sum);
        e2 += DIMH;
    }
}

void bench_expert_forward(float *x, float *e1, float *e2, float *xo)
{
    for (int k = 0; k < 1000; k++)
        expert_forward(x, e1, e2, xo);
}

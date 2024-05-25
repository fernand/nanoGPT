#include <immintrin.h>
#include <stdlib.h>
#include <stdio.h>

#include "thpool.h"
#include "xoshiro.h"

#define BS 16
#define SEQ 1024
#define NUM_EXPERTS 1024
#define TOPK 16
#define DIM 768
#define DIMH 64

static inline float hsum128(__m128 x)
{
    x = _mm_add_ps(x, _mm_movehl_ps(x, x));
    x = _mm_add_ss(x, _mm_movehdup_ps(x));
    return _mm_cvtss_f32(x);
}

static inline float hsum(__m256 x)
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
            // FMA is slower than mul and add on Zen2.
            __m256 v1 = _mm256_loadu_ps(&act[i]);
            __m256 v2 = _mm256_loadu_ps(&e2[i]);
            __m256 prod = _mm256_mul_ps(v1, v2);
            sum = _mm256_add_ps(sum, prod);
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

typedef struct efwd_args
{
    float *x;
    float *e1;
    float *e2;
    float *xo;
} efwd_args;

static void expert_forward_thread(void *arg)
{
    efwd_args *args = (efwd_args *)arg;
    expert_forward(args->x, args->e1, args->e2, args->xo);
}

static uint32_t *malloc_rand_expert(int numel, rnd_state *state)
{
    uint32_t *output = malloc(sizeof(uint32_t) * numel);
    for (int i = 0; i < numel; i++)
        output[i] = next(state) % NUM_EXPERTS;
    return output;
}

static float *fmalloc_rand(int numel, rnd_state *state)
{
    float *output = malloc(sizeof(float) * numel);
    for (int i = 0; i < numel; i++)
        output[i] = random_float(state);
    return output;
}

static void route_and_compute_token(float *x, uint32_t *experts, float *E1, float *E2, float *Xo, threadpool pool, efwd_args *argpool)
{
    for (int i = 0; i < TOPK; i++)
    {
        int expert_idx = (int)experts[i];
        argpool[i].x = x;
        argpool[i].e1 = &E1[expert_idx * DIM * DIMH];
        argpool[i].e2 = &E2[expert_idx * DIMH * DIM];
        argpool[i].xo = Xo;
        thpool_add_work(pool, expert_forward_thread, &argpool[i]);
        Xo += DIM;
    }
}

void route_and_compute(float *X, uint32_t *Ei, float *E1, float *E2, float *Xo, threadpool pool)
{
    efwd_args argpool1[TOPK];
    efwd_args argpool2[TOPK];
    // Move two tokens at a time given that we have a 32 thread pool.
    for (int token_idx = 0; token_idx < BS * SEQ; token_idx+=2)
    {
        route_and_compute_token(X, Ei, E1, E2, Xo, pool, argpool1);
        X += DIM;
        Ei += TOPK;
        route_and_compute_token(X, Ei, E1, E2, Xo, pool, argpool2);
        X += DIM;
        Ei += TOPK;
        thpool_wait(pool);
    }
}

int main()
{
    rnd_state state = {{257, 566}};
    float *X = fmalloc_rand(BS * SEQ * DIM, &state);
    // Chosen topk experts after gating.
    uint32_t *Ei = malloc_rand_expert(BS * SEQ * TOPK, &state);
    float *E1 = fmalloc_rand(NUM_EXPERTS * DIM * DIMH, &state);
    float *E2 = fmalloc_rand(NUM_EXPERTS * DIMH * DIM, &state);
    // The reduction across TOPK will by done by torch on the CPU with the softmax scores.
    float *Xo = calloc(BS * SEQ * TOPK * DIM, sizeof(float));

    threadpool pool = thpool_init(32);

    route_and_compute(X, Ei, E1, E2, Xo, pool);

    thpool_destroy(pool);

    return 0;
}

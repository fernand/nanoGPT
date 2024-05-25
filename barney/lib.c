#include <immintrin.h>
#include <pthread.h>
#include <stdlib.h>
#include <stdio.h>
#include <time.h>

#include "xoshiro.h"

#define BS 16
#define SEQ 1024
#define NUM_EXPERTS 1024
#define TOPK 16
#define DIM 768
#define DIMH 64
#define NUM_THREADS 16

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

static void route_and_compute_token(float *x, uint32_t *experts, float *E1, float *E2, float *Xo)
{
    for (int i = 0; i < TOPK; i++)
    {
        int expert_idx = (int)experts[i];
        expert_forward(x, &E1[expert_idx * DIM * DIMH], &E2[expert_idx * DIMH * DIM], Xo);
        Xo += DIM;
    }
}

typedef struct thread_args
{
    int chunk_size;
    float *X;
    uint32_t *Ei;
    float *E1;
    float *E2;
    float *Xo;
} thread_args;

void *compute_chunk(void *arg)
{
    thread_args *args = (thread_args *)arg;
    for (int i = 0; i < args->chunk_size; i++)
    {
        route_and_compute_token(args->X, args->Ei, args->E1, args->E2, args->Xo);
        args->X += DIM;
        args->Ei += TOPK;
    }
    return NULL;
}

void compute(float *X, uint32_t *Ei, float *E1, float *E2, float *Xo)
{
    int chunk_size = BS * SEQ / NUM_THREADS;
    pthread_t threads[NUM_THREADS];
    thread_args args[NUM_THREADS];
    int token_idx = 0;
    for (int i = 0; i < NUM_THREADS; i++)
    {
        args[i].chunk_size = chunk_size;
        args[i].X = X;
        args[i].Ei = Ei;
        args[i].E1 = E1;
        args[i].E2 = E2;
        args[i].Xo = Xo;
        if (pthread_create(&threads[i], NULL, compute_chunk, &args[i]) != 0)
        {
            perror("Failed to create thread");
            exit(EXIT_FAILURE);
        }
        X += chunk_size * DIM;
        Ei += chunk_size * TOPK;
        Xo += chunk_size * TOPK * DIM;
        token_idx += chunk_size;
    }
    for (int i = 0; i < NUM_THREADS; i++)
    {
        if (pthread_join(threads[i], NULL) != 0)
        {
            perror("Failed to join thread");
            exit(EXIT_FAILURE);
        }
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

    struct timespec t1, t2;
    clock_gettime(CLOCK_MONOTONIC, &t1);
    compute(X, Ei, E1, E2, Xo);
    clock_gettime(CLOCK_MONOTONIC, &t2);
    long seconds = t2.tv_sec - t1.tv_sec;
    long nanoseconds = t2.tv_nsec - t1.tv_nsec;
    double elapsed = seconds + nanoseconds * 1e-9;
    printf("Time: %f ms\n", elapsed * 1000);

    return 0;
}

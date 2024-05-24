// xoshiro128+ https://prng.di.unimi.it/xoroshiro128plus.c
#include <stdint.h>

typedef struct rnd_state
{
    uint64_t s[2];
} rnd_state;

static inline uint64_t rotl(const uint64_t x, int k)
{
    return (x << k) | (x >> (64 - k));
}

static uint64_t next(rnd_state *state)
{
    const uint64_t s0 = state->s[0];
    uint64_t s1 = state->s[1];
    const uint64_t result = s0 + s1;

    s1 ^= s0;
    state->s[0] = rotl(s0, 24) ^ s1 ^ (s1 << 16);
    state->s[1] = rotl(s1, 37);

    return result;
}

// Returns a random float from -1 to 1
float random_float(rnd_state *state)
{
    uint32_t high_bits = next(state) >> 32;
    float normalized = (float)high_bits / (float)UINT32_MAX; // 0-1 range
    return normalized * 2.0f - 1.0f;
}

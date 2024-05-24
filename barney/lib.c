#define DIM 768
#define DIMH 64

#define max(a,b) ((a)>(b) ? (a) : (b))

// Input will be X(batch, seq, dim)
// Each expert inputs will be E1(dim, 64) and E2(64, dim)

float dot(float *a, float *b, int dim)
{
    float result = 0;
    for (int i = 0; i < dim; ++i)
        result += a[i] * b[i];
    return result;
}

float dotrelu(float *a, float *b, int dim)
{
    float result = 0;
    for (int i = 0; i < dim; ++i)
        result += max(0.0f, a[i] * b[i]);
    return result;
}
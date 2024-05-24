#define DIM 768
#define DIMH 64

#define max(a, b) ((a) > (b) ? (a) : (b))

/*
 * x is a single token of shape (DIM,)
 * e1 is a row oriented matrix of shape (DIM, DIMH)
 * e2 is a row oriented matrix of shape (DIMH, DIM)
 * xo points to pre-allocated memory of shape (DIM,)
*/
// Currently 35 us with one core, 2x slower than Pytorch.
void expert_forward(float *x, float *e1, float *e2, float *xo)
{
    float tmp[DIMH] = {0};
    for (int j = 0; j < DIMH; j++)
    {
        for (int i=0; i < DIM; i++)
            tmp[j] += max(0, e1[i] * x[i]);
        e1 += DIM;
    }
    for (int j = 0; j < DIM; j++)
    {
        for (int i=0; i < DIMH; i++)
            xo[j] += e2[i] * tmp[i];
        e2 += DIMH;
    }
}

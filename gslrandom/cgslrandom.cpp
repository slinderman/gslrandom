#include "cgslrandom.h"


BasicRNG::BasicRNG()
{
    r = gsl_rng_alloc(gsl_rng_mt19937);
    gsl_rng_set (r, time(NULL));
}

BasicRNG::BasicRNG(unsigned long seed)
{
    r = gsl_rng_alloc(gsl_rng_mt19937);
    gsl_rng_set (r, seed);
}

void BasicRNG::multinomial(int K,
                           unsigned int N,
                           const double p[],
                           unsigned int n[])
{
    gsl_ran_multinomial(r, K, N, p, n);
}

void sample_multinomial(BasicRNG* brng,
                        int K,
                        unsigned int N,
                        const double p[],
                        unsigned int n[])
{
    brng->multinomial(K, N, p, n);
}

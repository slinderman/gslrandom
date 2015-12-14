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

void BasicRNG::multinomial(const size_t K,
                           const unsigned int N,
                           const double p[],
                           unsigned int n[])
{
    gsl_ran_multinomial(r, K, N, p, n);
}

void BasicRNG::dirichlet (const size_t K,
                          const double alpha[],
                          double theta[])
{
    gsl_ran_dirichlet(r, K, alpha, theta);
}

void sample_multinomial(BasicRNG* brng,
                        const size_t K,
                        const unsigned int N,
                        const double p[],
                        unsigned int n[])
{
    brng->multinomial(K, N, p, n);
}

void seeded_sample_multinomial(unsigned long seed,
                               const size_t K,
                               const unsigned int N,
                               const double p[],
                               unsigned int n[])
{
    BasicRNG* brng = new BasicRNG(seed);
    brng->multinomial(K, N, p, n);
    delete brng;
}

void sample_dirichlet(BasicRNG* brng,
                      const size_t K,
                      const double alpha[],
                      double theta[])
{
    brng->dirichlet(K, alpha, theta);
}

float sample_uniform()
{
    return rand() / float(RAND_MAX);
}

bool sample_bernoulli(const double p)
{
    double u = sample_uniform();
    if (u < p) { return 1; }
    else { return 0;}
}

unsigned int sample_crt(const unsigned int m,
                        const double r)
{
    if ((r == 0) || (m == 0)) { return 0; }

    if (m == 1) { return 1; }

    unsigned int l = 0;
    for (size_t n = 1; n <= m; n++)
    {
        l = l + sample_bernoulli(r / (r + n - 1));
    }
    return l;
}

void cbincount(size_t n_obs,
              const unsigned int obs[],
              const unsigned int weights[],
              size_t minlength,
              unsigned int bins[],
              int reset_bins)
{
    unsigned int i;
    if (reset_bins == 1) {
      for (size_t i = 0; i < minlength; i ++) 
      { 
          bins[i] = 0;
      }
    }
    unsigned int w;
    for (size_t j = 0; j < n_obs; j++)
    {
        i = obs[j];
        w = weights[j];
        bins[i] += w;
    }
}

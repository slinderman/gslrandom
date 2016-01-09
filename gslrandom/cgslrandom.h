#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>
#include <gsl/gsl_vector.h>
#include <ctime>
#include <stdlib.h>

class BasicRNG {

 protected:

  gsl_rng * r;

 public:

  // Constructors and destructors.
  BasicRNG();
  BasicRNG(unsigned long seed);

  virtual ~BasicRNG()
    { gsl_rng_free (r); }

  // Get rng -- be careful.  Needed for other random variates.
  gsl_rng* getrng()
  {
      return r;
  }

  void multinomial (const size_t K,
                    const unsigned int N,
                    const double p[],
                    unsigned int n[]);

  void dirichlet (const size_t K,
                  const double alpha[],
                  double theta[]);

}; // BasicRNG

void sample_multinomial(BasicRNG* brng,
                        const size_t K,
                        const unsigned int N,
                        const double p[],
                        unsigned int n[]);

// Alternative method, that doesn't rely on an external BasicRNG object.
// Creates and deletes a BasicRNG. Requires a seed for the BasicRNG.
void seeded_sample_multinomial(const unsigned long seed,
                               const size_t K,
                               const unsigned int N,
                               const double p[],
                               unsigned int n[]);

void sample_dirichlet (BasicRNG* brng,
                       const size_t K,
                       const double alpha[],
                       double theta[]);

float sample_uniform();

bool sample_bernoulli(const double p);

// Chinese Restaurant Table (CRT) distribution
unsigned int sample_crt(const unsigned int m,
                        const double r);

// Bincount
void cbincount(size_t n_obs,
              const unsigned int obs[],
              const unsigned int weights[],
              size_t minlength,
              unsigned int bins[],
              int reset_bins);

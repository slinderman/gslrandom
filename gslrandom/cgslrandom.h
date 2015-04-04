#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>
#include <ctime>

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

  void multinomial (int K,
                    unsigned int N,
                    const double p[],
                    unsigned int n[]);

}; // BasicRNG

void sample_multinomial(BasicRNG* brng,
                        int K,
                        unsigned int N,
                        const double p[],
                        unsigned int n[]);

void dumb_sample_multinomial(int K,
                             unsigned int N,
                             const double p[],
                             unsigned int n[]);


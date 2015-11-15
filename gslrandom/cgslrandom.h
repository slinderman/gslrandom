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

// Alternative method, that doesn't rely on an external BasicRNG object.
// Creates and deletes a BasicRNG every time.
// Requires an initial seed for the BasicRNG.
void seeded_sample_multinomial(unsigned long seed,
                               int K,
                               unsigned int N,
                               const double p[],
                               unsigned int n[]);
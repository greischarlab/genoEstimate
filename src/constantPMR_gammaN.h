# ifndef __ESTIMATE_PMR_CONSTANTPMR_GAMMAN_H
# define __ESTIMATE_PMR_CONSTANTPMR_GAMMAN_H

#include <RcppArmadillo.h>
#include <cmath>

#include "ode.h"

using namespace Rcpp;



// constantPMR_gammaN function
struct ConstantPMRgammaN {

    double cycleLength; // Length of one full cycle
    double mu;          // Non-sequestered iRBC mortality rate
    double museq;       // Sequestered iRBC mortality rate
    double R;           // Parasite multiplication rate
    uint32_t n;         // Number of age classes
    double lambdaN;     // Transition rate for non-sequestered iRBCs
    double lambdaS;     // Transition rate for sequestered iRBCs
    double inflec;
    arma::vec qValues;

    ConstantPMRgammaN(const double& cycleLength_,
                      const double& mu_,
                      const double& museq_,
                      const double& R_,
                      const double& n_,
                      const double& inflec_)
        : cycleLength(cycleLength_),
          mu(mu_),
          museq(museq_),
          R(R_),
          n(static_cast<uint32_t>(n_)),
          lambdaN(n_ / cycleLength),
          lambdaS(lambdaN),
          inflec(inflec_),
          qValues(n, arma::fill::zeros) {

        double dt = 1 / lambdaN;
        double yValue0 = yfx1(dt, inflec);
        double yValue;

        for (uint32_t i = 1; i <= n; ++i) {
            yValue = yfx1(dt + (i-1) * dt, inflec);
            qValues(i-1) = 1 - yValue / yValue0; // Proportional change in yValues.
            yValue0 = yValue;

        }

    }


    void operator()(const VecType& x, VecType& dxdt, const double t) {

        // For the first age class (i = 0):
        dxdt[0] = R * (lambdaN * x[n - 1] + lambdaS * x[2 * n - 1]) -
            (lambdaN + mu) * x[0];
        dxdt[n] = -(lambdaS + museq) * x[n];

        for (uint32_t i = 1; i < n; ++i) {

            const double& Ni(x[i]);     // non-sequestered, age class i
            const double& Ni1(x[i-1]);  // non-sequestered, age class i-1
            double& dNi(dxdt[i]);       // change in non-sequestered, age class i

            const double& Si(x[n+i]);   // sequestered, age class i
            const double& Si1(x[n+i-1]);// sequestered, age class i-1
            double& dSi(dxdt[n+i]);     // change in sequestered, age class i
            const double& q(qValues(i-1));

            dNi = (1 - q) * lambdaN * Ni1 - (lambdaN + mu) * Ni;
            dSi = q * lambdaN * Ni1 + lambdaS * Si1 - (lambdaS + museq) * Si;

        }

        return;

    }


private:

    // Define the values of all parameters used inside yfx.
    const double p1 = 11.3869/467.6209; // Lower bound
    const double p2 = 1.0;               // Upper bound
    const double p4 = 0.2242 * 2.0;       // Slope

    double yfx1(const double& age, double inflec) const {

        double yVal = 1 - (p1 + ((p2 - p1) / (1 + std::pow(10, p4 * (inflec - age)))));

        return yVal;
    }

};




arma::mat constPMR_gammaN_ode_cpp(std::vector<double> x0,
                                  const double& cycleLength,
                                  const double& mu,
                                  const double& museq,
                                  const double& R,
                                  const int& n,
                                  const double& inflec,
                                  const double& max_t,
                                  const double& dt);


#endif



#include <RcppArmadillo.h>
#include <cmath>

#include "ode.h"
#include "constantPMR_gammaN.h"

using namespace Rcpp;






arma::mat constPMR_gammaN_ode_cpp(std::vector<double> x0,
                                  const double& cycleLength,
                                  const double& mu,
                                  const double& museq,
                                  const double& R,
                                  const int& n,
                                  const double& inflec,
                                  const double& max_t,
                                  const double& dt) {

    if (max_t <= 0) stop("max_t must be > 0");
    if (dt >= max_t || dt <= 0) stop("dt must be < max_t and > 0");

    if (n < 0) stop("`n` cannot be < 0");

    if (x0.size() != (2*n)) stop("`x0` must be twice length as `n`");

    GreedyObserver<VecType> obs;
    ConstantPMRgammaN system(cycleLength, mu, museq, R, n, inflec);

    boost::numeric::odeint::integrate_const(
        VecStepperType(), std::ref(system),
        x0, 0.0, max_t, dt, std::ref(obs));

    uint32_t n_steps = obs.data.size();
    arma::mat output(n_steps, x0.size()+1U);
    for (uint32_t i = 0; i < n_steps; i++) {
        output(i,0) = obs.time[i];
        for (uint32_t j = 0; j < x0.size(); j++) {
            output(i,j+1U) = obs.data[i][j];
        }
    }

    return output;

}




//' Run ODE for constant PMR and gamma N.
//'
//' @param x0 Initial conditions for all stages.
//' @param Maximum time point to simulate to.
//' @param Time step to use for ODE.
//'
//'
//' @export
//'
//' @return Numeric matrix where rows are time and columns are stages.
//'
//[[Rcpp::export]]
arma::mat constPMR_gammaN_ode(const std::vector<double>& x0,
                              const double& cycleLength,
                              const double& mu,
                              const double& museq,
                              const double& R,
                              const int& n,
                              const double& inflec,
                              const double& max_t,
                              const double& dt) {

    arma::mat out = constPMR_gammaN_ode_cpp(x0, cycleLength, mu, museq,
                                            R, n, inflec, max_t, dt);

    return out;

}



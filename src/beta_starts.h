# ifndef __ESTIMATE_PMR_BETA_STARTS_H
# define __ESTIMATE_PMR_BETA_STARTS_H


#include <RcppArmadillo.h>
#include <vector>
#include <math.h>
#include <algorithm>


using namespace Rcpp;


// Internal C++ code for use in functions exported to R
arma::vec beta_starts_cpp(const double& shape,
                          const double& offset,
                          const double& total0,
                          const uint32_t& compartments);



#endif






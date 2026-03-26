# ifndef __ESTIMATE_PMR_ARCHER_SHARED_H
# define __ESTIMATE_PMR_ARCHER_SHARED_H


#include <RcppArmadillo.h>
#include <cmath>
#include <vector>
#include <algorithm>
#include <stdexcept>
#include <iostream>
#include <boost/math/distributions/beta.hpp>
#include "ode.h"
#include "beta_starts.h"
#include "constantPMR_gammaN.h"



using namespace Rcpp;


// Define the values of all parameters used inside yfx.
constexpr double p1 = 11.3869 / 467.6209; // Lower bound
constexpr double p2 = 1.0;               // Upper bound
constexpr double p4 = 0.2242 * 2.0;      // Slope




struct ArcherInfo {
  
  double betaShape;
  double offset;
  double R;
  double I0;
  int n;
  double pfCycleLength;
  double inflec;
  double ring_duration;
  
};





// Function to calculate yfx
arma::vec yfx(const arma::vec& age, const double& inflec);

// Function to subset rows of a matrix and return a NumericMatrix
arma::mat subsetRows(const arma::mat& input, const int& step, const bool& geno);

// Repeat subvector a number of times
arma::vec repeat_subvector(const arma::vec& x, const bool& geno);






// double last_stage(const double& pfCycleLength,
//                   const double& R,
//                   const int& n,
//                   const double& inflec,
//                   const double& betaShape,
//                   const double& offset,
//                   const double& I0,
//                   const double& ring_duration,
//                   const DataFrame& data);







#endif
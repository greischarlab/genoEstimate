
#include <RcppArmadillo.h>
#include <Rmath.h>
#include <vector>
#include <math.h>
#include <algorithm>

#include "beta_starts.h"

using namespace Rcpp;


// Internal C++ code for use in functions exported to R
arma::vec beta_starts_cpp(const double& shape,
                          const double& offset,
                          const double& total0,
                          const uint32_t& compartments) {

    arma::vec times = arma::linspace<arma::vec>(0, 1, compartments+1U);
    times += offset;

    arma::vec stage_abunds0(compartments, arma::fill::none);
    double sum_stage_abunds0 = 0;

    double corr_time, pbeta_val, pbeta_val0;
    for (uint32_t i = 0; i <= compartments; i++) {
        corr_time = times(i);
        if (corr_time > 1) corr_time -= 1;
        pbeta_val = R::pbeta(corr_time, shape, shape, true, false);
        if (times(i) > 1) pbeta_val += 1;
        if (i > 0) {
            stage_abunds0(i-1) = total0 * (pbeta_val - pbeta_val0);
            sum_stage_abunds0 += stage_abunds0(i-1);
        }
        pbeta_val0 = pbeta_val;
    }

    bool equal_sums = std::round(sum_stage_abunds0) == std::round(total0);
    if (!equal_sums) {
        stage_abunds0 *= (total0 / sum_stage_abunds0);
        equal_sums = std::round(arma::accu(stage_abunds0)) == std::round(total0);
    }
    if (!equal_sums) {
        double summ_diff = std::round(arma::accu(stage_abunds0)) - std::round(total0);
        std::string err = "beta_starts magnitude error (";
        err += std::to_string(summ_diff) + ", shape = ";
        err += std::to_string(shape) + ", offset = ";
        err += std::to_string(offset) + ")";
        Rcpp::warning(err.c_str());
    }
    return stage_abunds0;
}


//' Produce a vector of starting abundances based on a symmetrical beta distribution
//'
//' @param shape Single numeric giving the Beta distribution shape parameter.
//' @param offset Single numeric giving the Beta distribution offset parameter.
//' @param total0 Single numeric giving the total abundance across all stages.
//' @param compartments Integer giving the number of compartments for
//'     stage structure.
//'
//' @export
//'
//[[Rcpp::export]]
NumericVector beta_starts(const double& shape,
                          const double& offset,
                          const double& total0,
                          const int& compartments) {

    if (shape <= 0) stop("shape <= 0");
    if (offset < 0) stop("offset < 0");
    if (offset > 1) stop("offset > 1");
    if (total0 <= 0) stop("total0 <= 0");
    if (compartments <= 0) stop("compartments <= 0");

    arma::vec stage_abunds0 = beta_starts_cpp(shape, offset, total0,
                                              static_cast<uint32_t>(compartments));
    NumericVector out(stage_abunds0.begin(), stage_abunds0.end());

    return out;
}



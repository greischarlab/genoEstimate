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
#include "archer_shared.h"



using namespace Rcpp;






ArcherInfo extract_parms_cpp(const NumericVector& parms,
                             const double& pfCycleLength,
                             const double& inflec,
                             const double& ring_duration,
                             const double& upper) {

    ArcherInfo info;

    if (parms.size() < 5) {
        stop("parms must have length >= 5");
    }

    // param 1: offset [0, 0.5]
    info.offset = std::exp(-std::exp(parms[0]));
    if (info.offset < 0) info.offset = 0;
    if (info.offset > 0.5) info.offset = 0.5;

    // param 2: R
    info.R = std::exp((parms[1]));

    // param 3: n [4, 500]
    double cv_cycleLength = std::exp(-std::exp(parms[2]));
    double result = std::round(1.0 / (cv_cycleLength * cv_cycleLength));
    info.n = static_cast<int>(result);
    if (info.n < 4) info.n = 4;
    if (info.n > 500) info.n = 500;

    // param 4: I0
    info.I0 = std::exp((parms[3]));

    // param 5: start age [0, 5]
    double start_age1 = std::exp(-std::exp(parms[4]));
    info.start_age = start_age1 * 5;

    int parms_idx = 5;

    // param 6: pfCycleLength [20, 28]
    if (Rcpp::NumericVector::is_na(pfCycleLength)) { // if there's no values specified for pfCycleLength
        if (parms.size() <= parms_idx) { // if not proper length...
            stop("Not enough items for pfCycleLength");
        }
        double pfCycleLength1 = std::exp(-std::exp(parms[parms_idx]));
        info.pfCycleLength = pfCycleLength1 * 8.0 + 20.0;
        parms_idx++;
    } else info.pfCycleLength = pfCycleLength;

    // param 7: inflec [14, 22]
    if (Rcpp::NumericVector::is_na(inflec)) {
        if (parms.size() <= parms_idx) { // if not proper length...
            stop("Not enough items for inflec");
        }
        info.inflec = std::exp(-std::exp(parms[parms_idx])) * 8.0 + 14.0;
        parms_idx++;
    } else info.inflec = inflec;

    // param 8: ring_duration [3, 9]
    if (Rcpp::NumericVector::is_na(ring_duration)) {
        if (parms.size() <= parms_idx) { // if not proper length...
            stop("Not enough items for ring_duration");
        }
        info.ring_duration = std::exp(-std::exp(parms[parms_idx])) * 6.0 + 3.0;
        parms_idx++;
    } else info.ring_duration = ring_duration;

    // param 9: upper bound [0.25, 1]
    if (Rcpp::NumericVector::is_na(upper)) {
        if (parms.size() <= parms_idx) { // if not proper length...
            Rcpp::Rcout << "upper received: " << upper << std::endl;
            stop("Not enough items for upper bound");
        }
        info.upper = std::exp(-std::exp(parms[parms_idx]))*0.75 + 0.25;
        parms_idx++;
    } else info.upper = upper;


    return info;

}


//' Extract parameters.
//'
//' @export
//'
//[[Rcpp::export]]
NumericVector extract_parms(const NumericVector& parms,
                            const double& pfCycleLength = NA_REAL,
                            const double& inflec = NA_REAL,
                            const double& ring_duration = NA_REAL,
                            const double& upper = NA_REAL) {

    ArcherInfo info = extract_parms_cpp(parms, pfCycleLength, inflec, ring_duration, upper);

    NumericVector fit_parms = {info.offset, info.R, static_cast<double>(info.n), info.I0,
                               info.start_age, info.pfCycleLength, info.inflec, info.ring_duration, info.upper};

    return fit_parms;

}





//' Main optimization function for model with 5-8 parameters.
//'
//' @param parms A numeric vector of parameter values.
//' @param data A dataframe containing necessary data (columns
//'     `"Circ"` and `"ring_prop"`).
//' @param geno Decides which dataset is being optimized (if set as false,
//'     then the function is fitting the data from O'Donnell et al., Parasite
//'     Immunology, 2021; if set as true, then the function is fitting the data
//'     from Prior et al., Scientifc Reports, 2019).
//' @param pfCycleLength Single numeric indicating the cycle length.
//'     Defaults to `NA`, which results in it being extracted from `parms`.
//' @param inflec Single numeric indicating the inflection point.
//'     Defaults to `NA`, which results in it being extracted from `parms`.
//' @param ring_duration Single numeric indicating the ring duration.
//'     Defaults to `NA`, which results in it being extracted from `parms`.
//' @param upper Single numeric indicating the upper bound of the sequestration curve.
//'     Defaults to `NA`, which results in it being extracted from `parms`.
//' @param circ_return Single logical indicating whether to output
//'     circulating iRBCs.
//'     Defaults to `FALSE`.
//' @param seq_return Single logical indicating whether to output
//'     sequestered iRBCs.
//'     Defaults to `FALSE`.
//' @param ring_prop_return Single logical indicating whether to output
//'     ring proportions.
//'     Defaults to `FALSE`.
//' @param output_full_return Single logical indicating whether to output
//'     full ODE output.
//'     Defaults to `FALSE`.
//'
//' @export
//'
// [[Rcpp::export]]
SEXP archer_fitN_odeint(NumericVector parms,
                        DataFrame data,
                        const bool& geno,
                        const double& pfCycleLength = NA_REAL,
                        const double& inflec = NA_REAL,
                        const double& ring_duration = NA_REAL,
                        const double& upper = NA_REAL,
                        const bool& circ_return = false,
                        const bool& seq_return = false,
                        const bool& ring_prop_return = false,
                        const bool& output_full_return = false) {

    ArcherInfo info = extract_parms_cpp(parms, pfCycleLength, inflec, ring_duration, upper);

    int n = info.n;
    double n_dbl = n;

    int betaShape = 200; // set shape parameter to 200

    // setting up initial conditions
    arma::vec ages(n);
    for (int i = 0; i < n; ++i) ages[i] = (i + 1) * info.pfCycleLength / n_dbl;

    arma::vec ys = yfx(ages, info.inflec, info.upper);

    arma::vec startI0All = beta_starts_cpp(betaShape, info.offset, info.I0, info.n);
    std::vector<double> x0(2 * n);
    for (int i = 0; i < n; ++i) x0[i] = ys[i] * startI0All[i];
    for (int i = n; i < 2 * n; ++i) x0[i] = (1 - ys[i - n]) * startI0All[i - n];

    // running ODE simulation
    double max_t, step_size;
    if (geno) {
        max_t = 120;
        step_size = 30;
    } else {
        max_t = 128.5;
        step_size = 40;
    }

    double dt = 0.1;
    arma::mat odeint_output = constPMR_gammaN_ode_cpp(x0, info.pfCycleLength,
                                                      0.0, 0.0, info.R, info.n,
                                                      info.inflec, info.upper, max_t, dt);

    // calculate circ.iRBC with high resolution
    int numRows = odeint_output.n_rows;
    arma::vec circ_iRBC(numRows, arma::fill::zeros);
    for (int j = 1; j < n + 1; ++j) {
        circ_iRBC += odeint_output.col(j);
    }

    // calculate seq.iRBC with high resolution
    arma::vec seq_iRBC(numRows, arma::fill::zeros);
    for (int j = n+1; j < 2*n + 1; ++j) {
        seq_iRBC += odeint_output.col(j);
    }

    // subsetting the dataset to make the same resolution as the original time series
    arma::mat subsetMatrix = subsetRows(odeint_output, step_size, geno);

    // calculate circ.iRBC.unique
    int numRows_u = subsetMatrix.n_rows;
    arma::vec circ_iRBC_unique(numRows_u, arma::fill::zeros);
    for (int j = 1; j < n + 1; ++j) {
        circ_iRBC_unique += subsetMatrix.col(j);
    }

    // calculate seq.iRBC.unique
    arma::vec seq_iRBC_unique(numRows_u, arma::fill::zeros);
    for (int j = n+1; j < 2*n + 1; ++j) {
        seq_iRBC_unique += subsetMatrix.col(j);
    }


    arma::vec circ_iRBC_rep = repeat_subvector(circ_iRBC_unique, geno);
    // Rcpp::Rcout << "circ_iRBC_rep: " << circ_iRBC_rep << std::endl;

    double ring_first_stage = std::round(info.start_age / info.pfCycleLength * n_dbl) + 1;
    double ring_last_stage = std::round((info.start_age + info.ring_duration) / info.pfCycleLength * n_dbl) + 1;
    //Rcpp::Rcout << "ring_last_stage: " << ring_last_stage << std::endl;

    if (ring_last_stage <= 2) ring_last_stage = 3;

    // low resolution ring_prop_estim
    arma::vec circ_ring_tot(numRows_u, arma::fill::zeros);

    for (int j = ring_first_stage; j < ring_last_stage; ++j) {
        circ_ring_tot += subsetMatrix.col(j);
    }

    arma::vec ring_prop_estim = circ_ring_tot/circ_iRBC_unique;


    //  high resolution ring_prop_estim
    arma::vec circ_ring_tot_h(numRows, arma::fill::zeros);

    for (int j = ring_first_stage; j < ring_last_stage; ++j) {
        circ_ring_tot_h += odeint_output.col(j);
    }

    arma::vec ring_prop_estim_h = circ_ring_tot_h/circ_iRBC;

    // repeat the low resolution ring_prop_estim
    arma::vec ring_prop_rep = repeat_subvector(ring_prop_estim, geno);

    // Calculate SSE
    NumericVector circ_data = data["Circ"];
    arma::vec transformed_data(circ_data.size());

    for (int i = 0; i < circ_data.size(); ++i) {
        if (NumericVector::is_na(circ_data[i])) {
            transformed_data[i] = NA_REAL;
        } else {
            transformed_data[i] = std::log10(circ_data[i] + 1.0);
        }
    }

    // Rcpp::Rcout << "transformed_data: " << transformed_data<< std::endl;

    arma::vec circ_pred = arma::log10(circ_iRBC_rep + 1.0);
    // Rcpp::Rcout << "circ_pred: " << circ_pred << std::endl;

    arma::vec squared_diff_iRBC(transformed_data.size());
    for (int i = 0; i < circ_data.size(); ++i) {
        if (NumericVector::is_na(transformed_data[i])) {
            // If circ_data[i] is NA, set squared_diff[i] to 0
            squared_diff_iRBC[i] = 0.0;
        } else {
            // Otherwise, calculate the squared difference
            squared_diff_iRBC[i] = std::pow(transformed_data[i] - circ_pred[i], 2U);
        }
    }

    double sse_iRBC = arma::accu(squared_diff_iRBC);
    //Rcpp::Rcout << "sse_iRBC: " << sse_iRBC << std::endl;

    NumericVector ring_data = data["ring_prop"];

    //Rcpp::Rcout << "ring_data: " << ring_data << std::endl;

    arma::vec squared_diff_ring(ring_data.size());
    for (int i = 0; i < ring_data.size(); ++i) {
        if (NumericVector::is_na(ring_data[i])) {
            squared_diff_ring[i] = 0.0;
        } else {
            squared_diff_ring[i] = std::pow(ring_data[i] - ring_prop_rep[i], 2U);
        }
    }

    double sse_ring = arma::accu(squared_diff_ring);
    //Rcpp::Rcout << "sse_ring: " << sse_ring << std::endl;

    double sse = sse_iRBC + sse_ring;


    // ---- RETURN MODE ----
    int n_trues = 0;
    if (circ_return) n_trues++;
    if (seq_return) n_trues++;
    if (ring_prop_return) n_trues++;
    if (output_full_return) n_trues++;
    if (n_trues > 1) {
        std::string err = "Of the arguments circ_return, seq_return, ";
        err += "ring_prop_return, and output_full_return, ";
        err += "at most 1 is allowed to be true.";
        stop(err.c_str());
    }

    if (circ_return == true) return wrap(circ_iRBC);
    if (seq_return == true) return wrap(seq_iRBC);
    if (ring_prop_return == true) return wrap(ring_prop_estim_h);
    if (output_full_return == true) return wrap(odeint_output);

    return wrap(sse);
}





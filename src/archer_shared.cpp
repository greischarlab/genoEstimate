
#include <RcppArmadillo.h>
#include "archer_shared.h"

using namespace Rcpp;



// Function to calculate yfx
arma::vec yfx(const arma::vec& age, const double& inflec) {
  int n = age.size();
  arma::vec yVals(n);

  for (int i = 0; i < n; ++i) {
    // Use Rcpp's `pow` for element-wise calculation
    yVals[i] = 1 - (p1 + ((p2 - p1) / (1 + std::pow(10, p4 * (inflec - age[i])))));
  }
  return yVals;
}


// Function to subset rows of a list
// Function to subset rows of a matrix and return a NumericMatrix
arma::mat subsetRows(const arma::mat& input, const int& step, const bool& geno) {
  if (geno == true){ // If the dataset is the genotype ones
    int n_rows = input.n_rows;
    if (n_rows == 0) {
      return arma::mat(0, 0);  // Return an empty matrix if input is empty
    }

    // Calculate the number of rows in the subset
    int subset_n_rows = (n_rows + step - 1) / step;

    // Initialize a arma::mat to store the subset
    arma::mat subset(subset_n_rows, input.n_cols);

    // Fill the subset matrix
    for (int col = 0; col < input.n_cols; col++) {
      int row = 0; // Reset row counter for each column
      for (int i = 0; i < n_rows; i += step) {
        if (row >= subset_n_rows) {
          break; // Prevent out-of-bounds access
        }
        subset(row, col) = input(i, col);
        row++;
      }
    }
    return subset;
  }else{ // If the datasets are the mistimed treatment groups
    int n_rows = input.n_rows;
    if (n_rows == 0) {
      return arma::mat(0, 0);  // Return an empty matrix if input is empty
    }

    // Calculate the number of rows in the subset
    int subset_n_rows = (n_rows -5 + step - 1) / step;

    // Initialize a arma::mat to store the subset
    arma::mat subset(subset_n_rows, input.n_cols);

    // Fill the subset matrix
    for (int col = 0; col < input.n_cols; col++) {
      int row = 0; // Reset row counter for each column
      for (int i = 5; i < n_rows; i += step) {
        if (row >= subset_n_rows) {
          break; // Prevent out-of-bounds access
        }
        subset(row, col) = input(i, col);
        row++;
      }
    }
    return subset;
  }
}




arma::vec repeat_subvector(const arma::vec& x, const bool& geno) {
  if (geno){
    std::vector<int> reps = {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                             6, 6, 6, 6, 6, 6, 6,
                             6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6};

    int totalSize = 146;

    // Create the output vector
    arma::vec result(totalSize);

    // Fill the output vector with repeated values
    int index = 0;
    for (int i = 0; i < x.size(); i++) {
      for (int j = 0; j < reps[i]; j++) {
        result[index] = x[i];
        index++;
      }
    }

    return result;
  } else{
    std::vector<int> reps = {1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                             4, 4, 4, 4, 4, 4, 8, 8,
                             4, 4, 4, 4, 8, 8,
                             4, 4, 4, 4, 8, 8,
                             4, 4, 4};

    int totalSize = 126;

    // Create the output vector
    arma::vec result(totalSize);

    // Fill the output vector with repeated values
    int index = 0;
    for (int i = 0; i < x.size(); i++) {
      for (int j = 0; j < reps[i]; j++) {
        result[index] = x[i];
        index++;
      }
    }

    return result;
  }
}











// double last_stage(const double& pfCycleLength,
//                   const double& R,
//                   const int& n,
//                   const double& inflec,
//                   const double& betaShape,
//                   const double& offset,
//                   const double& I0,
//                   const double& ring_duration,
//                   const DataFrame& data) {
//
//     double dbl_n = static_cast<double>(n);
//
//     // Initializing parameters
//     std::vector<double> parmValues = {pfCycleLength, 0.0, 0.0, R, dbl_n, inflec};
//     //  std::cout << "parmValues: ";
//     //for (size_t i = 0; i < parmValues.size(); ++i) {
//     //  std::cout << parmValues[i];
//     //   if (i < parmValues.size() - 1) {
//     //     std::cout << ", ";  // Add a comma between elements
//     //}
//     // }
//     // std::cout << std::endl;
//
//     NumericVector ages(n);
//     double step = pfCycleLength / dbl_n;
//     for (int i = 0; i < n; ++i) {
//         ages[i] = (i + 1) * step;  // Assign values directly
//     }
//
//     NumericVector ys = yfx(ages, inflec);
//
//     arma::vec startI0All = beta_starts_cpp(betaShape, offset, I0, n);
//
//     std::vector<double> x0;
//     x0.reserve(2 * n);
//
//     for (int i = 0; i < n; ++i) {
//         x0.push_back(ys[i] * startI0All[i]);
//     }
//     for (int i = n; i < 2 * n; ++i) {
//         x0.push_back((1 - ys[i - n]) * startI0All[i - n]);
//     }
//
//     //   std::cout << "x0: ";
//     //   for (size_t i = 0; i < x0.size(); ++i) {
//     //      std::cout << x0[i];
//     //      if (i < x0.size() - 1) {
//     //       std::cout << ", ";  // Add a comma between elements
//     //       }
//     //  }
//     //   std::cout << std::endl;
//
//     // Running ODE simulation
//     double max_t = 120;
//     double dt = 0.1;
//     arma::mat odeint_output = constPMR_gammaN_ode_cpp(x0, parmValues, max_t, dt);
//
//     double step_size = 30;
//     NumericMatrix subsetMatrix = subsetRows(odeint_output, step_size);
//     // Rcpp::Rcout << "subsetMatrix: " << subsetMatrix << std::endl;
//
//     // Get number of rows
//     int numRows = subsetMatrix.nrow();
//     NumericVector circ_iRBC_unique(numRows, 0.0);
//
//     for (int j = 1; j <= n; ++j) {
//         circ_iRBC_unique += subsetMatrix(_, j);
//     }
//     //Rcpp::Rcout << "circ_iRBC_unique: " << circ_iRBC_unique << std::endl;
//
//     NumericVector circ_iRBC_rep = repeat_subvector(circ_iRBC_unique);
//     // Rcpp::Rcout << "circ_iRBC_rep: " << circ_iRBC_rep << std::endl;
//
//     double ring_last_stage = round(ring_duration / 24 * n) + 1;
//     //Rcpp::Rcout << "ring_last_stage: " << ring_last_stage << std::endl;
//
//     if (ring_last_stage <= 2) ring_last_stage = 3;
//
//     NumericVector circ_ring_tot(numRows, 0.0);
//
//     for (int j = 1; j < ring_last_stage; ++j) {
//         circ_ring_tot += subsetMatrix(_, j);
//     }
//
//     // Rcpp::Rcout << "circ_ring_tot: " << circ_ring_tot << std::endl;
//
//     NumericVector ring_prop_estim = circ_ring_tot/circ_iRBC_unique;
//     // Rcpp::Rcout << "ring_prop_estim: " << ring_prop_estim << std::endl;
//     NumericVector ring_prop_rep = repeat_subvector(ring_prop_estim);
//     // Rcpp::Rcout << "ring_prop_rep: " << ring_prop_rep << std::endl;
//
//
//     // Calculate SSE
//     NumericVector circ_data = data["Circ"];
//     NumericVector transformed_data(circ_data.size());
//
//     for (int i = 0; i < circ_data.size(); ++i) {
//         if (NumericVector::is_na(circ_data[i])) {
//             transformed_data[i] = NA_REAL;
//         } else {
//             transformed_data[i] = std::log10(circ_data[i] + 1.0);
//         }
//     }
//
//     // Rcpp::Rcout << "transformed_data: " << transformed_data<< std::endl;
//
//     NumericVector circ_pred = log10(circ_iRBC_rep + 1.0);
//     // Rcpp::Rcout << "circ_pred: " << circ_pred << std::endl;
//
//     NumericVector squared_diff_iRBC(transformed_data.size());
//     for (int i = 0; i < circ_data.size(); ++i) {
//         if (NumericVector::is_na(transformed_data[i])) {
//             // If circ_data[i] is NA, set squared_diff[i] to 0
//             squared_diff_iRBC[i] = 0.0;
//         } else {
//             // Otherwise, calculate the squared difference
//             squared_diff_iRBC[i] = std::pow(transformed_data[i] - circ_pred[i], 2U);
//         }
//     }
//
//     double sse_iRBC = sum(squared_diff_iRBC);
//     //Rcpp::Rcout << "sse_iRBC: " << sse_iRBC << std::endl;
//
//     NumericVector ring_data = data["ring_prop"];
//
//     //Rcpp::Rcout << "ring_data: " << ring_data << std::endl;
//
//     NumericVector squared_diff_ring(ring_data.size());
//     for (int i = 0; i < ring_data.size(); ++i) {
//         if (NumericVector::is_na(ring_data[i])) {
//             squared_diff_ring[i] = 0.0;
//         } else {
//             squared_diff_ring[i] = std::pow(ring_data[i] - ring_prop_rep[i], 2U);
//         }
//     }
//
//     double sse_ring = sum(squared_diff_ring);
//     //Rcpp::Rcout << "sse_ring: " << sse_ring << std::endl;
//
//     double sse = sse_iRBC + sse_ring;
//
//     return sse;
//
// }


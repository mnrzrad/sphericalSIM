#include <RcppArmadillo.h>
// [[Rcpp::depends(RcppArmadillo)]]

using namespace Rcpp;
using namespace arma;

//' @name cpp_functions
//' @title C++ Accelerated Functions for Spherical SIM
//' @description Fast C++ implementations of computationally intensive functions
//' @keywords internal


// [[Rcpp::export]]
arma::mat inv_stereo_cpp(const arma::mat& u) {
  int n = u.n_rows;
  int q_minus_1 = u.n_cols;
  int q = q_minus_1 + 1;
  
  arma::mat result(n, q);
  arma::vec u_norm_sq = sum(square(u), 1);
  arma::vec denom = 1.0 + u_norm_sq;
  
  // Fill first q-1 columns
  for(int j = 0; j < q_minus_1; j++) {
    result.col(j) = 2.0 * u.col(j) / denom;
  }
  
  // Fill last column
  result.col(q_minus_1) = (1.0 - u_norm_sq) / denom;
  
  return result;
}


// [[Rcpp::export]]
arma::mat jacobian_inv_stereo_cpp(const arma::vec& u) {
  int q_minus_1 = u.n_elem;
  int q = q_minus_1 + 1;
  
  double u_norm_sq = sum(square(u));
  double denom = 1.0 + u_norm_sq;
  double denom_sq = denom * denom;
  
  arma::mat J(q, q_minus_1, fill::zeros);
  
  // Fill first q-1 rows
  for(int i = 0; i < q_minus_1; i++) {
    for(int j = 0; j < q_minus_1; j++) {
      J(i, j) = 2.0 * (i == j ? 1.0 : 0.0) / denom - 4.0 * u(i) * u(j) / denom_sq;
    }
  }
  
  // Fill last row
  J.row(q_minus_1) = -4.0 * u.t() / denom_sq;
  
  return J;
}


// [[Rcpp::export]]
arma::vec prox_group_lasso_cpp(const arma::vec& beta, 
                                const List& group_idx,
                                const arma::vec& weights,
                                double lambda,
                                double step) {
  int G = group_idx.size();
  arma::vec beta_new = beta;
  
  for(int g = 0; g < G; g++) {
    arma::uvec idx = as<arma::uvec>(group_idx[g]) - 1; // R to C++ indexing
    arma::vec beta_g = beta.elem(idx);
    double norm_g = norm(beta_g, 2);
    double threshold = step * lambda * weights(g);
    
    if(norm_g > threshold) {
      beta_new.elem(idx) = (1.0 - threshold / norm_g) * beta_g;
    } else {
      beta_new.elem(idx).zeros();
    }
  }
  
  return beta_new;
}


// [[Rcpp::export]]
double compute_group_penalty_cpp(const arma::vec& beta,
                                  const List& group_idx,
                                  const arma::vec& weights,
                                  double lambda) {
  int G = group_idx.size();
  double penalty = 0.0;
  
  for(int g = 0; g < G; g++) {
    arma::uvec idx = as<arma::uvec>(group_idx[g]) - 1;
    double group_norm = norm(beta.elem(idx), 2);
    penalty += weights(g) * group_norm;
  }
  
  return lambda * penalty;
}


// [[Rcpp::export]]
arma::vec grad_beta_cpp(const arma::vec& beta,
                        const arma::mat& Theta,
                        const arma::mat& X,
                        const arma::mat& Y,
                        const arma::mat& B,
                        const arma::mat& B_deriv) {
  int n = X.n_rows;
  
  // Compute U = B * Theta
  arma::mat U = B * Theta;
  
  // Compute Y_hat via inverse stereographic projection
  arma::mat Y_hat = inv_stereo_cpp(U);
  
  // Compute residuals
  arma::mat E = Y_hat - Y;
  
  // Compute m_prime = B_deriv * Theta
  arma::mat m_prime = B_deriv * Theta;
  
  // Compute gradient contributions for each observation
  arma::vec t_vec(n);
  for(int i = 0; i < n; i++) {
    arma::vec u_i = U.row(i).t();
    arma::mat J_i = jacobian_inv_stereo_cpp(u_i);
    arma::vec dy_dz = J_i * m_prime.row(i).t();
    t_vec(i) = dot(E.row(i), dy_dz);
  }
  
  return (2.0 / n) * X.t() * t_vec;
}


// [[Rcpp::export]]
double compute_objective_cpp(const arma::vec& beta,
                             const arma::mat& Theta,
                             const arma::mat& X,
                             const arma::mat& Y,
                             const arma::mat& B,
                             const arma::mat& Omega,
                             double lambda,
                             double gamma,
                             const List& group_idx,
                             const arma::vec& weights) {
  int n = X.n_rows;
  
  // Compute U = B * Theta
  arma::mat U = B * Theta;
  
  // Compute Y_hat
  arma::mat Y_hat = inv_stereo_cpp(U);
  
  // Compute loss
  arma::mat residuals = Y - Y_hat;
  double loss = mean(sum(square(residuals), 1));
  
  // Compute roughness penalty
  double roughness = gamma * trace(Theta.t() * Omega * Theta);
  
  // Compute group penalty
  double group_penalty = compute_group_penalty_cpp(beta, group_idx, weights, lambda);
  
  return loss + roughness + group_penalty;
}


// [[Rcpp::export]]
List compute_group_norms_cpp(const arma::vec& beta,
                              const List& group_idx) {
  int G = group_idx.size();
  arma::vec group_norms(G);
  arma::uvec selected(G, fill::zeros);
  
  for(int g = 0; g < G; g++) {
    arma::uvec idx = as<arma::uvec>(group_idx[g]) - 1;
    group_norms(g) = norm(beta.elem(idx), 2);
    selected(g) = (group_norms(g) > 1e-6) ? 1 : 0;
  }
  
  return List::create(
    Named("norms") = group_norms,
    Named("selected") = selected
  );
}


// [[Rcpp::export]]
arma::mat compute_prediction_cpp(const arma::mat& B_new,
                                 const arma::mat& Theta) {
  arma::mat U_new = B_new * Theta;
  return inv_stereo_cpp(U_new);
}


// Efficient matrix-vector products for cross-validation
// [[Rcpp::export]]
arma::vec fast_matvec(const arma::mat& A, const arma::vec& x) {
  return A * x;
}


// Compute multiple objectives for lambda path (used in CV)
// [[Rcpp::export]]
arma::vec compute_objectives_lambda_path(const arma::mat& X,
                                         const arma::mat& Y,
                                         const arma::mat& beta_mat,
                                         const arma::cube& Theta_cube,
                                         const arma::cube& B_array,
                                         const arma::mat& Omega,
                                         const arma::vec& lambdas,
                                         double gamma,
                                         const List& group_idx,
                                         const arma::vec& weights) {
  int n_lambda = lambdas.n_elem;
  arma::vec objectives(n_lambda);
  
  for(int k = 0; k < n_lambda; k++) {
    arma::vec beta = beta_mat.col(k);
    arma::mat Theta = Theta_cube.slice(k);
    arma::mat B = B_array.slice(k);
    
    objectives(k) = compute_objective_cpp(beta, Theta, X, Y, B, Omega,
                                          lambdas(k), gamma, group_idx, weights);
  }
  
  return objectives;
}


// Weighted group lasso soft-thresholding for adaptive weights
// [[Rcpp::export]]
arma::vec adaptive_prox_cpp(const arma::vec& beta,
                            const List& group_idx,
                            const arma::vec& adaptive_weights,
                            double lambda,
                            double step) {
  return prox_group_lasso_cpp(beta, group_idx, adaptive_weights, lambda, step);
}


// Compute adaptive weights from initial beta estimate
// [[Rcpp::export]]
arma::vec compute_adaptive_weights_from_beta(const arma::vec& beta_init,
                                              const List& group_idx,
                                              double gamma_power,
                                              double epsilon = 1e-4) {
  int G = group_idx.size();
  arma::vec weights(G);
  arma::vec group_sizes(G);
  
  // Compute group norms and sizes
  for(int g = 0; g < G; g++) {
    arma::uvec idx = as<arma::uvec>(group_idx[g]) - 1;
    double norm_g = norm(beta_init.elem(idx), 2);
    group_sizes(g) = std::sqrt(static_cast<double>(idx.n_elem));
    weights(g) = group_sizes(g) / std::pow(norm_g + epsilon, gamma_power);
  }
  
  // Normalize weights
  double sum_weights = sum(weights);
  weights = weights * (static_cast<double>(G) / sum_weights);
  
  return weights;
}


// Fast computation of residual sum of squares for CV
// [[Rcpp::export]]
double compute_rss_cpp(const arma::mat& Y_pred, const arma::mat& Y_true) {
  arma::mat diff = Y_true - Y_pred;
  return accu(square(diff));
}


// Batch compute predictions for cross-validation
// [[Rcpp::export]]
arma::mat batch_predict_cpp(const arma::mat& X,
                            const arma::vec& beta,
                            const arma::mat& B,
                            const arma::mat& Theta) {
  arma::mat U = B * Theta;
  return inv_stereo_cpp(U);
}


// Compute fold-wise CV errors efficiently
// [[Rcpp::export]]
arma::vec compute_cv_errors_cpp(const arma::mat& X_test,
                                const arma::mat& Y_test,
                                const List& beta_list,
                                const List& Theta_list,
                                const List& B_list) {
  int n_models = beta_list.size();
  arma::vec errors(n_models);
  
  for(int k = 0; k < n_models; k++) {
    arma::vec beta = as<arma::vec>(beta_list[k]);
    arma::mat Theta = as<arma::mat>(Theta_list[k]);
    arma::mat B = as<arma::mat>(B_list[k]);
    
    arma::mat Y_pred = batch_predict_cpp(X_test, beta, B, Theta);
    errors(k) = compute_rss_cpp(Y_pred, Y_test) / Y_test.n_rows;
  }
  
  return errors;
}


// Optimized loop for proximal gradient iterations
// [[Rcpp::export]]
List proximal_gradient_step(const arma::vec& beta_current,
                            const arma::mat& Theta,
                            const arma::mat& X,
                            const arma::mat& Y,
                            const arma::mat& B,
                            const arma::mat& B_deriv,
                            const List& group_idx,
                            const arma::vec& weights,
                            double lambda,
                            double step_size,
                            int max_backtrack = 20) {
  
  // Compute gradient
  arma::vec grad = grad_beta_cpp(beta_current, Theta, X, Y, B, B_deriv);
  
  // Gradient step
  arma::vec beta_temp = beta_current - step_size * grad;
  
  // Proximal step
  arma::vec beta_new = prox_group_lasso_cpp(beta_temp, group_idx, weights, lambda, step_size);
  
  // Project to unit norm
  double beta_norm = norm(beta_new, 2);
  if(beta_norm > 1e-10) {
    beta_new = beta_new / beta_norm;
  }
  
  return List::create(
    Named("beta") = beta_new,
    Named("grad_norm") = norm(grad, 2)
  );
}

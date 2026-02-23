#' @useDynLib sphericalSIM, .registration = TRUE
#' @importFrom Rcpp sourceCpp
NULL

#' Use C++ Accelerated Functions
#'
#' Enables or disables C++ acceleration for computationally intensive functions.
#' By default, C++ acceleration is enabled if the package was compiled with
#' Rcpp support.
#'
#' @param use_cpp Logical; if TRUE (default), use C++ implementations when available
#'
#' @details
#' The package contains both R and C++ implementations of key functions. The C++
#' versions are typically 10-100x faster for large datasets. This function allows
#' you to switch between implementations, which can be useful for:
#' \itemize{
#'   \item Debugging (R versions easier to step through)
#'   \item Platforms where C++ compilation is unavailable
#'   \item Benchmarking performance improvements
#' }
#'
#' @return Invisibly returns the previous setting
#'
#' @examples
#' \dontrun{
#' # Use C++ (default)
#' set_cpp_enabled(TRUE)
#'
#' # Disable C++ for debugging
#' set_cpp_enabled(FALSE)
#' }
#'
#' @export
set_cpp_enabled <- function(use_cpp = TRUE) {
  old_val <- getOption("sphericalSIM.use_cpp", TRUE)
  options(sphericalSIM.use_cpp = use_cpp)
  invisible(old_val)
}

#' Check if C++ Acceleration is Enabled
#'
#' @return Logical indicating if C++ acceleration is currently enabled
#' @export
is_cpp_enabled <- function() {
  getOption("sphericalSIM.use_cpp", TRUE)
}


# Internal wrapper functions that choose between R and C++ implementations

inv_stereo_internal <- function(u) {
  if (is_cpp_enabled()) {
    return(inv_stereo_cpp(u))
  } else {
    return(inv_stereo(u))
  }
}

jacobian_inv_stereo_internal <- function(u) {
  if (is_cpp_enabled()) {
    return(jacobian_inv_stereo_cpp(u))
  } else {
    return(jacobian_inv_stereo(u))
  }
}

prox_group_lasso_internal <- function(beta, group_idx, weights, lambda, step) {
  if (is_cpp_enabled()) {
    return(prox_group_lasso_cpp(beta, group_idx, weights, lambda, step))
  } else {
    return(prox_group_lasso(beta, group_idx, weights, lambda, step))
  }
}

grad_beta_internal <- function(beta, Theta, X, Y, knots, degree) {
  if (!is_cpp_enabled()) {
    return(grad_beta(beta, Theta, X, Y, knots, degree))
  }
  
  # For C++ version, pre-compute B matrices
  z <- as.vector(X %*% beta)
  z_clip <- pmax(pmin(z, max(knots) - 1e-6), min(knots) + 1e-6)
  B <- splineDesign(knots, z_clip, ord = degree + 1)
  B_deriv <- splineDesign(knots, z_clip, ord = degree + 1, derivs = 1)
  
  return(grad_beta_cpp(beta, Theta, X, Y, B, B_deriv))
}

compute_objective_internal <- function(beta, Theta, X, Y, knots, degree, Omega,
                                        lambda, gamma, group_idx, weights) {
  if (!is_cpp_enabled()) {
    return(compute_objective(beta, Theta, X, Y, knots, degree, Omega,
                             lambda, gamma, group_idx, weights))
  }
  
  # For C++ version, pre-compute B matrix
  z <- as.vector(X %*% beta)
  z_clip <- pmax(pmin(z, max(knots) - 1e-6), min(knots) + 1e-6)
  B <- splineDesign(knots, z_clip, ord = degree + 1)
  
  return(compute_objective_cpp(beta, Theta, X, Y, B, Omega,
                               lambda, gamma, group_idx, weights))
}


#' Benchmark R vs C++ Performance
#'
#' Compare performance of R and C++ implementations on a test dataset.
#'
#' @param n Number of observations (default 200)
#' @param p Number of predictors (default 30)
#' @param G Number of groups (default 6)
#' @param q Dimension of sphere (default 3)
#'
#' @return A data frame with timing comparisons
#'
#' @examples
#' \dontrun{
#' # Benchmark on medium-sized problem
#' benchmark_cpp(n = 200, p = 30, G = 6)
#'
#' # Larger problem shows more dramatic speedup
#' benchmark_cpp(n = 500, p = 60, G = 10)
#' }
#'
#' @export
benchmark_cpp <- function(n = 200, p = 30, G = 6, q = 3) {
  # Generate test data
  set.seed(42)
  X <- matrix(rnorm(n * p), n, p)
  beta <- rnorm(p)
  beta <- beta / sqrt(sum(beta^2))
  
  q_minus_1 <- q - 1
  u <- matrix(rnorm(n * q_minus_1), n, q_minus_1)
  Y <- inv_stereo(u)
  
  group_idx <- split(1:p, rep(1:G, length.out = p))
  weights <- sapply(group_idx, function(idx) sqrt(length(idx)))
  
  # Setup for gradient computation
  n_knots <- 5
  knots <- seq(-3, 3, length.out = n_knots + 2)
  full_knots <- c(rep(knots[1], 4), knots, rep(knots[length(knots)], 4))
  n_basis <- n_knots + 4
  Theta <- matrix(rnorm(n_basis * q_minus_1), n_basis, q_minus_1)
  
  z <- as.vector(X %*% beta)
  z_clip <- pmax(pmin(z, max(full_knots) - 1e-6), min(full_knots) + 1e-6)
  B <- splineDesign(full_knots, z_clip, ord = 4)
  B_deriv <- splineDesign(full_knots, z_clip, ord = 4, derivs = 1)
  
  Omega <- diag(n_basis)
  
  results <- list()
  
  # Benchmark inv_stereo
  cat("Benchmarking inv_stereo...\n")
  t_r <- system.time(replicate(100, inv_stereo(u)))["elapsed"]
  t_cpp <- system.time(replicate(100, inv_stereo_cpp(u)))["elapsed"]
  results$inv_stereo <- data.frame(
    function_name = "inv_stereo",
    R_time = t_r,
    CPP_time = t_cpp,
    speedup = t_r / t_cpp
  )
  
  # Benchmark prox_group_lasso
  cat("Benchmarking prox_group_lasso...\n")
  t_r <- system.time(replicate(100, prox_group_lasso(beta, group_idx, weights, 0.1, 0.01)))["elapsed"]
  t_cpp <- system.time(replicate(100, prox_group_lasso_cpp(beta, group_idx, weights, 0.1, 0.01)))["elapsed"]
  results$prox <- data.frame(
    function_name = "prox_group_lasso",
    R_time = t_r,
    CPP_time = t_cpp,
    speedup = t_r / t_cpp
  )
  
  # Benchmark grad_beta
  cat("Benchmarking grad_beta...\n")
  t_r <- system.time(replicate(10, grad_beta(beta, Theta, X, Y, full_knots, 3)))["elapsed"]
  t_cpp <- system.time(replicate(10, grad_beta_cpp(beta, Theta, X, Y, B, B_deriv)))["elapsed"]
  results$grad <- data.frame(
    function_name = "grad_beta",
    R_time = t_r,
    CPP_time = t_cpp,
    speedup = t_r / t_cpp
  )
  
  # Benchmark compute_objective
  cat("Benchmarking compute_objective...\n")
  t_r <- system.time(replicate(10, compute_objective(beta, Theta, X, Y, full_knots, 3, 
                                                      Omega, 0.1, 0.01, group_idx, weights)))["elapsed"]
  t_cpp <- system.time(replicate(10, compute_objective_cpp(beta, Theta, X, Y, B, 
                                                            Omega, 0.1, 0.01, group_idx, weights)))["elapsed"]
  results$objective <- data.frame(
    function_name = "compute_objective",
    R_time = t_r,
    CPP_time = t_cpp,
    speedup = t_r / t_cpp
  )
  
  result_df <- do.call(rbind, results)
  rownames(result_df) <- NULL
  
  cat("\nBenchmark Results:\n")
  print(result_df, digits = 3)
  cat("\nAverage speedup:", mean(result_df$speedup), "x\n")
  
  invisible(result_df)
}

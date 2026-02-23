#' Proximal Operator for Group Lasso Penalty
#'
#' Computes the proximal operator for the weighted group lasso penalty, which
#' performs soft-thresholding at the group level. This is a key component in
#' proximal gradient methods for group-sparse optimization.
#'
#' @param beta Current coefficient vector (length p)
#' @param group_idx List of length G, where each element contains indices of predictors in that group
#' @param weights Vector of group-specific weights (length G)
#' @param lambda Regularization parameter controlling overall sparsity level
#' @param step Step size (learning rate) for the proximal gradient algorithm
#'
#' @details
#' The proximal operator for the weighted group lasso penalty is defined as:
#' \deqn{\text{prox}(β) = \arg\min_u \left\{ \frac{1}{2}||u - β||^2 + \text{step} · λ \sum_{g=1}^G w_g ||u_g||_2 \right\}}
#'
#' The solution has a closed form for each group g:
#' \deqn{u_g = \left(1 - \frac{\text{step} · λ · w_g}{||β_g||_2}\right)_+ β_g}
#' where (x)_+ = max(0, x) is the positive part function.
#'
#' \strong{Group-Level Soft-Thresholding:}
#' For each group g:
#' \itemize{
#'   \item Compute the L2 norm: ||β_g||_2 = sqrt(sum(β_g^2))
#'   \item Compute threshold: τ = step * λ * w_g
#'   \item If ||β_g||_2 > τ: shrink the group by factor (1 - τ/||β_g||_2)
#'   \item If ||β_g||_2 ≤ τ: set entire group to zero (group selection)
#' }
#'
#' \strong{Role in Proximal Gradient Method:}
#' In the proximal gradient algorithm for solving:
#' \deqn{\min f(β) + λ \sum_g w_g ||β_g||_2}
#' each iteration performs:
#' \enumerate{
#'   \item Gradient step: β_temp = β - step * ∇f(β)
#'   \item Proximal step: β_new = prox_group_lasso(β_temp, ...)
#' }
#'
#' \strong{Adaptive Weights:}
#' The weights w_g allow for adaptive group lasso, where w_g = ||β_init_g||^(-γ)
#' gives heavier penalties to groups with smaller initial estimates, encouraging
#' sparser solutions while protecting strong signals.
#'
#' @return Vector of updated coefficients (same length as beta) with group-level
#'   sparsity. Entire groups may be set to zero if their norm is below the threshold.
#'
#' @examples
#' # Create simple example with 3 groups
#' p <- 12
#' group_idx <- list(1:4, 5:8, 9:12)
#' weights <- c(2, 2, 2)  # Equal weights (times sqrt(group size))
#'
#' # Current coefficient estimate
#' beta <- c(0.5, 0.3, 0.2, 0.1,   # Group 1: moderate signal
#'           0.01, 0.02, 0.01, 0.01, # Group 2: weak signal
#'           1.0, 0.8, 0.9, 0.7)     # Group 3: strong signal
#'
#' # Apply proximal operator with moderate penalty
#' beta_new <- prox_group_lasso(beta, group_idx, weights,
#'                              lambda = 0.5, step = 0.1)
#'
#' # Check which groups survived
#' sapply(group_idx, function(idx) sqrt(sum(beta_new[idx]^2)))
#' # Group 2 (weak) likely set to zero, groups 1 and 3 shrunk
#'
#' # Visualize effect of proximal operator
#' par(mfrow = c(1, 2))
#' plot(beta, col = rep(1:3, each = 4), pch = 19,
#'      main = "Before Proximal Operator", ylab = "Coefficient")
#' abline(h = 0, lty = 2)
#' plot(beta_new, col = rep(1:3, each = 4), pch = 19,
#'      main = "After Proximal Operator", ylab = "Coefficient")
#' abline(h = 0, lty = 2)
#'
#' # Effect of different lambda values
#' lambdas <- c(0.1, 0.5, 1.0, 2.0)
#' results <- sapply(lambdas, function(lam) {
#'   beta_prox <- prox_group_lasso(beta, group_idx, weights,
#'                                 lambda = lam, step = 0.1)
#'   sapply(group_idx, function(idx) sqrt(sum(beta_prox[idx]^2)))
#' })
#'
#' matplot(lambdas, t(results), type = "l", lwd = 2,
#'         xlab = "Lambda", ylab = "Group Norm",
#'         main = "Group Norms vs Regularization")
#' legend("topright", paste("Group", 1:3), col = 1:3, lty = 1:3)
#'
#' # Adaptive weights example
#' initial_norms <- c(0.5, 0.05, 1.0)  # Initial estimates
#' adaptive_weights <- initial_norms^(-1)  # Inverse weighting
#'
#' beta_adaptive <- prox_group_lasso(beta, group_idx, adaptive_weights,
#'                                   lambda = 0.5, step = 0.1)
#'
#' # Compare standard vs adaptive
#' cat("Standard group norms:",
#'     sapply(group_idx, function(idx) sqrt(sum(beta_new[idx]^2))), "\n")
#' cat("Adaptive group norms:",
#'     sapply(group_idx, function(idx) sqrt(sum(beta_adaptive[idx]^2))), "\n")
#' # Adaptive version protects strong groups more
#'
#' # Step size effect (with fixed lambda)
#' steps <- c(0.01, 0.05, 0.1, 0.2)
#' step_results <- sapply(steps, function(s) {
#'   beta_prox <- prox_group_lasso(beta, group_idx, weights,
#'                                 lambda = 1.0, step = s)
#'   sapply(group_idx, function(idx) sqrt(sum(beta_prox[idx]^2)))
#' })
#'
#' matplot(steps, t(step_results), type = "l", lwd = 2,
#'         xlab = "Step Size", ylab = "Group Norm",
#'         main = "Group Norms vs Step Size")
#' }
#'
#' @references
#' - Yuan, M., & Lin, Y. (2006). Model selection and estimation in regression
#'   with grouped variables. Journal of the Royal Statistical Society: Series B,
#'   68(1), 49-67.
#' - Parikh, N., & Boyd, S. (2014). Proximal algorithms. Foundations and Trends
#'   in Optimization, 1(3), 127-239.
#' - Zou, H. (2006). The adaptive lasso and its oracle properties. Journal of
#'   the American Statistical Association, 101(476), 1418-1429.
#'
#' @seealso
#' \code{\link{spherical_sim_group}} for the main optimization routine using this operator,
#' \code{\link{compute_adaptive_weights_fast}} for computing adaptive weights,
#' \code{\link{compute_adaptive_weights_grouplasso}} for alternative weight computation
#'
#' @export
prox_group_lasso <- function(beta, group_idx, weights, lambda, step) {
  # Use C++ version if enabled
  if (getOption("sphericalSIM.use_cpp", TRUE)) {
    return(prox_group_lasso_cpp(beta, group_idx, weights, lambda, step))
  }

  # Original R implementation
  beta_new <- beta
  for (g in seq_along(group_idx)) {
    idx <- group_idx[[g]]
    beta_g <- beta[idx]
    norm_g <- sqrt(sum(beta_g^2))
    threshold <- step * lambda * weights[g]
    if (norm_g > threshold) {
      beta_new[idx] <- (1 - threshold / norm_g) * beta_g
    } else {
      beta_new[idx] <- 0
    }
  }
  beta_new
}

#' Compute Objective Function for Spherical Single-Index Model with Group Lasso
#'
#' Computes the complete objective function including prediction loss, link function
#' roughness penalty, and group lasso penalty for sparse index parameter estimation.
#'
#' @param beta Index parameter vector (length p)
#' @param Theta B-spline coefficient matrix for link function approximation (n_basis × q_minus_1)
#' @param X Design matrix (n × p)
#' @param Y Response matrix on unit sphere (n × q)
#' @param knots Vector of B-spline knots for link function approximation
#' @param degree Degree of B-spline basis (typically 3 for cubic splines)
#' @param Omega Penalty matrix for roughness control (n_basis × n_basis)
#' @param lambda Regularization parameter for group lasso penalty
#' @param gamma Regularization parameter for link function roughness
#' @param group_idx List of group indices
#' @param weights Vector of group-specific weights (length G)
#'
#' @details
#' The objective function is:
#' \deqn{Q(β, Θ) = L(β, Θ) + γR(Θ) + λP(β)}
#'
#' where:
#' \describe{
#'   \item{\strong{Loss term:}}{L(β, Θ) = (1/n) Σ_i ||Y_i - m(X_i^T β)||^2,
#'         measuring prediction error on the sphere}
#'   \item{\strong{Roughness penalty:}}{R(Θ) = tr(Θ^T Ω Θ),
#'         controlling smoothness of the link function via the penalty matrix Ω}
#'   \item{\strong{Group lasso penalty:}}{P(β) = Σ_g w_g ||β_g||_2,
#'         inducing group-level sparsity in the index parameter}
#' }
#'
#' \strong{Computational Steps:}
#' \enumerate{
#'   \item Compute single index: z_i = X_i^T β
#'   \item Clip z values to valid knot range to avoid extrapolation
#'   \item Evaluate B-spline basis: B(z_i)
#'   \item Compute link function in R^{q-1}: U_i = B(z_i) Θ
#'   \item Map to sphere via inverse stereographic projection: Ŷ_i = π^{-1}(U_i)
#'   \item Calculate squared error: ||Y_i - Ŷ_i||^2
#'   \item Add roughness penalty: γ tr(Θ^T Ω Θ)
#'   \item Add group lasso penalty: λ Σ_g w_g ||β_g||_2
#' }
#'
#' \strong{Parameter Roles:}
#' \itemize{
#'   \item λ controls sparsity: larger values select fewer groups
#'   \item γ controls smoothness: larger values produce smoother link functions
#'   \item weights allow adaptive penalties: typically w_g ∝ ||β_init_g||^{-γ}
#' }
#'
#' @return Scalar value of the objective function
#'
#' @examples
#' \dontrun{
#' # Generate synthetic data
#' set.seed(123)
#' data <- generate_spherical_data(n = 100, p = 20, G = 4)
#'
#' # Setup for objective computation
#' group_idx <- data$group_idx
#' weights <- sapply(group_idx, function(idx) sqrt(length(idx)))
#'
#' # B-spline setup
#' n_knots <- 5
#' z_range <- range(data$X %*% data$beta_true)
#' knots <- seq(z_range[1] - 0.1, z_range[2] + 0.1, length.out = n_knots + 2)
#' full_knots <- c(rep(knots[1], 4), knots, rep(knots[length(knots)], 4))
#'
#' # Initialize parameters
#' beta <- data$beta_true + rnorm(20, 0, 0.1)
#' beta <- beta / sqrt(sum(beta^2))
#'
#' n_basis <- n_knots + 4
#' Theta <- matrix(rnorm(n_basis * 2), n_basis, 2)
#'
#' # Penalty matrix for roughness
#' D2 <- diff(diag(n_basis), differences = 2)
#' Omega <- t(D2) %*% D2
#'
#' # Compute objective for different lambda values
#' lambdas <- c(0.01, 0.1, 0.5, 1.0)
#' objectives <- sapply(lambdas, function(lam) {
#'   compute_objective(beta, Theta, data$X, data$Y,
#'                    full_knots, degree = 3, Omega,
#'                    lambda = lam, gamma = 0.1,
#'                    group_idx, weights)
#' })
#'
#' plot(lambdas, objectives, type = "b", log = "x",
#'      xlab = "Lambda", ylab = "Objective Value",
#'      main = "Objective vs Regularization")
#'
#' # Decompose objective into components
#' lambda <- 0.1
#' gamma <- 0.1
#'
#' # Full objective
#' obj_full <- compute_objective(beta, Theta, data$X, data$Y,
#'                               full_knots, 3, Omega,
#'                               lambda, gamma, group_idx, weights)
#'
#' # Loss only (lambda = gamma = 0)
#' obj_loss <- compute_objective(beta, Theta, data$X, data$Y,
#'                               full_knots, 3, Omega,
#'                               lambda = 0, gamma = 0, group_idx, weights)
#'
#' # Loss + roughness (lambda = 0)
#' obj_smooth <- compute_objective(beta, Theta, data$X, data$Y,
#'                                 full_knots, 3, Omega,
#'                                 lambda = 0, gamma, group_idx, weights)
#'
#' cat("Loss component:", obj_loss, "\n")
#' cat("+ Roughness:", obj_smooth - obj_loss, "\n")
#' cat("+ Group penalty:", obj_full - obj_smooth, "\n")
#' cat("Total objective:", obj_full, "\n")
#' }
#'
#' @seealso
#' \code{\link{grad_beta}} for gradient computation,
#' \code{\link{spherical_sim_group}} for optimization using this objective,
#' \code{\link{inv_stereo}} for inverse stereographic projection
#'
#' @export
compute_objective <- function(beta, Theta, X, Y, knots, degree, Omega,
                              lambda, gamma, group_idx, weights) {
  # Use C++ version if enabled
  if (getOption("sphericalSIM.use_cpp", TRUE)) {
    z <- as.vector(X %*% beta)
    z_clip <- pmax(pmin(z, max(knots) - 1e-6), min(knots) + 1e-6)
    B <- splineDesign(knots, z_clip, ord = degree + 1)
    return(compute_objective_cpp(beta, Theta, X, Y, B, Omega,
                                 lambda, gamma, group_idx, weights))
  }

  # Original R implementation
  n <- nrow(X)
  q_minus_1 <- ncol(Theta)

  z <- as.vector(X %*% beta)
  z_clip <- pmax(pmin(z, max(knots) - 1e-6), min(knots) + 1e-6)

  B <- splineDesign(knots, z_clip, ord = degree + 1)
  U <- B %*% Theta
  Y_hat <- inv_stereo(U)

  residuals <- Y - Y_hat
  loss <- mean(rowSums(residuals^2))

  roughness <- gamma * sum(diag(t(Theta) %*% Omega %*% Theta))

  group_penalty <- 0
  for (g in seq_along(group_idx)) {
    idx <- group_idx[[g]]
    group_penalty <- group_penalty + weights[g] * sqrt(sum(beta[idx]^2))
  }
  group_penalty <- lambda * group_penalty

  loss + roughness + group_penalty
}


#' Compute Gradient of Loss with Respect to Index Parameter
#'
#' Computes the gradient of the spherical prediction loss with respect to the
#' index parameter β using the chain rule through the B-spline basis and
#' inverse stereographic projection.
#'
#' @param beta Index parameter vector (length p)
#' @param Theta B-spline coefficient matrix for link function (n_basis × q_minus_1)
#' @param X Design matrix (n × p)
#' @param Y Response matrix on unit sphere (n × q)
#' @param knots Vector of B-spline knots
#' @param degree Degree of B-spline basis (typically 3 for cubic splines)
#'
#' @details
#' For the loss function L(β) = (1/n) Σ_i ||Y_i - m(X_i^T β)||^2, the gradient is:
#' \deqn{∇_β L = (2/n) X^T t}
#' where t_i = (Y_i - Ŷ_i)^T (∂Ŷ_i/∂z_i) with z_i = X_i^T β.
#'
#' \strong{Chain Rule Decomposition:}
#' The derivative ∂Ŷ_i/∂z_i is computed via the chain rule:
#' \deqn{∂Ŷ_i/∂z_i = (∂π^{-1}/∂u)(U_i) · (∂u/∂z)(z_i)}
#' where:
#' \itemize{
#'   \item π^{-1} is the inverse stereographic projection
#'   \item (∂π^{-1}/∂u)(U_i) is the Jacobian matrix (q × (q-1)) computed by
#'         \code{\link{jacobian_inv_stereo}}
#'   \item (∂u/∂z)(z_i) = B'(z_i) Θ where B'(z_i) is the derivative of the
#'         B-spline basis
#' }
#'
#' \strong{Computational Steps:}
#' \enumerate{
#'   \item Compute single index: z_i = X_i^T β
#'   \item Clip z to valid knot range
#'   \item Evaluate B-spline basis B(z_i) and its derivative B'(z_i)
#'   \item Compute link function: U_i = B(z_i) Θ
#'   \item Map to sphere: Ŷ_i = π^{-1}(U_i)
#'   \item Compute residuals: E_i = Ŷ_i - Y_i
#'   \item For each i, compute Jacobian J_i = (∂π^{-1}/∂u)(U_i)
#'   \item Compute link derivative: m'_i = B'(z_i) Θ
#'   \item Apply chain rule: t_i = E_i^T · (J_i · m'_i)
#'   \item Aggregate: ∇L = (2/n) X^T t
#' }
#'
#' \strong{Usage in Optimization:}
#' This gradient is used in the proximal gradient algorithm:
#' \enumerate{
#'   \item Gradient step: β_temp = β - step * grad_beta(β, Θ, X, Y, ...)
#'   \item Proximal step: β_new = prox_group_lasso(β_temp, ...)
#' }
#'
#' The gradient computation does not include the group lasso penalty because
#' it is handled separately via the proximal operator.
#'
#' @return Gradient vector of length p: ∇_β L(β)
#'
#' @examples
#' \dontrun{
#' # Generate synthetic data
#' set.seed(123)
#' data <- generate_spherical_data(n = 100, p = 20)
#'
#' # B-spline setup
#' n_knots <- 5
#' z_range <- range(data$X %*% data$beta_true)
#' knots <- seq(z_range[1] - 0.1, z_range[2] + 0.1, length.out = n_knots + 2)
#' full_knots <- c(rep(knots[1], 4), knots, rep(knots[length(knots)], 4))
#'
#' # Initialize parameters
#' beta <- rnorm(20)
#' beta <- beta / sqrt(sum(beta^2))
#'
#' n_basis <- n_knots + 4
#' Theta <- matrix(rnorm(n_basis * 2), n_basis, 2)
#'
#' # Compute gradient
#' grad <- grad_beta(beta, Theta, data$X, data$Y, full_knots, degree = 3)
#'
#' cat("Gradient norm:", sqrt(sum(grad^2)), "\n")
#' cat("Gradient range: [", min(grad), ",", max(grad), "]\n")
#'
#' # Verify gradient numerically
#' eps <- 1e-6
#' grad_numerical <- numeric(length(beta))
#'
#' for (j in 1:length(beta)) {
#'   beta_plus <- beta
#'   beta_plus[j] <- beta[j] + eps
#'
#'   beta_minus <- beta
#'   beta_minus[j] <- beta[j] - eps
#'
#'   # Compute loss at both points
#'   loss_plus <- mean(rowSums((data$Y - inv_stereo(
#'     splineDesign(full_knots, as.vector(data$X %*% beta_plus), 4) %*% Theta
#'   ))^2))
#'
#'   loss_minus <- mean(rowSums((data$Y - inv_stereo(
#'     splineDesign(full_knots, as.vector(data$X %*% beta_minus), 4) %*% Theta
#'   ))^2))
#'
#'   grad_numerical[j] <- (loss_plus - loss_minus) / (2 * eps)
#' }
#'
#' # Compare analytical vs numerical gradient
#' plot(grad, grad_numerical, asp = 1,
#'      xlab = "Analytical Gradient", ylab = "Numerical Gradient",
#'      main = "Gradient Verification")
#' abline(0, 1, col = "red", lty = 2)
#'
#' cat("Max absolute difference:", max(abs(grad - grad_numerical)), "\n")
#' cat("Relative error:",
#'     max(abs(grad - grad_numerical)) / max(abs(grad)), "\n")
#'
#' # Visualize gradient along coordinates
#' barplot(grad, main = "Gradient Components",
#'         xlab = "Coefficient Index", ylab = "Gradient Value")
#' abline(h = 0, lty = 2)
#' }
#'
#' @references
#' - de Boor, C. (2001). A Practical Guide to Splines. Springer.
#' - Parikh, N., & Boyd, S. (2014). Proximal algorithms. Foundations and Trends
#'   in Optimization, 1(3), 127-239.
#'
#' @seealso
#' \code{\link{compute_objective}} for the full objective function,
#' \code{\link{prox_group_lasso}} for the proximal operator,
#' \code{\link{jacobian_inv_stereo}} for Jacobian computation,
#' \code{\link{spherical_sim_group}} for the complete optimization algorithm
#'
#' @importFrom splines splineDesign
#' @export
grad_beta <- function(beta, Theta, X, Y, knots, degree) {
  # Use C++ version if enabled
  if (getOption("sphericalSIM.use_cpp", TRUE)) {
    z <- as.vector(X %*% beta)
    z_clip <- pmax(pmin(z, max(knots) - 1e-6), min(knots) + 1e-6)
    B <- splineDesign(knots, z_clip, ord = degree + 1)
    B_deriv <- splineDesign(knots, z_clip, ord = degree + 1, derivs = 1)
    return(grad_beta_cpp(beta, Theta, X, Y, B, B_deriv))
  }

  # Original R implementation
  n <- nrow(X)

  z <- as.vector(X %*% beta)
  z_clip <- pmax(pmin(z, max(knots) - 1e-6), min(knots) + 1e-6)

  B <- splineDesign(knots, z_clip, ord = degree + 1)
  B_deriv <- splineDesign(knots, z_clip, ord = degree + 1, derivs = 1)

  U <- B %*% Theta
  Y_hat <- inv_stereo(U)
  E <- Y_hat - Y

  m_prime <- B_deriv %*% Theta

  t_vec <- numeric(n)
  for (i in 1:n) {
    J_i <- jacobian_inv_stereo(U[i, ])
    dy_dz <- J_i %*% m_prime[i, ]
    t_vec[i] <- sum(E[i, ] * dy_dz)
  }

  (2 / n) * t(X) %*% t_vec
}

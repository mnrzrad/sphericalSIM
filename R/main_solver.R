#' Fit Spherical Single-Index Model with Group Lasso
#'
#' Estimates a spherical single-index model with group-sparse index parameter
#' using alternating optimization with proximal gradient descent for β and
#' L-BFGS for the link function coefficients Θ.
#'
#' @param X Design matrix (n × p)
#' @param Y Response matrix on unit sphere (n × q), where each row satisfies ||Y_i|| = 1
#' @param group_idx List of length G, where each element contains indices of predictors in that group
#' @param lambda Regularization parameter for group lasso penalty (controls sparsity)
#' @param gamma Regularization parameter for link function roughness (controls smoothness)
#' @param n_knots Number of internal B-spline knots for link function approximation (default 10)
#' @param degree Degree of B-spline basis functions (default 3 for cubic splines)
#' @param max_iter Maximum number of alternating optimization iterations (default 200)
#' @param tol Convergence tolerance for relative change in objective and β (default 1e-6)
#' @param beta_init Initial value for index parameter. If NULL, initialized randomly (default NULL)
#' @param weights Vector of group-specific weights (length G). If NULL, uses sqrt(group_size) (default NULL)
#' @param verbose Print progress messages every 10 iterations (default TRUE)
#'
#' @details
#' \strong{Model:}
#' The spherical single-index model is:
#' \deqn{Y_i = m(X_i^T β) + ε_i}
#' where Y_i ∈ S^{q-1}, β ∈ R^p with ||β|| = 1, and m: R → S^{q-1} is an unknown smooth link function.
#'
#' \strong{Objective Function:}
#' \deqn{Q(β, Θ) = \frac{1}{n}\sum_{i=1}^n ||Y_i - m(X_i^T β)||^2 + γ·\text{tr}(Θ^T Ω Θ) + λ\sum_{g=1}^G w_g ||β_g||_2}
#'
#' Components:
#' \itemize{
#'   \item \strong{Prediction loss}: Mean squared error on the sphere
#'   \item \strong{Roughness penalty}: Controls smoothness of link function via penalty matrix Ω
#'   \item \strong{Group lasso penalty}: Induces group-level sparsity in β
#' }
#'
#' \strong{Link Function Approximation:}
#' The link function is approximated using B-splines:
#' \enumerate{
#'   \item Compute single index: z_i = X_i^T β
#'   \item Represent link in R^{q-1}: u(z) = B(z)Θ where B(z) is the B-spline basis
#'   \item Map to sphere: m(z) = π^{-1}(u(z)) via inverse stereographic projection
#' }
#'
#' \strong{Alternating Optimization Algorithm:}
#' \enumerate{
#'   \item \strong{Fix β, update Θ}: Solve smooth optimization problem using L-BFGS
#'         \deqn{\min_Θ L(β, Θ) + γ·\text{tr}(Θ^T Ω Θ)}
#'   \item \strong{Fix Θ, update β}: Use proximal gradient method with backtracking
#'         \itemize{
#'           \item Gradient step: β̃ = β - η∇_β L(β, Θ)
#'           \item Proximal step: β_tilde = prox_η λ P(β̃) (group-level soft-thresholding)
#'           \item Projection: β_new = β_tilde / ||β_tilde|| (enforce unit norm)
#'           \item Backtracking line search to ensure sufficient decrease
#'         }
#'   \item Iterate until convergence (relative change < tol in both objective and β)
#' }
#'
#' \strong{Convergence Criteria:}
#' Algorithm stops when:
#' \itemize{
#'   \item Relative objective change < tol: |Q_t - Q_{t-1}| / |Q_{t-1}| < tol
#'   \item Parameter change < tol: ||β_t - β_{t-1}|| < tol
#'   \item Maximum iterations reached
#' }
#'
#' \strong{Adaptive Weights:}
#' Adaptive group lasso uses data-driven weights w_g = ||β̂_g^{init}||^{-γ} to reduce
#' bias. These should be computed externally (e.g., via \code{\link{compute_adaptive_weights_fast}})
#' and passed through the weights argument.
#'
#' @return A list with components:
#' \describe{
#'   \item{beta}{Estimated index parameter (length p, unit norm)}
#'   \item{Theta}{B-spline coefficient matrix for link function (n_basis × q_minus_1)}
#'   \item{knots}{Extended knot sequence used for B-spline basis}
#'   \item{degree}{Degree of B-spline basis}
#'   \item{Omega}{Penalty matrix for roughness control (n_basis × n_basis)}
#'   \item{objective}{Vector of objective values across iterations}
#'   \item{selected_groups}{Indices of selected groups (with ||β_g|| > 1e-6)}
#'   \item{group_norms}{Vector of L2 norms for each group}
#'   \item{fitted}{Fitted values on sphere (n × q matrix)}
#'   \item{index}{Single index values X %*% β (length n)}
#'   \item{n_iter}{Number of iterations until convergence}
#' }
#'
#' @examples
#' \dontrun{
#' # Generate synthetic data
#' set.seed(123)
#' data <- generate_spherical_data(n = 200, p = 30, G = 6,
#'                                 active_groups = c(1, 3))
#'
#' # Fit with standard weights
#' fit <- spherical_sim_group(
#'   X = data$X,
#'   Y = data$Y,
#'   group_idx = data$group_idx,
#'   lambda = 0.1,
#'   gamma = 0.05,
#'   n_knots = 8,
#'   verbose = TRUE
#' )
#'
#' # Check selected groups
#' cat("Selected groups:", fit$selected_groups, "\n")
#' cat("True active groups:", data$active_groups, "\n")
#'
#' # Plot convergence
#' plot(fit$objective, type = "l", log = "y",
#'      xlab = "Iteration", ylab = "Objective",
#'      main = "Convergence Path")
#'
#' # Visualize group selection
#' barplot(fit$group_norms, names.arg = 1:length(fit$group_norms),
#'         xlab = "Group", ylab = "Group Norm",
#'         main = "Estimated Group Structure")
#' abline(h = 0, lty = 2)
#'
#' # Fit with adaptive weights
#' adaptive_res <- compute_adaptive_weights_fast(
#'   data$X, data$Y, data$group_idx
#' )
#'
#' fit_adaptive <- spherical_sim_group(
#'   X = data$X,
#'   Y = data$Y,
#'   group_idx = data$group_idx,
#'   lambda = 0.1,
#'   gamma = 0.05,
#'   weights = adaptive_res$weights,
#'   verbose = TRUE
#' )
#'
#' # Compare selection
#' cat("Standard selected:", fit$selected_groups, "\n")
#' cat("Adaptive selected:", fit_adaptive$selected_groups, "\n")
#'
#' # Visualize fitted link function
#' z_grid <- seq(min(fit$index), max(fit$index), length.out = 100)
#' z_clip <- pmax(pmin(z_grid, max(fit$knots) - 1e-6),
#'                min(fit$knots) + 1e-6)
#' B_grid <- splineDesign(fit$knots, z_clip, ord = fit$degree + 1)
#' U_grid <- B_grid %*% fit$Theta
#' m_grid <- inv_stereo(U_grid)
#'
#' par(mfrow = c(1, 2))
#' plot(z_grid, m_grid[,1], type = "l", lwd = 2,
#'      xlab = "z = X^T β", ylab = "m_1(z)",
#'      main = "Link Function Component 1")
#' plot(z_grid, m_grid[,2], type = "l", lwd = 2,
#'      xlab = "z = X^T β", ylab = "m_2(z)",
#'      main = "Link Function Component 2")
#'
#' # Compute prediction error
#' pred_error <- mean(rowSums((data$Y - fit$fitted)^2))
#' cat("Prediction error:", pred_error, "\n")
#'
#' # Compare true vs estimated β (accounting for sign)
#' cor_pos <- cor(data$beta_true, fit$beta)
#' cor_neg <- cor(data$beta_true, -fit$beta)
#' cat("Correlation with true β:", max(abs(cor_pos), abs(cor_neg)), "\n")
#'
#' # Plot true vs estimated coefficients
#' plot(data$beta_true, fit$beta, pch = 19,
#'      col = data$groups, asp = 1,
#'      xlab = "True β", ylab = "Estimated β",
#'      main = "Parameter Recovery")
#' abline(0, 1, col = "red", lty = 2)
#' abline(0, -1, col = "blue", lty = 2)
#' }
#'
#' @references
#' - Yuan, M., & Lin, Y. (2006). Model selection and estimation in regression
#'   with grouped variables. Journal of the Royal Statistical Society: Series B,
#'   68(1), 49-67.
#' - Eilers, P. H., & Marx, B. D. (1996). Flexible smoothing with B-splines and
#'   penalties. Statistical Science, 11(2), 89-121.
#' - Parikh, N., & Boyd, S. (2014). Proximal algorithms. Foundations and Trends
#'   in Optimization, 1(3), 127-239.
#'
#' @seealso
#' \code{\link{predict_spherical}} for prediction on new data,
#' \code{\link{cv_two_stage_adaptive}} for cross-validation,
#' \code{\link{compute_adaptive_weights_fast}} for adaptive weight computation,
#' \code{\link{prox_group_lasso}} for the proximal operator,
#' \code{\link{generate_spherical_data}} for data generation
#'
#' @importFrom splines splineDesign
#' @export
spherical_sim_group <- function(X, Y, group_idx, lambda, gamma,
                                n_knots = 10, degree = 3,
                                max_iter = 200, tol = 1e-6,
                                beta_init = NULL,
                                weights = NULL,  # ADDED: custom weights
                                verbose = TRUE) {

  n <- nrow(X)
  p <- ncol(X)
  q <- ncol(Y)
  q_minus_1 <- q - 1
  G <- length(group_idx)

  # MODIFIED: Use provided weights or compute standard
  if (is.null(weights)) {
    weights <- sapply(group_idx, function(idx) sqrt(length(idx)))
  }

  if (is.null(beta_init)) {
    beta <- rnorm(p)
  } else {
    beta <- beta_init
  }
  beta <- beta / sqrt(sum(beta^2))

  z <- as.vector(X %*% beta)
  knots <- build_knots(z, n_internal = n_knots, degree = degree)
  K <- length(knots) - degree - 1

  Omega <- compute_omega(knots, degree)
  Theta <- matrix(0, nrow = K, ncol = q_minus_1)

  obj_history <- numeric(max_iter)
  lbfgs_maxit <- 50

  for (iter in 1:max_iter) {
    beta_old <- beta

    # Update Theta
    z <- as.vector(X %*% beta)
    z_clip <- pmax(pmin(z, max(knots) - 1e-6), min(knots) + 1e-6)
    B_mat <- splineDesign(knots, z_clip, ord = degree + 1)

    fn_Theta <- function(theta_vec) {
      Theta_mat <- matrix(theta_vec, nrow = K, ncol = q_minus_1)
      U <- B_mat %*% Theta_mat
      Y_hat <- inv_stereo(U)
      E <- Y - Y_hat
      loss <- mean(rowSums(E^2))
      roughness <- gamma * sum(diag(t(Theta_mat) %*% Omega %*% Theta_mat))
      loss + roughness
    }

    gr_Theta <- function(theta_vec) {
      Theta_mat <- matrix(theta_vec, nrow = K, ncol = q_minus_1)
      U <- B_mat %*% Theta_mat
      Y_hat <- inv_stereo(U)
      E <- Y_hat - Y

      S <- matrix(0, nrow = n, ncol = q_minus_1)
      for (i in 1:n) {
        J_i <- jacobian_inv_stereo(U[i, ])
        S[i, ] <- t(J_i) %*% E[i, ]
      }

      grad_mat <- (2 / n) * t(B_mat) %*% S + 2 * gamma * Omega %*% Theta_mat
      as.vector(grad_mat)
    }

    opt_Theta <- optim(
      par = as.vector(Theta),
      fn = fn_Theta,
      gr = gr_Theta,
      method = "L-BFGS-B",
      control = list(maxit = lbfgs_maxit, factr = 1e7)
    )
    Theta <- matrix(opt_Theta$par, nrow = K, ncol = q_minus_1)

    # Update beta
    eta <- 1.0
    eta_min <- 1e-10
    shrink <- 0.5

    for (bt in 1:20) {
      g_beta <- grad_beta(beta, Theta, X, Y, knots, degree)
      beta_tilde <- beta - eta * g_beta
      beta_tilde <- prox_group_lasso(beta_tilde, group_idx, weights, lambda, eta)

      beta_norm <- sqrt(sum(beta_tilde^2))
      if (beta_norm < 1e-10) {
        beta_tilde <- rnorm(p)
        beta_norm <- sqrt(sum(beta_tilde^2))
      }
      beta_new <- beta_tilde / beta_norm

      if (eta < eta_min) break

      obj_old <- compute_objective(beta, Theta, X, Y, knots, degree, Omega,
                                   lambda, gamma, group_idx, weights)
      obj_new <- compute_objective(beta_new, Theta, X, Y, knots, degree, Omega,
                                   lambda, gamma, group_idx, weights)

      if (obj_new <= obj_old + 1e-4 * sum(g_beta * (beta_new - beta))) {
        break
      }
      eta <- eta * shrink
    }

    beta <- beta_new

    obj_val <- compute_objective(beta, Theta, X, Y, knots, degree, Omega,
                                 lambda, gamma, group_idx, weights)
    obj_history[iter] <- obj_val

    beta_change <- sqrt(sum((beta - beta_old)^2))

    if (verbose && iter %% 10 == 0) {
      active_groups <- sum(sapply(group_idx, function(idx) sqrt(sum(beta[idx]^2)) > 1e-6))
      cat(sprintf("Iter %3d: Obj = %.6f, ||beta_change|| = %.2e, Active = %d/%d\n",
                  iter, obj_val, beta_change, active_groups, G))
    }

    if (iter > 1 && abs(obj_history[iter] - obj_history[iter-1]) /
        (abs(obj_history[iter-1]) + 1e-10) < tol && beta_change < tol) {
      if (verbose) cat("Converged at iteration", iter, "\n")
      break
    }
  }

  obj_history <- obj_history[1:iter]
  group_norms <- sapply(group_idx, function(idx) sqrt(sum(beta[idx]^2)))
  selected_groups <- which(group_norms > 1e-6)

  z_final <- as.vector(X %*% beta)
  z_clip <- pmax(pmin(z_final, max(knots) - 1e-6), min(knots) + 1e-6)
  B_final <- splineDesign(knots, z_clip, ord = degree + 1)
  U_final <- B_final %*% Theta
  Y_hat <- inv_stereo(U_final)

  list(
    beta = beta,
    Theta = Theta,
    knots = knots,
    degree = degree,
    Omega = Omega,
    objective = obj_history,
    selected_groups = selected_groups,
    group_norms = group_norms,
    fitted = Y_hat,
    index = z_final,
    n_iter = iter
  )
}

#' Predict Spherical Responses for New Data
#'
#' Generates predictions on the unit sphere for new covariate observations using
#' a fitted spherical single-index model with group lasso.
#'
#' @param fit Fitted model object returned by \code{\link{spherical_sim_group}}
#' @param X_new New design matrix (n_new × p) for prediction
#'
#' @details
#' The prediction process follows these steps:
#' \enumerate{
#'   \item Compute single index for new data: z_new = X_new β̂
#'   \item Clip z values to valid knot range to avoid extrapolation
#'   \item Evaluate B-spline basis at z_new: B(z_new)
#'   \item Compute link function in R^{q-1}: U_new = B(z_new) Θ̂
#'   \item Map to unit sphere via inverse stereographic projection: Ŷ_new = π^{-1}(U_new)
#' }
#'
#' \strong{Valid Prediction Range:}
#' Predictions are most reliable when z_new values fall within the range of
#' single index values observed in the training data. The function clips z values
#' to the knot range, but extrapolation beyond the training range may be unreliable.
#'
#' \strong{Properties of Predictions:}
#' \itemize{
#'   \item Each predicted response lies exactly on the unit sphere: ||Ŷ_i|| = 1
#'   \item Predictions respect the single-index structure: similar X^T β values
#'         produce similar predictions
#'   \item Group sparsity in β̂ means predictions only depend on selected groups
#' }
#'
#' @return Matrix of predicted responses (n_new × q) where each row is a unit
#'   vector on S^{q-1}. The i-th row is the predicted spherical response for
#'   the i-th row of X_new.
#'
#' @examples
#' \dontrun{
#' # Generate training data
#' set.seed(123)
#' data <- generate_spherical_data(n = 200, p = 20, G = 4, seed = 123)
#'
#' # Fit model
#' fit <- spherical_sim_group(
#'   X = data$X,
#'   Y = data$Y,
#'   group_idx = data$group_idx,
#'   lambda = 0.1,
#'   gamma = 0.1,
#'   n_knots = 5,
#'   verbose = FALSE
#' )
#'
#' # Generate test data (same distribution)
#' test_data <- generate_spherical_data(n = 50, p = 20, G = 4, seed = 456)
#'
#' # Make predictions
#' Y_pred <- predict_spherical(fit, test_data$X)
#'
#' # Verify predictions are on unit sphere
#' all(abs(rowSums(Y_pred^2) - 1) < 1e-10)  # TRUE
#'
#' # Compute prediction error
#' pred_error <- mean(rowSums((Y_pred - test_data$Y)^2))
#' cat("Test prediction error:", pred_error, "\n")
#'
#' # Compare with training error
#' train_error <- mean(rowSums((fit$fitted - data$Y)^2))
#' cat("Train error:", train_error, "\n")
#' cat("Test error:", pred_error, "\n")
#'
#' # Visualize predictions vs true responses (first 2 dimensions)
#' par(mfrow = c(1, 2))
#' plot(test_data$Y[,1], test_data$Y[,2], asp = 1,
#'      col = "blue", pch = 19, cex = 0.8,
#'      xlab = "Y[,1]", ylab = "Y[,2]",
#'      main = "True Responses")
#' plot(Y_pred[,1], Y_pred[,2], asp = 1,
#'      col = "red", pch = 19, cex = 0.8,
#'      xlab = "Ŷ[,1]", ylab = "Ŷ[,2]",
#'      main = "Predicted Responses")
#'
#' # Prediction error vs single index
#' z_test <- as.vector(test_data$X %*% fit$beta)
#' errors <- rowSums((Y_pred - test_data$Y)^2)
#' plot(z_test, errors, pch = 19, cex = 0.8,
#'      xlab = "Single Index z", ylab = "Squared Error",
#'      main = "Prediction Error vs Single Index")
#'
#' # Out-of-sample prediction on new design points
#' X_new <- matrix(rnorm(10 * 20), nrow = 10, ncol = 20)
#' Y_new_pred <- predict_spherical(fit, X_new)
#'
#' # Check which groups are used for prediction
#' cat("Selected groups:", fit$selected_groups, "\n")
#' cat("Only these groups affect predictions\n")
#' }
#'
#' @seealso
#' \code{\link{spherical_sim_group}} for model fitting,
#' \code{\link{eval_link}} for evaluating the estimated link function,
#' \code{\link{inv_stereo}} for inverse stereographic projection
#'
#' @importFrom splines splineDesign
#' @export
predict_spherical <- function(fit, X_new) {
  z_new <- as.vector(X_new %*% fit$beta)
  z_clip <- pmax(pmin(z_new, max(fit$knots) - 1e-6), min(fit$knots) + 1e-6)
  B_new <- splineDesign(fit$knots, z_clip, ord = fit$degree + 1)
  U_new <- B_new %*% fit$Theta
  inv_stereo(U_new)
}


#' Evaluate Estimated Link Function
#'
#' Evaluates the estimated link function m̂(z) from a fitted spherical single-index
#' model on a specified grid or default sequence of single index values.
#'
#' @param fit Fitted model object returned by \code{\link{spherical_sim_group}}
#' @param z_grid Optional vector of single index values at which to evaluate the
#'   link function. If NULL, a uniform grid of n_grid points is created spanning
#'   the knot range (default NULL)
#' @param n_grid Number of grid points if z_grid is not provided (default 100)
#'
#' @details
#' The link function m: R → S^{q-1} maps single index values to the unit sphere.
#' It is estimated using B-spline approximation in R^{q-1} followed by inverse
#' stereographic projection.
#'
#' \strong{Evaluation Process:}
#' \enumerate{
#'   \item Create or validate z_grid (clipped to valid knot range)
#'   \item Evaluate B-spline basis: B(z_grid)
#'   \item Compute link in R^{q-1}: m̂(z) = B(z) Θ̂
#'   \item Map to sphere: ŷ(z) = π^{-1}(m̂(z))
#' }
#'
#' \strong{Return Values:}
#' The function returns both:
#' \itemize{
#'   \item The link function in R^{q-1} space (m)
#'   \item The link function on the sphere S^{q-1} (y_sphere)
#' }
#' The R^{q-1} representation is useful for visualization and understanding the
#' link structure before projection, while the spherical representation shows
#' the actual predicted curve on the sphere.
#'
#' \strong{Valid Evaluation Range:}
#' The link function should only be evaluated within the range of observed single
#' index values from training. The function automatically clips z_grid to the
#' valid knot range [min(knots) + ε, max(knots) - ε].
#'
#' @return List with three components:
#' \describe{
#'   \item{z}{Vector of single index values where link is evaluated (length n_grid or length(z_grid))}
#'   \item{m}{Matrix of link function values in R^{q-1} (n_grid × (q-1))}
#'   \item{y_sphere}{Matrix of link function values on unit sphere S^{q-1} (n_grid × q).
#'         Each row satisfies ||y_sphere[i,]|| = 1}
#' }
#'
#' @examples
#' \dontrun{
#' # Fit model to generated data
#' set.seed(123)
#' data <- generate_spherical_data(n = 200, p = 20, seed = 123)
#'
#' fit <- spherical_sim_group(
#'   X = data$X,
#'   Y = data$Y,
#'   group_idx = data$group_idx,
#'   lambda = 0.1,
#'   gamma = 0.1,
#'   n_knots = 5,
#'   verbose = FALSE
#' )
#'
#' # Evaluate link on default grid
#' link_eval <- eval_link(fit)
#'
#' # Verify points are on unit sphere
#' all(abs(rowSums(link_eval$y_sphere^2) - 1) < 1e-10)  # TRUE
#'
#' # Plot estimated link function (first component)
#' plot(link_eval$z, link_eval$m[,1], type = "l", lwd = 2,
#'      xlab = "Single Index z", ylab = "m̂₁(z)",
#'      main = "Estimated Link Function (Component 1)")
#'
#' # Add true link if known
#' z_true <- link_eval$z
#' m_true <- cbind(sin(z_true), 0.5 * cos(z_true))
#' lines(z_true, m_true[,1], col = "blue", lty = 2, lwd = 2)
#' legend("topright", c("Estimated", "True"),
#'        col = c("black", "blue"), lty = 1:2, lwd = 2)
#'
#' # Plot both components
#' par(mfrow = c(1, 2))
#' plot(link_eval$z, link_eval$m[,1], type = "l", lwd = 2,
#'      xlab = "z", ylab = "m̂₁(z)", main = "Component 1")
#' lines(z_true, m_true[,1], col = "blue", lty = 2, lwd = 2)
#'
#' plot(link_eval$z, link_eval$m[,2], type = "l", lwd = 2,
#'      xlab = "z", ylab = "m̂₂(z)", main = "Component 2")
#' lines(z_true, m_true[,2], col = "blue", lty = 2, lwd = 2)
#'
#' # Visualize link as curve on sphere (3D)
#' # First two dimensions
#' plot(link_eval$y_sphere[,1], link_eval$y_sphere[,2],
#'      type = "l", lwd = 2, asp = 1,
#'      xlab = "y₁", ylab = "y₂",
#'      main = "Link Function on Sphere (2D projection)")
#'
#' # Add unit circle for reference
#' theta <- seq(0, 2*pi, length.out = 100)
#' lines(cos(theta), sin(theta), col = "gray", lty = 2)
#'
#' # Color by z value
#' cols <- rainbow(nrow(link_eval$y_sphere))
#' plot(link_eval$y_sphere[,1], link_eval$y_sphere[,2],
#'      col = cols, pch = 19, cex = 0.5, asp = 1,
#'      xlab = "y₁", ylab = "y₂",
#'      main = "Link Colored by Single Index")
#'
#' # Evaluate at specific points
#' z_specific <- seq(-2, 2, by = 0.5)
#' link_specific <- eval_link(fit, z_grid = z_specific)
#'
#' print(data.frame(
#'   z = link_specific$z,
#'   m1 = link_specific$m[,1],
#'   m2 = link_specific$m[,2]
#' ))
#'
#' # Compare with training data
#' plot(link_eval$z, link_eval$m[,1], type = "l", lwd = 2,
#'      xlab = "Single Index", ylab = "Response Component 1",
#'      main = "Link Function vs Training Data")
#' points(fit$index, data$Y[,1], pch = 19, cex = 0.5, col = "red")
#' legend("topright", c("Estimated Link", "Training Data"),
#'        col = c("black", "red"), lty = c(1, NA), pch = c(NA, 19))
#'
#' # Compute smoothness of estimated link
#' m_diff <- diff(link_eval$m[,1])
#' z_diff <- diff(link_eval$z)
#' derivatives <- m_diff / z_diff
#'
#' plot(link_eval$z[-1], derivatives, type = "l", lwd = 2,
#'      xlab = "z", ylab = "dm̂₁/dz",
#'      main = "Estimated Link Derivative")
#' }
#'
#' @seealso
#' \code{\link{spherical_sim_group}} for model fitting,
#' \code{\link{predict_spherical}} for predictions on new data,
#' \code{\link{inv_stereo}} for inverse stereographic projection
#'
#' @importFrom splines splineDesign
#' @export
eval_link <- function(fit, z_grid = NULL, n_grid = 100) {
  if (is.null(z_grid)) {
    # Create default grid within valid knot range
    z_min <- min(fit$knots) + 1e-6
    z_max <- max(fit$knots) - 1e-6
    z_grid <- seq(z_min, z_max, length.out = n_grid)
  } else {
    # Clip provided grid to valid range
    z_min <- min(fit$knots) + 1e-6
    z_max <- max(fit$knots) - 1e-6
    z_grid <- pmax(pmin(z_grid, z_max), z_min)
  }

  # Evaluate B-spline basis at grid points
  B_grid <- splineDesign(fit$knots, z_grid, ord = fit$degree + 1)

  # Compute link function in R^{q-1}
  m_grid <- B_grid %*% fit$Theta

  list(
    z = z_grid,
    m = m_grid,
    y_sphere = inv_stereo(m_grid)
  )
}

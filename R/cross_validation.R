#' Cross-Validation for Lambda with Early Stopping
#'
#' Performs k-fold cross-validation over a sequence of lambda values with early
#' stopping to efficiently select the optimal regularization parameter for group
#' lasso in spherical single-index models. Includes constraints on group selection
#' and robust error handling.
#'
#' @param X Design matrix (n × p)
#' @param Y Response matrix on unit sphere (n × q)
#' @param group_idx List of length G, where each element contains indices of predictors in that group
#' @param lambda_seq Vector of lambda values to evaluate (typically in decreasing order)
#' @param gamma Roughness penalty parameter for link function smoothness
#' @param weights Vector of group-specific weights (length G). If NULL, uses sqrt(group_size) (default NULL)
#' @param n_folds Number of cross-validation folds (default 5)
#' @param n_knots Number of internal B-spline knots for link function approximation (default 10)
#' @param patience Number of lambda values without improvement before early stopping (default 3)
#' @param min_groups Minimum number of groups that must be selected (default 1)
#' @param max_groups Maximum number of groups allowed to be selected. If NULL, uses G (default NULL)
#' @param min_valid_folds Minimum number of folds with valid fits required for a lambda to be considered (default 2)
#' @param verbose Print detailed progress information (default FALSE)
#' @param ... Additional arguments passed to \code{\link{spherical_sim_group}}
#'
#' @details
#' This function efficiently searches for the optimal lambda by evaluating a sequence
#' of lambda values and stopping early when no improvement is observed.
#'
#' \strong{Cross-Validation Procedure:}
#' For each lambda value:
#' \enumerate{
#'   \item Split data into n_folds folds
#'   \item For each fold k:
#'     \itemize{
#'       \item Fit model on training folds (all except k)
#'       \item Predict on test fold k
#'       \item Compute prediction error if number of selected groups is in [min_groups, max_groups]
#'     }
#'   \item Average prediction errors across valid folds
#'   \item Track best lambda and check for early stopping
#' }
#'
#' \strong{Early Stopping Criteria:}
#' The search stops when any of these conditions is met:
#' \itemize{
#'   \item No improvement in CV error for 'patience' consecutive lambda values
#'   \item Too many consecutive failed fold evaluations
#'   \item Average number of selected groups exceeds max_groups
#'   \item All lambda values have been evaluated
#' }
#'
#' \strong{Robust Error Handling:}
#' \itemize{
#'   \item If a fold fails to fit, that fold contributes Inf to CV error
#'   \item If a fold selects too many/few groups, it contributes Inf
#'   \item Lambda values with fewer than min_valid_folds successful folds get CV error = Inf
#'   \item Gracefully handles numerical issues in fitting and prediction
#' }
#'
#' \strong{Lambda Sequence Strategy:}
#' For best results, provide lambda_seq in decreasing order (large to small):
#' \itemize{
#'   \item Large lambda: sparse models (few groups), fast fitting
#'   \item Small lambda: dense models (many groups), slower fitting
#'   \item Early stopping saves time when CV error stops improving
#' }
#'
#' \strong{Group Selection Constraints:}
#' The min_groups and max_groups parameters allow you to:
#' \itemize{
#'   \item Avoid degenerate solutions (min_groups = 1)
#'   \item Control model complexity (max_groups < G)
#'   \item Focus search on interpretable models (max_groups = 3 or 5)
#' }
#'
#' @return List with the following components:
#' \describe{
#'   \item{lambda}{Vector of evaluated lambda values (may be shorter than lambda_seq due to early stopping)}
#'   \item{cv_error}{Vector of CV errors for each evaluated lambda (Inf for invalid)}
#'   \item{n_selected}{Vector of average number of groups selected across folds}
#'   \item{n_valid_folds}{Vector indicating number of valid folds for each lambda}
#'   \item{best_lambda}{Lambda value with minimum CV error}
#'   \item{best_error}{Minimum CV error achieved}
#'   \item{best_idx}{Index of best_lambda in the lambda vector}
#'   \item{n_evaluated}{Total number of lambda values evaluated}
#'   \item{n_valid}{Number of lambda values with finite CV error}
#' }
#'
#' @examples
#' \dontrun{
#' # Generate data
#' set.seed(123)
#' data <- generate_spherical_data(n = 200, p = 20, G = 4, seed = 123)
#'
#' # Create lambda sequence (decreasing)
#' lambda_max <- 1.0
#' lambda_seq <- exp(seq(log(lambda_max), log(lambda_max * 0.01), length.out = 20))
#'
#' # Basic cross-validation
#' cv_result <- cv_lambda_path_early_stop(
#'   X = data$X,
#'   Y = data$Y,
#'   group_idx = data$group_idx,
#'   lambda_seq = lambda_seq,
#'   gamma = 0.1,
#'   n_folds = 5,
#'   verbose = TRUE
#' )
#'
#' # Examine results
#' cat("Best lambda:", cv_result$best_lambda, "\n")
#' cat("Best CV error:", cv_result$best_error, "\n")
#' cat("Evaluated:", cv_result$n_evaluated, "out of", length(lambda_seq), "lambdas\n")
#'
#' # Plot CV curve
#' finite_idx <- is.finite(cv_result$cv_error)
#' plot(cv_result$lambda[finite_idx], cv_result$cv_error[finite_idx],
#'      type = "b", log = "x", pch = 19,
#'      xlab = "Lambda", ylab = "CV Error",
#'      main = "Cross-Validation Curve")
#' abline(v = cv_result$best_lambda, col = "red", lty = 2)
#'
#' # Plot number of selected groups
#' plot(cv_result$lambda[finite_idx], cv_result$n_selected[finite_idx],
#'      type = "b", log = "x", pch = 19,
#'      xlab = "Lambda", ylab = "Number of Selected Groups",
#'      main = "Model Sparsity vs Lambda")
#'
#' # CV with group constraints
#' cv_constrained <- cv_lambda_path_early_stop(
#'   X = data$X,
#'   Y = data$Y,
#'   group_idx = data$group_idx,
#'   lambda_seq = lambda_seq,
#'   gamma = 0.1,
#'   min_groups = 2,
#'   max_groups = 3,  # Force selection of 2-3 groups
#'   n_folds = 5,
#'   verbose = TRUE
#' )
#'
#' # CV with adaptive weights
#' # First compute adaptive weights
#' initial_fit <- spherical_sim_group(
#'   X = data$X, Y = data$Y,
#'   group_idx = data$group_idx,
#'   lambda = 0.05, gamma = 0.1,
#'   verbose = FALSE
#' )
#' adaptive_weights <- sapply(data$group_idx, function(idx) {
#'   norm <- sqrt(sum(initial_fit$beta[idx]^2))
#'   sqrt(length(idx)) / max(norm, 1e-3)
#' })
#'
#' cv_adaptive <- cv_lambda_path_early_stop(
#'   X = data$X,
#'   Y = data$Y,
#'   group_idx = data$group_idx,
#'   lambda_seq = lambda_seq,
#'   gamma = 0.1,
#'   weights = adaptive_weights,
#'   n_folds = 5,
#'   verbose = TRUE
#' )
#'
#' # Compare standard vs adaptive
#' par(mfrow = c(1, 2))
#' plot(cv_result$lambda, cv_result$cv_error, type = "b", log = "x",
#'      main = "Standard Weights", xlab = "Lambda", ylab = "CV Error")
#' plot(cv_adaptive$lambda, cv_adaptive$cv_error, type = "b", log = "x",
#'      main = "Adaptive Weights", xlab = "Lambda", ylab = "CV Error")
#'
#' # Check early stopping effectiveness
#' cat("Standard evaluated:", cv_result$n_evaluated, "/", length(lambda_seq), "\n")
#' cat("Adaptive evaluated:", cv_adaptive$n_evaluated, "/", length(lambda_seq), "\n")
#'
#' # Examine fold stability
#' fold_success_rate <- cv_result$n_valid_folds / 5
#' plot(cv_result$lambda, fold_success_rate, type = "b", log = "x",
#'      xlab = "Lambda", ylab = "Fraction of Valid Folds",
#'      main = "Fold Success Rate",
#'      ylim = c(0, 1))
#' abline(h = 0.4, col = "red", lty = 2)  # min_valid_folds/n_folds threshold
#'
#' # Fit final model with best lambda
#' final_fit <- spherical_sim_group(
#'   X = data$X,
#'   Y = data$Y,
#'   group_idx = data$group_idx,
#'   lambda = cv_result$best_lambda,
#'   gamma = 0.1,
#'   n_knots = 10,
#'   verbose = TRUE
#' )
#'
#' cat("Selected groups:", final_fit$selected_groups, "\n")
#' cat("True active groups:", data$active_groups, "\n")
#' }
#'
#' @references
#' - Hastie, T., Tibshirani, R., & Friedman, J. (2009). The Elements of
#'   Statistical Learning (2nd ed.). Springer.
#' - Bühlmann, P., & van de Geer, S. (2011). Statistics for High-Dimensional
#'   Data: Methods, Theory and Applications. Springer.
#'
#' @seealso
#' \code{\link{cv_two_stage_adaptive}} for two-stage CV with gamma search,
#' \code{\link{spherical_sim_group}} for model fitting,
#' \code{\link{predict_spherical}} for predictions,
#' \code{\link{compute_lambda_max_with_weights}} for computing lambda_max
#'
#' @export
cv_lambda_path_early_stop <- function(X, Y, group_idx,
                                      lambda_seq,
                                      gamma,
                                      weights = NULL,  # ADDED
                                      n_folds = 5,
                                      n_knots = 10,
                                      patience = 3,
                                      min_groups = 1,
                                      max_groups = NULL,
                                      min_valid_folds = 2,
                                      verbose = FALSE,
                                      ...) {

  n <- nrow(X)
  G <- length(group_idx)
  folds <- sample(rep(1:n_folds, length.out = n))

  if (is.null(max_groups)) max_groups <- G
  if (is.null(weights)) {
    weights <- sapply(group_idx, function(idx) sqrt(length(idx)))
  }

  nlambda <- length(lambda_seq)
  cv_errors <- rep(NA, nlambda)
  n_selected_vec <- rep(NA, nlambda)
  n_valid_folds <- rep(NA, nlambda)

  best_cv_error <- Inf
  best_idx <- 1
  no_improvement_count <- 0

  for (i in 1:nlambda) {
    lam <- lambda_seq[i]

    fold_errors <- rep(NA, n_folds)
    fold_n_selected <- rep(NA, n_folds)

    for (k in 1:n_folds) {
      train_idx <- which(folds != k)
      test_idx <- which(folds == k)

      fit_k <- tryCatch({
        spherical_sim_group(
          X[train_idx, , drop = FALSE],
          Y[train_idx, , drop = FALSE],
          group_idx, lam, gamma,
          weights = weights,  # Pass weights
          n_knots = n_knots,
          max_iter = 100,
          verbose = FALSE,
          ...
        )
      }, error = function(e) NULL)

      if (!is.null(fit_k)) {
        fold_n_selected[k] <- length(fit_k$selected_groups)

        if (fold_n_selected[k] >= min_groups && fold_n_selected[k] <= max_groups) {
          Y_pred <- tryCatch({
            predict_spherical(fit_k, X[test_idx, , drop = FALSE])
          }, error = function(e) NULL)

          if (!is.null(Y_pred)) {
            error <- mean(rowSums((Y[test_idx, ] - Y_pred)^2))
            if (is.finite(error)) {
              fold_errors[k] <- error
            } else {
              fold_errors[k] <- Inf
            }
          } else {
            fold_errors[k] <- Inf
          }
        } else {
          fold_errors[k] <- Inf
        }
      } else {
        fold_errors[k] <- Inf
      }
    }

    finite_errors <- fold_errors[is.finite(fold_errors)]
    finite_n_selected <- fold_n_selected[is.finite(fold_n_selected)]
    n_valid_folds[i] <- length(finite_errors)

    if (n_valid_folds[i] >= min_valid_folds) {
      cv_errors[i] <- mean(finite_errors)
      n_selected_vec[i] <- mean(finite_n_selected)
    } else {
      cv_errors[i] <- Inf
      n_selected_vec[i] <- NA
    }

    if (verbose && is.finite(cv_errors[i])) {
      cat(sprintf("[%d/%d] λ=%.5f: CV=%.4f, n_sel=%.1f (valid: %d/%d)\n",
                  i, nlambda, lam, cv_errors[i], n_selected_vec[i],
                  n_valid_folds[i], n_folds))
    }

    # Early stopping
    if (is.finite(cv_errors[i])) {
      if (cv_errors[i] < best_cv_error) {
        best_cv_error <- cv_errors[i]
        best_idx <- i
        no_improvement_count <- 0
      } else {
        no_improvement_count <- no_improvement_count + 1
      }

      if (no_improvement_count >= patience) {
        if (verbose) cat("Early stop: No improvement\n")
        break
      }
    } else {
      no_improvement_count <- no_improvement_count + 1
      if (no_improvement_count >= patience) {
        if (verbose) cat("Early stop: Failures\n")
        break
      }
    }

    if (is.finite(n_selected_vec[i]) && n_selected_vec[i] >= max_groups - 0.5) {
      if (verbose) cat("Early stop: Too many groups\n")
      break
    }
  }

  cv_errors <- cv_errors[1:i]
  n_selected_vec <- n_selected_vec[1:i]
  lambda_seq <- lambda_seq[1:i]
  n_valid_folds <- n_valid_folds[1:i]

  finite_idx <- which(is.finite(cv_errors))
  if (length(finite_idx) == 0) {
    best_idx <- 1
    warning("No valid CV evaluations")
  } else {
    best_idx <- finite_idx[which.min(cv_errors[finite_idx])]
  }

  list(
    lambda = lambda_seq,
    cv_error = cv_errors,
    n_selected = n_selected_vec,
    n_valid_folds = n_valid_folds,
    best_lambda = lambda_seq[best_idx],
    best_error = cv_errors[best_idx],
    best_idx = best_idx,
    n_evaluated = i,
    n_valid = length(finite_idx)
  )
}


#' Two-Stage Cross-Validation with Adaptive Weights
#'
#' Performs efficient two-stage cross-validation to jointly select optimal lambda
#' (group lasso penalty) and gamma (link function smoothness) parameters. Optionally
#' computes adaptive weights to improve variable selection performance. Features
#' early stopping at both stages for computational efficiency.
#'
#' @param X Design matrix (n × p)
#' @param Y Response matrix on unit sphere (n × q)
#' @param group_idx List of length G, where each element contains indices of predictors in that group
#' @param use_adaptive Use adaptive group lasso weights (default TRUE)
#' @param adaptive_method Method for computing adaptive weights: "grouplasso" or "fast" (default "grouplasso")
#' @param adaptive_gamma_power Power for adaptive weight computation: w_g = ||β_init_g||^{-γ} (default 1)
#' @param initial_lambda_fraction Fraction of lambda_max for initial group lasso fit when computing adaptive weights (default 0.3)
#' @param nlambda Number of lambda values in search grid (default 12)
#' @param ngamma_coarse Number of gamma values in coarse stage (default 5)
#' @param ngamma_fine Number of gamma values in fine stage (default 8)
#' @param n_folds Number of cross-validation folds (default 3)
#' @param n_knots Number of internal B-spline knots for link function approximation (default 5)
#' @param lambda_multiplier Multiplier for lambda_max computation (default 5.0)
#' @param lambda_min_ratio Ratio of minimum to maximum lambda (default 0.1)
#' @param lambda_patience Number of lambda values without improvement before early stopping (default 3)
#' @param gamma_patience Number of gamma values without improvement before early stopping (default 2)
#' @param min_groups Minimum number of groups that must be selected (default 1)
#' @param max_groups Maximum number of groups allowed. If NULL, uses G (default NULL)
#' @param min_valid_folds Minimum number of valid CV folds required (default 2)
#' @param verbose Print detailed progress messages (default TRUE)
#' @param ... Additional arguments passed to \code{\link{spherical_sim_group}}
#'
#' @details
#' This function implements a sophisticated two-stage cross-validation strategy to
#' efficiently search the (lambda, gamma) parameter space while optionally using
#' adaptive weights to improve variable selection.
#'
#' \strong{Overall Procedure:}
#' \enumerate{
#'   \item \strong{Adaptive Weights} (if use_adaptive = TRUE):
#'     \itemize{
#'       \item Fit initial model to compute group-wise coefficient norms
#'       \item Compute weights: w_g = ||β_init_g||^{-γ} * sqrt(|G_g|)
#'       \item Groups with smaller initial estimates get larger penalties
#'     }
#'   \item \strong{Parameter Range Computation}:
#'     \itemize{
#'       \item lambda_max: largest lambda that keeps at least one group
#'       \item gamma_max: computed from data characteristics
#'       \item Create logarithmic grids for both parameters
#'     }
#'   \item \strong{Stage 1 - Coarse Gamma Search}:
#'     \itemize{
#'       \item Evaluate ngamma_coarse gamma values on coarse grid
#'       \item For each gamma, find optimal lambda via CV with early stopping
#'       \item Track best (lambda, gamma) combination
#'       \item Stop early if no improvement for gamma_patience iterations
#'     }
#'   \item \strong{Stage 2 - Fine Gamma Search}:
#'     \itemize{
#'       \item Create fine grid around best gamma from Stage 1 (±3x range)
#'       \item Re-evaluate with ngamma_fine values
#'       \item Find optimal (lambda, gamma) on refined grid
#'       \item Early stopping with gamma_patience
#'     }
#' }
#'
#' \strong{Adaptive Weight Methods:}
#' \describe{
#'   \item{grouplasso}{Fits initial group lasso at moderate lambda to get coefficient estimates.
#'         More accurate but slower. Recommended for most applications.}
#'   \item{fast}{Uses correlation-based screening to quickly estimate group importance.
#'         Much faster but potentially less accurate. Good for very large p.}
#' }
#'
#' \strong{Early Stopping Strategy:}
#' Early stopping occurs at two levels:
#' \itemize{
#'   \item \emph{Lambda level}: Within each gamma, stop lambda search after lambda_patience
#'         values without CV improvement (via \code{\link{cv_lambda_path_early_stop}})
#'   \item \emph{Gamma level}: Stop gamma search after gamma_patience gamma values
#'         without CV improvement
#' }
#'
#' \strong{Computational Efficiency:}
#' The two-stage approach is much faster than full grid search:
#' \itemize{
#'   \item Full grid: O(nlambda × (ngamma_coarse + ngamma_fine)) evaluations
#'   \item Two-stage with early stopping: Often < 50% of full grid
#'   \item Lambda sequence evaluated in decreasing order (sparse to dense)
#'   \item Early stopping saves time on unpromising parameter regions
#' }
#'
#' \strong{Choosing Parameters:}
#' \itemize{
#'   \item \emph{n_folds}: 3-5 is typical. Larger values give more stable estimates but are slower.
#'   \item \emph{ngamma_coarse/fine}: 5-8 is usually sufficient. Gamma is often less sensitive than lambda.
#'   \item \emph{nlambda}: 10-15 provides good coverage. More may be needed for difficult problems.
#'   \item \emph{patience parameters}: 2-3 works well. Increase if search seems to stop too early.
#'   \item \emph{adaptive_gamma_power}: 1 is standard. Values > 1 give stronger adaptive effect.
#' }
#'
#' @return List with the following components:
#' \describe{
#'   \item{best_lambda}{Optimal lambda from two-stage CV}
#'   \item{best_gamma}{Optimal gamma from two-stage CV}
#'   \item{best_cv_error}{Minimum CV error achieved}
#'   \item{weights}{Vector of group weights (length G) used in final model}
#'   \item{adaptive_info}{If use_adaptive = TRUE, contains information from adaptive weight computation
#'         including initial fit and group norms. NULL if use_adaptive = FALSE}
#'   \item{stage1}{List of CV results from coarse gamma stage. Each element contains output
#'         from \code{\link{cv_lambda_path_early_stop}}}
#'   \item{stage2}{List of CV results from fine gamma stage}
#' }
#'
#' @examples
#' \dontrun{
#' # Generate data
#' set.seed(123)
#' data <- generate_spherical_data(n = 200, p = 40, G = 8,
#'                                 active_groups = c(1, 3, 5),
#'                                 seed = 123)
#'
#' # Basic adaptive group lasso CV
#' cv_result <- cv_two_stage_adaptive(
#'   X = data$X,
#'   Y = data$Y,
#'   group_idx = data$group_idx,
#'   use_adaptive = TRUE,
#'   adaptive_method = "grouplasso",
#'   verbose = TRUE
#' )
#'
#' # Examine results
#' cat("Best lambda:", cv_result$best_lambda, "\n")
#' cat("Best gamma:", cv_result$best_gamma, "\n")
#' cat("Best CV error:", cv_result$best_cv_error, "\n")
#'
#' # Fit final model with selected parameters
#' final_fit <- spherical_sim_group(
#'   X = data$X,
#'   Y = data$Y,
#'   group_idx = data$group_idx,
#'   lambda = cv_result$best_lambda,
#'   gamma = cv_result$best_gamma,
#'   weights = cv_result$weights,
#'   n_knots = 5,
#'   verbose = TRUE
#' )
#'
#' cat("Selected groups:", final_fit$selected_groups, "\n")
#' cat("True active groups:", data$active_groups, "\n")
#'
#' # Compare adaptive vs standard weights
#' cv_standard <- cv_two_stage_adaptive(
#'   X = data$X,
#'   Y = data$Y,
#'   group_idx = data$group_idx,
#'   use_adaptive = FALSE,
#'   verbose = FALSE
#' )
#'
#' cat("\nAdaptive CV error:", cv_result$best_cv_error, "\n")
#' cat("Standard CV error:", cv_standard$best_cv_error, "\n")
#'
#' # Visualize Stage 1 results
#' stage1_errors <- sapply(cv_result$stage1, function(x) x$best_error)
#' stage1_gammas <- sapply(cv_result$stage1, function(x) {
#'   # Extract gamma from the CV call (stored in parent environment)
#'   cv_result$stage1[[1]]  # Would need to store gamma explicitly
#' })
#'
#' # Alternative: visualize lambda paths for selected gammas
#' par(mfrow = c(1, 2))
#'
#' # Stage 1: First gamma
#' res1 <- cv_result$stage1[[1]]
#' finite_idx <- is.finite(res1$cv_error)
#' plot(res1$lambda[finite_idx], res1$cv_error[finite_idx],
#'      type = "b", log = "x", pch = 19,
#'      xlab = "Lambda", ylab = "CV Error",
#'      main = "Stage 1: First Gamma")
#' abline(v = res1$best_lambda, col = "red", lty = 2)
#'
#' # Stage 2: Best gamma
#' best_stage2_idx <- which.min(sapply(cv_result$stage2,
#'                                     function(x) x$best_error))
#' res2 <- cv_result$stage2[[best_stage2_idx]]
#' finite_idx <- is.finite(res2$cv_error)
#' plot(res2$lambda[finite_idx], res2$cv_error[finite_idx],
#'      type = "b", log = "x", pch = 19,
#'      xlab = "Lambda", ylab = "CV Error",
#'      main = "Stage 2: Best Gamma")
#' abline(v = res2$best_lambda, col = "red", lty = 2)
#'
#' # Fast adaptive method for large problems
#' cv_fast <- cv_two_stage_adaptive(
#'   X = data$X,
#'   Y = data$Y,
#'   group_idx = data$group_idx,
#'   use_adaptive = TRUE,
#'   adaptive_method = "fast",  # Much faster
#'   verbose = TRUE
#' )
#'
#' # CV with constraints on model complexity
#' cv_constrained <- cv_two_stage_adaptive(
#'   X = data$X,
#'   Y = data$Y,
#'   group_idx = data$group_idx,
#'   min_groups = 2,
#'   max_groups = 4,  # Force selection of 2-4 groups
#'   verbose = TRUE
#' )
#'
#' # Fine-tuned search with more grid points
#' cv_fine <- cv_two_stage_adaptive(
#'   X = data$X,
#'   Y = data$Y,
#'   group_idx = data$group_idx,
#'   nlambda = 20,           # More lambda values
#'   ngamma_coarse = 8,      # More coarse gamma
#'   ngamma_fine = 12,       # More fine gamma
#'   n_folds = 5,            # More folds
#'   lambda_patience = 4,    # More patient
#'   gamma_patience = 3,
#'   verbose = TRUE
#' )
#'
#' # Examine adaptive weights
#' if (!is.null(cv_result$adaptive_info)) {
#'   cat("\nAdaptive Weights:\n")
#'   print(data.frame(
#'     Group = 1:length(cv_result$weights),
#'     Weight = cv_result$weights,
#'     InitialNorm = cv_result$adaptive_info$group_norms
#'   ))
#'
#'   # Plot weights
#'   barplot(cv_result$weights, names.arg = 1:length(cv_result$weights),
#'           xlab = "Group", ylab = "Adaptive Weight",
#'           main = "Adaptive Weights by Group",
#'           col = ifelse(1:length(cv_result$weights) %in% data$active_groups,
#'                       "lightblue", "lightgray"))
#'   legend("topright", c("Active", "Inactive"),
#'          fill = c("lightblue", "lightgray"))
#' }
#'
#' # Time comparison: adaptive methods
#' system.time({
#'   cv_grouplasso <- cv_two_stage_adaptive(
#'     X = data$X, Y = data$Y, group_idx = data$group_idx,
#'     adaptive_method = "grouplasso", verbose = FALSE
#'   )
#' })
#'
#' system.time({
#'   cv_fast <- cv_two_stage_adaptive(
#'     X = data$X, Y = data$Y, group_idx = data$group_idx,
#'     adaptive_method = "fast", verbose = FALSE
#'   )
#' })
#'
#' # Compare selection performance
#' fit_grouplasso <- spherical_sim_group(
#'   data$X, data$Y, data$group_idx,
#'   cv_grouplasso$best_lambda, cv_grouplasso$best_gamma,
#'   weights = cv_grouplasso$weights, verbose = FALSE
#' )
#'
#' fit_fast <- spherical_sim_group(
#'   data$X, data$Y, data$group_idx,
#'   cv_fast$best_lambda, cv_fast$best_gamma,
#'   weights = cv_fast$weights, verbose = FALSE
#' )
#'
#' cat("\nGroup Lasso method selected:", fit_grouplasso$selected_groups, "\n")
#' cat("Fast method selected:", fit_fast$selected_groups, "\n")
#' cat("True active groups:", data$active_groups, "\n")
#' }
#'
#' @references
#' - Zou, H. (2006). The adaptive lasso and its oracle properties. Journal of
#'   the American Statistical Association, 101(476), 1418-1429.
#' - Yuan, M., & Lin, Y. (2006). Model selection and estimation in regression
#'   with grouped variables. Journal of the Royal Statistical Society: Series B,
#'   68(1), 49-67.
#' - Friedman, J., Hastie, T., & Tibshirani, R. (2010). Regularization paths
#'   for generalized linear models via coordinate descent. Journal of Statistical
#'   Software, 33(1), 1-22.
#'
#' @seealso
#' \code{\link{cv_lambda_path_early_stop}} for single gamma CV,
#' \code{\link{compute_adaptive_weights_grouplasso}} for group lasso adaptive weights,
#' \code{\link{compute_adaptive_weights_fast}} for fast adaptive weights,
#' \code{\link{spherical_sim_group}} for model fitting,
#' \code{\link{run_single_simulation}} for using CV in simulations
#'
#' @export
cv_two_stage_adaptive <- function(X, Y, group_idx,
                                  use_adaptive = TRUE,
                                  adaptive_method = "grouplasso",  # or "fast"
                                  adaptive_gamma_power = 1,
                                  initial_lambda_fraction = 0.3,
                                  nlambda = 12,
                                  ngamma_coarse = 5,
                                  ngamma_fine = 8,
                                  n_folds = 3,
                                  n_knots = 5,
                                  lambda_multiplier = 5.0,
                                  lambda_min_ratio = 0.1,
                                  lambda_patience = 3,
                                  gamma_patience = 2,
                                  min_groups = 1,
                                  max_groups = NULL,
                                  min_valid_folds = 2,
                                  verbose = TRUE,
                                  ...) {

  if (is.null(max_groups)) max_groups <- length(group_idx)

  if (verbose) cat("=== ADAPTIVE GROUP LASSO ===\n")

  # Compute adaptive weights
  if (use_adaptive) {

    if (adaptive_method == "grouplasso") {
      adaptive_res <- compute_adaptive_weights_grouplasso(
        X, Y, group_idx,
        gamma_power = adaptive_gamma_power,
        initial_lambda_fraction = initial_lambda_fraction,
        n_knots = n_knots,
        verbose = verbose
      )

    } else if (adaptive_method == "fast") {
      adaptive_res <- compute_adaptive_weights_fast(
        X, Y, group_idx,
        gamma_power = adaptive_gamma_power,
        verbose = verbose
      )

    } else {
      stop("Unknown adaptive_method: ", adaptive_method)
    }

    weights <- adaptive_res$weights

  } else {
    if (verbose) cat("Using standard weights\n")
    weights <- sapply(group_idx, function(idx) sqrt(length(idx)))
    adaptive_res <- NULL
  }

  # Compute parameter ranges
  if (verbose) cat("\n=== Computing parameter ranges ===\n")

  lambda_max <- compute_lambda_max_with_weights(
    X, Y, group_idx, weights, n_knots, lambda_multiplier
  )

  gamma_max <- compute_gamma_max(X, Y, group_idx, n_knots)

  if (verbose) {
    cat(sprintf("Lambda_max = %.5f\n", lambda_max))
    cat(sprintf("Gamma_max = %.5f\n", gamma_max))
  }

  lambda_seq <- exp(seq(log(lambda_max), log(lambda_max * lambda_min_ratio),
                        length.out = nlambda))

  # Stage 1
  if (verbose) cat("\n=== Stage 1: Coarse gamma ===\n")

  gamma_coarse <- exp(seq(log(gamma_max), log(gamma_max * 0.001),
                          length.out = ngamma_coarse))

  best_gamma_stage1 <- NULL
  best_error_stage1 <- Inf
  no_improvement_gamma <- 0

  stage1_results <- list()

  for (i in 1:ngamma_coarse) {
    gam <- gamma_coarse[i]

    if (verbose) cat(sprintf("\n[%d/%d] Gamma = %.5f\n", i, ngamma_coarse, gam))

    cv_lam <- cv_lambda_path_early_stop(
      X, Y, group_idx,
      lambda_seq = lambda_seq,
      gamma = gam,
      weights = weights,
      n_folds = n_folds,
      n_knots = n_knots,
      patience = lambda_patience,
      min_groups = min_groups,
      max_groups = max_groups,
      min_valid_folds = min_valid_folds,
      verbose = verbose,
      ...
    )

    stage1_results[[i]] <- cv_lam

    if (verbose) {
      cat(sprintf("  Best: λ=%.5f, CV=%.4f\n",
                  cv_lam$best_lambda, cv_lam$best_error))
    }

    if (is.finite(cv_lam$best_error)) {
      if (cv_lam$best_error < best_error_stage1) {
        best_error_stage1 <- cv_lam$best_error
        best_gamma_stage1 <- gam
        no_improvement_gamma <- 0
      } else {
        no_improvement_gamma <- no_improvement_gamma + 1
      }
    } else {
      no_improvement_gamma <- no_improvement_gamma + 1
    }

    if (no_improvement_gamma >= gamma_patience) {
      if (verbose) cat("\nEarly stop gamma\n")
      break
    }
  }

  if (is.null(best_gamma_stage1)) {
    best_gamma_stage1 <- gamma_coarse[1]
  }

  # Stage 2
  if (verbose) cat("\n=== Stage 2: Fine gamma ===\n")

  gamma_fine <- exp(seq(
    log(best_gamma_stage1 * 3),
    log(best_gamma_stage1 / 3),
    length.out = ngamma_fine
  ))

  best_gamma_final <- best_gamma_stage1
  best_lambda_final <- stage1_results[[1]]$best_lambda
  best_error_final <- best_error_stage1
  no_improvement_final <- 0

  stage2_results <- list()

  for (i in 1:ngamma_fine) {
    gam <- gamma_fine[i]

    if (verbose) cat(sprintf("\n[%d/%d] Gamma = %.5f\n", i, ngamma_fine, gam))

    cv_lam <- cv_lambda_path_early_stop(
      X, Y, group_idx,
      lambda_seq = lambda_seq,
      gamma = gam,
      weights = weights,
      n_folds = n_folds,
      n_knots = n_knots,
      patience = lambda_patience,
      min_groups = min_groups,
      max_groups = max_groups,
      min_valid_folds = min_valid_folds,
      verbose = verbose,
      ...
    )

    stage2_results[[i]] <- cv_lam

    if (verbose) {
      cat(sprintf("  Best: λ=%.5f, CV=%.4f\n",
                  cv_lam$best_lambda, cv_lam$best_error))
    }

    if (is.finite(cv_lam$best_error)) {
      if (cv_lam$best_error < best_error_final) {
        best_error_final <- cv_lam$best_error
        best_gamma_final <- gam
        best_lambda_final <- cv_lam$best_lambda
        no_improvement_final <- 0
      } else {
        no_improvement_final <- no_improvement_final + 1
      }
    } else {
      no_improvement_final <- no_improvement_final + 1
    }

    if (no_improvement_final >= gamma_patience) {
      if (verbose) cat("\nEarly stop\n")
      break
    }
  }

  if (verbose) {
    cat(sprintf("\n=== FINAL ===\n"))
    cat(sprintf("Lambda: %.5f\n", best_lambda_final))
    cat(sprintf("Gamma: %.5f\n", best_gamma_final))
    cat(sprintf("CV error: %.4f\n", best_error_final))
  }

  list(
    best_lambda = best_lambda_final,
    best_gamma = best_gamma_final,
    best_cv_error = best_error_final,
    weights = weights,
    adaptive_info = adaptive_res,
    stage1 = stage1_results,
    stage2 = stage2_results
  )
}

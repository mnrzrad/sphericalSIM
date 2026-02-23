#' Compute Adaptive Weights via Initial Group Lasso
#'
#' Computes adaptive weights for the adaptive group lasso by fitting an initial
#' group lasso model and using the inverse of group coefficient norms. This approach
#' follows the adaptive lasso methodology and tends to give better variable selection
#' performance than standard group lasso.
#'
#' @param X Design matrix (n × p)
#' @param Y Response matrix on unit sphere (n × q)
#' @param group_idx List of length G, where each element contains indices of predictors in that group
#' @param gamma_power Power for adaptive weight formula (default 1). Higher values give stronger adaptation
#' @param initial_lambda_fraction Fraction of lambda_max for initial group lasso fit (default 0.3).
#'   Should be large enough to get sparse solution but not so large that all groups are zeroed
#' @param initial_gamma Roughness penalty for initial fit (default 0.01). Usually set small
#' @param n_knots Number of internal B-spline knots for initial fit (default 5)
#' @param verbose Print progress information (default FALSE)
#'
#' @details
#' The adaptive group lasso uses data-dependent weights to improve variable selection
#' properties. This function implements the two-stage procedure:
#'
#' \strong{Adaptive Weight Computation:}
#' \enumerate{
#'   \item \strong{Compute lambda_max}: Determine the regularization strength that
#'         zeros out all groups using standard weights w_g^{(0)} = sqrt(|G_g|)
#'   \item \strong{Initial fit}: Fit group lasso with λ = initial_lambda_fraction × λ_max
#'         using standard weights to obtain β̂_init
#'   \item \strong{Compute group norms}: For each group g, calculate
#'         ||β̂_init_g||_2 = sqrt(Σ_{j∈G_g} β̂_init_j^2)
#'   \item \strong{Adaptive weights}: Compute
#'         \deqn{w_g^{(adapt)} = \frac{\sqrt{|G_g|}}{(||β̂_init_g||_2 + ε)^γ}}
#'         where ε = 10^{-4} prevents division by zero
#'   \item \strong{Normalize}: Scale weights so Σ_g w_g = G
#' }
#'
#' \strong{Mathematical Formulation:}
#' The adaptive weights satisfy:
#' \deqn{w_g^{(adapt)} = w_g^{(0)} × (||β̂_init_g||_2 + ε)^{-γ} × \text{normalizer}}
#' where:
#' \itemize{
#'   \item w_g^{(0)} = sqrt(|G_g|) is the standard weight
#'   \item γ is the gamma_power parameter (typically 1)
#'   \item ε prevents numerical issues when ||β̂_init_g|| ≈ 0
#'   \item Normalizer ensures Σ_g w_g = G
#' }
#'
#' \strong{Effect of Adaptive Weights:}
#' \itemize{
#'   \item \emph{Large ||β̂_init_g||}: Small weight w_g → less penalty → easier to select
#'   \item \emph{Small ||β̂_init_g||}: Large weight w_g → more penalty → harder to select
#'   \item Encourages consistent variable selection (oracle property)
#'   \item Reduces false positives compared to standard group lasso
#' }
#'
#' \strong{Choosing initial_lambda_fraction:}
#' \itemize{
#'   \item Too small (< 0.1): Initial fit too dense, poor discrimination
#'   \item Too large (> 0.5): Initial fit too sparse, may miss important groups
#'   \item Recommended: 0.2-0.4 works well in practice
#'   \item Default 0.3 is a good balance for most problems
#' }
#'
#' \strong{Choosing gamma_power:}
#' \itemize{
#'   \item γ = 1: Standard adaptive lasso (most common)
#'   \item γ > 1: Stronger adaptation, more aggressive penalty on weak groups
#'   \item γ < 1: Weaker adaptation, closer to standard group lasso
#'   \item Theory supports γ = 1 for oracle properties
#' }
#'
#' \strong{Robustness:}
#' If the initial fit fails (convergence issues, numerical problems), the function
#' falls back to standard weights and issues a warning. This ensures the procedure
#' always returns valid weights.
#'
#' @return List with the following components:
#' \describe{
#'   \item{weights}{Vector of adaptive weights (length G). These should be passed
#'         to \code{\link{spherical_sim_group}} or \code{\link{cv_lambda_path_early_stop}}}
#'   \item{beta_init}{Initial coefficient vector from group lasso fit (length p).
#'         NULL if initial fit failed}
#'   \item{group_norms_init}{Vector of group norms from initial fit (length G).
#'         Shows ||β̂_init_g|| for each group g}
#'   \item{n_selected_init}{Number of groups selected in initial fit (scalar)}
#'   \item{method}{Character string: "grouplasso" if successful, "standard" if fallback used}
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
#' # Compute adaptive weights with defaults
#' adaptive_res <- compute_adaptive_weights_grouplasso(
#'   X = data$X,
#'   Y = data$Y,
#'   group_idx = data$group_idx,
#'   verbose = TRUE
#' )
#'
#' # Examine weights
#' cat("Adaptive weights:\n")
#' print(data.frame(
#'   Group = 1:length(adaptive_res$weights),
#'   Weight = round(adaptive_res$weights, 4),
#'   InitNorm = round(adaptive_res$group_norms_init, 4),
#'   TrueActive = 1:length(adaptive_res$weights) %in% data$active_groups
#' ))
#'
#' # Visualize weights vs initial norms
#' par(mfrow = c(1, 2))
#'
#' # Weights by group
#' barplot(adaptive_res$weights, names.arg = 1:8,
#'         xlab = "Group", ylab = "Adaptive Weight",
#'         main = "Adaptive Weights",
#'         col = ifelse(1:8 %in% data$active_groups, "lightblue", "lightgray"))
#' legend("topright", c("Active", "Inactive"),
#'        fill = c("lightblue", "lightgray"))
#'
#' # Initial group norms
#' barplot(adaptive_res$group_norms_init, names.arg = 1:8,
#'         xlab = "Group", ylab = "||β_init||",
#'         main = "Initial Group Norms",
#'         col = ifelse(1:8 %in% data$active_groups, "lightblue", "lightgray"))
#'
#' # Compare standard vs adaptive weights
#' standard_weights <- sapply(data$group_idx, function(idx) sqrt(length(idx)))
#'
#' plot(standard_weights, adaptive_res$weights, pch = 19,
#'      xlab = "Standard Weight", ylab = "Adaptive Weight",
#'      main = "Standard vs Adaptive Weights")
#' abline(0, 1, col = "red", lty = 2)
#' text(standard_weights, adaptive_res$weights,
#'      labels = 1:8, pos = 3, cex = 0.8)
#'
#' # Use adaptive weights in CV
#' cv_result <- cv_two_stage_adaptive(
#'   X = data$X,
#'   Y = data$Y,
#'   group_idx = data$group_idx,
#'   use_adaptive = TRUE,
#'   adaptive_method = "grouplasso",
#'   verbose = FALSE
#' )
#'
#' # Fit model with adaptive weights
#' fit_adaptive <- spherical_sim_group(
#'   X = data$X,
#'   Y = data$Y,
#'   group_idx = data$group_idx,
#'   lambda = cv_result$best_lambda,
#'   gamma = cv_result$best_gamma,
#'   weights = adaptive_res$weights,
#'   verbose = FALSE
#' )
#'
#' # Compare with standard group lasso
#' fit_standard <- spherical_sim_group(
#'   X = data$X,
#'   Y = data$Y,
#'   group_idx = data$group_idx,
#'   lambda = cv_result$best_lambda,
#'   gamma = cv_result$best_gamma,
#'   weights = standard_weights,
#'   verbose = FALSE
#' )
#'
#' cat("\nStandard selected:", fit_standard$selected_groups, "\n")
#' cat("Adaptive selected:", fit_adaptive$selected_groups, "\n")
#' cat("True active:", data$active_groups, "\n")
#'
#' # Effect of gamma_power
#' gamma_powers <- c(0.5, 1, 1.5, 2)
#'
#' weights_list <- lapply(gamma_powers, function(gp) {
#'   res <- compute_adaptive_weights_grouplasso(
#'     X = data$X, Y = data$Y, group_idx = data$group_idx,
#'     gamma_power = gp, verbose = FALSE
#'   )
#'   res$weights
#' })
#'
#' # Plot weights for different gamma_power
#' matplot(1:8, do.call(cbind, weights_list),
#'         type = "b", pch = 19, lty = 1,
#'         xlab = "Group", ylab = "Weight",
#'         main = "Weights vs gamma_power")
#' legend("topright", paste("γ =", gamma_powers),
#'        col = 1:4, lty = 1, pch = 19)
#' abline(v = data$active_groups, col = "gray", lty = 2)
#'
#' # Effect of initial_lambda_fraction
#' lambda_fractions <- c(0.1, 0.2, 0.3, 0.5, 0.7)
#'
#' n_selected_list <- sapply(lambda_fractions, function(frac) {
#'   res <- compute_adaptive_weights_grouplasso(
#'     X = data$X, Y = data$Y, group_idx = data$group_idx,
#'     initial_lambda_fraction = frac, verbose = FALSE
#'   )
#'   res$n_selected_init
#' })
#'
#' plot(lambda_fractions, n_selected_list, type = "b", pch = 19,
#'      xlab = "initial_lambda_fraction",
#'      ylab = "Groups Selected in Initial Fit",
#'      main = "Initial Sparsity vs Lambda Fraction")
#' abline(h = length(data$active_groups), col = "red", lty = 2)
#'
#' # Correlation between initial norms and true signals
#' true_active_indicator <- as.numeric(1:8 %in% data$active_groups)
#' cor_norms_truth <- cor(adaptive_res$group_norms_init, true_active_indicator)
#' cat("\nCorrelation between initial norms and true active groups:",
#'     round(cor_norms_truth, 3), "\n")
#'
#' # Weight ratio: active vs inactive groups
#' active_weights <- adaptive_res$weights[data$active_groups]
#' inactive_weights <- adaptive_res$weights[setdiff(1:8, data$active_groups)]
#'
#' cat("Mean active group weight:", mean(active_weights), "\n")
#' cat("Mean inactive group weight:", mean(inactive_weights), "\n")
#' cat("Ratio:", mean(inactive_weights) / mean(active_weights), "\n")
#'
#' # Examine detailed initial fit
#' cat("\nInitial fit details:\n")
#' cat("Method:", adaptive_res$method, "\n")
#' cat("Groups selected:", adaptive_res$n_selected_init, "\n")
#' cat("Group norms range: [",
#'     min(adaptive_res$group_norms_init), ",",
#'     max(adaptive_res$group_norms_init), "]\n")
#' }
#'
#' @references
#' - Zou, H. (2006). The adaptive lasso and its oracle properties. Journal of
#'   the American Statistical Association, 101(476), 1418-1429.
#' - Huang, J., Ma, S., & Zhang, C. H. (2008). Adaptive Lasso for sparse
#'   high-dimensional regression models. Statistica Sinica, 18(4), 1603-1618.
#' - Wang, H., Li, B., & Leng, C. (2009). Shrinkage tuning parameter selection
#'   with a diverging number of parameters. Journal of the Royal Statistical
#'   Society: Series B, 71(3), 671-683.
#'
#' @seealso
#' \code{\link{compute_adaptive_weights_fast}} for faster alternative method,
#' \code{\link{cv_two_stage_adaptive}} for using adaptive weights in CV,
#' \code{\link{spherical_sim_group}} for model fitting with weights,
#' \code{compute_lambda_max_with_weights} for computing lambda_max
#'
#' @export
compute_adaptive_weights_grouplasso <- function(X, Y, group_idx,
                                                gamma_power = 1,
                                                initial_lambda_fraction = 0.3,
                                                initial_gamma = 0.01,
                                                n_knots = 5,
                                                verbose = FALSE) {

  n <- nrow(X)
  p <- ncol(X)
  q <- ncol(Y)
  G <- length(group_idx)

  if (verbose) cat("Computing adaptive weights from initial group lasso...\n")

  # Step 1: Compute lambda_max for standard weights
  base_weights <- sapply(group_idx, function(idx) sqrt(length(idx)))

  lambda_max_init <- compute_lambda_max_with_weights(
    X, Y, group_idx,
    weights = base_weights,
    n_knots = n_knots,
    multiplier = 3.0
  )

  # Step 2: Fit initial group lasso
  initial_lambda <- lambda_max_init * initial_lambda_fraction

  if (verbose) {
    cat(sprintf("  Initial fit: lambda = %.5f\n", initial_lambda))
  }

  initial_fit <- tryCatch({
    spherical_sim_group(
      X, Y, group_idx,
      lambda = initial_lambda,
      gamma = initial_gamma,
      weights = base_weights,
      n_knots = n_knots,
      max_iter = 100,
      verbose = FALSE
    )
  }, error = function(e) {
    warning("Initial fit failed: ", e$message)
    NULL
  })

  if (is.null(initial_fit)) {
    if (verbose) cat("  WARNING: Using standard weights\n")
    return(list(
      weights = base_weights,
      beta_init = NULL,
      n_selected_init = 0,
      method = "standard"
    ))
  }

  # Step 3: Compute group norms
  beta_init <- initial_fit$beta

  group_norms_init <- sapply(group_idx, function(idx) {
    sqrt(sum(beta_init[idx]^2))
  })

  n_selected_init <- length(initial_fit$selected_groups)

  if (verbose) {
    cat(sprintf("  Initial selected: %d/%d groups\n", n_selected_init, G))
    cat(sprintf("  Group norms: [%.4f, %.4f]\n",
                min(group_norms_init), max(group_norms_init)))
  }

  # Step 4: Adaptive weights
  epsilon <- 1e-4
  group_norms_init[group_norms_init < epsilon] <- epsilon

  adaptive_weights <- base_weights / (group_norms_init^gamma_power)
  adaptive_weights <- adaptive_weights * G / sum(adaptive_weights)

  if (verbose) {
    cat(sprintf("  Adaptive weights: [%.4f, %.4f]\n",
                min(adaptive_weights), max(adaptive_weights)))
  }

  list(
    weights = adaptive_weights,
    beta_init = beta_init,
    group_norms_init = group_norms_init,
    n_selected_init = n_selected_init,
    method = "grouplasso"
  )
}

#' Compute Adaptive Weights via Fast Correlation Screening
#'
#' Computes adaptive weights for the adaptive group lasso using a fast correlation-based
#' screening method. This approach avoids fitting an initial model, making it much
#' faster than \code{\link{compute_adaptive_weights_grouplasso}} while providing
#' reasonable weight estimates for variable selection.
#'
#' @param X Design matrix (n × p)
#' @param Y Response matrix on unit sphere (n × q)
#' @param group_idx List of length G, where each element contains indices of predictors in that group
#' @param gamma_power Power for adaptive weight formula (default 1). Higher values give stronger adaptation
#' @param verbose Print progress information (default FALSE)
#'
#' @details
#' This function provides a computationally efficient alternative to fitting an
#' initial group lasso model for computing adaptive weights.
#'
#' \strong{Fast Weight Computation:}
#' \enumerate{
#'   \item \strong{Group scoring}: For each group g, compute
#'         \deqn{s_g = \max_{j \in G_g, k=1,...,q} |cor(X_j, Y_k)|}
#'         This captures the maximum absolute correlation between any predictor
#'         in group g and any response component
#'   \item \strong{Adaptive weights}: Compute
#'         \deqn{w_g^{(adapt)} = \frac{\sqrt{|G_g|}}{(s_g + \epsilon)^\gamma}}
#'         where ε = 0.01 prevents division by zero
#'   \item \strong{Normalize}: Scale weights so Σ_g w_g = G
#' }
#'
#' \strong{Mathematical Rationale:}
#' The correlation score s_g serves as a proxy for the importance of group g:
#' \itemize{
#'   \item \emph{High s_g}: Strong marginal association → likely important → small weight
#'   \item \emph{Low s_g}: Weak marginal association → likely unimportant → large weight
#' }
#' This mimics the behavior of adaptive weights from an initial lasso fit, where
#' groups with large coefficients receive smaller penalties.
#'
#' \strong{Comparison with Group Lasso Method:}
#' \tabular{lll}{
#'   \strong{Aspect} \tab \strong{Fast Method} \tab \strong{Group Lasso Method} \cr
#'   Speed \tab Very fast (O(npq)) \tab Slow (requires optimization) \cr
#'   Accuracy \tab Marginal associations \tab Joint model \cr
#'   Correlation handling \tab Poor with high correlation \tab Better with correlation \cr
#'   Use case \tab Large p, quick screening \tab Final analysis, moderate p \cr
#'   Recommended \tab p > 1000 or initial screening \tab p < 500, final model
#' }
#'
#' \strong{When to Use Fast Method:}
#' \itemize{
#'   \item Very high-dimensional problems (p > 1000)
#'   \item Rapid exploratory analysis
#'   \item Initial screening before more careful analysis
#'   \item When computational resources are limited
#'   \item When initial group lasso fits are unstable
#' }
#'
#' \strong{When to Use Group Lasso Method:}
#' \itemize{
#'   \item Final analysis requiring best possible selection
#'   \item Moderate dimensional problems (p < 500)
#'   \item When predictors are highly correlated
#'   \item When computational time is not critical
#'   \item Published results requiring highest accuracy
#' }
#'
#' \strong{Limitations:}
#' \itemize{
#'   \item Uses marginal correlations, ignoring predictor relationships
#'   \item May give suboptimal weights when predictors are highly correlated
#'   \item Does not account for joint effects or confounding
#'   \item Fixed epsilon (0.01) may not be optimal for all problems
#' }
#'
#' @return List with the following components:
#' \describe{
#'   \item{weights}{Vector of adaptive weights (length G). These should be passed
#'         to \code{\link{spherical_sim_group}} or \code{\link{cv_lambda_path_early_stop}}}
#'   \item{group_scores}{Vector of correlation scores for each group (length G).
#'         Shows s_g = max absolute correlation for group g}
#'   \item{method}{Character string: always "fast"}
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
#' # Compute adaptive weights (fast method)
#' adaptive_fast <- compute_adaptive_weights_fast(
#'   X = data$X,
#'   Y = data$Y,
#'   group_idx = data$group_idx,
#'   verbose = TRUE
#' )
#'
#' # Examine results
#' cat("Fast adaptive weights:\n")
#' print(data.frame(
#'   Group = 1:length(adaptive_fast$weights),
#'   Weight = round(adaptive_fast$weights, 4),
#'   Score = round(adaptive_fast$group_scores, 4),
#'   TrueActive = 1:length(adaptive_fast$weights) %in% data$active_groups
#' ))
#'
#' # Visualize weights and scores
#' par(mfrow = c(1, 2))
#'
#' # Correlation scores by group
#' barplot(adaptive_fast$group_scores, names.arg = 1:8,
#'         xlab = "Group", ylab = "Correlation Score",
#'         main = "Group Correlation Scores",
#'         col = ifelse(1:8 %in% data$active_groups, "lightblue", "lightgray"))
#' legend("topright", c("Active", "Inactive"),
#'        fill = c("lightblue", "lightgray"))
#'
#' # Adaptive weights
#' barplot(adaptive_fast$weights, names.arg = 1:8,
#'         xlab = "Group", ylab = "Adaptive Weight",
#'         main = "Adaptive Weights (Fast)",
#'         col = ifelse(1:8 %in% data$active_groups, "lightblue", "lightgray"))
#'
#' # Compare fast vs group lasso methods
#' adaptive_grouplasso <- compute_adaptive_weights_grouplasso(
#'   X = data$X,
#'   Y = data$Y,
#'   group_idx = data$group_idx,
#'   verbose = TRUE
#' )
#'
#' # Plot comparison
#' plot(adaptive_grouplasso$weights, adaptive_fast$weights,
#'      pch = 19, cex = 1.5,
#'      xlab = "Group Lasso Weights",
#'      ylab = "Fast Weights",
#'      main = "Weight Comparison",
#'      col = ifelse(1:8 %in% data$active_groups, "blue", "red"))
#' abline(0, 1, col = "gray", lty = 2)
#' legend("topleft", c("Active", "Inactive"),
#'        col = c("blue", "red"), pch = 19)
#'
#' # Correlation between weight methods
#' cor_weights <- cor(adaptive_grouplasso$weights, adaptive_fast$weights)
#' cat("\nCorrelation between weight methods:", round(cor_weights, 3), "\n")
#'
#' # Timing comparison
#' system.time({
#'   weights_fast <- compute_adaptive_weights_fast(
#'     data$X, data$Y, data$group_idx
#'   )
#' })
#'
#' system.time({
#'   weights_grouplasso <- compute_adaptive_weights_grouplasso(
#'     data$X, data$Y, data$group_idx
#'   )
#' })
#'
#' # Use in cross-validation
#' cv_fast <- cv_two_stage_adaptive(
#'   X = data$X,
#'   Y = data$Y,
#'   group_idx = data$group_idx,
#'   use_adaptive = TRUE,
#'   adaptive_method = "fast",
#'   verbose = FALSE
#' )
#'
#' cv_grouplasso <- cv_two_stage_adaptive(
#'   X = data$X,
#'   Y = data$Y,
#'   group_idx = data$group_idx,
#'   use_adaptive = TRUE,
#'   adaptive_method = "grouplasso",
#'   verbose = FALSE
#' )
#'
#' # Compare selection performance
#' fit_fast <- spherical_sim_group(
#'   data$X, data$Y, data$group_idx,
#'   cv_fast$best_lambda, cv_fast$best_gamma,
#'   weights = adaptive_fast$weights, verbose = FALSE
#' )
#'
#' fit_grouplasso <- spherical_sim_group(
#'   data$X, data$Y, data$group_idx,
#'   cv_grouplasso$best_lambda, cv_grouplasso$best_gamma,
#'   weights = adaptive_grouplasso$weights, verbose = FALSE
#' )
#'
#' cat("\nFast method selected:", fit_fast$selected_groups, "\n")
#' cat("Group lasso method selected:", fit_grouplasso$selected_groups, "\n")
#' cat("True active groups:", data$active_groups, "\n")
#'
#' # Compute selection metrics
#' TPR_fast <- length(intersect(fit_fast$selected_groups, data$active_groups)) /
#'             length(data$active_groups)
#' TPR_grouplasso <- length(intersect(fit_grouplasso$selected_groups, data$active_groups)) /
#'                   length(data$active_groups)
#'
#' cat("\nTPR (fast):", TPR_fast, "\n")
#' cat("TPR (group lasso):", TPR_grouplasso, "\n")
#'
#' # Effect of gamma_power
#' gamma_powers <- c(0.5, 1, 1.5, 2)
#'
#' weights_list <- lapply(gamma_powers, function(gp) {
#'   res <- compute_adaptive_weights_fast(
#'     X = data$X, Y = data$Y, group_idx = data$group_idx,
#'     gamma_power = gp
#'   )
#'   res$weights
#' })
#'
#' # Plot weights for different gamma_power
#' matplot(1:8, do.call(cbind, weights_list),
#'         type = "b", pch = 19, lty = 1,
#'         xlab = "Group", ylab = "Weight",
#'         main = "Fast Weights vs gamma_power")
#' legend("topright", paste("γ =", gamma_powers),
#'        col = 1:4, lty = 1, pch = 19)
#' abline(v = data$active_groups, col = "gray", lty = 2)
#'
#' # High-dimensional example showing speed advantage
#' set.seed(456)
#' data_large <- generate_spherical_data(n = 500, p = 1000, G = 100,
#'                                       active_groups = 1:5,
#'                                       seed = 456)
#'
#' cat("\nHigh-dimensional example (p = 1000, G = 100):\n")
#'
#' time_fast <- system.time({
#'   weights_fast_large <- compute_adaptive_weights_fast(
#'     data_large$X, data_large$Y, data_large$group_idx
#'   )
#' })
#'
#' cat("Fast method time:", time_fast["elapsed"], "seconds\n")
#' cat("(Group lasso method would take much longer)\n")
#'
#' # Correlation between scores and true signals
#' true_active <- as.numeric(1:8 %in% data$active_groups)
#' cor_scores_truth <- cor(adaptive_fast$group_scores, true_active)
#' cat("\nCorrelation between correlation scores and true active groups:",
#'     round(cor_scores_truth, 3), "\n")
#'
#' # Distribution of weights
#' active_weights <- adaptive_fast$weights[data$active_groups]
#' inactive_weights <- adaptive_fast$weights[setdiff(1:8, data$active_groups)]
#'
#' boxplot(list(Active = active_weights, Inactive = inactive_weights),
#'         main = "Weight Distribution by Group Type",
#'         ylab = "Adaptive Weight",
#'         col = c("lightblue", "lightgray"))
#'
#' cat("\nMean active group weight:", mean(active_weights), "\n")
#' cat("Mean inactive group weight:", mean(inactive_weights), "\n")
#' cat("Weight ratio (inactive/active):",
#'     mean(inactive_weights) / mean(active_weights), "\n")
#' }
#'
#' @references
#' - Fan, J., & Lv, J. (2008). Sure independence screening for ultrahigh
#'   dimensional feature space. Journal of the Royal Statistical Society:
#'   Series B, 70(5), 849-911.
#' - Zhao, P., & Yu, B. (2006). On model selection consistency of Lasso.
#'   Journal of Machine Learning Research, 7, 2541-2563.
#' - Zou, H. (2006). The adaptive lasso and its oracle properties. Journal of
#'   the American Statistical Association, 101(476), 1418-1429.
#'
#' @seealso
#' \code{\link{compute_adaptive_weights_grouplasso}} for more accurate but slower method,
#' \code{\link{cv_two_stage_adaptive}} for using adaptive weights in CV,
#' \code{\link{spherical_sim_group}} for model fitting with weights
#'
#' @export
compute_adaptive_weights_fast <- function(X, Y, group_idx,
                                          gamma_power = 1,
                                          verbose = FALSE) {

  n <- nrow(X)
  p <- ncol(X)
  q <- ncol(Y)
  G <- length(group_idx)

  if (verbose) cat("Computing adaptive weights (fast)...\n")

  group_scores <- numeric(G)

  for (g in 1:G) {
    idx <- group_idx[[g]]
    X_g <- X[, idx, drop = FALSE]
    cors <- cor(X_g, Y)
    group_scores[g] <- max(abs(cors))
  }

  if (verbose) {
    cat(sprintf("  Group scores: [%.4f, %.4f]\n",
                min(group_scores), max(group_scores)))
  }

  base_weights <- sapply(group_idx, function(idx) sqrt(length(idx)))

  epsilon <- 0.01
  group_scores[group_scores < epsilon] <- epsilon

  adaptive_weights <- base_weights / (group_scores^gamma_power)
  adaptive_weights <- adaptive_weights * G / sum(adaptive_weights)

  if (verbose) {
    cat(sprintf("  Adaptive weights: [%.4f, %.4f]\n",
                min(adaptive_weights), max(adaptive_weights)))
  }

  list(
    weights = adaptive_weights,
    group_scores = group_scores,
    method = "fast"
  )
}

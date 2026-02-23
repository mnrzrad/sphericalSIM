#' Run Single Simulation for Spherical Single-Index Model with Group Lasso
#'
#' Performs a complete simulation run including data generation, cross-validation
#' for tuning parameter selection, final model fitting, and comprehensive evaluation
#' of variable selection and estimation performance.
#'
#' @param sim_id Simulation ID number for tracking and seed generation
#' @param n Sample size (default 500)
#' @param p Total number of predictors (default 100)
#' @param q Response dimension on sphere (default 3, i.e., unit sphere in R^3)
#' @param G Total number of groups (default 10)
#' @param active_groups Vector of true active group indices (default c(1, 3, 5))
#' @param kappa Concentration parameter for von Mises-Fisher distribution (default 50)
#' @param rho Within-group correlation for predictor generation (default 0.5)
#' @param use_adaptive Use adaptive group lasso weights (default TRUE)
#' @param adaptive_method Method for computing adaptive weights: "fast" or "grouplasso" (default "fast")
#' @param nlambda Number of lambda values in CV grid (default 12)
#' @param ngamma_coarse Number of gamma values in coarse CV stage (default 5)
#' @param ngamma_fine Number of gamma values in fine CV stage (default 8)
#' @param n_folds Number of cross-validation folds (default 3)
#' @param n_knots Number of internal B-spline knots for link function approximation (default 5)
#' @param verbose Print detailed progress messages (default FALSE)
#' @param seed_offset Base seed offset for reproducibility; actual seed is seed_offset + sim_id (default 10000)
#'
#' @details
#' The function executes a complete simulation pipeline:
#' \enumerate{
#'   \item \strong{Data Generation}: Generates synthetic data from a spherical single-index
#'         model with specified parameters using \code{\link{generate_spherical_data}}
#'   \item \strong{Cross-Validation}: Performs two-stage cross-validation with adaptive
#'         weights to select optimal lambda and gamma parameters
#'   \item \strong{Final Fitting}: Fits the final model with selected tuning parameters
#'         using \code{\link{spherical_sim_group}}
#'   \item \strong{Link Evaluation}: Estimates the link function on a grid of 100 points
#'   \item \strong{Performance Metrics}: Computes comprehensive evaluation metrics including
#'         variable selection accuracy and estimation errors
#' }
#'
#' The true link function used is m(z) = [sin(z), 0.5*cos(z)]^T, which is evaluated
#' on the same grid as the estimated link for fair comparison.
#'
#' \strong{Evaluation Metrics:}
#' \itemize{
#'   \item \emph{Variable Selection}: TPR (true positive rate), FPR (false positive rate)
#'   \item \emph{Parameter Estimation}: L2 error (with sign correction), angular error
#'   \item \emph{Function Estimation}: Link MSE, prediction error
#'   \item \emph{Convergence}: Number of iterations, convergence status
#' }
#'
#' \strong{Sign Ambiguity:} Since the spherical single-index model has identification
#' up to sign (beta and -beta give the same model), the L2 error is computed as
#' min(||beta_est - beta_true||, ||beta_est + beta_true||).
#'
#' @return A list containing:
#' \describe{
#'   \item{sim_id}{Simulation identifier}
#'   \item{beta_true}{True index parameter vector (length p)}
#'   \item{m_true}{True link function evaluated on grid (n_grid × q matrix)}
#'   \item{z_grid}{Grid points where link function is evaluated}
#'   \item{true_active_groups}{True active group indices}
#'   \item{beta_est}{Estimated index parameter vector (length p)}
#'   \item{m_est}{Estimated link function on grid (n_grid × q matrix)}
#'   \item{selected_groups}{Selected group indices from final model}
#'   \item{best_lambda}{Optimal lambda from cross-validation}
#'   \item{best_gamma}{Optimal gamma from cross-validation}
#'   \item{weights}{Adaptive weights used in final model}
#'   \item{n_selected}{Number of groups selected}
#'   \item{TPR}{True positive rate: proportion of true active groups selected}
#'   \item{FPR}{False positive rate: proportion of inactive groups incorrectly selected}
#'   \item{beta_l2_error}{L2 error for beta (accounting for sign ambiguity)}
#'   \item{angular_error}{Angular error between true and estimated beta (in degrees)}
#'   \item{link_mse}{Mean squared error of link function estimation}
#'   \item{pred_error}{Prediction error on training data}
#'   \item{converged}{Logical indicating convergence (TRUE if n_iter < 200)}
#'   \item{n_iter}{Number of iterations until convergence}
#' }
#'
#' @examples
#' \dontrun{
#' # Run single simulation with default settings
#' result <- run_single_simulation(sim_id = 1)
#'
#' # Check variable selection performance
#' cat("Selected groups:", result$selected_groups, "\n")
#' cat("True active groups:", result$true_active_groups, "\n")
#' cat("TPR:", result$TPR, "FPR:", result$FPR, "\n")
#'
#' # Examine estimation accuracy
#' cat("Angular error:", result$angular_error, "degrees\n")
#' cat("Link MSE:", result$link_mse, "\n")
#'
#' # Plot true vs estimated link function
#' plot(result$z_grid, result$m_true[,1], type = "l", col = "blue",
#'      xlab = "z", ylab = "m_1(z)", main = "Link Function Component 1")
#' lines(result$z_grid, result$m_est[,1], col = "red", lty = 2)
#' legend("topright", c("True", "Estimated"), col = c("blue", "red"), lty = 1:2)
#'
#' # Run with custom settings and verbose output
#' result <- run_single_simulation(
#'   sim_id = 1,
#'   n = 200,
#'   p = 50,
#'   G = 5,
#'   active_groups = c(1, 2),
#'   use_adaptive = TRUE,
#'   adaptive_method = "grouplasso",
#'   verbose = TRUE
#' )
#'
#' # Multiple simulations for Monte Carlo study
#' results <- lapply(1:100, function(i) {
#'   run_single_simulation(sim_id = i, verbose = FALSE)
#' })
#'
#' # Summarize performance across simulations
#' mean_tpr <- mean(sapply(results, function(x) x$TPR))
#' mean_angular <- mean(sapply(results, function(x) x$angular_error))
#' cat("Average TPR:", mean_tpr, "\n")
#' cat("Average angular error:", mean_angular, "degrees\n")
#' }
#'
#' @seealso
#' \code{\link{generate_spherical_data}} for data generation,
#' \code{\link{cv_two_stage_adaptive}} for cross-validation,
#' \code{\link{spherical_sim_group}} for model fitting,
#' \code{\link{run_simulation_study}} for running multiple simulations,
#' \code{\link{summarize_results}} for analyzing simulation results
#'
#' @export
run_single_simulation <- function(sim_id,
                                  n = 500, p = 100, q = 3, G = 10,
                                  active_groups = c(1, 3, 5),
                                  kappa = 50, rho = 0.5,
                                  use_adaptive = TRUE,
                                  adaptive_method = "fast",
                                  nlambda = 12,
                                  ngamma_coarse = 5,
                                  ngamma_fine = 8,
                                  n_folds = 3,
                                  n_knots = 5,
                                  verbose = FALSE,
                                  seed_offset = 10000) {

  if (verbose) {
    cat(sprintf("\n========================================\n"))
    cat(sprintf("Simulation %d\n", sim_id))
    cat(sprintf("========================================\n"))
  }

  # Generate data
  data <- generate_spherical_data(
    n = n, p = p, q = q, G = G,
    active_groups = active_groups,
    kappa = kappa, rho = rho,
    seed = seed_offset + sim_id
  )

  X <- data$X
  Y <- data$Y
  beta_true <- data$beta_true
  z_true <- data$z_true
  m_true <- data$mu_true
  group_idx <- data$group_idx

  # CV
  if (verbose) cat("Step 1: Cross-validation...\n")

  cv_results <- cv_two_stage_adaptive(
    X = X, Y = Y, group_idx = group_idx,
    use_adaptive = use_adaptive,
    adaptive_method = adaptive_method,
    nlambda = nlambda,
    ngamma_coarse = ngamma_coarse,
    ngamma_fine = ngamma_fine,
    n_folds = n_folds,
    n_knots = n_knots,
    verbose = verbose
  )

  # Final model
  if (verbose) cat("\nStep 2: Final fit...\n")

  final_fit <- spherical_sim_group(
    X = X, Y = Y,
    group_idx = group_idx,
    lambda = cv_results$best_lambda,
    gamma = cv_results$best_gamma,
    weights = cv_results$weights,
    n_knots = n_knots,
    max_iter = 200,
    verbose = verbose
  )

  beta_est <- final_fit$beta

  # Evaluate link
  link_eval <- eval_link(final_fit, z_grid = NULL, n_grid = 100)
  m_est <- link_eval$m
  z_grid_used <- link_eval$z

  m_true_grid <- do.call(rbind, lapply(z_grid_used, function(x) c(sin(x), 0.5*cos(x))))

  # Metrics
  selected_groups <- final_fit$selected_groups
  n_selected <- length(selected_groups)

  TPR <- length(intersect(selected_groups, active_groups)) / length(active_groups)
  FPR <- length(setdiff(selected_groups, active_groups)) / max(1, G - length(active_groups))

  beta_error1 <- sqrt(sum((beta_est - beta_true)^2))
  beta_error2 <- sqrt(sum((beta_est + beta_true)^2))
  beta_l2_error <- min(beta_error1, beta_error2)

  cos_angle <- abs(sum(beta_est * beta_true))
  cos_angle <- min(cos_angle, 1)
  angular_error <- acos(cos_angle) * 180 / pi

  link_mse <- mean(rowSums((m_est - m_true_grid)^2))
  pred_error <- mean(rowSums((Y - final_fit$fitted)^2))

  if (verbose) {
    cat(sprintf("\nResults:\n"))
    cat(sprintf("  Selected: %d (True: %d)\n", n_selected, length(active_groups)))
    cat(sprintf("  TPR: %.3f, FPR: %.3f\n", TPR, FPR))
    cat(sprintf("  Angular error: %.2f degrees\n", angular_error))
  }

  list(
    sim_id = sim_id,
    beta_true = beta_true,
    m_true = m_true_grid,
    z_grid = z_grid_used,
    true_active_groups = active_groups,
    beta_est = beta_est,
    m_est = m_est,
    selected_groups = selected_groups,
    best_lambda = cv_results$best_lambda,
    best_gamma = cv_results$best_gamma,
    weights = cv_results$weights,
    n_selected = n_selected,
    TPR = TPR,
    FPR = FPR,
    beta_l2_error = beta_l2_error,
    angular_error = angular_error,
    link_mse = link_mse,
    pred_error = pred_error,
    converged = final_fit$n_iter < 200,
    n_iter = final_fit$n_iter
  )
}

#' Run Simulation Study for Spherical Single-Index Model
#'
#' Conducts a comprehensive simulation study by running multiple replications
#' of spherical single-index model estimation with group lasso. Supports both
#' sequential and parallel execution.
#'
#' @param n_sim Number of simulation replications to run (default 50)
#' @param n Sample size for each simulation (default 500)
#' @param p Total number of predictors (default 100)
#' @param q Response dimension on sphere (default 3)
#' @param G Total number of groups (default 10)
#' @param active_groups Vector of true active group indices (default c(1, 3, 5))
#' @param use_adaptive Use adaptive group lasso weights (default TRUE)
#' @param adaptive_method Method for computing adaptive weights: "fast" or "grouplasso" (default "fast")
#' @param parallel Use parallel processing (default TRUE)
#' @param n_cores Number of cores for parallel processing. If NULL, uses detectCores() - 1 (default NULL)
#' @param save_file File path to save results as RDS file. If NULL, results are not saved (default "simulation_results.rds")
#' @param verbose_sim Print detailed progress for each simulation (default FALSE)
#'
#' @details
#' The function performs the following steps:
#' \enumerate{
#'   \item Prints study configuration and start time
#'   \item Runs n_sim independent simulations via \code{\link{run_single_simulation}}
#'   \item Handles errors gracefully, excluding failed simulations from results
#'   \item Saves results to RDS file if save_file is specified
#'   \item Reports completion summary and end time
#' }
#'
#' When parallel = TRUE, the function uses \code{mclapply} from the parallel package
#' to distribute simulations across multiple cores. Error handling ensures that
#' failures in individual simulations do not crash the entire study.
#'
#' Each simulation uses a unique seed (seed_offset + sim_id) to ensure reproducibility
#' while maintaining independence between replications.
#'
#' @return List of results from all successful simulations. Each element is a list
#'   returned by \code{\link{run_single_simulation}} containing:
#'   \itemize{
#'     \item Estimation results (beta_est, m_est, selected_groups)
#'     \item Performance metrics (TPR, FPR, angular_error, link_mse, etc.)
#'     \item Tuning parameters (best_lambda, best_gamma, weights)
#'     \item Convergence information (converged, n_iter)
#'   }
#'
#' @examples
#' \dontrun{
#' # Run 50 simulations with default settings (parallel)
#' results <- run_simulation_study(
#'   n_sim = 50,
#'   save_file = "my_results.rds"
#' )
#'
#' # Sequential execution with verbose output
#' results <- run_simulation_study(
#'   n_sim = 10,
#'   parallel = FALSE,
#'   verbose_sim = TRUE,
#'   save_file = NULL
#' )
#'
#' # Custom settings with parallel processing
#' results <- run_simulation_study(
#'   n_sim = 100,
#'   n = 200,
#'   p = 50,
#'   G = 5,
#'   active_groups = c(1, 2),
#'   adaptive_method = "grouplasso",
#'   n_cores = 4,
#'   save_file = "custom_study.rds"
#' )
#'
#' # Analyze results
#' summary <- summarize_results(results)
#' }
#'
#' @seealso
#' \code{\link{run_single_simulation}} for single simulation execution,
#' \code{\link{summarize_results}} for analyzing simulation results
#'
#' @importFrom parallel mclapply detectCores
#' @export
run_simulation_study <- function(n_sim = 50,
                                 n = 500, p = 100, q = 3, G = 10,
                                 active_groups = c(1, 3, 5),
                                 use_adaptive = TRUE,
                                 adaptive_method = "fast",
                                 parallel = TRUE,
                                 n_cores = NULL,
                                 save_file = "simulation_results.rds",
                                 verbose_sim = FALSE) {

  cat("========================================\n")
  cat("SIMULATION STUDY\n")
  cat("========================================\n")
  cat(sprintf("Number of simulations: %d\n", n_sim))
  cat(sprintf("Sample size: n=%d, p=%d, q=%d, G=%d\n", n, p, q, G))
  cat(sprintf("Method: %s\n", ifelse(use_adaptive,
                                     paste0("Adaptive (", adaptive_method, ")"), "Standard")))
  cat(sprintf("Start: %s\n", Sys.time()))
  cat("========================================\n\n")

  if (parallel) {
    if (is.null(n_cores)) n_cores <- max(1, detectCores() - 1)
    cat(sprintf("Running parallel with %d cores\n\n", n_cores))

    results <- mclapply(1:n_sim, function(i) {
      tryCatch({
        run_single_simulation(
          sim_id = i,
          n = n, p = p, q = q, G = G,
          active_groups = active_groups,
          use_adaptive = use_adaptive,
          adaptive_method = adaptive_method,
          verbose = verbose_sim
        )
      }, error = function(e) {
        cat(sprintf("ERROR in sim %d: %s\n", i, e$message))
        NULL
      })
    }, mc.cores = n_cores)

  } else {
    results <- list()
    for (i in 1:n_sim) {
      cat(sprintf("\n>>> Simulation %d/%d <<<\n", i, n_sim))
      results[[i]] <- tryCatch({
        run_single_simulation(
          sim_id = i,
          n = n, p = p, q = q, G = G,
          active_groups = active_groups,
          use_adaptive = use_adaptive,
          adaptive_method = adaptive_method,
          verbose = TRUE
        )
      }, error = function(e) {
        cat(sprintf("ERROR: %s\n", e$message))
        NULL
      })
    }
  }

  results <- results[!sapply(results, is.null)]

  cat("\n========================================\n")
  cat(sprintf("Completed: %d/%d\n", length(results), n_sim))
  cat(sprintf("End: %s\n", Sys.time()))
  cat("========================================\n")

  if (!is.null(save_file)) {
    saveRDS(results, file = save_file)
    cat(sprintf("\nSaved to: %s\n", save_file))
  }

  return(results)
}

#' Summarize Simulation Study Results
#'
#' Computes comprehensive summary statistics from existing simulation results,
#' including metrics that may not have been computed originally. This function
#' can work with both old and new result formats by computing missing metrics
#' from available data.
#'
#' @param results List of simulation results from \code{run_simulation_study()}
#'   or multiple calls to \code{run_single_simulation()}
#'
#' @details
#' This function extracts or computes all evaluation metrics from Section 3:
#'
#' \strong{Coefficient Metrics (Section 3.1):}
#' \itemize{
#'   \item MSE_beta, RMSE_beta, MAE_beta: Computed from beta_true and beta_est
#'   \item rho_beta: Cosine similarity (after sign normalization)
#'   \item angular_error: Angular error in degrees
#' }
#'
#' \strong{Link Function Metrics (Section 3.2):}
#' \itemize{
#'   \item MSE_m, MAE_m: Computed from m_true and m_est
#' }
#'
#' \strong{Predictive Performance (Section 3.3):}
#' \itemize{
#'   \item W1: 1D Wasserstein distance (computed from m_true and m_est)
#'   \item pred_error: Prediction error
#' }
#'
#' \strong{Variable Selection Metrics (Section 3.4):}
#' \itemize{
#'   \item Precision, Recall, F1, FPR: Computed from selected_groups and true_active_groups
#'   \item TP, FP, FN, TN: Confusion matrix counts
#' }
#'
#' @return List with two components:
#' \describe{
#'   \item{metrics}{Data frame with one row per simulation containing all metrics}
#'   \item{summary}{Data frame with summary statistics for each metric category}
#' }
#'
#' @export
summarize_results <- function(results) {

  n_sim <- length(results)

  cat("\n========================================\n")
  cat("PROCESSING SIMULATION RESULTS\n")
  cat("========================================\n")
  cat(sprintf("Number of simulations: %d\n", n_sim))
  cat("Computing comprehensive metrics...\n\n")

  # Initialize lists to store computed metrics
  metrics_list <- list()

  for (i in 1:n_sim) {
    res <- results[[i]]

    # Get basic info
    sim_id <- if (!is.null(res$sim_id)) res$sim_id else i
    p <- length(res$beta_true)

    # Get group info
    true_active <- res$true_active_groups
    selected <- res$selected_groups

    # Determine G (total number of groups)
    G <- if (!is.null(res$weights)) {
      length(res$weights)
    } else {
      # Estimate from max group index
      max(c(true_active, selected, 0))
    }

    # ========================================
    # Sign normalization
    # ========================================
    beta_true <- res$beta_true
    beta_est <- res$beta_est

    # Apply sign normalization if needed
    if (sum(beta_true * beta_est) < 0) {
      beta_est <- -beta_est
    }

    # ========================================
    # 3.1 Coefficient Metrics
    # ========================================

    MSE_beta <- sum((beta_est - beta_true)^2) / p
    RMSE_beta <- sqrt(MSE_beta)
    MAE_beta <- sum(abs(beta_est - beta_true)) / p

    # Cosine similarity
    rho_beta <- sum(beta_true * beta_est) /
      (sqrt(sum(beta_true^2)) * sqrt(sum(beta_est^2)))
    rho_beta <- min(max(rho_beta, -1), 1)

    # Angular error (degrees)
    angular_error <- acos(abs(rho_beta)) * 180 / pi

    # ========================================
    # 3.2 Link Function Metrics
    # ========================================

    m_true <- res$m_true
    m_est <- res$m_est

    M <- nrow(m_est)  # Number of grid points
    q_minus_1 <- ncol(m_est)  # Should be q - 1

    MSE_m <- sum((m_est - m_true)^2) / (M * q_minus_1)
    MAE_m <- sum(abs(m_est - m_true)) / (M * q_minus_1)

    # ========================================
    # 3.3 Predictive Performance: Wasserstein
    # ========================================

    W1_components <- numeric(q_minus_1)
    for (j in 1:q_minus_1) {
      sorted_true <- sort(m_true[, j])
      sorted_est <- sort(m_est[, j])
      W1_components[j] <- sum(abs(sorted_true - sorted_est)) / M
    }
    W1 <- mean(W1_components)

    # ========================================
    # 3.4 Variable Selection Metrics
    # ========================================

    S <- true_active
    S_hat <- selected
    S_c <- setdiff(1:G, S)

    TP <- length(intersect(S_hat, S))
    FP <- length(setdiff(S_hat, S))
    FN <- length(setdiff(S, S_hat))
    TN <- length(setdiff(S_c, S_hat))

    # Precision (Eq. 3.8)
    Precision <- if (length(S_hat) > 0) TP / length(S_hat) else 0

    # Recall (Eq. 3.9)
    Recall <- if (length(S) > 0) TP / length(S) else 0

    # F1-score (Eq. 3.10)
    F1 <- if (Precision + Recall > 0) {
      2 * (Precision * Recall) / (Precision + Recall)
    } else {
      0
    }

    # False positive rate (Eq. 3.11)
    FPR <- if (length(S_c) > 0) FP / length(S_c) else 0

    # ========================================
    # Additional metrics
    # ========================================

    pred_error <- if (!is.null(res$pred_error)) {
      res$pred_error
    } else {
      NA
    }

    best_lambda <- if (!is.null(res$best_lambda)) res$best_lambda else NA
    best_gamma <- if (!is.null(res$best_gamma)) res$best_gamma else NA
    converged <- if (!is.null(res$converged)) res$converged else NA
    n_iter <- if (!is.null(res$n_iter)) res$n_iter else NA

    # Store all metrics
    metrics_list[[i]] <- data.frame(
      sim_id = sim_id,

      # Selection counts
      n_selected = length(selected),
      TP = TP,
      FP = FP,
      FN = FN,
      TN = TN,

      # Coefficient metrics (Section 3.1)
      MSE_beta = MSE_beta,
      RMSE_beta = RMSE_beta,
      MAE_beta = MAE_beta,
      rho_beta = rho_beta,
      angular_error = angular_error,

      # Link function metrics (Section 3.2)
      MSE_m = MSE_m,
      MAE_m = MAE_m,

      # Predictive performance (Section 3.3)
      W1 = W1,
      pred_error = pred_error,

      # Variable selection metrics (Section 3.4)
      Precision = Precision,
      Recall = Recall,
      F1 = F1,
      FPR = FPR,

      # Tuning parameters
      best_lambda = best_lambda,
      best_gamma = best_gamma,

      # Convergence
      converged = converged,
      n_iter = n_iter
    )
  }

  # Combine all metrics
  metrics <- do.call(rbind, metrics_list)
  rownames(metrics) <- NULL

  # ========================================
  # Print Summary Statistics
  # ========================================

  cat("========================================\n")
  cat("SUMMARY STATISTICS\n")
  cat("========================================\n\n")

  cat("COEFFICIENT METRICS (Section 3.1):\n")
  cat("-----------------------------------\n")
  coef_stats <- data.frame(
    Metric = c("MSE_beta", "RMSE_beta", "MAE_beta", "rho_beta", "angular_error"),
    Mean = c(mean(metrics$MSE_beta, na.rm = TRUE),
             mean(metrics$RMSE_beta, na.rm = TRUE),
             mean(metrics$MAE_beta, na.rm = TRUE),
             mean(metrics$rho_beta, na.rm = TRUE),
             mean(metrics$angular_error, na.rm = TRUE)),
    SD = c(sd(metrics$MSE_beta, na.rm = TRUE),
           sd(metrics$RMSE_beta, na.rm = TRUE),
           sd(metrics$MAE_beta, na.rm = TRUE),
           sd(metrics$rho_beta, na.rm = TRUE),
           sd(metrics$angular_error, na.rm = TRUE)),
    Median = c(median(metrics$MSE_beta, na.rm = TRUE),
               median(metrics$RMSE_beta, na.rm = TRUE),
               median(metrics$MAE_beta, na.rm = TRUE),
               median(metrics$rho_beta, na.rm = TRUE),
               median(metrics$angular_error, na.rm = TRUE)),
    Min = c(min(metrics$MSE_beta, na.rm = TRUE),
            min(metrics$RMSE_beta, na.rm = TRUE),
            min(metrics$MAE_beta, na.rm = TRUE),
            min(metrics$rho_beta, na.rm = TRUE),
            min(metrics$angular_error, na.rm = TRUE)),
    Max = c(max(metrics$MSE_beta, na.rm = TRUE),
            max(metrics$RMSE_beta, na.rm = TRUE),
            max(metrics$MAE_beta, na.rm = TRUE),
            max(metrics$rho_beta, na.rm = TRUE),
            max(metrics$angular_error, na.rm = TRUE))
  )
  print(coef_stats, digits = 4, row.names = FALSE)

  cat("\nLINK FUNCTION METRICS (Section 3.2):\n")
  cat("-------------------------------------\n")
  link_stats <- data.frame(
    Metric = c("MSE_m", "MAE_m"),
    Mean = c(mean(metrics$MSE_m, na.rm = TRUE),
             mean(metrics$MAE_m, na.rm = TRUE)),
    SD = c(sd(metrics$MSE_m, na.rm = TRUE),
           sd(metrics$MAE_m, na.rm = TRUE)),
    Median = c(median(metrics$MSE_m, na.rm = TRUE),
               median(metrics$MAE_m, na.rm = TRUE)),
    Min = c(min(metrics$MSE_m, na.rm = TRUE),
            min(metrics$MAE_m, na.rm = TRUE)),
    Max = c(max(metrics$MSE_m, na.rm = TRUE),
            max(metrics$MAE_m, na.rm = TRUE))
  )
  print(link_stats, digits = 4, row.names = FALSE)

  cat("\nPREDICTIVE PERFORMANCE (Section 3.3):\n")
  cat("--------------------------------------\n")
  pred_stats <- data.frame(
    Metric = c("W1", "pred_error"),
    Mean = c(mean(metrics$W1, na.rm = TRUE),
             mean(metrics$pred_error, na.rm = TRUE)),
    SD = c(sd(metrics$W1, na.rm = TRUE),
           sd(metrics$pred_error, na.rm = TRUE)),
    Median = c(median(metrics$W1, na.rm = TRUE),
               median(metrics$pred_error, na.rm = TRUE)),
    Min = c(min(metrics$W1, na.rm = TRUE),
            min(metrics$pred_error, na.rm = TRUE)),
    Max = c(max(metrics$W1, na.rm = TRUE),
            max(metrics$pred_error, na.rm = TRUE))
  )
  print(pred_stats, digits = 4, row.names = FALSE)

  cat("\nVARIABLE SELECTION METRICS (Section 3.4):\n")
  cat("------------------------------------------\n")
  sel_stats <- data.frame(
    Metric = c("Precision", "Recall", "F1", "FPR"),
    Mean = c(mean(metrics$Precision, na.rm = TRUE),
             mean(metrics$Recall, na.rm = TRUE),
             mean(metrics$F1, na.rm = TRUE),
             mean(metrics$FPR, na.rm = TRUE)),
    SD = c(sd(metrics$Precision, na.rm = TRUE),
           sd(metrics$Recall, na.rm = TRUE),
           sd(metrics$F1, na.rm = TRUE),
           sd(metrics$FPR, na.rm = TRUE)),
    Median = c(median(metrics$Precision, na.rm = TRUE),
               median(metrics$Recall, na.rm = TRUE),
               median(metrics$F1, na.rm = TRUE),
               median(metrics$FPR, na.rm = TRUE)),
    Min = c(min(metrics$Precision, na.rm = TRUE),
            min(metrics$Recall, na.rm = TRUE),
            min(metrics$F1, na.rm = TRUE),
            min(metrics$FPR, na.rm = TRUE)),
    Max = c(max(metrics$Precision, na.rm = TRUE),
            max(metrics$Recall, na.rm = TRUE),
            max(metrics$F1, na.rm = TRUE),
            max(metrics$FPR, na.rm = TRUE))
  )
  print(sel_stats, digits = 4, row.names = FALSE)

  cat("\nSELECTION SUMMARY:\n")
  cat("------------------\n")
  cat(sprintf("Mean # selected: %.2f (SD: %.2f)\n",
              mean(metrics$n_selected, na.rm = TRUE),
              sd(metrics$n_selected, na.rm = TRUE)))
  cat(sprintf("Mean TP: %.2f, FP: %.2f, FN: %.2f, TN: %.2f\n",
              mean(metrics$TP, na.rm = TRUE),
              mean(metrics$FP, na.rm = TRUE),
              mean(metrics$FN, na.rm = TRUE),
              mean(metrics$TN, na.rm = TRUE)))

  cat("\nCONVERGENCE:\n")
  cat("------------\n")
  if (all(!is.na(metrics$converged))) {
    cat(sprintf("Convergence rate: %.1f%%\n",
                100 * mean(metrics$converged, na.rm = TRUE)))
  }
  if (all(!is.na(metrics$n_iter))) {
    cat(sprintf("Mean iterations: %.1f (SD: %.1f)\n",
                mean(metrics$n_iter, na.rm = TRUE),
                sd(metrics$n_iter, na.rm = TRUE)))
    cat(sprintf("Iteration range: [%d, %d]\n",
                min(metrics$n_iter, na.rm = TRUE),
                max(metrics$n_iter, na.rm = TRUE)))
  }

  # Combine all summary statistics
  all_summary <- rbind(
    coef_stats,
    link_stats,
    pred_stats,
    sel_stats
  )

  cat("\n========================================\n")
  cat("SUMMARY COMPLETE\n")
  cat("========================================\n")
  cat(sprintf("\nMetrics computed for %d simulations\n", n_sim))
  cat("Access results via:\n")
  cat("  summary$metrics - Full metrics data frame\n")
  cat("  summary$summary - Summary statistics table\n\n")

  invisible(list(metrics = metrics, summary = all_summary))
}



create_group_indices <- function(groups) {
  groups <- as.integer(as.factor(groups))
  G <- max(groups)
  lapply(1:G, function(g) which(groups == g))
}

compute_lambda_max_with_weights <- function(X, Y, group_idx, weights,
                                            n_knots = 5, multiplier = 5.0) {

  n <- nrow(X)
  p <- ncol(X)
  q <- ncol(Y)
  q_minus_1 <- q - 1

  beta_init <- rnorm(p)
  beta_init <- beta_init / sqrt(sum(beta_init^2))

  z <- as.vector(X %*% beta_init)
  knots <- build_knots(z, n_internal = n_knots, degree = 3)
  K <- length(knots) - 3 - 1

  Theta <- matrix(rnorm(K * q_minus_1, sd = 0.1), nrow = K, ncol = q_minus_1)

  grad <- grad_beta(beta_init, Theta, X, Y, knots, degree = 3)

  lambda_max <- 0
  for (g in seq_along(group_idx)) {
    idx <- group_idx[[g]]
    grad_g_norm <- sqrt(sum(grad[idx]^2))
    lambda_max <- max(lambda_max, grad_g_norm / weights[g])
  }

  if (!is.finite(lambda_max) || lambda_max <= 0) {
    lambda_max <- 0.1
  }

  return(lambda_max * multiplier)
}

compute_gamma_max <- function(X, Y, group_idx, n_knots = 5) {

  n <- nrow(X)
  p <- ncol(X)
  q <- ncol(Y)
  q_minus_1 <- q - 1

  beta_ref <- rnorm(p)
  beta_ref <- beta_ref / sqrt(sum(beta_ref^2))

  z <- as.vector(X %*% beta_ref)
  knots <- build_knots(z, n_internal = n_knots, degree = 3)
  K <- length(knots) - 3 - 1

  Omega <- compute_omega(knots, degree = 3)

  z_clip <- pmax(pmin(z, max(knots) - 1e-6), min(knots) + 1e-6)
  B_mat <- splineDesign(knots, z_clip, ord = 3 + 1)

  fn_Theta <- function(theta_vec) {
    Theta_mat <- matrix(theta_vec, nrow = K, ncol = q_minus_1)
    U <- B_mat %*% Theta_mat
    Y_hat <- inv_stereo(U)
    E <- Y - Y_hat
    mean(rowSums(E^2))
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

    grad_mat <- (2 / n) * t(B_mat) %*% S
    as.vector(grad_mat)
  }

  Theta_init <- matrix(rnorm(K * q_minus_1, sd = 0.1), nrow = K, ncol = q_minus_1)

  opt <- optim(
    par = as.vector(Theta_init),
    fn = fn_Theta,
    gr = gr_Theta,
    method = "L-BFGS-B",
    control = list(maxit = 50)
  )

  Theta_ref <- matrix(opt$par, nrow = K, ncol = q_minus_1)

  loss_ref <- opt$value
  roughness_ref <- sum(diag(t(Theta_ref) %*% Omega %*% Theta_ref))

  if (roughness_ref > 1e-10) {
    gamma_max <- loss_ref / roughness_ref
  } else {
    gamma_max <- 1.0
  }

  if (!is.finite(gamma_max) || gamma_max <= 0) {
    gamma_max <- 1.0
  }

  gamma_max <- min(gamma_max, 10.0)

  return(gamma_max)
}

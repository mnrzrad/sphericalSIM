
build_knots <- function(z, n_internal = 10, degree = 3) {
  z_range <- range(z)
  if (n_internal > 0) {
    probs <- seq(0, 1, length.out = n_internal + 2)[-c(1, n_internal + 2)]
    internal <- quantile(z, probs)
  } else {
    internal <- numeric(0)
  }
  c(rep(z_range[1], degree + 1), internal, rep(z_range[2], degree + 1))
}

compute_omega <- function(knots, degree = 3, n_grid = 500) {
  K <- length(knots) - degree - 1
  z_min <- min(knots)
  z_max <- max(knots)
  grid <- seq(z_min, z_max, length.out = n_grid)
  h <- grid[2] - grid[1]

  B2 <- splineDesign(knots, grid, ord = degree + 1, derivs = 2)

  Omega <- (t(B2) %*% B2) * h
  Omega <- Omega - 0.5 * h * (outer(B2[1, ], B2[1, ]) + outer(B2[n_grid, ], B2[n_grid, ]))

  Omega
}

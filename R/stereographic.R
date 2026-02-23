#' Inverse Stereographic Projection
#'
#' Maps points from Euclidean space R^(q-1) to the unit sphere S^(q-1) in R^q
#' using the inverse stereographic projection from the north pole.
#'
#' @param u Vector or matrix of points in R^(q-1). If a vector, it is treated as
#'   a single point. If a matrix, each row represents a point.
#'
#' @details
#' The inverse stereographic projection π^(-1): R^(q-1) → S^(q-1) is defined as:
#' \deqn{π^(-1)(u) = \frac{1}{1 + ||u||^2} (2u_1, 2u_2, ..., 2u_{q-1}, 1 - ||u||^2)}
#'
#' This maps the entire Euclidean space R^(q-1) onto the unit sphere S^(q-1),
#' excluding the north pole (0, 0, ..., 0, 1). The projection is from the north
#' pole, meaning:
#' \itemize{
#'   \item The origin u = 0 maps to the south pole (0, 0, ..., 0, -1)
#'   \item Points near the origin map to the southern hemisphere
#'   \item Points far from the origin approach the north pole
#' }
#'
#' @return Matrix of dimension n × q, where each row is a point on the unit sphere
#'   in R^q. Each row satisfies ||π^(-1)(u)|| = 1.
#'
#' @examples
#' # Single point: origin maps to south pole
#' inv_stereo(c(0, 0))  # Returns approximately [0, 0, -1]
#'
#' # Multiple points
#' u_mat <- matrix(c(0, 0, 1, 0, 0, 1), nrow = 3, byrow = TRUE)
#' y <- inv_stereo(u_mat)
#' rowSums(y^2)  # Verify all points are on unit sphere (all ~1)
#'
#' # Point on circle in R^2 maps to curve on S^2
#' theta <- seq(0, 2*pi, length.out = 100)
#' u_circle <- cbind(cos(theta), sin(theta))
#' y_sphere <- inv_stereo(u_circle)
#'
#' @seealso \code{\link{jacobian_inv_stereo}} for the Jacobian matrix
#'
#' @export
inv_stereo <- function(u) {
  # Use C++ version if enabled
  if (getOption("sphericalSIM.use_cpp", TRUE)) {
    if (is.vector(u)) u <- matrix(u, nrow = 1)
    return(inv_stereo_cpp(u))
  }

  # Original R implementation
  if (is.vector(u)) u <- matrix(u, nrow = 1)
  u_norm_sq <- rowSums(u^2)
  denom <- 1 + u_norm_sq
  cbind(2 * u / denom, (1 - u_norm_sq) / denom)
}


#' Jacobian of Inverse Stereographic Projection
#'
#' Computes the Jacobian matrix of the inverse stereographic projection at a
#' given point u in R^(q-1).
#'
#' @param u Vector of length (q-1) representing a point in Euclidean space.
#'
#' @details
#' For the inverse stereographic projection π^(-1): R^(q-1) → S^(q-1), the
#' Jacobian J(u) is a q × (q-1) matrix where each column j contains the partial
#' derivatives with respect to u_j:
#'
#' For i = 1, ..., q-1:
#' \deqn{∂π_i^(-1)/∂u_j = \frac{2δ_{ij}}{1 + ||u||^2} - \frac{4u_i u_j}{(1 + ||u||^2)^2}}
#'
#' For i = q (the last component):
#' \deqn{∂π_q^(-1)/∂u_j = -\frac{4u_j}{(1 + ||u||^2)^2}}
#'
#' where δ_{ij} is the Kronecker delta (1 if i=j, 0 otherwise).
#'
#' The Jacobian is used in:
#' \itemize{
#'   \item Change-of-variables calculations
#'   \item Computing derivatives of composed functions via chain rule
#'   \item Gradient-based optimization on spherical manifolds
#' }
#'
#' @return Matrix of dimension q × (q-1) containing the Jacobian. Each column j
#'   contains the partial derivatives of π^(-1)(u) with respect to u_j.
#'
#' @examples
#' # Jacobian at origin
#' J0 <- jacobian_inv_stereo(c(0, 0))
#' print(J0)  # Simple diagonal-like structure at origin
#'
#' # Jacobian at non-origin point
#' u <- c(1, 1)
#' J <- jacobian_inv_stereo(u)
#'
#' # Verify dimensions
#' q_minus_1 <- length(u)
#' q <- q_minus_1 + 1
#' dim(J)  # Should be [q, q-1]
#'
#' # Numerical verification using finite differences
#' \dontrun{
#' library(numDeriv)
#' u_test <- c(0.5, 0.5)
#'
#' # Our analytical Jacobian
#' J_analytical <- jacobian_inv_stereo(u_test)
#'
#' # Numerical Jacobian
#' J_numerical <- jacobian(inv_stereo, u_test)
#'
#' # Compare
#' max(abs(J_analytical - J_numerical))  # Should be very small
#' }
#'
#' @seealso \code{\link{inv_stereo}} for the inverse stereographic projection
#'
#' @export
jacobian_inv_stereo <- function(u) {
  # Use C++ version if enabled
  if (getOption("sphericalSIM.use_cpp", TRUE)) {
    return(jacobian_inv_stereo_cpp(u))
  }

  # Original R implementation
  q_minus_1 <- length(u)
  q <- q_minus_1 + 1
  u_norm_sq <- sum(u^2)
  denom <- 1 + u_norm_sq
  denom_sq <- denom^2

  J <- matrix(0, nrow = q, ncol = q_minus_1)

  for (i in 1:q_minus_1) {
    for (j in 1:q_minus_1) {
      J[i, j] <- 2 * (i == j) / denom - 4 * u[i] * u[j] / denom_sq
    }
  }

  J[q, ] <- -4 * u / denom_sq

  J
}

#' Generate Synthetic Data for Spherical Single-Index Model
#'
#' Generates synthetic data from a spherical single-index model with group structure
#' in the predictors. The response lies on the unit sphere and follows a single-index
#' structure with a known link function.
#'
#' @param n Sample size (default 500)
#' @param p Total number of predictors (default 100)
#' @param q Dimension of the response sphere (default 3, i.e., unit sphere in R^3)
#' @param G Total number of groups (default 10)
#' @param active_groups Vector of indices indicating which groups have non-zero coefficients (default c(1, 3, 5))
#' @param kappa Concentration parameter controlling noise level; higher values mean less noise (default 50)
#' @param rho Correlation parameter for auto-regressive covariance structure in X (default 0.5)
#' @param mean_beta_true Mean for generating non-zero coefficients in active groups (default 0.5)
#' @param sd_beta_true Standard deviation for generating non-zero coefficients in active groups (default 0.2)
#' @param seed Random seed for reproducibility. If NULL, no seed is set (default NULL)
#'
#' @details
#' The function generates data according to the spherical single-index model:
#' \deqn{Y_i = m(X_i^T β) + ε_i}
#' where:
#' \itemize{
#'   \item Y_i ∈ S^{q-1} (unit sphere in R^q)
#'   \item X_i ∈ R^p with auto-regressive correlation structure
#'   \item β ∈ R^p is the index parameter with ||β|| = 1
#'   \item m: R → S^{q-1} is the link function
#'   \item ε_i is spherical noise with concentration κ
#' }
#'
#' \strong{Data Generation Process:}
#' \enumerate{
#'   \item \strong{Group Structure}: Divides p predictors into G groups as evenly as possible
#'   \item \strong{Index Parameter}: For active groups, coefficients are drawn from
#'         N(mean_beta_true, sd_beta_true^2). Inactive groups have zero coefficients.
#'         The vector is normalized: β = β_raw / ||β_raw||
#'   \item \strong{Predictors}: X ~ N(0, Σ) where Σ_{ij} = ρ^{|i-j|} (AR(1) structure)
#'   \item \strong{Single Index}: z_i = X_i^T β
#'   \item \strong{Link Function}: m(z) = [sin(z), 0.5*cos(z)]^T, mapped to S^{q-1}
#'         via inverse stereographic projection
#'   \item \strong{Noise}: Spherical noise is added by perturbing with N(0, 1/κ)^q
#'         and re-normalizing to the unit sphere. Higher κ gives more concentrated
#'         responses (less noise)
#' }
#'
#' \strong{True Link Function:}
#' The true link function is m(z) = [sin(z), 0.5*cos(z)]^T in R^{q-1}, which is
#' then mapped to S^{q-1} using the inverse stereographic projection. This creates
#' a smooth curve on the sphere.
#'
#' \strong{Group Structure:}
#' Groups are created by dividing predictors as evenly as possible. If p is not
#' divisible by G, the first (p mod G) groups will have one extra predictor.
#'
#' @return A list containing:
#' \describe{
#'   \item{X}{Design matrix (n × p) with AR(1) correlation structure}
#'   \item{Y}{Response matrix (n × q) where each row is a unit vector on S^{q-1}}
#'   \item{beta_true}{True index parameter vector (length p, normalized to unit length)}
#'   \item{z_true}{Single index values X %*% beta_true (length n)}
#'   \item{groups}{Vector of length p indicating group membership for each predictor}
#'   \item{active_groups}{Vector of active group indices (groups with non-zero coefficients)}
#'   \item{G}{Total number of groups}
#'   \item{group_idx}{List of length G, where each element contains indices of predictors in that group}
#'   \item{mu_true}{Clean mean responses before noise addition (n × (q-1) matrix in R^{q-1})}
#' }
#'
#' @examples
#' \dontrun{
#' # Generate data with default settings
#' data <- generate_spherical_data(seed = 123)
#'
#' # Check that Y values are on unit sphere
#' all(abs(rowSums(data$Y^2) - 1) < 1e-10)  # TRUE
#'
#' # Verify beta is normalized
#' sum(data$beta_true^2)  # Should be 1
#'
#' # Check which groups are active
#' data$active_groups  # c(1, 3, 5)
#'
#' # Visualize group structure
#' plot(data$beta_true, col = data$groups, pch = 19,
#'      xlab = "Predictor Index", ylab = "Coefficient",
#'      main = "True Coefficients by Group")
#' abline(h = 0, lty = 2, col = "gray")
#'
#' # Custom data generation
#' data <- generate_spherical_data(
#'   n = 200,           # Smaller sample
#'   p = 50,            # Fewer predictors
#'   G = 5,             # 5 groups
#'   active_groups = c(1, 2),  # Only 2 active groups
#'   kappa = 100,       # Less noise
#'   rho = 0.8,         # Higher correlation
#'   seed = 456
#' )
#'
#' # Verify group sizes
#' table(data$groups)
#'
#' # Plot first two dimensions of responses
#' plot(data$Y[,1], data$Y[,2], asp = 1,
#'      col = rainbow(10)[cut(data$z_true, 10)],
#'      pch = 19, cex = 0.5,
#'      main = "Spherical Response (first 2 dimensions)")
#'
#' # Examine relationship between single index and response
#' par(mfrow = c(1, 2))
#' plot(data$z_true, data$Y[,1], pch = 19, cex = 0.5,
#'      xlab = "Single Index z", ylab = "Y[,1]")
#' plot(data$z_true, data$Y[,2], pch = 19, cex = 0.5,
#'      xlab = "Single Index z", ylab = "Y[,2]")
#'
#' # Generate data for different noise levels
#' data_high_noise <- generate_spherical_data(kappa = 10, seed = 100)
#' data_low_noise <- generate_spherical_data(kappa = 100, seed = 100)
#'
#' # Compare noise levels visually
#' par(mfrow = c(1, 2))
#' plot(data_high_noise$Y[,1], data_high_noise$Y[,2],
#'      main = "High Noise (kappa=10)", asp = 1)
#' plot(data_low_noise$Y[,1], data_low_noise$Y[,2],
#'      main = "Low Noise (kappa=100)", asp = 1)
#' }
#'
#' @references
#' The von Mises-Fisher distribution and spherical single-index models:
#' - Mardia, K. V., & Jupp, P. E. (2000). Directional Statistics. Wiley.
#' - Ding, S., & Cook, R. D. (2018). Matrix variate regressions and envelope models.
#'   Journal of the Royal Statistical Society: Series B, 80(2), 387-408.
#'
#' @seealso
#' \code{\link{inv_stereo}} for inverse stereographic projection,
#' \code{create_group_indices} for group structure creation,
#' \code{\link{run_single_simulation}} for using generated data in simulations,
#' \code{\link{spherical_sim_group}} for fitting models to generated data
#'
#' @importFrom MASS mvrnorm
#' @export
generate_spherical_data <- function(n = 500, p = 100, q = 3, G = 10,
                                    active_groups = c(1, 3, 5),
                                    kappa = 50, rho = 0.5,
                                    mean_beta_true = 0.5,
                                    sd_beta_true = 0.2,
                                    seed = NULL) {

  if (!is.null(seed)) set.seed(seed)

  pg <- rep(floor(p/G), G)
  remainder <- p %% G
  if (remainder > 0) pg[1:remainder] <- pg[1:remainder] + 1
  groups <- rep(1:G, times = pg)
  group_idx <- create_group_indices(groups)

  beta_true <- rep(0, p)
  for (g in active_groups) {
    idx <- group_idx[[g]]
    beta_true[idx] <- rnorm(length(idx), mean = mean_beta_true, sd = sd_beta_true)
  }
  beta_true <- beta_true / sqrt(sum(beta_true^2))

  Sigma <- matrix(rho^abs(outer(1:p, 1:p, "-")), p, p)
  X <- mvrnorm(n, mu = rep(0, p), Sigma = Sigma)

  z_true <- as.vector(X %*% beta_true)
  U_true <- do.call(rbind, lapply(z_true, function(x) c(sin(x), 0.5*cos(x))))
  Y_clean <- inv_stereo(U_true)

  Y <- Y_clean
  for (i in 1:n) {
    noise <- rnorm(q) / sqrt(kappa)
    y_perturbed <- Y_clean[i, ] + noise
    Y[i, ] <- y_perturbed / sqrt(sum(y_perturbed^2))
  }

  list(
    X = X, Y = Y, beta_true = beta_true, z_true = z_true,
    groups = groups, active_groups = active_groups, G = G,
    group_idx = group_idx, mu_true = U_true
  )
}

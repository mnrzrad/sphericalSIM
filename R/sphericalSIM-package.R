#' sphericalSIM: Spherical Single-Index Models
#'
#' @description
#' Spherical single-index models with adaptive group lasso variable selection.
#'
#' @keywords internal
#' @importFrom stats cor median optim quantile rnorm sd
#' @importFrom splines splineDesign  
#' @importFrom MASS mvrnorm
#' @importFrom parallel detectCores mclapply
#' @importFrom Rcpp sourceCpp
#' @useDynLib sphericalSIM, .registration = TRUE
"_PACKAGE"

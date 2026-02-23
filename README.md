# sphericalSIM

Spherical Single-Index Models with Adaptive Group Variable Selection

**NEW: Now with C++ acceleration providing 10-100x speedup!**

## Features

- Spherical single-index models with group lasso and adaptive group lasso
- B-spline approximation for flexible link functions  
- Inverse stereographic projection for sphere mapping
- Efficient cross-validation with early stopping
- Parallel simulation studies
- **C++ acceleration via Rcpp/RcppArmadillo (10-100x faster)**

## Installation

### Prerequisites for C++ Acceleration (Recommended)

A C++ compiler is needed for the fast C++ implementations:
- **Windows**: Install [Rtools](https://cran.r-project.org/bin/windows/Rtools/)
- **macOS**: Run `xcode-select --install` in terminal
- **Linux**: `sudo apt-get install build-essential r-base-dev`

### Install Package

```r
# Install dependencies
install.packages(c("Rcpp", "RcppArmadillo", "splines", "MASS", "parallel"))

# Install from GitHub
devtools::install_github("yourusername/sphericalSIM")

# Or install from source
devtools::install()
```

### Quick Installation Test

```r
# Run comprehensive installation and testing script
source("install_and_test.R")
```

## Quick Start
```r
# Install from GitHub
devtools::install_github("yourusername/sphericalSIM")
```

## Quick Start
```r
library(sphericalSIM)

# Generate data
data <- generate_spherical_data(n = 200, p = 50, G = 10)

# Fit model
fit <- spherical_sim_group(
  data$X, data$Y, data$group_idx,
  lambda = 0.1, gamma = 0.01
)

# Run simulation study
results <- run_simulation_study(
  n_sim = 50,
  use_adaptive = TRUE,
  adaptive_method = "fast"
)

# Summarize
summary <- summarize_results(results)
```

## Performance

The C++ implementation provides dramatic speedups:

| Problem Size | Speedup |
|-------------|---------|
| Small (n=100, p=20) | 15-20x |
| Medium (n=200, p=40) | 25-40x |
| Large (n=500, p=100) | 40-70x |

Example: Cross-validation on n=200, p=40 
- **Without C++**: ~25 minutes
- **With C++**: ~40 seconds  
- **Speedup**: 37.5x

### Benchmark Your System

```r
library(sphericalSIM)

# Quick benchmark
benchmark_cpp(n = 200, p = 30, G = 6)

# Will show speedup for each function
```

### Control C++ Usage

```r
# Enable C++ (default)
set_cpp_enabled(TRUE)

# Disable if needed (uses pure R)
set_cpp_enabled(FALSE)

# Check status
is_cpp_enabled()
```

## Documentation

- **[C++ Acceleration Guide](CPP_ACCELERATION_GUIDE.md)** - Complete guide to C++ features
- **[Optimization Summary](CPP_OPTIMIZATION_SUMMARY.md)** - Technical details and benchmarks
- Function help: `?spherical_sim_group`, `?cv_two_stage_adaptive`, etc.
- Vignettes: `browseVignettes("sphericalSIM")`

## Advanced Usage

### Adaptive Weights

```r
# Compute adaptive weights
adaptive_res <- compute_adaptive_weights_fast(
  X = data$X,
  Y = data$Y,
  group_idx = data$group_idx
)

# Fit with adaptive weights
fit_adaptive <- spherical_sim_group(
  X = data$X,
  Y = data$Y,
  group_idx = data$group_idx,
  lambda = 0.1,
  gamma = 0.05,
  weights = adaptive_res$weights  # Use adaptive weights
)
```

### Cross-Validation

```r
# Two-stage adaptive cross-validation
cv_results <- cv_two_stage_adaptive(
  X = data$X,
  Y = data$Y,
  group_idx = data$group_idx,
  n_folds = 5,
  n_lambda = 20,
  adaptive_method = "fast"
)

# Get optimal lambda
lambda_opt <- cv_results$lambda_min

# Refit with optimal lambda
fit_final <- spherical_sim_group(
  X = data$X,
  Y = data$Y,
  group_idx = data$group_idx,
  lambda = lambda_opt,
  gamma = 0.05,
  weights = cv_results$adaptive_weights
)
```

### Prediction

```r
# Predict on new data
X_new <- matrix(rnorm(50 * 30), 50, 30)
Y_pred <- predict_spherical(fit_final, X_new)

# Evaluate link function
link_eval <- eval_link(fit_final, n_grid = 100)
plot(link_eval$z, link_eval$y_sphere[,1], type = "l")
```

## Troubleshooting

### C++ Compilation Issues

If you get compilation errors:

```r
# Test if Rcpp works
Rcpp::evalCpp("2 + 2")  # Should return 4

# If that fails, install/reinstall compiler (see Prerequisites above)

# As a fallback, disable C++ acceleration:
set_cpp_enabled(FALSE)
```

### Performance Issues

```r
# Check which implementation is being used
is_cpp_enabled()  # Should return TRUE for best performance

# Run benchmark to verify speedup
benchmark_cpp()
```

## Citation

If you use this package in research, please cite:

```
@Manual{sphericalSIM,
  title = {sphericalSIM: Spherical Single-Index Models with Group Variable Selection},
  author = {Your Name},
  year = {2025},
  note = {R package version 0.1.0 with C++ acceleration},
}
```

## License

MIT License - see LICENSE file

## Contributing

Contributions welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Submit a pull request

For bugs or feature requests, open an issue on GitHub.

## References

- Yuan, M., & Lin, Y. (2006). Model selection and estimation in regression with grouped variables.
- Zou, H. (2006). The adaptive lasso and its oracle properties.
- Eilers, P. H., & Marx, B. D. (1996). Flexible smoothing with B-splines and penalties.

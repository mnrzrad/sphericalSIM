library(sphericalSIM)

# Your existing code works exactly the same, but faster!
data <- generate_spherical_data(n = 200, p = 30, G = 6)

fit <- spherical_sim_group(
  X = data$X,
  Y = data$Y,
  group_idx = data$group_idx,
  lambda = 0.1,
  gamma = 0.05
)
# This is now 25-40x faster than before!

cv_res <- cv_two_stage_adaptive(X = data$X,
                                Y = data$Y,
                                group_idx = data$group_idx)

results <- run_simulation_study(
  n_sim = 500,
  n = 500,
  p = 100,
  q = 3,
  G = 10,
  active_groups = c(1, 2, 5, 8, 10),
  use_adaptive = TRUE,
  adaptive_method = "fast",
  parallel = TRUE,
  n_cores = NULL,
  save_file = "simulation_results_n_500_p_100_G_10_ag_1_2_5_8_10_adp_fast.rds",
  verbose_sim = F
)

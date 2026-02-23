#include <R.h>
#include <Rinternals.h>
#include <stdlib.h> // for NULL
#include <R_ext/Rdynload.h>

/* .Call calls */
extern SEXP _sphericalSIM_inv_stereo_cpp(SEXP);
extern SEXP _sphericalSIM_jacobian_inv_stereo_cpp(SEXP);
extern SEXP _sphericalSIM_prox_group_lasso_cpp(SEXP, SEXP, SEXP, SEXP, SEXP);
extern SEXP _sphericalSIM_compute_group_penalty_cpp(SEXP, SEXP, SEXP, SEXP);
extern SEXP _sphericalSIM_grad_beta_cpp(SEXP, SEXP, SEXP, SEXP, SEXP, SEXP);
extern SEXP _sphericalSIM_compute_objective_cpp(SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP);
extern SEXP _sphericalSIM_compute_group_norms_cpp(SEXP, SEXP);
extern SEXP _sphericalSIM_compute_prediction_cpp(SEXP, SEXP);
extern SEXP _sphericalSIM_compute_adaptive_weights_from_beta(SEXP, SEXP, SEXP, SEXP);
extern SEXP _sphericalSIM_proximal_gradient_step(SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP);

static const R_CallMethodDef CallEntries[] = {
    {"_sphericalSIM_inv_stereo_cpp", (DL_FUNC) &_sphericalSIM_inv_stereo_cpp, 1},
    {"_sphericalSIM_jacobian_inv_stereo_cpp", (DL_FUNC) &_sphericalSIM_jacobian_inv_stereo_cpp, 1},
    {"_sphericalSIM_prox_group_lasso_cpp", (DL_FUNC) &_sphericalSIM_prox_group_lasso_cpp, 5},
    {"_sphericalSIM_compute_group_penalty_cpp", (DL_FUNC) &_sphericalSIM_compute_group_penalty_cpp, 4},
    {"_sphericalSIM_grad_beta_cpp", (DL_FUNC) &_sphericalSIM_grad_beta_cpp, 6},
    {"_sphericalSIM_compute_objective_cpp", (DL_FUNC) &_sphericalSIM_compute_objective_cpp, 10},
    {"_sphericalSIM_compute_group_norms_cpp", (DL_FUNC) &_sphericalSIM_compute_group_norms_cpp, 2},
    {"_sphericalSIM_compute_prediction_cpp", (DL_FUNC) &_sphericalSIM_compute_prediction_cpp, 2},
    {"_sphericalSIM_compute_adaptive_weights_from_beta", (DL_FUNC) &_sphericalSIM_compute_adaptive_weights_from_beta, 4},
    {"_sphericalSIM_proximal_gradient_step", (DL_FUNC) &_sphericalSIM_proximal_gradient_step, 11},
    {NULL, NULL, 0}
};

void R_init_sphericalSIM(DllInfo *dll)
{
    R_registerRoutines(dll, NULL, CallEntries, NULL, NULL);
    R_useDynamicSymbols(dll, FALSE);
}

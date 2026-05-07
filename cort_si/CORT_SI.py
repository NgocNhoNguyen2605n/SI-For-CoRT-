import random
from joblib import Parallel, delayed
import multiprocessing
import numpy as np
from threadpoolctl import threadpool_limits

try:
    from . import sub_prob
    from . import algorithms
    from . import utils
except ImportError:  # pragma: no cover - fallback for direct script execution
    import sub_prob
    import algorithms
    import utils



def SI(X0, Y0, XS_list, YS_list, lambda_sel, lambda0, lambdak_list, SigmaS_list, Sigma0, folds=None, T=5, z_min=-20, z_max=20, verbose=False):
    if len(XS_list) != len(YS_list):
        raise ValueError("XS_list and YS_list must have the same length")
    if len(lambdak_list) != len(XS_list):
        raise ValueError("lambdak_list must have the same length as XS_list")

    if folds is None:
        folds = utils.construct_folds(X0.shape[0], T=T, shuffle=False)
    if len(folds) % 2 == 0:
        raise ValueError("The number of folds T must be odd")

    I_obs = algorithms.adaptive_source_selection(X0, Y0, XS_list, YS_list, folds, lambda_sel, verbose=verbose)
    _, beta0_hat, _, _ = algorithms.solve_cort_model(
        X0, Y0, XS_list, YS_list, I_obs, lambda0, lambdak_list,
        verbose=verbose, label="Observed CoRT fit",
    )
    M_obs = [idx for idx, value in enumerate(beta0_hat) if value != 0]

    if verbose:
        print(f"observed informative source set = {I_obs}")
        print(f"observed target active set = {M_obs}")

    if not M_obs:
        return None

    Y_obs = utils.construct_Y(YS_list, Y0)
    Sigma = utils.construct_Sigma(SigmaS_list, Sigma0)
    X0M = X0[:, M_obs]
    print(f"length of M_obs: {len(M_obs)}")
    p_values = []
    for j in M_obs:
        print(f"feature j: {j}")
        etaj, etajTy = utils.construct_test_statistic(j, X0M, Y_obs, M_obs, Y0.shape[0], Y_obs.shape[0])
        a, b = utils.calculate_a_b(etaj, Y_obs, Sigma, Y_obs.shape[0])

        intervals_source = sub_prob.compute_Z1_region(X0, Y0, XS_list, YS_list, a, b, folds, I_obs, lambda_sel, z_min, z_max)
        intervals_model = sub_prob.compute_Z2_region(
            X0, Y0, XS_list, YS_list, a, b, I_obs, M_obs,
            intervals_source, lambda0, lambdak_list,
        )
        
        intervals = utils.intersect_interval_unions(intervals_source, intervals_model)
        print(f"length of intervals: {len(intervals)}")
        for left, right in intervals:
            print(left, right)
        p_sel = None
        if intervals:
            p_sel = utils.calculate_TN_p_value(intervals, etaj, etajTy, Sigma, 0.0)
        p_values.append((j, p_sel))

    return p_values

def SI_parallel(X0, Y0, XS_list, YS_list, lambda_sel, lambda0, lambdak_list, SigmaS_list, Sigma0, folds=None, T=5, z_min=-20, z_max=20, verbose=False):
    if len(XS_list) != len(YS_list):
        raise ValueError("XS_list and YS_list must have the same length")
    if len(lambdak_list) != len(XS_list):
        raise ValueError("lambdak_list must have the same length as XS_list")

    if folds is None:
        folds = utils.construct_folds(X0.shape[0], T=T, shuffle=False)
    if len(folds) % 2 == 0:
        raise ValueError("The number of folds T must be odd")

    I_obs = algorithms.adaptive_source_selection(X0, Y0, XS_list, YS_list, folds, lambda_sel, verbose=verbose)
    _, beta0_hat, _, _ = algorithms.solve_cort_model(
        X0, Y0, XS_list, YS_list, I_obs, lambda0, lambdak_list,
        verbose=verbose, label="Observed CoRT fit",
    )
    M_obs = [idx for idx, value in enumerate(beta0_hat) if value != 0]

    if verbose:
        print(f"observed informative source set = {I_obs}")
        print(f"observed target active set = {M_obs}")

    if not M_obs:
        return None

    Y_obs = utils.construct_Y(YS_list, Y0)
    Sigma = utils.construct_Sigma(SigmaS_list, Sigma0)
    X0M = X0[:, M_obs]
    print(f"length of M_obs: {len(M_obs)}")
    p_values = []
    for j in M_obs:
        print(f"feature j: {j}")
        etaj, etajTy = utils.construct_test_statistic(j, X0M, Y_obs, M_obs, Y0.shape[0], Y_obs.shape[0])
        a, b = utils.calculate_a_b(etaj, Y_obs, Sigma, Y_obs.shape[0])
        S = multiprocessing.cpu_count() - 1
        z_points = np.linspace(z_min, z_max, S + 1)
        sub_intervals = [(z_points[i], z_points[i+1]) for i in range(S)]
        parallel_results_source = Parallel(n_jobs=S)(
            delayed(sub_prob.compute_Z1_region)(
                X0, Y0, XS_list, YS_list, a, b, folds, I_obs, lambda_sel, z_start, z_end
            ) for z_start, z_end in sub_intervals
        )
        intervals_source = [interval for sublist in parallel_results_source if sublist for interval in sublist]
        intervals_model = sub_prob.compute_Z2_region(
            X0, Y0, XS_list, YS_list, a, b, I_obs, M_obs, 
            intervals_source, lambda0, lambdak_list
        )
        intervals = utils.intersect_interval_unions(intervals_source, intervals_model)
        print(f"length of intervals: {len(intervals)}")
        for left, right in intervals:
            print(left, right)
        p_sel = None
        if intervals:
            p_sel = utils.calculate_TN_p_value(intervals, etaj, etajTy, Sigma, 0.0)
        p_values.append((j, p_sel))

    return p_values


def SI_randj(X0, Y0, XS_list, YS_list, lambda_sel, lambda0, lambdak_list, SigmaS_list, Sigma0, folds=None, T=5, z_min=-20, z_max=20, verbose=False):
    if len(XS_list) != len(YS_list):
        raise ValueError("XS_list and YS_list must have the same length")
    if len(lambdak_list) != len(XS_list):
        raise ValueError("lambdak_list must have the same length as XS_list")
    if folds is None:
        folds = utils.construct_folds(X0.shape[0], T=T, shuffle=False)
    if len(folds) % 2 == 0:
        raise ValueError("The number of folds T must be odd")
    I_obs = algorithms.adaptive_source_selection(X0, Y0, XS_list, YS_list, folds, lambda_sel, verbose=verbose)
    _, beta0_hat, _, _ = algorithms.solve_cort_model(
        X0, Y0, XS_list, YS_list, I_obs, lambda0, lambdak_list,
        verbose=verbose, label="Observed CoRT fit",
    )
    M_obs = [idx for idx, value in enumerate(beta0_hat) if value != 0]
    print(f"length of M_obs: {len(M_obs)}")
    if verbose:
        print(f"observed informative source set = {I_obs}")
        print(f"observed target active set = {M_obs}")
    if not M_obs:
        return None

    Y_obs = utils.construct_Y(YS_list, Y0)
    Sigma = utils.construct_Sigma(SigmaS_list, Sigma0)
    X0M = X0[:, M_obs]
    j = random.choice(M_obs)

    etaj, etajTy = utils.construct_test_statistic(j, X0M, Y_obs, M_obs, Y0.shape[0], Y_obs.shape[0])
    a, b = utils.calculate_a_b(etaj, Y_obs, Sigma, Y_obs.shape[0])
    intervals_source = sub_prob.compute_Z1_region(X0, Y0, XS_list, YS_list, a, b, folds, I_obs, lambda_sel, z_min, z_max)
    intervals_model = sub_prob.compute_Z2_region(
        X0, Y0, XS_list, YS_list, a, b, I_obs, M_obs, 
        intervals_source, lambda0, lambdak_list
    )
    intervals = utils.intersect_interval_unions(intervals_source, intervals_model)
    print(f"length of intervals: {len(intervals)}, length of intervals_model: {len(intervals_model)}")
    if not intervals:
        return None

    p_sel = utils.calculate_TN_p_value(intervals, etaj, etajTy, Sigma, 0.0)
    return j, p_sel

def SI_parallel_randj(X0, Y0, XS_list, YS_list, lambda_sel, lambda0, lambdak_list, SigmaS_list, Sigma0, folds=None, T=5, z_min=-20, z_max=20, verbose=False):
    if len(XS_list) != len(YS_list):
        raise ValueError("XS_list and YS_list must have the same length")
    if len(lambdak_list) != len(XS_list):
        raise ValueError("lambdak_list must have the same length as XS_list")

    if folds is None:
        folds = utils.construct_folds(X0.shape[0], T=T, shuffle=False)
    if len(folds) % 2 == 0:
        raise ValueError("The number of folds T must be odd")

    I_obs = algorithms.adaptive_source_selection(X0, Y0, XS_list, YS_list, folds, lambda_sel, verbose=verbose)
    _, beta0_hat, _, _ = algorithms.solve_cort_model(
        X0, Y0, XS_list, YS_list, I_obs, lambda0, lambdak_list,
        verbose=verbose, label="Observed CoRT fit",
    )
    M_obs = [idx for idx, value in enumerate(beta0_hat) if value != 0]
    # print(f"length of M_obs: {len(M_obs)}")
    if verbose:
        print(f"observed informative source set = {I_obs}")
        print(f"observed target active set = {M_obs}")

    if not M_obs:
        return None

    Y_obs = utils.construct_Y(YS_list, Y0)
    Sigma = utils.construct_Sigma(SigmaS_list, Sigma0)
    X0M = X0[:, M_obs]
    j = random.choice(M_obs)
    etaj, etajTy = utils.construct_test_statistic(j, X0M, Y_obs, M_obs, Y0.shape[0], Y_obs.shape[0])
    a, b = utils.calculate_a_b(etaj, Y_obs, Sigma, Y_obs.shape[0])
    S = multiprocessing.cpu_count() - 1
    z_points = np.linspace(z_min, z_max, S + 1)
    sub_intervals = [(z_points[i], z_points[i+1]) for i in range(S)]
    with threadpool_limits(limits=1, user_api='blas'):
        parallel_results_source = Parallel(n_jobs=S)(
            delayed(sub_prob.compute_Z1_region)(
                X0, Y0, XS_list, YS_list, a, b, folds, I_obs, lambda_sel, z_start, z_end
            ) for z_start, z_end in sub_intervals
        )
    intervals_source = [interval for sublist in parallel_results_source if sublist for interval in sublist]
    intervals_model = sub_prob.compute_Z2_region(
        X0, Y0, XS_list, YS_list, a, b, I_obs, M_obs, 
        intervals_source, lambda0, lambdak_list
    )
    intervals = utils.intersect_interval_unions(intervals_source, intervals_model)
    if not intervals:
        return None

    p_sel = utils.calculate_TN_p_value(intervals, etaj, etajTy, Sigma, 0.0)
    return j, p_sel
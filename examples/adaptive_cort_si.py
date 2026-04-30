import pathlib
import sys

import numpy as np

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from cort_si import SI, gen_data


def run_adaptive_cort_si():
    np.random.seed(0)

    XS_list, YS_list, X0, Y0, _, SigmaS_list, Sigma0, _ = gen_data.generate_data(
        p=5,
        s=2,
        nS=6,
        nT=7,
        true_beta=1.0,
        num_info_aux=1,
        num_uninfo_aux=1,
        gamma=0.05,
    )

    p_values = SI(
        X0,
        Y0,
        XS_list,
        YS_list,
        lambda_sel=0.05,
        lambda0=0.05,
        lambdak_list=[0.05] * len(XS_list),
        SigmaS_list=SigmaS_list,
        Sigma0=Sigma0,
        T=3,
        z_min=-5,
        z_max=5,
    )

    print("Adaptive CoRT-SI p-values:")
    print(p_values)


if __name__ == "__main__":
    run_adaptive_cort_si()
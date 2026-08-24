import numpy as np
from pathlib import Path
import pickle
import yaml
import sys
from os.path import exists
import matplotlib.pyplot as plt
from bootstrap import bootstrap
import analysis.stats as stats
from analysis.formatting import err_brackets
from analysis import fitfunc as ff

_metadata = {"Author": "Mischa Batelaan", "Creator": __file__}
_colors = [
    "#377eb8",
    "#4daf4a",
    "#f781bf",
    "#a65628",
    "#ff7f00",
    "#984ea3",
    "#999999",
    "#e41a1c",
    "#dede00",
]
_markers = ["s", "o", "^", "*", "v", ">", "<", "s", "s"]


def main():
    """
    Read the fit results to the 3pt ratios at qsq=0
    Use these to extract the value of f^0_{K pi}(0)
    """

    mystyle = Path("mystyle.txt")
    plt.style.use(mystyle.as_posix())
    plt.rc("text.latex", preamble=r"\usepackage{physics}")
    plotdir = Path("./plots/twisted/")
    datadir = Path("./plots/twisted/data/")
    datadir_tau13 = Path("./data/twisted_tau13/")

    nboot = 500
    nbin = 1

    # ======================================================================
    # Open two-point function data:
    twopt_datadir = Path(
        "/Users/mbatelaan/Research/Adelaide2026/analysis/six_point_fn/data/pickles/run1_meson/"
    )
    with open(twopt_datadir / (f"time_window_loop_pion_Cosh.pkl"), "rb") as file_in:
        fitlist_pion_cosh = pickle.load(file_in)
    with open(twopt_datadir / (f"time_window_loop_kaon_Cosh.pkl"), "rb") as file_in:
        fitlist_kaon_cosh = pickle.load(file_in)

    weights_pion = np.array([i["weight"] for i in fitlist_pion_cosh])
    energies_pion = np.array([i["param"][:, 1] for i in fitlist_pion_cosh])
    pion_mass = np.dot(weights_pion, energies_pion)
    # high_weight_pion = np.argmax(weights_pion)
    # pion_mass = fitlist_pion_cosh[high_weight_pion]["param"][:, 1]
    # # pion_mass = np.average(fitlist_pion_cosh[high_weight_pion]["param"][:, 1])
    # pion_fit = fitlist_pion_cosh[high_weight_pion]["param"]
    print(np.average(pion_mass))

    weights_kaon = np.array([i["weight"] for i in fitlist_kaon_cosh])
    energies_kaon = np.array([i["param"][:, 1] for i in fitlist_kaon_cosh])
    kaon_mass = np.dot(weights_kaon, energies_kaon)
    # high_weight_kaon = np.argmax(weights_kaon)
    # kaon_mass = fitlist_kaon_cosh[high_weight_kaon]["param"][:, 1]
    # # kaon_mass = np.average(fitlist_kaon_cosh[high_weight_kaon]["param"][:, 1])
    # kaon_fit = fitlist_kaon_cosh[high_weight_kaon]["param"]
    print(np.average(kaon_mass))

    # ======================================================================
    # Open two-point function with twist data:
    pion_utwist_file = plotdir / (f"time_window_loop_pion_u-twist_Cosh.pkl")
    kaon_stwist_file = plotdir / (f"time_window_loop_kaon_s-twist_Cosh.pkl")
    if pion_utwist_file.is_file():
        with open(pion_utwist_file, "rb") as file_in:
            fitlist_pion_utwist = pickle.load(file_in)
    else:
        print("pion u-twist two-point function fit not found")
        exit()
    if kaon_stwist_file.is_file():
        with open(kaon_stwist_file, "rb") as file_in:
            fitlist_kaon_stwist = pickle.load(file_in)
    else:
        print("kaon s-twist two-point function fit not found")
        exit()

    weights_pion_utwist = np.array([i["weight"] for i in fitlist_pion_utwist])
    energies_pion_utwist = np.array([i["param"][:, 1] for i in fitlist_pion_utwist])
    pion_utwist_energy = np.dot(weights_pion_utwist, energies_pion_utwist)
    # sort_weight = np.argsort(weights_pion_utwist)
    # pion_fit_utwist = fitlist_pion_utwist[sort_weight[0]]["param"]
    # pion_utwist_energy = pion_fit_utwist[:, 1]
    # pion_utwist_energy = np.average(pion_fit_utwist[:, 1])

    weights_kaon_stwist = np.array([i["weight"] for i in fitlist_kaon_stwist])
    energies_kaon_stwist = np.array([i["param"][:, 1] for i in fitlist_kaon_stwist])
    kaon_stwist_energy = np.dot(weights_kaon_stwist, energies_kaon_stwist)
    # sort_weight = np.argsort(weights_kaon_stwist)
    # kaon_fit_stwist = fitlist_kaon_stwist[sort_weight[0]]["param"]
    # kaon_stwist_energy = kaon_fit_stwist[:, 1]
    # kaon_stwist_energy = np.average(kaon_fit_stwist[:, 1])

    # ======================================================================
    # Open the quad ratio fit results
    # threepoint_file4 = datadir / (
    threepoint_file4 = datadir_tau13 / (
        "time_window_loop_3pt_quad_kpi_u-twist_ps_ps_fit_constant.pkl"
        # "time_window_loop_3pt_quad_kpi_u-twist_fwd_fit_constant.pkl"
    )
    # threepoint_file5 = datadir / (
    threepoint_file5 = datadir_tau13 / (
        "time_window_loop_3pt_quad_kpi_s-twist_ps_ps_fit_constant.pkl"
        # "time_window_loop_3pt_quad_kpi_s-twist_fwd_fit_constant.pkl"
    )

    with open(threepoint_file4, "rb") as file_in:
        data_3pt_4 = pickle.load(file_in)
    bestfit4 = data_3pt_4[0]
    quad_fit_utwist = bestfit4["param"][:, 0]

    with open(threepoint_file5, "rb") as file_in:
        data_3pt_5 = pickle.load(file_in)
    bestfit5 = data_3pt_5[0]
    quad_fit_stwist = bestfit5["param"][:, 0]

    # # ----------------------------------------------------------------------
    # # Read fit to rescaled threept function
    # threepoint_file4b = datadir / (
    #     "time_window_loop_3pt_double_fitparam_kpi_fit_utwist_constant.pkl"
    # )
    # threepoint_file5b = datadir / (
    #     "time_window_loop_3pt_double_fitparam_kpi_fit_stwist_constant.pkl"
    # )
    # with open(threepoint_file4b, "rb") as file_in:
    #     data_3pt_4b = pickle.load(file_in)
    # bestfit4b = data_3pt_4b[0]
    # resc_fit_utwist = bestfit4b["param"][:, 0]
    # with open(threepoint_file5b, "rb") as file_in:
    #     data_3pt_5b = pickle.load(file_in)
    # bestfit5b = data_3pt_5b[0]
    # resc_fit_stwist = bestfit5b["param"][:, 0]

    # ----------------------------------------------------------------------
    # Read fit to ratio of threept over twopt function
    threepoint_file4c = datadir / (
        "time_window_loop_3pt_double_kpi_fit_utwist_constant.pkl"
    )
    threepoint_file5c = datadir / (
        "time_window_loop_3pt_double_kpi_fit_stwist_constant.pkl"
    )
    with open(threepoint_file4c, "rb") as file_in:
        data_3pt_4c = pickle.load(file_in)
    bestfit4c = data_3pt_4c[0]
    dbl_fit_utwist = bestfit4c["param"][:, 0]
    with open(threepoint_file5c, "rb") as file_in:
        data_3pt_5c = pickle.load(file_in)
    bestfit5c = data_3pt_5c[0]
    dbl_fit_stwist = bestfit5c["param"][:, 0]

    # ----------------------------------------------------------------------
    print()
    print(np.average(quad_fit_utwist), " +- ", np.std(quad_fit_utwist))
    print(np.average(quad_fit_stwist), " +- ", np.std(quad_fit_stwist))

    # print()
    # print(np.average(resc_fit_utwist), " +- ", np.std(resc_fit_utwist))
    # print(np.average(resc_fit_stwist), " +- ", np.std(resc_fit_stwist))

    print()
    print(np.average(dbl_fit_utwist), " +- ", np.std(dbl_fit_utwist))
    print(np.average(dbl_fit_stwist), " +- ", np.std(dbl_fit_stwist))
    print()

    # ----------------------------------------------------------------------
    # make the ratios
    ratio_f0 = (
        quad_fit_stwist * (kaon_mass - pion_utwist_energy)
        - quad_fit_utwist * (kaon_stwist_energy - pion_mass)
    ) / (
        (kaon_stwist_energy + pion_mass) * (kaon_mass - pion_utwist_energy)
        - (kaon_mass + pion_utwist_energy) * (kaon_stwist_energy - pion_mass)
    )
    print("f^0_{K pi}(0) = ")
    print(np.average(ratio_f0), " +- ", np.std(ratio_f0))

    # ratio_f0_resc = (
    #     resc_fit_stwist * (kaon_mass - pion_utwist_energy)
    #     - resc_fit_utwist * (kaon_stwist_energy - pion_mass)
    # ) / (
    #     (kaon_stwist_energy + pion_mass) * (kaon_mass - pion_utwist_energy)
    #     - (kaon_mass + pion_utwist_energy) * (kaon_stwist_energy - pion_mass)
    # )
    # print("f^0_{K pi}(0) = ")
    # print(np.average(ratio_f0_resc), " +- ", np.std(ratio_f0_resc))

    ratio_f0_dbl = (
        dbl_fit_stwist * (kaon_mass - pion_utwist_energy)
        - dbl_fit_utwist * (kaon_stwist_energy - pion_mass)
    ) / (
        (kaon_stwist_energy + pion_mass) * (kaon_mass - pion_utwist_energy)
        - (kaon_mass + pion_utwist_energy) * (kaon_stwist_energy - pion_mass)
    )
    print("f^0_{K pi}(0) = ")
    print(np.average(ratio_f0_dbl), " +- ", np.std(ratio_f0_dbl))

    # ----------------------------------------------------------------------
    # Write the values for f^0(0)
    filename = "f0_0_3pt_quad_kpi.pkl"
    with open(datadir / filename, "wb") as file_out:
        pickle.dump(ratio_f0, file_out)

    # filename = "f0_0_3pt_double_fitparam_kpi.pkl"
    # with open(datadir / filename, "wb") as file_out:
    #     pickle.dump(ratio_f0_resc, file_out)

    filename = "f0_0_3pt_double_kpi.pkl"
    with open(datadir / filename, "wb") as file_out:
        pickle.dump(ratio_f0_dbl, file_out)

    return


if __name__ == "__main__":
    main()

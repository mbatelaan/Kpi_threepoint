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

from analysis_3pt import fit_loop_driver
from analysis_3pt import fit_loop_new
from analysis_3pt import plot_corr
from analysis_3pt import plot_corr_log
from analysis_3pt import plot_eff_corr
from analysis_3pt import fit_correlator

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


def main_meson4():
    """
    Read the meson 3pt functions and 2pt functions. Construct a ratio to fit and plot
    For 3pt correlators with partially twisted boundary conditions
    """

    mystyle = Path("mystyle.txt")
    plt.style.use(mystyle.as_posix())
    plt.rc("text.latex", preamble=r"\usepackage{physics}")
    plotdir = Path("./plots/twisted/")

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
    high_weight_pion = np.argmax(weights_pion)
    pion_mass = fitlist_pion_cosh[high_weight_pion]["param"][:, 1]
    pion_fit = fitlist_pion_cosh[high_weight_pion]["param"]
    print(np.average(pion_mass))

    weights_kaon = np.array([i["weight"] for i in fitlist_kaon_cosh])
    high_weight_kaon = np.argmax(weights_kaon)
    kaon_mass = fitlist_kaon_cosh[high_weight_kaon]["param"][:, 1]
    kaon_fit = fitlist_kaon_cosh[high_weight_kaon]["param"]
    print(np.average(kaon_mass))

    # ----------------------------------------------------------------------
    # Twisted energies
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
    sort_weight = np.argsort(weights_pion_utwist)
    pion_fit_utwist = fitlist_pion_utwist[sort_weight[0]]["param"]
    pion_utwist_energy = pion_fit_utwist[:, 1]

    weights_kaon_stwist = np.array([i["weight"] for i in fitlist_kaon_stwist])
    sort_weight = np.argsort(weights_kaon_stwist)
    kaon_fit_stwist = fitlist_kaon_stwist[sort_weight[0]]["param"]
    kaon_stwist_energy = kaon_fit_stwist[:, 1]

    # ======================================================================
    # Open three-point function data:
    # Twisted u-quark
    filename_p_k_utwist = "./threept_run4/meson-3pt_US_SU_u-twist/messpec/32x64/slrc/kp120620kp121040/sh_gij_p21_90-sh_gij_p21_90/p+0+0+0/messpec_g5-g5_495cfgs.pickle"
    with open(filename_p_k_utwist, "rb") as file_in:
        data = pickle.load(file_in)
    bsdata_p_k_utwist = bootstrap(data, config_ax=0, nboot=nboot, nbin=nbin)[:, :, 0]

    filename_k_p_utwist = "./threept_run4/meson-3pt_US_SU_u-twist/messpec/32x64/slrc/kp121040kp121040/sh_gij_p21_90-sh_gij_p21_90/p+0+0+0/messpec_g5-g5_495cfgs.pickle"
    with open(filename_k_p_utwist, "rb") as file_in:
        data = pickle.load(file_in)
    bsdata_k_p_utwist = bootstrap(data, config_ax=0, nboot=nboot, nbin=nbin)[:, :, 0]

    # Twisted s-quark
    filename_p_k_stwist = "./threept_run4/meson-3pt_US_SU_s-twist/messpec/32x64/slrc/kp120620kp121040/sh_gij_p21_90-sh_gij_p21_90/p+0+0+0/messpec_g5-g5_495cfgs.pickle"
    with open(filename_p_k_stwist, "rb") as file_in:
        data = pickle.load(file_in)
    bsdata_p_k_stwist = bootstrap(data, config_ax=0, nboot=nboot, nbin=nbin)[:, :, 0]

    filename_k_p_stwist = "./threept_run4/meson-3pt_US_SU_s-twist/messpec/32x64/slrc/kp121040kp121040/sh_gij_p21_90-sh_gij_p21_90/p+0+0+0/messpec_g5-g5_495cfgs.pickle"
    with open(filename_k_p_stwist, "rb") as file_in:
        data = pickle.load(file_in)
    bsdata_k_p_stwist = bootstrap(data, config_ax=0, nboot=nboot, nbin=nbin)[:, :, 0]

    # Twisted u-quark and s-quark
    filename_p_k_utwist_stwist = "./threept_run4/meson-3pt_US_SU_u-twist_s-twist/messpec/32x64/slrc/kp120620kp121040/sh_gij_p21_90-sh_gij_p21_90/p+0+0+0/messpec_g5-g5_495cfgs.pickle"
    with open(filename_p_k_utwist_stwist, "rb") as file_in:
        data = pickle.load(file_in)
    bsdata_p_k_utwist_stwist = bootstrap(data, config_ax=0, nboot=nboot, nbin=nbin)[
        :, :, 0
    ]

    filename_k_p_utwist_stwist = "./threept_run4/meson-3pt_US_SU_u-twist_s-twist/messpec/32x64/slrc/kp121040kp121040/sh_gij_p21_90-sh_gij_p21_90/p+0+0+0/messpec_g5-g5_495cfgs.pickle"
    with open(filename_k_p_utwist_stwist, "rb") as file_in:
        data = pickle.load(file_in)
    bsdata_k_p_utwist_stwist = bootstrap(data, config_ax=0, nboot=nboot, nbin=nbin)[
        :, :, 0
    ]

    # ----------------------------------------------------------------------
    # Flavour-diagonal 3pt functions
    # Twisted u-quark
    filename_p_p_utwist = "./threept_run4/meson-3pt_UU_SS_u-twist_s-twist/messpec/32x64/slrc/kp121040kp121040/sh_gij_p21_90-sh_gij_p21_90/p+0+0+0/messpec_g5-g5_495cfgs.pickle"
    with open(filename_p_p_utwist, "rb") as file_in:
        data_pion = pickle.load(file_in)
    bsdata_p_p_utwist = bootstrap(data_pion, config_ax=0, nboot=nboot, nbin=nbin)[
        :, :, 0
    ]
    # Twisted s-quark
    filename_k_k_stwist = "./threept_run4/meson-3pt_UU_SS_u-twist_s-twist/messpec/32x64/slrc/kp120620kp121040/sh_gij_p21_90-sh_gij_p21_90/p+0+0+0/messpec_g5-g5_495cfgs.pickle"
    with open(filename_k_k_stwist, "rb") as file_in:
        data_kaon = pickle.load(file_in)
    bsdata_k_k_stwist = bootstrap(data_kaon, config_ax=0, nboot=nboot, nbin=nbin)[
        :, :, 0
    ]

    # At rest
    filename_p_p = "./threept_run4/meson-3pt_UU_US/messpec/32x64/slrc/kp121040kp121040/sh_gij_p21_90-sh_gij_p21_90/p+0+0+0/messpec_g5-g5_495cfgs.pickle"
    with open(filename_p_p, "rb") as file_in:
        data_pion = pickle.load(file_in)
    bsdata_p_p = bootstrap(data_pion, config_ax=0, nboot=nboot, nbin=nbin)[:, :, 0]

    filename_k_k = "./threept_run4/meson-3pt_SS_SU/messpec/32x64/slrc/kp120620kp121040/sh_gij_p21_90-sh_gij_p21_90/p+0+0+0/messpec_g5-g5_495cfgs.pickle"
    with open(filename_k_k, "rb") as file_in:
        data_kaon = pickle.load(file_in)
    bsdata_k_k = bootstrap(data_kaon, config_ax=0, nboot=nboot, nbin=nbin)[:, :, 0]

    # ----------------------------------------------------------------------
    # Two-point functions
    filename_p = "./threept_run4/meson_qcdsf/messpec/32x64/slrc/kp121040kp121040/sh_gij_p21_90-sh_gij_p21_90/p+0+0+0/messpec_g5-g5_495cfgs.pickle"
    with open(filename_p, "rb") as file_in:
        data_pion = pickle.load(file_in)
    bsdata_pion = bootstrap(data_pion, config_ax=0, nboot=nboot, nbin=nbin)[:, :, 0]

    filename_k = "./threept_run4/meson_qcdsf/messpec/32x64/slrc/kp120620kp121040/sh_gij_p21_90-sh_gij_p21_90/p+0+0+0/messpec_g5-g5_495cfgs.pickle"
    with open(filename_k, "rb") as file_in:
        data_kaon = pickle.load(file_in)
    bsdata_kaon = bootstrap(data_kaon, config_ax=0, nboot=nboot, nbin=nbin)[:, :, 0]

    # Two-point functions with twist
    filename_p = "./threept_run4/meson-2pt_u-twist/messpec/32x64/slrc/kp121040kp121040/sh_gij_p21_90-sh_gij_p21_90/p+0+0+0/messpec_g5-g5_495cfgs.pickle"
    with open(filename_p, "rb") as file_in:
        data_pion = pickle.load(file_in)
    bsdata_pion_utwist = bootstrap(data_pion, config_ax=0, nboot=nboot, nbin=nbin)[
        :, :, 0
    ]

    filename_k = "./threept_run4/meson-2pt_s-twist/messpec/32x64/slrc/kp120620kp121040/sh_gij_p21_90-sh_gij_p21_90/p+0+0+0/messpec_g5-g5_495cfgs.pickle"
    with open(filename_k, "rb") as file_in:
        data_kaon = pickle.load(file_in)
    bsdata_kaon_stwist = bootstrap(data_kaon, config_ax=0, nboot=nboot, nbin=nbin)[
        :, :, 0
    ]

    # ======================================================================
    # Ratios
    # tmax = 32
    tmax = 50
    tau = 10
    # energy_factor = np.sqrt(4 * pion_mass * kaon_mass)
    energy_factor_utwist = np.sqrt(4 * pion_utwist_energy * kaon_mass)
    energy_factor_stwist = np.sqrt(4 * pion_mass * kaon_stwist_energy)
    energy_factor_utwist_stwist = np.sqrt(4 * pion_utwist_energy * kaon_stwist_energy)
    energy_factor2 = pion_mass + kaon_mass
    norm_factor = 0.863

    utwist = True
    stwist = True
    bothtwist = True
    # ======================================================================
    # Twisted u-quark
    if utwist:
        # ----------------------------------------------------------------------
        # Ratio with all four 3-point functions
        denominator = np.abs(bsdata_k_k[:, :tmax] * bsdata_p_p_utwist[:, :tmax]) ** (-1)
        numerator = np.abs(bsdata_p_k_utwist[:, :tmax] * bsdata_k_p_utwist[:, :tmax])
        bsdata_7a = np.sqrt(np.einsum("ij,ij->ij", numerator, denominator))
        quad_ratio_bs = np.einsum("ij,i->ij", bsdata_7a, energy_factor_utwist)
        # quad_ratio_bs = np.einsum("ij,i->ij", bsdata_7a, energy_factor_utwist / norm_factor)
        plot_corr(
            quad_ratio_bs,
            plotname="3pt_pi-k_quad-ratio_utwist",
            plotdir=plotdir,
            v_line=10,
        )
        plot_corr(
            quad_ratio_bs,
            plotname="3pt_pi-k_quad-ratio_ylim_utwist",
            plotdir=plotdir,
            v_line=10,
            ylim=(0.2, 0.37),
            # ylim=(0.35, 0.5),
        )

        # ----------------------------------------------------------------------
        # Ratio with two 3-point functions divided by two 2-point functions
        denominator1 = np.abs(bsdata_kaon[:, :tmax] * bsdata_pion_utwist[:, :tmax]) ** (
            -1
        )
        bsdata_1 = np.abs(bsdata_p_k_utwist[:, :tmax] * bsdata_k_p_utwist[:, :tmax])
        bsdata_1a = np.sqrt(np.einsum("ij,ij->ij", bsdata_1, denominator1))
        bsdata_1b = np.einsum(
            "ij,i->ij",
            bsdata_1a,
            energy_factor_utwist * norm_factor,
            # "ij,i->ij", bsdata_1a, energy_factor
        )
        plot_corr(
            bsdata_1b,
            plotname="3pt_pi-k_double-ratio_utwist",
            plotdir=plotdir,
            v_line=10,
        )
        plot_corr(
            bsdata_1b,
            plotname="3pt_pi-k_double-ratio_utwist_ylim",
            plotdir=plotdir,
            v_line=10,
            ylim=(0.2, 0.37),
        )

        # ----------------------------------------------------------------------
        # double ratio with fit params divided out
        t_vals = np.arange(0, tmax)
        bsdata_dbl_fit = np.abs(
            bsdata_p_k_utwist[:, :tmax]
            * bsdata_k_p_utwist[:, :tmax]
            * np.exp(np.einsum("i,j->ij", pion_utwist_energy, t_vals))
            * np.exp(np.einsum("i,j->ij", kaon_mass, t_vals))
        )
        denominator_fit = (kaon_fit[:, 0] * pion_fit_utwist[:, 0] / 4) ** (-1)
        bsdata_dbl_fit1 = np.sqrt(
            np.einsum("ij,i->ij", bsdata_dbl_fit, denominator_fit)
        )
        bsdata_dbl_fit2 = np.einsum(
            "ij,i->ij",
            bsdata_dbl_fit1,
            energy_factor_utwist * norm_factor,
        )
        plot_corr(
            bsdata_dbl_fit2,
            plotname="3pt_pi-k_double-ratio_utwist_fitparam",
            plotdir=plotdir,
            v_line=10,
            ylim=(0.2, 0.37),
        )

        # ----------------------------------------------------------------------
        # Fits
        fit_correlator(
            bsdata_1b,
            plotdir,
            name="3pt_double_kpi_fit_utwist",
            ylabel=r"$R_1(t)$",
            ylim=(0.315, 0.33),  # ylim=(0.2, 0.37)
            time_limits=np.array([[[11, 15], [13, 20]]]),
        )
        fit_correlator(
            quad_ratio_bs,
            plotdir,
            name="3pt_quad_kpi_fit_utwist",
            ylabel=r"$R_2(t)$",
            ylim=(0.315, 0.33),  # ,ylim=(0.35, 0.5)
            time_limits=np.array([[[11, 20], [13, 30]]]),
        )
        fit_correlator(
            bsdata_dbl_fit2,
            plotdir,
            name="3pt_double_fitparam_kpi_fit_utwist",
            ylabel=r"$R_3(t)$",
            ylim=(0.315, 0.33),  # ,ylim=(0.35, 0.5)
            time_limits=np.array([[[11, 20], [13, 30]]]),
        )

    # ======================================================================
    # Twisted s-quark
    if stwist:
        # ----------------------------------------------------------------------
        # Ratio with all four 3-point functions
        denominator = np.abs(bsdata_k_k_stwist[:, :tmax] * bsdata_p_p[:, :tmax]) ** (-1)
        numerator = np.abs(bsdata_p_k_stwist[:, :tmax] * bsdata_k_p_stwist[:, :tmax])
        bsdata_7a = np.sqrt(np.einsum("ij,ij->ij", numerator, denominator))
        quad_ratio_bs = np.einsum("ij,i->ij", bsdata_7a, energy_factor_stwist)
        plot_corr(
            quad_ratio_bs,
            plotname="3pt_pi-k_quad-ratio_stwist",
            plotdir=plotdir,
            v_line=10,
        )
        plot_corr(
            quad_ratio_bs,
            plotname="3pt_pi-k_quad-ratio_ylim_stwist",
            plotdir=plotdir,
            v_line=10,
            ylim=(0.2, 0.37),
            # ylim=(0.35, 0.5),
        )

        # ----------------------------------------------------------------------
        # Ratio with two 3-point functions divided by two 2-point functions
        denominator1 = (bsdata_kaon_stwist[:, :tmax] * bsdata_pion[:, :tmax]) ** (-1)
        bsdata_1 = bsdata_p_k_stwist[:, :tmax] * bsdata_k_p_stwist[:, :tmax]
        bsdata_1a = np.sqrt(np.abs(np.einsum("ij,ij->ij", bsdata_1, denominator1)))
        dbl_ratio_bs = np.einsum(
            "ij,i->ij",
            bsdata_1a,
            energy_factor_stwist * norm_factor,
        )
        plot_corr(
            dbl_ratio_bs,
            plotname="3pt_pi-k_double-ratio_stwist",
            plotdir=plotdir,
            v_line=10,
        )
        plot_corr(
            dbl_ratio_bs,
            plotname="3pt_pi-k_double-ratio_stwist_ylim",
            plotdir=plotdir,
            v_line=10,
            ylim=(0.2, 0.37),
        )

        # ----------------------------------------------------------------------
        # double ratio with fit params divided out
        t_vals = np.arange(0, tmax)
        bsdata_dbl_fit = np.abs(
            bsdata_p_k_stwist[:, :tmax]
            * bsdata_k_p_stwist[:, :tmax]
            * np.exp(np.einsum("i,j->ij", pion_mass, t_vals))
            * np.exp(np.einsum("i,j->ij", kaon_stwist_energy, t_vals))
        )
        denominator_fit = (kaon_fit_stwist[:, 0] * pion_fit[:, 0] / 4) ** (-1)
        bsdata_dbl_fit1 = np.sqrt(
            np.einsum("ij,i->ij", bsdata_dbl_fit, denominator_fit)
        )
        bsdata_dbl_fit2 = np.einsum(
            "ij,i->ij",
            bsdata_dbl_fit1,
            energy_factor_stwist * norm_factor,
        )
        plot_corr(
            bsdata_dbl_fit2,
            plotname="3pt_pi-k_double-ratio_stwist_fitparam",
            plotdir=plotdir,
            v_line=10,
            ylim=(0.2, 0.37),
        )

        # ----------------------------------------------------------------------
        # Fits
        fit_correlator(
            dbl_ratio_bs,
            plotdir,
            name="3pt_double_kpi_fit_stwist",
            ylabel=r"$R_1(t)$",
            ylim=(0.315, 0.33),  # ylim=(0.2, 0.37)
            time_limits=np.array([[[11, 15], [13, 20]]]),
        )
        fit_correlator(
            quad_ratio_bs,
            plotdir,
            name="3pt_quad_kpi_fit_stwist",
            ylabel=r"$R_2(t)$",
            ylim=(0.315, 0.33),  # ,ylim=(0.35, 0.5)
            time_limits=np.array([[[11, 20], [13, 30]]]),
        )
        fit_correlator(
            bsdata_dbl_fit2,
            plotdir,
            name="3pt_double_fitparam_kpi_fit_stwist",
            ylabel=r"$R_3(t)$",
            ylim=(0.315, 0.33),  # ,ylim=(0.35, 0.5)
            time_limits=np.array([[[11, 20], [13, 30]]]),
        )

    # ======================================================================
    # Twisted u-quark and s-quark
    if bothtwist:
        # ----------------------------------------------------------------------
        # Ratio with all four 3-point functions
        denominator = np.abs(
            bsdata_k_k_stwist[:, :tmax] * bsdata_p_p_utwist[:, :tmax]
        ) ** (-1)
        numerator = np.abs(
            bsdata_p_k_utwist_stwist[:, :tmax] * bsdata_k_p_utwist_stwist[:, :tmax]
        )
        bsdata_7a = np.sqrt(np.einsum("ij,ij->ij", numerator, denominator))
        quad_ratio_bs = np.einsum("ij,i->ij", bsdata_7a, energy_factor_utwist_stwist)

        plot_corr(
            quad_ratio_bs,
            plotname="3pt_pi-k_quad-ratio_utwist_stwist",
            plotdir=plotdir,
            v_line=10,
        )
        plot_corr(
            quad_ratio_bs,
            plotname="3pt_pi-k_quad-ratio_ylim_utwist_stwist",
            plotdir=plotdir,
            v_line=10,
            ylim=(0.2, 0.37),
            # ylim=(0.35, 0.5),
        )

        # ----------------------------------------------------------------------
        # Ratio with two 3-point functions divided by two 2-point functions
        denominator1 = np.abs(
            bsdata_kaon_stwist[:, :tmax] * bsdata_pion_utwist[:, :tmax]
        ) ** (-1)
        bsdata_1 = np.abs(
            bsdata_p_k_utwist_stwist[:, :tmax] * bsdata_k_p_utwist_stwist[:, :tmax]
        )
        bsdata_1a = np.sqrt(np.einsum("ij,ij->ij", bsdata_1, denominator1))
        double_ratio_bs = np.einsum(
            "ij,i->ij",
            bsdata_1a,
            energy_factor_utwist_stwist * norm_factor,
        )

        plot_corr(
            double_ratio_bs,
            plotname="3pt_pi-k_double-ratio_utwist_stwist",
            plotdir=plotdir,
            v_line=10,
        )
        plot_corr(
            double_ratio_bs,
            plotname="3pt_pi-k_double-ratio_utwist_stwist_ylim",
            plotdir=plotdir,
            v_line=10,
            ylim=(0.2, 0.37),
        )

        # ----------------------------------------------------------------------
        # single ratios with fit params divided out
        # Kaon -> pion
        t_vals = np.arange(0, tmax)
        bsdata_kpi_fit = bsdata_k_p_utwist_stwist[:, :tmax] * np.exp(
            np.einsum("i,j->ij", pion_utwist_energy, t_vals - tau)
        )
        const_factor1 = (
            energy_factor_utwist_stwist
            * norm_factor
            * np.exp(kaon_stwist_energy * tau)
            / np.sqrt(
                kaon_fit_stwist[:, 0] * pion_fit_utwist[:, 0] / 4
            )  # factor of 4 due to cosh fits
        )
        bsdata_kpi_fit1 = np.einsum("ij,i->ij", bsdata_kpi_fit, const_factor1)
        plot_corr(
            bsdata_kpi_fit1,
            plotname="3pt_k-pi_rescaled_utwist_stwist",
            plotdir=plotdir,
            v_line=10,
        )

        # pion -> Kaon
        t_vals = np.arange(0, tmax)
        bsdata_pik_fit = bsdata_p_k_utwist_stwist[:, :tmax] * np.exp(
            np.einsum("i,j->ij", kaon_stwist_energy, t_vals - tau)
        )
        const_factor2 = (
            energy_factor_utwist_stwist
            * norm_factor
            * np.exp(pion_utwist_energy * tau)
            / np.sqrt(
                kaon_fit_stwist[:, 0] * pion_fit_utwist[:, 0] / 4
            )  # factor of 4 due to cosh fits
        )
        bsdata_pik_fit1 = np.einsum("ij,i->ij", bsdata_pik_fit, const_factor2)
        plot_corr(
            bsdata_pik_fit1,
            plotname="3pt_pi-k_rescaled_utwist_stwist",
            plotdir=plotdir,
            v_line=10,
        )

        # ----------------------------------------------------------------------
        # double ratio with fit params divided out
        t_vals = np.arange(0, tmax)
        bsdata_dbl_fit = np.abs(
            bsdata_p_k_utwist_stwist[:, :tmax]
            * bsdata_k_p_utwist_stwist[:, :tmax]
            * np.exp(np.einsum("i,j->ij", pion_utwist_energy, t_vals))
            * np.exp(np.einsum("i,j->ij", kaon_stwist_energy, t_vals))
        )
        denominator_fit = (kaon_fit_stwist[:, 0] * pion_fit_utwist[:, 0] / 4) ** (-1)
        bsdata_dbl_fit1 = np.sqrt(
            np.einsum("ij,i->ij", bsdata_dbl_fit, denominator_fit)
        )
        bsdata_dbl_fit2 = np.einsum(
            "ij,i->ij",
            bsdata_dbl_fit1,
            energy_factor_utwist_stwist * norm_factor,
        )
        plot_corr(
            bsdata_dbl_fit2,
            plotname="3pt_pi-k_double-ratio_utwist_stwist_fitparam",
            plotdir=plotdir,
            v_line=10,
        )

        # ----------------------------------------------------------------------
        # Fits
        fitlist_R1 = fit_correlator(
            double_ratio_bs,
            plotdir,
            name="3pt_double_kpi_fit_utwist_stwist",
            ylabel=r"$R_1(t)$",
            ylim=(0.33, 0.35),  # ylim=(0.2, 0.37)
            time_limits=np.array([[[11, 15], [13, 20]]]),
        )
        fitlist_R2 = fit_correlator(
            quad_ratio_bs,
            plotdir,
            name="3pt_quad_kpi_fit_utwist_stwist",
            ylabel=r"$R_2(t)$",
            ylim=(0.33, 0.35),  # ,ylim=(0.35, 0.5)
            time_limits=np.array([[[11, 20], [13, 30]]]),
        )
        fitlist_R3 = fit_correlator(
            bsdata_dbl_fit2,
            plotdir,
            name="3pt_double_fitparam_kpi_fit_utwist_stwist",
            ylabel=r"$R_3(t)$",
            ylim=(0.32, 0.36),
            time_limits=np.array([[[11, 20], [13, 30]]]),
        )
        fitlist_R3a = fit_correlator(
            bsdata_kpi_fit1,
            plotdir,
            name="3pt_k-pi_rescaled_utwist_stwist_fits",
            ylabel=r"$R_3(t)$",
            # ylim=(0.16, 0.18),  # ,ylim=(0.35, 0.5)
            ylim=(0.32, 0.36),
            time_limits=np.array([[[11, 20], [13, 30]]]),
        )
        fitlist_R3b = fit_correlator(
            bsdata_pik_fit1,
            plotdir,
            name="3pt_pi-k_rescaled_utwist_stwist_fits",
            ylabel=r"$R_3(t)$",
            # ylim=(0.16, 0.18),  # ,ylim=(0.35, 0.5)
            ylim=(0.32, 0.36),
            time_limits=np.array([[[11, 20], [13, 30]]]),
        )

        plot_fit_comparison(
            [fitlist_R1, fitlist_R2, fitlist_R3, fitlist_R3a, fitlist_R3b],
            [r"$R_1$", r"$R_2$", r"$R_3$", r"$R_3^{K\to \pi}$", r"$R_3^{\pi \to K}$"],
            plotdir,
            name="fit_comp_utwist_stwist",
        )

    return


def plot_fit_comparison(
    fitlists,
    labels,
    plotdir,
    ylabel=r"$\mel{\pi(\mathbf{p}_{\pi})}{V_4}{K(\mathbf{p}_K)}$",
    name="",
):
    """
    Plot all the fit results nex to each other to compare them
    """

    w_avg_list = []
    red_chisq_list = []
    for fitlist_kpi in fitlists:
        bestfit = fitlist_kpi[0]
        weighted_avg = bestfit["param"][:, 0]
        fit_redchisq = bestfit["redchisq"]

        w_avg_list.append(weighted_avg)
        red_chisq_list.append(fit_redchisq)

    f, axs = plt.subplots(1, 1, figsize=(6, 6))
    ymax = 0
    ymin = 1e30
    for ielem, fit_elem in enumerate(w_avg_list):
        axs.errorbar(
            ielem,
            np.average(fit_elem),
            np.std(fit_elem),
            capsize=4,
            elinewidth=1,
            color=_colors[0],
            fmt="s",
            mfc="white",
        )
        axs.text(
            ielem,
            np.average(fit_elem) + 1.2 * np.std(fit_elem),
            s=rf"$\chi^2_{{\textrm{{dof}}}}={red_chisq_list[ielem]:.2f}$",
            rotation=30,
            rotation_mode="anchor",
            fontsize=10,
            color="k",
        )
        if np.average(fit_elem) + 2 * np.std(fit_elem) > ymax:
            ymax = np.average(fit_elem) + 2 * np.std(fit_elem)
        if np.average(fit_elem) - 1.2 * np.std(fit_elem) < ymin:
            ymin = np.average(fit_elem) - 1.2 * np.std(fit_elem)

    xvals = np.arange(len(fitlists))
    plt.xticks(xvals, labels, fontsize="x-small")
    plt.ylabel(ylabel, fontsize="x-small")

    plt.xlim(-0.5, len(fitlists) + 0.5)
    plt.ylim(ymin, ymax)

    plt.tight_layout()
    plt.savefig(plotdir / f"{name}.pdf")
    plt.close()


if __name__ == "__main__":
    main_meson4()

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
    plotdir = Path("./plots/")

    nboot = 500
    nbin = 1

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
    pion_energy = fitlist_pion_cosh[high_weight_pion]["param"][:, 1]
    pion_fit = fitlist_pion_cosh[high_weight_pion]["param"]
    print(np.average(pion_energy))

    weights_kaon = np.array([i["weight"] for i in fitlist_kaon_cosh])
    high_weight_kaon = np.argmax(weights_kaon)
    kaon_energy = fitlist_kaon_cosh[high_weight_kaon]["param"][:, 1]
    kaon_fit = fitlist_kaon_cosh[high_weight_kaon]["param"]
    print(np.average(kaon_energy))

    # ======================================================================
    # Twisted u-quark
    # Open three-point function data:
    filename_p_k = "./threept_run4/meson-3pt_US_SU_u-twist/messpec/32x64/slrc/kp120620kp121040/sh_gij_p21_90-sh_gij_p21_90/p+0+0+0/messpec_g5-g5_495cfgs.pickle"
    with open(filename_p_k, "rb") as file_in:
        data = pickle.load(file_in)
    bsdata_p_k = bootstrap(data, config_ax=0, nboot=nboot, nbin=nbin)[:, :, 0]

    filename_k_p = "./threept_run4/meson-3pt_US_SU_u-twist/messpec/32x64/slrc/kp121040kp121040/sh_gij_p21_90-sh_gij_p21_90/p+0+0+0/messpec_g5-g5_495cfgs.pickle"
    with open(filename_k_p, "rb") as file_in:
        data = pickle.load(file_in)
    bsdata_k_p = bootstrap(data, config_ax=0, nboot=nboot, nbin=nbin)[:, :, 0]

    filename_p_p = "./threept_run4/meson-3pt_UU_SS_u-twist_s-twist/messpec/32x64/slrc/kp121040kp121040/sh_gij_p21_90-sh_gij_p21_90/p+0+0+0/messpec_g5-g5_495cfgs.pickle"
    with open(filename_p_p, "rb") as file_in:
        data_pion = pickle.load(file_in)
    bsdata_p_p = bootstrap(data_pion, config_ax=0, nboot=nboot, nbin=nbin)[:, :, 0]

    filename_k_k = "./threept_run4/meson-3pt_UU_SS_u-twist_s-twist/messpec/32x64/slrc/kp120620kp121040/sh_gij_p21_90-sh_gij_p21_90/p+0+0+0/messpec_g5-g5_495cfgs.pickle"
    with open(filename_k_k, "rb") as file_in:
        data_kaon = pickle.load(file_in)
    bsdata_k_k = bootstrap(data_kaon, config_ax=0, nboot=nboot, nbin=nbin)[:, :, 0]

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
    bsdata_pion_utwist = bootstrap(data_pion, config_ax=0, nboot=nboot, nbin=nbin)[:, :, 0]

    filename_k = "./threept_run4/meson-2pt_u-twist/messpec/32x64/slrc/kp120620kp121040/sh_gij_p21_90-sh_gij_p21_90/p+0+0+0/messpec_g5-g5_495cfgs.pickle"
    with open(filename_k, "rb") as file_in:
        data_kaon = pickle.load(file_in)
    bsdata_kaon_utwist = bootstrap(data_kaon, config_ax=0, nboot=nboot, nbin=nbin)[:, :, 0]

    # ======================================================================
    # Ratios
    tmax = 32
    tau = 10
    energy_factor = np.sqrt(4 * pion_energy * kaon_energy)
    energy_factor2 = pion_energy + kaon_energy
    norm_factor = 0.863

    # ----------------------------------------------------------------------
    # Ratio with all four 3-point functions
    denominator = np.abs(bsdata_k_k[:, :tmax] * bsdata_p_p[:, :tmax]) ** (-1)
    numerator = np.abs(bsdata_p_k[:, :tmax] * bsdata_k_p[:, :tmax])
    bsdata_7a = np.sqrt(np.einsum("ij,ij->ij", numerator, denominator))
    # quad_ratio_bs = np.einsum("ij,i->ij", bsdata_7a, energy_factor / energy_factor2)
    quad_ratio_bs = np.einsum("ij,i->ij", bsdata_7a, energy_factor / norm_factor)
    plot_corr(quad_ratio_bs, plotname="3pt_pi-k_quad-ratio_utwist", plotdir=plotdir, v_line=10)
    plot_corr(
        quad_ratio_bs,
        plotname="3pt_pi-k_quad-ratio_ylim_utwist",
        plotdir=plotdir,
        v_line=10,
        ylim=(0.35, 0.5),
    )

    # ----------------------------------------------------------------------
    # Ratio with two 3-point functions divided by two 2-point functions
    denominator1 = np.abs(bsdata_kaon[:, :tmax] * bsdata_pion_utwist[:, :tmax]) ** (-1)
    bsdata_1 = np.abs(bsdata_p_k[:, :tmax] * bsdata_k_p[:, :tmax])
    bsdata_1a = np.sqrt(np.einsum("ij,ij->ij", bsdata_1, denominator1))
    bsdata_1b = np.einsum(
        # "ij,i->ij", bsdata_1a, energy_factor / energy_factor2 * norm_factor
        "ij,i->ij", bsdata_1a, energy_factor
    )
    plot_corr(bsdata_1b, plotname="3pt_pi-k_double-ratio_utwist", plotdir=plotdir, v_line=10)
    plot_corr(bsdata_1b, plotname="3pt_pi-k_double-ratio_utwist_ylim", plotdir=plotdir, v_line=10, ylim=(0.2,0.37))

    # # double ratio with fit params divided out
    # t_vals = np.arange(0, tmax)
    # bsdata_dbl_fit = np.abs(
    #     bsdata_p_k[:, :tmax]
    #     * bsdata_k_p[:, :tmax]
    #     * np.exp(np.einsum("i,j->ij", pion_fit[:, 1], t_vals))
    #     * np.exp(np.einsum("i,j->ij", kaon_fit[:, 1], t_vals))
    # )
    # denominator_fit = (kaon_fit[:, 0] * pion_fit[:, 0] / 4) ** (-1)
    # bsdata_dbl_fit1 = np.sqrt(np.einsum("ij,i->ij", bsdata_dbl_fit, denominator_fit))
    # bsdata_dbl_fit2 = np.einsum(
    #     # "ij,i->ij", bsdata_dbl_fit1, energy_factor / energy_factor2 * norm_factor
    #     "ij,i->ij", bsdata_dbl_fit1, energy_factor
    # )
    # plot_corr(
    #     bsdata_dbl_fit2,
    #     plotname="3pt_pi-k_double-ratio_fit_utwist",
    #     plotdir=plotdir,
    #     v_line=10,
    #     # ylim=(0.95, 1.05),
    # )

    fit_correlator(bsdata_1b, plotdir, name="3pt_double_kpi_fit_utwist", ylim=(0.2,0.37))
    fit_correlator(quad_ratio_bs, plotdir, name="3pt_quad_kpi_fit_utwist", ylim=(0.35, 0.5))

    # fit_correlator(
    #     bsdata_dbl_fit2, plotdir, name="3pt_double_kpi_fitparam_fit_utwist"
    # )
    
    return

def main_baryon():
    """
    Read the baryon 3pt functions and 2pt functions. Construct a ratio to fit and plot
    """

    mystyle = Path("mystyle.txt")
    plt.style.use(mystyle.as_posix())
    plt.rc("text.latex", preamble=r"\usepackage{physics}")
    nboot = 500
    nbin = 1

    filename_sig_nucl = "./baryon-3pt/sigma-nucleon/barspec_nucleon_rel_496cfgs.pickle"
    with open(filename_sig_nucl, "rb") as file_in:
        data = pickle.load(file_in)
    # bsdata_sig_nucl = bootstrap(data, config_ax=0, nboot=nboot, nbin=nbin)[:,:,0]
    bsdata_sig_nucl = bootstrap(data, config_ax=0, nboot=nboot, nbin=nbin)

    filename_nucl_sig = "./baryon-3pt/nucleon-sigma/barspec_nucleon_rel_496cfgs.pickle"
    with open(filename_nucl_sig, "rb") as file_in:
        data = pickle.load(file_in)
    # bsdata_nucl_sig = bootstrap(data, config_ax=0, nboot=nboot, nbin=nbin)[:,:,0]
    bsdata_nucl_sig = bootstrap(data, config_ax=0, nboot=nboot, nbin=nbin)

    filename_n = "./baryon-2pt/slrc/kp121040kp121040/sh_gij_p21_90-sh_gij_p21_90/p+0+0+0/barspec_nucleon_rel_496cfgs.pickle"
    with open(filename_n, "rb") as file_in:
        data_nucl = pickle.load(file_in)
    bsdata_nucl = bootstrap(data_nucl, config_ax=0, nboot=nboot, nbin=nbin)[:, :, 0]

    filename_s = "./baryon-2pt/slrc/kp121040kp120620/sh_gij_p21_90-sh_gij_p21_90/p+0+0+0/barspec_nucleon_rel_496cfgs.pickle"
    with open(filename_s, "rb") as file_in:
        data_sigm = pickle.load(file_in)
    bsdata_sigm = bootstrap(data_sigm, config_ax=0, nboot=nboot, nbin=nbin)[:, :, 0]

    # ======================================================================
    # Ratios
    tmax = 25
    tau = 10

    nucl_energy = 0.422
    sigm_energy = 0.4631
    const_factor = np.sqrt(4 * nucl_energy * sigm_energy)
    bsdata_7 = np.abs(bsdata_sig_nucl[:, :tmax] * bsdata_nucl_sig[:, :tmax])
    denominator = np.abs(bsdata_sigm[:, :tmax] * bsdata_nucl[:, :tmax]) ** (-1)
    bsdata_7a = np.sqrt(np.einsum("ijk,ij->ijk", bsdata_7, denominator)[:, :, 0])
    bsdata_7b = bsdata_7a * const_factor
    print(f"{const_factor=}")

    plot_corr(bsdata_7b, plotname="3pt_nucl_sigm_double-ratio", v_line=10)

    # ======================================================================
    # Fitting the 3pt function
    fit_corr = bsdata_7b
    const_function = ff.initffncs("Constant")
    time_limits = np.array([[[11, 13], [15, 20]]])
    datadir = Path("./")
    [fitlist_nsig] = fit_loop_new(
        fit_corr,
        [const_function],
        time_limits,
        datadir,
        "nucleon_sigma_3pt",
        Nt_min=2,
    )

    bestfit = fitlist_nsig[0]
    print(np.shape(bestfit["param"]))
    fit_time = bestfit["x"]

    time = np.arange(0, np.shape(fit_corr)[1])
    yavg = np.average(fit_corr, axis=0)
    ystd = np.std(fit_corr, axis=0)
    weighted_energy = bestfit["param"][:, 0]
    fit_redchisq = bestfit["redchisq"]

    f, axs = plt.subplots(1, 1, figsize=(9, 6))
    axs.errorbar(
        time,
        yavg,
        ystd,
        capsize=4,
        elinewidth=1,
        color=_colors[0],
        fmt="s",
        mfc="white",
    )
    axs.fill_between(
        fit_time,
        np.ones(len(fit_time))
        * (np.average(bestfit["param"][:, 0]) - np.std(bestfit["param"][:, 0])),
        np.ones(len(fit_time))
        * (np.average(bestfit["param"][:, 0]) + np.std(bestfit["param"][:, 0])),
        color="r",
        alpha=0.6,
        linewidth=0,
        label=rf"$|ME|={err_brackets(np.average(weighted_energy), np.std(weighted_energy))}$; $\chi^2_{{\textrm{{dof}}}} = {fit_redchisq:.2f}$",
        zorder=3,
    )
    axs.axvline(10, color="k", linewidth=1, linestyle="--")

    plt.xlabel(r"$t/a$")
    # plt.ylabel(
    #     r"$\mel{N}{\bar{u}\gamma_{4}s}{\Sigma}$",
    #     fontsize="small",
    # )
    plt.ylabel(r"$R(t)$", fontsize="small")
    plt.legend(fontsize="x-small")
    plt.savefig("./3pt_plot_nsig_fit.pdf")
    plt.ylim(0, 0.5)
    plt.savefig("./3pt_plot_nsig_fit_zoom.pdf")
    plt.close()
    return


if __name__ == "__main__":
    # main_baryon()
    # main()
    # main_meson2()
    # main_meson3()
    main_meson4()

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

from gevpanalysis.plotting_scripts import plot_weighted_avg2

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


def fit_meson_energies():
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

    # Two-point functions with u-twist
    filename_p_utwist = "./threept_run4/meson-2pt_u-twist/messpec/32x64/slrc/kp121040kp121040/sh_gij_p21_90-sh_gij_p21_90/p+0+0+0/messpec_g5-g5_495cfgs.pickle"
    with open(filename_p_utwist, "rb") as file_in:
        data_pion = pickle.load(file_in)
    bsdata_pion_utwist = bootstrap(data_pion, config_ax=0, nboot=nboot, nbin=nbin)[
        :, :, 0
    ]

    # filename_k_utwist = "./threept_run4/meson-2pt_u-twist/messpec/32x64/slrc/kp120620kp121040/sh_gij_p21_90-sh_gij_p21_90/p+0+0+0/messpec_g5-g5_495cfgs.pickle"
    # with open(filename_k_utwist, "rb") as file_in:
    #     data_kaon = pickle.load(file_in)
    # bsdata_kaon_utwist = bootstrap(data_kaon, config_ax=0, nboot=nboot, nbin=nbin)[:, :, 0]

    # # Two-point functions with s-twist
    # filename_p_stwist = "./threept_run4/meson-2pt_s-twist/messpec/32x64/slrc/kp121040kp121040/sh_gij_p21_90-sh_gij_p21_90/p+0+0+0/messpec_g5-g5_495cfgs.pickle"
    # with open(filename_p_stwist, "rb") as file_in:
    #     data_pion = pickle.load(file_in)
    # bsdata_pion_stwist = bootstrap(data_pion, config_ax=0, nboot=nboot, nbin=nbin)[:, :, 0]

    filename_k_stwist = "./threept_run4/meson-2pt_s-twist/messpec/32x64/slrc/kp120620kp121040/sh_gij_p21_90-sh_gij_p21_90/p+0+0+0/messpec_g5-g5_495cfgs.pickle"
    with open(filename_k_stwist, "rb") as file_in:
        data_kaon = pickle.load(file_in)
    bsdata_kaon_stwist = bootstrap(data_kaon, config_ax=0, nboot=nboot, nbin=nbin)[
        :, :, 0
    ]

    lat_T = 64
    cosh_function = ff.initffncs("Cosh", T=lat_T)
    time_limits_pion = np.array(
        [
            [[5, 24], [lat_T - 32, lat_T - 5]],
        ]
    )

    pion_utwist_file = plotdir / (f"time_window_loop_pion_u-twist_Cosh.pkl")
    kaon_stwist_file = plotdir / (f"time_window_loop_kaon_s-twist_Cosh.pkl")

    if pion_utwist_file.is_file():
        with open(pion_utwist_file, "rb") as file_in:
            fitlist_pion_utwist = pickle.load(file_in)
    else:
        [fitlist_pion_utwist] = fit_loop_new(
            bsdata_pion_utwist,
            [cosh_function],
            time_limits_pion,
            plotdir,
            "pion_u-twist",
            Nt_min=10,
        )

    if kaon_stwist_file.is_file():
        with open(kaon_stwist_file, "rb") as file_in:
            fitlist_kaon_stwist = pickle.load(file_in)
    else:
        [fitlist_kaon_stwist] = fit_loop_new(
            bsdata_kaon_stwist,
            [cosh_function],
            time_limits_pion,
            plotdir,
            "kaon_s-twist",
            Nt_min=10,
        )

    plot_weighted_avg2(fitlist_pion_utwist, plotdir, "pion_u-twist")
    plot_weighted_avg2(fitlist_kaon_stwist, plotdir, "kaon_s-twist")

    weights_pion_utwist = np.array([i["weight"] for i in fitlist_pion_utwist])
    sort_weight = np.argsort(weights_pion_utwist)
    chosen_pion_fit = fitlist_pion_utwist[sort_weight[0]]
    pion_t_range = chosen_pion_fit["x"]
    print("=" * 70, "\nPion fits:")
    for fit_ in fitlist_pion_utwist:
        print(f"\nFit t={fit_['x'][0]}-{fit_['x'][-1]}")
        print(f"red. chisq = {fit_['redchisq']}")
        print(f"weight = {fit_['weight']}")

    weights_kaon_stwist = np.array([i["weight"] for i in fitlist_kaon_stwist])
    sort_weight = np.argsort(weights_kaon_stwist)
    chosen_kaon_fit = fitlist_kaon_stwist[sort_weight[0]]
    kaon_t_range = chosen_kaon_fit["x"]
    print("=" * 70, "\nKaon fits:")
    for fit_ in fitlist_kaon_stwist:
        print(f"\nFit t={fit_['x'][0]}-{fit_['x'][-1]}")
        print(f"red. chisq = {fit_['redchisq']}")
        print(f"weight = {fit_['weight']}")

    plot_unpert_mesons(
        bsdata_pion_utwist,
        bsdata_kaon_stwist,
        chosen_pion_fit,
        chosen_kaon_fit,
        cosh_function,
        cosh_function,
        plotdir,
        name="_unpert_mesons",
        show=False,
    )

    return


def plot_unpert_mesons(
    correlator1,
    correlator2,
    fitvals1,
    fitvals2,
    fitfunc1,
    fitfunc2,
    plotdir,
    name="",
    show=False,
):
    spacing = 2
    xlim = 62
    time = np.arange(0, np.shape(correlator1)[1])
    efftime1 = time[:-spacing] + 0.5
    efftime2 = time[:-spacing] + 0.5

    effcorr1 = stats.bs_effmass(correlator1, time_axis=1, spacing=spacing)
    effcorr2 = stats.bs_effmass(correlator2, time_axis=1, spacing=spacing)
    yavg_1 = np.average(effcorr1, axis=0)
    ystd_1 = np.std(effcorr1, axis=0)
    yavg_2 = np.average(effcorr2, axis=0)
    ystd_2 = np.std(effcorr2, axis=0)

    plt.figure(figsize=(7, 5))
    plt.errorbar(
        efftime1,
        yavg_1,
        ystd_1,
        capsize=4,
        elinewidth=1,
        color=_colors[0],
        fmt="s",
        markerfacecolor="none",
        # label=f"{redchisqs[0]:.2f}"
    )
    plt.errorbar(
        efftime2[:xlim],
        yavg_2[:xlim],
        ystd_2[:xlim],
        capsize=4,
        elinewidth=1,
        color=_colors[1],
        fmt="s",
        markerfacecolor="none",
        # label=f"{redchisqs[1]:.2f}"
    )

    # Plot the pion fit results
    fit_energy_pion = fitvals1["param"][:, 1]
    fit_redchisq_pion = fitvals1["redchisq"]
    pion_t_range = fitvals1["x"]
    pion_eval = fitvals1["fitted_func"]
    pion_fit_eff = stats.bs_effmass(pion_eval, time_axis=1, spacing=spacing)

    plt.plot(
        pion_t_range[:-spacing] + 0.5,
        np.average(pion_fit_eff, axis=0),
        color=_colors[0],
    )
    plt.fill_between(
        pion_t_range[:-spacing] + 0.5,
        np.average(pion_fit_eff, axis=0) - np.std(pion_fit_eff, axis=0),
        np.average(pion_fit_eff, axis=0) + np.std(pion_fit_eff, axis=0),
        color=_colors[0],
        alpha=0.3,
        label=rf"$E_{{\pi}}(\mathbf{{0}}) = {err_brackets(np.average(fit_energy_pion),np.std(fit_energy_pion))}$; $\chi^2_{{\textrm{{dof}}}} = {fit_redchisq_pion:.2f}$",
    )

    # Plot the kaon fit results
    fit_energy_kaon = fitvals2["param"][:, 1]
    fit_redchisq_kaon = fitvals2["redchisq"]
    kaon_t_range = fitvals2["x"]
    kaon_eval = fitvals2["fitted_func"]
    kaon_fit_eff = stats.bs_effmass(kaon_eval, time_axis=1, spacing=spacing)

    plt.plot(
        kaon_t_range[:-spacing] + 0.5,
        np.average(kaon_fit_eff, axis=0),
        color=_colors[1],
    )
    plt.fill_between(
        kaon_t_range[:-spacing] + 0.5,
        np.average(kaon_fit_eff, axis=0) - np.std(kaon_fit_eff, axis=0),
        np.average(kaon_fit_eff, axis=0) + np.std(kaon_fit_eff, axis=0),
        color=_colors[1],
        alpha=0.3,
        label=rf"$E_{{K}}(\mathbf{{0}}) = {err_brackets(np.average(fit_energy_kaon),np.std(fit_energy_kaon))}$; $\chi^2_{{\textrm{{dof}}}} = {fit_redchisq_kaon:.2f}$",
    )

    plt.legend(fontsize="x-small")
    plt.ylabel(r"$\textrm{Effective energy}$")
    plt.xlabel(r"$t/a$")
    plt.axhline(y=0, color="k", alpha=0.3, linewidth=0.5)
    # plt.setp(axs, xlim=(0, xlim), ylim=(0, 2))
    plt.xlim(0, xlim)
    plt.ylim(-0.4, 0.4)
    plt.grid(True, alpha=0.3)
    plt.savefig(plotdir / ("unpert_energies.pdf"), metadata=_metadata)
    plt.close()

    return


if __name__ == "__main__":
    fit_meson_energies()

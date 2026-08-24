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
from gevpanalysis.common import read_pickle
from gevpanalysis.common import normalize_matrices

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


def fit_loop_driver(
    data,
    fitfnc,
    time_limits,
    unpert_fits=np.zeros((500, 2)),
    N_fits=5,
    Nt_min=7,
    plot=False,
    disp=False,
    time=False,
):
    """
    Fit the correlator by looping over time ranges and calculating the weight for each fit.

    time_limits = [[tminmin,tminmax],[tmaxmin, tmaxmax]]
    """

    ### Get the effective mass and amplitude for p0
    # Nt_min = len(fitfnc.initpar) + 2
    # Nt_min = 7
    # N_fits = 4
    unpert_energy = unpert_fits[:, 1]

    ### Set the initial guesses for the parameters
    fitfnc.initparfnc(data)
    # fitfnc.bounds = np.array([[-1e50, 1e50], [-1e3, 1e3]])
    # fitfnc.priorsigma = np.array([1e50, 1e3])
    fitfnc.E = np.average(unpert_energy)

    # ======================================================================
    # Fit tot the averages of the correlators first
    fitlist = []
    [[tminmin, tminmax], [tmaxmin, tmaxmax]] = time_limits
    for tmin in range(tminmin, tminmax + 1):
        for tmax in range(tmaxmin, tmaxmax + 1):
            if tmax - tmin > Nt_min:
                timerange = np.arange(tmin, tmax + 1)
                if disp:
                    # print(f"time range = {tmin}-{tmax}\r")
                    sys.stdout.write("time range = %s-%s\r" % (tmin, tmax))
                ydata = data[:, timerange]

                ### Perform the fit
                fitparam_avg = stats.fit_avg_bayes(
                    fitfnc.eval,
                    fitfnc.initpar,
                    fitfnc.priorsigma,
                    timerange,
                    ydata,
                    bounds=fitfnc.bounds,
                    time=time,
                )
                fitparam_avg["y"] = data
                fitlist.append(fitparam_avg)
    if disp:
        print("")
        print(f"priors = {fitfnc.initpar}")
        print(f"priors sigma = {fitfnc.priorsigma}")
        print(f"parameter values = {fitparam_avg['paramavg']}")
        print(f"chi-sq. per dof. = {fitparam_avg['redchisq']}")
    ### Calculate the weights of each of the fits
    doflist = np.array([i["dof"] for i in fitlist])
    # chisqlist = np.array([i["redchisq"] for i in fitlist]) * doflist
    # errorlist = np.array([np.std(i["param"], axis=0)[1] for i in fitlist])
    weightlist = stats.bayesian_weights(fitlist)
    for i, elem in enumerate(fitlist):
        elem["weight"] = weightlist[i]

    # Get the N_fits best fits
    # weightlist[np.isnan(weightlist)] = 0
    sort_fitweights = np.argsort(weightlist)[::-1]
    best_fits_weights = weightlist[sort_fitweights[:N_fits]]
    best_fits = [fitlist[index_] for index_ in sort_fitweights[:N_fits]]
    best_tranges = np.array([[fit_["x"][0], fit_["x"][-1]] for fit_ in best_fits])
    best_redchisq = np.array([fit_["redchisq"] for fit_ in best_fits])
    if disp:
        print(f"{best_tranges=}")
        print(f"{best_fits_weights=}")
        print(f"{best_redchisq=}")
        print(f"best parameter values = {best_fits[0]['paramavg']}")
    fitfnc.initpar = best_fits[0]["paramavg"]

    # ======================================================================
    # Do a full bootstrap fit for the N_fits best fits
    fitlist_bs = []
    for iwin, window in enumerate(best_tranges):
        timerange = np.arange(window[0], window[1] + 1)
        if disp:
            # print(f"time range = {timerange[0]}-{timerange[-1]}\r")
            sys.stdout.write("time range = %s-%s\r" % (timerange[0], timerange[-1]))
        ydata = data[:, timerange]
        ### Perform the fit
        fitparam_bs = stats.fit_bootstrap_bayes_shift(
            fitfnc,
            fitfnc.initpar,
            fitfnc.priorsigma,
            timerange,
            ydata,
            unpert_energy,
            bounds=fitfnc.bounds,
            time=time,
        )
        fitparam_bs["y"] = data
        fitlist_bs.append(fitparam_bs)
    weightlist_bs = stats.bayesian_weights(fitlist_bs)
    for i, elem in enumerate(fitlist_bs):
        elem["weight"] = weightlist_bs[i]
    # for i, elem in enumerate(fitlist_bs):
    #     elem["weight"] = weightlist[sort_fitweights][i]

    best_fits_params = np.array([fit_["param"] for fit_ in fitlist_bs])
    # print(np.shape(weightlist[sort_fitweights]))
    # print(np.shape(best_fits_params))
    weighted_average = np.einsum("i, ijk -> jk", weightlist_bs, best_fits_params)
    redchisq_bs = np.array([fit_["redchisq"] for fit_ in fitlist_bs])
    # print(f"{redchisq_bs=}")
    tranges_bs = np.array([[fit_["x"][0], fit_["x"][-1] + 1] for fit_ in fitlist_bs])
    # print(f"{tranges_bs=}")

    return fitlist_bs, weighted_average


def fit_loop_new(correlator, fitfunctions, time_limits, datadir, data_label, Nt_min=3):
    """Take a correlator with values for each bootstrap and each timeslice and a fitfunction, then loop over the fit windows given by the parameters and return a fit for each window.
    The function here is an object which has labels and initial parameters defined

    fitfunctions: A list of objects which contain fitting functions
    time_limits: A list of arrays defining the extent of the fit windows in the format [[tminmin, tminmax], [tmaxmin, tmaxmax]] for each fit function.
    """
    fitlist_list = []
    for ifunc, function in enumerate(fitfunctions):
        fitlist, bootfit = fit_loop_driver(
            correlator,
            function,
            time_limits[ifunc],
            N_fits=10,
            Nt_min=Nt_min,
            plot=False,
            disp=True,
            time=False,
        )

        fitlist_list.append(fitlist)
        # with open(datadir / (f"time_window_loop_nucl_1exp.pkl"), "wb") as file_out:
        filename = f"time_window_loop_" + data_label + "_" + function.label + ".pkl"
        with open(datadir / filename, "wb") as file_out:
            pickle.dump(fitlist, file_out)

    return fitlist_list


def plot_corr(bsdata, plotname="", plotdir=Path("./"), ylim=False, v_line=0):
    time = np.arange(0, np.shape(bsdata)[1])
    yavg = np.average(bsdata, axis=0)
    ystd = np.std(bsdata, axis=0)

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
    if v_line:
        axs.axvline(v_line, color="k", linewidth=1, linestyle="--")
    if ylim:
        axs.set_ylim(ylim)

    plt.xlabel(r"$t/a$")
    plt.ylabel(r"$R$")
    plt.savefig(plotdir / f"{plotname}.pdf")
    plt.close()
    return


def plot_corr_log(bsdata, plotname="", plotdir=Path("./")):
    time = np.arange(0, np.shape(bsdata)[1])
    yavg = np.average(bsdata, axis=0)
    ystd = np.std(bsdata, axis=0)

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
    axs.axvline(10, color="k", linewidth=1, linestyle="--")
    axs.set_yscale("log")
    plt.savefig(plotdir / f"{plotname}.pdf")
    plt.close()
    return


def plot_eff_corr(bsdata, plotname="", ylim=False):

    time = np.arange(0, np.shape(bsdata)[1])
    eff_time = np.arange(0, np.shape(bsdata)[1] - 1)
    effmass = stats.bs_effmass(bsdata, time_axis=1, spacing=1)

    yavg = np.average(effmass, axis=0)
    ystd = np.std(effmass, axis=0)

    f, axs = plt.subplots(1, 1, figsize=(9, 6))
    axs.errorbar(
        eff_time,
        yavg,
        ystd,
        capsize=4,
        elinewidth=1,
        color=_colors[0],
        fmt="s",
        mfc="white",
    )
    axs.axvline(10, color="k", linewidth=1, linestyle="--")
    if ylim:
        axs.set_ylim(ylim)
    plt.savefig(f"{plotname}.pdf")
    plt.close()
    return


def main():
    """
    Read the meson 3pt functions and 2pt functions. Construct a ratio to fit and plot
    """

    mystyle = Path("mystyle.txt")
    plt.style.use(mystyle.as_posix())
    plt.rc("text.latex", preamble=r"\usepackage{physics}")

    nboot = 500
    nbin = 1
    datadir = Path(
        "/Users/mbatelaan/Research/Adelaide2026/analysis/six_point_fn/data/pickles/run1_meson/"
    )
    with open(datadir / (f"time_window_loop_pion_Cosh.pkl"), "rb") as file_in:
        fitlist_pion_cosh = pickle.load(file_in)
    with open(datadir / (f"time_window_loop_kaon_Cosh.pkl"), "rb") as file_in:
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

    filename_p_k = "./meson-3pt/pion-kaon/messpec_g5-g5_496cfgs.pickle"
    with open(filename_p_k, "rb") as file_in:
        data = pickle.load(file_in)
    bsdata_p_k = bootstrap(data, config_ax=0, nboot=nboot, nbin=nbin)[:, :, 0]

    filename_k_p = "./meson-3pt/kaon-pion/messpec_g5-g5_496cfgs.pickle"
    with open(filename_k_p, "rb") as file_in:
        data = pickle.load(file_in)
    bsdata_k_p = bootstrap(data, config_ax=0, nboot=nboot, nbin=nbin)[:, :, 0]

    filename_p = "./meson-2pt/slrc/kp121040kp121040/sh_gij_p21_90-sh_gij_p21_90/p+0+0+0/messpec_g5-g5_496cfgs.pickle"
    with open(filename_p, "rb") as file_in:
        data_pion = pickle.load(file_in)
    bsdata_pion = bootstrap(data_pion, config_ax=0, nboot=nboot, nbin=nbin)[:, :, 0]

    filename_k = "./meson-2pt/slrc/kp121040kp120620/sh_gij_p21_90-sh_gij_p21_90/p+0+0+0/messpec_g5-g5_496cfgs.pickle"
    with open(filename_k, "rb") as file_in:
        data_kaon = pickle.load(file_in)
    bsdata_kaon = bootstrap(data_kaon, config_ax=0, nboot=nboot, nbin=nbin)[:, :, 0]

    # ======================================================================
    # Plot the correlators
    plot_corr(bsdata_p_k, plotname="3pt_pi-k", v_line=10)
    plot_corr_log(bsdata_p_k, plotname="3pt_pi-k_log")
    plot_eff_corr(bsdata_p_k, plotname="3pt_pi-k_eff", ylim=(-1, 1.5))

    bsdata_3 = bsdata_p_k / bsdata_pion
    plot_corr(bsdata_3, plotname="3pt_pi-k_divpion", v_line=10)

    # bsdata_4 = bsdata_p_k / (np.einsum("ij,i->ij",bsdata_kaon, np.sqrt(bsdata_pion[:,10])) )
    # bsdata_4 = bsdata_p_k / np.einsum("ij,i->ij",bsdata_kaon, np.sqrt(bsdata_pion[:,10]))
    bsdata_4 = bsdata_p_k / bsdata_kaon
    plot_corr(bsdata_4, plotname="3pt_pi-k_divkaon", v_line=10)

    bsdata_4p1 = bsdata_k_p / bsdata_pion
    plot_corr(bsdata_4p1, plotname="3pt_k-pi_divpion", v_line=10)

    # bsdata_4p5 = bsdata_p_k[:,:-10] / bsdata_kaon[:,10:]
    bsdata_4p5 = bsdata_p_k[:, 10:] / bsdata_kaon[:, :-10]
    plot_corr(bsdata_4p5, plotname="3pt_pi-k_divkaon_2", ylim=(-0.5, 0.13))

    bsdata_5 = bsdata_p_k / (np.sqrt(bsdata_pion) * np.sqrt(bsdata_kaon))
    plot_corr(bsdata_5, plotname="3pt_pi-k_divpionkaon", v_line=10)

    # ======================================================================
    # Ratios
    tmax = 32
    tau = 10
    energy_factor = np.sqrt(4 * pion_energy * kaon_energy)
    # energy_factor = np.sqrt(4*pion_energy * kaon_energy)**(-1)
    # energy_factor = pion_energy / pion_energy
    print(np.average(energy_factor))

    bsdata_6 = bsdata_p_k[:, tau : tmax + tau] * np.sqrt(
        bsdata_kaon[:, :tmax]
        / (
            bsdata_kaon[:, tau : tmax + tau]
            * bsdata_pion[:, tau : tmax + tau]
            * bsdata_pion[:, :tmax]
        )
    )
    # bsdata_6 = bsdata_p_k[:,tau:tmax+tau] * np.sqrt( bsdata_kaon[:,tau:tmax+tau] / ( bsdata_kaon[:,:tmax] * bsdata_pion[:,tau:tmax+tau] * bsdata_pion[:,:tmax]) )
    const_factor = np.sqrt(bsdata_pion[:, tau] / bsdata_kaon[:, tau]) * energy_factor
    bsdata_6b = np.einsum("ij,i->ij", bsdata_6, const_factor)
    # plot_corr(bsdata_6b, plotname="3pt_pi-k_ratio", ylim=(0,0.6))
    plot_corr(bsdata_6b, plotname="3pt_pi-k_ratio", ylim=(0, 1.3))

    bsdata_6p1 = bsdata_k_p[:, tau : tmax + tau] * np.sqrt(
        bsdata_pion[:, :tmax]
        / (
            bsdata_pion[:, tau : tmax + tau]
            * bsdata_kaon[:, tau : tmax + tau]
            * bsdata_kaon[:, :tmax]
        )
    )
    # bsdata_6p1 = bsdata_k_p[:,tau:tmax+tau] * np.sqrt( bsdata_pion[:,tau:tmax+tau] / (bsdata_pion[:,:tmax] * bsdata_kaon[:,tau:tmax+tau] * bsdata_kaon[:,:tmax]) )
    const_factor = np.sqrt(bsdata_kaon[:, tau] / bsdata_pion[:, tau]) * energy_factor
    bsdata_6c = np.einsum("ij,i->ij", bsdata_6p1, const_factor)
    # plot_corr(bsdata_6c, plotname="3pt_k-pi_ratio", ylim=(0,0.6))
    plot_corr(bsdata_6c, plotname="3pt_k-pi_ratio", ylim=(0, 1.3))

    denominator = np.abs(bsdata_kaon[:, :tmax] * bsdata_pion[:, :tmax]) ** (-1)
    bsdata_7 = np.abs(bsdata_p_k[:, :tmax] * bsdata_k_p[:, :tmax])
    bsdata_7a = np.sqrt(np.einsum("ij,ij->ij", bsdata_7, denominator))
    bsdata_7b = np.einsum("ij,i->ij", bsdata_7a, energy_factor)
    plot_corr(bsdata_7b, plotname="3pt_pi-k_double-ratio", v_line=10, ylim=(0, 1.3))

    # pion to kaon with fit params divided out
    t_vals = np.arange(0, tmax)
    bsdata_p2k_fit = bsdata_p_k[:, tau : tmax + tau] * np.exp(
        np.einsum("i,j->ij", kaon_fit[:, 1], t_vals)
    )
    const_factor = np.exp(pion_fit[:, 1] * tau) / np.sqrt(
        pion_fit[:, 0] * kaon_fit[:, 0] / 4
    )
    bsdata_p2k_fit1 = np.einsum("ij,i->ij", bsdata_p2k_fit, const_factor)
    plot_corr(bsdata_p2k_fit1, plotname="3pt_pi-k_ratio_fit", ylim=(0, 1.3))

    # kaon to pion with fit params divided out
    t_vals = np.arange(0, tmax)
    bsdata_k2p_fit = bsdata_k_p[:, tau : tmax + tau] * np.exp(
        np.einsum("i,j->ij", pion_fit[:, 1], t_vals)
    )
    const_factor = np.exp(kaon_fit[:, 1] * tau) / np.sqrt(
        kaon_fit[:, 0] * pion_fit[:, 0] / 4
    )
    bsdata_k2p_fit1 = np.einsum("ij,i->ij", bsdata_k2p_fit, const_factor)
    plot_corr(bsdata_k2p_fit1, plotname="3pt_k-pi_ratio_fit", ylim=(0, 1.3))

    # double ratio with fit params divided out
    # denominator = np.abs(bsdata_kaon[:,:tmax] * bsdata_pion[:,:tmax])**(-1)
    bsdata_dbl_fit = np.abs(
        bsdata_p_k[:, :tmax]
        * bsdata_k_p[:, :tmax]
        * np.exp(np.einsum("i,j->ij", pion_fit[:, 1], t_vals))
        * np.exp(np.einsum("i,j->ij", kaon_fit[:, 1], t_vals))
    )
    denominator_fit = (kaon_fit[:, 0] * pion_fit[:, 0] / 4) ** (-1)
    bsdata_dbl_fit1 = np.sqrt(np.einsum("ij,i->ij", bsdata_dbl_fit, denominator_fit))
    plot_corr(
        bsdata_dbl_fit1, plotname="3pt_pi-k_double-ratio_fit", v_line=10, ylim=(0, 1.3)
    )
    # plot_corr(bsdata_dbl_fit1, plotname="3pt_pi-k_double-ratio_fit", v_line=10)

    # ======================================================================
    # Fitting the 3pt function
    fit_corr = bsdata_7b
    const_function = ff.initffncs("Constant")
    time_limits = np.array([[[11, 15], [13, 20]]])
    datadir = Path("./")
    [fitlist_kpi] = fit_loop_new(
        fit_corr,
        [const_function],
        time_limits,
        datadir,
        "kaon_pion_3pt",
        Nt_min=2,
    )

    bestfit = fitlist_kpi[0]
    fit_time = bestfit["x"]

    time = np.arange(0, np.shape(fit_corr)[1])
    yavg = np.average(fit_corr, axis=0)
    ystd = np.std(fit_corr, axis=0)
    weighted_energy = bestfit["param"][:, 0]
    fit_redchisq = bestfit["redchisq"]
    print([fit["redchisq"] for fit in fitlist_kpi])

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
        label=rf"$|M|={err_brackets(np.average(weighted_energy), np.std(weighted_energy))}$; $\chi^2_{{\textrm{{dof}}}} = {fit_redchisq:.2f}$",
        zorder=3,
    )
    axs.axvline(10, color="k", linewidth=1, linestyle="--")

    plt.xlabel(r"$t/a$")
    # plt.ylabel(
    #     r"$\frac{\mel{\pi}{\bar{u}\gamma_{4}s}{K}}{\sqrt{4E_{\pi}E_K}}$",
    #     fontsize="small",
    # )
    # plt.ylabel(
    #     r"$\mel{\pi}{\bar{u}\gamma_{4}s}{K}$",
    #     fontsize="small",
    # )
    plt.ylabel(r"$R(t)$", fontsize="small")
    plt.legend(fontsize="x-small")
    plt.savefig("./3pt_plot_kpi_fit.pdf")
    plt.ylim(0, 0.5)
    plt.savefig("./3pt_plot_kpi_fit_zoom.pdf")
    plt.close()
    return


def main_meson2():
    """
    Read the meson 3pt functions and 2pt functions. Construct a ratio to fit and plot
    """

    mystyle = Path("mystyle.txt")
    plt.style.use(mystyle.as_posix())
    plt.rc("text.latex", preamble=r"\usepackage{physics}")

    nboot = 500
    nbin = 1

    # Open two-point function data:
    datadir = Path(
        "/Users/mbatelaan/Research/Adelaide2026/analysis/six_point_fn/data/pickles/run1_meson/"
    )
    with open(datadir / (f"time_window_loop_pion_Cosh.pkl"), "rb") as file_in:
        fitlist_pion_cosh = pickle.load(file_in)
    with open(datadir / (f"time_window_loop_kaon_Cosh.pkl"), "rb") as file_in:
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

    # Open three-point function data:
    filename_p_k = "./u-twist0.47829_s-twist0.67308_g8/meson-3pt_US_SU_s-twist/kp121040kp120620/messpec_g5-g5_495cfgs.pickle"
    with open(filename_p_k, "rb") as file_in:
        data = pickle.load(file_in)
    bsdata_p_k = bootstrap(data, config_ax=0, nboot=nboot, nbin=nbin)[:, :, 0]

    filename_k_p = "./u-twist0.47829_s-twist0.67308_g8/meson-3pt_US_SU_s-twist/kp121040kp121040/messpec_g5-g5_495cfgs.pickle"
    with open(filename_k_p, "rb") as file_in:
        data = pickle.load(file_in)
    bsdata_k_p = bootstrap(data, config_ax=0, nboot=nboot, nbin=nbin)[:, :, 0]

    filename_p = "./u-twist0.47829_s-twist0.67308_g8/meson-2pt_s-twist/kp121040kp121040/messpec_g5-g5_495cfgs.pickle"
    with open(filename_p, "rb") as file_in:
        data_pion = pickle.load(file_in)
    bsdata_pion = bootstrap(data_pion, config_ax=0, nboot=nboot, nbin=nbin)[:, :, 0]

    filename_k = "./u-twist0.47829_s-twist0.67308_g8/meson-2pt_s-twist/kp121040kp120620/messpec_g5-g5_495cfgs.pickle"
    with open(filename_k, "rb") as file_in:
        data_kaon = pickle.load(file_in)
    bsdata_kaon = bootstrap(data_kaon, config_ax=0, nboot=nboot, nbin=nbin)[:, :, 0]

    # # ======================================================================
    # # Plot the correlators
    # plot_corr(bsdata_p_k, plotname="3pt_pi-k", v_line=10)
    # plot_corr_log(bsdata_p_k, plotname="3pt_pi-k_log")
    # plot_eff_corr(bsdata_p_k, plotname="3pt_pi-k_eff", ylim=(-1, 1.5))

    # bsdata_3 = bsdata_p_k / bsdata_pion
    # plot_corr(bsdata_3, plotname="3pt_pi-k_divpion", v_line=10)

    # # bsdata_4 = bsdata_p_k / (np.einsum("ij,i->ij",bsdata_kaon, np.sqrt(bsdata_pion[:,10])) )
    # # bsdata_4 = bsdata_p_k / np.einsum("ij,i->ij",bsdata_kaon, np.sqrt(bsdata_pion[:,10]))
    # bsdata_4 = bsdata_p_k / bsdata_kaon
    # plot_corr(bsdata_4, plotname="3pt_pi-k_divkaon", v_line=10)

    # bsdata_4p1 = bsdata_k_p / bsdata_pion
    # plot_corr(bsdata_4p1, plotname="3pt_k-pi_divpion", v_line=10)

    # # bsdata_4p5 = bsdata_p_k[:,:-10] / bsdata_kaon[:,10:]
    # bsdata_4p5 = bsdata_p_k[:,10:] / bsdata_kaon[:,:-10]
    # plot_corr(bsdata_4p5, plotname="3pt_pi-k_divkaon_2", ylim=(-0.5,0.13))

    # bsdata_5 = bsdata_p_k / (np.sqrt(bsdata_pion) * np.sqrt(bsdata_kaon))
    # plot_corr(bsdata_5, plotname="3pt_pi-k_divpionkaon", v_line=10)

    # ======================================================================
    # Ratios
    tmax = 32
    tau = 10
    # energy_factor = np.sqrt(4*pion_energy * kaon_energy)
    # # energy_factor = np.sqrt(4*pion_energy * kaon_energy)**(-1)
    energy_factor = pion_energy / pion_energy
    # print(np.average(energy_factor))

    # bsdata_6 = bsdata_p_k[:,tau:tmax+tau] * np.sqrt( bsdata_kaon[:,:tmax] / (bsdata_kaon[:,tau:tmax+tau] * bsdata_pion[:,tau:tmax+tau] * bsdata_pion[:,:tmax]) )
    # # bsdata_6 = bsdata_p_k[:,tau:tmax+tau] * np.sqrt( bsdata_kaon[:,tau:tmax+tau] / ( bsdata_kaon[:,:tmax] * bsdata_pion[:,tau:tmax+tau] * bsdata_pion[:,:tmax]) )
    # const_factor = np.sqrt(bsdata_pion[:,tau] / bsdata_kaon[:,tau]) * energy_factor
    # bsdata_6b = np.einsum("ij,i->ij",bsdata_6, const_factor)
    # # plot_corr(bsdata_6b, plotname="3pt_pi-k_ratio", ylim=(0,0.6))
    # plot_corr(bsdata_6b, plotname="3pt_pi-k_ratio", ylim=(0,1.3))

    # bsdata_6p1 = bsdata_k_p[:,tau:tmax+tau] * np.sqrt( bsdata_pion[:,:tmax] / (bsdata_pion[:,tau:tmax+tau] * bsdata_kaon[:,tau:tmax+tau] * bsdata_kaon[:,:tmax]) )
    # # bsdata_6p1 = bsdata_k_p[:,tau:tmax+tau] * np.sqrt( bsdata_pion[:,tau:tmax+tau] / (bsdata_pion[:,:tmax] * bsdata_kaon[:,tau:tmax+tau] * bsdata_kaon[:,:tmax]) )
    # const_factor = np.sqrt(bsdata_kaon[:,tau] / bsdata_pion[:,tau]) * energy_factor
    # bsdata_6c = np.einsum("ij,i->ij",bsdata_6p1, const_factor)
    # # plot_corr(bsdata_6c, plotname="3pt_k-pi_ratio", ylim=(0,0.6))
    # plot_corr(bsdata_6c, plotname="3pt_k-pi_ratio", ylim=(0,1.3))

    denominator = np.abs(bsdata_kaon[:, :tmax] * bsdata_pion[:, :tmax]) ** (-1)
    bsdata_7 = np.abs(bsdata_p_k[:, :tmax] * bsdata_k_p[:, :tmax])
    bsdata_7a = np.sqrt(np.einsum("ij,ij->ij", bsdata_7, denominator))
    bsdata_7b = np.einsum("ij,i->ij", bsdata_7a, energy_factor)
    plot_corr(
        bsdata_7b, plotname="s-twist_3pt_pi-k_double-ratio", v_line=10, ylim=(0, 1.3)
    )

    # # pion to kaon with fit params divided out
    # t_vals = np.arange(0,tmax)
    # bsdata_p2k_fit = bsdata_p_k[:,tau:tmax+tau] * np.exp(np.einsum("i,j->ij",kaon_fit[:,1],t_vals))
    # const_factor = np.exp(pion_fit[:,1] * tau) / np.sqrt(pion_fit[:,0]*kaon_fit[:,0]/4)
    # bsdata_p2k_fit1 = np.einsum("ij,i->ij",bsdata_p2k_fit, const_factor)
    # plot_corr(bsdata_p2k_fit1, plotname="3pt_pi-k_ratio_fit", ylim=(0,1.3))

    # # kaon to pion with fit params divided out
    # t_vals = np.arange(0,tmax)
    # bsdata_k2p_fit = bsdata_k_p[:,tau:tmax+tau] * np.exp(np.einsum("i,j->ij",pion_fit[:,1],t_vals))
    # const_factor = np.exp(kaon_fit[:,1] * tau) / np.sqrt(kaon_fit[:,0]*pion_fit[:,0]/4)
    # bsdata_k2p_fit1 = np.einsum("ij,i->ij",bsdata_k2p_fit, const_factor)
    # plot_corr(bsdata_k2p_fit1, plotname="3pt_k-pi_ratio_fit", ylim=(0,1.3))

    # # double ratio with fit params divided out
    # # denominator = np.abs(bsdata_kaon[:,:tmax] * bsdata_pion[:,:tmax])**(-1)
    # bsdata_dbl_fit = np.abs( bsdata_p_k[:,:tmax] * bsdata_k_p[:,:tmax] * np.exp(np.einsum("i,j->ij",pion_fit[:,1],t_vals)) * np.exp(np.einsum("i,j->ij",kaon_fit[:,1],t_vals)) )
    # denominator_fit = (kaon_fit[:,0] * pion_fit[:,0]/4) ** (-1)
    # bsdata_dbl_fit1 = np.sqrt(np.einsum("ij,i->ij",bsdata_dbl_fit, denominator_fit))
    # plot_corr(bsdata_dbl_fit1, plotname="3pt_pi-k_double-ratio_fit", v_line=10, ylim=(0,1.3))
    # # plot_corr(bsdata_dbl_fit1, plotname="3pt_pi-k_double-ratio_fit", v_line=10)

    exit()

    # ======================================================================
    # Fitting the 3pt function
    fit_corr = bsdata_7b
    const_function = ff.initffncs("Constant")
    time_limits = np.array([[[11, 15], [13, 20]]])
    datadir = Path("./")
    [fitlist_kpi] = fit_loop_new(
        fit_corr,
        [const_function],
        time_limits,
        datadir,
        "kaon_pion_3pt",
        Nt_min=2,
    )

    bestfit = fitlist_kpi[0]
    fit_time = bestfit["x"]

    time = np.arange(0, np.shape(fit_corr)[1])
    yavg = np.average(fit_corr, axis=0)
    ystd = np.std(fit_corr, axis=0)
    weighted_energy = bestfit["param"][:, 0]
    fit_redchisq = bestfit["redchisq"]
    print([fit["redchisq"] for fit in fitlist_kpi])

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
        label=rf"$|M|={err_brackets(np.average(weighted_energy), np.std(weighted_energy))}$; $\chi^2_{{\textrm{{dof}}}} = {fit_redchisq:.2f}$",
        zorder=3,
    )
    axs.axvline(10, color="k", linewidth=1, linestyle="--")

    plt.xlabel(r"$t/a$")
    # plt.ylabel(
    #     r"$\frac{\mel{\pi}{\bar{u}\gamma_{4}s}{K}}{\sqrt{4E_{\pi}E_K}}$",
    #     fontsize="small",
    # )
    # plt.ylabel(
    #     r"$\mel{\pi}{\bar{u}\gamma_{4}s}{K}$",
    #     fontsize="small",
    # )
    plt.ylabel(r"$R(t)$", fontsize="small")
    plt.legend(fontsize="x-small")
    plt.savefig("./3pt_plot_kpi_fit.pdf")
    plt.ylim(0, 0.5)
    plt.savefig("./3pt_plot_kpi_fit_zoom.pdf")
    plt.close()
    return


def main_meson3():
    """
    Read the meson 3pt functions and 2pt functions. Construct a ratio to fit and plot
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

    # Open three-point function data:
    filename_p_k = "./threept_run3/meson-3pt_UU_US/messpec/32x64/slrc/kp120620kp121040/sh_gij_p21_90-sh_gij_p21_90/p+0+0+0/messpec_g5-g5_497cfgs.pickle"
    with open(filename_p_k, "rb") as file_in:
        data = pickle.load(file_in)
    bsdata_p_k = bootstrap(data, config_ax=0, nboot=nboot, nbin=nbin)[:, :, 0]

    filename_k_p = "./threept_run3/meson-3pt_SS_SU/messpec/32x64/slrc/kp121040kp121040/sh_gij_p21_90-sh_gij_p21_90/p+0+0+0/messpec_g5-g5_497cfgs.pickle"
    with open(filename_k_p, "rb") as file_in:
        data = pickle.load(file_in)
    bsdata_k_p = bootstrap(data, config_ax=0, nboot=nboot, nbin=nbin)[:, :, 0]

    filename_p_p = "./threept_run3/meson-3pt_UU_US/messpec/32x64/slrc/kp121040kp121040/sh_gij_p21_90-sh_gij_p21_90/p+0+0+0/messpec_g5-g5_497cfgs.pickle"
    with open(filename_p_p, "rb") as file_in:
        data_pion = pickle.load(file_in)
    bsdata_p_p = bootstrap(data_pion, config_ax=0, nboot=nboot, nbin=nbin)[:, :, 0]

    filename_k_k = "./threept_run3/meson-3pt_SS_SU/messpec/32x64/slrc/kp120620kp121040/sh_gij_p21_90-sh_gij_p21_90/p+0+0+0/messpec_g5-g5_497cfgs.pickle"
    with open(filename_k_k, "rb") as file_in:
        data_kaon = pickle.load(file_in)
    bsdata_k_k = bootstrap(data_kaon, config_ax=0, nboot=nboot, nbin=nbin)[:, :, 0]

    # Two-point functions
    filename_p = "./threept_run3/meson_qcdsf/messpec/32x64/slrc/kp121040kp121040/sh_gij_p21_90-sh_gij_p21_90/p+0+0+0/messpec_g5-g5_497cfgs.pickle"
    with open(filename_p, "rb") as file_in:
        data_pion = pickle.load(file_in)
    bsdata_pion = bootstrap(data_pion, config_ax=0, nboot=nboot, nbin=nbin)[:, :, 0]

    filename_k = "./threept_run3/meson_qcdsf/messpec/32x64/slrc/kp120620kp121040/sh_gij_p21_90-sh_gij_p21_90/p+0+0+0/messpec_g5-g5_497cfgs.pickle"
    with open(filename_k, "rb") as file_in:
        data_kaon = pickle.load(file_in)
    bsdata_kaon = bootstrap(data_kaon, config_ax=0, nboot=nboot, nbin=nbin)[:, :, 0]

    # ======================================================================
    # Ratios
    tmax = 32
    tau = 10
    energy_factor = np.sqrt(4 * pion_energy * kaon_energy)
    energy_factor2 = pion_energy + kaon_energy
    norm_factor = 0.863

    # Ratio with all four 3-point functions
    denominator = np.abs(bsdata_k_k[:, :tmax] * bsdata_p_p[:, :tmax]) ** (-1)
    numerator = np.abs(bsdata_p_k[:, :tmax] * bsdata_k_p[:, :tmax])
    bsdata_7a = np.sqrt(np.einsum("ij,ij->ij", numerator, denominator))
    # quad_ratio_bs = np.einsum("ij,i->ij", bsdata_7a, energy_factor / energy_factor2)
    quad_ratio_bs = np.einsum("ij,i->ij", bsdata_7a, energy_factor / norm_factor)
    plot_corr(quad_ratio_bs, plotname="3pt_pi-k_quad-ratio", plotdir=plotdir, v_line=10)

    plot_corr(
        quad_ratio_bs,
        plotname="3pt_pi-k_quad-ratio_ylim",
        plotdir=plotdir,
        v_line=10,
        ylim=(0.99, 1.02),
    )

    # Ratio with two 3-point functions divided by two 2-point functions
    denominator1 = np.abs(bsdata_kaon[:, :tmax] * bsdata_pion[:, :tmax]) ** (-1)
    bsdata_1 = np.abs(bsdata_p_k[:, :tmax] * bsdata_k_p[:, :tmax])
    bsdata_1a = np.sqrt(np.einsum("ij,ij->ij", bsdata_1, denominator1))
    bsdata_1b = np.einsum(
        # "ij,i->ij", bsdata_1a, energy_factor / energy_factor2 * norm_factor
        "ij,i->ij",
        bsdata_1a,
        energy_factor,
    )
    plot_corr(bsdata_1b, plotname="3pt_pi-k_double-ratio", plotdir=plotdir, v_line=10)
    plot_corr(
        bsdata_1b,
        plotname="3pt_pi-k_double-ratio_ylim",
        plotdir=plotdir,
        v_line=10,
        ylim=(0.95, 1.05),
        # ylim=(0.35, 0.375),
    )

    # double ratio with fit params divided out
    t_vals = np.arange(0, tmax)
    bsdata_dbl_fit = np.abs(
        bsdata_p_k[:, :tmax]
        * bsdata_k_p[:, :tmax]
        * np.exp(np.einsum("i,j->ij", pion_fit[:, 1], t_vals))
        * np.exp(np.einsum("i,j->ij", kaon_fit[:, 1], t_vals))
    )
    denominator_fit = (kaon_fit[:, 0] * pion_fit[:, 0] / 4) ** (-1)
    bsdata_dbl_fit1 = np.sqrt(np.einsum("ij,i->ij", bsdata_dbl_fit, denominator_fit))
    bsdata_dbl_fit2 = np.einsum(
        # "ij,i->ij", bsdata_dbl_fit1, energy_factor / energy_factor2 * norm_factor
        "ij,i->ij",
        bsdata_dbl_fit1,
        energy_factor,
    )
    plot_corr(
        bsdata_dbl_fit2,
        plotname="3pt_pi-k_double-ratio_fit",
        plotdir=plotdir,
        v_line=10,
        ylim=(0.95, 1.05),
        # ylim=(0.34, 0.395),
    )

    fit_correlator(bsdata_1b, plotdir, name="3pt_double_kpi_fit", ylim=(0.95, 1.05))
    fit_correlator(
        bsdata_dbl_fit2, plotdir, name="3pt_double_kpi_fitparam_fit", ylim=(0.95, 1.05)
    )
    fit_correlator(quad_ratio_bs, plotdir, name="3pt_quad_kpi_fit", ylim=(0.99, 1.02))

    return


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
    filename_p_k = "./threept_run4/meson-3pt_US_SU_u-twist/messpec/32x64/slrc/kp121040kp120620/sh_gij_p21_90-sh_gij_p21_90/p+0+0+0/messpec_g5-g5_495cfgs.pickle"
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

    filename_k_k = "./threept_run4/meson-3pt_UU_SS_u-twist_s-twist/messpec/32x64/slrc/kp121040kp120620/sh_gij_p21_90-sh_gij_p21_90/p+0+0+0/messpec_g5-g5_495cfgs.pickle"
    with open(filename_k_k, "rb") as file_in:
        data_kaon = pickle.load(file_in)
    bsdata_k_k = bootstrap(data_kaon, config_ax=0, nboot=nboot, nbin=nbin)[:, :, 0]

    # Two-point functions
    filename_p = "./threept_run4/meson_qcdsf/messpec/32x64/slrc/kp121040kp121040/sh_gij_p21_90-sh_gij_p21_90/p+0+0+0/messpec_g5-g5_495cfgs.pickle"
    with open(filename_p, "rb") as file_in:
        data_pion = pickle.load(file_in)
    bsdata_pion = bootstrap(data_pion, config_ax=0, nboot=nboot, nbin=nbin)[:, :, 0]

    filename_k = "./threept_run4/meson_qcdsf/messpec/32x64/slrc/kp121040kp120620/sh_gij_p21_90-sh_gij_p21_90/p+0+0+0/messpec_g5-g5_495cfgs.pickle"
    with open(filename_k, "rb") as file_in:
        data_kaon = pickle.load(file_in)
    bsdata_kaon = bootstrap(data_kaon, config_ax=0, nboot=nboot, nbin=nbin)[:, :, 0]

    # Two-point functions with twist
    filename_p = "./threept_run4/meson-2pt_TBC/messpec/32x64/slrc/kp121040kp121040/sh_gij_p21_90-sh_gij_p21_90/p+0+0+0/messpec_g5-g5_495cfgs.pickle"
    with open(filename_p, "rb") as file_in:
        data_pion = pickle.load(file_in)
    bsdata_pion_twist = bootstrap(data_pion, config_ax=0, nboot=nboot, nbin=nbin)[
        :, :, 0
    ]

    filename_k = "./threept_run4/meson-2pt_TBC/messpec/32x64/slrc/kp121040kp120620/sh_gij_p21_90-sh_gij_p21_90/p+0+0+0/messpec_g5-g5_495cfgs.pickle"
    with open(filename_k, "rb") as file_in:
        data_kaon = pickle.load(file_in)
    bsdata_kaon_twist = bootstrap(data_kaon, config_ax=0, nboot=nboot, nbin=nbin)[
        :, :, 0
    ]

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
    plot_corr(
        quad_ratio_bs, plotname="3pt_pi-k_quad-ratio_twist", plotdir=plotdir, v_line=10
    )
    plot_corr(
        quad_ratio_bs,
        plotname="3pt_pi-k_quad-ratio_ylim_twist",
        plotdir=plotdir,
        v_line=10,
        ylim=(0.35, 0.5),
    )

    # ----------------------------------------------------------------------
    # Ratio with two 3-point functions divided by two 2-point functions
    denominator1 = np.abs(bsdata_kaon[:, :tmax] * bsdata_pion_twist[:, :tmax]) ** (-1)
    bsdata_1 = np.abs(bsdata_p_k[:, :tmax] * bsdata_k_p[:, :tmax])
    bsdata_1a = np.sqrt(np.einsum("ij,ij->ij", bsdata_1, denominator1))
    bsdata_1b = np.einsum(
        # "ij,i->ij", bsdata_1a, energy_factor / energy_factor2 * norm_factor
        "ij,i->ij",
        bsdata_1a,
        energy_factor,
    )
    plot_corr(
        bsdata_1b, plotname="3pt_pi-k_double-ratio_twist", plotdir=plotdir, v_line=10
    )
    plot_corr(
        bsdata_1b,
        plotname="3pt_pi-k_double-ratio_twist_ylim",
        plotdir=plotdir,
        v_line=10,
        ylim=(0.2, 0.37),
    )

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
    #     plotname="3pt_pi-k_double-ratio_fit_twist",
    #     plotdir=plotdir,
    #     v_line=10,
    #     # ylim=(0.95, 1.05),
    # )

    fit_correlator(
        bsdata_1b, plotdir, name="3pt_double_kpi_fit_twist", ylim=(0.2, 0.37)
    )
    fit_correlator(
        quad_ratio_bs, plotdir, name="3pt_quad_kpi_fit_twist", ylim=(0.35, 0.5)
    )

    # fit_correlator(
    #     bsdata_dbl_fit2, plotdir, name="3pt_double_kpi_fitparam_fit_twist"
    # )

    return


def gevp_bootstrap_mesons(
    corr_matrix, time_choice=10, delta_t=1, prev_evecs=None, name="", show=None
):
    """Solve the GEVP for a given correlation matrix

    corr_matrix has the matrix indices as the first two, then the bootstrap index and then the time index
    time_choice is the timeslice on which the GEVP will be set
    delta_t is the size of the time evolution which will be used to solve the GEVP
    """

    # Apply the GEVP to the ensemble average to project the states we will use for extracting the energy shift.
    mat_0_avg = np.average(corr_matrix[:, :, :, time_choice], axis=2)
    mat_1_avg = np.average(corr_matrix[:, :, :, time_choice + delta_t], axis=2)
    eval_left_avg, evec_left_avg = np.linalg.eig(
        np.matmul(mat_1_avg, np.linalg.inv(mat_0_avg)).T
    )
    eval_right_avg, evec_right_avg = np.linalg.eig(
        np.matmul(np.linalg.inv(mat_0_avg), mat_1_avg)
    )

    print("GEVP matrices: ")
    print(mat_0_avg)
    print("")
    print(mat_1_avg)
    print("")

    # Ordering of the eigenvalues
    if eval_left_avg[0] < eval_left_avg[1]:
        eval_left_avg = eval_left_avg[::-1]
        evec_left_avg = evec_left_avg[:, ::-1]
    if eval_right_avg[0] < eval_right_avg[1]:
        eval_right_avg = eval_right_avg[::-1]
        evec_right_avg = evec_right_avg[:, ::-1]

    # # Ordering of the eigenvalues
    # if eval_left_avg[0] > eval_left_avg[1]:
    #     eval_left_avg = eval_left_avg[::-1]
    #     evec_left_avg = evec_left_avg[:, ::-1]
    # if eval_right_avg[0] > eval_right_avg[1]:
    #     eval_right_avg = eval_right_avg[::-1]
    #     evec_right_avg = evec_right_avg[:, ::-1]

    evec_left_list = []
    evec_right_list = []
    eval_left_list = []
    eval_right_list = []
    nboot = np.shape(corr_matrix)[2]

    for boot in range(nboot):
        mat_0 = corr_matrix[:, :, boot, time_choice]
        mat_1 = corr_matrix[:, :, boot, time_choice + delta_t]

        eval_left, evec_left = np.linalg.eig(np.matmul(mat_1, np.linalg.inv(mat_0)).T)
        eval_right, evec_right = np.linalg.eig(np.matmul(np.linalg.inv(mat_0), mat_1))

        # # Ordering of the eigenvalues
        if eval_left[0] > eval_left[1]:
            eval_left = eval_left[::-1]
            evec_left = evec_left[:, ::-1]
        if eval_right[0] > eval_right[1]:
            eval_right = eval_right[::-1]
            evec_right = evec_right[:, ::-1]

        evec_left_list.append(evec_left)
        evec_right_list.append(evec_right)
        eval_left_list.append(eval_left)
        eval_right_list.append(eval_right)

    # Complex corelators
    proj_corr = []
    for i in range(len(eval_left_avg)):
        Gti = np.einsum(
            "i,ijkl,j->kl",
            evec_left_avg[:, i],
            corr_matrix,
            evec_right_avg[:, i],
        )
        proj_corr.append(Gti)
    # # Complex corelators
    # Gt1 = np.einsum(
    #     "i,ijkl,j->kl", evec_left_avg[:, 0], corr_matrix, evec_right_avg[:, 0]
    # )
    # Gt2 = np.einsum(
    #     "i,ijkl,j->kl", evec_left_avg[:, 1], corr_matrix, evec_right_avg[:, 1]
    # )
    # if show:
    #     stats.ploteffmass(Gt1, "eig_1" + name, plotdir, show=True)
    #     stats.ploteffmass(Gt2, "eig_2" + name, plotdir, show=True)

    gevp_data = [
        np.array(eval_left_avg),
        np.array(evec_left_avg),
        np.array(eval_right_avg),
        np.array(evec_right_avg),
    ]
    return proj_corr, gevp_data
    # return Gt1, Gt2, gevp_data


def plot_eff_proj_corrs(
    corrs,
    plotdir,
    plot_name="",
    labels=[r"$C_1$", r"$C_1$"],
    ylim=None,
    title="",
    ylabel=r"$R(t)$",
):

    spacing = 1

    corr_list = []
    time_list = []
    for corr in corrs:
        real_1 = np.real(corr)
        time = np.arange(0, np.shape(corr)[1])
        effmass_1 = stats.bs_effmass(real_1, time_axis=1, spacing=spacing)
        efftime = time[:-spacing] + 0.5
        corr_list.append(effmass_1)
        time_list.append(efftime)

    # ======================================================================
    # Plotting the principal correlators
    f, axs = plt.subplots(1, 1, figsize=(7, 5), sharex=True, sharey=True)
    for icorr, corr in enumerate(corr_list):
        plt.errorbar(
            time_list[icorr],
            np.average(corr, axis=0),
            np.std(corr, axis=0),
            capsize=4,
            elinewidth=1,
            color=_colors[icorr],
            fmt=_markers[icorr],
            ms=4,
            mfc="white",
            label=labels[icorr],
            zorder=1,
        )

    plt.grid(True, alpha=0.3)
    plt.legend(fontsize="x-small")
    plt.xlabel(r"$t/a$")
    # plt.ylabel(r"$C_n(t)$")
    # plt.ylabel(r"$R(t)$")
    plt.ylabel(ylabel)
    plt.title(title)
    # plt.ylim(-0.36, 0.36)

    if ylim is not None:
        plt.ylim(ylim)

    plt.tight_layout()
    plt.savefig(
        plotdir / (f"{plot_name}.pdf"),
        metadata=_metadata,
    )
    # plt.yscale("log")
    # plt.savefig(
    #     plotdir / (f"GEVP_proj_corrs_{plot_name}_log.pdf"),
    #     metadata=_metadata,
    # )
    plt.close()

    return


def meson_2pt_projections_big(
    pickledir,
    plotdir,
    datadir,
    pion_dir="meson_qcdsf",
    kaon_dir="meson_qcdsf",
    time_choice=29,
    delta_t=6,
    label="",
    prev_evecs=None,
    kappa_1="kp121040kp121040",
    kappa_2="kp120620kp121040",
):
    """
    Read in the two-point functions for the pion and kaon, construct the correlator matrices and do the GEVP to get the eigenvectors, eigenvalues and the projected correlators.
    """

    nboot = 500
    nbin = 1

    conf_num_u = "495"
    dirpart_u = f"slrc/{kappa_1}/"
    dirpart_s = f"slrc/{kappa_2}/"

    # ------------------------------------------------------------
    # Find conf number
    test_file = (
        pickledir
        / Path(
            f"{pion_dir}/messpec/32x64/{dirpart_u}sh_gij_p21_90-sh_gij_p21_90/p+0+0+0/"
        )
    ).glob("messpec_g5-g5_*cfgs.pickle")

    conf_num_u = [
        int("".join(filter(str.isdigit, l.name.split("_")[-1])))
        for l in list(test_file)
    ][0]
    print(f"conf_num = {conf_num_u}")

    ps = 5
    a4 = 53
    a2 = 51
    opnames = [ps, a4, a2]

    for iop, op1 in enumerate(opnames):
        for jop, op2 in enumerate(opnames):
            sign = 1.0
            if iop == 1:
                sign = sign * 1.0j
            if jop == 1:
                sign = sign * 1.0j

            # pions
            pion_file = pickledir / Path(
                f"{pion_dir}/messpec/32x64/{dirpart_u}sh_gij_p21_90-sh_gij_p21_90/p+0+0+0/messpec_g{op1}-g{op2}_{conf_num_u}cfgs.pickle"
            )
            pion_data = read_pickle(pion_file, nboot=nboot, nbin=nbin)
            pion_complex = pion_data[:, :, 0] + 1j * pion_data[:, :, 1]
            if iop == 0 and jop == 0:
                corr_matrix_pions = np.empty(
                    (
                        len(opnames),
                        len(opnames),
                        pion_complex.shape[0],
                        pion_complex.shape[1],
                    ),
                    dtype=complex,
                )
            corr_matrix_pions[iop, jop] = sign * pion_complex

            # kaons
            kaon_file = pickledir / Path(
                f"{kaon_dir}/messpec/32x64/{dirpart_s}sh_gij_p21_90-sh_gij_p21_90/p+0+0+0/messpec_g{op1}-g{op2}_{conf_num_u}cfgs.pickle"
            )
            kaon_data = read_pickle(kaon_file, nboot=nboot, nbin=nbin)
            kaon_complex = kaon_data[:, :, 0] + 1j * kaon_data[:, :, 1]
            if iop == 0 and jop == 0:
                corr_matrix_kaons = np.empty(
                    (
                        len(opnames),
                        len(opnames),
                        kaon_complex.shape[0],
                        kaon_complex.shape[1],
                    ),
                    dtype=complex,
                )
            corr_matrix_kaons[iop, jop] = sign * kaon_complex

    # ------------------------------------------------------------
    # Hermitianize
    # pions_herm = np.swapaxes(
    #     np.concatenate(
    #         (corr_matrix_pions[:, :, :, :1], corr_matrix_pions[:, :, :, :0:-1]), axis=3
    #     ),
    #     0,
    #     1,
    # ).conj()
    # np.swapaxes(
    #     np.concatenate(
    #         (corr_matrix_kaons[:, :, :, :1], corr_matrix_kaons[:, :, :, :0:-1]), axis=3
    #     ),
    #     0,
    #     1,
    # ).conj()

    pions_herm = np.swapaxes(corr_matrix_pions, 0, 1).conj()
    kaons_herm = np.swapaxes(corr_matrix_kaons, 0, 1).conj()
    corr_matrix_pions = 0.5 * (corr_matrix_pions + pions_herm)
    corr_matrix_kaons = 0.5 * (corr_matrix_kaons + kaons_herm)

    # # ------------------------------------------------------------
    # # Normalize the matrices
    # corr_matrix_pions = normalize_matrices([corr_matrix_pions], time_choice=3)[0]
    # corr_matrix_kaons = normalize_matrices([corr_matrix_kaons], time_choice=3)[0]

    # ------------------------------------------------------------
    # Pion GEVP
    [
        Gt1_pion,
        Gt2_pion,
        Gt3_pion,
    ], [
        eval_left_pion,
        evec_left_pion,
        eval_right_pion,
        evec_right_pion,
    ] = gevp_bootstrap_mesons(
        corr_matrix_pions,
        time_choice,
        delta_t,
        prev_evecs=prev_evecs,
        name="_test",
        # show=False,
        show=True,
    )

    # ----------------------------------------------------------------------
    pion_mat_avg = np.average(corr_matrix_pions[:, :, :, time_choice], axis=2)
    corr_tZ = np.average(corr_matrix_pions[:, :, :, time_choice + 2], axis=2)
    S = np.diag([1, -1, 1])
    v0 = evec_right_pion[:, 0]
    v0_renorm = v0 / np.sqrt(np.einsum("i,ij,j->", v0.conj(), corr_tZ, v0))

    zF = corr_tZ @ v0_renorm
    b0 = S @ zF
    PB_perp = np.eye(3, dtype=complex) - np.outer(b0, b0.conj()) / np.vdot(b0, b0)
    v_simple = PB_perp @ zF
    norm2 = np.real(np.vdot(v_simple, pion_mat_avg @ v_simple))
    v_simple /= np.sqrt(norm2)

    evec_right_pion[:, 0] = v_simple
    evec_left_pion[:, 0] = v_simple.conj()
    Gt1_pion = np.einsum(
        "i,ijkl,j->kl",
        v_simple.conj(),
        corr_matrix_pions,
        v_simple,
    )

    # Plotting
    plot_eff_proj_corrs(
        [Gt1_pion, Gt2_pion, Gt3_pion],
        plotdir,
        plot_name=f"pion_GEVP_proj_{label}",
        labels=[
            r"corr 1",
            r"corr 2",
            r"corr 3",
        ],
        ylim=(-0.8, 0.8),
    )

    # ======================================================================
    # Kaon GEVP

    [
        Gt1_kaon,
        Gt2_kaon,
        Gt3_kaon,
    ], [
        eval_left_kaon,
        evec_left_kaon,
        eval_right_kaon,
        evec_right_kaon,
    ] = gevp_bootstrap_mesons(
        corr_matrix_kaons,
        time_choice,
        delta_t,
        prev_evecs=prev_evecs,
        name="_test",
        show=False,
    )

    # ------------------------------------------------------------
    # Testing
    kaon_mat_avg = np.average(corr_matrix_kaons[:, :, :, time_choice], axis=2)
    corr_tZ = np.average(corr_matrix_kaons[:, :, :, time_choice + 2], axis=2)
    S = np.diag([1, -1, 1])
    v0 = evec_right_kaon[:, 0]
    v0_renorm = v0 / np.sqrt(np.einsum("i,ij,j->", v0.conj(), corr_tZ, v0))

    zF = corr_tZ @ v0_renorm
    b0 = S @ zF
    PB_perp = np.eye(3, dtype=complex) - np.outer(b0, b0.conj()) / np.vdot(b0, b0)
    v_simple = PB_perp @ zF
    norm2 = np.real(np.vdot(v_simple, kaon_mat_avg @ v_simple))
    v_simple /= np.sqrt(norm2)

    evec_right_kaon[:, 0] = v_simple
    evec_left_kaon[:, 0] = v_simple.conj()
    Gt1_kaon = np.einsum(
        "i,ijkl,j->kl",
        v_simple.conj(),
        corr_matrix_kaons,
        v_simple,
    )

    # # ======================================================================
    # # FIRST TEST
    # kaon_mat_avg = np.average(corr_matrix_kaons[:, :, :, time_choice], axis=2)
    # # z_0_F = np.einsum(
    # #     "ij,j->i",
    # #     kaon_mat_avg,
    # #     evec_right_kaon[:, 0],
    # # )
    # S_mat = np.diag([1, -1, 1])
    # # z0F_norm = z_0_F / np.linalg.norm(z_0_F)
    # # z0B = S_mat @ z_0_F

    # # print(evec_right_kaon[:, 0])
    # # print(z_0_F)
    # # print(z0F_norm)
    # # print(z0B)

    # v0 = evec_right_kaon[:, 0]
    # v0_renorm = v0 / np.sqrt(np.einsum("i,ij,j->", v0.conj(), kaon_mat_avg, v0))
    # # v0_renorm = np.asarray(v0_renorm).reshape(-1)
    # b0 = np.einsum(
    #     "ki,ij,j->i",
    #     S_mat,
    #     kaon_mat_avg,
    #     v0_renorm,
    #     # v0,
    # )

    # # chat
    # b0 = np.asarray(b0).reshape(-1)
    # # v0 = np.asarray(v0_renorm).reshape(-1)
    # v0 = np.asarray(v0).reshape(-1)

    # denom = np.vdot(b0, b0)
    # PB_perp = np.eye(b0.size, dtype=complex) - np.outer(b0, b0.conj()) / denom

    # vnull = PB_perp @ v0

    # norm2 = np.abs(np.vdot(vnull, kaon_mat_avg @ vnull))
    # vnull_renorm = vnull / np.sqrt(norm2)
    # # vnull_renorm = vnull

    # test = np.vdot(b0, vnull_renorm)

    # print("vnull:")
    # print(vnull)
    # print(vnull_renorm)
    # print("b0† vnull:", test)
    # print(
    #     "relative residual:",
    #     abs(test) / (np.linalg.norm(b0) * np.linalg.norm(vnull_renorm)),
    # )
    # print("v0:")
    # print(evec_right_kaon[:, 0])

    # evec_right_kaon[:, 0] = vnull_renorm
    # evec_left_kaon[:, 0] = vnull_renorm.conj()

    # Gt1_kaon = np.einsum(
    #     "i,ijkl,j->kl",
    #     vnull_renorm.conj(),
    #     corr_matrix_kaons,
    #     vnull_renorm,
    # )

    # # Me
    # # PB_perp = np.diag([1, 1, 1]) - (z0B @ z0B.conj().T) / (z0B.conj().T @ z0B)
    # PB_perp = np.diag([1, 1, 1]) - np.outer(b0, b0.conj()) / np.vdot(b0, b0)
    # vnull = PB_perp @ v0_renorm
    # # norm2 = np.real(np.vdot(vnull, kaon_mat_avg @ vnull))
    # # vnull_renorm = vnull / np.sqrt(norm2)
    # vnull_renorm = vnull / np.sqrt(
    #     np.einsum("i,ij,j->", vnull.conj().T, kaon_mat_avg, vnull)
    # )

    # print("vnull")
    # print(vnull_renorm)
    # # test = b0.conj().T @ vnull
    # test = np.vdot(b0, vnull_renorm)
    # print(test)
    # print(np.abs(test))

    # exit()
    # ------------------------------------------------------------

    # Plotting
    plot_eff_proj_corrs(
        [Gt1_kaon, Gt2_kaon, Gt3_kaon],
        plotdir,
        plot_name=f"kaon_GEVP_proj_{label}",
        labels=[
            r"corr 1",
            r"corr 2",
            r"corr 3",
        ],
        ylim=(-0.8, 0.8),
    )

    pion_ps_ps = corr_matrix_pions[0, 0]
    pion_a_a = corr_matrix_pions[1, 1]
    pion_a2_a2 = corr_matrix_pions[2, 2]
    kaon_ps_ps = corr_matrix_kaons[0, 0]
    kaon_a_a = corr_matrix_kaons[1, 1]
    kaon_a2_a2 = corr_matrix_kaons[2, 2]

    # Plotting
    if label == "0twist":
        title1 = r"$\mathbf{p}^{\pi} = \mathbf{0}$"
        title2 = r"$\mathbf{p}^{K} = \mathbf{0}$"
    else:
        title1 = r"$\mathbf{p}^{\pi}_2 = \mathbf{\theta}_2$"
        title2 = r"$\mathbf{p}^{K}_2 = \mathbf{\theta}_2$"
    plot_eff_proj_corrs(
        [pion_ps_ps, pion_a_a, pion_a2_a2],
        plotdir,
        plot_name=f"pion_GEVP_ops_{label}",
        labels=[
            r"$\pi_{\gamma_5,\gamma_5}$",
            r"$\pi_{\gamma_4\gamma_5,\gamma_4\gamma_5}$",
            r"$\pi_{\gamma_2\gamma_5,\gamma_2\gamma_5}$",
        ],
        title=title1,
        ylim=(-0.8, 0.8),
    )
    # Plotting
    plot_eff_proj_corrs(
        [kaon_ps_ps, kaon_a_a, kaon_a2_a2],
        plotdir,
        plot_name=f"kaon_GEVP_ops_{label}",
        labels=[
            r"$K_{\gamma_5,\gamma_5}$",
            r"$K_{\gamma_4\gamma_5,\gamma_4\gamma_5}$",
            r"$K_{\gamma_2\gamma_5,\gamma_2\gamma_5}$",
        ],
        title=title2,
        ylim=(-0.8, 0.8),
    )

    # print("evals = ")
    # print(np.real(eval_left_pion))
    # print(np.real(eval_left_kaon))

    return (
        [eval_left_pion, eval_right_pion, eval_left_kaon, eval_right_kaon],
        [evec_left_pion, evec_right_pion, evec_left_kaon, evec_right_kaon],
        [
            Gt1_pion,
            Gt1_kaon,
            Gt2_pion,
            Gt2_kaon,
            Gt3_pion,
            Gt3_kaon,
        ],
        [
            pion_ps_ps,
            kaon_ps_ps,
            pion_a_a,
            kaon_a_a,
            pion_a2_a2,
            kaon_a2_a2,
        ],
    )


def meson_2pt_projections(
    pickledir,
    plotdir,
    datadir,
    pion_dir="meson_qcdsf",
    kaon_dir="meson_qcdsf",
    time_choice=29,
    delta_t=6,
    label="",
    prev_evecs=None,
    kappa_1="kp121040kp121040",
    kappa_2="kp120620kp121040",
):
    """
    Read in the two-point functions for the pion and kaon, construct the correlator matrices and do the GEVP to get the eigenvectors, eigenvalues and the projected correlators.
    """

    nboot = 500
    nbin = 1

    conf_num_u = "495"
    dirpart_u = f"slrc/{kappa_1}/"
    dirpart_s = f"slrc/{kappa_2}/"

    # ------------------------------------------------------------
    # Find conf number
    test_file = (
        pickledir
        / Path(
            f"{pion_dir}/messpec/32x64/{dirpart_u}sh_gij_p21_90-sh_gij_p21_90/p+0+0+0/"
        )
    ).glob("messpec_g5-g5_*cfgs.pickle")

    conf_num_u = [
        int("".join(filter(str.isdigit, l.name.split("_")[-1])))
        for l in list(test_file)
    ][0]
    print(f"conf_num = {conf_num_u}")

    ps = 5
    a4 = 53
    # a2 = 51
    opnames = [ps, a4]

    for iop, op1 in enumerate(opnames):
        for jop, op2 in enumerate(opnames):
            # pions
            sign = 1.0
            if iop == 1:
                sign = sign * 1.0j
            if jop == 1:
                sign = sign * 1.0j

            pion_file = pickledir / Path(
                f"{pion_dir}/messpec/32x64/{dirpart_u}sh_gij_p21_90-sh_gij_p21_90/p+0+0+0/messpec_g{op1}-g{op2}_{conf_num_u}cfgs.pickle"
            )
            pion_data = read_pickle(pion_file, nboot=nboot, nbin=nbin)
            pion_complex = pion_data[:, :, 0] + 1j * pion_data[:, :, 1]
            if iop == 0 and jop == 0:
                corr_matrix_pions = np.empty(
                    (
                        len(opnames),
                        len(opnames),
                        pion_complex.shape[0],
                        pion_complex.shape[1],
                    ),
                    dtype=complex,
                )
            corr_matrix_pions[iop, jop] = sign * pion_complex

            # kaons
            kaon_file = pickledir / Path(
                f"{kaon_dir}/messpec/32x64/{dirpart_s}sh_gij_p21_90-sh_gij_p21_90/p+0+0+0/messpec_g{op1}-g{op2}_{conf_num_u}cfgs.pickle"
            )
            kaon_data = read_pickle(kaon_file, nboot=nboot, nbin=nbin)
            kaon_complex = kaon_data[:, :, 0] + 1j * kaon_data[:, :, 1]
            if iop == 0 and jop == 0:
                corr_matrix_kaons = np.empty(
                    (
                        len(opnames),
                        len(opnames),
                        kaon_complex.shape[0],
                        kaon_complex.shape[1],
                    ),
                    dtype=complex,
                )
            corr_matrix_kaons[iop, jop] = sign * kaon_complex

    # # ------------------------------------------------------------
    # Pion GEVP

    [
        Gt1_pion,
        Gt2_pion,
    ], [
        eval_left_pion,
        evec_left_pion,
        eval_right_pion,
        evec_right_pion,
    ] = gevp_bootstrap_mesons(
        corr_matrix_pions,
        time_choice,
        delta_t,
        prev_evecs=prev_evecs,
        name="_test",
        # show=False,
        show=True,
    )

    # Plotting
    plot_eff_proj_corrs(
        [Gt1_pion, Gt2_pion],
        plotdir,
        plot_name=f"pion_GEVP_proj_{label}",
        labels=[
            r"corr 1",
            r"corr 2",
            r"corr 3",
        ],
        ylim=(-0.8, 0.8),
    )

    # ======================================================================
    # Kaon GEVP

    [
        Gt1_kaon,
        Gt2_kaon,
    ], [
        eval_left_kaon,
        evec_left_kaon,
        eval_right_kaon,
        evec_right_kaon,
    ] = gevp_bootstrap_mesons(
        corr_matrix_kaons,
        time_choice,
        delta_t,
        prev_evecs=prev_evecs,
        name="_test",
        show=False,
    )

    # Plotting
    plot_eff_proj_corrs(
        [Gt1_kaon, Gt2_kaon],
        plotdir,
        plot_name=f"kaon_GEVP_proj_{label}",
        labels=[
            r"corr 1",
            r"corr 2",
            r"corr 3",
        ],
        ylim=(-0.8, 0.8),
    )

    pion_ps_ps = corr_matrix_pions[0, 0]
    pion_a_a = corr_matrix_pions[1, 1]
    kaon_ps_ps = corr_matrix_kaons[0, 0]
    kaon_a_a = corr_matrix_kaons[1, 1]

    print("evals = ")
    # print(np.real(eval_left_pion))
    # print(np.real(eval_left_kaon))

    return (
        [eval_left_pion, eval_right_pion, eval_left_kaon, eval_right_kaon],
        [evec_left_pion, evec_right_pion, evec_left_kaon, evec_right_kaon],
        [
            Gt1_pion,
            Gt1_kaon,
            Gt2_pion,
            Gt2_kaon,
        ],
        [
            pion_ps_ps,
            kaon_ps_ps,
            pion_a_a,
            kaon_a_a,
        ],
    )


def meson_3pt_projections(
    pickledir,
    plotdir,
    datadir,
    evecs_pion,
    evecs_kaon,
    p_k_dir="meson_qcdsf",
    k_p_dir="meson_qcdsf",
    p_p_dir="meson_qcdsf",
    k_k_dir="meson_qcdsf",
    time_choice=29,
    delta_t=6,
    label="",
    # switch_kappa=False,
    kappa_1="kp121040kp121040",
    kappa_2="kp120620kp121040",
    # kappa_2="kp121040kp120620",
):
    print(np.shape(evecs_pion))
    print(np.shape(evecs_kaon))
    # GEVP
    # time_choice = 25
    # delta_t = 4
    nboot = 500
    nbin = 1
    prev_evecs = None

    conf_num_u = "495"
    dirpart_u = f"slrc/{kappa_1}/"
    dirpart_s = f"slrc/{kappa_2}/"
    # if switch_kappa:
    #     dirpart_s = "slrc/kp121040kp120620/"
    # else:
    #     dirpart_s = "slrc/kp120620kp121040/"

    # ------------------------------------------------------------
    # Find conf number
    test_file = (
        pickledir
        / Path(
            f"{p_k_dir}/messpec/32x64/{dirpart_u}sh_gij_p21_90-sh_gij_p21_90/p+0+0+0/"
        )
    ).glob("messpec_g5-g5_*cfgs.pickle")
    conf_num_u = [
        int("".join(filter(str.isdigit, l.name.split("_")[-1])))
        for l in list(test_file)
    ][0]
    print(f"conf_num = {conf_num_u}")

    pvec_pvec_name = f"/messpec_g5-g5_{conf_num_u}cfgs.pickle"
    pvec_axial_name = f"/messpec_g5-g53_{conf_num_u}cfgs.pickle"
    axial_pvec_name = f"/messpec_g53-g5_{conf_num_u}cfgs.pickle"
    axial_axial_name = f"/messpec_g53-g53_{conf_num_u}cfgs.pickle"

    # ------------------------------------------------------------
    # pion to kaon
    filename_p_k_PS_PS = pickledir / Path(
        f"{p_k_dir}/messpec/32x64/{dirpart_s}/sh_gij_p21_90-sh_gij_p21_90/p+0+0+0/{pvec_pvec_name}"
    )
    filename_p_k_A_PS = pickledir / Path(
        f"{p_k_dir}/messpec/32x64/{dirpart_s}/sh_gij_p21_90-sh_gij_p21_90/p+0+0+0/{axial_pvec_name}"
    )
    filename_p_k_PS_A = pickledir / Path(
        f"{p_k_dir}/messpec/32x64/{dirpart_s}/sh_gij_p21_90-sh_gij_p21_90/p+0+0+0/{pvec_axial_name}"
    )
    filename_p_k_A_A = pickledir / Path(
        f"{p_k_dir}/messpec/32x64/{dirpart_s}/sh_gij_p21_90-sh_gij_p21_90/p+0+0+0/{axial_axial_name}"
    )

    p_k_pvec_pvec = read_pickle(filename_p_k_PS_PS, nboot=nboot, nbin=nbin)
    p_k_pvec_axial = read_pickle(filename_p_k_PS_A, nboot=nboot, nbin=nbin)
    p_k_axial_pvec = read_pickle(filename_p_k_A_PS, nboot=nboot, nbin=nbin)
    p_k_axial_axial = read_pickle(filename_p_k_A_A, nboot=nboot, nbin=nbin)

    p_k_ps_ps = p_k_pvec_pvec[:, :, 0] + 1j * p_k_pvec_pvec[:, :, 1]
    p_k_a_a = p_k_axial_axial[:, :, 0] + 1j * p_k_axial_axial[:, :, 1]
    p_k_ps_a = p_k_pvec_axial[:, :, 0] + 1j * p_k_pvec_axial[:, :, 1]
    p_k_a_ps = p_k_axial_pvec[:, :, 0] + 1j * p_k_axial_pvec[:, :, 1]

    # ------------------------------------------------------------
    # kaon to pion
    filename_k_p_PS_PS = pickledir / Path(
        f"{k_p_dir}/messpec/32x64/{dirpart_u}/sh_gij_p21_90-sh_gij_p21_90/p+0+0+0/{pvec_pvec_name}"
    )
    filename_k_p_A_PS = pickledir / Path(
        f"{k_p_dir}/messpec/32x64/{dirpart_u}/sh_gij_p21_90-sh_gij_p21_90/p+0+0+0/{axial_pvec_name}"
    )
    filename_k_p_PS_A = pickledir / Path(
        f"{k_p_dir}/messpec/32x64/{dirpart_u}/sh_gij_p21_90-sh_gij_p21_90/p+0+0+0/{pvec_axial_name}"
    )
    filename_k_p_A_A = pickledir / Path(
        f"{k_p_dir}/messpec/32x64/{dirpart_u}/sh_gij_p21_90-sh_gij_p21_90/p+0+0+0/{axial_axial_name}"
    )

    k_p_pvec_pvec = read_pickle(filename_k_p_PS_PS, nboot=nboot, nbin=nbin)
    k_p_pvec_axial = read_pickle(filename_k_p_PS_A, nboot=nboot, nbin=nbin)
    k_p_axial_pvec = read_pickle(filename_k_p_A_PS, nboot=nboot, nbin=nbin)
    k_p_axial_axial = read_pickle(filename_k_p_A_A, nboot=nboot, nbin=nbin)

    k_p_ps_ps = k_p_pvec_pvec[:, :, 0] + 1j * k_p_pvec_pvec[:, :, 1]
    k_p_a_a = k_p_axial_axial[:, :, 0] + 1j * k_p_axial_axial[:, :, 1]
    k_p_ps_a = k_p_pvec_axial[:, :, 0] + 1j * k_p_pvec_axial[:, :, 1]
    k_p_a_ps = k_p_axial_pvec[:, :, 0] + 1j * k_p_axial_pvec[:, :, 1]

    # ------------------------------------------------------------
    # pion to pion
    filename_p_p_PS_PS = pickledir / Path(
        f"{p_p_dir}/messpec/32x64/{dirpart_u}/sh_gij_p21_90-sh_gij_p21_90/p+0+0+0/{pvec_pvec_name}"
    )
    filename_p_p_A_PS = pickledir / Path(
        f"{p_p_dir}/messpec/32x64/{dirpart_u}/sh_gij_p21_90-sh_gij_p21_90/p+0+0+0/{axial_pvec_name}"
    )
    filename_p_p_PS_A = pickledir / Path(
        f"{p_p_dir}/messpec/32x64/{dirpart_u}/sh_gij_p21_90-sh_gij_p21_90/p+0+0+0/{pvec_axial_name}"
    )
    filename_p_p_A_A = pickledir / Path(
        f"{p_p_dir}/messpec/32x64/{dirpart_u}/sh_gij_p21_90-sh_gij_p21_90/p+0+0+0/{axial_axial_name}"
    )

    p_p_pvec_pvec = read_pickle(filename_p_p_PS_PS, nboot=nboot, nbin=nbin)
    p_p_pvec_axial = read_pickle(filename_p_p_PS_A, nboot=nboot, nbin=nbin)
    p_p_axial_pvec = read_pickle(filename_p_p_A_PS, nboot=nboot, nbin=nbin)
    p_p_axial_axial = read_pickle(filename_p_p_A_A, nboot=nboot, nbin=nbin)

    p_p_ps_ps = p_p_pvec_pvec[:, :, 0] + 1j * p_p_pvec_pvec[:, :, 1]
    p_p_a_a = p_p_axial_axial[:, :, 0] + 1j * p_p_axial_axial[:, :, 1]
    p_p_ps_a = p_p_pvec_axial[:, :, 0] + 1j * p_p_pvec_axial[:, :, 1]
    p_p_a_ps = p_p_axial_pvec[:, :, 0] + 1j * p_p_axial_pvec[:, :, 1]

    # ------------------------------------------------------------
    # kaon to kaon
    filename_k_k_PS_PS = pickledir / Path(
        f"{k_k_dir}/messpec/32x64/{dirpart_s}/sh_gij_p21_90-sh_gij_p21_90/p+0+0+0/{pvec_pvec_name}"
    )
    filename_k_k_A_PS = pickledir / Path(
        f"{k_k_dir}/messpec/32x64/{dirpart_s}/sh_gij_p21_90-sh_gij_p21_90/p+0+0+0/{axial_pvec_name}"
    )
    filename_k_k_PS_A = pickledir / Path(
        f"{k_k_dir}/messpec/32x64/{dirpart_s}/sh_gij_p21_90-sh_gij_p21_90/p+0+0+0/{pvec_axial_name}"
    )
    filename_k_k_A_A = pickledir / Path(
        f"{k_k_dir}/messpec/32x64/{dirpart_s}/sh_gij_p21_90-sh_gij_p21_90/p+0+0+0/{axial_axial_name}"
    )

    k_k_pvec_pvec = read_pickle(filename_k_k_PS_PS, nboot=nboot, nbin=nbin)
    k_k_pvec_axial = read_pickle(filename_k_k_PS_A, nboot=nboot, nbin=nbin)
    k_k_axial_pvec = read_pickle(filename_k_k_A_PS, nboot=nboot, nbin=nbin)
    k_k_axial_axial = read_pickle(filename_k_k_A_A, nboot=nboot, nbin=nbin)

    k_k_ps_ps = k_k_pvec_pvec[:, :, 0] + 1j * k_k_pvec_pvec[:, :, 1]
    k_k_a_a = k_k_axial_axial[:, :, 0] + 1j * k_k_axial_axial[:, :, 1]
    k_k_ps_a = k_k_pvec_axial[:, :, 0] + 1j * k_k_pvec_axial[:, :, 1]
    k_k_a_ps = k_k_axial_pvec[:, :, 0] + 1j * k_k_axial_pvec[:, :, 1]

    # ======================================================================
    # Make projected operator threept functions
    # pion to kaon
    corr_matrix_p_k = np.array(
        [
            [p_k_ps_ps, p_k_ps_a],
            [p_k_a_ps, p_k_a_a],
        ]
    )
    p_k_fwd = np.einsum(
        # "i,ijkl,j->kl",
        # evecs_pion[0][:, 0],
        # corr_matrix_p_k,
        # evecs_kaon[1][:, 0],
        "i,ijkl,j->kl",
        evecs_kaon[0][:, 0].conj(),
        corr_matrix_p_k,
        evecs_pion[1][:, 0],
    )

    # ------------------------------------------------------------
    # kaon to pion
    corr_matrix_k_p = np.array(
        [
            [k_p_ps_ps, k_p_ps_a],
            [k_p_a_ps, k_p_a_a],
        ]
    )
    k_p_fwd = np.einsum(
        # "i,ijkl,j->kl",
        # evecs_kaon[0][:, 0],
        # corr_matrix_k_p,
        # evecs_pion[1][:, 0],
        "i,ijkl,j->kl",
        evecs_pion[0][:, 0].conj(),
        corr_matrix_k_p,
        evecs_kaon[1][:, 0],
    )
    # ------------------------------------------------------------
    # pion to pion
    corr_matrix_p_p = np.array(
        [
            [p_p_ps_ps, p_p_ps_a],
            [p_p_a_ps, p_p_a_a],
        ]
    )
    p_p_fwd = np.einsum(
        "i,ijkl,j->kl",
        evecs_pion[0][:, 0].conj(),
        corr_matrix_p_p,
        evecs_pion[1][:, 0],
    )

    # ------------------------------------------------------------
    corr_matrix_k_k = np.array(
        [
            [k_k_ps_ps, k_k_ps_a],
            [k_k_a_ps, k_k_a_a],
        ]
    )
    k_k_fwd = np.einsum(
        "i,ijkl,j->kl",
        evecs_kaon[0][:, 0].conj(),
        corr_matrix_k_k,
        evecs_kaon[1][:, 0],
    )

    # # ------------------------------------------------------------
    # # Plotting tests
    # k_k_fwd_2 = np.einsum(
    #     "i,ijkl,j->kl", evecs_kaon[0][:, 0], corr_matrix_k_k, evecs_kaon[1][:, 0].T
    # )
    # plot_2corr_log(
    #     # -k_k_fwd_2,
    #     # -k_p_a_a.real,
    #     k_p_a_ps.real,
    #     # -k_k_a_a.imag,
    #     k_p_ps_ps.real,
    #     # k_k_ps_ps.imag,
    #     plotname="k_p_ps-ps_a-ps",
    #     # plotname="k_p_ps-ps_a-a",
    #     plotdir=plotdir,
    #     # v_line=10,
    # )
    # exit()
    # # # ------------------------------------------------------------

    return (
        [p_k_fwd, k_p_fwd, p_p_fwd, k_k_fwd],
        [
            p_k_ps_ps,
            k_p_ps_ps,
            p_p_ps_ps,
            k_k_ps_ps,
        ],
        [
            p_k_a_a,
            k_p_a_a,
            p_p_a_a,
            k_k_a_a,
        ],
    )


def meson_3pt_projections_big(
    pickledir,
    plotdir,
    datadir,
    evecs_pion,
    evecs_kaon,
    fwd_index=0,
    p_k_dir="meson_qcdsf",
    k_p_dir="meson_qcdsf",
    p_p_dir="meson_qcdsf",
    k_k_dir="meson_qcdsf",
    time_choice=29,
    delta_t=6,
    label="",
    kappa_1="kp121040kp121040",
    kappa_2="kp120620kp121040",
):
    print(np.shape(evecs_pion))
    print(np.shape(evecs_kaon))
    # GEVP
    # time_choice = 25
    # delta_t = 4
    nboot = 500
    nbin = 1
    prev_evecs = None

    conf_num_u = "495"
    dirpart_u = f"slrc/{kappa_1}/"
    dirpart_s = f"slrc/{kappa_2}/"

    # ------------------------------------------------------------
    # Find conf number
    test_file = (
        pickledir
        / Path(
            f"{p_k_dir}/messpec/32x64/{dirpart_u}sh_gij_p21_90-sh_gij_p21_90/p+0+0+0/"
        )
    ).glob("messpec_g5-g5_*cfgs.pickle")
    conf_num_u = [
        int("".join(filter(str.isdigit, l.name.split("_")[-1])))
        for l in list(test_file)
    ][0]
    print(f"conf_num = {conf_num_u}")

    # ------------------------------------------------------------
    ps = 5
    a4 = 53
    a2 = 51
    opnames = [ps, a4, a2]
    if label == "utwist":
        opnames_p = [ps, a4, a2]
        opnames_k = [ps, a4]
    elif label == "stwist":
        opnames_k = [ps, a4, a2]
        opnames_p = [ps, a4]
    # opnames_k = opnames
    for iop, op1 in enumerate(opnames_p):
        for jop, op2 in enumerate(opnames_k):
            sign = 1.0
            if iop == 1:
                sign = sign * 1.0j
            if jop == 1:
                sign = sign * 1.0j
            # ------------------------------------------------------------
            # pion to kaon
            filename_p_k = pickledir / Path(
                f"{k_p_dir}/messpec/32x64/{dirpart_u}/sh_gij_p21_90-sh_gij_p21_90/p+0+0+0/messpec_g{op1}-g{op2}_{conf_num_u}cfgs.pickle"
            )
            p_k_data = read_pickle(filename_p_k, nboot=nboot, nbin=nbin)
            p_k_complex = p_k_data[:, :, 0] + 1j * p_k_data[:, :, 1]
            if iop == 0 and jop == 0:
                corr_matrix_p_k = np.empty(
                    (
                        len(opnames_p),
                        len(opnames_k),
                        p_k_complex.shape[0],
                        p_k_complex.shape[1],
                    ),
                    dtype=complex,
                )
            corr_matrix_p_k[iop, jop] = sign * p_k_complex

    for iop, op1 in enumerate(opnames_k):
        for jop, op2 in enumerate(opnames_p):
            sign = 1.0
            if iop == 1:
                sign = sign * 1.0j
            if jop == 1:
                sign = sign * 1.0j
            # ------------------------------------------------------------
            # kaon to pion
            filename_k_p = pickledir / Path(
                f"{p_k_dir}/messpec/32x64/{dirpart_s}/sh_gij_p21_90-sh_gij_p21_90/p+0+0+0/messpec_g{op1}-g{op2}_{conf_num_u}cfgs.pickle"
            )
            k_p_data = read_pickle(filename_k_p, nboot=nboot, nbin=nbin)
            k_p_complex = k_p_data[:, :, 0] + 1j * k_p_data[:, :, 1]
            if iop == 0 and jop == 0:
                corr_matrix_k_p = np.empty(
                    (
                        len(opnames_k),
                        len(opnames_p),
                        k_p_complex.shape[0],
                        k_p_complex.shape[1],
                    ),
                    dtype=complex,
                )
            corr_matrix_k_p[iop, jop] = sign * k_p_complex

    for iop, op1 in enumerate(opnames_p):
        for jop, op2 in enumerate(opnames_p):
            # ------------------------------------------------------------
            # pion to pion
            filename_p_p = pickledir / Path(
                f"{p_p_dir}/messpec/32x64/{dirpart_u}/sh_gij_p21_90-sh_gij_p21_90/p+0+0+0/messpec_g{op1}-g{op2}_{conf_num_u}cfgs.pickle"
            )
            p_p_data = read_pickle(filename_p_p, nboot=nboot, nbin=nbin)
            p_p_complex = p_p_data[:, :, 0] + 1j * p_p_data[:, :, 1]
            if iop == 0 and jop == 0:
                corr_matrix_p_p = np.empty(
                    (
                        len(opnames_p),
                        len(opnames_p),
                        p_p_complex.shape[0],
                        p_p_complex.shape[1],
                    ),
                    dtype=complex,
                )
            corr_matrix_p_p[iop, jop] = p_p_complex

    for iop, op1 in enumerate(opnames_k):
        for jop, op2 in enumerate(opnames_k):
            # ------------------------------------------------------------
            # kaon to kaon
            filename_k_k = pickledir / Path(
                f"{k_k_dir}/messpec/32x64/{dirpart_s}/sh_gij_p21_90-sh_gij_p21_90/p+0+0+0/messpec_g{op1}-g{op2}_{conf_num_u}cfgs.pickle"
            )
            k_k_data = read_pickle(filename_k_k, nboot=nboot, nbin=nbin)
            k_k_complex = k_k_data[:, :, 0] + 1j * k_k_data[:, :, 1]
            if iop == 0 and jop == 0:
                corr_matrix_k_k = np.empty(
                    (
                        len(opnames_k),
                        len(opnames_k),
                        k_k_complex.shape[0],
                        k_k_complex.shape[1],
                    ),
                    dtype=complex,
                )
            corr_matrix_k_k[iop, jop] = k_k_complex

    # ------------------------------------------------------------
    # pion to kaon
    p_k_fwd = np.einsum(
        "i,ijkl,j->kl",
        evecs_kaon[0][:, fwd_index],
        corr_matrix_k_p,
        # corr_matrix_p_k,
        evecs_pion[1][:, fwd_index],
    )

    # ------------------------------------------------------------
    # kaon to pion
    k_p_fwd = np.einsum(
        "i,ijkl,j->kl",
        evecs_pion[0][:, fwd_index],
        corr_matrix_p_k,
        # corr_matrix_k_p,
        evecs_kaon[1][:, fwd_index],
    )
    # ------------------------------------------------------------
    # pion to pion
    p_p_fwd = np.einsum(
        "i,ijkl,j->kl",
        evecs_pion[0][:, fwd_index],
        corr_matrix_p_p,
        evecs_pion[1][:, fwd_index],
    )

    # ------------------------------------------------------------
    k_k_fwd = np.einsum(
        "i,ijkl,j->kl",
        evecs_kaon[0][:, fwd_index],
        corr_matrix_k_k,
        evecs_kaon[1][:, fwd_index],
    )

    p_k_ps_ps = corr_matrix_p_k[0, 0]
    k_p_ps_ps = corr_matrix_k_p[0, 0]
    p_p_ps_ps = corr_matrix_p_p[0, 0]
    k_k_ps_ps = corr_matrix_k_k[0, 0]

    p_k_a_a = corr_matrix_p_k[1, 1]
    k_p_a_a = corr_matrix_k_p[1, 1]
    p_p_a_a = corr_matrix_p_p[1, 1]
    k_k_a_a = corr_matrix_k_k[1, 1]

    return (
        [p_k_fwd, k_p_fwd, p_p_fwd, k_k_fwd],
        [
            p_k_ps_ps,
            k_p_ps_ps,
            p_p_ps_ps,
            k_k_ps_ps,
        ],
        [
            p_k_a_a,
            k_p_a_a,
            p_p_a_a,
            k_k_a_a,
        ],
    )


def fit_correlator(
    fit_corr,
    plotdir,
    datadir,
    name="",
    ylabel=r"$R(t)$",
    ylim=False,
    time_limits=np.array([[[11, 15], [13, 20]]]),
    tau=10,
):
    # ======================================================================
    # Fitting the 3pt function
    const_function = ff.initffncs("Constant")
    [fitlist_kpi] = fit_loop_new(
        fit_corr,
        [const_function],
        time_limits,
        datadir,
        # plotdir / "data/",
        name,
        Nt_min=2,
    )

    bestfit = fitlist_kpi[0]
    fit_time = bestfit["x"]

    time = np.arange(0, np.shape(fit_corr)[1])
    yavg = np.average(fit_corr, axis=0)
    ystd = np.std(fit_corr, axis=0)
    weighted_energy = bestfit["param"][:, 0]
    fit_redchisq = bestfit["redchisq"]
    print([fit["redchisq"] for fit in fitlist_kpi])
    print("\nweights:")
    print([fit["weight"] for fit in fitlist_kpi])

    # f, axs = plt.subplots(1, 1, figsize=(9, 6))
    f, axs = plt.subplots(1, 1, figsize=(7, 6))
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
        label=rf"$|M|={err_brackets(np.average(weighted_energy), np.std(weighted_energy))}$; $\chi^2_{{\textrm{{dof}}}} = {fit_redchisq:.2f}$",
        zorder=3,
    )
    axs.axvline(tau, color="k", linewidth=1, linestyle="--")

    plt.xlabel(r"$t/a$")
    plt.ylabel(ylabel, fontsize="small")
    plt.legend(fontsize="x-small")
    if ylim:
        axs.set_ylim(ylim)
    plt.tight_layout()
    plt.savefig(plotdir / f"{name}.pdf")
    # plt.savefig(plotdir / f"{name}_zoom.pdf")
    plt.close()

    return fitlist_kpi


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
        label=rf"$|M|={err_brackets(np.average(weighted_energy), np.std(weighted_energy))}$; $\chi^2_{{\textrm{{dof}}}} = {fit_redchisq:.2f}$",
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

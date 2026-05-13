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

def main_meson3():
    """
    Read the meson 3pt functions and 2pt functions. Construct a ratio to fit and plot
    For 3pt correlators at qsq_max
    """

    mystyle = Path("mystyle.txt")
    plt.style.use(mystyle.as_posix())
    plt.rc("text.latex", preamble=r"\usepackage{physics}")
    plotdir = Path("./plots/qsq_max/")

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
    plot_corr_log(bsdata_k_p, plotname="threepoint_k_p",plotdir=plotdir)
    
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
    # Ratio should be equal to f_0(qsq_max)
    quad_ratio_bs = np.einsum("ij,i->ij", bsdata_7a, energy_factor / energy_factor2)
    # quad_ratio_bs = np.einsum("ij,i->ij", bsdata_7a, energy_factor / norm_factor)
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
        "ij,i->ij", bsdata_1a, energy_factor / energy_factor2 * norm_factor
        # "ij,i->ij", bsdata_1a, energy_factor
    )
    plot_corr(bsdata_1b, plotname="3pt_pi-k_double-ratio", plotdir=plotdir, v_line=10)
    plot_corr(
        bsdata_1b,
        plotname="3pt_pi-k_double-ratio_ylim",
        plotdir=plotdir,
        v_line=10,
        ylim=(0.95, 1.04),
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
        "ij,i->ij", bsdata_dbl_fit1, energy_factor / energy_factor2 * norm_factor
        # "ij,i->ij", bsdata_dbl_fit1, energy_factor
    )
    plot_corr(
        bsdata_dbl_fit2,
        plotname="3pt_pi-k_double-ratio_fit",
        plotdir=plotdir,
        v_line=10,
        ylim=(0.95, 1.04),
        # ylim=(0.34, 0.395),
    )

    fit_correlator(bsdata_1b, plotdir, name="3pt_double_kpi_fit", ylim=(0.95, 1.04))
    fit_correlator(
        bsdata_dbl_fit2, plotdir, name="3pt_double_kpi_fitparam_fit", ylim=(0.95, 1.04)
    )
    fit_correlator(quad_ratio_bs, plotdir, name="3pt_quad_kpi_fit", ylim=(0.95, 1.04))

    return


def fit_correlator(fit_corr, plotdir, name="", ylim=False):
    # ======================================================================
    # Fitting the 3pt function
    const_function = ff.initffncs("Constant")
    time_limits = np.array([[[11, 15], [13, 20]]])
    [fitlist_kpi] = fit_loop_new(
        fit_corr,
        [const_function],
        time_limits,
        plotdir,
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
    plt.ylabel(r"$R(t)$", fontsize="small")
    plt.legend(fontsize="x-small")
    if ylim:
        axs.set_ylim(ylim)
    plt.savefig(plotdir / f"{name}.pdf")
    # plt.savefig(plotdir / f"{name}_zoom.pdf")
    plt.close()
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
    main_meson3()

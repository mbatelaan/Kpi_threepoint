import numpy as np
from pathlib import Path
import pickle
import yaml
import shutil
import os
import sys
import scipy.optimize as syopt
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from matplotlib import rcParams

from analysis import stats
from analysis.bootstrap import bootstrap
from analysis.formatting import err_brackets
from analysis import fitfunc as ff

from gevpanalysis.params import params


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
# _colors = ["r", "g", "b", "k", "y", "m", "k", "k"]
_markers = ["s", "o", "^", "*", "v", ">", "<", "s", "s"]
# sys.stdout = open("output.txt", "wt")
# From the theta tuning:
m_N = 0.4179255
m_S = 0.4641829


def read_pickle(filename, nboot=200, nbin=1):
    """Get the data from the pickle file and output a bootstrapped numpy array.

    The output is a numpy matrix with:
    axis=0: bootstraps
    axis=2: time axis
    axis=3: real & imaginary parts
    """
    with open(filename, "rb") as file_in:
        data = pickle.load(file_in)
    bsdata = bootstrap(data, config_ax=0, nboot=nboot, nbin=nbin)
    return bsdata


def fit_value(diffG, t_range):
    """Fit a constant function to the diffG correlator

    diffG is a correlator with the bootstraps on the first index and the time on the second
    t_range is an array of time values to fit over
    the function will return an array of fit parameters for each bootstrap
    """
    data_set = diffG[:, t_range]
    diffG_avg = np.average(data_set, axis=0)
    covmat = np.cov(data_set.T)
    diag = np.diagonal(covmat)
    norms = np.einsum("i,j->ij", diag, diag) ** 0.5
    covmat_norm = covmat / norms
    # plt.figure(figsize=(11, 11))
    # plt.figure(figsize=(11, 11))
    # mat = plt.matshow(np.linalg.inv(covmat))
    # plt.colorbar(mat, shrink=0.5)
    # plt.savefig("cov_matrix_corr.pdf")

    diag_sigma = np.diag(np.std(data_set, axis=0) ** 2)
    popt_avg, pcov_avg = curve_fit(ff.constant, t_range, diffG_avg, sigma=covmat)
    chisq = ff.chisqfn(
        *popt_avg, ff.constant, t_range, diffG_avg, np.linalg.inv(covmat)
    )
    redchisq = chisq / len(t_range)
    bootfit = []
    for iboot, values in enumerate(diffG):
        popt, pcov = curve_fit(ff.constant, t_range, values[t_range], sigma=diag_sigma)
        bootfit.append(popt)
    bootfit = np.array(bootfit)

    return bootfit, redchisq


def fit_value2(diffG, t_range, function):
    """Fit a function to the diffG correlator

    diffG is a correlator with the bootstraps on the first index and the time on the second
    t_range is an array of time values to fit over
    the function will return an array of fit parameters for each bootstrap
    """
    data_set = diffG[:, t_range]
    diffG_avg = np.average(data_set, axis=0)
    covmat = np.cov(data_set.T)
    diag_sigma = np.diag(np.std(data_set, axis=0) ** 2)
    function.initparfnc(diffG, timeslice=8)
    fitparam = stats.fit_bootstrap(
        function.eval,
        function.initpar,
        t_range,
        data_set,
        bounds=None,
        time=False,
        fullcov=False,
    )

    bootfit = fitparam["param"]
    return bootfit, fitparam["redchisq"]


def fit_value3(diffG, t_range, function, norm=1):
    """Fit a function to the diffG correlator

    diffG is a correlator with the bootstraps on the first index and the time on the second
    t_range is an array of time values to fit over
    the function will return an array of fit parameters for each bootstrap
    """
    diffG = diffG / norm
    data_set = diffG[:, t_range]
    diffG_avg = np.average(data_set, axis=0)
    covmat = np.cov(data_set.T)
    diag_sigma = np.diag(np.std(data_set, axis=0) ** 2)
    function.initparfnc(diffG, timeslice=7)
    # print("initpar = ", function.initpar)

    popt_avg, pcov_avg = curve_fit(
        function.eval_2, t_range, diffG_avg, sigma=diag_sigma, p0=function.initpar
    )

    chisq = ff.chisqfn2(
        popt_avg, function.eval_2, t_range, diffG_avg, np.linalg.inv(covmat)
    )
    redchisq = chisq / len(t_range)
    bootfit = []
    for iboot, values in enumerate(diffG):
        # print("p0 = ", function.initpar)
        popt, pcov = curve_fit(
            function.eval_2,
            t_range,
            values[t_range],
            sigma=diag_sigma,
            p0=function.initpar,
        )
        bootfit.append(popt)
    bootfit = np.array(bootfit)
    return bootfit, redchisq


def fit_value4(corr, t_range, function, unpert_energy, norm=1):
    """Fit a function to the corr correlator

    corr is a correlator with the bootstraps on the first index and the time on the second
    t_range is an array of time values to fit over
    unpert_energy is an array containing the bootstrap data of the unperturbed energy
    the function will return an array of fit parameters for each bootstrap
    """
    corr = corr / norm
    data_set = corr[:, t_range]
    corr_avg = np.average(data_set, axis=0)
    covmat = np.cov(data_set.T)
    diag_sigma = np.diag(np.std(data_set, axis=0) ** 2)
    function.initparfnc(corr, timeslice=7)
    function.E = np.average(unpert_energy, axis=0)

    popt_avg, pcov_avg = curve_fit(
        function.eval_2, t_range, corr_avg, sigma=diag_sigma, p0=function.initpar
    )

    chisq = ff.chisqfn2(
        popt_avg, function.eval_2, t_range, corr_avg, np.linalg.inv(covmat)
    )
    redchisq = chisq / len(t_range)
    bootfit = []
    for iboot, values in enumerate(corr):
        # print("p0 = ", function.initpar)
        function.E = unpert_energy[iboot]
        popt, pcov = curve_fit(
            function.eval_2,
            t_range,
            values[t_range],
            sigma=diag_sigma,
            p0=function.initpar,
        )
        bootfit.append(popt)
    bootfit = np.array(bootfit)
    return bootfit, redchisq


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
    print(f"{fitfnc.initpar=}")
    # fitfnc.initpar = np.array([np.average(unpert_fits[:, 0], axis=0), 0.0])
    # print(f"{fitfnc.initpar=}")
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
                # if disp:
                #     print(f"priors = {fitfnc.initpar}")
                #     print(f"priors sigma = {fitfnc.priorsigma}")
                #     print(f"parameter values = {fitparam_avg['paramavg']}")
                #     print(f"chi-sq. per dof. = {fitparam_avg['redchisq']}")
                fitparam_avg["y"] = data
                fitlist.append(fitparam_avg)

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


def transfer_pickle_data(pars, pickledir, pickledir2, datadir, qmax=False):
    """Read the correlator data from the pickle files, then save it in the data directory.
    The data will be placed in the following tree structure:
    datadir
        |- correlators_unperturbed
        |   |- corr_UU_lmb0
        |   |- corr_SS_lmb0
        |- correlators_lmb1
        |   |- corr_US_lmb1
        |   |- corr_SU_lmb1
        |- correlators_lmb2
        |   |- corr_UU_lmb2
        |   |- corr_SS_lmb2
        |- correlators_lmb3
        |   |- corr_US_lmb3
        |   |- corr_SU_lmb3
        |- correlators_lmb4
            |- corr_UU_lmb4
            |- corr_SS_lmb4
    """
    mom_strings = ["p-1+0+0", "p+0+0+0", "p+1+0+0"]

    ### find the highest number of configurations available
    ### Unperturbed correlators
    fileup = (
        pickledir
        / Path(
            "baryon-3pt_US_lmb_TBC/barspec/32x64/slrc/kp121040kp120620/sh_gij_p21_90-sh_gij_p21_90/"
            + mom_strings[1]
        )
    ).glob("barspec_nucleon_rel_[0-9]*cfgs.pickle")
    # print(list(fileup))
    conf_num_list_u = np.array(
        [int("".join(filter(str.isdigit, l.name))) for l in list(fileup)]
    )
    print(conf_num_list_u)
    conf_num_u = conf_num_list_u[np.argmax(conf_num_list_u)]
    barspec_nameU = "/barspec_nucleon_rel_" + str(conf_num_u) + "cfgs.pickle"

    filestrange = (
        pickledir2
        / Path(
            "baryon_qcdsf/barspec/32x64/slrc/kp121040kp120620/sh_gij_p21_90-sh_gij_p21_90/"
            + mom_strings[1]
        )
    ).glob("barspec_nucleon_rel_[0-9]*cfgs.pickle")
    conf_num_list_s = np.array(
        [int("".join(filter(str.isdigit, l.name))) for l in list(filestrange)]
    )
    print(conf_num_list_s)
    conf_num_s = conf_num_list_s[np.argmax(conf_num_list_s)]
    barspec_nameS = "/barspec_nucleon_rel_" + str(conf_num_s) + "cfgs.pickle"

    # We now have the names of the files with the correct number of configurations in the name

    dirpart_u = "slrc/kp121040kp121040/"
    dirpart_s = "slrc/kp121040kp120620/"
    if qmax:
        dirpart_nucl = "slrc/kp121040kp121040/"
        unpertfile_nucleon = pickledir2 / Path(
            f"baryon_qcdsf/barspec/32x64/{dirpart_nucl}sh_gij_p21_90-sh_gij_p21_90/"
            + mom_strings[1]
            + barspec_nameS
        )
    else:
        unpertfile_nucleon = pickledir / Path(
            f"baryon_qcdsf_TBC/barspec/32x64/{dirpart_u}sh_gij_p21_90-sh_gij_p21_90/"
            + mom_strings[1]
            + barspec_nameU
        )

    ### ----------------------------------------------------------------------
    ### Unperturbed correlators
    # unpertfile_nucleon = nucleon_pickledir / Path(
    #     f"baryon_qcdsf_TBC/barspec/32x64/{dirpart_u}sh_gij_p21_90-sh_gij_p21_90/"
    #     + mom_strings[1]
    #     + nucleon_name
    # )
    unpertfile_sigma = pickledir2 / Path(
        "baryon_qcdsf/barspec/32x64/slrc/kp121040kp120620/sh_gij_p21_90-sh_gij_p21_90/"
        + mom_strings[1]
        + barspec_nameS
    )

    # Define the destination file names
    write_file_nucleon_lmb0 = datadir / Path(
        f"correlator_data/corr_UU_lmb0_{mom_strings[1]}.pkl"
    )
    write_file_sigma_lmb0 = datadir / Path(
        f"correlator_data/corr_SS_lmb0_{mom_strings[1]}.pkl"
    )

    print(write_file_nucleon_lmb0)

    # Copy the pickle files to the new locations in the data directory
    shutil.copyfile(unpertfile_nucleon, write_file_nucleon_lmb0)
    shutil.copyfile(unpertfile_sigma, write_file_sigma_lmb0)

    print("here")

    ### ----------------------------------------------------------------------
    ### SU & SS
    filelist_SU1 = pickledir2 / Path(
        f"baryon-3pt_SU_lmb_TBC/barspec/32x64/{dirpart_u}sh_gij_p21_90-sh_gij_p21_90/"
        + mom_strings[1]
        + barspec_nameS
    )
    filelist_SU3 = pickledir2 / Path(
        f"baryon-3pt_SU_lmb3_TBC/barspec/32x64/{dirpart_u}sh_gij_p21_90-sh_gij_p21_90/"
        + mom_strings[1]
        + barspec_nameS
    )
    filelist_SS2 = pickledir2 / Path(
        f"baryon-3pt_SS_lmb2_TBC/barspec/32x64/{dirpart_s}sh_gij_p21_90-sh_gij_p21_90/"
        + mom_strings[1]
        + barspec_nameS
    )
    filelist_SS4 = pickledir2 / Path(
        f"baryon-3pt_SS_lmb4_TBC/barspec/32x64/{dirpart_s}sh_gij_p21_90-sh_gij_p21_90/"
        + mom_strings[1]
        + barspec_nameS
    )

    # Define the destination file names
    write_file_sigma_lmb1 = datadir / Path(
        f"correlator_data/corr_SU_lmb1_{mom_strings[1]}.pkl"
    )
    write_file_sigma_lmb3 = datadir / Path(
        f"correlator_data/corr_SU_lmb3_{mom_strings[1]}.pkl"
    )
    write_file_sigma_lmb2 = datadir / Path(
        f"correlator_data/corr_SS_lmb2_{mom_strings[1]}.pkl"
    )
    write_file_sigma_lmb4 = datadir / Path(
        f"correlator_data/corr_SS_lmb4_{mom_strings[1]}.pkl"
    )

    # Copy the pickle files to the new locations in the data directory
    shutil.copyfile(filelist_SU1, write_file_sigma_lmb1)
    shutil.copyfile(filelist_SU3, write_file_sigma_lmb3)
    shutil.copyfile(filelist_SS2, write_file_sigma_lmb2)
    shutil.copyfile(filelist_SS4, write_file_sigma_lmb4)

    ### ----------------------------------------------------------------------
    ### US & UU
    filelist_US1 = pickledir / Path(
        f"baryon-3pt_US_lmb_TBC/barspec/32x64/{dirpart_s}sh_gij_p21_90-sh_gij_p21_90/"
        + mom_strings[1]
        + barspec_nameU
    )
    filelist_UU2 = pickledir / Path(
        f"baryon-3pt_UU_lmb2_TBC/barspec/32x64/{dirpart_u}sh_gij_p21_90-sh_gij_p21_90/"
        + mom_strings[1]
        + barspec_nameU
    )
    filelist_US3 = pickledir / Path(
        f"baryon-3pt_US_lmb3_TBC/barspec/32x64/{dirpart_s}sh_gij_p21_90-sh_gij_p21_90/"
        + mom_strings[1]
        + barspec_nameU
    )
    filelist_UU4 = pickledir / Path(
        f"baryon-3pt_UU_lmb4_TBC/barspec/32x64/{dirpart_u}sh_gij_p21_90-sh_gij_p21_90/"
        + mom_strings[1]
        + barspec_nameU
    )

    # Define the destination file names
    write_file_nucleon_lmb1 = datadir / Path(
        f"correlator_data/corr_US_lmb1_{mom_strings[1]}.pkl"
    )
    write_file_nucleon_lmb3 = datadir / Path(
        f"correlator_data/corr_US_lmb3_{mom_strings[1]}.pkl"
    )
    write_file_nucleon_lmb2 = datadir / Path(
        f"correlator_data/corr_UU_lmb2_{mom_strings[1]}.pkl"
    )
    write_file_nucleon_lmb4 = datadir / Path(
        f"correlator_data/corr_UU_lmb4_{mom_strings[1]}.pkl"
    )

    # Copy the pickle files to the new locations in the data directory
    shutil.copyfile(filelist_US1, write_file_nucleon_lmb1)
    shutil.copyfile(filelist_US3, write_file_nucleon_lmb3)
    shutil.copyfile(filelist_UU2, write_file_nucleon_lmb2)
    shutil.copyfile(filelist_UU4, write_file_nucleon_lmb4)
    return


def transfer_pickle_data_mesons(pars, pickledir, pickledir2, datadir):
    """Read the correlator data from the pickle files, then save it in the data directory.
    The data will be placed in the following tree structure:
    datadir
        |- correlators_unperturbed
        |   |- meson_g5-g5_UU_lmb0
        |   |- meson_g5-g5_SS_lmb0
        |- correlators_lmb1
        |   |- meson_g5-g5_US_lmb1
        |   |- meson_g5-g5_SU_lmb1
        |- correlators_lmb2
        |   |- meson_g5-g5_UU_lmb2
        |   |- meson_g5-g5_SS_lmb2
        |- correlators_lmb3
        |   |- meson_g5-g5_US_lmb3
        |   |- meson_g5-g5_SU_lmb3
        |- correlators_lmb4
            |- meson_g5-g5_UU_lmb4
            |- meson_g5-g5_SS_lmb4
    """
    mom_strings = ["p-1+0+0", "p+0+0+0", "p+1+0+0"]

    ### find the highest number of configurations available
    ### Unperturbed correlators
    fileup = (
        pickledir
        / Path(
            "meson-3pt_US_lmb_TBC/messpec/32x64/slrc/kp121040kp120620/sh_gij_p21_90-sh_gij_p21_90/"
            + mom_strings[1]
        )
    ).glob("messpec_g5-g5_[0-9]*cfgs.pickle")
    fileup_list = list(fileup)
    # print(list(fileup))
    # conf_num_list_u = np.array(
    #     [int("".join(filter(str.isdigit, l.name))) for l in list(fileup)]
    # )
    test1 = np.array([l.name for l in fileup_list])
    test2 = np.array([l.name.split("_")[-1] for l in fileup_list])
    print(test1)
    print(test2)
    conf_num_list_u = np.array(
        [int("".join(filter(str.isdigit, l.name.split("_")[-1]))) for l in fileup_list]
    )
    print(conf_num_list_u)
    conf_num_u = conf_num_list_u[np.argmax(conf_num_list_u)]
    print(conf_num_u)
    messpec_nameU = "/messpec_g5-g5_" + str(conf_num_u) + "cfgs.pickle"

    filestrange = (
        pickledir2
        / Path(
            "meson_qcdsf/messpec/32x64/slrc/kp121040kp120620/sh_gij_p21_90-sh_gij_p21_90/"
            + mom_strings[1]
        )
    ).glob("messpec_g5-g5_[0-9]*cfgs.pickle")
    # conf_num_list_s = np.array(
    #     [int("".join(filter(str.isdigit, l.name))) for l in list(filestrange)]
    # )
    conf_num_list_s = np.array(
        [
            int("".join(filter(str.isdigit, l.name.split("_")[-1].split(".")[0])))
            for l in list(filestrange)
        ]
    )

    print(conf_num_list_s)
    conf_num_s = conf_num_list_s[np.argmax(conf_num_list_s)]
    messpec_nameS = "/messpec_g5-g5_" + str(conf_num_s) + "cfgs.pickle"

    # We now have the names of the files with the correct number of configurations in the name
    dirpart_u = "slrc/kp121040kp121040/"
    dirpart_s = "slrc/kp121040kp120620/"
    unpertfile_pion = pickledir / Path(
        f"meson_qcdsf_TBC/messpec/32x64/{dirpart_u}sh_gij_p21_90-sh_gij_p21_90/"
        + mom_strings[1]
        + messpec_nameU
    )

    ### ----------------------------------------------------------------------
    ### Unperturbed correlators
    unpertfile_kaon = pickledir2 / Path(
        "meson_qcdsf/messpec/32x64/slrc/kp121040kp120620/sh_gij_p21_90-sh_gij_p21_90/"
        + mom_strings[1]
        + messpec_nameS
    )

    # Define the destination file names
    write_file_pion_lmb0 = datadir / Path(
        f"correlator_data/meson_g5-g5_UU_lmb0_{mom_strings[1]}.pkl"
    )
    write_file_kaon_lmb0 = datadir / Path(
        f"correlator_data/meson_g5-g5_SS_lmb0_{mom_strings[1]}.pkl"
    )

    # Copy the pickle files to the new locations in the data directory
    shutil.copyfile(unpertfile_pion, write_file_pion_lmb0)
    shutil.copyfile(unpertfile_kaon, write_file_kaon_lmb0)

    ### ----------------------------------------------------------------------
    ### SU & SS
    filelist_SU1 = pickledir2 / Path(
        f"meson-3pt_SU_lmb_TBC/messpec/32x64/{dirpart_u}sh_gij_p21_90-sh_gij_p21_90/"
        + mom_strings[1]
        + messpec_nameS
    )
    filelist_SU3 = pickledir2 / Path(
        f"meson-3pt_SU_lmb3_TBC/messpec/32x64/{dirpart_u}sh_gij_p21_90-sh_gij_p21_90/"
        + mom_strings[1]
        + messpec_nameS
    )
    filelist_SS2 = pickledir2 / Path(
        f"meson-3pt_SS_lmb2_TBC/messpec/32x64/{dirpart_s}sh_gij_p21_90-sh_gij_p21_90/"
        + mom_strings[1]
        + messpec_nameS
    )
    filelist_SS4 = pickledir2 / Path(
        f"meson-3pt_SS_lmb4_TBC/messpec/32x64/{dirpart_s}sh_gij_p21_90-sh_gij_p21_90/"
        + mom_strings[1]
        + messpec_nameS
    )

    # Define the destination file names
    write_file_kaon_lmb1 = datadir / Path(
        f"correlator_data/meson_g5-g5_SU_lmb1_{mom_strings[1]}.pkl"
    )
    write_file_kaon_lmb3 = datadir / Path(
        f"correlator_data/meson_g5-g5_SU_lmb3_{mom_strings[1]}.pkl"
    )
    write_file_kaon_lmb2 = datadir / Path(
        f"correlator_data/meson_g5-g5_SS_lmb2_{mom_strings[1]}.pkl"
    )
    write_file_kaon_lmb4 = datadir / Path(
        f"correlator_data/meson_g5-g5_SS_lmb4_{mom_strings[1]}.pkl"
    )

    # Copy the pickle files to the new locations in the data directory
    shutil.copyfile(filelist_SU1, write_file_kaon_lmb1)
    shutil.copyfile(filelist_SU3, write_file_kaon_lmb3)
    shutil.copyfile(filelist_SS2, write_file_kaon_lmb2)
    shutil.copyfile(filelist_SS4, write_file_kaon_lmb4)

    ### ----------------------------------------------------------------------
    ### US & UU
    filelist_US1 = pickledir / Path(
        f"meson-3pt_US_lmb_TBC/messpec/32x64/{dirpart_s}sh_gij_p21_90-sh_gij_p21_90/"
        + mom_strings[1]
        + messpec_nameU
    )
    filelist_UU2 = pickledir / Path(
        f"meson-3pt_UU_lmb2_TBC/messpec/32x64/{dirpart_u}sh_gij_p21_90-sh_gij_p21_90/"
        + mom_strings[1]
        + messpec_nameU
    )
    filelist_US3 = pickledir / Path(
        f"meson-3pt_US_lmb3_TBC/messpec/32x64/{dirpart_s}sh_gij_p21_90-sh_gij_p21_90/"
        + mom_strings[1]
        + messpec_nameU
    )
    filelist_UU4 = pickledir / Path(
        f"meson-3pt_UU_lmb4_TBC/messpec/32x64/{dirpart_u}sh_gij_p21_90-sh_gij_p21_90/"
        + mom_strings[1]
        + messpec_nameU
    )

    # Define the destination file names
    write_file_pion_lmb1 = datadir / Path(
        f"correlator_data/meson_g5-g5_US_lmb1_{mom_strings[1]}.pkl"
    )
    write_file_pion_lmb3 = datadir / Path(
        f"correlator_data/meson_g5-g5_US_lmb3_{mom_strings[1]}.pkl"
    )
    write_file_pion_lmb2 = datadir / Path(
        f"correlator_data/meson_g5-g5_UU_lmb2_{mom_strings[1]}.pkl"
    )
    write_file_pion_lmb4 = datadir / Path(
        f"correlator_data/meson_g5-g5_UU_lmb4_{mom_strings[1]}.pkl"
    )

    # Copy the pickle files to the new locations in the data directory
    shutil.copyfile(filelist_US1, write_file_pion_lmb1)
    shutil.copyfile(filelist_US3, write_file_pion_lmb3)
    shutil.copyfile(filelist_UU2, write_file_pion_lmb2)
    shutil.copyfile(filelist_UU4, write_file_pion_lmb4)
    return


def read_correlators(pars, pickledir, pickledir2, mom_strings):
    """Read the pickle files which contain the correlator data

    The script will check the folders for available files and pick out the files with the highest number of configurations.
    """
    ### find the highest number of configurations available
    ### Unperturbed correlators
    fileup = (
        pickledir
        / Path(
            "baryon-3pt_US_lmb_TBC/barspec/32x64/slrc/kp121040kp120620/sh_gij_p21_90-sh_gij_p21_90/"
            + mom_strings[1]
        )
    ).glob("barspec_nucleon_rel_[0-9]*cfgs.pickle")
    conf_num_list = np.array(
        [int("".join(filter(str.isdigit, l.name))) for l in list(fileup)]
    )
    conf_num = conf_num_list[np.argmax(conf_num_list)]
    print("conf_numU:", conf_num)
    barspec_nameU = "/barspec_nucleon_rel_" + str(conf_num) + "cfgs.pickle"
    filestrange = (
        pickledir2
        / Path(
            "baryon_qcdsf/barspec/32x64/slrc/kp121040kp120620/sh_gij_p21_90-sh_gij_p21_90/"
            + mom_strings[1]
        )
    ).glob("barspec_nucleon_rel_[0-9]*cfgs.pickle")
    conf_num_list = np.array(
        [int("".join(filter(str.isdigit, l.name))) for l in list(filestrange)]
    )
    conf_num = conf_num_list[np.argmax(conf_num_list)]
    print("conf_numS:", conf_num)
    barspec_nameS = "/barspec_nucleon_rel_" + str(conf_num) + "cfgs.pickle"

    ### ----------------------------------------------------------------------
    G2_nucl = []
    G2_sigm = []
    ### ----------------------------------------------------------------------
    ### Unperturbed correlators
    unpertfile_nucleon_pos = pickledir / Path(
        "baryon_qcdsf_TBC/barspec/32x64/slrc/kp121040kp121040/sh_gij_p21_90-sh_gij_p21_90/"
        + mom_strings[2]
        + barspec_nameU
    )
    unpertfile_sigma = pickledir2 / Path(
        "baryon_qcdsf/barspec/32x64/slrc/kp121040kp120620/sh_gij_p21_90-sh_gij_p21_90/"
        + mom_strings[1]
        + barspec_nameS
    )
    G2_unpert_qp100_nucl = read_pickle(
        unpertfile_nucleon_pos, nboot=pars.nboot, nbin=pars.nbin
    )
    G2_unpert_q000_sigma = read_pickle(
        unpertfile_sigma, nboot=pars.nboot, nbin=pars.nbin
    )
    G2_nucl.append(G2_unpert_qp100_nucl)
    G2_sigm.append(G2_unpert_q000_sigma)

    ### ----------------------------------------------------------------------
    ### SU & SS
    filelist_SU1 = pickledir2 / Path(
        "baryon-3pt_SU_lmb_TBC/barspec/32x64/slrc/kp121040kp121040/sh_gij_p21_90-sh_gij_p21_90/"
        + mom_strings[2]
        + barspec_nameS
    )
    filelist_SU3 = pickledir2 / Path(
        "baryon-3pt_SU_lmb3_TBC/barspec/32x64/slrc/kp121040kp121040/sh_gij_p21_90-sh_gij_p21_90/"
        + mom_strings[2]
        + barspec_nameS
    )
    filelist_SS2 = pickledir2 / Path(
        "baryon-3pt_SS_lmb2_TBC/barspec/32x64/slrc/kp121040kp120620/sh_gij_p21_90-sh_gij_p21_90/"
        + mom_strings[1]
        + barspec_nameS
    )
    filelist_SS4 = pickledir2 / Path(
        "baryon-3pt_SS_lmb4_TBC/barspec/32x64/slrc/kp121040kp120620/sh_gij_p21_90-sh_gij_p21_90/"
        + mom_strings[1]
        + barspec_nameS
    )

    G2_q100_SU_lmb = read_pickle(filelist_SU1, nboot=pars.nboot, nbin=pars.nbin)
    G2_q000_SS_lmb2 = read_pickle(filelist_SS2, nboot=pars.nboot, nbin=pars.nbin)
    G2_q100_SU_lmb3 = read_pickle(filelist_SU3, nboot=pars.nboot, nbin=pars.nbin)
    G2_q000_SS_lmb4 = read_pickle(filelist_SS4, nboot=pars.nboot, nbin=pars.nbin)
    G2_sigm.append(G2_q100_SU_lmb)
    G2_sigm.append(G2_q000_SS_lmb2)
    G2_sigm.append(G2_q100_SU_lmb3)
    G2_sigm.append(G2_q000_SS_lmb4)

    ### ----------------------------------------------------------------------
    ### US & UU
    filelist_US1 = pickledir / Path(
        "baryon-3pt_US_lmb_TBC/barspec/32x64/slrc/kp121040kp120620/sh_gij_p21_90-sh_gij_p21_90/"
        + mom_strings[1]
        + barspec_nameU
    )
    filelist_UU2 = pickledir / Path(
        "baryon-3pt_UU_lmb2_TBC/barspec/32x64/slrc/kp121040kp121040/sh_gij_p21_90-sh_gij_p21_90/"
        + mom_strings[2]
        + barspec_nameU
    )
    filelist_US3 = pickledir / Path(
        "baryon-3pt_US_lmb3_TBC/barspec/32x64/slrc/kp121040kp120620/sh_gij_p21_90-sh_gij_p21_90/"
        + mom_strings[1]
        + barspec_nameU
    )
    filelist_UU4 = pickledir / Path(
        "baryon-3pt_UU_lmb4_TBC/barspec/32x64/slrc/kp121040kp121040/sh_gij_p21_90-sh_gij_p21_90/"
        + mom_strings[2]
        + barspec_nameU
    )

    G2_q000_US_lmb = read_pickle(filelist_US1, nboot=pars.nboot, nbin=pars.nbin)
    G2_q100_UU_lmb2 = read_pickle(filelist_UU2, nboot=pars.nboot, nbin=pars.nbin)
    G2_q000_US_lmb3 = read_pickle(filelist_US3, nboot=pars.nboot, nbin=pars.nbin)
    G2_q100_UU_lmb4 = read_pickle(filelist_UU4, nboot=pars.nboot, nbin=pars.nbin)
    G2_nucl.append(G2_q000_US_lmb)
    G2_nucl.append(G2_q100_UU_lmb2)
    G2_nucl.append(G2_q000_US_lmb3)
    G2_nucl.append(G2_q100_UU_lmb4)

    return G2_nucl, G2_sigm


def read_correlators2(pars, pickledir, pickledir2, mom_strings):
    """Read the pickle files which contain the correlator data

    The script will check the folders for available files and pick out the files with the highest number of configurations.

    This version reads only zero momentum correlators as it is for the case where all the momentum is contained in the twisted boundary conditions.
    """

    ### ----------------------------------------------------------------------
    G2_nucl = []
    G2_sigm = []
    ### ----------------------------------------------------------------------
    ### Unperturbed correlators
    fileup = (
        pickledir
        / Path(
            "baryon_qcdsf_TBC/barspec/32x64/slrc/kp121040kp121040/sh_gij_p21_90-sh_gij_p21_90/"
            + mom_strings[1]
        )
    ).glob("barspec_nucleon_rel_[0-9]*cfgs.pickle")
    conf_num_list = np.array(
        [int("".join(filter(str.isdigit, l.name))) for l in list(fileup)]
    )
    conf_num = conf_num_list[np.argmax(conf_num_list)]
    print("conf_numU:", conf_num)
    barspec_nameU = "/barspec_nucleon_rel_" + str(conf_num) + "cfgs.pickle"

    filestrange = (
        pickledir2
        / Path(
            "baryon_qcdsf/barspec/32x64/slrc/kp121040kp120620/sh_gij_p21_90-sh_gij_p21_90/"
            + mom_strings[1]
        )
    ).glob("barspec_nucleon_rel_[0-9]*cfgs.pickle")
    conf_num_list = np.array(
        [int("".join(filter(str.isdigit, l.name))) for l in list(filestrange)]
    )
    conf_num = conf_num_list[np.argmax(conf_num_list)]
    print("conf_numS:", conf_num)
    barspec_nameS = "/barspec_nucleon_rel_" + str(conf_num) + "cfgs.pickle"

    unpertfile_nucleon_pos = pickledir / Path(
        "baryon_qcdsf_TBC/barspec/32x64/slrc/kp121040kp121040/sh_gij_p21_90-sh_gij_p21_90/"
        + mom_strings[1]
        + barspec_nameU
    )
    unpertfile_sigma = pickledir2 / Path(
        "baryon_qcdsf/barspec/32x64/slrc/kp121040kp120620/sh_gij_p21_90-sh_gij_p21_90/"
        + mom_strings[1]
        + barspec_nameS
    )
    G2_unpert_qp100_nucl = read_pickle(
        unpertfile_nucleon_pos, nboot=pars.nboot, nbin=pars.nbin
    )
    G2_unpert_q000_sigma = read_pickle(
        unpertfile_sigma, nboot=pars.nboot, nbin=pars.nbin
    )
    G2_nucl.append(G2_unpert_qp100_nucl)
    G2_sigm.append(G2_unpert_q000_sigma)

    ### ----------------------------------------------------------------------
    ### SU & SS
    filelist_SU1 = pickledir2 / Path(
        "baryon-3pt_SU_lmb_TBC/barspec/32x64/slrc/kp121040kp121040/sh_gij_p21_90-sh_gij_p21_90/"
        + mom_strings[1]
        + barspec_nameS
    )
    filelist_SU3 = pickledir2 / Path(
        "baryon-3pt_SU_lmb3_TBC/barspec/32x64/slrc/kp121040kp121040/sh_gij_p21_90-sh_gij_p21_90/"
        + mom_strings[1]
        + barspec_nameS
    )
    filelist_SS2 = pickledir2 / Path(
        "baryon-3pt_SS_lmb2_TBC/barspec/32x64/slrc/kp121040kp120620/sh_gij_p21_90-sh_gij_p21_90/"
        + mom_strings[1]
        + barspec_nameS
    )
    filelist_SS4 = pickledir2 / Path(
        "baryon-3pt_SS_lmb4_TBC/barspec/32x64/slrc/kp121040kp120620/sh_gij_p21_90-sh_gij_p21_90/"
        + mom_strings[1]
        + barspec_nameS
    )

    G2_q100_SU_lmb = read_pickle(filelist_SU1, nboot=pars.nboot, nbin=pars.nbin)
    G2_q000_SS_lmb2 = read_pickle(filelist_SS2, nboot=pars.nboot, nbin=pars.nbin)
    G2_q100_SU_lmb3 = read_pickle(filelist_SU3, nboot=pars.nboot, nbin=pars.nbin)
    G2_q000_SS_lmb4 = read_pickle(filelist_SS4, nboot=pars.nboot, nbin=pars.nbin)
    G2_sigm.append(G2_q100_SU_lmb)
    G2_sigm.append(G2_q000_SS_lmb2)
    G2_sigm.append(G2_q100_SU_lmb3)
    G2_sigm.append(G2_q000_SS_lmb4)

    ### ----------------------------------------------------------------------
    ### US & UU
    filelist_US1 = pickledir / Path(
        "baryon-3pt_US_lmb_TBC/barspec/32x64/slrc/kp121040kp120620/sh_gij_p21_90-sh_gij_p21_90/"
        + mom_strings[1]
        + barspec_nameU
    )
    filelist_UU2 = pickledir / Path(
        "baryon-3pt_UU_lmb2_TBC/barspec/32x64/slrc/kp121040kp121040/sh_gij_p21_90-sh_gij_p21_90/"
        + mom_strings[1]
        + barspec_nameU
    )
    filelist_US3 = pickledir / Path(
        "baryon-3pt_US_lmb3_TBC/barspec/32x64/slrc/kp121040kp120620/sh_gij_p21_90-sh_gij_p21_90/"
        + mom_strings[1]
        + barspec_nameU
    )
    filelist_UU4 = pickledir / Path(
        "baryon-3pt_UU_lmb4_TBC/barspec/32x64/slrc/kp121040kp121040/sh_gij_p21_90-sh_gij_p21_90/"
        + mom_strings[1]
        + barspec_nameU
    )

    G2_q000_US_lmb = read_pickle(filelist_US1, nboot=pars.nboot, nbin=pars.nbin)
    G2_q100_UU_lmb2 = read_pickle(filelist_UU2, nboot=pars.nboot, nbin=pars.nbin)
    G2_q000_US_lmb3 = read_pickle(filelist_US3, nboot=pars.nboot, nbin=pars.nbin)
    G2_q100_UU_lmb4 = read_pickle(filelist_UU4, nboot=pars.nboot, nbin=pars.nbin)
    G2_nucl.append(G2_q000_US_lmb)
    G2_nucl.append(G2_q100_UU_lmb2)
    G2_nucl.append(G2_q000_US_lmb3)
    G2_nucl.append(G2_q100_UU_lmb4)

    return G2_nucl, G2_sigm


def read_correlators3(pars, pickledir, pickledir2, mom_strings):
    """Read the pickle files which contain the correlator data

    The script will check the folders for available files and pick out the files with the highest number of configurations.
    This version reads the correlators for all momenta in mom_strings
    Used in the qmax_analysis script
    """

    ### ----------------------------------------------------------------------
    G2_nucl = []
    G2_sigm = []
    ### ----------------------------------------------------------------------
    ### Unperturbed correlators
    fileup = (
        pickledir
        / Path(
            "baryon-3pt_US_lmb_TBC/barspec/32x64/slrc/kp121040kp120620/sh_gij_p21_90-sh_gij_p21_90/"
            + mom_strings[1]
        )
    ).glob("barspec_nucleon_rel_[0-9]*cfgs.pickle")
    conf_num_list = np.array(
        [int("".join(filter(str.isdigit, l.name))) for l in list(fileup)]
    )
    conf_num = conf_num_list[np.argmax(conf_num_list)]
    print("conf_numU:", conf_num)
    barspec_nameU = "/barspec_nucleon_rel_" + str(conf_num) + "cfgs.pickle"

    filestrange = (
        pickledir2
        / Path(
            "baryon_qcdsf/barspec/32x64/slrc/kp121040kp120620/sh_gij_p21_90-sh_gij_p21_90/"
            + mom_strings[1]
        )
    ).glob("barspec_nucleon_rel_[0-9]*cfgs.pickle")
    conf_num_list = np.array(
        [int("".join(filter(str.isdigit, l.name))) for l in list(filestrange)]
    )
    conf_num = conf_num_list[np.argmax(conf_num_list)]
    print("conf_numS:", conf_num)
    barspec_nameS = "/barspec_nucleon_rel_" + str(conf_num) + "cfgs.pickle"

    for mom in mom_strings:
        unpertfile_nucleon_pos = pickledir2 / Path(
            "baryon_qcdsf/barspec/32x64/slrc/kp121040kp121040/sh_gij_p21_90-sh_gij_p21_90/"
            + mom
            + barspec_nameS
        )
        print(unpertfile_nucleon_pos)
        unpertfile_sigma = pickledir2 / Path(
            "baryon_qcdsf/barspec/32x64/slrc/kp121040kp120620/sh_gij_p21_90-sh_gij_p21_90/"
            + mom
            + barspec_nameS
        )
        print(unpertfile_sigma)
        G2_unpert_qp100_nucl = read_pickle(
            unpertfile_nucleon_pos, nboot=pars.nboot, nbin=pars.nbin
        )
        G2_unpert_q000_sigma = read_pickle(
            unpertfile_sigma, nboot=pars.nboot, nbin=pars.nbin
        )
        G2_nucl.append(G2_unpert_qp100_nucl)
        G2_sigm.append(G2_unpert_q000_sigma)

    return G2_nucl, G2_sigm

    ### ----------------------------------------------------------------------
    ### SU & SS
    filelist_SU1 = pickledir2 / Path(
        "baryon-3pt_SU_lmb_TBC/barspec/32x64/slrc/kp121040kp121040/sh_gij_p21_90-sh_gij_p21_90/"
        + mom_strings[1]
        + barspec_nameS
    )
    filelist_SU3 = pickledir2 / Path(
        "baryon-3pt_SU_lmb3_TBC/barspec/32x64/slrc/kp121040kp121040/sh_gij_p21_90-sh_gij_p21_90/"
        + mom_strings[1]
        + barspec_nameS
    )
    filelist_SS2 = pickledir2 / Path(
        "baryon-3pt_SS_lmb2_TBC/barspec/32x64/slrc/kp121040kp120620/sh_gij_p21_90-sh_gij_p21_90/"
        + mom_strings[1]
        + barspec_nameS
    )
    filelist_SS4 = pickledir2 / Path(
        "baryon-3pt_SS_lmb4_TBC/barspec/32x64/slrc/kp121040kp120620/sh_gij_p21_90-sh_gij_p21_90/"
        + mom_strings[1]
        + barspec_nameS
    )

    G2_q100_SU_lmb = read_pickle(filelist_SU1, nboot=pars.nboot, nbin=pars.nbin)
    G2_q000_SS_lmb2 = read_pickle(filelist_SS2, nboot=pars.nboot, nbin=pars.nbin)
    G2_q100_SU_lmb3 = read_pickle(filelist_SU3, nboot=pars.nboot, nbin=pars.nbin)
    G2_q000_SS_lmb4 = read_pickle(filelist_SS4, nboot=pars.nboot, nbin=pars.nbin)
    G2_sigm.append(G2_q100_SU_lmb)
    G2_sigm.append(G2_q000_SS_lmb2)
    G2_sigm.append(G2_q100_SU_lmb3)
    G2_sigm.append(G2_q000_SS_lmb4)

    ### ----------------------------------------------------------------------
    ### US & UU
    filelist_US1 = pickledir / Path(
        "baryon-3pt_US_lmb_TBC/barspec/32x64/slrc/kp121040kp120620/sh_gij_p21_90-sh_gij_p21_90/"
        + mom_strings[1]
        + barspec_nameU
    )
    filelist_UU2 = pickledir / Path(
        "baryon-3pt_UU_lmb2_TBC/barspec/32x64/slrc/kp121040kp121040/sh_gij_p21_90-sh_gij_p21_90/"
        + mom_strings[1]
        + barspec_nameU
    )
    filelist_US3 = pickledir / Path(
        "baryon-3pt_US_lmb3_TBC/barspec/32x64/slrc/kp121040kp120620/sh_gij_p21_90-sh_gij_p21_90/"
        + mom_strings[1]
        + barspec_nameU
    )
    filelist_UU4 = pickledir / Path(
        "baryon-3pt_UU_lmb4_TBC/barspec/32x64/slrc/kp121040kp121040/sh_gij_p21_90-sh_gij_p21_90/"
        + mom_strings[1]
        + barspec_nameU
    )

    G2_q000_US_lmb = read_pickle(filelist_US1, nboot=pars.nboot, nbin=pars.nbin)
    G2_q100_UU_lmb2 = read_pickle(filelist_UU2, nboot=pars.nboot, nbin=pars.nbin)
    G2_q000_US_lmb3 = read_pickle(filelist_US3, nboot=pars.nboot, nbin=pars.nbin)
    G2_q100_UU_lmb4 = read_pickle(filelist_UU4, nboot=pars.nboot, nbin=pars.nbin)
    G2_nucl.append(G2_q000_US_lmb)
    G2_nucl.append(G2_q100_UU_lmb2)
    G2_nucl.append(G2_q000_US_lmb3)
    G2_nucl.append(G2_q100_UU_lmb4)

    return G2_nucl, G2_sigm


def read_correlators4(pars, pickledir, pickledir2, mom_strings):
    """Read the pickle files which contain the correlator data

    The script will check the folders for available files and pick out the files with the highest number of configurations.
    This version reads only zero momentum correlators as it is for the case where all the momentum is contained in the twisted boundary conditions.
    Used when the "qmax" parameter is in the data file
    """

    ### ----------------------------------------------------------------------
    G2_nucl = []
    G2_sigm = []
    ### ----------------------------------------------------------------------
    ### Find the pickle files with the highest number of configurations
    fileup = (
        pickledir
        / Path(
            "baryon-3pt_US_lmb_TBC/barspec/32x64/slrc/kp121040kp120620/sh_gij_p21_90-sh_gij_p21_90/"
            + mom_strings[1]
        )
    ).glob("barspec_nucleon_rel_[0-9]*cfgs.pickle")
    conf_num_list = np.array(
        [int("".join(filter(str.isdigit, l.name))) for l in list(fileup)]
    )
    conf_num = conf_num_list[np.argmax(conf_num_list)]
    print("conf_numU:", conf_num)
    barspec_nameU = "/barspec_nucleon_rel_" + str(conf_num) + "cfgs.pickle"

    filestrange = (
        pickledir2
        / Path(
            "baryon_qcdsf/barspec/32x64/slrc/kp121040kp120620/sh_gij_p21_90-sh_gij_p21_90/"
            + mom_strings[1]
        )
    ).glob("barspec_nucleon_rel_[0-9]*cfgs.pickle")
    conf_num_list = np.array(
        [int("".join(filter(str.isdigit, l.name))) for l in list(filestrange)]
    )
    conf_num = conf_num_list[np.argmax(conf_num_list)]
    print("conf_numS:", conf_num)
    barspec_nameS = "/barspec_nucleon_rel_" + str(conf_num) + "cfgs.pickle"

    ### ----------------------------------------------------------------------
    ### Unperturbed correlators
    unpertfile_nucleon_pos = pickledir2 / Path(
        "baryon_qcdsf/barspec/32x64/slrc/kp121040kp121040/sh_gij_p21_90-sh_gij_p21_90/"
        + mom_strings[1]
        + barspec_nameS
    )
    unpertfile_sigma = pickledir2 / Path(
        "baryon_qcdsf/barspec/32x64/slrc/kp121040kp120620/sh_gij_p21_90-sh_gij_p21_90/"
        + mom_strings[1]
        + barspec_nameS
    )

    G2_unpert_qp100_nucl = read_pickle(
        unpertfile_nucleon_pos, nboot=pars.nboot, nbin=pars.nbin
    )
    G2_unpert_q000_sigma = read_pickle(
        unpertfile_sigma, nboot=pars.nboot, nbin=pars.nbin
    )
    G2_nucl.append(G2_unpert_qp100_nucl[:, :, 0] + 1j * G2_unpert_qp100_nucl[:, :, 1])
    G2_sigm.append(G2_unpert_q000_sigma[:, :, 0] + 1j * G2_unpert_q000_sigma[:, :, 1])
    # G2_nucl.append(G2_unpert_qp100_nucl)
    # G2_sigm.append(G2_unpert_q000_sigma)

    ### ----------------------------------------------------------------------
    ### SU & SS
    filelist_SU1 = pickledir2 / Path(
        "baryon-3pt_SU_lmb_TBC/barspec/32x64/slrc/kp121040kp121040/sh_gij_p21_90-sh_gij_p21_90/"
        + mom_strings[1]
        + barspec_nameS
    )
    filelist_SU3 = pickledir2 / Path(
        "baryon-3pt_SU_lmb3_TBC/barspec/32x64/slrc/kp121040kp121040/sh_gij_p21_90-sh_gij_p21_90/"
        + mom_strings[1]
        + barspec_nameS
    )
    filelist_SS2 = pickledir2 / Path(
        "baryon-3pt_SS_lmb2_TBC/barspec/32x64/slrc/kp121040kp120620/sh_gij_p21_90-sh_gij_p21_90/"
        + mom_strings[1]
        + barspec_nameS
    )
    filelist_SS4 = pickledir2 / Path(
        "baryon-3pt_SS_lmb4_TBC/barspec/32x64/slrc/kp121040kp120620/sh_gij_p21_90-sh_gij_p21_90/"
        + mom_strings[1]
        + barspec_nameS
    )

    G2_q100_SU_lmb = read_pickle(filelist_SU1, nboot=pars.nboot, nbin=pars.nbin)
    G2_q000_SS_lmb2 = read_pickle(filelist_SS2, nboot=pars.nboot, nbin=pars.nbin)
    G2_q100_SU_lmb3 = read_pickle(filelist_SU3, nboot=pars.nboot, nbin=pars.nbin)
    G2_q000_SS_lmb4 = read_pickle(filelist_SS4, nboot=pars.nboot, nbin=pars.nbin)
    G2_sigm.append(G2_q100_SU_lmb[:, :, 0] + 1j * G2_q100_SU_lmb[:, :, 1])
    G2_sigm.append(G2_q000_SS_lmb2[:, :, 0] + 1j * G2_q000_SS_lmb2[:, :, 1])
    G2_sigm.append(G2_q100_SU_lmb3[:, :, 0] + 1j * G2_q100_SU_lmb3[:, :, 1])
    G2_sigm.append(G2_q000_SS_lmb4[:, :, 0] + 1j * G2_q000_SS_lmb4[:, :, 1])
    # G2_sigm.append(G2_q100_SU_lmb)
    # G2_sigm.append(G2_q000_SS_lmb2)
    # G2_sigm.append(G2_q100_SU_lmb3)
    # G2_sigm.append(G2_q000_SS_lmb4)

    ### ----------------------------------------------------------------------
    ### US & UU
    filelist_US1 = pickledir / Path(
        "baryon-3pt_US_lmb_TBC/barspec/32x64/slrc/kp121040kp120620/sh_gij_p21_90-sh_gij_p21_90/"
        + mom_strings[1]
        + barspec_nameU
    )
    filelist_UU2 = pickledir / Path(
        "baryon-3pt_UU_lmb2_TBC/barspec/32x64/slrc/kp121040kp121040/sh_gij_p21_90-sh_gij_p21_90/"
        + mom_strings[1]
        + barspec_nameU
    )
    filelist_US3 = pickledir / Path(
        "baryon-3pt_US_lmb3_TBC/barspec/32x64/slrc/kp121040kp120620/sh_gij_p21_90-sh_gij_p21_90/"
        + mom_strings[1]
        + barspec_nameU
    )
    filelist_UU4 = pickledir / Path(
        "baryon-3pt_UU_lmb4_TBC/barspec/32x64/slrc/kp121040kp121040/sh_gij_p21_90-sh_gij_p21_90/"
        + mom_strings[1]
        + barspec_nameU
    )

    G2_q000_US_lmb = read_pickle(filelist_US1, nboot=pars.nboot, nbin=pars.nbin)
    G2_q100_UU_lmb2 = read_pickle(filelist_UU2, nboot=pars.nboot, nbin=pars.nbin)
    G2_q000_US_lmb3 = read_pickle(filelist_US3, nboot=pars.nboot, nbin=pars.nbin)
    G2_q100_UU_lmb4 = read_pickle(filelist_UU4, nboot=pars.nboot, nbin=pars.nbin)
    G2_nucl.append(G2_q000_US_lmb[:, :, 0] + 1j * G2_q000_US_lmb[:, :, 1])
    G2_nucl.append(G2_q100_UU_lmb2[:, :, 0] + 1j * G2_q100_UU_lmb2[:, :, 1])
    G2_nucl.append(G2_q000_US_lmb3[:, :, 0] + 1j * G2_q000_US_lmb3[:, :, 1])
    G2_nucl.append(G2_q100_UU_lmb4[:, :, 0] + 1j * G2_q100_UU_lmb4[:, :, 1])
    # G2_nucl.append(G2_q000_US_lmb)
    # G2_nucl.append(G2_q100_UU_lmb2)
    # G2_nucl.append(G2_q000_US_lmb3)
    # G2_nucl.append(G2_q100_UU_lmb4)

    return G2_nucl, G2_sigm


def read_correlators5(pars, pickledir, pickledir2, mom_strings):
    """Read the pickle files which contain the correlator data

    The script will check the folders for available files and pick out the files with the highest number of configurations.
    """
    ### ----------------------------------------------------------------------
    ### find the highest number of configurations available
    ### Unperturbed correlators
    fileup = (
        pickledir
        / Path(
            "baryon-3pt_US_lmb_TBC/barspec/32x64/slrc/kp121040kp120620/sh_gij_p21_90-sh_gij_p21_90/"
            + mom_strings[1]
        )
    ).glob("barspec_nucleon_rel_[0-9]*cfgs.pickle")
    conf_num_list = np.array(
        [int("".join(filter(str.isdigit, l.name))) for l in list(fileup)]
    )
    conf_num = conf_num_list[np.argmax(conf_num_list)]
    print("conf_numU:", conf_num)
    barspec_nameU = "/barspec_nucleon_rel_" + str(conf_num) + "cfgs.pickle"

    filestrange = (
        pickledir2
        / Path(
            "baryon_qcdsf/barspec/32x64/slrc/kp121040kp120620/sh_gij_p21_90-sh_gij_p21_90/"
            + mom_strings[1]
        )
    ).glob("barspec_nucleon_rel_[0-9]*cfgs.pickle")
    conf_num_list = np.array(
        [int("".join(filter(str.isdigit, l.name))) for l in list(filestrange)]
    )
    conf_num = conf_num_list[np.argmax(conf_num_list)]
    print("conf_numS:", conf_num)
    barspec_nameS = "/barspec_nucleon_rel_" + str(conf_num) + "cfgs.pickle"

    # files = (
    #     pickledir
    #     / Path(
    #         "baryon_qcdsf_TBC/barspec/32x64/slrc/kp121040kp121040/sh_gij_p21_90-sh_gij_p21_90/p+0+0+0/"
    #     )
    # ).glob("barspec_nucleon_rel_[0-9]*cfgs.pickle")
    # # Strip the conf number from the file names
    # conf_num_list = np.array(
    #     [int("".join(filter(str.isdigit, l.name))) for l in list(files)]
    # )
    # print(conf_num_list)
    # # conf_num_list = [100] # hard code a choice
    # conf_num = conf_num_list[np.argmax(conf_num_list)]
    # print("conf_num:", conf_num)
    # barspec_nameU = "/barspec_nucleon_rel_" + str(conf_num) + "cfgs.pickle"
    # barspec_name = "/barspec_nucleon_rel_" + str(conf_num) + "cfgs.pickle"

    ### ----------------------------------------------------------------------
    G2_nucl = []
    G2_sigm = []
    ### ----------------------------------------------------------------------
    ### Unperturbed correlators
    unpertfile_nucleon_pos = pickledir / Path(
        "baryon_qcdsf_TBC/barspec/32x64/slrc/kp121040kp121040/sh_gij_p21_90-sh_gij_p21_90/"
        + mom_strings[1]
        + barspec_nameU
    )
    unpertfile_sigma = pickledir2 / Path(
        "baryon_qcdsf/barspec/32x64/slrc/kp121040kp120620/sh_gij_p21_90-sh_gij_p21_90/"
        + mom_strings[1]
        + barspec_nameS
    )
    G2_unpert_qp100_nucl = read_pickle(
        unpertfile_nucleon_pos, nboot=pars.nboot, nbin=pars.nbin
    )
    G2_unpert_q000_sigma = read_pickle(
        unpertfile_sigma, nboot=pars.nboot, nbin=pars.nbin
    )
    G2_nucl.append(G2_unpert_qp100_nucl)
    G2_sigm.append(G2_unpert_q000_sigma)

    ### ----------------------------------------------------------------------
    ### SU & SS
    filelist_SU1 = pickledir2 / Path(
        "baryon-3pt_SU_lmb_TBC/barspec/32x64/slrc/kp121040kp121040/sh_gij_p21_90-sh_gij_p21_90/"
        + mom_strings[1]
        + barspec_nameS
    )
    filelist_SU3 = pickledir2 / Path(
        "baryon-3pt_SU_lmb3_TBC/barspec/32x64/slrc/kp121040kp121040/sh_gij_p21_90-sh_gij_p21_90/"
        + mom_strings[1]
        + barspec_nameS
    )
    filelist_SS2 = pickledir2 / Path(
        "baryon-3pt_SS_lmb2_TBC/barspec/32x64/slrc/kp121040kp120620/sh_gij_p21_90-sh_gij_p21_90/"
        + mom_strings[1]
        + barspec_nameS
    )
    filelist_SS4 = pickledir2 / Path(
        "baryon-3pt_SS_lmb4_TBC/barspec/32x64/slrc/kp121040kp120620/sh_gij_p21_90-sh_gij_p21_90/"
        + mom_strings[1]
        + barspec_nameS
    )

    G2_q100_SU_lmb = read_pickle(filelist_SU1, nboot=pars.nboot, nbin=pars.nbin)
    G2_q000_SS_lmb2 = read_pickle(filelist_SS2, nboot=pars.nboot, nbin=pars.nbin)
    G2_q100_SU_lmb3 = read_pickle(filelist_SU3, nboot=pars.nboot, nbin=pars.nbin)
    G2_q000_SS_lmb4 = read_pickle(filelist_SS4, nboot=pars.nboot, nbin=pars.nbin)
    G2_sigm.append(G2_q100_SU_lmb)
    G2_sigm.append(G2_q000_SS_lmb2)
    G2_sigm.append(G2_q100_SU_lmb3)
    G2_sigm.append(G2_q000_SS_lmb4)

    ### ----------------------------------------------------------------------
    ### US & UU
    filelist_US1 = pickledir / Path(
        "baryon-3pt_US_lmb_TBC/barspec/32x64/slrc/kp121040kp120620/sh_gij_p21_90-sh_gij_p21_90/"
        + mom_strings[1]
        + barspec_nameU
    )
    filelist_UU2 = pickledir / Path(
        "baryon-3pt_UU_lmb2_TBC/barspec/32x64/slrc/kp121040kp121040/sh_gij_p21_90-sh_gij_p21_90/"
        + mom_strings[1]
        + barspec_nameU
    )
    filelist_US3 = pickledir / Path(
        "baryon-3pt_US_lmb3_TBC/barspec/32x64/slrc/kp121040kp120620/sh_gij_p21_90-sh_gij_p21_90/"
        + mom_strings[1]
        + barspec_nameU
    )
    filelist_UU4 = pickledir / Path(
        "baryon-3pt_UU_lmb4_TBC/barspec/32x64/slrc/kp121040kp121040/sh_gij_p21_90-sh_gij_p21_90/"
        + mom_strings[1]
        + barspec_nameU
    )

    G2_q000_US_lmb = read_pickle(filelist_US1, nboot=pars.nboot, nbin=pars.nbin)
    G2_q100_UU_lmb2 = read_pickle(filelist_UU2, nboot=pars.nboot, nbin=pars.nbin)
    G2_q000_US_lmb3 = read_pickle(filelist_US3, nboot=pars.nboot, nbin=pars.nbin)
    G2_q100_UU_lmb4 = read_pickle(filelist_UU4, nboot=pars.nboot, nbin=pars.nbin)
    G2_nucl.append(G2_q000_US_lmb)
    G2_nucl.append(G2_q100_UU_lmb2)
    G2_nucl.append(G2_q000_US_lmb3)
    G2_nucl.append(G2_q100_UU_lmb4)

    return G2_nucl, G2_sigm


def read_correlators5_complex(pars, pickledir, pickledir2, mom_strings):
    """Read the pickle files which contain the correlator data

    The script will check the folders for available files and pick out the files with the highest number of configurations.
    """
    ### ----------------------------------------------------------------------
    ### find the highest number of configurations available
    ### Unperturbed correlators
    fileup = (
        pickledir
        / Path(
            "baryon-3pt_US_lmb_TBC/barspec/32x64/slrc/kp121040kp120620/sh_gij_p21_90-sh_gij_p21_90/"
            + mom_strings[1]
        )
    ).glob("barspec_nucleon_rel_[0-9]*cfgs.pickle")
    conf_num_list = np.array(
        [int("".join(filter(str.isdigit, l.name))) for l in list(fileup)]
    )
    conf_num = conf_num_list[np.argmax(conf_num_list)]
    print("conf_numU:", conf_num)
    barspec_nameU = "/barspec_nucleon_rel_" + str(conf_num) + "cfgs.pickle"

    filestrange = (
        pickledir2
        / Path(
            "baryon_qcdsf/barspec/32x64/slrc/kp121040kp120620/sh_gij_p21_90-sh_gij_p21_90/"
            + mom_strings[1]
        )
    ).glob("barspec_nucleon_rel_[0-9]*cfgs.pickle")
    conf_num_list = np.array(
        [int("".join(filter(str.isdigit, l.name))) for l in list(filestrange)]
    )
    conf_num = conf_num_list[np.argmax(conf_num_list)]
    print("conf_numS:", conf_num)
    barspec_nameS = "/barspec_nucleon_rel_" + str(conf_num) + "cfgs.pickle"

    ### ----------------------------------------------------------------------
    G2_nucl = []
    G2_sigm = []
    ### ----------------------------------------------------------------------
    ### Unperturbed correlators
    unpertfile_nucleon_pos = pickledir / Path(
        "baryon_qcdsf_TBC/barspec/32x64/slrc/kp121040kp121040/sh_gij_p21_90-sh_gij_p21_90/"
        + mom_strings[1]
        + barspec_nameU
    )
    unpertfile_sigma = pickledir2 / Path(
        "baryon_qcdsf/barspec/32x64/slrc/kp121040kp120620/sh_gij_p21_90-sh_gij_p21_90/"
        + mom_strings[1]
        + barspec_nameS
    )
    G2_unpert_qp100_nucl = read_pickle(
        unpertfile_nucleon_pos, nboot=pars.nboot, nbin=pars.nbin
    )
    G2_unpert_q000_sigma = read_pickle(
        unpertfile_sigma, nboot=pars.nboot, nbin=pars.nbin
    )
    G2_nucl.append(G2_unpert_qp100_nucl[:, :, 0] + 1j * G2_unpert_qp100_nucl[:, :, 1])
    G2_sigm.append(G2_unpert_q000_sigma[:, :, 0] + 1j * G2_unpert_q000_sigma[:, :, 1])

    ### ----------------------------------------------------------------------
    ### SU & SS
    filelist_SU1 = pickledir2 / Path(
        "baryon-3pt_SU_lmb_TBC/barspec/32x64/slrc/kp121040kp121040/sh_gij_p21_90-sh_gij_p21_90/"
        + mom_strings[1]
        + barspec_nameS
    )
    filelist_SU3 = pickledir2 / Path(
        "baryon-3pt_SU_lmb3_TBC/barspec/32x64/slrc/kp121040kp121040/sh_gij_p21_90-sh_gij_p21_90/"
        + mom_strings[1]
        + barspec_nameS
    )
    filelist_SS2 = pickledir2 / Path(
        "baryon-3pt_SS_lmb2_TBC/barspec/32x64/slrc/kp121040kp120620/sh_gij_p21_90-sh_gij_p21_90/"
        + mom_strings[1]
        + barspec_nameS
    )
    filelist_SS4 = pickledir2 / Path(
        "baryon-3pt_SS_lmb4_TBC/barspec/32x64/slrc/kp121040kp120620/sh_gij_p21_90-sh_gij_p21_90/"
        + mom_strings[1]
        + barspec_nameS
    )

    G2_q100_SU_lmb = read_pickle(filelist_SU1, nboot=pars.nboot, nbin=pars.nbin)
    G2_q000_SS_lmb2 = read_pickle(filelist_SS2, nboot=pars.nboot, nbin=pars.nbin)
    G2_q100_SU_lmb3 = read_pickle(filelist_SU3, nboot=pars.nboot, nbin=pars.nbin)
    G2_q000_SS_lmb4 = read_pickle(filelist_SS4, nboot=pars.nboot, nbin=pars.nbin)
    G2_sigm.append(G2_q100_SU_lmb[:, :, 0] + 1j * G2_q100_SU_lmb[:, :, 1])
    G2_sigm.append(G2_q000_SS_lmb2[:, :, 0] + 1j * G2_q000_SS_lmb2[:, :, 1])
    G2_sigm.append(G2_q100_SU_lmb3[:, :, 0] + 1j * G2_q100_SU_lmb3[:, :, 1])
    G2_sigm.append(G2_q000_SS_lmb4[:, :, 0] + 1j * G2_q000_SS_lmb4[:, :, 1])

    ### ----------------------------------------------------------------------
    ### US & UU
    filelist_US1 = pickledir / Path(
        "baryon-3pt_US_lmb_TBC/barspec/32x64/slrc/kp121040kp120620/sh_gij_p21_90-sh_gij_p21_90/"
        + mom_strings[1]
        + barspec_nameU
    )
    filelist_UU2 = pickledir / Path(
        "baryon-3pt_UU_lmb2_TBC/barspec/32x64/slrc/kp121040kp121040/sh_gij_p21_90-sh_gij_p21_90/"
        + mom_strings[1]
        + barspec_nameU
    )
    filelist_US3 = pickledir / Path(
        "baryon-3pt_US_lmb3_TBC/barspec/32x64/slrc/kp121040kp120620/sh_gij_p21_90-sh_gij_p21_90/"
        + mom_strings[1]
        + barspec_nameU
    )
    filelist_UU4 = pickledir / Path(
        "baryon-3pt_UU_lmb4_TBC/barspec/32x64/slrc/kp121040kp121040/sh_gij_p21_90-sh_gij_p21_90/"
        + mom_strings[1]
        + barspec_nameU
    )

    G2_q000_US_lmb = read_pickle(filelist_US1, nboot=pars.nboot, nbin=pars.nbin)
    G2_q100_UU_lmb2 = read_pickle(filelist_UU2, nboot=pars.nboot, nbin=pars.nbin)
    G2_q000_US_lmb3 = read_pickle(filelist_US3, nboot=pars.nboot, nbin=pars.nbin)
    G2_q100_UU_lmb4 = read_pickle(filelist_UU4, nboot=pars.nboot, nbin=pars.nbin)
    G2_nucl.append(G2_q000_US_lmb[:, :, 0] + 1j * G2_q000_US_lmb[:, :, 1])
    G2_nucl.append(G2_q100_UU_lmb2[:, :, 0] + 1j * G2_q100_UU_lmb2[:, :, 1])
    G2_nucl.append(G2_q000_US_lmb3[:, :, 0] + 1j * G2_q000_US_lmb3[:, :, 1])
    G2_nucl.append(G2_q100_UU_lmb4[:, :, 0] + 1j * G2_q100_UU_lmb4[:, :, 1])

    return G2_nucl, G2_sigm


def read_correlators5_complex_mesons(pars, datadir, mom_strings):
    """Read the pickle files which contain the correlator data

    The script will check the folders for available files and pick out the files with the highest number of configurations.
    """
    ### ----------------------------------------------------------------------
    ### find the highest number of configurations available
    ### Unperturbed correlators
    # fileup = (
    #     pickledir
    #     / Path(
    #         "meson-3pt_US_lmb_TBC/messpec/32x64/slrc/kp121040kp120620/sh_gij_p21_90-sh_gij_p21_90/"
    #         + mom_strings[1]
    #     )
    # ).glob("meson_g5-g5_[0-9]*cfgs.pickle")
    # conf_num_list = np.array(
    #     [int("".join(filter(str.isdigit, l.name))) for l in list(fileup)]
    # )
    # conf_num = conf_num_list[np.argmax(conf_num_list)]
    # print("conf_numU:", conf_num)
    # barspec_nameU = "/barspec_nucleon_rel_" + str(conf_num) + "cfgs.pickle"

    # filestrange = (
    #     pickledir2
    #     / Path(
    #         "baryon_qcdsf/barspec/32x64/slrc/kp121040kp120620/sh_gij_p21_90-sh_gij_p21_90/"
    #         + mom_strings[1]
    #     )
    # ).glob("barspec_nucleon_rel_[0-9]*cfgs.pickle")
    # conf_num_list = np.array(
    #     [int("".join(filter(str.isdigit, l.name))) for l in list(filestrange)]
    # )
    # conf_num = conf_num_list[np.argmax(conf_num_list)]
    # print("conf_numS:", conf_num)
    # barspec_nameS = "/barspec_nucleon_rel_" + str(conf_num) + "cfgs.pickle"

    ### ----------------------------------------------------------------------
    G2_pion = []
    G2_kaon = []
    ### ----------------------------------------------------------------------
    ### Unperturbed correlators
    unpert_pion_file = datadir / Path(
        f"correlator_data/meson_g5-g5_UU_lmb0_{mom_strings[1]}.pkl"
    )
    unpert_kaon_file = datadir / Path(
        f"correlator_data/meson_g5-g5_SS_lmb0_{mom_strings[1]}.pkl"
    )
    G2_unpert_pion = read_pickle(unpert_pion_file, nboot=pars.nboot, nbin=pars.nbin)
    G2_unpert_kaon = read_pickle(unpert_kaon_file, nboot=pars.nboot, nbin=pars.nbin)
    G2_pion.append(G2_unpert_pion[:, :, 0] + 1j * G2_unpert_pion[:, :, 1])
    G2_kaon.append(G2_unpert_kaon[:, :, 0] + 1j * G2_unpert_kaon[:, :, 1])

    ### ----------------------------------------------------------------------
    ### SU & SS
    filelist_SU1 = datadir / Path(
        f"correlator_data/meson_g5-g5_SU_lmb1_{mom_strings[1]}.pkl"
    )
    filelist_SU3 = datadir / Path(
        f"correlator_data/meson_g5-g5_SU_lmb3_{mom_strings[1]}.pkl"
    )
    filelist_SS2 = datadir / Path(
        f"correlator_data/meson_g5-g5_SS_lmb2_{mom_strings[1]}.pkl"
    )
    filelist_SS4 = datadir / Path(
        f"correlator_data/meson_g5-g5_SS_lmb4_{mom_strings[1]}.pkl"
    )

    G2_SU_lmb = read_pickle(filelist_SU1, nboot=pars.nboot, nbin=pars.nbin)
    G2_SS_lmb2 = read_pickle(filelist_SS2, nboot=pars.nboot, nbin=pars.nbin)
    G2_SU_lmb3 = read_pickle(filelist_SU3, nboot=pars.nboot, nbin=pars.nbin)
    G2_SS_lmb4 = read_pickle(filelist_SS4, nboot=pars.nboot, nbin=pars.nbin)
    G2_kaon.append(G2_SU_lmb[:, :, 0] + 1j * G2_SU_lmb[:, :, 1])
    G2_kaon.append(G2_SS_lmb2[:, :, 0] + 1j * G2_SS_lmb2[:, :, 1])
    G2_kaon.append(G2_SU_lmb3[:, :, 0] + 1j * G2_SU_lmb3[:, :, 1])
    G2_kaon.append(G2_SS_lmb4[:, :, 0] + 1j * G2_SS_lmb4[:, :, 1])

    ### ----------------------------------------------------------------------
    ### US & UU
    filelist_US1 = datadir / Path(
        f"correlator_data/meson_g5-g5_US_lmb1_{mom_strings[1]}.pkl"
    )
    filelist_US3 = datadir / Path(
        f"correlator_data/meson_g5-g5_US_lmb3_{mom_strings[1]}.pkl"
    )
    filelist_UU2 = datadir / Path(
        f"correlator_data/meson_g5-g5_UU_lmb2_{mom_strings[1]}.pkl"
    )
    filelist_UU4 = datadir / Path(
        f"correlator_data/meson_g5-g5_UU_lmb4_{mom_strings[1]}.pkl"
    )

    G2_US_lmb = read_pickle(filelist_US1, nboot=pars.nboot, nbin=pars.nbin)
    G2_UU_lmb2 = read_pickle(filelist_UU2, nboot=pars.nboot, nbin=pars.nbin)
    G2_US_lmb3 = read_pickle(filelist_US3, nboot=pars.nboot, nbin=pars.nbin)
    G2_UU_lmb4 = read_pickle(filelist_UU4, nboot=pars.nboot, nbin=pars.nbin)
    G2_pion.append(G2_US_lmb[:, :, 0] + 1j * G2_US_lmb[:, :, 1])
    G2_pion.append(G2_UU_lmb2[:, :, 0] + 1j * G2_UU_lmb2[:, :, 1])
    G2_pion.append(G2_US_lmb3[:, :, 0] + 1j * G2_US_lmb3[:, :, 1])
    G2_pion.append(G2_UU_lmb4[:, :, 0] + 1j * G2_UU_lmb4[:, :, 1])

    return G2_pion, G2_kaon


def read_correlators5_complex_data(pars, datadir, mom_strings):
    """Read the pickle files which contain the correlator data

    The script will check the folders for available files and pick out the files with the highest number of configurations.
    This is my attempt to use the correlator data from the copied pickle files which are in the data directory of this analysis code (for easier copying).
    """
    ### ----------------------------------------------------------------------
    ### find the highest number of configurations available
    ### Unperturbed correlators
    fileup = (
        pickledir
        / Path(
            "baryon-3pt_US_lmb_TBC/barspec/32x64/slrc/kp121040kp120620/sh_gij_p21_90-sh_gij_p21_90/"
            + mom_strings[1]
        )
    ).glob("barspec_nucleon_rel_[0-9]*cfgs.pickle")
    conf_num_list = np.array(
        [int("".join(filter(str.isdigit, l.name))) for l in list(fileup)]
    )
    conf_num = conf_num_list[np.argmax(conf_num_list)]
    print("conf_numU:", conf_num)
    barspec_nameU = "/barspec_nucleon_rel_" + str(conf_num) + "cfgs.pickle"

    filestrange = (
        pickledir2
        / Path(
            "baryon_qcdsf/barspec/32x64/slrc/kp121040kp120620/sh_gij_p21_90-sh_gij_p21_90/"
            + mom_strings[1]
        )
    ).glob("barspec_nucleon_rel_[0-9]*cfgs.pickle")
    conf_num_list = np.array(
        [int("".join(filter(str.isdigit, l.name))) for l in list(filestrange)]
    )
    conf_num = conf_num_list[np.argmax(conf_num_list)]
    print("conf_numS:", conf_num)
    barspec_nameS = "/barspec_nucleon_rel_" + str(conf_num) + "cfgs.pickle"

    ### ----------------------------------------------------------------------
    G2_nucl = []
    G2_sigm = []
    ### ----------------------------------------------------------------------
    ### Unperturbed correlators
    unpertfile_nucleon_pos = pickledir / Path(
        "baryon_qcdsf_TBC/barspec/32x64/slrc/kp121040kp121040/sh_gij_p21_90-sh_gij_p21_90/"
        + mom_strings[1]
        + barspec_nameU
    )
    unpertfile_sigma = pickledir2 / Path(
        "baryon_qcdsf/barspec/32x64/slrc/kp121040kp120620/sh_gij_p21_90-sh_gij_p21_90/"
        + mom_strings[1]
        + barspec_nameS
    )
    G2_unpert_qp100_nucl = read_pickle(
        unpertfile_nucleon_pos, nboot=pars.nboot, nbin=pars.nbin
    )
    G2_unpert_q000_sigma = read_pickle(
        unpertfile_sigma, nboot=pars.nboot, nbin=pars.nbin
    )
    G2_nucl.append(G2_unpert_qp100_nucl[:, :, 0] + 1j * G2_unpert_qp100_nucl[:, :, 1])
    G2_sigm.append(G2_unpert_q000_sigma[:, :, 0] + 1j * G2_unpert_q000_sigma[:, :, 1])

    ### ----------------------------------------------------------------------
    ### SU & SS
    filelist_SU1 = pickledir2 / Path(
        "baryon-3pt_SU_lmb_TBC/barspec/32x64/slrc/kp121040kp121040/sh_gij_p21_90-sh_gij_p21_90/"
        + mom_strings[1]
        + barspec_nameS
    )
    filelist_SU3 = pickledir2 / Path(
        "baryon-3pt_SU_lmb3_TBC/barspec/32x64/slrc/kp121040kp121040/sh_gij_p21_90-sh_gij_p21_90/"
        + mom_strings[1]
        + barspec_nameS
    )
    filelist_SS2 = pickledir2 / Path(
        "baryon-3pt_SS_lmb2_TBC/barspec/32x64/slrc/kp121040kp120620/sh_gij_p21_90-sh_gij_p21_90/"
        + mom_strings[1]
        + barspec_nameS
    )
    filelist_SS4 = pickledir2 / Path(
        "baryon-3pt_SS_lmb4_TBC/barspec/32x64/slrc/kp121040kp120620/sh_gij_p21_90-sh_gij_p21_90/"
        + mom_strings[1]
        + barspec_nameS
    )

    G2_q100_SU_lmb = read_pickle(filelist_SU1, nboot=pars.nboot, nbin=pars.nbin)
    G2_q000_SS_lmb2 = read_pickle(filelist_SS2, nboot=pars.nboot, nbin=pars.nbin)
    G2_q100_SU_lmb3 = read_pickle(filelist_SU3, nboot=pars.nboot, nbin=pars.nbin)
    G2_q000_SS_lmb4 = read_pickle(filelist_SS4, nboot=pars.nboot, nbin=pars.nbin)
    G2_sigm.append(G2_q100_SU_lmb[:, :, 0] + 1j * G2_q100_SU_lmb[:, :, 1])
    G2_sigm.append(G2_q000_SS_lmb2[:, :, 0] + 1j * G2_q000_SS_lmb2[:, :, 1])
    G2_sigm.append(G2_q100_SU_lmb3[:, :, 0] + 1j * G2_q100_SU_lmb3[:, :, 1])
    G2_sigm.append(G2_q000_SS_lmb4[:, :, 0] + 1j * G2_q000_SS_lmb4[:, :, 1])

    ### ----------------------------------------------------------------------
    ### US & UU
    filelist_US1 = pickledir / Path(
        "baryon-3pt_US_lmb_TBC/barspec/32x64/slrc/kp121040kp120620/sh_gij_p21_90-sh_gij_p21_90/"
        + mom_strings[1]
        + barspec_nameU
    )
    filelist_UU2 = pickledir / Path(
        "baryon-3pt_UU_lmb2_TBC/barspec/32x64/slrc/kp121040kp121040/sh_gij_p21_90-sh_gij_p21_90/"
        + mom_strings[1]
        + barspec_nameU
    )
    filelist_US3 = pickledir / Path(
        "baryon-3pt_US_lmb3_TBC/barspec/32x64/slrc/kp121040kp120620/sh_gij_p21_90-sh_gij_p21_90/"
        + mom_strings[1]
        + barspec_nameU
    )
    filelist_UU4 = pickledir / Path(
        "baryon-3pt_UU_lmb4_TBC/barspec/32x64/slrc/kp121040kp121040/sh_gij_p21_90-sh_gij_p21_90/"
        + mom_strings[1]
        + barspec_nameU
    )

    G2_q000_US_lmb = read_pickle(filelist_US1, nboot=pars.nboot, nbin=pars.nbin)
    G2_q100_UU_lmb2 = read_pickle(filelist_UU2, nboot=pars.nboot, nbin=pars.nbin)
    G2_q000_US_lmb3 = read_pickle(filelist_US3, nboot=pars.nboot, nbin=pars.nbin)
    G2_q100_UU_lmb4 = read_pickle(filelist_UU4, nboot=pars.nboot, nbin=pars.nbin)

    G2_nucl.append(G2_q000_US_lmb[:, :, 0] + 1j * G2_q000_US_lmb[:, :, 1])
    G2_nucl.append(G2_q100_UU_lmb2[:, :, 0] + 1j * G2_q100_UU_lmb2[:, :, 1])
    G2_nucl.append(G2_q000_US_lmb3[:, :, 0] + 1j * G2_q000_US_lmb3[:, :, 1])
    G2_nucl.append(G2_q100_UU_lmb4[:, :, 0] + 1j * G2_q100_UU_lmb4[:, :, 1])

    return G2_nucl, G2_sigm


def read_correlators6(pars, pickledir, pickledir2, mom_strings):
    """Read the pickle files which contain the correlator data

    The script will check the folders for available files and pick out the files with the highest number of configurations.
    This version reads only zero momentum correlators as it is for the case where all the momentum is contained in the twisted boundary conditions.
    Used when the "one_fourier" parameter is in the data file
    """

    ### ----------------------------------------------------------------------
    G2_nucl = []
    G2_sigm = []
    ### ----------------------------------------------------------------------
    ### Find the pickle files with the highest number of configurations
    fileup = (
        pickledir
        / Path(
            "baryon-3pt_US_lmb_TBC/barspec/32x64/slrc/kp121040kp120620/sh_gij_p21_90-sh_gij_p21_90/"
            + mom_strings[1]
        )
    ).glob("barspec_nucleon_rel_[0-9]*cfgs.pickle")
    conf_num_list = np.array(
        [int("".join(filter(str.isdigit, l.name))) for l in list(fileup)]
    )
    conf_num = conf_num_list[np.argmax(conf_num_list)]
    print("conf_numU:", conf_num)
    barspec_nameU = "/barspec_nucleon_rel_" + str(conf_num) + "cfgs.pickle"

    filestrange = (
        pickledir2
        / Path(
            "baryon_qcdsf/barspec/32x64/slrc/kp121040kp120620/sh_gij_p21_90-sh_gij_p21_90/"
            + mom_strings[1]
        )
    ).glob("barspec_nucleon_rel_[0-9]*cfgs.pickle")
    conf_num_list = np.array(
        [int("".join(filter(str.isdigit, l.name))) for l in list(filestrange)]
    )
    conf_num = conf_num_list[np.argmax(conf_num_list)]
    print("conf_numS:", conf_num)
    barspec_nameS = "/barspec_nucleon_rel_" + str(conf_num) + "cfgs.pickle"

    ### ----------------------------------------------------------------------
    ### Unperturbed correlators
    unpertfile_nucleon_pos = pickledir2 / Path(
        "baryon_qcdsf/barspec/32x64/slrc/kp121040kp121040/sh_gij_p21_90-sh_gij_p21_90/"
        + mom_strings[2]
        + barspec_nameS
    )
    unpertfile_sigma = pickledir2 / Path(
        "baryon_qcdsf/barspec/32x64/slrc/kp121040kp120620/sh_gij_p21_90-sh_gij_p21_90/"
        + mom_strings[1]
        + barspec_nameS
    )

    G2_unpert_qp100_nucl = read_pickle(
        unpertfile_nucleon_pos, nboot=pars.nboot, nbin=pars.nbin
    )
    G2_unpert_q000_sigma = read_pickle(
        unpertfile_sigma, nboot=pars.nboot, nbin=pars.nbin
    )
    G2_nucl.append(G2_unpert_qp100_nucl[:, :, 0] + 1j * G2_unpert_qp100_nucl[:, :, 1])
    G2_sigm.append(G2_unpert_q000_sigma[:, :, 0] + 1j * G2_unpert_q000_sigma[:, :, 1])

    ### ----------------------------------------------------------------------
    ### SU & SS
    filelist_SU1 = pickledir2 / Path(
        "baryon-3pt_SU_lmb_TBC/barspec/32x64/slrc/kp121040kp121040/sh_gij_p21_90-sh_gij_p21_90/"
        + mom_strings[2]
        + barspec_nameS
    )
    filelist_SU3 = pickledir2 / Path(
        "baryon-3pt_SU_lmb3_TBC/barspec/32x64/slrc/kp121040kp121040/sh_gij_p21_90-sh_gij_p21_90/"
        + mom_strings[2]
        + barspec_nameS
    )
    filelist_SS2 = pickledir2 / Path(
        "baryon-3pt_SS_lmb2_TBC/barspec/32x64/slrc/kp121040kp120620/sh_gij_p21_90-sh_gij_p21_90/"
        + mom_strings[1]
        + barspec_nameS
    )
    filelist_SS4 = pickledir2 / Path(
        "baryon-3pt_SS_lmb4_TBC/barspec/32x64/slrc/kp121040kp120620/sh_gij_p21_90-sh_gij_p21_90/"
        + mom_strings[1]
        + barspec_nameS
    )

    G2_q100_SU_lmb = read_pickle(filelist_SU1, nboot=pars.nboot, nbin=pars.nbin)
    G2_q000_SS_lmb2 = read_pickle(filelist_SS2, nboot=pars.nboot, nbin=pars.nbin)
    G2_q100_SU_lmb3 = read_pickle(filelist_SU3, nboot=pars.nboot, nbin=pars.nbin)
    G2_q000_SS_lmb4 = read_pickle(filelist_SS4, nboot=pars.nboot, nbin=pars.nbin)
    G2_sigm.append(G2_q100_SU_lmb[:, :, 0] + 1j * G2_q100_SU_lmb[:, :, 1])
    G2_sigm.append(G2_q000_SS_lmb2[:, :, 0] + 1j * G2_q000_SS_lmb2[:, :, 1])
    G2_sigm.append(G2_q100_SU_lmb3[:, :, 0] + 1j * G2_q100_SU_lmb3[:, :, 1])
    G2_sigm.append(G2_q000_SS_lmb4[:, :, 0] + 1j * G2_q000_SS_lmb4[:, :, 1])

    ### ----------------------------------------------------------------------
    ### US & UU
    filelist_US1 = pickledir / Path(
        "baryon-3pt_US_lmb_TBC/barspec/32x64/slrc/kp121040kp120620/sh_gij_p21_90-sh_gij_p21_90/"
        + mom_strings[1]
        + barspec_nameU
    )
    filelist_UU2 = pickledir / Path(
        "baryon-3pt_UU_lmb2_TBC/barspec/32x64/slrc/kp121040kp121040/sh_gij_p21_90-sh_gij_p21_90/"
        + mom_strings[2]
        + barspec_nameU
    )
    filelist_US3 = pickledir / Path(
        "baryon-3pt_US_lmb3_TBC/barspec/32x64/slrc/kp121040kp120620/sh_gij_p21_90-sh_gij_p21_90/"
        + mom_strings[1]
        + barspec_nameU
    )
    filelist_UU4 = pickledir / Path(
        "baryon-3pt_UU_lmb4_TBC/barspec/32x64/slrc/kp121040kp121040/sh_gij_p21_90-sh_gij_p21_90/"
        + mom_strings[2]
        + barspec_nameU
    )

    G2_q000_US_lmb = read_pickle(filelist_US1, nboot=pars.nboot, nbin=pars.nbin)
    G2_q100_UU_lmb2 = read_pickle(filelist_UU2, nboot=pars.nboot, nbin=pars.nbin)
    G2_q000_US_lmb3 = read_pickle(filelist_US3, nboot=pars.nboot, nbin=pars.nbin)
    G2_q100_UU_lmb4 = read_pickle(filelist_UU4, nboot=pars.nboot, nbin=pars.nbin)
    G2_nucl.append(G2_q000_US_lmb[:, :, 0] + 1j * G2_q000_US_lmb[:, :, 1])
    G2_nucl.append(G2_q100_UU_lmb2[:, :, 0] + 1j * G2_q100_UU_lmb2[:, :, 1])
    G2_nucl.append(G2_q000_US_lmb3[:, :, 0] + 1j * G2_q000_US_lmb3[:, :, 1])
    G2_nucl.append(G2_q100_UU_lmb4[:, :, 0] + 1j * G2_q100_UU_lmb4[:, :, 1])

    return G2_nucl, G2_sigm


def normalize_matrices(matrices, time_choice=1):
    """Normalize each of the correlator matrices in a list of matrices.

    Using the square root of the product of the diagonal elements of the correlator matrix.
    :param time_choice: sets the time at wich to take the values of the correlators used to normalize the matrices.
    """
    matrix_list = []
    for matrix in matrices:
        matrix_copy = matrix.copy()
        # matrix = matrix.copy()/1e36
        for i, elemi in enumerate(matrix):
            for j, elemj in enumerate(elemi):
                matrix[i, j] = np.einsum(
                    "kl,k->kl",
                    matrix_copy[i, j],
                    np.sqrt(
                        np.abs(
                            matrix_copy[i, i, :, time_choice]
                            * matrix_copy[j, j, :, time_choice]
                        )
                    )
                    ** (-1),
                )
        matrix_list.append(matrix)
    return matrix_list


def make_matrices_real(G2_nucl, G2_sigm, lmb_val):
    matrix_1 = np.array(
        [
            [G2_nucl[0][:, :, 0], lmb_val * G2_nucl[1][:, :, 0]],
            [lmb_val * G2_sigm[1][:, :, 0], G2_sigm[0][:, :, 0]],
        ]
    )
    matrix_2 = np.array(
        [
            [
                G2_nucl[0][:, :, 0] + lmb_val**2 * G2_nucl[2][:, :, 0],
                lmb_val * G2_nucl[1][:, :, 0],
            ],
            [
                lmb_val * G2_sigm[1][:, :, 0],
                G2_sigm[0][:, :, 0] + lmb_val**2 * G2_sigm[2][:, :, 0],
            ],
        ]
    )
    matrix_3 = np.array(
        [
            [
                G2_nucl[0][:, :, 0] + lmb_val**2 * G2_nucl[2][:, :, 0],
                lmb_val * G2_nucl[1][:, :, 0] + lmb_val**3 * G2_nucl[3][:, :, 0],
            ],
            [
                lmb_val * G2_sigm[1][:, :, 0] + lmb_val**3 * G2_sigm[3][:, :, 0],
                G2_sigm[0][:, :, 0] + lmb_val**2 * G2_sigm[2][:, :, 0],
            ],
        ]
    )
    matrix_4 = np.array(
        [
            [
                G2_nucl[0][:, :, 0]
                + (lmb_val**2) * G2_nucl[2][:, :, 0]
                + (lmb_val**4) * G2_nucl[4][:, :, 0],
                lmb_val * G2_nucl[1][:, :, 0] + (lmb_val**3) * G2_nucl[3][:, :, 0],
            ],
            [
                lmb_val * G2_sigm[1][:, :, 0] + (lmb_val**3) * G2_sigm[3][:, :, 0],
                G2_sigm[0][:, :, 0]
                + (lmb_val**2) * G2_sigm[2][:, :, 0]
                + (lmb_val**4) * G2_sigm[4][:, :, 0],
            ],
        ]
    )

    return matrix_1, matrix_2, matrix_3, matrix_4


def make_matrices(G2_nucl, G2_sigm, lmb_val):
    """Construct the matrices for the GEVP

    The 2pt functions have a real and imag dimension.
    """

    matrix_1 = np.array(
        [
            [G2_nucl[0][:, :], lmb_val * G2_nucl[1][:, :]],
            [lmb_val * G2_sigm[1][:, :], G2_sigm[0][:, :]],
        ]
    )
    matrix_2 = np.array(
        [
            [
                G2_nucl[0][:, :] + lmb_val**2 * G2_nucl[2][:, :],
                lmb_val * G2_nucl[1][:, :],
            ],
            [
                lmb_val * G2_sigm[1][:, :],
                G2_sigm[0][:, :] + lmb_val**2 * G2_sigm[2][:, :],
            ],
        ]
    )
    matrix_3 = np.array(
        [
            [
                G2_nucl[0][:, :] + lmb_val**2 * G2_nucl[2][:, :],
                lmb_val * G2_nucl[1][:, :] + lmb_val**3 * G2_nucl[3][:, :],
            ],
            [
                lmb_val * G2_sigm[1][:, :] + lmb_val**3 * G2_sigm[3][:, :],
                G2_sigm[0][:, :] + lmb_val**2 * G2_sigm[2][:, :],
            ],
        ]
    )
    matrix_4 = np.array(
        [
            [
                G2_nucl[0][:, :]
                + (lmb_val**2) * G2_nucl[2][:, :]
                + (lmb_val**4) * G2_nucl[4][:, :],
                lmb_val * G2_nucl[1][:, :] + (lmb_val**3) * G2_nucl[3][:, :],
            ],
            [
                lmb_val * G2_sigm[1][:, :] + (lmb_val**3) * G2_sigm[3][:, :],
                G2_sigm[0][:, :]
                + (lmb_val**2) * G2_sigm[2][:, :]
                + (lmb_val**4) * G2_sigm[4][:, :],
            ],
        ]
    )

    return matrix_1, matrix_2, matrix_3, matrix_4


def fit_correlation_matrix(matrix, t_range, function, diag=False):
    """
    Fit to the correlators in the 2x2 matrix with function.
    If diag is true, then it will skip the off-diagonal elements of the matrix

    Returns a list in the same shape as the matrix with the fit results.
    """
    bootfit_list = [[[], []], [[], []]]
    redchisq_list = [[[], []], [[], []]]
    # print(np.shape(matrix))
    for i, icorr in enumerate(matrix):
        for j, jcorr in enumerate(icorr):
            if diag and ((i == 0 and j == 1) or (i == 1 and j == 0)):
                continue
            else:
                if i == 0 and j == 0:
                    time_range = t_range[0]
                elif i == 1 and j == 1:
                    time_range = t_range[1]
                else:
                    time_range = t_range[2]

                bootfit_, redchisq = fit_value3(
                    np.abs(jcorr), time_range, function, norm=1
                )

                # amp0 = effamp(data, plot=False)
                # mass0 = bs_effmass(data, plot=False)

                # # Set the initial values for the parameters
                # fitfnc.initparfnc(data)

                # fit_bootstrap(
                #     fitfnc, p0, x, data, bounds=None, time=False, fullcov=False
                # )

                # fitlist_2pt = stats.fit_loop_bayes(
                #     twopoint_fn_real,
                #     fitfunction,
                #     time_limits[imom],
                #     plot=False,
                #     disp=True,
                #     time=False,
                #     weights_=True,
                # )

                bootfit_list[i][j] = bootfit_
                redchisq_list[i][j] = redchisq

    return bootfit_list, redchisq_list


def gevp(corr_matrix, time_choice=10, delta_t=1, name="", show=None):
    """Solve the GEVP for a given correlation matrix

    corr_matrix has the matrix indices as the first two, then the bootstrap index and then the time index
    time_choice is the timeslice on which the GEVP will be set
    delta_t is the size of the time evolution which will be used to solve the GEVP
    """
    mat_0 = np.average(corr_matrix[:, :, :, time_choice], axis=2)
    mat_1 = np.average(corr_matrix[:, :, :, time_choice + delta_t], axis=2)

    eval_left, evec_left = np.linalg.eig(np.matmul(mat_1, np.linalg.inv(mat_0)).T)
    eval_right, evec_right = np.linalg.eig(np.matmul(np.linalg.inv(mat_0), mat_1))

    # # Ordering by the square of the first value of the eigenvectors
    # if evec_left[0,0]**2 < evec_left[0,1]**2:
    #     print("sorting left")
    #     eval_left = eval_left[::-1]
    #     evec_left = evec_left[:,::-1]
    # if evec_right[0,0]**2 < evec_right[0,1]**2:
    #     print("sorting right")
    #     eval_right = eval_right[::-1]
    #     evec_right = evec_right[:,::-1]

    # Ordering of the eigenvalues
    if eval_left[0] > eval_left[1]:
        eval_left = eval_left[::-1]
        evec_left = evec_left[:, ::-1]
        # eval_left = eval_left.T[::-1].T
        # evec_left = evec_left.T[::-1].T
    if eval_right[0] > eval_right[1]:
        eval_right = eval_right[::-1]
        evec_right = evec_right[:, ::-1]
        # eval_right = eval_right.T[::-1].T
        # evec_right = evec_right.T[::-1].T
    # print("left:", eval_left, evec_left)
    # print("right:", eval_right, evec_right)

    Gt1 = np.einsum("i,ijkl,j->kl", evec_left[:, 0], corr_matrix, evec_right[:, 0])
    Gt2 = np.einsum("i,ijkl,j->kl", evec_left[:, 1], corr_matrix, evec_right[:, 1])

    if show:
        stats.ploteffmass(Gt1, "eig_1" + name, plotdir, show=True)
        stats.ploteffmass(Gt2, "eig_2" + name, plotdir, show=True)

    gevp_data = [eval_left, evec_left, eval_right, evec_right]

    return Gt1, Gt2, gevp_data


def gevp_bootstrap(corr_matrix, time_choice=10, delta_t=1, name="", show=None):
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

    # Ordering of the eigenvalues
    if eval_left_avg[0] > eval_left_avg[1]:
        eval_left_avg = eval_left_avg[::-1]
        evec_left_avg = evec_left_avg[:, ::-1]
    if eval_right_avg[0] > eval_right_avg[1]:
        eval_right_avg = eval_right_avg[::-1]
        evec_right_avg = evec_right_avg[:, ::-1]

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
        # if boot == 1:
        # if evec_left[1,0]**2 > 0.4:
        # print('\nboot 1  = ', np.matmul(mat_1, np.linalg.inv(mat_0)).T)
        # print('evec sq = ', evec_left[:,0]**2)
        # print('evec sq = ', evec_left, evec_right,'\n')

        # # Ordering by the square of the first value of the eigenvectors
        # if evec_left[0,0]**2 < evec_left[0,1]**2:
        #     print("sorting left")
        #     eval_left = eval_left[::-1]
        #     evec_left = evec_left[:,::-1]
        # if evec_right[0,0]**2 < evec_right[0,1]**2:
        #     print("sorting right")
        #     eval_right = eval_right[::-1]
        #     evec_right = evec_right[:,::-1]

        # # Ordering of the eigenvalues
        if eval_left[0] > eval_left[1]:
            # print("sorting left")
            eval_left = eval_left[::-1]
            evec_left = evec_left[:, ::-1]
        if eval_right[0] > eval_right[1]:
            # print("sorting right")
            eval_right = eval_right[::-1]
            evec_right = evec_right[:, ::-1]
            # eval_right = eval_right.T[::-1].T
            # evec_right = evec_right.T[::-1].T

        evec_left_list.append(evec_left)
        evec_right_list.append(evec_right)
        eval_left_list.append(eval_left)
        eval_right_list.append(eval_right)

    # # Average the bootstrap eigenvectors
    # eval_left_avg = np.average(eval_left_list,axis=0)
    # eval_right_avg = np.average(eval_right_list,axis=0)
    # evec_left_avg = np.average(evec_left_list,axis=0)
    # evec_right_avg = np.average(evec_right_list,axis=0)
    # # Ordering of the eigenvalues
    # if eval_left_avg[0] > eval_left_avg[1]:
    #     eval_left_avg = eval_left_avg[::-1]
    #     evec_left_avg = evec_left_avg[:, ::-1]
    # if eval_right_avg[0] > eval_right_avg[1]:
    #     eval_right_avg = eval_right_avg[::-1]
    #     evec_right_avg = evec_right_avg[:, ::-1]

    # evec_left = np.average(evec_left_list, axis=0)
    # evec_right = np.average(evec_right_list, axis=0)
    # evec_left = np.average(evec_left_list, axis=0)
    # evec_right = np.average(evec_right_list, axis=0)

    # # Ordering of the eigenvalues
    # if eval_left[0] > eval_left[1]:
    #     eval_left_list = [evalu[::-1] for evalu in eval_left_list]
    #     evec_left_list = [evec[:,::-1] for evec in evec_left_list]
    # if eval_right[0] > eval_right[1]:
    #     eval_right_list = [evalu[::-1] for evalu in eval_right_list]
    #     evec_right_list = [evec[:,::-1] for evec in evec_right_list]

    # Gt1 = np.abs(
    #     np.einsum(
    #         "i,ijkl,j->kl", evec_left_avg[:, 0], corr_matrix, evec_right_avg[:, 0]
    #     )
    # )
    # Gt2 = np.abs(
    #     np.einsum(
    #         "i,ijkl,j->kl", evec_left_avg[:, 1], corr_matrix, evec_right_avg[:, 1]
    #     )
    # )

    # Complex corelators
    Gt1 = np.einsum(
        "i,ijkl,j->kl", evec_left_avg[:, 0], corr_matrix, evec_right_avg[:, 0]
    )
    Gt2 = np.einsum(
        "i,ijkl,j->kl", evec_left_avg[:, 1], corr_matrix, evec_right_avg[:, 1]
    )

    if show:
        stats.ploteffmass(Gt1, "eig_1" + name, plotdir, show=True)
        stats.ploteffmass(Gt2, "eig_2" + name, plotdir, show=True)

    gevp_data = [
        np.array(eval_left_list),
        np.array(evec_left_list),
        np.array(eval_right_list),
        np.array(evec_right_list),
    ]
    return Gt1, Gt2, gevp_data


def weighted_avg_1_2_exp(fitlist_1exp, fitlist_2exp, print_=False, tmax_choice=None):
    """Take two lists of dictionaries, one for a fit using the one-exponential function and one using a two-exponential function, Return the weighted average of the energies across these fits"""

    dE_1exp = np.std([i["param"][:, 1] for i in fitlist_1exp], axis=1)
    dof_1exp = np.array([i["dof"] for i in fitlist_1exp])
    chisq_1exp = np.array([i["chisq"] for i in fitlist_1exp])

    if len(fitlist_2exp) > 0:
        dE_2exp = np.std([i["param"][:, 1] for i in fitlist_2exp], axis=1)
        dof_2exp = np.array([i["dof"] for i in fitlist_2exp])
        chisq_2exp = np.array([i["chisq"] for i in fitlist_2exp])
        dof_list = np.append(dof_1exp, dof_2exp)
        chisq_list = np.append(chisq_1exp, chisq_2exp)
        dE_list = np.append(dE_1exp, dE_2exp)
        energies_comb = np.append(
            np.array([i["param"][:, 1] for i in fitlist_1exp]),
            np.array([i["param"][:, 1] for i in fitlist_2exp]),
            axis=0,
        )

    else:
        dof_list = dof_1exp
        chisq_list = chisq_1exp
        dE_list = dE_1exp
        energies_comb = np.array([i["param"][:, 1] for i in fitlist_1exp])

    fitweights = np.array(
        stats.fitweights(
            dof_list,
            chisq_list,
            dE_list,
        )
    )

    weighted_energy = np.dot(fitweights, energies_comb)

    # Rescale the bootstrap error to include the systematic error
    E_avg = np.average(weighted_energy)
    E_staterr = np.std(weighted_energy)
    E_systerr = np.sqrt(
        np.dot(
            fitweights,
            np.array([(E_avg - np.average(energy)) ** 2 for energy in energies_comb]),
        )
    )
    E_comberr = np.sqrt(E_staterr**2 + E_systerr**2)
    for ival, value in enumerate(weighted_energy):
        weighted_energy[ival] = E_avg + (value - E_avg) * E_comberr / E_staterr

    if print_:
        print(
            f"\n+++++\nweighted energy = {err_brackets(np.average(weighted_energy), np.std(weighted_energy))}\n+++++"
        )
    return weighted_energy, fitweights


def weighted_avg(
    fitlist_1exp,
    fitlist_2exp,
    plotdir,
    name,
    tmax_choice=24,
    tminmin_1exp=3,
    tminmax_1exp=16,
    tminmin_2exp=0,
    tminmax_2exp=4,
    plot=False,
):
    """
    Take two lists of dictionaries containing fit information. A selection of the fits will be chosen and the weighted average of these calculated.
    """
    print("\n")

    # Filter out the desired one-exponential fits
    tmax_1exp = np.array([i["x"][-1] for i in fitlist_1exp])
    tmin_1exp = np.array([i["x"][0] for i in fitlist_1exp])
    indices = np.where(tmax_1exp == tmax_choice)[0]
    indices = indices[np.where(tminmin_1exp < tmin_1exp[indices])]
    print(indices)
    indices = indices[np.where(tminmax_1exp > tmin_1exp[indices])]
    print(indices)
    tmax_ = tmax_1exp[indices]
    tmin_ = tmin_1exp[indices]
    reduced_fitlist_1exp = [fitlist_1exp[index] for index in indices]

    # Filter out the desired two-exponential fits
    tmax_2exp = np.array([i["x"][-1] for i in fitlist_2exp])
    tmin_2exp = np.array([i["x"][0] for i in fitlist_2exp])
    indices_2exp = np.where(tmax_2exp == tmax_choice)
    indices_2exp = indices_2exp[0][np.where(tminmin_2exp < tmin_2exp[indices_2exp])]
    indices_2exp = indices_2exp[np.where(tminmax_2exp > tmin_2exp[indices_2exp])]
    tmin_2 = tmin_2exp[indices_2exp]
    reduced_fitlist_2exp = [fitlist_2exp[index] for index in indices_2exp]

    # Calculate the weighted average of the chosen fits.
    weighted_energy, fitweights = weighted_avg_1_2_exp(
        reduced_fitlist_1exp, reduced_fitlist_2exp, print_=False
    )

    if plot:
        # Separate the weights of the two fit functions to allow better plotting
        weights = fitweights[: len(reduced_fitlist_1exp)]
        weights_2 = fitweights[len(reduced_fitlist_1exp) :]

        energies_1exp = np.array([i["param"][:, 1] for i in reduced_fitlist_1exp])
        energies_avg = np.average(energies_1exp, axis=1)
        energies_std = np.std(energies_1exp, axis=1)
        weights_ = weights

        if len(reduced_fitlist_2exp) > 0:
            energies_2exp = np.array([i["param"][:, 1] for i in reduced_fitlist_2exp])
            energies_avg_2 = np.average(energies_2exp, axis=1)
            energies_std_2 = np.std(energies_2exp, axis=1)
            weights_2 = weights_2

        # fig, ax1 = plt.subplots(figsize=(5, 4))
        fig, ax1 = plt.subplots(figsize=(7, 5))
        ax1.errorbar(
            tmin_,
            energies_avg,
            energies_std,
            fmt="s",
            label=r"1-exp",
            color=_colors[0],
            capsize=4,
            elinewidth=1,
            markerfacecolor="none",
        )
        if len(reduced_fitlist_2exp) > 0:
            ax1.errorbar(
                tmin_2,
                energies_avg_2,
                energies_std_2,
                fmt="s",
                label=r"2-exp",
                color=_colors[1],
                capsize=4,
                elinewidth=1,
                markerfacecolor="none",
            )
        ax1.fill_between(
            np.arange(-1, 20),
            np.ones(21) * (np.average(weighted_energy) - np.std(weighted_energy)),
            np.ones(21) * (np.average(weighted_energy) + np.std(weighted_energy)),
            color="r",
            alpha=0.4,
            linewidth=0,
            label=rf"$E={err_brackets(np.average(weighted_energy), np.std(weighted_energy))}$",
        )
        ax1.set_xlabel(r"$t_{\textrm{min}}$")
        ax1.set_ylabel(r"$E$")
        ax1.set_xlim(0, tmin_[-1] + 1)
        ax1.set_xticks(np.arange(np.max(tmin_) + 1))

        ax2 = ax1.twinx()
        ax2.bar(tmin_, weights_, color=_colors[0], alpha=0.3)
        ax2.bar(tmin_2, weights_2, color=_colors[1], alpha=0.3)
        ax2.set_ylabel(r"$\textrm{Weights}$")

        # ax1.legend(fontsize="xx-small", framealpha=1, facecolor='blue')
        # fig.legend(fontsize="xx-small", framealpha=0.8, loc="upper right")
        fig.legend(fontsize="x-small", framealpha=1)
        fig.savefig(
            plotdir / ("tmin_energies_weights_" + name + ".pdf"), metadata=_metadata
        )
    return weighted_energy, fitweights

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

from gevpanalysis.common import read_correlators_complex_mesons_gamma
from gevpanalysis.params import params
from gevpanalysis.common import read_pickle

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
        np.array(eval_left_avg),
        np.array(evec_left_avg),
        np.array(eval_right_avg),
        np.array(evec_right_avg),
    ]
    return Gt1, Gt2, gevp_data


def meson_2pt_projections(
    pickledir,
    plotdir,
    datadir,
    pion_dir="meson_qcdsf",
    kaon_dir="meson_qcdsf",
    time_choice=29,
    delta_t=6,
    label="",
):
    # GEVP
    # time_choice = 25
    # delta_t = 4
    nboot = 500
    nbin = 1
    prev_evecs = None

    conf_num_u = "495"
    # conf_num_u = "496"
    # unpert_pion_dir = "meson_qcdsf"
    # unpert_kaon_dir = "meson_qcdsf"
    dirpart_u = "slrc/kp121040kp121040/"
    dirpart_s = "slrc/kp121040kp120620/"
    # dirpart_s = "slrc/kp120620kp121040/"

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

    pvec_pvec_name = f"/messpec_g5-g5_{conf_num_u}cfgs.pickle"
    pvec_axial_name = f"/messpec_g5-g53_{conf_num_u}cfgs.pickle"
    axial_pvec_name = f"/messpec_g53-g5_{conf_num_u}cfgs.pickle"
    axial_axial_name = f"/messpec_g53-g53_{conf_num_u}cfgs.pickle"

    # ------------------------------------------------------------

    pion_pvec_pvec_file = pickledir / Path(
        f"{pion_dir}/messpec/32x64/{dirpart_u}sh_gij_p21_90-sh_gij_p21_90/p+0+0+0/"
        + pvec_pvec_name
    )
    pion_pvec_axial_file = pickledir / Path(
        f"{pion_dir}/messpec/32x64/{dirpart_u}sh_gij_p21_90-sh_gij_p21_90/p+0+0+0/"
        + pvec_axial_name
    )
    pion_axial_pvec_file = pickledir / Path(
        f"{pion_dir}/messpec/32x64/{dirpart_u}sh_gij_p21_90-sh_gij_p21_90/p+0+0+0/"
        + axial_pvec_name
    )
    pion_axial_axial_file = pickledir / Path(
        f"{pion_dir}/messpec/32x64/{dirpart_u}sh_gij_p21_90-sh_gij_p21_90/p+0+0+0/"
        + axial_axial_name
    )

    pion_pvec_pvec = read_pickle(pion_pvec_pvec_file, nboot=nboot, nbin=nbin)
    pion_pvec_axial = read_pickle(pion_pvec_axial_file, nboot=nboot, nbin=nbin)
    pion_axial_pvec = read_pickle(pion_axial_pvec_file, nboot=nboot, nbin=nbin)
    pion_axial_axial = read_pickle(pion_axial_axial_file, nboot=nboot, nbin=nbin)

    pion_ps_ps = pion_pvec_pvec[:, :, 0] + 1j * pion_pvec_pvec[:, :, 1]
    pion_a_a = pion_axial_axial[:, :, 0] + 1j * pion_axial_axial[:, :, 1]
    pion_ps_a = pion_pvec_axial[:, :, 0] + 1j * pion_pvec_axial[:, :, 1]
    pion_a_ps = pion_axial_pvec[:, :, 0] + 1j * pion_axial_pvec[:, :, 1]

    corr_matrix_pion = np.array(
        [
            [pion_ps_ps, pion_ps_a],
            [pion_a_ps, pion_a_a],
        ]
    )

    (
        Gt1_pion,
        Gt2_pion,
        [eval_left_pion, evec_left_pion, eval_right_pion, evec_right_pion],
    ) = gevp_bootstrap_mesons(
        corr_matrix_pion,
        time_choice,
        delta_t,
        prev_evecs=prev_evecs,
        name="_test",
        show=False,
    )

    # # Plotting
    # plot_proj_corr(
    #     Gt1_pion,
    #     Gt2_pion,
    #     time_choice,
    #     delta_t,
    #     plotdir,
    #     plot_name=f"_pion_{label}",
    # )

    # ======================================================================
    # Kaon GEVP
    kaon_pvec_pvec_file = pickledir / Path(
        f"{kaon_dir}/messpec/32x64/{dirpart_s}sh_gij_p21_90-sh_gij_p21_90/p+0+0+0/"
        + pvec_pvec_name
    )
    kaon_pvec_axial_file = pickledir / Path(
        f"{kaon_dir}/messpec/32x64/{dirpart_s}sh_gij_p21_90-sh_gij_p21_90/p+0+0+0/"
        + pvec_axial_name
    )
    kaon_axial_pvec_file = pickledir / Path(
        f"{kaon_dir}/messpec/32x64/{dirpart_s}sh_gij_p21_90-sh_gij_p21_90/p+0+0+0/"
        + axial_pvec_name
    )
    kaon_axial_axial_file = pickledir / Path(
        f"{kaon_dir}/messpec/32x64/{dirpart_s}sh_gij_p21_90-sh_gij_p21_90/p+0+0+0/"
        + axial_axial_name
    )

    kaon_pvec_pvec = read_pickle(kaon_pvec_pvec_file, nboot=nboot, nbin=nbin)
    kaon_pvec_axial = read_pickle(kaon_pvec_axial_file, nboot=nboot, nbin=nbin)
    kaon_axial_pvec = read_pickle(kaon_axial_pvec_file, nboot=nboot, nbin=nbin)
    kaon_axial_axial = read_pickle(kaon_axial_axial_file, nboot=nboot, nbin=nbin)

    kaon_ps_ps = kaon_pvec_pvec[:, :, 0] + 1j * kaon_pvec_pvec[:, :, 1]
    kaon_a_a = kaon_axial_axial[:, :, 0] + 1j * kaon_axial_axial[:, :, 1]
    kaon_ps_a = kaon_pvec_axial[:, :, 0] + 1j * kaon_pvec_axial[:, :, 1]
    kaon_a_ps = kaon_axial_pvec[:, :, 0] + 1j * kaon_axial_pvec[:, :, 1]

    corr_matrix_kaon = np.array(
        [
            [kaon_ps_ps, kaon_ps_a],
            [kaon_a_ps, kaon_a_a],
        ]
    )

    (
        Gt1_kaon,
        Gt2_kaon,
        [eval_left_kaon, evec_left_kaon, eval_right_kaon, evec_right_kaon],
    ) = gevp_bootstrap_mesons(
        corr_matrix_kaon,
        time_choice,
        delta_t,
        prev_evecs=prev_evecs,
        name="_test",
        show=False,
    )

    # # Plotting
    # plot_proj_corr(
    #     Gt1_kaon,
    #     Gt2_kaon,
    #     time_choice,
    #     delta_t,
    #     plotdir,
    #     plot_name=f"_kaon_{label}",
    # )

    # Save the meson e-vecs.
    with open(
        datadir / (f"pion_GEVP_t0{time_choice}_dt{delta_t}_{label}.pkl"),
        "wb",
    ) as file_out:
        pickle.dump(np.array([evec_left_pion, evec_right_pion]), file_out)
    with open(
        datadir / (f"kaon_GEVP_t0{time_choice}_dt{delta_t}_{label}.pkl"),
        "wb",
    ) as file_out:
        pickle.dump(np.array([evec_left_kaon, evec_right_kaon]), file_out)

    return (
        [evec_left_pion, evec_right_pion, evec_left_kaon, evec_right_kaon],
        [
            Gt1_pion,
            Gt2_pion,
            Gt1_kaon,
            Gt2_kaon,
        ],
        [
            pion_ps_ps,
            pion_a_a,
            kaon_ps_ps,
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
):
    print(np.shape(evecs_pion))
    print(np.shape(evecs_kaon))
    # GEVP
    # time_choice = 25
    # delta_t = 4
    nboot = 500
    nbin = 1
    prev_evecs = None

    conf_num_u = "497"
    dirpart_u = "slrc/kp121040kp121040/"
    # dirpart_s = "slrc/kp121040kp120620/"
    dirpart_s = "slrc/kp120620kp121040/"

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
        evecs_kaon[0][:, 0],
        corr_matrix_p_k,
        evecs_pion[1][:, 0].conj(),
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
        evecs_pion[0][:, 0],
        corr_matrix_k_p,
        evecs_kaon[1][:, 0].conj(),
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
        "i,ijkl,j->kl", evecs_pion[0][:, 0], corr_matrix_p_p, evecs_pion[1][:, 0].conj()
    )

    # ------------------------------------------------------------
    corr_matrix_k_k = np.array(
        [
            [k_k_ps_ps, k_k_ps_a],
            [k_k_a_ps, k_k_a_a],
        ]
    )
    k_k_fwd = np.einsum(
        "i,ijkl,j->kl", evecs_kaon[0][:, 0], corr_matrix_k_k, evecs_kaon[1][:, 0].conj()
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


def plot_proj_corr(
    corr1,
    corr2,
    time_choice,
    delta_t,
    plotdir,
    plot_name="",
    labels=[r"$C_1$", r"$C_1$"],
    ylim=None,
):

    real_1 = np.real(corr1)
    real_2 = np.real(corr2)

    spacing = 1
    effmass_1 = stats.bs_effmass(real_1, time_axis=1, spacing=spacing)
    effmass_2 = stats.bs_effmass(real_2, time_axis=1, spacing=spacing)
    time = np.arange(0, np.shape(corr1)[1])
    efftime = time[:-spacing] + 0.5

    mask1 = np.array([True for time_ in efftime])
    mask2 = np.array([True for time_ in efftime])

    # ======================================================================
    # Plotting the principal correlators
    f, axs = plt.subplots(1, 1, figsize=(7, 5), sharex=True, sharey=True)
    plt.errorbar(
        time,
        np.average(real_1, axis=0),
        np.std(real_1, axis=0),
        capsize=4,
        elinewidth=1,
        color=_colors[0],
        fmt="s",
        mfc="white",
        # label=r"$C_1$",
        label=labels[0],
        zorder=1,
    )
    plt.errorbar(
        time,
        np.average(real_2, axis=0),
        np.std(real_2, axis=0),
        capsize=4,
        elinewidth=1,
        color=_colors[1],
        fmt="^",
        mfc="white",
        # label=r"$C_2$",
        label=labels[1],
        zorder=1,
    )

    plt.grid(True, alpha=0.3)
    plt.legend(fontsize="x-small")
    plt.xlabel(r"$t/a$")
    # plt.ylabel(r"$C_n(t)$")
    plt.ylabel(r"$R(t)$")
    # plt.title(rf"$\lambda={lmb_val:.5f}$")
    # plt.ylim(-0.36, 0.36)

    if ylim is not None:
        plt.ylim(ylim)

    plt.tight_layout()
    plt.savefig(
        plotdir / (f"GEVP_proj_corrs_{plot_name}.pdf"),
        metadata=_metadata,
    )
    # plt.yscale("log")
    # plt.savefig(
    #     plotdir / (f"GEVP_proj_corrs_{plot_name}_log.pdf"),
    #     metadata=_metadata,
    # )
    plt.close()

    # # ======================================================================
    # # Plotting the principal correlators
    # f, axs = plt.subplots(1, 1, figsize=(7, 5), sharex=True, sharey=True)
    # plt.errorbar(
    #     efftime[mask1],
    #     np.average(effmass_1, axis=0)[mask1],
    #     np.std(effmass_1, axis=0)[mask1],
    #     capsize=4,
    #     elinewidth=1,
    #     color=_colors[0],
    #     fmt="s",
    #     mfc="white",
    #     label=r"$C_1$",
    #     zorder=1,
    # )
    # plt.errorbar(
    #     efftime[~mask1],
    #     np.average(effmass_1, axis=0)[~mask1],
    #     np.std(effmass_1, axis=0)[~mask1],
    #     capsize=4,
    #     elinewidth=1,
    #     color=_colors[0],
    #     fmt="s",
    #     mfc="white",
    #     alpha=0.5,
    #     zorder=1,
    # )

    # plt.errorbar(
    #     efftime[mask2],
    #     np.average(effmass_2, axis=0)[mask2],
    #     np.std(effmass_2, axis=0)[mask2],
    #     capsize=4,
    #     elinewidth=1,
    #     color=_colors[1],
    #     fmt="^",
    #     mfc="white",
    #     label=r"$C_2$",
    #     zorder=1,
    # )
    # plt.errorbar(
    #     efftime[~mask2],
    #     np.average(effmass_2, axis=0)[~mask2],
    #     np.std(effmass_2, axis=0)[~mask2],
    #     capsize=4,
    #     elinewidth=1,
    #     color=_colors[1],
    #     fmt="^",
    #     mfc="white",
    #     alpha=0.5,
    #     zorder=1,
    # )

    # plt.grid(True, alpha=0.3)
    # plt.legend(fontsize="x-small")
    # plt.xlabel(r"$t/a$")
    # plt.ylabel(r"$E_{\textrm{eff}}(C_n)$")
    # # plt.ylabel(r"$C_n(t)$")
    # # plt.ylim(-0.36, 0.36)
    # plt.ylim(-0.5, 0.5)
    # plt.tight_layout()
    # plt.savefig(
    #     plotdir / (f"GEVP_eff_proj_corrs_t0{time_choice}_dt{delta_t}{plot_name}.pdf"),
    #     metadata=_metadata,
    # )
    # plt.close()

    return


def plot_proj_corrs(
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
        # effmass_1 = stats.bs_effmass(real_1, time_axis=1, spacing=spacing)
        # efftime = time[:-spacing] + 0.5
        corr_list.append(real_1)
        time_list.append(time)

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
        plotdir / (f"GEVP_proj_corrs_{plot_name}.pdf"),
        metadata=_metadata,
    )
    # plt.yscale("log")
    # plt.savefig(
    #     plotdir / (f"GEVP_proj_corrs_{plot_name}_log.pdf"),
    #     metadata=_metadata,
    # )
    plt.close()

    return


def ratio1(p_k_3pt, k_p_3pt, pion_2pt, kaon_2pt, energy_factor, norm_factor):
    """Ratio with two 3-point functions divided by two 2-point functions"""

    bsdata_1 = np.abs(p_k_3pt.real * k_p_3pt.real)
    denominator1 = (np.abs(kaon_2pt.real * pion_2pt.real)) ** (-1)
    bsdata_1a = np.sqrt(np.einsum("ij,ij->ij", bsdata_1, denominator1))
    bsdata_dbl = np.einsum(
        "ij,i->ij",
        bsdata_1a,
        energy_factor * norm_factor,
    )
    return bsdata_dbl


def meson_gevp_twist():
    """
    Read the meson 3pt functions and 2pt functions. Construct a ratio to fit and plot
    For 3pt correlators at qsq_max
    Making ratios which should equal the renormalised matrix element
    Use the projected operators for the pion and kaon (mixing PS and A_4 operators)
    """

    mystyle = Path("mystyle.txt")
    plt.style.use(mystyle.as_posix())
    plt.rc("text.latex", preamble=r"\usepackage{physics}")

    pickledir = Path("./threept_u-twist0.47829_s-twist0.67308_g8_tau13/")
    pickledir_3pt = Path("./threept_u-twist0.47829_s-twist0.67308_g8_tau13/")
    plotdir = Path("./plots/twisted_tau13/")
    datadir = Path("./data/twisted_tau13/")
    # pickledir = Path("./threept_u-twist0.47829_s-twist0.67308_g8_tau13_2/")
    # pickledir_3pt = Path("./threept_u-twist0.47829_s-twist0.67308_g8_tau13_2/")
    # plotdir = Path("./plots/twisted_tau13_2/")
    # datadir = Path("./data/twisted_tau13_2/")

    datadir2 = Path("./data/")
    datadir.mkdir(parents=True, exist_ok=True)
    plotdir.mkdir(parents=True, exist_ok=True)

    nboot = 500
    nbin = 1
    conf_num_u = "497"

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
    pion_energy = fitlist_pion_cosh[high_weight_pion]["param"][:, 1]
    pion_fit = fitlist_pion_cosh[high_weight_pion]["param"]
    print("pion energy = ", np.average(pion_energy))

    weights_kaon = np.array([i["weight"] for i in fitlist_kaon_cosh])
    high_weight_kaon = np.argmax(weights_kaon)
    kaon_energy = fitlist_kaon_cosh[high_weight_kaon]["param"][:, 1]
    kaon_fit = fitlist_kaon_cosh[high_weight_kaon]["param"]
    print("kaon energy = ", np.average(kaon_energy))

    # ----------------------------------------------------------------------
    # Twisted energies
    pion_utwist_file = datadir2 / (f"time_window_loop_pion_u-twist_Cosh.pkl")
    kaon_stwist_file = datadir2 / (f"time_window_loop_kaon_s-twist_Cosh.pkl")
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
    # Open two-point function data:
    twopt_datadir = Path(
        "/Users/mbatelaan/Research/Adelaide2026/analysis/six_point_fn/data/pickles/run5_meson/"
    )
    with open(twopt_datadir / (f"time_window_loop_pion_Cosh.pkl"), "rb") as file_in:
        fitlist_pion_cosh = pickle.load(file_in)
    weights_pion = np.array([i["weight"] for i in fitlist_pion_cosh])
    high_weight_pion = np.argmax(weights_pion)
    pion_energy_degen = fitlist_pion_cosh[high_weight_pion]["param"][:, 1]

    # ======================================================================
    energy_factor_rest = np.sqrt(4 * pion_energy * kaon_energy)
    energy_factor_utwist = np.sqrt(4 * pion_utwist_energy * kaon_energy)
    energy_factor_stwist = np.sqrt(4 * pion_energy * kaon_stwist_energy)
    norm_factor = 0.863
    print("\n\nenergy factors:")
    print("utwist = ", np.average(energy_factor_utwist))
    print("stwist = ", np.average(energy_factor_stwist))
    print("rest = ", np.average(energy_factor_rest))
    print("\n\n")
    print("pion energy = ", np.average(pion_energy))
    print("kaon energy = ", np.average(kaon_energy))
    print("utwist energy = ", np.average(pion_utwist_energy))
    print("stwist energy = ", np.average(kaon_stwist_energy))
    print("\n\n")

    L = 32
    utwist = 0.47829
    stwist = 0.67308

    # # ======================================================================
    # # Read the twopt GEVP eigenvectors
    # datadir_gevp = Path(
    #     "/Users/mbatelaan/Research/Adelaide2026/analysis/six_point_fn/data/pickles/run1_meson/"
    # )
    # time_choice = 29
    # delta_t = 6

    # # pion
    # with open(
    #     datadir_gevp / (f"pion_GEVP_t0{time_choice}_dt{delta_t}.pkl"),
    #     "rb",
    # ) as file_in:
    #     [evec_left_pion, evec_right_pion] = pickle.load(file_in)

    # # kaon
    # with open(
    #     datadir_gevp / (f"kaon_GEVP_t0{time_choice}_dt{delta_t}.pkl"),
    #     "rb",
    # ) as file_in:
    #     [evec_left_kaon, evec_right_kaon] = pickle.load(file_in)
    # ======================================================================
    # Read the 2pt functions and do the GEVP to get the e-vecs
    # pickledir = Path("./threept_run4")
    # pickledir = Path("./threept_run5")
    pion_dir = "meson_qcdsf"
    kaon_dir = "meson_qcdsf"
    evecs_0twist, Gt_0twist, mesons_2pt = meson_2pt_projections(
        pickledir,
        plotdir,
        datadir,
        pion_dir=pion_dir,
        kaon_dir=kaon_dir,
        label="0twist",
    )

    print("\n\nNow twist")

    pion_dir_twist = "meson-2pt_u-twist"
    kaon_dir_twist = "meson-2pt_s-twist"
    # pion_dir_twist = "meson-2pt_TBC"
    # kaon_dir_twist = "meson-2pt_TBC"
    evecs_twist, Gt_twist, mesons_2pt_twist = meson_2pt_projections(
        pickledir,
        plotdir,
        datadir,
        pion_dir=pion_dir_twist,
        kaon_dir=kaon_dir_twist,
        label="utwist-stwist",
    )
    pion_ps_ps = mesons_2pt[0]
    kaon_ps_ps = mesons_2pt[2]
    pion_a_a = mesons_2pt[1]
    kaon_a_a = mesons_2pt[3]
    pion_utwist_ps_ps = mesons_2pt_twist[0]
    kaon_stwist_ps_ps = mesons_2pt_twist[2]
    pion_utwist_a_a = mesons_2pt_twist[1]
    kaon_stwist_a_a = mesons_2pt_twist[3]
    pion_fwd = Gt_0twist[0]
    kaon_fwd = Gt_0twist[2]
    pion_utwist_fwd = Gt_twist[0]
    kaon_stwist_fwd = Gt_twist[2]

    # stats.ploteffmass(pion_utwist_fwd, "pion", plotdir, show=True)
    # stats.ploteffmass(kaon_stwist_fwd, "kaon", plotdir, show=True)

    threept_dir_utwist = "meson-3pt_US_SU_u-twist"
    threept_dir_stwist = "meson-3pt_US_SU_s-twist"
    # threept_dir_utwist_stwist = "meson-3pt_US_SU_u-twist_s-twist"
    threept_dir_pion = "meson-3pt_UU_SS"
    threept_dir_pion_utwist = "meson-3pt_UU_SS_u-twist_s-twist"
    threept_dir_kaon = "meson-3pt_UU_SS"
    threept_dir_kaon_stwist = "meson-3pt_UU_SS_u-twist_s-twist"

    # ======================================================================
    # pion twisted, kaon at rest
    (
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
    ) = meson_3pt_projections(
        pickledir,
        plotdir,
        datadir,
        evecs_twist[:2],
        evecs_0twist[2:],
        p_k_dir=threept_dir_utwist,
        k_p_dir=threept_dir_utwist,
        p_p_dir=threept_dir_pion_utwist,
        k_k_dir=threept_dir_kaon,
        label="utwist",
    )

    # ======================================================================
    # Make the ratios for the 3pt functions
    tmax = 30
    tau = 13

    ratio1_dbl_utwist_fwd = ratio1(
        p_k_fwd, k_p_fwd, pion_utwist_fwd, kaon_fwd, energy_factor_utwist, norm_factor
    )[:, :tmax]

    plot_corr(
        ratio1_dbl_utwist_fwd,
        plotname="3pt_pi-k_double-ratio_gevp_utwist",
        plotdir=plotdir,
        v_line=10,
        # ylim=(0.315, 0.4),
        ylim=(0.315, 0.33),
    )

    # ----------------------------------------------------------------------
    # double ratio with PS-PS operators
    ratio1_dbl_utwist_ps_ps = ratio1(
        p_k_ps_ps,
        k_p_ps_ps,
        pion_utwist_ps_ps,
        kaon_ps_ps,
        energy_factor_utwist,
        norm_factor,
    )[:, :tmax]

    # ----------------------------------------------------------------------
    # double ratio with A-A operators
    ratio1_dbl_utwist_a_a = ratio1(
        p_k_a_a,
        k_p_a_a,
        pion_utwist_a_a,
        kaon_a_a,
        energy_factor_utwist,
        norm_factor,
    )[:, :tmax]

    # ----------------------------------------------------------------------
    # Plot all ratio1
    plot_proj_corrs(
        # [bsdata_dbl_ps_ps, bsdata_dbl_a_a, bsdata_dbl],
        [ratio1_dbl_utwist_ps_ps, ratio1_dbl_utwist_a_a, ratio1_dbl_utwist_fwd],
        plotdir,
        plot_name="_kaon_and_pion_dbl_ratio1_utwist",
        labels=[r"$\gamma_5,\gamma_5$", r"$\gamma_4\gamma_5,\gamma_4\gamma_5$", "GEVP"],
        # ylim=(0.315, 0.33),
        ylim=(0.30, 0.33),
        # ylim=(0.30, 0.4),
        # ylim=(0.355, 0.385),
        # ylim=(0.30, 0.34),
        title="u-quark twisted",
        ylabel=r"$R_1(t)$",
    )

    # ----------------------------------------------------------------------
    # Ratio with all four 3-point functions
    denominator = np.abs(k_k_fwd[:, :tmax] * p_p_fwd[:, :tmax]) ** (-1)
    numerator = np.abs(p_k_fwd[:, :tmax] * k_p_fwd[:, :tmax])
    bsdata_7a = np.sqrt(np.einsum("ij,ij->ij", numerator, denominator))
    quad_ratio_bs = np.einsum("ij,i->ij", bsdata_7a, energy_factor_utwist)

    plot_corr(
        quad_ratio_bs,
        plotname="3pt_pi-k_quad-ratio_u-twist_fwd",
        plotdir=plotdir,
        v_line=10,
    )
    plot_corr(
        quad_ratio_bs,
        plotname="3pt_pi-k_quad-ratio_u-twist_fwd_ylim",
        plotdir=plotdir,
        v_line=10,
        ylim=(0.315, 0.33),
    )

    # quad ratio with PS-PS operators
    denominator_ps_ps = np.abs(k_k_ps_ps[:, :tmax] * p_p_ps_ps[:, :tmax]) ** (-1)
    numerator_ps_ps = np.abs(p_k_ps_ps[:, :tmax] * k_p_ps_ps[:, :tmax])
    bsdata_7a_ps_ps = np.sqrt(
        np.einsum("ij,ij->ij", numerator_ps_ps, denominator_ps_ps)
    )
    quad_ratio_bs_ps_ps = np.einsum("ij,i->ij", bsdata_7a_ps_ps, energy_factor_utwist)

    denominator_a_a = np.abs(k_k_a_a[:, :tmax] * p_p_a_a[:, :tmax]) ** (-1)
    numerator_a_a = np.abs(p_k_a_a[:, :tmax] * k_p_a_a[:, :tmax])
    bsdata_7a_a_a = np.sqrt(np.einsum("ij,ij->ij", numerator_a_a, denominator_a_a))
    quad_ratio_bs_a_a = np.einsum("ij,i->ij", bsdata_7a_a_a, energy_factor_utwist)

    plot_proj_corrs(
        [quad_ratio_bs_ps_ps, quad_ratio_bs_a_a, quad_ratio_bs],
        plotdir,
        plot_name="_kaon_and_pion_quad_ratio_u-twist",
        labels=[r"$\gamma_5,\gamma_5$", r"$\gamma_4\gamma_5,\gamma_4\gamma_5$", "GEVP"],
        ylim=(0.315, 0.33),
        title="u-quark twisted",
        ylabel=r"$R_2(t)$",
    )

    fit_correlator(
        quad_ratio_bs,
        # quad_ratio_bs_ps_ps,
        plotdir,
        datadir,
        name="3pt_quad_kpi_u-twist_fwd_fit",
        # name="3pt_quad_kpi_u-twist_ps_ps_fit",
        ylabel=r"$R_2(t)$",
        ylim=(0.315, 0.33),
        # time_limits=np.array([[[11, 35], [13, 35]]]),
        time_limits=np.array([[[11, 25], [13, 25]]]),
        tau=13,
    )

    ps_a_ratio = quad_ratio_bs_ps_ps / quad_ratio_bs_a_a
    plot_proj_corrs(
        [ps_a_ratio],
        plotdir,
        plot_name="_kaon_and_pion_quad_ps_a_ratio_u-twist",
        labels=[r"$\gamma_5,\gamma_5/\gamma_4\gamma_5,\gamma_4\gamma_5$"],
        ylim=(0.96, 1.05),
        title="u-quark twisted",
        ylabel=r"$R_2(t)$",
    )

    # ======================================================================
    # pion at rest, kaon twisted
    # s-twist
    (
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
    ) = meson_3pt_projections(
        pickledir_3pt,
        plotdir,
        datadir,
        evecs_0twist[:2],
        evecs_twist[2:],
        # evecs_0twist[2:],
        # evecs_twist[:2],
        # evecs_0twist[2:],
        # evecs_twist[2:],
        p_k_dir=threept_dir_stwist,
        k_p_dir=threept_dir_stwist,
        p_p_dir=threept_dir_pion,
        k_k_dir=threept_dir_kaon_stwist,
        label="stwist",
    )
    # plot_2corr_log(
    #     -k_k_fwd,
    #     k_k_ps_ps,
    #     plotname="k_k_fwd_ps_ps",
    #     plotdir=plotdir,
    #     # v_line=10,
    # )

    # ======================================================================
    # Make the ratios for the 3pt functions
    tmax = 30
    tau = 13

    # ----------------------------------------------------------------------
    # double ratio with two 3-point functions divided by two 2-point functions
    ratio1_dbl_stwist_fwd = ratio1(
        p_k_fwd, k_p_fwd, pion_fwd, kaon_stwist_fwd, energy_factor_stwist, norm_factor
    )[:, :tmax]

    plot_corr(
        # bsdata_dbl,
        ratio1_dbl_stwist_fwd,
        plotname="3pt_pi-k_double-ratio_gevp_stwist",
        plotdir=plotdir,
        v_line=13,
        ylim=(0.315, 0.33),
        # ylim=(0.315, 0.4),
    )

    # ----------------------------------------------------------------------
    # double ratio with PS-PS operators
    ratio1_dbl_stwist_ps_ps = ratio1(
        p_k_ps_ps,
        k_p_ps_ps,
        pion_ps_ps,
        kaon_stwist_ps_ps,
        energy_factor_stwist,
        norm_factor,
    )[:, :tmax]

    # ----------------------------------------------------------------------
    # double ratio with A-A operators
    ratio1_dbl_stwist_a_a = ratio1(
        p_k_a_a,
        k_p_a_a,
        pion_a_a,
        kaon_stwist_a_a,
        energy_factor_stwist,
        norm_factor,
    )[:, :tmax]

    # ----------------------------------------------------------------------
    # Plot all ratio1
    plot_proj_corrs(
        # [bsdata_dbl_ps_ps, bsdata_dbl_a_a, bsdata_dbl],
        [ratio1_dbl_stwist_ps_ps, ratio1_dbl_stwist_a_a, ratio1_dbl_stwist_fwd],
        plotdir,
        plot_name="_kaon_and_pion_dbl_ratio1_stwist",
        labels=[r"$\gamma_5,\gamma_5$", r"$\gamma_4\gamma_5,\gamma_4\gamma_5$", "GEVP"],
        # ylim=(0.315, 0.33),
        ylim=(0.30, 0.33),
        # ylim=(0.30, 0.4),
        # ylim=(0.355, 0.385),
        # ylim=(0.30, 0.34),
        title="s-quark twisted",
        ylabel=r"$R_1(t)$",
    )

    # # ----------------------------------------------------------------------
    # # single 3pt ratio with PS-PS operators
    # # pi -> k
    # numerator1 = np.abs(p_k_ps_ps[:, tau : tau + tmax].real)
    # denominator1 = np.abs(kaon_stwist_ps_ps[:, tau : tau + tmax].real) ** (-1)
    # numerator2 = np.sqrt(
    #     np.abs(pion_ps_ps[:, :tmax].real * kaon_stwist_ps_ps[:, tau : tau + tmax].real)
    # )
    # denominator2 = np.sqrt(
    #     np.abs(kaon_stwist_ps_ps[:, :tmax].real * pion_ps_ps[:, tau : tau + tmax].real)
    # ) ** (-1)
    # const_factor = (
    #     np.sqrt(np.abs(kaon_stwist_ps_ps[:, tau].real / pion_ps_ps[:, tau].real))
    #     * energy_factor_stwist
    #     * norm_factor
    # )

    # prem_ratio1 = numerator1 * denominator1 * numerator2 * denominator2
    # single_ratio_p_k = np.einsum("ij,i->ij", prem_ratio1, const_factor)

    # # k -> pi
    # numerator1 = np.abs(k_p_ps_ps[:, tau : tau + tmax].real)
    # denominator1 = np.abs(pion_ps_ps[:, tau : tau + tmax].real) ** (-1)
    # numerator2 = np.sqrt(
    #     np.abs(kaon_stwist_ps_ps[:, :tmax].real * pion_ps_ps[:, tau : tau + tmax].real)
    # )
    # denominator2 = np.sqrt(
    #     np.abs(pion_ps_ps[:, :tmax].real * kaon_stwist_ps_ps[:, tau : tau + tmax].real)
    # ) ** (-1)
    # const_factor = (
    #     np.sqrt(np.abs(pion_ps_ps[:, tau].real / kaon_stwist_ps_ps[:, tau].real))
    #     * energy_factor_stwist
    #     * norm_factor
    # )

    # prem_ratio1 = numerator1 * denominator1 * numerator2 * denominator2
    # single_ratio_k_p = np.einsum("ij,i->ij", prem_ratio1, const_factor)

    # plot_proj_corr(
    #     single_ratio_p_k,
    #     single_ratio_k_p,
    #     0,
    #     0,
    #     plotdir,
    #     plot_name="_kaon_and_pion_single_ratio_stwist_ps_ps",
    #     # labels=["GEVP", r"$\gamma_5,\gamma_5$"],
    #     labels=[r"$\pi \to K$", r"$K\to \pi$"],
    #     ylim=(0.28, 0.34),
    # )

    # # ----------------------------------------------------------------------
    # # single 3pt ratio with fwd operators
    # # pi -> k
    # numerator1 = np.abs(p_k_fwd[:, tau : tau + tmax].real)
    # denominator1 = np.abs(kaon_stwist_fwd[:, tau : tau + tmax].real) ** (-1)
    # numerator2 = np.sqrt(
    #     np.abs(pion_fwd[:, :tmax].real * kaon_stwist_fwd[:, tau : tau + tmax].real)
    # )
    # denominator2 = np.sqrt(
    #     np.abs(kaon_stwist_fwd[:, :tmax].real * pion_fwd[:, tau : tau + tmax].real)
    # ) ** (-1)
    # const_factor = (
    #     np.sqrt(np.abs(kaon_stwist_fwd[:, tau].real / pion_fwd[:, tau].real))
    #     * energy_factor_stwist
    #     * norm_factor
    # )

    # prem_ratio1 = numerator1 * denominator1 * numerator2 * denominator2
    # single_ratio_p_k_fwd = np.einsum("ij,i->ij", prem_ratio1, const_factor)

    # # k -> pi
    # numerator1 = np.abs(k_p_fwd[:, tau : tau + tmax].real)
    # denominator1 = np.abs(pion_fwd[:, tau : tau + tmax].real) ** (-1)
    # numerator2 = np.sqrt(
    #     np.abs(kaon_stwist_fwd[:, :tmax].real * pion_fwd[:, tau : tau + tmax].real)
    # )
    # denominator2 = np.sqrt(
    #     np.abs(pion_fwd[:, :tmax].real * kaon_stwist_fwd[:, tau : tau + tmax].real)
    # ) ** (-1)
    # const_factor = (
    #     np.sqrt(np.abs(pion_fwd[:, tau].real / kaon_stwist_fwd[:, tau].real))
    #     * energy_factor_stwist
    #     * norm_factor
    # )

    # prem_ratio1 = numerator1 * denominator1 * numerator2 * denominator2
    # single_ratio_k_p_fwd = np.einsum("ij,i->ij", prem_ratio1, const_factor)

    # plot_proj_corr(
    #     single_ratio_p_k_fwd,
    #     single_ratio_k_p_fwd,
    #     0,
    #     0,
    #     plotdir,
    #     plot_name="_kaon_and_pion_single_ratio_stwist_fwd",
    #     # labels=["GEVP", r"$\gamma_5,\gamma_5$"],
    #     labels=[r"$\pi \to K$", r"$K\to \pi$"],
    #     ylim=(0.28, 0.34),
    # )

    # plot_proj_corr(
    #     single_ratio_k_p_fwd,
    #     single_ratio_k_p,
    #     0,
    #     0,
    #     plotdir,
    #     plot_name="_kaon_to_pion_single_ratio_stwist",
    #     labels=["GEVP", r"$\gamma_5,\gamma_5$"],
    #     ylim=(0.28, 0.34),
    # )

    # plot_proj_corr(
    #     single_ratio_p_k_fwd,
    #     single_ratio_p_k,
    #     0,
    #     0,
    #     plotdir,
    #     plot_name="_pion_to_kaon_single_ratio_stwist",
    #     labels=["GEVP", r"$\gamma_5,\gamma_5$"],
    #     ylim=(0.28, 0.34),
    # )

    # ----------------------------------------------------------------------
    # Quad ratio with all four 3-point functions
    denominator = np.abs(k_k_fwd[:, :tmax] * p_p_fwd[:, :tmax]) ** (-1)
    numerator = np.abs(p_k_fwd[:, :tmax] * k_p_fwd[:, :tmax])
    bsdata_7a = np.sqrt(np.einsum("ij,ij->ij", numerator, denominator))
    quad_ratio_bs = np.einsum("ij,i->ij", bsdata_7a, energy_factor_stwist)

    plot_corr(
        quad_ratio_bs,
        plotname="3pt_pi-k_quad-ratio_s-twist_fwd",
        plotdir=plotdir,
        v_line=10,
    )
    plot_corr(
        quad_ratio_bs,
        plotname="3pt_pi-k_quad-ratio_s-twist_fwd_ylim",
        plotdir=plotdir,
        v_line=10,
        ylim=(0.315, 0.33),
    )

    # quad ratio with PS-PS operators
    denominator_ps_ps = np.abs(k_k_ps_ps[:, :tmax] * p_p_ps_ps[:, :tmax]) ** (-1)
    numerator_ps_ps = np.abs(p_k_ps_ps[:, :tmax] * k_p_ps_ps[:, :tmax])
    bsdata_7a_ps_ps = np.sqrt(
        np.einsum("ij,ij->ij", numerator_ps_ps, denominator_ps_ps)
    )
    quad_ratio_bs_ps_ps = np.einsum("ij,i->ij", bsdata_7a_ps_ps, energy_factor_stwist)

    denominator_a_a = np.abs(k_k_a_a[:, :tmax] * p_p_a_a[:, :tmax]) ** (-1)
    numerator_a_a = np.abs(p_k_a_a[:, :tmax] * k_p_a_a[:, :tmax])
    bsdata_7a_a_a = np.sqrt(np.einsum("ij,ij->ij", numerator_a_a, denominator_a_a))
    quad_ratio_bs_a_a = np.einsum("ij,i->ij", bsdata_7a_a_a, energy_factor_stwist)

    plot_proj_corrs(
        [quad_ratio_bs_ps_ps, quad_ratio_bs_a_a, quad_ratio_bs],
        plotdir,
        plot_name="_kaon_and_pion_quad_ratio_s-twist",
        labels=[r"$\gamma_5,\gamma_5$", r"$\gamma_4\gamma_5,\gamma_4\gamma_5$", "GEVP"],
        ylim=(0.315, 0.33),
        title="s-quark twisted",
        ylabel=r"$R_2(t)$",
    )
    fit_correlator(
        quad_ratio_bs,
        # quad_ratio_bs_ps_ps,
        plotdir,
        datadir,
        name="3pt_quad_kpi_s-twist_fwd_fit",
        # name="3pt_quad_kpi_s-twist_ps_ps_fit",
        ylabel=r"$R_2(t)$",
        ylim=(0.315, 0.33),
        # time_limits=np.array([[[11, 35], [13, 35]]]),
        time_limits=np.array([[[11, 25], [13, 25]]]),
        tau=13,
    )
    ps_a_ratio = quad_ratio_bs_ps_ps / quad_ratio_bs_a_a
    plot_proj_corrs(
        [ps_a_ratio],
        plotdir,
        plot_name="_kaon_and_pion_quad_ps_a_ratio_s-twist",
        labels=[r"$\gamma_5,\gamma_5/\gamma_4\gamma_5,\gamma_4\gamma_5$"],
        ylim=(0.96, 1.05),
        title="s-quark twisted",
        ylabel=r"$R_2(t)$",
    )

    return


def plot_2corr_log(bsdata1, bsdata2, plotname="", plotdir=Path("./")):
    time1 = np.arange(0, np.shape(bsdata1)[1])
    yavg1 = np.average(bsdata1, axis=0)
    ystd1 = np.std(bsdata1, axis=0)

    time2 = np.arange(0, np.shape(bsdata2)[1])
    yavg2 = np.average(bsdata2, axis=0)
    ystd2 = np.std(bsdata2, axis=0)

    f, axs = plt.subplots(1, 1, figsize=(9, 6))
    axs.errorbar(
        time1,
        yavg1,
        ystd1,
        capsize=4,
        elinewidth=1,
        color=_colors[0],
        fmt="s",
        mfc="white",
    )
    axs.errorbar(
        time2,
        yavg2,
        ystd2,
        capsize=4,
        elinewidth=1,
        color=_colors[1],
        fmt="o",
        mfc="white",
    )
    # axs.axvline(10, color="k", linewidth=1, linestyle="--")
    axs.set_yscale("log")
    plt.savefig(plotdir / f"{plotname}.pdf")
    plt.show()
    plt.close()
    return


if __name__ == "__main__":
    # meson_gevp_qsqmax()
    meson_gevp_twist()

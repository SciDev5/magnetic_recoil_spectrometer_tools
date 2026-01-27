# Generate Figures for MERGS device

from typing import Protocol

import numpy as np
import matplotlib.pyplot as plt

from mrs_ion_optics import gen_charictaristic_rays, MRSIonOptics
import mrs_ion_optics as K
from cross_sections import (
    gen_cross_section_compton,
    gen_cross_section_pairproduction,
    load_pairproduction_cross_section,
)
from acceptance import (
    SRXMData,
    foil_trace,
    Foil,
    aperture,
    AngleEnergyId,
    ThetaPhiXYEnergyId,
    rays_into_relative,
)
from physical_constants import mol, MeV, millimeter, centimeter
from util import full_width_half_max


#### --------------------------- configuration constants ---------------------------- ####


R_FOIL = 0.015
L_DRIFT = 0.25
R_APERTURE = 0.015


# ---- COSY parameters (taken from a good run of the optimizer)

sess = (
    MRSIonOptics()
    .disable_fit()
    .config_order(1)
    .config_vis_lab_coordinates(False)
    .config_outputs(
        [
            "ME(1,1)",  # magnification (dispersive direction)
            "ME(1,2)",  # focus (dispersive direction)
            "ME(1,6)",  # dispersion (dispersive direction)
            "ME(3,1)",  # magnification (vertical direction)
            "ME(3,2)",  # focus (vertical direction)
            "(VMAX(RAY(1))-VMIN(RAY(1)))/2",  # image size (dispersive direction)
            "(VMAX(RAY(3))-VMIN(RAY(3)))/2",  # image size (vertical direction)
        ],
        do_beamsize=True,
    )
    .set_parameters(
        {
            K.K_p_bend_radius: 0.55,
            K.K_p_bend_angle: 110.0,
            K.K_p_drift_post_aperture: 0.05,
            K.K_p_drift_pre_bend: 0.1711294047732842,
            K.K_p_drift_post_bend: 0.2014765625,
            K.K_p_drift_pre_hodoscope: 0.228115896571998,
            K.K_p_m5a_length: 0.1,
            K.K_p_m5a_quad: -0.06233481854188321,
            K.K_p_m5a_hex: 0.0,
            K.K_p_m5a_oct: 0.0,
            K.K_p_m5a_dec: 0.0,
            K.K_p_m5a_dodec: 0.0,
            K.K_p_m5b_length: 0.1,
            K.K_p_m5b_quad: 0.115469210096311,
            K.K_p_m5b_hex: 0.0,
            K.K_p_m5b_oct: 0.0,
            K.K_p_m5b_dec: 0.0,
            K.K_p_m5b_dodec: 0.0,
            K.K_p_shape_in_1: 0.2985833653414149,
            K.K_p_shape_in_2: 0.0,
            K.K_p_shape_in_3: 0.0,
            K.K_p_shape_in_4: 0.0,
            K.K_p_shape_in_5: 0.0,
            K.K_p_shape_out_1: 0.176719510472977,
            K.K_p_shape_out_2: 0.0,
            K.K_p_shape_out_3: 0.0,
            K.K_p_shape_out_4: 0.0,
            K.K_p_shape_out_5: 0.0,
            K.K_p_drift_m5a_m5b: 0.1,
            K.K_p_m5c_length: 0.1,
            K.K_p_m5c_quad: -0.01585503500342181,
            K.K_p_m5c_hex: 0.0,
            K.K_p_m5c_oct: 0.0,
            K.K_p_m5c_dec: 0.0,
            K.K_p_m5c_dodec: 0.0,
            K.K_p_drift_m5c_m5d: 0.1,
            K.K_p_m5d_length: 0.1,
            K.K_p_m5d_quad: -0.5,
            K.K_p_m5d_hex: 0.0,
            K.K_p_m5d_oct: 0.0,
            K.K_p_m5d_dec: 0.0,
            K.K_p_m5d_dodec: 0.0,
        }
    )
)

# ---- cosy svg output

(
    sess.set_rays(gen_charictaristic_rays(R_FOIL, L_DRIFT, R_APERTURE, n=3), color=6)
    .add_rays(
        gen_charictaristic_rays(R_FOIL, L_DRIFT, R_APERTURE, n=3, energy=-0.1), color=8
    )
    .add_rays(
        gen_charictaristic_rays(R_FOIL, L_DRIFT, R_APERTURE, n=3, energy=0.1), color=7
    )
    .add_rays(
        gen_charictaristic_rays(R_FOIL, L_DRIFT, R_APERTURE, n=3, energy=-0.01), color=3
    )
    .add_rays(
        gen_charictaristic_rays(R_FOIL, L_DRIFT, R_APERTURE, n=3, energy=0.01), color=2
    )
)

with open("./scripts/fig_outputs/mergs_sys.svg", "w", encoding="utf-8") as f:
    f.write(sess.config_vis_lab_coordinates(True).config_order(1).exec_svg()[0])

sess.set_rays(gen_charictaristic_rays(R_FOIL, L_DRIFT, R_APERTURE, n=3), color=6)

(
    _,
    (
        V_magnification,
        V_focus,
        V_dispersion,
        V_magnification_y,
        V_focus_y,
        V_imagesize_x,
        V_imagesize_y,
    ),
    _,
    _,
) = sess.config_order(5).exec()
V_resolution_order_1 = np.abs(2 * R_APERTURE * V_magnification / V_dispersion)
V_resolution_order_5 = np.abs(2 * V_imagesize_x / V_dispersion)
print(f"Magnetic Optics Resolution [order = 1]: {V_resolution_order_1}")
print(f"Magnetic Optics Resolution [order = 5]: {V_resolution_order_5}")

# ---- utility for doing the monte carlo


class EvalMCFn(Protocol):
    def __call__(
        self,
        foil_depth: float,
        do_aperture: bool,
        gamma_energy: float,
        gamma_count: int | None = None,
        elec_count: int | None = None,
    ) -> tuple[ThetaPhiXYEnergyId | AngleEnergyId, float]: ...


def prepare_monte_carlo(
    Z: int,
    x_density: float,
    x_atomic_weight: float,
    material_name_str: str,
) -> tuple[int, str, EvalMCFn]:
    x_number_density = x_density * 1e6 / x_atomic_weight * mol  # [/m^3]

    with open(f"./_data/estar_{material_name_str}.txt", "r", encoding="utf8") as srem:
        x_srem = np.array(
            [[float(y) for y in x.split(" ")[:2]] for x in srem.readlines()[9:]]
        )
        x_srem[:, 0] *= MeV  #  MeV  ->  J
        x_srem[:, 1] *= x_density * MeV / centimeter  #  MeV cm^2 / g  ->  J / m
        x_srem: SRXMData = x_srem[:, 0], x_srem[:, 1]

    x_crosssection_compton = gen_cross_section_compton(
        x_number_density * Z,
        np.linspace(2, 20, 50) * MeV,
        np.linspace(0.0, np.pi, 1000),
    )
    x_crosssection_pairproduction = gen_cross_section_pairproduction(
        Z,
        x_number_density,
        load_pairproduction_cross_section("./_data/pairprod_xsctn_medium.npz"),
    )
    x_foil: Foil = x_srem, [
        x_crosssection_compton,
        x_crosssection_pairproduction,
    ]

    def eval_mc(
        foil_depth: float,
        do_aperture: bool,
        gamma_energy: float,
        gamma_count: int | None = None,
        elec_count: int | None = None,
    ):
        if gamma_count is not None:
            # use gamma_count, ignore elec_count
            pre_aperture, _ = foil_trace(
                n_rays_base=gamma_count,
                n_srxm_steps=100,
                phot_energy_in=gamma_energy,
                foil_properties=x_foil,
                foil_depth=foil_depth,
                apply_conversion_efficiency=True,
            )
            if do_aperture:
                post_aperture = aperture(pre_aperture, R_FOIL, R_APERTURE, L_DRIFT)
                return post_aperture, 1.0
            else:
                # elec_angle, elec_energy, ids, raw_conversion_efficiency = pre_aperture
                return pre_aperture, 1.0
        if elec_count is not None:
            total_gamma_in = 0.0
            if do_aperture:
                post_aperture = (
                    np.zeros(0),
                    np.zeros(0),
                    np.zeros(0),
                    np.zeros(0),
                    np.zeros(0),
                    np.zeros(0, dtype=np.int64),
                )
                while post_aperture[0].shape[0] < elec_count:
                    target_raycount = elec_count - post_aperture[0].shape[0]
                    pre_aperture, efficiency_base = foil_trace(
                        n_rays_base=target_raycount,
                        n_srxm_steps=100,
                        phot_energy_in=gamma_energy,
                        foil_properties=x_foil,
                        foil_depth=foil_depth,
                        apply_conversion_efficiency=False,
                    )
                    post_aperture_ = aperture(
                        pre_aperture,
                        R_FOIL,
                        R_APERTURE,
                        L_DRIFT,
                    )
                    post_aperture = (
                        np.concatenate([post_aperture[0], post_aperture_[0]]),
                        np.concatenate([post_aperture[1], post_aperture_[1]]),
                        np.concatenate([post_aperture[2], post_aperture_[2]]),
                        np.concatenate([post_aperture[3], post_aperture_[3]]),
                        np.concatenate([post_aperture[4], post_aperture_[4]]),
                        np.concatenate([post_aperture[5], post_aperture_[5]]),
                    )
                    total_gamma_in += target_raycount / efficiency_base
                # t, p, x, y, en, i = post_aperture
                return post_aperture, elec_count / total_gamma_in
            else:
                elec_angle = np.zeros(0)
                elec_energy = np.zeros(0)
                ids = np.zeros(0, dtype=np.int64)
                while ids.shape[0] < elec_count:
                    target_raycount = elec_count - ids.shape[0]
                    (elec_angle_, elec_energy_, ids_), efficiency_base = foil_trace(
                        n_rays_base=target_raycount,
                        n_srxm_steps=100,
                        phot_energy_in=gamma_energy,
                        foil_properties=x_foil,
                        foil_depth=foil_depth,
                        apply_conversion_efficiency=False,
                    )
                    total_gamma_in += target_raycount / efficiency_base
                    elec_angle = np.concatenate([elec_angle, elec_angle_])
                    elec_energy = np.concatenate([elec_energy, elec_energy_])
                    ids = np.concatenate([ids, ids_])
                return (elec_angle, elec_energy, ids), elec_count / total_gamma_in

        raise ValueError("Must specify input gamma count or output electron count")

    return (Z, material_name_str, eval_mc)


Material = tuple[int, str, EvalMCFn]
mat_li = prepare_monte_carlo(
    Z=3,
    x_density=0.5334,  # [g/cm^3]
    x_atomic_weight=6.94,  # [amu | g/mol]
    material_name_str="li",
)
mat_be = prepare_monte_carlo(
    Z=4,
    x_density=1.845,  # [g/cm^3]
    x_atomic_weight=9.0122,  # [amu | g/mol]
    material_name_str="be",
)
mat_graphite = prepare_monte_carlo(
    Z=6,
    x_density=2.18,  # [g/cm^3]
    x_atomic_weight=12.011,  # [amu | g/mol]
    material_name_str="graphite",
)
mat_si = prepare_monte_carlo(
    Z=16,
    x_density=2.329085,  # [g/cm^3]
    x_atomic_weight=28.085,  # [amu | g/mol]
    material_name_str="si",
)
mat_fe = prepare_monte_carlo(
    Z=26,
    x_density=7.874,  # [g/cm^3]
    x_atomic_weight=55.845,  # [amu | g/mol]
    material_name_str="fe",
)
mat_au = prepare_monte_carlo(
    Z=79,
    x_density=19.283,  # [g/cm^3]
    x_atomic_weight=196.966570,  # [amu | g/mol]
    material_name_str="au",
)


def material_name(mat: Material, capitalize=False):
    name = mat[1]
    return name[0].upper() + name[1:] if capitalize else name


# ---- energy/angle 2d hist plot (16MeV; Li, Si, Au; d=0.25mm)
def evals_pre_aperture_angle_energy_hist(
    foil_mat: Material,
    foil_depth: float,
    gamma_energy: float,
    gamma_count: int | None = None,
    elec_count: int | None = None,
):
    print(
        f"[energy/angle 2d hist plot] ({material_name(foil_mat)} {foil_depth/millimeter}mm) "
    )
    (elec_angle, elec_energy, _), _ = foil_mat[2](
        foil_depth=foil_depth,
        do_aperture=False,
        gamma_energy=gamma_energy,
        gamma_count=gamma_count,
        elec_count=elec_count,
    )

    plt.cla()
    plt.hist2d(elec_angle, elec_energy / MeV, bins=150)
    plt.title(
        f"Pre-Aperture Electron Angle and Energy \n [{material_name(foil_mat, capitalize=True)} {foil_depth/millimeter}mm]"
    )
    plt.xlabel("electron angle /rad")
    plt.ylabel("electron energy /MeV")
    plt.savefig(
        f"./scripts/fig_outputs/{material_name(foil_mat)}_{foil_depth/millimeter}mm_h2d.png"
    )


evals_pre_aperture_angle_energy_hist(
    mat_li,
    0.25 * millimeter,
    16 * MeV,
    elec_count=500_000,
)
evals_pre_aperture_angle_energy_hist(
    mat_si,
    0.25 * millimeter,
    16 * MeV,
    elec_count=500_000,
)
evals_pre_aperture_angle_energy_hist(
    mat_au,
    0.25 * millimeter,
    16 * MeV,
    elec_count=500_000,
)


# ---- post-aperture energy hist (16MeV; Li, Si, Au; d=0.25mm, cr=1.5cm, dl=25cm, ar=1.5cm)
def evals_post_aperture_energy_hist(
    foil_mat: Material,
    foil_depth: float,
    gamma_energy: float,
    gamma_count: int | None = None,
    elec_count: int | None = None,
):
    print(
        f"[post-aperture energy hist] ({material_name(foil_mat)} {foil_depth/millimeter}mm) "
    )
    (_, _, _, _, elec_energy, ids), _ = foil_mat[2](
        foil_depth=foil_depth,
        do_aperture=True,
        gamma_energy=gamma_energy,
        gamma_count=gamma_count,
        elec_count=elec_count,
    )

    plt.cla()

    plt.hist(
        [elec_energy[ids == 0] / MeV, elec_energy[ids == 1] / MeV],
        bins=int(16 / 0.150),
        stacked=True,
    )
    plt.title(
        f"Post-Aperture Electron Energy\n[{material_name(foil_mat, capitalize=True)} {foil_depth/millimeter}mm; foil r={R_FOIL/centimeter}cm, drift {L_DRIFT/centimeter}cm, aperture r={R_APERTURE/centimeter}cm]"
    )
    plt.legend(["incoherent", "pair-production"])
    plt.xlabel("electron energy /MeV")
    # plt.ylabel(f"electron counts ({ids.shape[0]} total)")
    plt.ylabel("relative electron counts")
    plt.gca().axes.yaxis.set_ticklabels([])
    plt.savefig(
        f"./scripts/fig_outputs/{material_name(foil_mat)}_{foil_depth/millimeter}mm_spectrum.png"
    )


evals_post_aperture_energy_hist(
    mat_li,
    0.25 * millimeter,
    16 * MeV,
    elec_count=500_000,
)
evals_post_aperture_energy_hist(
    mat_si,
    0.25 * millimeter,
    16 * MeV,
    elec_count=500_000,
)
evals_post_aperture_energy_hist(
    mat_au,
    0.25 * millimeter,
    16 * MeV,
    elec_count=500_000,
)


# ---- energy spectrum fwhm vs foil depth for incoherent (16Mev; [Li-Au])
# ---- counts/gamma vs foil depth for incoherent (16Mev; Si, [Li-Au])


def eval_performance_metrics(
    foil_mat: Material,
    foil_depth: float,
    gamma_energy: float,
    elec_count: int,
):

    print(
        f"[performance metrics array elt] ({material_name(foil_mat)} {foil_depth/millimeter}mm) "
    )
    (_, _, _, _, elec_energy, ids), total_efficiency = foil_mat[2](
        foil_depth=foil_depth,
        do_aperture=True,
        gamma_energy=gamma_energy,
        elec_count=elec_count,
    )

    foil_area = 3.14159 * (R_FOIL / 1e-2) ** 2
    gamma_per_cm2_MW = 1.5e3

    fwhm = full_width_half_max(elec_energy[ids == 0])
    compton_efficiency = np.mean(ids == 0) * total_efficiency

    # print(
    #     f"{material_name},{x_depth_mm},{elec_energy.size / N},{elec_energy[ids==0].size / N},{fwhm/MeV}"
    # )
    return (
        compton_efficiency,  # compton efficiency
        compton_efficiency * gamma_per_cm2_MW * foil_area,  # counts/s/MW
        np.mean(ids == 0) / np.mean(ids == 1),  # compton/pairprod snr
        fwhm,
    )


a0 = []
# depths = np.array([0.1, 1, 10]) * millimeter
# depths = np.array([0.1, 0.25, 0.5, 1, 2, 4, 10]) * millimeter
# depths = np.array([0.1, 0.2, 0.4, 0.6, 0.8, 1, 2, 3, 4, 5]) * millimeter
depths = (
    np.array([0.1, 0.2, 0.3, 0.4, 0.6, 0.8, 1, 1.5, 2, 3, 4, 5, 6, 8, 10]) * millimeter
)
# mats = [mat_si]
# mats = [mat_li, mat_si, mat_au]
mats = [mat_li, mat_be, mat_graphite, mat_si, mat_fe, mat_au]
for depth in depths:
    a1 = []
    a0.append(a1)
    for mat_ in mats:
        a1.append(
            eval_performance_metrics(
                foil_mat=mat_,
                foil_depth=depth,
                gamma_energy=16 * MeV,
                elec_count=100_000,
                # gamma_count=1000_000,
            )
        )
a0 = np.array(a0)


plt.cla()
plt.plot(depths / millimeter, a0[:, mats.index(mat_si), 0], ".r-")
plt.xlabel("foil depth /mm")
plt.ylabel("incoherent electron counts / gamma")
# plt.yscale("log")
plt.legend(
    [
        "Si [Z=16]",
    ]
)
plt.title(
    "Electron Counts per Gamma vs Foil Depth\n[incoherent scattering only, post-aperture, 16MeV gamma]"
)
plt.savefig("./scripts/fig_outputs/graph_efficiency_si.png")


plt.cla()
plt.plot(depths / millimeter, a0[:, 0, 0])
plt.plot(depths / millimeter, a0[:, 1, 0])
plt.plot(depths / millimeter, a0[:, 2, 0])
plt.plot(depths / millimeter, a0[:, 3, 0])
plt.plot(depths / millimeter, a0[:, 4, 0])
plt.plot(depths / millimeter, a0[:, 5, 0])
plt.xlabel("foil depth /mm")
plt.ylabel("incoherent electron counts / gamma")
# plt.yscale("log")
plt.legend(
    [
        "Li [Z=3]",
        "Be [Z=4]",
        "C [Z=6]",
        "Si [Z=16]",
        "Fe [Z=26]",
        "Au [Z=79]",
    ]
)
plt.title(
    "Electron Counts per Gamma vs Foil Depth\n[incoherent scattering only, post-aperture, 16MeV gamma]"
)
plt.savefig("./scripts/fig_outputs/graph_efficiencies.png")


plt.cla()
plt.plot(depths / millimeter, a0[:, 0, 3] / MeV)
plt.plot(depths / millimeter, a0[:, 1, 3] / MeV)
plt.plot(depths / millimeter, a0[:, 2, 3] / MeV)
plt.plot(depths / millimeter, a0[:, 3, 3] / MeV)
plt.plot(depths / millimeter, a0[:, 4, 3] / MeV)
plt.plot(depths / millimeter, a0[:, 5, 3] / MeV)
plt.xlabel("foil depth /mm")
plt.ylabel("incoherent electron counts / gamma")
# plt.yscale("log")
plt.legend(
    [
        "Li [Z=3]",
        "Be [Z=4]",
        "C [Z=6]",
        "Si [Z=16]",
        "Fe [Z=26]",
        "Au [Z=79]",
    ]
)
plt.ylim([0, 3])
plt.title(
    "Incoherent Electron Energy FWHM vs Foil Depth and Z \n [incoherent scattering only, post aperture, 16MeV gamma]"
)
plt.savefig("./scripts/fig_outputs/graph_fwhm_vs_depth.png")


# ---- location spectrum fwhm for incoherent (16MeV)


def evals_location_spectrum_fwhm(
    foil_mat: Material,
    foil_depth: float,
    gamma_energy: float,
    gamma_count: int | None = None,
    elec_count: int | None = None,
    NO_PAIR_PRODUCTION=True,
):
    print("[location spectrum fwhm for incoherent]")

    _, _, ray_map, _ = sess.config_order(1).exec()
    # _, _, ray_map, _ = sess.config_order(5).exec()

    rays_in_raw, _ = foil_mat[2](
        foil_depth=foil_depth,
        do_aperture=True,
        gamma_energy=gamma_energy,
        gamma_count=gamma_count,
        elec_count=elec_count,
    )
    rays_in = rays_into_relative(rays_in_raw, gamma_energy)
    if NO_PAIR_PRODUCTION:
        rays_in = rays_in[rays_in_raw[5] == 0]
    rays_out = []
    for x, a, y, b, e in rays_in:
        # (x,a,y,b,t,K) -> (x,a,y,b,t)
        x, a, y, b, _ = ray_map((x, a, y, b, 0, e))
        rays_out.append((x, a, y, b, e))
    x, a, y, b, e = np.transpose(rays_out)

    plt.cla()
    plt.hist(x, bins=200)
    plt.xlabel("dispersive direction /m")
    plt.title(
        f"Electron detection location by incident gamma ray energies.\n[incoherent only, $E_{{gamma}} = {gamma_energy/MeV}MeV$]"
    )
    plt.savefig("./scripts/fig_outputs/graph_fwhm_xpos.png")
    print(f"fwhm of location spectrum: {full_width_half_max(x,n_bins=200)}")

    plt.cla()
    plt.hist(y, bins=200)
    plt.savefig("./scripts/fig_outputs/graph_fwhm_ypos.png")


evals_location_spectrum_fwhm(
    foil_mat=mat_si,
    foil_depth=0.35 * millimeter,
    gamma_energy=16 * MeV,
    elec_count=100_000,
    # gamma_count=100_000_000,
)

# ---- detection position 2d scatter for various input energy around 16MeV


def evals_location_spectrum_2d(
    foil_mat: Material,
    foil_depth: float,
    gamma_energy: float,
    gamma_count: int | None = None,
    elec_count: int | None = None,
    NO_PAIR_PRODUCTION=True,
):
    print("[detection position 2d scatter for various input energy around 16MeV]")

    _, _, ray_map, _ = sess.config_order(1).exec()
    # _, _, ray_map, _ = sess.config_order(5).exec()

    def plot_monte_carlo(d_e: float, color):
        rays_in_raw, _ = foil_mat[2](
            foil_depth=foil_depth,
            do_aperture=True,
            gamma_energy=gamma_energy * (1 + d_e),
            gamma_count=gamma_count,
            elec_count=elec_count,
        )
        rays_in = rays_into_relative(rays_in_raw, gamma_energy)
        if NO_PAIR_PRODUCTION:
            rays_in = rays_in[rays_in_raw[5] == 0]
        rays_out = []
        for x, a, y, b, e in rays_in:
            # (x,a,y,b,t,K) -> (x,a,y,b,t)
            x, a, y, b, _ = ray_map((x, a, y, b, 0, e))
            rays_out.append((x, a, y, b, e))
        x, a, y, b, e = np.transpose(rays_out)
        plt.scatter(x, y, c=color, s=1)
        # plt.hist(x,bins=100,stacked=True)

    NO_PAIR_PRODUCTION = True

    plt.cla()
    plot_monte_carlo(0.1, "b")
    plot_monte_carlo(0.05, "c")
    plot_monte_carlo(0, "lime")
    plot_monte_carlo(-0.05, "y")
    plot_monte_carlo(-0.1, "r")
    plt.xlabel("dispersive direction /m")
    plt.ylabel("vertical direction /m")
    plt.legend(
        [
            "+0.10",
            "+0.05",
            "$\\delta E$ = 0.00",
            "-0.05",
            "-0.10",
        ]
    )
    plt.title(
        "Electron detection location by incident gamma ray energies."
        "\n[incoherent only, $E_{gamma} = 16MeV(1+\\delta E)]$ "
    )
    plt.savefig("./scripts/fig_outputs/graph_rainbow.png")


evals_location_spectrum_2d(
    foil_mat=mat_si,
    foil_depth=0.5 * millimeter,
    gamma_energy=16 * MeV,
    elec_count=10_000,
)

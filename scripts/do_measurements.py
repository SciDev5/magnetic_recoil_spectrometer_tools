import numpy as np
import matplotlib.pyplot as plt
from cross_sections import (
    gen_cross_section_compton,
    gen_cross_section_pairproduction,
    load_pairproduction_cross_section,
)
from acceptance import SRXMData, foil_trace, Foil, aperture
from physical_constants import mol, MeV, millimeter


DO_PLOTS = input("do plots? [y/N]: ").lower().strip() in ["y", "yes", "true"]


###################
def run_sim():
    x_number_density = x_density * 1e6 / x_atomic_weight * mol  # [/m^3]

    with open(f"./_data/estar_{material_name}.txt", "r", encoding="utf8") as srem:
        x_srem = np.array(
            [[float(y) for y in x.split(" ")[:2]] for x in srem.readlines()[9:]]
        )
        x_srem[:, 0] *= MeV  #  MeV  ->  J
        x_srem[:, 1] *= x_density * 100 * MeV  #  MeV cm^2 / g  ->  J / m
        x_srem: SRXMData = x_srem[:, 0], x_srem[:, 1]

    x_crosssection_compton = gen_cross_section_compton(
        x_number_density * Z,
        np.linspace(15, 17, 3) * MeV,
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

    elec_angle, elec_energy = foil_trace(
        n_rays_incident=N,
        n_srxm_steps=100,
        phot_energy_in=16 * MeV,
        foil_properties=x_foil,
        foil_depth=x_depth_mm * millimeter,
    )

    print(f"------ {material_name} {x_depth_mm}mm ------")
    print(f"pre-aperture efficiency: {elec_energy.size / N}")

    foil_area = 3.14159 * (R_FOIL / 1e-2) ** 2
    gamma_per_cm2_MW = 1.5e3

    t, p, x, y, en = aperture(
        (elec_angle, elec_energy), R_FOIL, R_APERTURE, DIST_APERTURE
    )
    print(f"post-aperture efficiency: {en.size / N}  [1/{N / en.size }]")
    print(f"electrons per megawatt: {(en.size / N) * gamma_per_cm2_MW * foil_area}")
    print(f"electrons @140MW: {(en.size / N) * gamma_per_cm2_MW * foil_area * 140}")

    if DO_PLOTS:
        plt.hist2d(elec_angle, elec_energy / MeV, bins=150)
        plt.title(f"{material_name[0].upper()}{material_name[1:]} {x_depth_mm}mm")
        plt.xlabel("electron angle /rad")
        plt.ylabel("electron energy /MeV")
        plt.show()
        plt.hist(en / MeV, bins=250)
        plt.title(
            f"Energy spectrum of electrons exiting aperature {material_name[0].upper()}{material_name[1:]} {x_depth_mm}mm"
        )
        plt.xlabel("electron energy /MeV")
        plt.ylabel("counts")
        plt.show()


# R_FOIL = 0.02
# R_APERTURE = 0.02
# DIST_APERTURE = 0.25
# x_depth_mm = 1.0e-1
# N_BY_Z = 1_000_000_000


# Z = 3
# N = N_BY_Z / Z
# x_density = 0.5334  # [g/cm^3]
# x_atomic_weight = 6.94  # [amu | g/mol]
# material_name = "li"
# run_sim()

# Z = 4
# N = N_BY_Z / Z
# x_density = 1.845  # [g/cm^3]
# x_atomic_weight = 9.0122  # [amu | g/mol]
# material_name = "be"
# run_sim()

# Z = 6
# N = N_BY_Z / Z
# x_density = 2.18  # [g/cm^3]
# x_atomic_weight = 12.011  # [amu | g/mol]
# material_name = "graphite"
# run_sim()

# Z = 16
# N = N_BY_Z / Z
# x_density = 2.329085  # [g/cm^3]
# x_atomic_weight = 28.085  # [amu | g/mol]
# material_name = "si"
# run_sim()

# Z = 26
# N = N_BY_Z / Z
# x_density = 7.874  # [g/cm^3]
# x_atomic_weight = 55.845  # [amu | g/mol]
# material_name = "fe"
# run_sim()

# Z = 79
# N = N_BY_Z / Z
# x_density = 19.283  # [g/cm^3]
# x_atomic_weight = 196.966570  # [amu | g/mol]
# material_name = "au"
# run_sim()

R_FOIL = 0.02
R_APERTURE = 0.03
DIST_APERTURE = 0.25
x_depth_mm = 0.25
N_BY_Z = 1_000_000_000


Z = 16
N = N_BY_Z / Z
x_density = 2.329085  # [g/cm^3]
x_atomic_weight = 28.085  # [amu | g/mol]
material_name = "si"
run_sim()

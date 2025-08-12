"""
Contains implementations for the total and differential
cross sections for all relevant processes.
"""

import numpy as np
import numpy.typing as npt

from acceptance import (
    CrossSection,
    CrossSectionTableTotal,
    AngleEnergy,
)
from util import (
    d_,
    dsolid_dangle,
    dsolid_dspherical,
    prep_distr_1_1,
    sample_distr_1_1,
    prep_distr_1_2,
    sample_distr_1_2,
    begin_timer,
)
from physical_constants import (
    elec_rest_energy,
    classical_electron_radius,
    speed_of_light,
    alpha,
)

########################################################################
# ==== INCOHERENT(COMPTON) SCATTERING [gamma -> electron + gamma]


def gen_cross_section_compton(
    electron_number_density: float,  # electrons/m^3
    phot_energy_range: npt.NDArray,  # :: [energy/J; n]
    phot_out_angle_range: npt.NDArray,  # :: [angle/rad; m]
    override_total_csd: CrossSectionTableTotal | None = None,
) -> CrossSection:
    """
    Generate the cross-section data for the compton effect.

    Requires:
    - `phot_energy_range` linearly spaced range of values
    for the input photons to be considered.
    - `phot_out_angle_range` linearly spaced valid output angles
    for the electron to exit at.

    The atom number density of the material in question, and optionally an
    array which overrides the total cross section.
    """

    # /m
    def csd_total(energy_in: npt.NDArray) -> npt.NDArray:
        s0 = 6.651e-29  # m2
        a0 = energy_in / elec_rest_energy
        return s0 * 3 / 8 / a0 * (np.log(2 * a0) + 0.5) * electron_number_density

    def csd_total_override(energy_in: npt.NDArray):
        return np.interp(energy_in, override_total_csd[0], override_total_csd[1])

    # /m /st
    def csd_differential(
        energy_in: npt.NDArray, phot_angle_out: npt.NDArray
    ) -> npt.NDArray:
        alpha_0 = energy_in / elec_rest_energy
        cos_theta = np.cos(phot_angle_out)
        return (
            classical_electron_radius**2
            * (1 + cos_theta**2)
            / (1 + alpha_0 * (1 - cos_theta)) ** 2
            * (
                1
                + (alpha_0**2 * (1 - cos_theta) ** 2)
                / ((1 + cos_theta**2) * (1 + alpha_0 * (1 - cos_theta)))
            )
        ) * electron_number_density

    def compton_elec_params(
        phot_energy_in: npt.NDArray, phot_angle_out: npt.NDArray
    ) -> tuple[
        npt.NDArray,  # elec_energy_out
        npt.NDArray,  # elec_angle_out
    ]:
        a_0 = phot_energy_in / elec_rest_energy
        a = a_0 / (1 + a_0 * (1 - np.cos(phot_angle_out)))
        elec_energy_out = (a_0 - a) * elec_rest_energy
        elec_angle_out = np.arctan(
            1 / ((a_0 + 1) * np.tan(phot_angle_out / 2))
        )  # * np.sqrt((2 * a_0 * a) / (a_0 - a) - 1)
        return elec_angle_out, elec_energy_out

    phot_scattering_angle_distr = prep_distr_1_1(
        csd_differential(
            np.expand_dims(phot_energy_range, axis=1).repeat(
                phot_out_angle_range.size, axis=1
            ),
            np.expand_dims(phot_out_angle_range, axis=0).repeat(
                phot_energy_range.size, axis=0
            ),
        ),
        phot_energy_range,
        phot_out_angle_range,
    )

    def gen_rays(count: int, phot_energy_in: float) -> AngleEnergy:
        phot_angle_out = sample_distr_1_1(
            count, phot_scattering_angle_distr, phot_energy_in
        )
        # VB.append(phot_angle_out)
        # VC.append(compton_elec_params(phot_energy_in, phot_angle_out)[0])
        return compton_elec_params(phot_energy_in, phot_angle_out)

    return (csd_total if override_total_csd is None else csd_total_override), gen_rays


########################################################################
# ==== PAIR PRODUCTION [gamma -> electron + positron]


PairProductionRawCrossSection = tuple[
    npt.NDArray, tuple[npt.NDArray, npt.NDArray, npt.NDArray]
]


def calculate_pairproduction_cross_section(
    energy_gamma: npt.NDArray,  # [J]
    energy_elec_frac: npt.NDArray,  # [1]
    angle_posi: npt.NDArray,  # [rad]
    angle_elec: npt.NDArray,  # [rad]
    angle_inter: npt.NDArray,  # [rad]
) -> PairProductionRawCrossSection:
    """
    Calculates the raw precursor to the pair production cross-section data
    (free of Z dependence), requires evenly spaced arrays to configure what
    areas of parameter space to consider:
    - `energy_gamma`: gamma input energy
    - `energy_elec_frac`: fraction of kinetic energy given to the electron or positron
    - `angle_posi`: off-beamline angle of positron
    - `angle_elec`: off-beamline angle of electron
    - `angle_inter`: angle between the planes formed by the beamline and the electron
    and positron momenta.
    ).
    """

    def genarg(arg: npt.NDArray, axis: int) -> npt.NDArray:
        axes = {0, 1} - {axis}
        arg = np.expand_dims(arg, axis=tuple(axes))
        for axis_i, size in [
            (i, x.size) for i, x in enumerate([angle_posi, angle_inter]) if i in axes
        ]:
            arg = arg.repeat(size, axis=axis_i)
        return arg

    d_energy_elec_frac = d_(energy_elec_frac)
    d_angle_posi = d_(angle_posi)
    d_angle_elec = d_(angle_elec)
    d_angle_inter = d_(angle_inter)

    d_sigma_for_elec = np.zeros(
        (energy_gamma.size, energy_elec_frac.size, angle_elec.size)
    )

    tick_timer = begin_timer("gen pair-production cross-section")

    n_done = 0
    for i_energy_gamma, i_energy_elec_frac, i_angle_elec in (
        (i_energy_gamma, i_energy_elec_frac, i_angle_elec)
        for i_energy_gamma in range(energy_gamma.size)
        for i_energy_elec_frac in range(energy_elec_frac.size)
        for i_angle_elec in range(angle_elec.size)
    ):

        energy_gamma_ = energy_gamma[i_energy_gamma]
        energy_elec_ = (
            energy_elec_frac[i_energy_elec_frac]
            * (energy_gamma_ - (elec_rest_energy * 1.001) * 2)
            + elec_rest_energy * 1.0001
        )
        angle_posi_ = genarg(angle_posi, 0)
        angle_elec_ = angle_elec[i_angle_elec]
        angle_inter_ = genarg(angle_inter, 1)

        d_energy_elec = d_energy_elec_frac * (
            energy_gamma_ - (elec_rest_energy * 1.001) * 2
        )

        # E^2 = m^2 c^4 + p^2 c^2
        # p^2 = (E^2 - E_rest^2)/c^2
        Ey = energy_gamma_
        En = energy_elec_
        Ep = energy_gamma_ - energy_elec_
        pn2 = ((En**2) - (elec_rest_energy**2)) / (speed_of_light**2)
        pp2 = ((Ep**2) - (elec_rest_energy**2)) / (speed_of_light**2)
        pn = np.sqrt(pn2)
        pp = np.sqrt(pp2)

        # k = Ey/c
        k = energy_gamma_ / speed_of_light

        # pnx = 0, ppx = -qx
        # pny + ppy + qy = 0
        # pnz + ppz + qz = k

        # pnx = 0
        # pny = |pn| sin(tn)
        # pnz = |pn| cos(tn)
        # ppx = |pp| sin(tp) sin(phi)
        # ppy = |pp| sin(tp) cos(phi)
        # ppz = |pn| cos(tp)

        # p^2 =         pny^2 + pnz^2
        # p^2 = ppx^2 + ppy^2 + ppz^2

        pny = pn * np.sin(angle_elec_)
        pnR = pny
        pnz = pn * np.cos(angle_elec_)
        ppR = pp * np.sin(angle_posi_)
        ppx = pp * np.sin(angle_posi_) * np.sin(angle_inter_)
        ppy = pp * np.sin(angle_posi_) * np.cos(angle_inter_)
        ppz = pp * np.cos(angle_posi_)
        qx = -ppx
        qy = -pny - ppy
        qz = k - pnz - ppz
        q2 = qx**2 + qy**2 + qz**2
        q4 = q2**2

        c = speed_of_light
        c2 = c * c

        d_sigma = (
            (alpha * classical_electron_radius**2 * elec_rest_energy**2)
            / (2 * np.pi) ** 2
            * ((pp * pn) / (q4 * Ey**3))
            * (
                d_energy_elec
                * (dsolid_dangle(angle_elec_) * d_angle_elec)
                * (dsolid_dspherical(angle_posi_) * d_angle_posi * d_angle_inter)
            )
        ) * (
            -((pnR / (En - pnz * c)) ** 2) * (4 * Ep**2 - q2 * c2)
            - ((ppR / (Ep - ppz * c)) ** 2) * (4 * En**2 - q2 * c2)
            + 2
            / ((Ep - ppz * c) * (En - pnz * c))
            * (
                (ppR**2 + pnR**2) * (Ey**2)
                + (ppy * pny) * (2 * Ep**2 + 2 * En**2 - q2 * c2)
            )
        )
        d_sigma[Ey - Ep - elec_rest_energy < 0] = 0
        d_sigma[Ey - En - elec_rest_energy < 0] = 0
        d_sigma[np.isnan(d_sigma)] = 0  #!!!!

        d_sigma_for_elec[i_energy_gamma, i_energy_elec_frac, i_angle_elec] = np.sum(
            d_sigma
        )

        # debug estimated time of completion printing:
        n_done += 1
        tick_timer(n_done / d_sigma_for_elec.size)
    tick_timer(1)

    # /m
    return d_sigma_for_elec, (
        energy_gamma,
        energy_elec_frac,
        angle_elec,
    )


def save_pairproduction_cross_section(
    filename: str, cross_section: PairProductionRawCrossSection
):
    """
    Saves the raw pair-production cross section to a file.
    """
    d_sigma, (
        energy_gamma,
        energy_elec_frac,
        angle_elec,
    ) = cross_section
    np.savez(
        filename,
        d_sigma=d_sigma,
        energy_gamma=energy_gamma,
        energy_elec_frac=energy_elec_frac,
        angle_elec=angle_elec,
    )


def load_pairproduction_cross_section(
    filename: str,
) -> PairProductionRawCrossSection:
    """
    Read the raw pair-production cross section from a file.
    """
    x = np.load(filename)

    d_sigma = x["d_sigma"]
    energy_gamma = x["energy_gamma"]
    energy_elec_frac = x["energy_elec_frac"]
    angle_elec = x["angle_elec"]
    return d_sigma, (
        energy_gamma,
        energy_elec_frac,
        angle_elec,
    )


def gen_cross_section_pairproduction(
    Z: int,
    atom_number_density: float,
    cross_section: PairProductionRawCrossSection,
    override_total_csd: CrossSectionTableTotal | None = None,
):
    """
    Generate the pair-production cross section from the raw Z-independent raw cross section.

    Requires Z (atomic number), of the material in question, the atom number density,
    and optionally an array which overrides the total cross section.
    """
    d_sigma, (
        energy_gamma,
        energy_elec_frac,
        angle_elec,
    ) = cross_section
    d_sigma *= Z**2

    sigma_total_for_gamma_in = d_sigma.sum(axis=(1, 2))

    def csd_total_numeric(energy_in: npt.NDArray) -> npt.NDArray:
        return (
            np.interp(energy_in, energy_gamma, sigma_total_for_gamma_in)
            * atom_number_density
        )

    def csd_total_override(energy_in: npt.NDArray):
        return np.interp(energy_in, override_total_csd[0], override_total_csd[1])

    raw_distr = prep_distr_1_2(d_sigma, energy_gamma, energy_elec_frac, angle_elec)

    def gen_rays(count: int, phot_energy_in: float) -> AngleEnergy:
        elec_energy_frac_out, elec_angle_out = sample_distr_1_2(
            count, raw_distr, phot_energy_in
        )
        elec_energy_out = (phot_energy_in - elec_rest_energy * 2) * elec_energy_frac_out
        return elec_angle_out, elec_energy_out

    return (
        csd_total_numeric if override_total_csd is None else csd_total_override
    ), gen_rays

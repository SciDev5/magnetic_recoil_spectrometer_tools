import numpy as np

from physical_constants import MeV
import cross_sections


xs = cross_sections.calculate_pairproduction_cross_section(
    np.linspace(1, 20, 50) * MeV,
    np.linspace(0, 1, 200),
    np.linspace(0, np.pi / 2, 500),
    np.linspace(0, np.pi / 2, 500),
    np.linspace(0, np.pi * 2, 100),
)
cross_sections.save_pairproduction_cross_section("./_data/pairprod_xsctn_medium", xs)


# xs = cross_sections.calculate_pairproduction_cross_section(
#     np.linspace(1, 20, 100) * MeV,
#     np.linspace(0, 1, 500),
#     np.linspace(0, np.pi / 2, 500),
#     np.linspace(0, np.pi / 2, 500),
#     np.linspace(0, np.pi * 2, 250),
# )
# cross_sections.save_pairproduction_cross_section("./_data/pairprod_xsctn_large", xs)

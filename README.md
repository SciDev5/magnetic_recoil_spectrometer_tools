# Magnetic Recoil Spectrometer Tools (MRS Tools)


## Project Setup

0. Dependencies:
    - COSY Infinity (https://www.bmtdynamics.org/cosy/)
    - Python 3.10.12 (recommended, other Python 3 versions may work)
    - Developed in a linux environment, may require minor tweaks to `/cosy/__init__.py` to function on other platforms.

1. Setup COSY INFINITY
    Place the COSY executable, `COSY.bin`, and `COSYGUI.jar` in the `cosy/eval/` directory.

## Documentation

### Walkthroughs
- `/docs/walkthrough.ipynb`: Contains a complete walkthrough of the software, following implementation of a hypothetical electron spectrometer.
- `/docs/creating_new_cross_section.ipynb`: Contains explanation and examples for adding new cross sections to the monte-carlo sim.

### In-line code documentation:
- All important python functions have python docstrings.
- Many of the more useful COSYScript functions have documentation in `/cosy/utils.fox`.


## `/scripts/*`

There are several scripts (which are called from the project root directory as `python -m scripts.<name of script>`):

- `gen_pairprod_crosssection`: Generate / integrate and save the raw cross-section data for pair-production (note: very long run time).
- `do_measurements`: Run the monte carlo sim several times and output figures.

## Implementation Notes

### Units

Almost all units are SI. Values with non-standard units should be tagged in their name with the units they are in (ex. `depth_mm` for depth in millimeters, `energy_MeV` for energy in MeV, etc.). To convert into the system units, import the unit constant from `physical_constants.py` and multiply it by the unit (ex. `my_energy = 12.34 * MeV`). To convert back to other units, divide out the unit constant (ex. `print(f"My Energy: {my_energy / MeV}MeV`, prints "My Energy: 12.34MeV").
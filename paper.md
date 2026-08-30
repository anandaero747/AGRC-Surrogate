---
title: 'AGRC-Surrogate: A Neural Network-Based C81 Aerodynamic Table Generator for Rotorcraft Airfoil Analysis and Design'
tags:
  - Python
  - aerodynamics
  - rotorcraft
  - machine learning
  - surrogate modeling
  - airfoil design
  - C81 tables
authors:
  - name: Apurva Anand
    orcid: 0000-0003-3746-0249
    affiliation: 1
affiliations:
  - name: Alfred Gessow Rotorcraft Center, Department of Aerospace Engineering, University of Maryland, College Park, MD, USA
    index: 1
date: 16 August 2026
bibliography: paper.bib
---

# Summary

`AGRC-Surrogate` is an open-source Python package that generates C81 aerodynamic coefficient tables for arbitrary airfoil geometries using a set of pre-trained neural network surrogate models. C81 tables encode lift ($C_l$), drag ($C_d$), and pitching moment ($C_m$) coefficients as a function of angle of attack (AoA) and Mach number and are the standard input format for comprehensive rotorcraft analysis codes such as CAMRAD II and FLIGHTLAB [@johnson1994rotorcraft; @ormiston2004rotorcraft]. Generating these tables with traditional RANS CFD solvers requires hours of compute time per airfoil; `AGRC-Surrogate` replaces this process with a sub-second inference pipeline.

The package accepts airfoil geometry in standard two-column coordinate format (`.dat`), parameterizes it using Chebyshev-based Class-Shape Transformation (CST) coefficients [@kulfan2008universal], and feeds a 20-dimensional feature vector into six independent TensorFlow/Keras models—one for each Mach number from 0.3 to 0.8 in 0.1 increments [@tensorflow2015]. Each model predicts PCA-reduced aerodynamic coefficient vectors that are inverse-transformed into full AoA sweeps spanning $-180°$ to $+180°$ at $1°$ resolution. Neural network predictions are blended with baseline NACA airfoil data at extreme angles of attack using cosine-weighted smoothing windows, ensuring physically consistent behavior across the full $360°$ range. An optional genetic algorithm optimization module, built on the DEAP framework [@fortin2012deap], enables inverse design workflows that search for airfoil geometries meeting user-specified aerodynamic targets.

# Statement of Need

The design and analysis of helicopter rotors requires aerodynamic performance data for candidate airfoil sections across a wide range of Mach numbers representative of the advancing and retreating blade environment. The conventional workflow involves running high-fidelity RANS CFD computations for each candidate airfoil to populate C81 tables, which are then consumed by comprehensive rotorcraft codes to simulate the full rotor system. This process is computationally expensive and creates a bottleneck in both analysis and optimization loops: a single RANS run for one Mach point can take from minutes to hours depending on mesh resolution and solver convergence, making population-based optimization methods and large parametric sweeps impractical.

Existing alternatives are insufficient for the subsonic compressible regime relevant to rotorcraft blades. Panel-method codes such as XFOIL are restricted to incompressible or low-Mach flows and cannot capture transonic effects that become significant on advancing blades [@drela1989xfoil]. Look-up-table catalogs of tested airfoil sections (e.g., the NACA 4- and 5-digit series) are limited to a small discrete set of geometries and cannot generalize to novel sections. While several machine learning approaches have been demonstrated for airfoil aerodynamic prediction, most target two-dimensional lift and drag at fixed conditions and do not produce the full multi-Mach C81 table structure required by rotorcraft codes [@bouhlel2020airfoil; @li2020machine].

`AGRC-Surrogate` addresses this gap by providing a trained, installable software tool that maps arbitrary airfoil geometry directly to ready-to-use C81 tables in sub-second inference time. It is designed for integration into rotorcraft design workflows: the package exposes a Python API (in addition to a CLI) that enables other codes to call C81 table generation programmatically, and the optional DEAP-based optimization module demonstrates its use in a genetic algorithm inverse design loop. The underlying models were trained on high-fidelity RANS data generated with the HAM2D structured CFD solver across a family of airfoil geometries representative of modern rotorcraft blade sections [@anand2026joa].

# Methodology

## Airfoil Parameterization

Airfoil geometry is represented using Chebyshev-based CST coefficients [@kulfan2008universal]. The upper and lower surfaces are each described by 10 coefficients extracted via least-squares fitting, yielding a compact 20-dimensional design vector. This parameterization smoothly captures camber, thickness, leading-edge radius, and trailing-edge angle variations, and serves as the input to all neural network models. The package handles arbitrary coordinate conventions—both clockwise and counterclockwise orderings—by automatically detecting and pre-processing the input orientation.

## Surrogate Model Architecture

Six independent neural network models are trained—one per Mach number (0.3, 0.4, 0.5, 0.6, 0.7, 0.8)—using high-fidelity RANS data from the HAM2D CFD solver [@anand2026joa]. Each model takes the 20-dimensional CST feature vector (scaled by a fitted StandardScaler) as input and outputs a PCA-reduced representation of the aerodynamic coefficient distributions. Separate PCA transforms and inverse scalers reconstruct $C_l$ (10 PCA components), $C_d$ (18 components, with exponential back-transform to enforce positivity), and $C_m$ (8 components) over the $-10°$ to $+20°$ AoA range where the models were validated.

## Physics-Consistent Extension to ±180°

Rotorcraft comprehensive codes require aerodynamic data over the full $360°$ AoA range to model retreating blade stall, autorotation, and vortex ring states. Neural network predictions are defined only within the validated AoA range. Beyond this range, the package blends smoothly to baseline NACA aerodynamic data using cosine-weighted transition windows over $10°$ blend zones. This ensures the output C81 tables are physically consistent and immediately usable by analysis codes without manual post-processing.

## Optimization Interface

The `opt_api` module exposes functions for extracting CST coefficients from an airfoil file, predicting C81 tables from a CST vector, and evaluating scalar aerodynamic objectives (e.g., lift-to-drag ratio at a target AoA and Mach). These are designed as objective function callbacks compatible with standard optimization frameworks. An example genetic algorithm workflow using DEAP is provided in the `examples/` directory, demonstrating population-based airfoil optimization with the surrogate model as the fitness evaluator.

# Installation and Usage

`AGRC-Surrogate` is installable from source via pip:

```bash
pip install -e .                          # core package
pip install -e ".[opt]"                   # with DEAP optimization support
```

Generating a C81 table from an airfoil coordinate file requires a single command:

```bash
agrc-c81 --airfoil my_airfoil.dat
```

This produces `C81_all_mach.dat`, a formatted table ready for use in CAMRAD II, FLIGHTLAB, or similar rotorcraft analysis codes. The Python API can be called programmatically:

```python
from agrc_surrogate.opt_api import cst_from_airfoil, c81_from_cst

cst = cst_from_airfoil("my_airfoil.dat")
tables = c81_from_cst(cst, write_output=True)
```

# Acknowledgements

The author acknowledges the Alfred Gessow Rotorcraft Center at the University of Maryland for supporting this research. The high-fidelity CFD training data was generated using the HAM2D RANS solver. The author declares that no AI-assisted tools were used in the development of this software or the preparation of this manuscript.

# References

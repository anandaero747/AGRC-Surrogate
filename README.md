AGRC Surrogate is a Neural Networks based surrogate model trained on high-fidelity CFD (RANS) data.
The model works from Mach 0.3 to 0.8 and generates a C81 table for any given airfoil geometry.
The airfoil coordinates should be in a .dat format with 2 columns (x and y).
The coordinates can be from trailing edge lower surfface through Leading edge to trailing edge
upper surface in a closckwise direction. OR
It can be in the opposite direction. The code is robust enough the detect the direction of the coordinates
and pre-process accordingly.
Currently, the model gives C81 table as outut. 
This can be directly used with comprehensive rotorcraft codes that needs airfoil table as inputs.
If you want to use this model for airfoil optimization, airfoil CST coefficients can be used as design variables.
This code also computes CST coefficients, but doesn't display it right now.
Feel free to contact the author "apurva01@umd.edu" to know about the application of thi model in airfoil
optimization.


## Installation

```bash
git clone https://github.com/anandaero747/AGRC-Surrogate.git
cd AGRC-Surrogate
pip install -e .

## Usage
agrc-c81 --airfoil your_airfoil.dat

This generates C81_all_mach.dat file



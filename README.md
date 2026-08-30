AGRC Surrogate is a Neural Networks based surrogate model trained on high-fidelity CFD (RANS) data.
The model works from Mach 0.3 to 0.8 and generates a C81 table for any given airfoil geometry.
The airfoil coordinates should be in a .dat format with 2 columns (x and y).
The coordinates can be from trailing edge lower surface through Leading edge to trailing edge
upper surface in a clockwise direction. OR
It can be in the opposite direction. The code is robust enough the detect the direction of the coordinates
and pre-process accordingly.
Currently, the model gives C81 table as output. 
This can be directly used with comprehensive rotorcraft codes that needs airfoil table as inputs.
If you want to use this model for airfoil optimization, airfoil CST coefficients can be used as design variables.
This code also computes CST coefficients, but doesn't display it right now.
Feel free to contact the author "apurva01@umd.edu" to know about the application of this model in airfoil
optimization.

An added functionality is also included in the package for airfoil optimization. This package uses airfoil C81 table and CST coefficients
with DEAP genetic algorithm (GA) library to optimize airfoil at any mach number and angle of attack. Currently there are no thickness constraints included in the optimization package, but it should be easy to modify. Feel free to contact anandaero747@gmail.com to get optimization code with thickness constraint.


## Installation and Environment Recommendation 

```bash

Recommended: Create an environment to use the package. "python -m venv agrc-env", "source agrc-env/bin/activate"

git clone https://github.com/anandaero747/AGRC-Surrogate.git
cd AGRC-Surrogate
pip install -e .

## Usage
agrc-c81 --airfoil your_airfoil.dat

This generates C81_all_mach.dat file


## Optimization (GA via DEAP)

Install with optimization extras:

pip install "agrc-surrogate[opt]"

Run example GA optimization:

python examples/ga_optimize_sc1095.py --airfoil path/to/airfoil.dat --aoa 2.0 --mach 0.3 --pop 80 --ngen 30

## Testing

Install the test dependencies and run the test suite:

```bash
pip install -e ".[dev]"
pytest tests/ -v
```

To skip the slow integration tests that load TensorFlow models:

```bash
pytest tests/ -v -m "not slow"
```

If you find the package useful, please cite our work

@article{anand2026generalizable,
  title={Generalizable deep learning module for rotorcraft inverse design applications},
  author={Anand, Apurva and Marepally, Koushik and Safdar, M Muneeb and Lee, Bumseok and Baeder, James D},
  journal={Journal of Aircraft},
  pages={1--15},
  year={2026},
  publisher={American Institute of Aeronautics and Astronautics}
}



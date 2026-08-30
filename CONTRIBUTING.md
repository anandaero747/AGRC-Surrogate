# Contributing to AGRC-Surrogate

Thank you for your interest in contributing. Contributions are welcome in the form of bug reports, feature requests, and pull requests.

## Reporting Issues

Please open an issue on the [GitHub issue tracker](https://github.com/anandaero747/AGRC-Surrogate/issues) and include:
- A clear description of the problem
- Steps to reproduce the issue
- Your Python version and operating system
- The airfoil file (if relevant), or a minimal example that reproduces the problem

## Submitting Changes

1. Fork the repository and create a branch from `main`.
2. Install the package in development mode with test dependencies:
   ```bash
   pip install -e ".[dev,opt]"
   ```
3. Make your changes and add tests where appropriate.
4. Run the test suite to confirm nothing is broken:
   ```bash
   pytest tests/ -v
   ```
5. Open a pull request with a clear description of what the change does and why.

## Scope

This package is primarily focused on:
- C81 table generation for rotorcraft airfoil sections in the Mach 0.3–0.8 range
- CST-based airfoil parameterization
- Integration with DEAP-based optimization workflows

Contributions that extend the Mach range, add new airfoil parameterizations, or improve the optimization interface are especially welcome.

## Contact

For questions not suited to a GitHub issue, contact the author at anandaero747@gmail.com.

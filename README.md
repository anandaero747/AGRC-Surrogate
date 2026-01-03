# AGRC Surrogate – C81 Generator

Streamlit web app to generate full 360° C81 airfoil tables using
AGRC-trained ANN surrogate models.

## Features
- Upload upper & lower airfoil geometry
- CST parametrization
- ANN-based Cl / Cd / Cm prediction
- 360° cosine-blended C81 tables
- Download C81_all_mach.dat

## Run locally
```bash
streamlit run streamlit_c81_app.py


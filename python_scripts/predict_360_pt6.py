#!/usr/bin/env python3
"""
predict_airfoil_cheb.py
-----------------------
Parametrize airfoil surfaces (upper/lower .dat) using Chebyshev-based CST,
scale inputs, predict Cl/Cd/Cm using pretrained NN model (m1.h5),
and inverse transform results.

Usage:
    python3 predict_airfoil_cheb.py --upper upper.dat --lower lower.dat
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
import tensorflow as tf
import argparse
from scipy.optimize import curve_fit
from numpy.polynomial.chebyshev import chebvander
import os, sys

# ----------------------------
# Helper functions
# ----------------------------

def read_airfoil_surface(file_path):
    """Read .dat file with two columns: x, y"""
    x, y = [], []
    with open(file_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 2:
                try:
                    x.append(float(parts[0]))
                    y.append(float(parts[1]))
                except ValueError:
                    continue
    return np.array(x), np.array(y)

def class_func(x):
    """CST class function"""
    return np.sqrt(x) * (1 - x)

def CST_chebyshev_TE(x, *a):
    """CST parameterization using Chebyshev polynomials with TE offset"""
    *cheb_coeffs, te_offset = a
    xi = 2 * x - 1  # map to [-1,1]
    T = chebvander(xi, len(cheb_coeffs)-1)
    shape = T @ np.array(cheb_coeffs)
    y = class_func(x) * shape
    return y + te_offset * x
    
    
def read_airfoil_dat(filepath):
    data = np.genfromtxt(filepath, filling_values=np.nan)
    data = data[~np.isnan(data).all(axis=1)]

    if data.ndim == 1:
        data = data.reshape(1, -1)

    n_cols = data.shape[1]
    colnames = ['AoA'] + [f"Mach_{round(0.1*i + 0.1, 1)}" for i in range(1, n_cols)]

    df = pd.DataFrame(data, columns=colnames)
    df = df.sort_values(by='AoA').reset_index(drop=True)
    return df

# ----------------------------
# Main
# ----------------------------

def main():
    parser = argparse.ArgumentParser(description="Airfoil CST + NN prediction")
    parser.add_argument("--upper", required=True, help="Path to upper surface .dat")
    parser.add_argument("--lower", required=True, help="Path to lower surface .dat")
    parser.add_argument("--model", default="Forward_pt6_non_opt_hyp_v2.h5", help="Path to trained model (.h5)")
    parser.add_argument("--scalerx", default="scaler_cst_v2.pkl", help="Path to input scaler")
    parser.add_argument("--pca_cl", default="pca_cl_pt6_v2.pkl", help="PCA model for Cl")
    parser.add_argument("--pca_cd", default="pca_cd_pt6_v2.pkl", help="PCA model for Cd")
    parser.add_argument("--pca_cm", default="pca_cm_pt6_v2.pkl", help="PCA model for Cm")
    parser.add_argument("--scaler_cl", default="scaler_cl_pt6_v2.pkl", help="Scaler for Cl")
    parser.add_argument("--scaler_cd", default="scaler_cd_pt6_v2.pkl", help="Scaler for Cd")
    parser.add_argument("--scaler_cm", default="scaler_cm_pt6_v2.pkl", help="Scaler for Cm")
    args = parser.parse_args()

    # ----------------------------
    # Load model and transformers
    # ----------------------------
    try:
        model = tf.keras.models.load_model(args.model, compile=False)
        scaler_x = joblib.load(args.scalerx)
        pca_cl = joblib.load(args.pca_cl)
        pca_cd = joblib.load(args.pca_cd)
        pca_cm = joblib.load(args.pca_cm)
        scaler_cl = joblib.load(args.scaler_cl)
        scaler_cd = joblib.load(args.scaler_cd)
        scaler_cm = joblib.load(args.scaler_cm)
        print("✅ Loaded model and scalers successfully.")
    except Exception as e:
        print(f"❌ Error loading model/scalers: {e}")
        sys.exit(1)

    # ----------------------------
    # Read and preprocess surfaces
    # ----------------------------
    xU, yU = read_airfoil_surface(args.upper)
    xL, yL = read_airfoil_surface(args.lower)
    xU, yU = xU[np.argsort(xU)], yU[np.argsort(xU)]
    xL, yL = xL[np.argsort(xL)], yL[np.argsort(xL)]
    xU, xL = xU / np.max(xU), xL / np.max(xL)

    # ----------------------------
    # Fit upper & lower surfaces
    # ----------------------------
    try:
        popt_upper, _ = curve_fit(CST_chebyshev_TE, xU, yU, p0=np.zeros(10))
        popt_lower, _ = curve_fit(CST_chebyshev_TE, xL, yL, p0=np.zeros(10))
        print("✅ CST fitting successful.")
    except Exception as e:
        print(f"❌ CST fitting failed: {e}")
        sys.exit(1)

    # Combine upper + lower coefficients (total 20)
    X_input = np.concatenate([popt_upper, popt_lower])[None, :]

    # ----------------------------
    # Scale and predict
    # ----------------------------
    X_scaled = scaler_x.transform(X_input)
    y_pred = model.predict(X_scaled)
    print("✅ Neural network prediction complete.")

    # Split into Cl, Cd, Cm parts
    Y_cl = y_pred[:, 0:10]
    Y_cd = y_pred[:, 10:28]
    Y_cm = y_pred[:, 28:36]

    # Inverse PCA + scaling
    Cl = scaler_cl.inverse_transform(pca_cl.inverse_transform(Y_cl))
    Cd = scaler_cd.inverse_transform(pca_cd.inverse_transform(Y_cd))
    Cm = scaler_cm.inverse_transform(pca_cm.inverse_transform(Y_cm))
    
    base_dir = "./data" 
#     print("✅ Inverse PCA and scaling done.")
    
    # Start blending C81 table
    
 	#base_dir = "./data"      # local folder with .dat files

    file_map = {
        "Cl": "Cl_360_23012.dat",
        "Cd": "Cd_360_23012.dat",
        "Cm": "Cm_360_23012.dat"
    }

    # ----------------------------------------------------------
    # STEP 3 — Read all NACA0012 tables
    # ----------------------------------------------------------
    tables = {}
    for key, fname in file_map.items():
        fpath = os.path.join(base_dir, fname)
        tables[key] = read_airfoil_dat(fpath)
        print(f"Loaded {key}: {tables[key].shape[0]} rows")

    # ----------------------------------------------------------
    # STEP 4 — Prepare ANN predictions
    # ----------------------------------------------------------
    aoa_ann = np.linspace(-10, 20, 31)

    Y_pred_cl_orig_pt3 = Cl.flatten()
    Y_pred_cd_orig_pt3 = Cd.flatten()
    Y_pred_cm_orig_pt3 = Cm.flatten()

    # ----------------------------------------------------------
    # STEP 5 — Extract NACA AoA / Mach 0.3 data
    # ----------------------------------------------------------
    def extract(df):
        return df["AoA"].values, df["Mach_0.6"].values

    aoa_cl_naca, cl_naca = extract(tables["Cl"])
    aoa_cd_naca, cd_naca = extract(tables["Cd"])
    aoa_cm_naca, cm_naca = extract(tables["Cm"])

    # ----------------------------------------------------------
    # STEP 6 — Cosine smoothing blend
    # ----------------------------------------------------------
    def blend_curves(aoa_naca, data_naca, aoa_ann, data_ann,
                     aoa_min=-10, aoa_max=20, blend_width=10):

        aoa_full = np.linspace(-180, 180, 361)

        aoa_naca = np.array(aoa_naca).flatten()
        data_naca = np.array(data_naca).flatten()
        aoa_ann = np.array(aoa_ann).flatten()
        data_ann = np.array(data_ann).flatten()

        data_naca_interp = np.interp(aoa_full, aoa_naca, data_naca)
        data_ann_interp = np.interp(aoa_full, aoa_ann, data_ann)

        blended = data_naca_interp.copy()

        # direct replacement inside ANN region
        idx_ann = (aoa_full >= aoa_min) & (aoa_full <= aoa_max)
        blended[idx_ann] = data_ann_interp[idx_ann]

        def smooth_window(x, x0, w):
            return 0.5 * (1 - np.cos(np.pi * (x - x0 + w) / w))

        # lower blend region
        mask_low = (aoa_full >= aoa_min - blend_width) & (aoa_full < aoa_min)
        if np.any(mask_low):
            w = smooth_window(aoa_full[mask_low], aoa_min, blend_width)
            blended[mask_low] = (1 - w) * data_naca_interp[mask_low] + w * data_ann_interp[mask_low]

        # upper blend region
        mask_high = (aoa_full > aoa_max) & (aoa_full <= aoa_max + blend_width)
        if np.any(mask_high):
            w = 1 - smooth_window(aoa_full[mask_high], aoa_max, blend_width)
            blended[mask_high] = (1 - w) * data_ann_interp[mask_high] + w * data_naca_interp[mask_high]

        return aoa_full, blended

    # ----------------------------------------------------------
    # STEP 7 — Blend curves
    # ----------------------------------------------------------
    aoa_full, cl_blend = blend_curves(aoa_cl_naca, cl_naca, aoa_ann, Y_pred_cl_orig_pt3)
    _, cd_blend = blend_curves(aoa_cd_naca, cd_naca, aoa_ann, Y_pred_cd_orig_pt3)
    _, cm_blend = blend_curves(aoa_cm_naca, cm_naca, aoa_ann, Y_pred_cm_orig_pt3)
    
    # Limit to 4 decimal places
    
    cl_blend = np.round(cl_blend,4)
    cd_blend = np.round(cd_blend,4)
    cm_blend = np.round(cm_blend,4)
    

    # ----------------------------------------------------------
    # STEP 8 — Save final C81 table
    # ----------------------------------------------------------
    c81 = pd.DataFrame({
        "AoA": aoa_full,
        "Cl": cl_blend,
        "Cd": cd_blend,
        "Cm": cm_blend
    })

    output_file = "C81_Mach0.6.dat"
    c81.to_csv(output_file, sep="\t", index=False)
    print(f"\nSaved C81 table → {output_file}")
    

    # Save results
#     np.savetxt("predicted_Cl.dat", Cl)
#     np.savetxt("predicted_Cd.dat", Cd)
#     np.savetxt("predicted_Cm.dat", Cm)
#     print("✅ Saved: predicted_Cl.dat, predicted_Cd.dat, predicted_Cm.dat")

if __name__ == "__main__":
    main()

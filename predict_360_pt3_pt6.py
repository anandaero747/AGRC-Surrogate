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
    parser.add_argument("--model_pt3", default="models/Forward_pt3_non_opt_hyp_v2.h5", help="Path to trained model (.h5)")
    parser.add_argument("--model_pt4", default="models/Forward_pt4_non_opt_hyp_v2.h5", help="Path to trained model (.h5)")
    parser.add_argument("--model_pt5", default="models/Forward_pt5_non_opt_hyp_v2.h5", help="Path to trained model (.h5)")
    parser.add_argument("--model_pt6", default="models/Forward_pt6_non_opt_hyp_v2.h5", help="Path to trained model (.h5)")
    parser.add_argument("--model_pt7", default="models/Forward_pt7_non_opt_hyp_v2.h5", help="Path to trained model (.h5)")
    
    parser.add_argument("--scalerx", default="preprocess/scaler_cst_v2.pkl", help="Path to input scaler")
    
    parser.add_argument("--pca_cl_pt3", default="preprocess/pca_cl_pt3_v2.pkl", help="PCA model for Cl")
    parser.add_argument("--pca_cd_pt3", default="preprocess/pca_cd_pt3_v2.pkl", help="PCA model for Cd")
    parser.add_argument("--pca_cm_pt3", default="preprocess/pca_cm_pt3_v2.pkl", help="PCA model for Cm")
    parser.add_argument("--scaler_cl_pt3", default="preprocess/scaler_cl_pt3_v2.pkl", help="Scaler for Cl")
    parser.add_argument("--scaler_cd_pt3", default="preprocess/scaler_cd_pt3_v2.pkl", help="Scaler for Cd")
    parser.add_argument("--scaler_cm_pt3", default="preprocess/scaler_cm_pt3_v2.pkl", help="Scaler for Cm")
    
    parser.add_argument("--pca_cl_pt4", default="preprocess/pca_cl_pt4_v2.pkl", help="PCA model for Cl")
    parser.add_argument("--pca_cd_pt4", default="preprocess/pca_cd_pt4_v2.pkl", help="PCA model for Cd")
    parser.add_argument("--pca_cm_pt4", default="preprocess/pca_cm_pt4_v2.pkl", help="PCA model for Cm")
    parser.add_argument("--scaler_cl_pt4", default="preprocess/scaler_cl_pt4_v2.pkl", help="Scaler for Cl")
    parser.add_argument("--scaler_cd_pt4", default="preprocess/scaler_cd_pt4_v2.pkl", help="Scaler for Cd")
    parser.add_argument("--scaler_cm_pt4", default="preprocess/scaler_cm_pt4_v2.pkl", help="Scaler for Cm")
    
    parser.add_argument("--pca_cl_pt5", default="preprocess/pca_cl_pt5_v2.pkl", help="PCA model for Cl")
    parser.add_argument("--pca_cd_pt5", default="preprocess/pca_cd_pt5_v2.pkl", help="PCA model for Cd")
    parser.add_argument("--pca_cm_pt5", default="preprocess/pca_cm_pt5_v2.pkl", help="PCA model for Cm")
    parser.add_argument("--scaler_cl_pt5", default="preprocess/scaler_cl_pt5_v2.pkl", help="Scaler for Cl")
    parser.add_argument("--scaler_cd_pt5", default="preprocess/scaler_cd_pt5_v2.pkl", help="Scaler for Cd")
    parser.add_argument("--scaler_cm_pt5", default="preprocess/scaler_cm_pt5_v2.pkl", help="Scaler for Cm")
    
    parser.add_argument("--pca_cl_pt6", default="preprocess/pca_cl_pt6_v2.pkl", help="PCA model for Cl")
    parser.add_argument("--pca_cd_pt6", default="preprocess/pca_cd_pt6_v2.pkl", help="PCA model for Cd")
    parser.add_argument("--pca_cm_pt6", default="preprocess/pca_cm_pt6_v2.pkl", help="PCA model for Cm")
    parser.add_argument("--scaler_cl_pt6", default="preprocess/scaler_cl_pt6_v2.pkl", help="Scaler for Cl")
    parser.add_argument("--scaler_cd_pt6", default="preprocess/scaler_cd_pt6_v2.pkl", help="Scaler for Cd")
    parser.add_argument("--scaler_cm_pt6", default="preprocess/scaler_cm_pt6_v2.pkl", help="Scaler for Cm")
    
    parser.add_argument("--pca_cl_pt7", default="preprocess/pca_cl_pt7_v2.pkl", help="PCA model for Cl")
    parser.add_argument("--pca_cd_pt7", default="preprocess/pca_cd_pt7_v2.pkl", help="PCA model for Cd")
    parser.add_argument("--pca_cm_pt7", default="preprocess/pca_cm_pt7_v2.pkl", help="PCA model for Cm")
    parser.add_argument("--scaler_cl_pt7", default="preprocess/scaler_cl_pt7_v2.pkl", help="Scaler for Cl")
    parser.add_argument("--scaler_cd_pt7", default="preprocess/scaler_cd_pt7_v2.pkl", help="Scaler for Cd")
    parser.add_argument("--scaler_cm_pt7", default="preprocess/scaler_cm_pt7_v2.pkl", help="Scaler for Cm")
    
    args = parser.parse_args()

    # ----------------------------
    # Load model and transformers
    # ----------------------------
    try:
        model_pt3 = tf.keras.models.load_model(args.model_pt3, compile=False)
        model_pt4 = tf.keras.models.load_model(args.model_pt4, compile=False)
        model_pt5 = tf.keras.models.load_model(args.model_pt5, compile=False)
        model_pt6 = tf.keras.models.load_model(args.model_pt6, compile=False)
        model_pt7 = tf.keras.models.load_model(args.model_pt7, compile=False)
        
        scaler_x = joblib.load(args.scalerx)
        
        pca_cl_pt3 = joblib.load(args.pca_cl_pt3)
        pca_cd_pt3 = joblib.load(args.pca_cd_pt3)
        pca_cm_pt3 = joblib.load(args.pca_cm_pt3)
        scaler_cl_pt3 = joblib.load(args.scaler_cl_pt3)
        scaler_cd_pt3 = joblib.load(args.scaler_cd_pt3)
        scaler_cm_pt3 = joblib.load(args.scaler_cm_pt3)
        
        pca_cl_pt4 = joblib.load(args.pca_cl_pt4)
        pca_cd_pt4 = joblib.load(args.pca_cd_pt4)
        pca_cm_pt4 = joblib.load(args.pca_cm_pt4)
        scaler_cl_pt4 = joblib.load(args.scaler_cl_pt4)
        scaler_cd_pt4 = joblib.load(args.scaler_cd_pt4)
        scaler_cm_pt4 = joblib.load(args.scaler_cm_pt4)
        
        pca_cl_pt5 = joblib.load(args.pca_cl_pt5)
        pca_cd_pt5 = joblib.load(args.pca_cd_pt5)
        pca_cm_pt5 = joblib.load(args.pca_cm_pt5)
        scaler_cl_pt5 = joblib.load(args.scaler_cl_pt5)
        scaler_cd_pt5 = joblib.load(args.scaler_cd_pt5)
        scaler_cm_pt5 = joblib.load(args.scaler_cm_pt5)
        
        pca_cl_pt6 = joblib.load(args.pca_cl_pt6)
        pca_cd_pt6 = joblib.load(args.pca_cd_pt6)
        pca_cm_pt6 = joblib.load(args.pca_cm_pt6)
        scaler_cl_pt6 = joblib.load(args.scaler_cl_pt6)
        scaler_cd_pt6 = joblib.load(args.scaler_cd_pt6)
        scaler_cm_pt6 = joblib.load(args.scaler_cm_pt6)
        
        pca_cl_pt7 = joblib.load(args.pca_cl_pt7)
        pca_cd_pt7 = joblib.load(args.pca_cd_pt7)
        pca_cm_pt7 = joblib.load(args.pca_cm_pt7)
        scaler_cl_pt7 = joblib.load(args.scaler_cl_pt7)
        scaler_cd_pt7 = joblib.load(args.scaler_cd_pt7)
        scaler_cm_pt7 = joblib.load(args.scaler_cm_pt7)
        
        
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
    y_pred_pt3 = model_pt3.predict(X_scaled)
    y_pred_pt4 = model_pt4.predict(X_scaled)
    y_pred_pt5 = model_pt5.predict(X_scaled)
    y_pred_pt6 = model_pt6.predict(X_scaled)
    y_pred_pt7 = model_pt7.predict(X_scaled)
    print("✅ Neural network prediction complete.")

    # Split into Cl, Cd, Cm parts
    Y_cl_pt3 = y_pred_pt3[:, 0:10]
    Y_cd_pt3 = y_pred_pt3[:, 10:28]
    Y_cm_pt3 = y_pred_pt3[:, 28:36]
    
    Y_cl_pt4 = y_pred_pt4[:, 0:10]
    Y_cd_pt4 = y_pred_pt4[:, 10:28]
    Y_cm_pt4 = y_pred_pt4[:, 28:36]
    
    Y_cl_pt5 = y_pred_pt5[:, 0:10]
    Y_cd_pt5 = y_pred_pt5[:, 10:28]
    Y_cm_pt5 = y_pred_pt5[:, 28:36]
    
    Y_cl_pt6 = y_pred_pt6[:, 0:10]
    Y_cd_pt6 = y_pred_pt6[:, 10:28]
    Y_cm_pt6 = y_pred_pt6[:, 28:36]
    
    Y_cl_pt7 = y_pred_pt7[:, 0:10]
    Y_cd_pt7 = y_pred_pt7[:, 10:28]
    Y_cm_pt7 = y_pred_pt7[:, 28:36]

    # Inverse PCA + scaling
    Cl_pt3 = scaler_cl_pt3.inverse_transform(pca_cl_pt3.inverse_transform(Y_cl_pt3))
    Cd_pt3 = scaler_cd_pt3.inverse_transform(pca_cd_pt3.inverse_transform(Y_cd_pt3))
    Cm_pt3 = scaler_cm_pt3.inverse_transform(pca_cm_pt3.inverse_transform(Y_cm_pt3))
    
    Cl_pt4 = scaler_cl_pt4.inverse_transform(pca_cl_pt4.inverse_transform(Y_cl_pt4))
    Cd_pt4 = scaler_cd_pt4.inverse_transform(pca_cd_pt4.inverse_transform(Y_cd_pt4))
    Cm_pt4 = scaler_cm_pt4.inverse_transform(pca_cm_pt4.inverse_transform(Y_cm_pt4))
    
    Cl_pt5 = scaler_cl_pt5.inverse_transform(pca_cl_pt5.inverse_transform(Y_cl_pt5))
    Cd_pt5 = scaler_cd_pt5.inverse_transform(pca_cd_pt5.inverse_transform(Y_cd_pt5))
    Cm_pt5 = scaler_cm_pt5.inverse_transform(pca_cm_pt5.inverse_transform(Y_cm_pt5))
    
    Cl_pt6 = scaler_cl_pt6.inverse_transform(pca_cl_pt6.inverse_transform(Y_cl_pt6))
    Cd_pt6 = scaler_cd_pt6.inverse_transform(pca_cd_pt6.inverse_transform(Y_cd_pt6))
    Cm_pt6 = scaler_cm_pt6.inverse_transform(pca_cm_pt6.inverse_transform(Y_cm_pt6))

    Cl_pt7 = scaler_cl_pt7.inverse_transform(pca_cl_pt7.inverse_transform(Y_cl_pt7))
    Cd_pt7 = scaler_cd_pt7.inverse_transform(pca_cd_pt7.inverse_transform(Y_cd_pt7))
    Cm_pt7 = scaler_cm_pt7.inverse_transform(pca_cm_pt7.inverse_transform(Y_cm_pt7))
    
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
    aoa_ann_pt7 = np.linspace(-4, 20, 25)

    Y_pred_cl_orig_pt3 = Cl_pt3.flatten()
    Y_pred_cd_orig_pt3 = Cd_pt3.flatten()
    Y_pred_cm_orig_pt3 = Cm_pt3.flatten()
    
    Y_pred_cl_orig_pt4 = Cl_pt4.flatten()
    Y_pred_cd_orig_pt4 = Cd_pt4.flatten()
    Y_pred_cm_orig_pt4 = Cm_pt4.flatten()
    
    Y_pred_cl_orig_pt5 = Cl_pt5.flatten()
    Y_pred_cd_orig_pt5 = Cd_pt5.flatten()
    Y_pred_cm_orig_pt5 = Cm_pt5.flatten()
    
    Y_pred_cl_orig_pt6 = Cl_pt6.flatten()
    Y_pred_cd_orig_pt6 = Cd_pt6.flatten()
    Y_pred_cm_orig_pt6 = Cm_pt6.flatten()
    
    Y_pred_cl_orig_pt7 = Cl_pt7.flatten()
    Y_pred_cd_orig_pt7 = Cd_pt7.flatten()
    Y_pred_cm_orig_pt7 = Cm_pt7.flatten()

    # ----------------------------------------------------------
    # STEP 5 — Extract NACA AoA / Mach 0.3 data
    # ----------------------------------------------------------
    def extract_pt3(df):
        return df["AoA"].values, df["Mach_0.3"].values

    aoa_cl_naca_pt3, cl_naca_pt3 = extract_pt3(tables["Cl"])
    aoa_cd_naca_pt3, cd_naca_pt3 = extract_pt3(tables["Cd"])
    aoa_cm_naca_pt3, cm_naca_pt3 = extract_pt3(tables["Cm"])
    
    def extract_pt4(df):
        return df["AoA"].values, df["Mach_0.4"].values

    aoa_cl_naca_pt4, cl_naca_pt4 = extract_pt4(tables["Cl"])
    aoa_cd_naca_pt4, cd_naca_pt4 = extract_pt4(tables["Cd"])
    aoa_cm_naca_pt4, cm_naca_pt4 = extract_pt4(tables["Cm"])
    
    def extract_pt5(df):
        return df["AoA"].values, df["Mach_0.5"].values

    aoa_cl_naca_pt5, cl_naca_pt5 = extract_pt5(tables["Cl"])
    aoa_cd_naca_pt5, cd_naca_pt5 = extract_pt5(tables["Cd"])
    aoa_cm_naca_pt5, cm_naca_pt5 = extract_pt5(tables["Cm"])
    
    def extract_pt6(df):
        return df["AoA"].values, df["Mach_0.6"].values

    aoa_cl_naca_pt6, cl_naca_pt6 = extract_pt6(tables["Cl"])
    aoa_cd_naca_pt6, cd_naca_pt6 = extract_pt6(tables["Cd"])
    aoa_cm_naca_pt6, cm_naca_pt6 = extract_pt6(tables["Cm"])
    
    def extract_pt7(df):
        return df["AoA"].values, df["Mach_0.7"].values

    aoa_cl_naca_pt7, cl_naca_pt7 = extract_pt7(tables["Cl"])
    aoa_cd_naca_pt7, cd_naca_pt7 = extract_pt7(tables["Cd"])
    aoa_cm_naca_pt7, cm_naca_pt7 = extract_pt7(tables["Cm"])

    # ----------------------------------------------------------
    # STEP 6 — Cosine smoothing blend
    # ----------------------------------------------------------
    
    # Blending for 0.3 to 0.6
    
    def blend_curves_pt7(aoa_naca, data_naca, aoa_ann, data_ann,
                     aoa_min=-4, aoa_max=20, blend_width=10):

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
        
        # Blending for 0.3 to 0.6
    
    
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
    aoa_full, cl_blend_pt3 = blend_curves(aoa_cl_naca_pt3, cl_naca_pt3, aoa_ann, Y_pred_cl_orig_pt3)
    _, cd_blend_pt3 = blend_curves(aoa_cd_naca_pt3, cd_naca_pt3, aoa_ann, Y_pred_cd_orig_pt3)
    _, cm_blend_pt3 = blend_curves(aoa_cm_naca_pt3, cm_naca_pt3, aoa_ann, Y_pred_cm_orig_pt3)
    
    # Limit to 4 decimal places
    
    cl_blend_pt3 = np.round(cl_blend_pt3,4)
    cd_blend_pt3 = np.round(cd_blend_pt3,4)
    cm_blend_pt3 = np.round(cm_blend_pt3,4)
    
    # Mach 0.4
    
    aoa_full, cl_blend_pt4 = blend_curves(aoa_cl_naca_pt4, cl_naca_pt4, aoa_ann, Y_pred_cl_orig_pt4)
    _, cd_blend_pt4 = blend_curves(aoa_cd_naca_pt4, cd_naca_pt4, aoa_ann, Y_pred_cd_orig_pt4)
    _, cm_blend_pt4 = blend_curves(aoa_cm_naca_pt4, cm_naca_pt4, aoa_ann, Y_pred_cm_orig_pt4)
    
    # Limit to 4 decimal places
    
    cl_blend_pt4 = np.round(cl_blend_pt4,4)
    cd_blend_pt4 = np.round(cd_blend_pt4,4)
    cm_blend_pt4 = np.round(cm_blend_pt4,4)
    
    # Mach 0.5
    
    aoa_full, cl_blend_pt5 = blend_curves(aoa_cl_naca_pt5, cl_naca_pt5, aoa_ann, Y_pred_cl_orig_pt5)
    _, cd_blend_pt5 = blend_curves(aoa_cd_naca_pt5, cd_naca_pt5, aoa_ann, Y_pred_cd_orig_pt5)
    _, cm_blend_pt5 = blend_curves(aoa_cm_naca_pt5, cm_naca_pt5, aoa_ann, Y_pred_cm_orig_pt5)
    
    # Limit to 4 decimal places
    
    cl_blend_pt5 = np.round(cl_blend_pt5,4)
    cd_blend_pt5 = np.round(cd_blend_pt5,4)
    cm_blend_pt5 = np.round(cm_blend_pt5,4)
    
    # Mach 0.6
    
    aoa_full, cl_blend_pt6 = blend_curves(aoa_cl_naca_pt6, cl_naca_pt6, aoa_ann, Y_pred_cl_orig_pt6)
    _, cd_blend_pt6 = blend_curves(aoa_cd_naca_pt6, cd_naca_pt6, aoa_ann, Y_pred_cd_orig_pt6)
    _, cm_blend_pt6 = blend_curves(aoa_cm_naca_pt6, cm_naca_pt6, aoa_ann, Y_pred_cm_orig_pt6)
    
    # Limit to 4 decimal places
    
    cl_blend_pt6 = np.round(cl_blend_pt6,4)
    cd_blend_pt6 = np.round(cd_blend_pt6,4)
    cm_blend_pt6 = np.round(cm_blend_pt6,4)
    
        # Mach 0.7
    
    aoa_full, cl_blend_pt7 = blend_curves_pt7(aoa_cl_naca_pt7, cl_naca_pt7, aoa_ann_pt7, Y_pred_cl_orig_pt7)
    _, cd_blend_pt7 = blend_curves_pt7(aoa_cd_naca_pt7, cd_naca_pt7, aoa_ann_pt7, Y_pred_cd_orig_pt7)
    _, cm_blend_pt7 = blend_curves_pt7(aoa_cm_naca_pt7, cm_naca_pt7, aoa_ann_pt7, Y_pred_cm_orig_pt7)
    
    # Limit to 4 decimal places
    
    cl_blend_pt7 = np.round(cl_blend_pt7,4)
    cd_blend_pt7 = np.round(cd_blend_pt7,4)
    cm_blend_pt7 = np.round(cm_blend_pt7,4)
    

    # ----------------------------------------------------------
    # STEP 8 — Save final C81 table
    # ----------------------------------------------------------
    c81_pt3 = pd.DataFrame({
        "AoA": aoa_full,
        "Cl": cl_blend_pt3,
        "Cd": cd_blend_pt3,
        "Cm": cm_blend_pt3
    })

    output_file_pt3 = "C81_Mach0.3.dat"
    c81_pt3.to_csv(output_file_pt3, sep="\t", index=False)
    
    # Mach 0.4
    
    c81_pt4 = pd.DataFrame({
        "AoA": aoa_full,
        "Cl": cl_blend_pt4,
        "Cd": cd_blend_pt4,
        "Cm": cm_blend_pt4
    })

    output_file_pt4 = "C81_Mach0.4.dat"
    c81_pt4.to_csv(output_file_pt4, sep="\t", index=False)
    
    # Mach 0.5
    
    c81_pt5 = pd.DataFrame({
        "AoA": aoa_full,
        "Cl": cl_blend_pt5,
        "Cd": cd_blend_pt5,
        "Cm": cm_blend_pt5
    })

    output_file_pt5 = "C81_Mach0.5.dat"
    c81_pt5.to_csv(output_file_pt5, sep="\t", index=False)
    
    # Mach 0.6
    
    c81_pt6 = pd.DataFrame({
        "AoA": aoa_full,
        "Cl": cl_blend_pt6,
        "Cd": cd_blend_pt6,
        "Cm": cm_blend_pt6
    })

    output_file_pt6 = "C81_Mach0.6.dat"
    c81_pt6.to_csv(output_file_pt6, sep="\t", index=False)
    print(f"\nSaved C81 table → {output_file_pt3}")
    
        # Mach 0.7
    
    c81_pt7 = pd.DataFrame({
        "AoA": aoa_full,
        "Cl": cl_blend_pt7,
        "Cd": cd_blend_pt7,
        "Cm": cm_blend_pt7
    })

    output_file_pt7 = "C81_Mach0.7.dat"
    c81_pt7.to_csv(output_file_pt7, sep="\t", index=False)
    print(f"\nSaved C81 table → {output_file_pt7}")
    
    output_file="C81_all_mach.dat"
    aoa = c81_pt3["AoA"].values

    # ---- 2) Stack Cl, Cd, Cm from all Mach numbers ----
    Cl_matrix = np.column_stack([
        c81_pt3["Cl"].values,
        c81_pt4["Cl"].values,
        c81_pt5["Cl"].values,
        c81_pt6["Cl"].values,
        c81_pt7["Cl"].values,
    ])

    Cd_matrix = np.column_stack([
        c81_pt3["Cd"].values,
        c81_pt4["Cd"].values,
        c81_pt5["Cd"].values,
        c81_pt6["Cd"].values,
        c81_pt7["Cd"].values,
    ])

    Cm_matrix = np.column_stack([
        c81_pt3["Cm"].values,
        c81_pt4["Cm"].values,
        c81_pt5["Cm"].values,
        c81_pt6["Cm"].values,
        c81_pt7["Cm"].values,
    ])

    # ---- 3) Mach labels ----
    mach_labels = ["M0.3", "M0.4", "M0.5", "M0.6", "M0.7"]

    # ---- 4) Write file ----
    with open(output_file, "w") as f:

        f.write("C81 Table\n\n")

        # -------------------- CL --------------------
        f.write("Cl vs alpha & Mach\n")
        f.write("alpha\t" + "\t".join(mach_labels) + "\n")

        for i, a in enumerate(aoa):
            row = [f"{a:6.2f}"] + [f"{Cl_matrix[i, j]:8.4f}" for j in range(5)]
            f.write("\t".join(row) + "\n")

        f.write("\n")

        # -------------------- CD --------------------
        f.write("Cd vs alpha & Mach\n")
        f.write("alpha\t" + "\t".join(mach_labels) + "\n")

        for i, a in enumerate(aoa):
            row = [f"{a:6.2f}"] + [f"{Cd_matrix[i, j]:8.4f}" for j in range(5)]
            f.write("\t".join(row) + "\n")

        f.write("\n")

        # -------------------- CM --------------------
        f.write("Cm vs alpha & Mach\n")
        f.write("alpha\t" + "\t".join(mach_labels) + "\n")

        for i, a in enumerate(aoa):
            row = [f"{a:6.2f}"] + [f"{Cm_matrix[i, j]:8.4f}" for j in range(5)]
            f.write("\t".join(row) + "\n")

    print(f"✅ C81 file written → {output_file}")
    

    # Save results
#     np.savetxt("predicted_Cl.dat", Cl)
#     np.savetxt("predicted_Cd.dat", Cd)
#     np.savetxt("predicted_Cm.dat", Cm)
#     print("✅ Saved: predicted_Cl.dat, predicted_Cd.dat, predicted_Cm.dat")

if __name__ == "__main__":
    main()

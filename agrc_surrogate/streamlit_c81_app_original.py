# streamlit_c81_app.py
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
import tensorflow as tf
from scipy.optimize import curve_fit
from numpy.polynomial.chebyshev import chebvander
import os, sys

st.set_page_config(layout="wide", page_title="C81 Generator")

# ----------------------------
# Helper functions (from your script)
# ----------------------------
def read_airfoil_surface_filelike(filelike):
    data = np.loadtxt(filelike)
    return data[:, 0], data[:, 1]

def class_func(x):
    return np.sqrt(x) * (1 - x)

def CST_chebyshev_TE(x, *a):
    *cheb_coeffs, te_offset = a
    xi = 2 * x - 1
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

# blending functions (as in your script)
def blend_curves(aoa_naca, data_naca, aoa_ann, data_ann, aoa_min=-10, aoa_max=20, blend_width=10):
    aoa_full = np.linspace(-180, 180, 361)
    aoa_naca = np.array(aoa_naca).flatten()
    data_naca = np.array(data_naca).flatten()
    aoa_ann = np.array(aoa_ann).flatten()
    data_ann = np.array(data_ann).flatten()
    data_naca_interp = np.interp(aoa_full, aoa_naca, data_naca)
    data_ann_interp = np.interp(aoa_full, aoa_ann, data_ann)
    blended = data_naca_interp.copy()
    idx_ann = (aoa_full >= aoa_min) & (aoa_full <= aoa_max)
    blended[idx_ann] = data_ann_interp[idx_ann]
    def smooth_window(x, x0, w):
        return 0.5 * (1 - np.cos(np.pi * (x - x0 + w) / w))
    mask_low = (aoa_full >= aoa_min - blend_width) & (aoa_full < aoa_min)
    if np.any(mask_low):
        w = smooth_window(aoa_full[mask_low], aoa_min, blend_width)
        blended[mask_low] = (1 - w) * data_naca_interp[mask_low] + w * data_ann_interp[mask_low]
    mask_high = (aoa_full > aoa_max) & (aoa_full <= aoa_max + blend_width)
    if np.any(mask_high):
        w = 1 - smooth_window(aoa_full[mask_high], aoa_max, blend_width)
        blended[mask_high] = (1 - w) * data_ann_interp[mask_high] + w * data_naca_interp[mask_high]
    return aoa_full, blended

def blend_curves_pt7(aoa_naca, data_naca, aoa_ann, data_ann, aoa_min=-4, aoa_max=20, blend_width=10):
    aoa_full = np.linspace(-180, 180, 361)
    aoa_naca = np.array(aoa_naca).flatten()
    data_naca = np.array(data_naca).flatten()
    aoa_ann = np.array(aoa_ann).flatten()
    data_ann = np.array(data_ann).flatten()
    data_naca_interp = np.interp(aoa_full, aoa_naca, data_naca)
    data_ann_interp = np.interp(aoa_full, aoa_ann, data_ann)
    blended = data_naca_interp.copy()
    idx_ann = (aoa_full >= aoa_min) & (aoa_full <= aoa_max)
    blended[idx_ann] = data_ann_interp[idx_ann]
    def smooth_window(x, x0, w):
        return 0.5 * (1 - np.cos(np.pi * (x - x0 + w) / w))
    mask_low = (aoa_full >= aoa_min - blend_width) & (aoa_full < aoa_min)
    if np.any(mask_low):
        w = smooth_window(aoa_full[mask_low], aoa_min, blend_width)
        blended[mask_low] = (1 - w) * data_naca_interp[mask_low] + w * data_ann_interp[mask_low]
    mask_high = (aoa_full > aoa_max) & (aoa_full <= aoa_max + blend_width)
    if np.any(mask_high):
        w = 1 - smooth_window(aoa_full[mask_high], aoa_max, blend_width)
        blended[mask_high] = (1 - w) * data_ann_interp[mask_high] + w * data_naca_interp[mask_high]
    return aoa_full, blended

# ----------------------------
# Streamlit UI layout
# ----------------------------
st.title("C81 Generator — Streamlit")
st.write("Upload upper/lower surface .dat files, parametrize, predict, blend and produce C81 output.")

col1, col2 = st.columns([1,1])
with col1:
    upper_file = st.file_uploader("Upload Upper Surface (.dat/.txt)", type=["dat","txt"])
with col2:
    lower_file = st.file_uploader("Upload Lower Surface (.dat/.txt)", type=["dat","txt"])

st.markdown("---")
st.header("Model & Preprocessor Paths")
colm1, colm2 = st.columns(2)
with colm1:
    model_pt3_path = st.text_input("Model M=0.3 .h5", "models/Forward_pt3_non_opt_hyp_v2.h5")
    model_pt4_path = st.text_input("Model M=0.4 .h5", "models/Forward_pt4_non_opt_hyp_v2.h5")
    model_pt5_path = st.text_input("Model M=0.5 .h5", "models/Forward_pt5_non_opt_hyp_v2.h5")
with colm2:
    model_pt6_path = st.text_input("Model M=0.6 .h5", "models/Forward_pt6_non_opt_hyp_v2.h5")
    model_pt7_path = st.text_input("Model M=0.7 .h5", "models/Forward_pt7_non_opt_hyp_v2.h5")
    scalerx_path = st.text_input("ScalerX .pkl", "preprocess/scaler_cst_v2.pkl")

st.write("Preprocess PCA/Scaler files default paths are from your script. Change if needed.")

# Allow user to pick Mach to plot
mach_list = [0.3,0.4,0.5,0.6,0.7]
mach_selected = st.selectbox("Select Mach for plotting", mach_list, index=3)

# Button to run generation
run_button = st.button("Generate C81 for uploaded airfoil")

if run_button:
    if (upper_file is None) or (lower_file is None):
        st.error("Please upload both upper and lower surface files.")
    else:
        try:
            # read surfaces
            xu, yu = read_airfoil_surface_filelike(upper_file)
            xl, yl = read_airfoil_surface_filelike(lower_file)

            # preprocess / normalize x as in your script
            xu, yu = xu[np.argsort(xu)], yu[np.argsort(xu)]
            xl, yl = xl[np.argsort(xl)], yl[np.argsort(xl)]
            xu = xu / np.max(xu)
            xl = xl / np.max(xl)

            st.success("Parsed surface files and normalized X coordinates.")

            # fit CST — using similar p0 as your script (10 params)
            try:
                popt_upper, _ = curve_fit(CST_chebyshev_TE, xu, yu, p0=np.zeros(10), maxfev=20000)
                popt_lower, _ = curve_fit(CST_chebyshev_TE, xl, yl, p0=np.zeros(10), maxfev=20000)
                st.success("CST fit complete.")
            except Exception as e:
                st.warning(f"CST fit failed or had issues: {e}\nContinuing with best-effort polyfit fallback.")
                popt_upper = np.polyfit(xu, yu, 9)
                popt_lower = np.polyfit(xl, yl, 9)

            X_input = np.concatenate([popt_upper, popt_lower])[None,:]

            # load models & preprocessors
            try:
                model_pt3 = tf.keras.models.load_model(model_pt3_path, compile=False)
                model_pt4 = tf.keras.models.load_model(model_pt4_path, compile=False)
                model_pt5 = tf.keras.models.load_model(model_pt5_path, compile=False)
                model_pt6 = tf.keras.models.load_model(model_pt6_path, compile=False)
                model_pt7 = tf.keras.models.load_model(model_pt7_path, compile=False)
                scaler_x = joblib.load(scalerx_path)
                st.success("Loaded models and scaler_x.")
            except Exception as e:
                st.error(f"Failed to load models/scalers: {e}")
                st.stop()

            # scale & predict
            X_scaled = scaler_x.transform(X_input)
            y_pred_pt3 = model_pt3.predict(X_scaled)
            y_pred_pt4 = model_pt4.predict(X_scaled)
            y_pred_pt5 = model_pt5.predict(X_scaled)
            y_pred_pt6 = model_pt6.predict(X_scaled)
            y_pred_pt7 = model_pt7.predict(X_scaled)
            st.success("NN predictions done for 5 Mach numbers.")

            # split and inverse PCA+scaler — load pca/scaler pkl paths like in your code
            # We'll try to auto-load using the same path pattern as your script
            def load_pca_scalers(prefix):
                pca_cl = joblib.load(f"preprocess/pca_cl_{prefix}_v2.pkl")
                pca_cd = joblib.load(f"preprocess/pca_cd_{prefix}_v2.pkl")
                pca_cm = joblib.load(f"preprocess/pca_cm_{prefix}_v2.pkl")
                scaler_cl = joblib.load(f"preprocess/scaler_cl_{prefix}_v2.pkl")
                scaler_cd = joblib.load(f"preprocess/scaler_cd_{prefix}_v2.pkl")
                scaler_cm = joblib.load(f"preprocess/scaler_cm_{prefix}_v2.pkl")
                return pca_cl,pca_cd,pca_cm,scaler_cl,scaler_cd,scaler_cm

            try:
                pca_cl_pt3,pca_cd_pt3,pca_cm_pt3,scaler_cl_pt3,scaler_cd_pt3,scaler_cm_pt3 = load_pca_scalers("pt3")
                pca_cl_pt4,pca_cd_pt4,pca_cm_pt4,scaler_cl_pt4,scaler_cd_pt4,scaler_cm_pt4 = load_pca_scalers("pt4")
                pca_cl_pt5,pca_cd_pt5,pca_cm_pt5,scaler_cl_pt5,scaler_cd_pt5,scaler_cm_pt5 = load_pca_scalers("pt5")
                pca_cl_pt6,pca_cd_pt6,pca_cm_pt6,scaler_cl_pt6,scaler_cd_pt6,scaler_cm_pt6 = load_pca_scalers("pt6")
                pca_cl_pt7,pca_cd_pt7,pca_cm_pt7,scaler_cl_pt7,scaler_cd_pt7,scaler_cm_pt7 = load_pca_scalers("pt7")
                st.success("Loaded PCA & scaler pickles.")
            except Exception as e:
                st.error(f"Failed to load PCA/scaler pickles automatically: {e}\nMake sure preprocess/ files exist.")
                st.stop()

            # split predicted arrays
            def split_and_inv(y_pred, pca_cl, pca_cd, pca_cm, s_cl, s_cd, s_cm):
                Y_cl = y_pred[:, 0:10]
                Y_cd = y_pred[:, 10:28]
                Y_cm = y_pred[:, 28:36]
                Cl = s_cl.inverse_transform(pca_cl.inverse_transform(Y_cl))
                Cd = np.exp(s_cd.inverse_transform(pca_cd.inverse_transform(Y_cd)))
                Cm = s_cm.inverse_transform(pca_cm.inverse_transform(Y_cm))
                return Cl.flatten(), Cd.flatten(), Cm.flatten()

            Cl_pt3, Cd_pt3, Cm_pt3 = split_and_inv(y_pred_pt3, pca_cl_pt3, pca_cd_pt3, pca_cm_pt3, scaler_cl_pt3, scaler_cd_pt3, scaler_cm_pt3)
            Cl_pt4, Cd_pt4, Cm_pt4 = split_and_inv(y_pred_pt4, pca_cl_pt4, pca_cd_pt4, pca_cm_pt4, scaler_cl_pt4, scaler_cd_pt4, scaler_cm_pt4)
            Cl_pt5, Cd_pt5, Cm_pt5 = split_and_inv(y_pred_pt5, pca_cl_pt5, pca_cd_pt5, pca_cm_pt5, scaler_cl_pt5, scaler_cd_pt5, scaler_cm_pt5)
            Cl_pt6, Cd_pt6, Cm_pt6 = split_and_inv(y_pred_pt6, pca_cl_pt6, pca_cd_pt6, pca_cm_pt6, scaler_cl_pt6, scaler_cd_pt6, scaler_cm_pt6)
            Cl_pt7, Cd_pt7, Cm_pt7 = split_and_inv(y_pred_pt7, pca_cl_pt7, pca_cd_pt7, pca_cm_pt7, scaler_cl_pt7, scaler_cd_pt7, scaler_cm_pt7)

            st.success("Inverse PCA + scaling completed.")

            # read NACA0012 baseline tables from ./data directory (same filenames as your script)
            base_dir = "./data"
            file_map = {"Cl":"Cl_360_23012.dat", "Cd":"Cd_360_23012.dat", "Cm":"Cm_360_23012.dat"}
            tables = {}
            for key,fname in file_map.items():
                fpath = os.path.join(base_dir, fname)
                tables[key] = read_airfoil_dat(fpath)

            # Prepare ANN AoA arrays
            aoa_ann = np.linspace(-10, 20, 31)
            aoa_ann_pt7 = np.linspace(-4, 20, 25)

            # Extract NACA arrays
            aoa_cl_naca_pt3, cl_naca_pt3 = tables["Cl"]["AoA"].values, tables["Cl"]["Mach_0.3"].values
            aoa_cd_naca_pt3, cd_naca_pt3 = tables["Cd"]["AoA"].values, tables["Cd"]["Mach_0.3"].values
            aoa_cm_naca_pt3, cm_naca_pt3 = tables["Cm"]["AoA"].values, tables["Cm"]["Mach_0.3"].values

            aoa_cl_naca_pt4, cl_naca_pt4 = tables["Cl"]["AoA"].values, tables["Cl"]["Mach_0.4"].values
            aoa_cd_naca_pt4, cd_naca_pt4 = tables["Cd"]["AoA"].values, tables["Cd"]["Mach_0.4"].values
            aoa_cm_naca_pt4, cm_naca_pt4 = tables["Cm"]["AoA"].values, tables["Cm"]["Mach_0.4"].values

            aoa_cl_naca_pt5, cl_naca_pt5 = tables["Cl"]["AoA"].values, tables["Cl"]["Mach_0.5"].values
            aoa_cd_naca_pt5, cd_naca_pt5 = tables["Cd"]["AoA"].values, tables["Cd"]["Mach_0.5"].values
            aoa_cm_naca_pt5, cm_naca_pt5 = tables["Cm"]["AoA"].values, tables["Cm"]["Mach_0.5"].values

            aoa_cl_naca_pt6, cl_naca_pt6 = tables["Cl"]["AoA"].values, tables["Cl"]["Mach_0.6"].values
            aoa_cd_naca_pt6, cd_naca_pt6 = tables["Cd"]["AoA"].values, tables["Cd"]["Mach_0.6"].values
            aoa_cm_naca_pt6, cm_naca_pt6 = tables["Cm"]["AoA"].values, tables["Cm"]["Mach_0.6"].values

            aoa_cl_naca_pt7, cl_naca_pt7 = tables["Cl"]["AoA"].values, tables["Cl"]["Mach_0.7"].values
            aoa_cd_naca_pt7, cd_naca_pt7 = tables["Cd"]["AoA"].values, tables["Cd"]["Mach_0.7"].values
            aoa_cm_naca_pt7, cm_naca_pt7 = tables["Cm"]["AoA"].values, tables["Cm"]["Mach_0.7"].values

            # compute blended curves for each Mach
            aoa_full, cl_blend_pt3 = blend_curves(aoa_cl_naca_pt3, cl_naca_pt3, aoa_ann, Cl_pt3)
            _, cd_blend_pt3 = blend_curves(aoa_cd_naca_pt3, cd_naca_pt3, aoa_ann, Cd_pt3)
            _, cm_blend_pt3 = blend_curves(aoa_cm_naca_pt3, cm_naca_pt3, aoa_ann, Cm_pt3)
            cl_blend_pt3 = np.round(cl_blend_pt3,4); cd_blend_pt3 = np.round(cd_blend_pt3,4); cm_blend_pt3 = np.round(cm_blend_pt3,4)

            aoa_full, cl_blend_pt4 = blend_curves(aoa_cl_naca_pt4, cl_naca_pt4, aoa_ann, Cl_pt4)
            _, cd_blend_pt4 = blend_curves(aoa_cd_naca_pt4, cd_naca_pt4, aoa_ann, Cd_pt4)
            _, cm_blend_pt4 = blend_curves(aoa_cm_naca_pt4, cm_naca_pt4, aoa_ann, Cm_pt4)
            cl_blend_pt4 = np.round(cl_blend_pt4,4); cd_blend_pt4 = np.round(cd_blend_pt4,4); cm_blend_pt4 = np.round(cm_blend_pt4,4)

            aoa_full, cl_blend_pt5 = blend_curves(aoa_cl_naca_pt5, cl_naca_pt5, aoa_ann, Cl_pt5)
            _, cd_blend_pt5 = blend_curves(aoa_cd_naca_pt5, cd_naca_pt5, aoa_ann, Cd_pt5)
            _, cm_blend_pt5 = blend_curves(aoa_cm_naca_pt5, cm_naca_pt5, aoa_ann, Cm_pt5)
            cl_blend_pt5 = np.round(cl_blend_pt5,4); cd_blend_pt5 = np.round(cd_blend_pt5,4); cm_blend_pt5 = np.round(cm_blend_pt5,4)

            aoa_full, cl_blend_pt6 = blend_curves(aoa_cl_naca_pt6, cl_naca_pt6, aoa_ann, Cl_pt6)
            _, cd_blend_pt6 = blend_curves(aoa_cd_naca_pt6, cd_naca_pt6, aoa_ann, Cd_pt6)
            _, cm_blend_pt6 = blend_curves(aoa_cm_naca_pt6, cm_naca_pt6, aoa_ann, Cm_pt6)
            cl_blend_pt6 = np.round(cl_blend_pt6,4); cd_blend_pt6 = np.round(cd_blend_pt6,4); cm_blend_pt6 = np.round(cm_blend_pt6,4)

            aoa_full, cl_blend_pt7 = blend_curves_pt7(aoa_cl_naca_pt7, cl_naca_pt7, aoa_ann_pt7, Cl_pt7)
            _, cd_blend_pt7 = blend_curves_pt7(aoa_cd_naca_pt7, cd_naca_pt7, aoa_ann_pt7, Cd_pt7)
            _, cm_blend_pt7 = blend_curves_pt7(aoa_cm_naca_pt7, cm_naca_pt7, aoa_ann_pt7, Cm_pt7)
            cl_blend_pt7 = np.round(cl_blend_pt7,4); cd_blend_pt7 = np.round(cd_blend_pt7,4); cm_blend_pt7 = np.round(cm_blend_pt7,4)

            # build dataframes
            c81_pt3 = pd.DataFrame({"AoA": aoa_full, "Cl": cl_blend_pt3, "Cd": cd_blend_pt3, "Cm": cm_blend_pt3})
            c81_pt4 = pd.DataFrame({"AoA": aoa_full, "Cl": cl_blend_pt4, "Cd": cd_blend_pt4, "Cm": cd_blend_pt4})
            c81_pt5 = pd.DataFrame({"AoA": aoa_full, "Cl": cl_blend_pt5, "Cd": cd_blend_pt5, "Cm": cm_blend_pt5})
            c81_pt6 = pd.DataFrame({"AoA": aoa_full, "Cl": cl_blend_pt6, "Cd": cd_blend_pt6, "Cm": cm_blend_pt6})
            c81_pt7 = pd.DataFrame({"AoA": aoa_full, "Cl": cl_blend_pt7, "Cd": cd_blend_pt7, "Cm": cm_blend_pt7})

            # display selected Mach table and plots
            c81_map = {0.3:c81_pt3, 0.4:c81_pt4, 0.5:c81_pt5, 0.6:c81_pt6, 0.7:c81_pt7}
            selected_df = c81_map[mach_selected]

            st.subheader(f"C81 table (Mach {mach_selected:.1f})")
            st.dataframe(selected_df)

            # plots side-by-side
            pcol1, pcol2, pcol3 = st.columns(3)
            with pcol1:
                fig, ax = plt.subplots()
                ax.plot(selected_df["AoA"], selected_df["Cl"], linewidth=1.5)
                ax.set_xlabel("AoA [deg]"); ax.set_ylabel("Cl"); ax.grid(True)
                st.pyplot(fig)
            with pcol2:
                fig, ax = plt.subplots()
                ax.plot(selected_df["AoA"], selected_df["Cd"], linewidth=1.5)
                ax.set_xlabel("AoA [deg]"); ax.set_ylabel("Cd"); ax.grid(True)
                st.pyplot(fig)
            with pcol3:
                fig, ax = plt.subplots()
                ax.plot(selected_df["AoA"], selected_df["Cm"], linewidth=1.5)
                ax.set_xlabel("AoA [deg]"); ax.set_ylabel("Cm"); ax.grid(True)
                st.pyplot(fig)

            # produce combined C81_all_mach.dat in your requested format and make downloadable
            output_text = []
            output_text.append("C81 Table\n\n")
            mach_labels = ["M0.3","M0.4","M0.5","M0.6","M0.7"]

            # CL block
            output_text.append("Cl vs alpha & Mach\n")
            output_text.append("alpha\t" + "\t".join(mach_labels) + "\n")
            Cl_matrix = np.column_stack([c81_pt3["Cl"].values, c81_pt4["Cl"].values, c81_pt5["Cl"].values, c81_pt6["Cl"].values, c81_pt7["Cl"].values])
            for i,a in enumerate(aoa_full):
                row = [f"{a:6.2f}"] + [f"{Cl_matrix[i,j]:8.4f}" for j in range(5)]
                output_text.append("\t".join(row) + "\n")
            output_text.append("\n")

            # CD block
            output_text.append("Cd vs alpha & Mach\n")
            output_text.append("alpha\t" + "\t".join(mach_labels) + "\n")
            Cd_matrix = np.column_stack([c81_pt3["Cd"].values, c81_pt4["Cd"].values, c81_pt5["Cd"].values, c81_pt6["Cd"].values, c81_pt7["Cd"].values])
            for i,a in enumerate(aoa_full):
                row = [f"{a:6.2f}"] + [f"{Cd_matrix[i,j]:8.4f}" for j in range(5)]
                output_text.append("\t".join(row) + "\n")
            output_text.append("\n")

            # CM block
            output_text.append("Cm vs alpha & Mach\n")
            output_text.append("alpha\t" + "\t".join(mach_labels) + "\n")
            Cm_matrix = np.column_stack([c81_pt3["Cm"].values, c81_pt4["Cm"].values, c81_pt5["Cm"].values, c81_pt6["Cm"].values, c81_pt7["Cm"].values])
            for i,a in enumerate(aoa_full):
                row = [f"{a:6.2f}"] + [f"{Cm_matrix[i,j]:8.4f}" for j in range(5)]
                output_text.append("\t".join(row) + "\n")

            c81_all_text = "".join(output_text)

            st.download_button("Download combined C81_all_mach.dat", data=c81_all_text, file_name="C81_all_mach.dat", mime="text/plain")
            st.success("C81 generation finished — download ready.")

        except Exception as e:
            st.error(f"Error during processing: {e}")
            raise


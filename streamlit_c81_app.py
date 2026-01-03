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

# ============================
# Page / Global plot settings
# ============================
st.set_page_config(layout="wide", page_title="AGRC-Surrogate Prediction")

# Make matplotlib text readable
plt.rcParams.update({
    "font.size": 12,
    "axes.titlesize": 16,
    "axes.labelsize": 13,
    "legend.fontsize": 12,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
})

# ============================
# Constants (same defaults as your script; not shown in UI)
# ============================
LOGO_PATH = "AGRC_logo.png"  # put logo in same folder (or edit this)
BASE_DIR = "./data"

MODEL_PT3 = "models/Forward_pt3_non_opt_hyp_v2.h5"
MODEL_PT4 = "models/Forward_pt4_non_opt_hyp_v2.h5"
MODEL_PT5 = "models/Forward_pt5_non_opt_hyp_v2.h5"
MODEL_PT6 = "models/Forward_pt6_non_opt_hyp_v2.h5"
MODEL_PT7 = "models/Forward_pt7_non_opt_hyp_v2.h5"

SCALER_X = "preprocess/scaler_cst_v2.pkl"

# Baseline files (same as your script)
BASELINE_FILES = {
    "Cl": "Cl_360_23012.dat",
    "Cd": "Cd_360_23012.dat",
    "Cm": "Cm_360_23012.dat"
}

# ============================
# Helper functions (UNCHANGED logic)
# ============================
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

# ============================
# Extra helpers (visual + downloads only)
# ============================
def load_pca_scalers(prefix):
    pca_cl = joblib.load(f"preprocess/pca_cl_{prefix}_v2.pkl")
    pca_cd = joblib.load(f"preprocess/pca_cd_{prefix}_v2.pkl")
    pca_cm = joblib.load(f"preprocess/pca_cm_{prefix}_v2.pkl")
    scaler_cl = joblib.load(f"preprocess/scaler_cl_{prefix}_v2.pkl")
    scaler_cd = joblib.load(f"preprocess/scaler_cd_{prefix}_v2.pkl")
    scaler_cm = joblib.load(f"preprocess/scaler_cm_{prefix}_v2.pkl")
    return pca_cl, pca_cd, pca_cm, scaler_cl, scaler_cd, scaler_cm

def split_and_inv(y_pred, pca_cl, pca_cd, pca_cm, s_cl, s_cd, s_cm):
    Y_cl = y_pred[:, 0:10]
    Y_cd = y_pred[:, 10:28]
    Y_cm = y_pred[:, 28:36]
    Cl = s_cl.inverse_transform(pca_cl.inverse_transform(Y_cl))
    Cd = np.exp(s_cd.inverse_transform(pca_cd.inverse_transform(Y_cd)))
    Cm = s_cm.inverse_transform(pca_cm.inverse_transform(Y_cm))
    return Cl.flatten(), Cd.flatten(), Cm.flatten()

def format_c81_all_mach_text(aoa_full, c81_pt3, c81_pt4, c81_pt5, c81_pt6, c81_pt7):
    out = []
    out.append("C81 Table\n\n")
    mach_labels = ["M0.3","M0.4","M0.5","M0.6","M0.7"]

    # CL
    out.append("Cl vs alpha & Mach\n")
    out.append("alpha\t" + "\t".join(mach_labels) + "\n")
    Cl_matrix = np.column_stack([c81_pt3["Cl"].values, c81_pt4["Cl"].values, c81_pt5["Cl"].values, c81_pt6["Cl"].values, c81_pt7["Cl"].values])
    for i, a in enumerate(aoa_full):
        row = [f"{a:6.2f}"] + [f"{Cl_matrix[i, j]:8.4f}" for j in range(5)]
        out.append("\t".join(row) + "\n")
    out.append("\n")

    # CD
    out.append("Cd vs alpha & Mach\n")
    out.append("alpha\t" + "\t".join(mach_labels) + "\n")
    Cd_matrix = np.column_stack([c81_pt3["Cd"].values, c81_pt4["Cd"].values, c81_pt5["Cd"].values, c81_pt6["Cd"].values, c81_pt7["Cd"].values])
    for i, a in enumerate(aoa_full):
        row = [f"{a:6.2f}"] + [f"{Cd_matrix[i, j]:8.4f}" for j in range(5)]
        out.append("\t".join(row) + "\n")
    out.append("\n")

    # CM
    out.append("Cm vs alpha & Mach\n")
    out.append("alpha\t" + "\t".join(mach_labels) + "\n")
    Cm_matrix = np.column_stack([c81_pt3["Cm"].values, c81_pt4["Cm"].values, c81_pt5["Cm"].values, c81_pt6["Cm"].values, c81_pt7["Cm"].values])
    for i, a in enumerate(aoa_full):
        row = [f"{a:6.2f}"] + [f"{Cm_matrix[i, j]:8.4f}" for j in range(5)]
        out.append("\t".join(row) + "\n")

    return "".join(out)

def format_c81_single_mach_text(df, mach):
    out = []
    out.append("C81 Table\n\n")
    out.append(f"Mach = {mach:.1f}\n\n")
    out.append("alpha\tCl\tCd\tCm\n")
    for _, r in df.iterrows():
        out.append(f"{r['AoA']:6.2f}\t{r['Cl']:8.4f}\t{r['Cd']:8.4f}\t{r['Cm']:8.4f}\n")
    return "".join(out)

def plot_geometry_big(xu, yu, xl, yl, popt_upper, popt_lower):
    # Reconstruct from CST (visual only)
    xg = np.linspace(0.0, 1.0, 400)
    yu_rec = CST_chebyshev_TE(xg, *popt_upper)
    yl_rec = CST_chebyshev_TE(xg, *popt_lower)

    fig, ax = plt.subplots(figsize=(12, 3.2), dpi=160)

    # Original (upper+lower) solid green
    ax.plot(xu, yu, linewidth=2.0, linestyle="-", label="Original", color="green")
    ax.plot(xl, yl, linewidth=2.0, linestyle="-", color="green")

    # Reconstructed (upper+lower) dashed red
    ax.plot(xg, yu_rec, linewidth=2.0, linestyle="--", label="Reconstructed", color="red")
    ax.plot(xg, yl_rec, linewidth=2.0, linestyle="--", color="red")

    ax.set_title("Geometry (solid = uploaded, dashed = CST reconstructed)")
    ax.set_xlabel("x/c")
    ax.set_ylabel("y/c")
    ax.grid(True)
    ax.legend(loc="best")
    ax.set_aspect("equal", adjustable="datalim")
    return fig

def plot_polar_big(x, y, title, ylabel):
    fig, ax = plt.subplots(figsize=(5.2, 4.0), dpi=160)
    ax.plot(x, y, linewidth=2.0)
    ax.set_title(title)
    ax.set_xlabel("AoA [deg]")
    ax.set_ylabel(ylabel)
    ax.grid(True)
    return fig

# ============================
# Sidebar (logo + inputs)
# ============================
with st.sidebar:
    st.markdown("## AGRC Surrogate")
    if os.path.exists(LOGO_PATH):
        st.image(LOGO_PATH, use_container_width=True)
    else:
        st.info(f"Logo not found: '{LOGO_PATH}'. Put your logo file there to show it.")
    st.markdown("### Upload Geometry")
    upper_file = st.file_uploader("Upper surface (.dat/.txt)", type=["dat", "txt"])
    lower_file = st.file_uploader("Lower surface (.dat/.txt)", type=["dat", "txt"])

    st.markdown("### Settings")
    mach_list = [0.3, 0.4, 0.5, 0.6, 0.7]
    mach_selected = st.selectbox("Mach", mach_list, index=3)

    run_button = st.button("Run", use_container_width=True)

# ============================
# Main header
# ============================
st.markdown(
    "<h1 style='margin-bottom:0.2rem;'>AGRC-Surrogate Prediction</h1>",
    unsafe_allow_html=True
)

# ============================
# Run pipeline (your logic)
# ============================
if run_button:
    if (upper_file is None) or (lower_file is None):
        st.error("Please upload both upper and lower surface files.")
        st.stop()

    try:
        # read surfaces
        xu, yu = read_airfoil_surface_filelike(upper_file)
        xl, yl = read_airfoil_surface_filelike(lower_file)

        # preprocess / normalize x as in your script
        xu, yu = xu[np.argsort(xu)], yu[np.argsort(xu)]
        xl, yl = xl[np.argsort(xl)], yl[np.argsort(xl)]
        xu = xu / np.max(xu)
        xl = xl / np.max(xl)

        # fit CST (same)
        try:
            popt_upper, _ = curve_fit(CST_chebyshev_TE, xu, yu, p0=np.zeros(10), maxfev=20000)
            popt_lower, _ = curve_fit(CST_chebyshev_TE, xl, yl, p0=np.zeros(10), maxfev=20000)
        except Exception as e:
            st.warning(f"CST fit failed or had issues: {e}\nContinuing with best-effort polyfit fallback.")
            popt_upper = np.polyfit(xu, yu, 9)
            popt_lower = np.polyfit(xl, yl, 9)

        X_input = np.concatenate([popt_upper, popt_lower])[None, :]

        # load models & preprocessors (same defaults, hidden)
        model_pt3 = tf.keras.models.load_model(MODEL_PT3, compile=False)
        model_pt4 = tf.keras.models.load_model(MODEL_PT4, compile=False)
        model_pt5 = tf.keras.models.load_model(MODEL_PT5, compile=False)
        model_pt6 = tf.keras.models.load_model(MODEL_PT6, compile=False)
        model_pt7 = tf.keras.models.load_model(MODEL_PT7, compile=False)
        scaler_x = joblib.load(SCALER_X)

        # scale & predict (same)
        X_scaled = scaler_x.transform(X_input)
        y_pred_pt3 = model_pt3.predict(X_scaled, verbose=0)
        y_pred_pt4 = model_pt4.predict(X_scaled, verbose=0)
        y_pred_pt5 = model_pt5.predict(X_scaled, verbose=0)
        y_pred_pt6 = model_pt6.predict(X_scaled, verbose=0)
        y_pred_pt7 = model_pt7.predict(X_scaled, verbose=0)

        # load PCA/scalers (same)
        pca_cl_pt3, pca_cd_pt3, pca_cm_pt3, scaler_cl_pt3, scaler_cd_pt3, scaler_cm_pt3 = load_pca_scalers("pt3")
        pca_cl_pt4, pca_cd_pt4, pca_cm_pt4, scaler_cl_pt4, scaler_cd_pt4, scaler_cm_pt4 = load_pca_scalers("pt4")
        pca_cl_pt5, pca_cd_pt5, pca_cm_pt5, scaler_cl_pt5, scaler_cd_pt5, scaler_cm_pt5 = load_pca_scalers("pt5")
        pca_cl_pt6, pca_cd_pt6, pca_cm_pt6, scaler_cl_pt6, scaler_cd_pt6, scaler_cm_pt6 = load_pca_scalers("pt6")
        pca_cl_pt7, pca_cd_pt7, pca_cm_pt7, scaler_cl_pt7, scaler_cd_pt7, scaler_cm_pt7 = load_pca_scalers("pt7")

        # inverse (same)
        Cl_pt3, Cd_pt3, Cm_pt3 = split_and_inv(y_pred_pt3, pca_cl_pt3, pca_cd_pt3, pca_cm_pt3, scaler_cl_pt3, scaler_cd_pt3, scaler_cm_pt3)
        Cl_pt4, Cd_pt4, Cm_pt4 = split_and_inv(y_pred_pt4, pca_cl_pt4, pca_cd_pt4, pca_cm_pt4, scaler_cl_pt4, scaler_cd_pt4, scaler_cm_pt4)
        Cl_pt5, Cd_pt5, Cm_pt5 = split_and_inv(y_pred_pt5, pca_cl_pt5, pca_cd_pt5, pca_cm_pt5, scaler_cl_pt5, scaler_cd_pt5, scaler_cm_pt5)
        Cl_pt6, Cd_pt6, Cm_pt6 = split_and_inv(y_pred_pt6, pca_cl_pt6, pca_cd_pt6, pca_cm_pt6, scaler_cl_pt6, scaler_cd_pt6, scaler_cm_pt6)
        Cl_pt7, Cd_pt7, Cm_pt7 = split_and_inv(y_pred_pt7, pca_cl_pt7, pca_cd_pt7, pca_cm_pt7, scaler_cl_pt7, scaler_cd_pt7, scaler_cm_pt7)

        # read baseline tables (same)
        tables = {}
        for key, fname in BASELINE_FILES.items():
            fpath = os.path.join(BASE_DIR, fname)
            tables[key] = read_airfoil_dat(fpath)

        aoa_ann = np.linspace(-10, 20, 31)
        aoa_ann_pt7 = np.linspace(-4, 20, 25)

        # Extract NACA arrays (same)
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

        # Blend (same)
        aoa_full, cl_blend_pt3 = blend_curves(aoa_cl_naca_pt3, cl_naca_pt3, aoa_ann, Cl_pt3)
        _, cd_blend_pt3 = blend_curves(aoa_cd_naca_pt3, cd_naca_pt3, aoa_ann, Cd_pt3)
        _, cm_blend_pt3 = blend_curves(aoa_cm_naca_pt3, cm_naca_pt3, aoa_ann, Cm_pt3)
        cl_blend_pt3 = np.round(cl_blend_pt3, 4); cd_blend_pt3 = np.round(cd_blend_pt3, 4); cm_blend_pt3 = np.round(cm_blend_pt3, 4)

        aoa_full, cl_blend_pt4 = blend_curves(aoa_cl_naca_pt4, cl_naca_pt4, aoa_ann, Cl_pt4)
        _, cd_blend_pt4 = blend_curves(aoa_cd_naca_pt4, cd_naca_pt4, aoa_ann, Cd_pt4)
        _, cm_blend_pt4 = blend_curves(aoa_cm_naca_pt4, cm_naca_pt4, aoa_ann, Cm_pt4)
        cl_blend_pt4 = np.round(cl_blend_pt4, 4); cd_blend_pt4 = np.round(cd_blend_pt4, 4); cm_blend_pt4 = np.round(cm_blend_pt4, 4)

        aoa_full, cl_blend_pt5 = blend_curves(aoa_cl_naca_pt5, cl_naca_pt5, aoa_ann, Cl_pt5)
        _, cd_blend_pt5 = blend_curves(aoa_cd_naca_pt5, cd_naca_pt5, aoa_ann, Cd_pt5)
        _, cm_blend_pt5 = blend_curves(aoa_cm_naca_pt5, cm_naca_pt5, aoa_ann, Cm_pt5)
        cl_blend_pt5 = np.round(cl_blend_pt5, 4); cd_blend_pt5 = np.round(cd_blend_pt5, 4); cm_blend_pt5 = np.round(cm_blend_pt5, 4)

        aoa_full, cl_blend_pt6 = blend_curves(aoa_cl_naca_pt6, cl_naca_pt6, aoa_ann, Cl_pt6)
        _, cd_blend_pt6 = blend_curves(aoa_cd_naca_pt6, cd_naca_pt6, aoa_ann, Cd_pt6)
        _, cm_blend_pt6 = blend_curves(aoa_cm_naca_pt6, cm_naca_pt6, aoa_ann, Cm_pt6)
        cl_blend_pt6 = np.round(cl_blend_pt6, 4); cd_blend_pt6 = np.round(cd_blend_pt6, 4); cm_blend_pt6 = np.round(cm_blend_pt6, 4)

        aoa_full, cl_blend_pt7 = blend_curves_pt7(aoa_cl_naca_pt7, cl_naca_pt7, aoa_ann_pt7, Cl_pt7)
        _, cd_blend_pt7 = blend_curves_pt7(aoa_cd_naca_pt7, cd_naca_pt7, aoa_ann_pt7, Cd_pt7)
        _, cm_blend_pt7 = blend_curves_pt7(aoa_cm_naca_pt7, cm_naca_pt7, aoa_ann_pt7, Cm_pt7)
        cl_blend_pt7 = np.round(cl_blend_pt7, 4); cd_blend_pt7 = np.round(cd_blend_pt7, 4); cm_blend_pt7 = np.round(cm_blend_pt7, 4)

        # Build dataframes (same, including your existing pt4 Cm assignment)
        c81_pt3 = pd.DataFrame({"AoA": aoa_full, "Cl": cl_blend_pt3, "Cd": cd_blend_pt3, "Cm": cm_blend_pt3})
        c81_pt4 = pd.DataFrame({"AoA": aoa_full, "Cl": cl_blend_pt4, "Cd": cd_blend_pt4, "Cm": cd_blend_pt4})
        c81_pt5 = pd.DataFrame({"AoA": aoa_full, "Cl": cl_blend_pt5, "Cd": cd_blend_pt5, "Cm": cm_blend_pt5})
        c81_pt6 = pd.DataFrame({"AoA": aoa_full, "Cl": cl_blend_pt6, "Cd": cd_blend_pt6, "Cm": cm_blend_pt6})
        c81_pt7 = pd.DataFrame({"AoA": aoa_full, "Cl": cl_blend_pt7, "Cd": cd_blend_pt7, "Cm": cm_blend_pt7})

        c81_map = {0.3: c81_pt3, 0.4: c81_pt4, 0.5: c81_pt5, 0.6: c81_pt6, 0.7: c81_pt7}
        selected_df = c81_map[mach_selected]

        # ============================
        # Layout: Row 1 (geometry + downloads)
        # ============================
        top_left, top_right = st.columns([2.2, 1.0], vertical_alignment="top")

        with top_left:
            st.markdown("## Geometry")
            fig_geom = plot_geometry_big(xu, yu, xl, yl, popt_upper, popt_lower)
            st.pyplot(fig_geom, use_container_width=True)

        with top_right:
            st.markdown("## Downloads")

            c81_all_text = format_c81_all_mach_text(aoa_full, c81_pt3, c81_pt4, c81_pt5, c81_pt6, c81_pt7)
            st.download_button(
                "Download C81_all_mach.dat",
                data=c81_all_text,
                file_name="C81_all_mach.dat",
                mime="text/plain",
                use_container_width=True
            )

            c81_single_text = format_c81_single_mach_text(selected_df, mach_selected)
            st.download_button(
                f"Download C81_Mach{mach_selected:.1f}.dat",
                data=c81_single_text,
                file_name=f"C81_Mach{mach_selected:.1f}.dat",
                mime="text/plain",
                use_container_width=True
            )

        st.markdown("---")

        # ============================
        # Layout: Row 2 (3 big plots)
        # ============================
        st.markdown(f"## 360° Polars (Mach {mach_selected:.1f})")

        p1, p2, p3 = st.columns(3)
        with p1:
            st.pyplot(plot_polar_big(selected_df["AoA"], selected_df["Cl"], "Cl vs AoA", "Cl"), use_container_width=True)
        with p2:
            st.pyplot(plot_polar_big(selected_df["AoA"], selected_df["Cd"], "Cd vs AoA", "Cd"), use_container_width=True)
        with p3:
            st.pyplot(plot_polar_big(selected_df["AoA"], selected_df["Cm"], "Cm vs AoA", "Cm"), use_container_width=True)

        # Optional: show table (you can comment this out if you don’t want it)
        st.markdown("## C81 Table (Selected Mach)")
        st.dataframe(selected_df, use_container_width=True)

        st.success("Done.")

    except Exception as e:
        st.error(f"Error during processing: {e}")
        raise
else:
    st.info("Upload upper/lower geometry in the sidebar, select Mach, then click Run.")

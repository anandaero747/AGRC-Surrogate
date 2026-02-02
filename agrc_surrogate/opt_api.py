# agrc_surrogate/opt_api.py
from __future__ import annotations

import numpy as np
from scipy.optimize import curve_fit

# Reuse the exact implementations you already have
from .predict_360_pt3_pt8 import (
    read_airfoil_single_dat_autofix,
    read_airfoil_surface,
    CST_chebyshev_TE,
)

def cst_from_airfoil(
    airfoil: str | None = None,
    upper: str | None = None,
    lower: str | None = None,
    eps: float = 1e-6,
    p0_n: int = 10,
) -> np.ndarray:
    """
    Compute CST Chebyshev coefficients (upper+lower) from coordinates.
    Returns: cst (20,) where first 10 are upper, next 10 are lower.
    """
    if airfoil is None:
        if upper is None or lower is None:
            raise ValueError("Provide either airfoil OR both upper and lower.")
    # 1) Read coords
    if airfoil is not None:
        xU, yU, xL, yL = read_airfoil_single_dat_autofix(airfoil)
    else:
        xU, yU = read_airfoil_surface(upper)
        xL, yL = read_airfoil_surface(lower)

    # 2) Sort by x
    xU, yU = xU[np.argsort(xU)], yU[np.argsort(xU)]
    xL, yL = xL[np.argsort(xL)], yL[np.argsort(xL)]

    # 3) Mask out x=0 (your CST class func has sqrt(x))
    maskU = xU > eps
    maskL = xL > eps
    xU, yU = xU[maskU], yU[maskU]
    xL, yL = xL[maskL], yL[maskL]

    # 4) Fit CST (Chebyshev + TE offset)
    p0 = np.zeros(p0_n)
    popt_upper, _ = curve_fit(CST_chebyshev_TE, xU, yU, p0=p0)
    popt_lower, _ = curve_fit(CST_chebyshev_TE, xL, yL, p0=p0)

    # 5) Return 20-D vector
    cst = np.concatenate([popt_upper, popt_lower]).astype(float)
    return cst
    
def c81_from_cst(
    cst: np.ndarray,
    *,
    # asset paths (defaults work for installed package because your predictor uses asset_path())
    model_pt3=None, model_pt4=None, model_pt5=None, model_pt6=None, model_pt7=None, model_pt8=None,
    scalerx=None,
    pca_cl_pt3=None, pca_cd_pt3=None, pca_cm_pt3=None, scaler_cl_pt3=None, scaler_cd_pt3=None, scaler_cm_pt3=None,
    pca_cl_pt4=None, pca_cd_pt4=None, pca_cm_pt4=None, scaler_cl_pt4=None, scaler_cd_pt4=None, scaler_cm_pt4=None,
    pca_cl_pt5=None, pca_cd_pt5=None, pca_cm_pt5=None, scaler_cl_pt5=None, scaler_cd_pt5=None, scaler_cm_pt5=None,
    pca_cl_pt6=None, pca_cd_pt6=None, pca_cm_pt6=None, scaler_cl_pt6=None, scaler_cd_pt6=None, scaler_cm_pt6=None,
    pca_cl_pt7=None, pca_cd_pt7=None, pca_cm_pt7=None, scaler_cl_pt7=None, scaler_cd_pt7=None, scaler_cm_pt7=None,
    pca_cl_pt8=None, pca_cd_pt8=None, pca_cm_pt8=None, scaler_cl_pt8=None, scaler_cd_pt8=None, scaler_cm_pt8=None,
    # optional: if you want to also write files later
    write_files: bool = False,
    out_prefix: str = "C81",
):
    """
    Evaluate a 20-D CST vector and return blended C81 arrays:
      returns dict with keys: aoa, Cl, Cd, Cm (each shape (361, 8))
      Mach order: [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    """
    import joblib
    import tensorflow as tf
    import pandas as pd
    from pathlib import Path

    # Reuse your exact helpers (and asset_path that works installed/from-source)
    from .predict_360_pt3_pt8 import asset_path, read_airfoil_dat

    cst = np.asarray(cst, dtype=float).reshape(1, -1)
    if cst.shape[1] != 20:
        raise ValueError(f"Expected cst shape (20,), got {cst.shape}")

    # ---- Load assets (simple, not optimized; fine for now)
    def _p(default_rel, override):
        return override if override is not None else asset_path(default_rel)

    m3 = tf.keras.models.load_model(_p("models/Forward_pt3_non_opt_hyp_v2.h5", model_pt3), compile=False)
    m4 = tf.keras.models.load_model(_p("models/Forward_pt4_non_opt_hyp_v2.h5", model_pt4), compile=False)
    m5 = tf.keras.models.load_model(_p("models/Forward_pt5_non_opt_hyp_v2.h5", model_pt5), compile=False)
    m6 = tf.keras.models.load_model(_p("models/Forward_pt6_non_opt_hyp_v2.h5", model_pt6), compile=False)
    m7 = tf.keras.models.load_model(_p("models/Forward_pt7_non_opt_hyp_v2.h5", model_pt7), compile=False)
    m8 = tf.keras.models.load_model(_p("models/Forward_pt8_non_opt_hyp_v2.h5", model_pt8), compile=False)

    sx = joblib.load(_p("preprocess/scaler_cst_v2.pkl", scalerx))

    # triplet loaders
    def load_trip(tag, o_pca_cl, o_pca_cd, o_pca_cm, o_scl, o_scd, o_scm):
        return (
            joblib.load(_p(f"preprocess/pca_cl_{tag}_v2.pkl", o_pca_cl)),
            joblib.load(_p(f"preprocess/pca_cd_{tag}_v2.pkl", o_pca_cd)),
            joblib.load(_p(f"preprocess/pca_cm_{tag}_v2.pkl", o_pca_cm)),
            joblib.load(_p(f"preprocess/scaler_cl_{tag}_v2.pkl", o_scl)),
            joblib.load(_p(f"preprocess/scaler_cd_{tag}_v2.pkl", o_scd)),
            joblib.load(_p(f"preprocess/scaler_cm_{tag}_v2.pkl", o_scm)),
        )

    p3 = load_trip("pt3", pca_cl_pt3, pca_cd_pt3, pca_cm_pt3, scaler_cl_pt3, scaler_cd_pt3, scaler_cm_pt3)
    p4 = load_trip("pt4", pca_cl_pt4, pca_cd_pt4, pca_cm_pt4, scaler_cl_pt4, scaler_cd_pt4, scaler_cm_pt4)
    p5 = load_trip("pt5", pca_cl_pt5, pca_cd_pt5, pca_cm_pt5, scaler_cl_pt5, scaler_cd_pt5, scaler_cm_pt5)
    p6 = load_trip("pt6", pca_cl_pt6, pca_cd_pt6, pca_cm_pt6, scaler_cl_pt6, scaler_cd_pt6, scaler_cm_pt6)
    p7 = load_trip("pt7", pca_cl_pt7, pca_cd_pt7, pca_cm_pt7, scaler_cl_pt7, scaler_cd_pt7, scaler_cm_pt7)
    p8 = load_trip("pt8", pca_cl_pt8, pca_cd_pt8, pca_cm_pt8, scaler_cl_pt8, scaler_cd_pt8, scaler_cm_pt8)

    # ---- Predict
    Xs = sx.transform(cst)

    y3 = m3.predict(Xs, verbose=0)
    y4 = m4.predict(Xs, verbose=0)
    y5 = m5.predict(Xs, verbose=0)
    y6 = m6.predict(Xs, verbose=0)
    y7 = m7.predict(Xs, verbose=0)
    y8 = m8.predict(Xs, verbose=0)

    def inv_trip(y, trip):
        pca_cl, pca_cd, pca_cm, scl_cl, scl_cd, scl_cm = trip
        Ycl = y[:, 0:10]
        Ycd = y[:, 10:28]
        Ycm = y[:, 28:36]
        cl = scl_cl.inverse_transform(pca_cl.inverse_transform(Ycl)).ravel()
        cd = scl_cd.inverse_transform(pca_cd.inverse_transform(Ycd)).ravel()
        cm = scl_cm.inverse_transform(pca_cm.inverse_transform(Ycm)).ravel()
        return cl, cd, cm

    cl3, cd3, cm3 = inv_trip(y3, p3)
    cl4, cd4, cm4 = inv_trip(y4, p4)
    cl5, cd5, cm5 = inv_trip(y5, p5)
    cl6, cd6, cm6 = inv_trip(y6, p6)
    cl7, cd7, cm7 = inv_trip(y7, p7)
    cl8, cd8, cm8 = inv_trip(y8, p8)

    # ---- Read baseline 360 tables (your data files)
    base_dir = Path(asset_path("data"))
    tables = {
        "Cl": read_airfoil_dat(str(base_dir / "Cl_360_23012.dat")),
        "Cd": read_airfoil_dat(str(base_dir / "Cd_360_23012.dat")),
        "Cm": read_airfoil_dat(str(base_dir / "Cm_360_23012.dat")),
    }

    # ---- Blend (same math as your predictor)
    aoa_full = np.linspace(-180, 180, 361)

    def smooth_window(x, x0, w):
        return 0.5 * (1 - np.cos(np.pi * (x - x0 + w) / w))

    def blend(aoa_naca, data_naca, aoa_ann, data_ann, aoa_min, aoa_max, blend_width=10):
        aoa_naca = np.asarray(aoa_naca).ravel()
        data_naca = np.asarray(data_naca).ravel()
        aoa_ann  = np.asarray(aoa_ann).ravel()
        data_ann = np.asarray(data_ann).ravel()

        naca_i = np.interp(aoa_full, aoa_naca, data_naca)
        ann_i  = np.interp(aoa_full, aoa_ann,  data_ann)

        out = naca_i.copy()
        idx = (aoa_full >= aoa_min) & (aoa_full <= aoa_max)
        out[idx] = ann_i[idx]

        low = (aoa_full >= aoa_min - blend_width) & (aoa_full < aoa_min)
        if np.any(low):
            w = smooth_window(aoa_full[low], aoa_min, blend_width)
            out[low] = (1 - w) * naca_i[low] + w * ann_i[low]

        high = (aoa_full > aoa_max) & (aoa_full <= aoa_max + blend_width)
        if np.any(high):
            w = 1 - smooth_window(aoa_full[high], aoa_max, blend_width)
            out[high] = (1 - w) * ann_i[high] + w * naca_i[high]

        return np.round(out, 4)

    # AoA grids for ANN regions (same as your code)
    aoa_ann_03_06 = np.linspace(-10, 20, 31)
    aoa_ann_07_08 = np.linspace(-4, 20, 25)

    # Helper extractors
    def col(df, name): return df[name].values

    # Build matrices for 0.3..0.8
    Cl03 = blend(col(tables["Cl"], "AoA"), col(tables["Cl"], "Mach_0.3"), aoa_ann_03_06, cl3, -10, 20)
    Cd03 = blend(col(tables["Cd"], "AoA"), col(tables["Cd"], "Mach_0.3"), aoa_ann_03_06, cd3, -10, 20)
    Cm03 = blend(col(tables["Cm"], "AoA"), col(tables["Cm"], "Mach_0.3"), aoa_ann_03_06, cm3, -10, 20)

    Cl04 = blend(col(tables["Cl"], "AoA"), col(tables["Cl"], "Mach_0.4"), aoa_ann_03_06, cl4, -10, 20)
    Cd04 = blend(col(tables["Cd"], "AoA"), col(tables["Cd"], "Mach_0.4"), aoa_ann_03_06, cd4, -10, 20)
    Cm04 = blend(col(tables["Cm"], "AoA"), col(tables["Cm"], "Mach_0.4"), aoa_ann_03_06, cm4, -10, 20)

    Cl05 = blend(col(tables["Cl"], "AoA"), col(tables["Cl"], "Mach_0.5"), aoa_ann_03_06, cl5, -10, 20)
    Cd05 = blend(col(tables["Cd"], "AoA"), col(tables["Cd"], "Mach_0.5"), aoa_ann_03_06, cd5, -10, 20)
    Cm05 = blend(col(tables["Cm"], "AoA"), col(tables["Cm"], "Mach_0.5"), aoa_ann_03_06, cm5, -10, 20)

    Cl06 = blend(col(tables["Cl"], "AoA"), col(tables["Cl"], "Mach_0.6"), aoa_ann_03_06, cl6, -10, 20)
    Cd06 = blend(col(tables["Cd"], "AoA"), col(tables["Cd"], "Mach_0.6"), aoa_ann_03_06, cd6, -10, 20)
    Cm06 = blend(col(tables["Cm"], "AoA"), col(tables["Cm"], "Mach_0.6"), aoa_ann_03_06, cm6, -10, 20)

    Cl07 = blend(col(tables["Cl"], "AoA"), col(tables["Cl"], "Mach_0.7"), aoa_ann_07_08, cl7, -4, 20)
    Cd07 = blend(col(tables["Cd"], "AoA"), col(tables["Cd"], "Mach_0.7"), aoa_ann_07_08, cd7, -4, 20)
    Cm07 = blend(col(tables["Cm"], "AoA"), col(tables["Cm"], "Mach_0.7"), aoa_ann_07_08, cm7, -4, 20)

    Cl08 = blend(col(tables["Cl"], "AoA"), col(tables["Cl"], "Mach_0.8"), aoa_ann_07_08, cl8, -4, 20)
    Cd08 = blend(col(tables["Cd"], "AoA"), col(tables["Cd"], "Mach_0.8"), aoa_ann_07_08, cd8, -4, 20)
    Cm08 = blend(col(tables["Cm"], "AoA"), col(tables["Cm"], "Mach_0.8"), aoa_ann_07_08, cm8, -4, 20)

    # 0.1 and 0.2 currently just copy 0.3 (same as your predictor file)
    Cl_mat = np.column_stack([Cl03, Cl03, Cl03, Cl04, Cl05, Cl06, Cl07, Cl08])
    Cd_mat = np.column_stack([Cd03, Cd03, Cd03, Cd04, Cd05, Cd06, Cd07, Cd08])
    Cm_mat = np.column_stack([Cm03, Cm03, Cm03, Cm04, Cm05, Cm06, Cm07, Cm08])

    result = {
        "aoa": aoa_full,
        "mach": np.array([0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8]),
        "Cl": Cl_mat,
        "Cd": Cd_mat,
        "Cm": Cm_mat,
        "cst": cst.ravel(),  # keep it for convenience
    }

    if write_files:
        # Minimal: write same "C81_all_mach.dat" format
        out_file = f"{out_prefix}_all_mach.dat"
        mach_labels = ["M0.1","M0.2","M0.3","M0.4","M0.5","M0.6","M0.7","M0.8"]
        with open(out_file, "w") as f:
            f.write("C81 Table\n\n")

            f.write("Cl vs alpha & Mach\n")
            f.write("alpha\t" + "\t".join(mach_labels) + "\n")
            for i, a in enumerate(aoa_full):
                row = [f"{a:6.2f}"] + [f"{Cl_mat[i,j]:8.4f}" for j in range(8)]
                f.write("\t".join(row) + "\n")
            f.write("\n")

            f.write("Cd vs alpha & Mach\n")
            f.write("alpha\t" + "\t".join(mach_labels) + "\n")
            for i, a in enumerate(aoa_full):
                row = [f"{a:6.2f}"] + [f"{Cd_mat[i,j]:8.4f}" for j in range(8)]
                f.write("\t".join(row) + "\n")
            f.write("\n")

            f.write("Cm vs alpha & Mach\n")
            f.write("alpha\t" + "\t".join(mach_labels) + "\n")
            for i, a in enumerate(aoa_full):
                row = [f"{a:6.2f}"] + [f"{Cm_mat[i,j]:8.4f}" for j in range(8)]
                f.write("\t".join(row) + "\n")

        result["written_file"] = out_file

    return result
    
def interp_at(aoa_grid, y_grid, aoa_query):
    """
    aoa_grid: (N,)
    y_grid:   (N,) or (N,M)
    returns:  scalar or (M,)
    """
    return np.interp(float(aoa_query), aoa_grid, y_grid)

def objective_ld_at_aoa(c81, aoa_target=2.0, mach_target=0.3, eps_cd=1e-6):
    """
    c81: dict from your c81_from_cst() that contains aoa, mach, Cl, Cd, Cm
    returns: scalar objective (maximize L/D => minimize -L/D)
    """
    aoa = c81["aoa"]          # (361,)
    mach = c81["mach"]        # (8,)
    Cl = c81["Cl"]            # (361,8)
    Cd = c81["Cd"]            # (361,8)

    # pick nearest Mach column (simple and robust)
    j = int(np.argmin(np.abs(mach - mach_target)))

    cl = interp_at(aoa, Cl[:, j], aoa_target)
    cd = interp_at(aoa, Cd[:, j], aoa_target)

    ld = cl / max(cd, eps_cd)
    return cd   # GA minimization form



import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

# =============================
# Helper Functions
# =============================
def read_surface(uploaded_file):
    data = np.loadtxt(uploaded_file)
    return data[:, 0], data[:, 1]


def plot_airfoil(xu, yu, xl, yl):
    fig, ax = plt.subplots()
    ax.plot(xu, yu, label="Upper Surface")
    ax.plot(xl, yl, label="Lower Surface")
    ax.set_aspect('equal')
    ax.grid(True)
    ax.legend()
    return fig


def cst_parametrize(x, y):
    # Placeholder for CST method
    # Return dummy coefficients for now
    return np.polyfit(x, y, 5)


def generate_c81():
    aoa = np.linspace(-10, 20, 31)
    cl = 0.1 * aoa
    cd = 0.01 + 0.002 * aoa**2
    cm = -0.05 * aoa
    return np.column_stack([aoa, cl, cd, cm])


# =============================
# Streamlit UI
# =============================
st.title("Airfoil C81 Generator Web App")
st.markdown("Upload airfoil surfaces, parametrize them, visualize, and generate C81.")

uploaded_upper = st.file_uploader("Upload Upper Surface File", type=["txt", "dat"])
uploaded_lower = st.file_uploader("Upload Lower Surface File", type=["txt", "dat"])

if uploaded_upper and uploaded_lower:
    xu, yu = read_surface(uploaded_upper)
    xl, yl = read_surface(uploaded_lower)

    st.subheader("Airfoil Plot")
    fig = plot_airfoil(xu, yu, xl, yl)
    st.pyplot(fig)

    st.subheader("Parametrization (CST Coefficients)")
    coeffs_upper = cst_parametrize(xu, yu)
    coeffs_lower = cst_parametrize(xl, yl)

    st.write("Upper Surface CST Coefficients:", coeffs_upper)
    st.write("Lower Surface CST Coefficients:", coeffs_lower)

    st.subheader("Generate C81 File")
    c81 = generate_c81()
    st.dataframe(c81)

    c81_text = "".join([f"{row[0]:.2f} {row[1]:.4f} {row[2]:.4f} {row[3]:.4f}\n" for row in c81])

    st.download_button(
        label="Download C81 File",
        data=c81_text,
        file_name="airfoil.c81",
        mime="text/plain"
    )

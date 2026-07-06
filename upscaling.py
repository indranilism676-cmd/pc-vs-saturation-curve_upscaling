import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from scipy.interpolate import interp1d
from catboost import CatBoostRegressor


st.set_page_config(page_title="Capillary Pressure Upscaling",page_icon="🛢️",layout="wide")

st.title("🛢️ Capillary Pressure Upscaling using Machine Learning")

st.markdown(
"""


## Overview
The Leverett J-function is one of the most widely used dimensionless relationships in petroleum engineering for capillary pressure upscaling. By normalizing capillary pressure with rock properties such as porosity and permeability, it enables laboratory-measured core capillary pressure curves to be transformed into representative reservoir-scale curves. This process is fundamental for reservoir characterization, reserve estimation, and reservoir simulation. However, applying the Leverett J-function manually to thousands of capillary pressure curves can be repetitive and time-intensive. In this project, I first implemented the traditional physics-based upscaling workflow in Python to generate a large dataset and then trained a machine learning model to learn the underlying scaling relationship. The result is an automated workflow that can generate accurate upscaled capillary pressure curves within seconds, significantly reducing processing time while remaining grounded in established reservoir engineering principles.



# Advantages

Compared with traditional manual upscaling

- Faster
- Automated
- Consistent
- Suitable for large datasets
- Easily integrated into reservoir workflows

---


---

# Disclaimer

This application is intended for educational and research purposes.

Predictions should be validated before use in field development or reservoir management decisions.

---

# Developer

Developed by **Indranil**

"""
)



@st.cache_resource
def load_model():

    model = CatBoostRegressor()

    model.load_model("ScalingFactorCatBoost4.cbm")

    return model

model = load_model()


st.sidebar.header("Rock Properties")

phi_core = st.sidebar.number_input(
    "Core Porosity",
    min_value=0.0001,
    value=0.18,
    format="%.4f"
)

k_core = st.sidebar.number_input(
    "Core Permeability (mD)",
    min_value=0.0001,
    value=100.0
)

phi_res = st.sidebar.number_input(
    "Reservoir Porosity",
    min_value=0.0001,
    value=0.25,
    format="%.4f"
)

k_res = st.sidebar.number_input(
    "Reservoir Permeability (mD)",
    min_value=0.0001,
    value=500.0
)


uploaded_file = st.file_uploader(
    "Upload Capillary Pressure CSV",
    type=["csv"]
)


if uploaded_file is not None:

    try:

        try:
            curve = pd.read_csv(uploaded_file)
        except UnicodeDecodeError:
            uploaded_file.seek(0)
            curve = pd.read_csv(uploaded_file, encoding="latin1")

        required = ["Sw", "Core_Pc"]

        if not all(col in curve.columns for col in required):
            st.error("CSV must contain columns: Sw and Core_Pc")
            st.stop()

        curve = curve[["Sw", "Core_Pc"]]

        curve = curve.replace(
            [np.inf, -np.inf],
            np.nan
        )

        curve = curve.dropna()

        curve = curve.sort_values("Sw")

        curve = curve.drop_duplicates(subset="Sw")

        curve.reset_index(drop=True, inplace=True)

        if len(curve) < 2:
            st.error("At least two data points are required.")
            st.stop()

        st.subheader("Uploaded Data")

        st.dataframe(curve)

        if st.button("Predict Upscaled Curve"):

            with st.spinner("Predicting..."):

                common_sw = np.linspace(
                    curve["Sw"].min(),
                    curve["Sw"].max(),
                    50
                )

                interpolation = interp1d(
                    curve["Sw"],
                    curve["Core_Pc"],
                    kind="linear",
                    bounds_error=False,
                    fill_value="extrapolate"
                )

                core_pc_interp = interpolation(common_sw)

                core_pc_norm = core_pc_interp / core_pc_interp[0]

                features = np.concatenate([
                    core_pc_norm,
                    [
                        phi_core,
                        k_core,
                        phi_res,
                        k_res
                    ]
                ])

                features = features.reshape(1, -1)

                log_scale = model.predict(features)[0]

                scale = np.exp(log_scale)

                curve["Upscaled_Pc"] = (
                    curve["Core_Pc"] * scale
                )

                st.metric(
                    "Predicted Scaling Factor",
                    f"{scale:.4f}"
                )

                fig, ax = plt.subplots(figsize=(8,6))

                ax.plot(
                    curve["Sw"],
                    curve["Core_Pc"],
                    linewidth=2,
                    label="Core Curve"
                )

                ax.plot(
                    curve["Sw"],
                    curve["Upscaled_Pc"],
                    linewidth=2,
                    label="Predicted Upscaled Curve"
                )

                ax.set_xlabel("Water Saturation")

                ax.set_ylabel("Capillary Pressure")

                ax.set_title("Capillary Pressure Upscaling")

                ax.grid(True)

                ax.legend()

                st.pyplot(fig)

                st.subheader("Predicted Data")

                st.dataframe(curve)

                csv = curve.to_csv(
                    index=False
                ).encode("utf-8")

                st.download_button(
                    "Download Upscaled Curve",
                    csv,
                    "upscaled_curve.csv",
                    "text/csv"
                )

    except Exception as e:

        st.error(f"Error while processing file:\n\n{e}")

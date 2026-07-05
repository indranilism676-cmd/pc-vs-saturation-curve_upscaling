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
# 🛢️ Capillary Pressure Curve Upscaling using Machine Learning

## Overview

This application predicts the **upscaled capillary pressure curve** for a reservoir using a machine learning model trained on the **Leverett J-function** scaling concept.

The application takes a laboratory-measured **core capillary pressure curve** together with the core and reservoir rock properties, predicts the **Leverett scaling factor** using a trained CatBoost model, and generates the corresponding **reservoir-scale capillary pressure curve**.

---

# Why is Capillary Pressure Upscaling Necessary?

Capillary pressure measurements are usually obtained from **small laboratory core plugs**. These measurements accurately represent the rock sample but **cannot be directly applied to an entire reservoir** because reservoir rocks often have:

- Different porosity
- Different permeability
- Different pore geometry
- Larger representative volume

Consequently, laboratory capillary pressure curves must be **upscaled** before being used in reservoir simulation, history matching, and reservoir characterization.

---

# What is Capillary Pressure?

Capillary pressure is the pressure difference between two immiscible fluids occupying the pore space.

\[
P_c=P_{non-wetting}-P_{wetting}
\]

where

- \(P_c\) = Capillary Pressure
- Non-wetting phase = Oil or Gas
- Wetting phase = Water

Capillary pressure controls

- Fluid distribution
- Initial water saturation
- Hydrocarbon column height
- Relative permeability
- Reservoir recovery

---

# What is the Leverett J-Function?

The Leverett J-function is a **dimensionless function** developed to normalize capillary pressure measurements from rocks having different porosity and permeability.

Instead of comparing capillary pressure directly, the J-function removes the effect of rock properties.

The mathematical expression is

\[
J(S_w)=\frac{P_c\sqrt{k/\phi}}{\sigma\cos\theta}
\]

where

- \(J(S_w)\) = Leverett J-function
- \(P_c\) = Capillary Pressure
- \(k\) = Permeability
- \(\phi\) = Porosity
- \(\sigma\) = Interfacial Tension
- \(\theta\) = Contact Angle

The J-function depends mainly on **water saturation** rather than absolute rock properties.

---

# Why is the J-Function Important?

The Leverett J-function allows engineers to

- Compare capillary pressure curves from different rocks
- Predict reservoir-scale capillary pressure
- Perform capillary pressure upscaling
- Estimate irreducible water saturation
- Construct reservoir simulation input

It is one of the most widely used scaling methods in petroleum engineering.

---

# Traditional Upscaling Workflow

Traditionally, engineers compute

1. Core capillary pressure
2. Core porosity
3. Core permeability
4. Reservoir porosity
5. Reservoir permeability

Then calculate

\[
P_{c,res}=P_{c,core}
\sqrt{
\frac{k_{core}\phi_{res}}
{k_{res}\phi_{core}}
}
\]

This procedure is repeated for every saturation point.

Although physically meaningful, it becomes computationally expensive when processing thousands of curves.

---

# Machine Learning Approach

Instead of directly predicting capillary pressure, this application predicts the **Leverett scaling factor**.

The scaling factor is

\[
Scaling\ Factor=
\frac{P_{c,reservoir}}
{P_{c,core}}
\]

The CatBoost model learns the relationship

```
Core Capillary Curve
+
Core Rock Properties
+
Reservoir Rock Properties

↓

Predicted Scaling Factor
```

The upscaled curve is then computed as

\[
P_{c,predicted}
=
P_{c,core}
\times
Scaling\ Factor
\]

This preserves the original capillary pressure curve while adjusting it to the target reservoir.

---

# Machine Learning Model

The application uses a **CatBoost Regressor**.

CatBoost is a gradient boosting algorithm particularly effective for structured engineering datasets because it

- Handles nonlinear relationships
- Requires minimal preprocessing
- Produces excellent prediction accuracy
- Is resistant to overfitting
- Works well with relatively small datasets

---

# Input Requirements

Upload a CSV file containing exactly two columns

| Column | Description |
|---------|-------------|
| Sw | Water Saturation |
| Core_Pc | Core Capillary Pressure |

Example

| Sw | Core_Pc |
|----|---------|
|1.00|5|
|0.95|8|
|0.90|15|

---

# Additional Inputs

The user must provide

### Core Properties

- Core Porosity
- Core Permeability

### Reservoir Properties

- Reservoir Porosity
- Reservoir Permeability

---

# Workflow

The application performs the following steps automatically.

## Step 1

Upload the core capillary pressure curve.

↓

## Step 2

Clean the data

- Remove missing values
- Remove duplicate saturation values
- Sort saturation

↓

## Step 3

Interpolate the curve to a standard 50-point representation.

↓

## Step 4

Normalize the curve.

↓

## Step 5

Construct the feature vector

- Normalized capillary pressure
- Core porosity
- Core permeability
- Reservoir porosity
- Reservoir permeability

↓

## Step 6

Predict the logarithm of the scaling factor using CatBoost.

↓

## Step 7

Convert back

\[
Scaling=e^{Prediction}
\]

↓

## Step 8

Generate the upscaled capillary pressure curve.

↓

## Step 9

Visualize the results.

↓

## Step 10

Download the predicted curve as a CSV file.

---

# Output

The application provides

- Predicted Leverett Scaling Factor
- Upscaled Capillary Pressure Curve
- Interactive Plot
- Downloadable CSV

---

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

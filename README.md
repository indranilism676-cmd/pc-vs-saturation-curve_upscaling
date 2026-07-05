# 🛢️ Capillary Pressure Curve Upscaling using Machine Learning

A Streamlit application that predicts **reservoir-scale capillary pressure curves** from laboratory-measured core capillary pressure data using a **CatBoost Machine Learning model** trained on the **Leverett J-function scaling principle**.

---

## 📌 Project Overview

Capillary pressure curves measured in the laboratory represent only a small core plug and cannot be directly applied to reservoir-scale simulations because reservoir rocks generally have different porosity and permeability.

Traditionally, engineers upscale these curves using the **Leverett J-function**. This project accelerates that workflow by training a machine learning model to predict the **Leverett scaling factor**, enabling rapid generation of reservoir-scale capillary pressure curves.

The application provides an easy-to-use interface where users upload a core capillary pressure curve, enter rock properties, and obtain the predicted upscaled curve instantly.

---

## ✨ Features

- 📂 Upload capillary pressure data in CSV format
- 🧹 Automatic data cleaning
- 📈 Automatic interpolation to a standard representation
- 🤖 CatBoost-based prediction of the Leverett scaling factor
- 📊 Visualization of:
  - Original Core Capillary Pressure Curve
  - Predicted Upscaled Capillary Pressure Curve
- 📥 Download predicted upscaled curve as CSV
- 🖥️ Interactive Streamlit interface

---

## 📁 Input File Format

The uploaded CSV must contain exactly **two columns**:

| Column | Description |
|---------|-------------|
| Sw | Water Saturation |
| Core_Pc | Core Capillary Pressure |

Example:

```csv
Sw,Core_Pc
1.00,5.00
0.98,5.45
0.96,6.20
0.94,7.10
...
```

---

## 📥 User Inputs

The application also requires:

### Core Rock Properties

- Core Porosity
- Core Permeability (mD)

### Reservoir Rock Properties

- Reservoir Porosity
- Reservoir Permeability (mD)

---

## ⚙️ Workflow

The application performs the following steps automatically:

1. Upload the core capillary pressure curve.
2. Validate the uploaded CSV.
3. Remove missing and duplicate values.
4. Sort the curve by water saturation.
5. Interpolate the curve to a fixed number of saturation points.
6. Normalize the capillary pressure curve.
7. Construct the machine learning feature vector.
8. Predict the logarithm of the Leverett scaling factor.
9. Convert the prediction back to the actual scaling factor.
10. Generate the predicted reservoir capillary pressure curve.
11. Display plots and allow CSV download.

---

## 🧠 Machine Learning Model

The application uses a **CatBoost Regressor**.

### Why CatBoost?

- Excellent performance on structured/tabular data
- Handles nonlinear relationships effectively
- Minimal preprocessing required
- Robust against overfitting
- High prediction accuracy

The model predicts

\[
\log(\text{Scaling Factor})
\]

instead of the scaling factor directly, improving numerical stability during training.

The final scaling factor is recovered using

\[
\text{Scaling Factor}=e^{\text{Prediction}}
\]

The predicted capillary pressure curve is computed as

\[
P_{c,\;predicted}
=
P_{c,\;core}
\times
\text{Scaling Factor}
\]

---

## 📊 Leverett J-Function

The Leverett J-function is a dimensionless relationship used to normalize capillary pressure curves obtained from rocks having different porosity and permeability.

It is defined as

\[
J(S_w)=
\frac{P_c\sqrt{k/\phi}}
{\sigma\cos\theta}
\]

where

- \(P_c\) = Capillary Pressure
- \(k\) = Permeability
- \(\phi\) = Porosity
- \(\sigma\) = Interfacial Tension
- \(\theta\) = Contact Angle

The J-function enables capillary pressure scaling between rocks of different properties and forms the physical basis of this project.




## 📂 Project Structure

```
Capillary-Pressure-Upscaling/
│
├── app.py
├── catboost_model.cbm
├── requirements.txt
├── README.md
└── sample_input.csv
```

---

## 📈 Outputs

The application generates:

- Predicted Leverett Scaling Factor
- Reservoir-scale Capillary Pressure Curve
- Interactive Plot
- Downloadable CSV file

---

## 🛠️ Technologies Used

- Python
- Streamlit
- CatBoost
- NumPy
- Pandas
- SciPy
- Matplotlib
- Scikit-learn



## 📖 Applications

This project can be applied in:

- Reservoir Characterization
- Reservoir Simulation
- Capillary Pressure Upscaling
- Digital Rock Physics
- Machine Learning in Petroleum Engineering
- Reservoir Property Prediction

---

## ⚠️ Disclaimer

This application is intended for educational and research purposes. Predictions should be validated before being used in engineering design or reservoir development decisions.

---

## 👨‍💻 Author

**Indranil**


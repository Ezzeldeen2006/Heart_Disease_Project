import streamlit as st
import pandas as pd
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pyngrok import ngrok

# Page Configuration
st.set_page_config(page_title="Heart Disease Prediction", page_icon="❤️", layout="wide")


# Load Assets
@st.cache_resource
def load_model():
    """Load the trained machine learning pipeline."""
    # Try multiple possible paths
    possible_paths = [
        "models/final_model.pkl",
        "../models/final_model.pkl",
        "./models/final_model.pkl",
        "Heart_Disease_Project/models/final_model.pkl",
    ]

    for model_path in possible_paths:
        if os.path.exists(model_path):
            try:
                return joblib.load(model_path)
            except Exception as e:
                st.error(f"Error loading model from {model_path}: {e}")

    st.error("Model file not found. Please check the file path.")
    st.info(f"Current working directory: {os.getcwd()}")
    st.info("Looking for model in these locations:")
    for path in possible_paths:
        st.info(f"  - {path} (exists: {os.path.exists(path)})")
    return None


@st.cache_data
def load_data(filename):
    """Load a CSV file, trying multiple possible locations."""
    possible_paths = [
        f"data/{filename}",
        f"../data/{filename}",
        f"./{filename}",
        f"Heart_Disease_Project/data/{filename}",
    ]

    for file_path in possible_paths:
        if os.path.exists(file_path):
            try:
                return pd.read_csv(file_path)
            except Exception as e:
                st.error(f"Error loading data from {file_path}: {e}")

    st.warning(f"Data file {filename} not found in expected locations.")
    return None


# Load the necessary assets
pipeline = load_model()

# Load raw heart disease data with proper column names
raw_heart_df = None
for filename in ["heart_disease.csv", "heart_disease_raw.csv"]:
    raw_data = load_data(filename)
    if raw_data is not None:
        if raw_data.shape[1] == 14:  # Expected number of columns
            column_names = [
                "age",
                "sex",
                "cp",
                "trestbps",
                "chol",
                "fbs",
                "restecg",
                "thalach",
                "exang",
                "oldpeak",
                "slope",
                "ca",
                "thal",
                "num",
            ]
            raw_data.columns = column_names
            raw_data = raw_data.replace("?", np.nan)
            raw_data["target"] = raw_data["num"].apply(
                lambda x: 1 if pd.notna(x) and x > 0 else 0
            )
            raw_heart_df = raw_data
            break

# Load tuning results
tuning_results_df = load_data("hyperparameter_tuning_comparison.csv")

# Define the exact 12 features your model uses
MODEL_FEATURES = [
    "thal_7.0",
    "cp_4.0",
    "exang_1.0",
    "thalach",
    "oldpeak",
    "slope_2.0",
    "ca_2.0",
    "ca_1.0",
    "cp_2.0",
    "cp_3.0",
    "sex_1.0",
    "trestbps",
]


def validate_inputs(**inputs):
    """Validate user inputs for realistic medical values."""
    errors = []

    if inputs["trestbps"] < 80 or inputs["trestbps"] > 250:
        errors.append("Resting blood pressure should be between 80-250 mmHg")

    if inputs["thalach"] < 60 or inputs["thalach"] > 220:
        errors.append("Maximum heart rate should be between 60-220 bpm")

    if inputs["oldpeak"] < 0 or inputs["oldpeak"] > 10:
        errors.append("ST depression should be between 0-10")

    if inputs.get("age", 0) < 0 or inputs.get("age", 0) > 120:
        errors.append("Age should be between 0-120 years")

    if inputs.get("chol", 0) < 0 or inputs.get("chol", 0) > 1000:
        errors.append("Cholesterol should be between 0-1000 mg/dl")

    return errors


def create_model_input(raw_inputs):
    """Convert raw user inputs to the exact format expected by the model."""
    # Initialize all features to 0
    model_input = pd.DataFrame(0, index=[0], columns=MODEL_FEATURES)

    # Set continuous features directly
    continuous_features = ["thalach", "oldpeak", "trestbps"]
    for feature in continuous_features:
        if feature in raw_inputs:
            model_input[feature] = raw_inputs[feature]

    # Handle binary and categorical features
    if raw_inputs.get("sex") == 1:  # Male
        model_input["sex_1.0"] = 1

    if raw_inputs.get("exang") == 1:  # Exercise induced angina
        model_input["exang_1.0"] = 1

    # Handle chest pain types (cp)
    cp_value = raw_inputs.get("cp", 1)
    cp_column = f"cp_{float(cp_value)}"
    if cp_column in MODEL_FEATURES:
        model_input[cp_column] = 1

    # Handle slope
    slope_value = raw_inputs.get("slope", 1)
    if slope_value == 2:  # Flat slope
        model_input["slope_2.0"] = 1

    # Handle ca (number of major vessels)
    ca_value = raw_inputs.get("ca", 0)
    if ca_value == 1 and "ca_1.0" in MODEL_FEATURES:
        model_input["ca_1.0"] = 1
    elif ca_value == 2 and "ca_2.0" in MODEL_FEATURES:
        model_input["ca_2.0"] = 1

    # Handle thalassemia
    thal_value = raw_inputs.get("thal", 3)
    if thal_value == 7 and "thal_7.0" in MODEL_FEATURES:
        model_input["thal_7.0"] = 1

    return model_input


# Main Application
st.title("Heart Disease Prediction Dashboard")
st.markdown("Enter patient details below to predict the likelihood of heart disease.")

if pipeline is None:
    st.error(
        "Cannot make predictions without the trained model. Please check file paths."
    )
    st.stop()

# User Input Section
st.header("Enter Patient Data")

# Create input validation info
with st.expander("Input Guidelines", expanded=False):
    st.markdown("""
    **Normal Ranges:**
    - Resting Blood Pressure: 90-140 mmHg (normal), >140 (high)
    - Max Heart Rate: ~220 - age (rough estimate)
    - ST Depression: 0-2 (normal), >2 (concerning)
    """)

col1, col2, col3 = st.columns(3)

with col1:
    age = st.slider("Age", 20, 100, 50)
    sex = st.selectbox("Sex", ("Female", "Male"))
    cp = st.selectbox(
        "Chest Pain Type",
        (1, 2, 3, 4),
        format_func=lambda x: {
            1: "Type 1: Typical Angina",
            2: "Type 2: Atypical Angina",
            3: "Type 3: Non-Anginal Pain",
            4: "Type 4: Asymptomatic",
        }[x],
    )
    exang = st.selectbox("Exercise Induced Angina", ("No", "Yes"))

with col2:
    trestbps = st.slider("Resting Blood Pressure (mmHg)", 80, 250, 120)
    chol = st.slider("Serum Cholesterol (mg/dl)", 100, 600, 200)
    thalach = st.slider("Maximum Heart Rate Achieved", 60, 220, 150)
    oldpeak = st.slider("ST Depression", 0.0, 10.0, 1.0, step=0.1)

with col3:
    fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", ("No", "Yes"))
    restecg = st.selectbox(
        "Resting ECG",
        (0, 1, 2),
        format_func=lambda x: {
            0: "Normal",
            1: "ST-T Wave Abnormality",
            2: "Left Ventricular Hypertrophy",
        }[x],
    )
    slope = st.selectbox(
        "Slope of Peak Exercise ST Segment",
        (1, 2, 3),
        format_func=lambda x: {1: "Upsloping", 2: "Flat", 3: "Downsloping"}[x],
    )
    ca = st.selectbox("Number of Major Vessels (0-3)", (0, 1, 2, 3))
    thal = st.selectbox(
        "Thalassemia",
        (3, 6, 7),
        format_func=lambda x: {3: "Normal", 6: "Fixed Defect", 7: "Reversible Defect"}[
            x
        ],
    )

# Prediction Logic
if st.button("Predict Heart Disease Risk", type="primary"):
    # Collect raw inputs
    raw_inputs = {
        "age": age,
        "sex": 1 if sex == "Male" else 0,
        "cp": cp,
        "trestbps": trestbps,
        "chol": chol,
        "fbs": 1 if fbs == "Yes" else 0,
        "restecg": restecg,
        "thalach": thalach,
        "exang": 1 if exang == "Yes" else 0,
        "oldpeak": oldpeak,
        "slope": slope,
        "ca": ca,
        "thal": thal,
    }

    # Validate inputs
    validation_errors = validate_inputs(**raw_inputs)
    if validation_errors:
        st.warning("Input validation warnings:")
        for error in validation_errors:
            st.warning(f"- {error}")
        st.info("You can still proceed, but please verify the values are correct.")

    # Convert to model format
    model_input = create_model_input(raw_inputs)

    # Debug: Show what the model receives
    with st.expander("Model Input Debug", expanded=False):
        st.write("Raw inputs:", raw_inputs)
        st.write("Model input shape:", model_input.shape)
        st.dataframe(model_input)
        st.write("Non-zero features:")
        non_zero = model_input.loc[0][model_input.loc[0] != 0]
        st.write(non_zero.to_dict())

    try:
        # Make prediction
        prediction = pipeline.predict(model_input)[0]
        prediction_proba = pipeline.predict_proba(model_input)[0]

        # Display results
        st.markdown("---")
        st.header("Prediction Result")

        col_result1, col_result2 = st.columns(2)

        with col_result1:
            if prediction == 1:
                st.error(f"**High Risk of Heart Disease**")
                st.markdown(f"**Probability:** {prediction_proba[1] * 100:.1f}%")
                st.warning(
                    "⚠️ Please consult a medical professional for further evaluation."
                )
            else:
                st.success(f"**Low Risk of Heart Disease**")
                st.markdown(f"**Probability:** {prediction_proba[0] * 100:.1f}%")
                st.info("✅ Continue maintaining a healthy lifestyle.")

        with col_result2:
            # Risk gauge
            risk_score = prediction_proba[1] * 100
            if risk_score < 25:
                risk_level = "Very Low"
                color = "green"
            elif risk_score < 50:
                risk_level = "Low"
                color = "lightgreen"
            elif risk_score < 75:
                risk_level = "Moderate"
                color = "orange"
            else:
                risk_level = "High"
                color = "red"

            st.metric(
                label="Risk Level",
                value=f"{risk_level} ({risk_score:.1f}%)",
                delta=f"{risk_score - 50:.1f}% from average",
            )

    except Exception as e:
        st.error(f"Prediction error: {str(e)}")
        st.error("This might indicate a problem with the model or input format.")

# Visualization Section
with st.expander("Explore Data and Model Performance", expanded=False):
    tab1, tab2 = st.tabs(["Data Insights", "Model Performance"])

    with tab1:
        st.subheader("Dataset Insights")
        if raw_heart_df is not None:
            col_viz1, col_viz2 = st.columns(2)

            with col_viz1:
                fig1, ax1 = plt.subplots(figsize=(8, 6))
                sns.histplot(
                    data=raw_heart_df,
                    x="age",
                    hue="target",
                    multiple="stack",
                    kde=True,
                    ax=ax1,
                )
                ax1.set_title("Age Distribution by Heart Disease Status")
                st.pyplot(fig1)

            with col_viz2:
                fig2, ax2 = plt.subplots(figsize=(8, 6))
                sns.countplot(data=raw_heart_df, x="cp", hue="target", ax=ax2)
                ax2.set_title("Chest Pain Type vs Heart Disease")
                st.pyplot(fig2)

            # Additional insights
            st.markdown("**Dataset Statistics:**")
            st.write(f"- Total patients: {len(raw_heart_df)}")
            st.write(
                f"- Heart disease prevalence: {raw_heart_df['target'].mean() * 100:.1f}%"
            )
            st.write(f"- Average age: {raw_heart_df['age'].mean():.1f} years")
        else:
            st.write("Raw data not available for visualization.")

    with tab2:
        st.subheader("Model Performance Comparison")
        if tuning_results_df is not None:
            fig3, ax3 = plt.subplots(figsize=(10, 6))
            sns.barplot(
                data=tuning_results_df, x="Model", y="AUC", hue="Method", ax=ax3
            )
            ax3.set_title("AUC Comparison of Tuned Models")
            ax3.set_ylabel("Area Under Curve (AUC)")
            plt.xticks(rotation=45)
            st.pyplot(fig3)

            # Show best model
            best_model = tuning_results_df.loc[tuning_results_df["AUC"].idxmax()]
            st.success(
                f"Best Model: {best_model['Model']} ({best_model['Method']}) - AUC: {best_model['AUC']:.4f}"
            )
        else:
            st.write("Model performance data not available.")

# Footer
st.markdown("---")
st.markdown(
    "**Disclaimer:** This tool is for educational purposes only and should not replace professional medical advice."
)
public_url = ngrok.connect(8501)
st.sidebar.success(f"Public app URL: {public_url}")

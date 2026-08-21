import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt

from utils import preprocess_data

st.set_page_config(
    page_title="Student Performance Prediction",
    page_icon="🎓",
    layout="wide"
)

st.title("🎓 Student Performance Prediction System")

st.write(
    """
Upload the original **student-mat.csv** dataset.

The application automatically preprocesses the data,
predicts whether each student will PASS or FAIL,
and allows you to download the results.
"""
)

model = joblib.load("../models/best_model.pkl")

uploaded_file = st.file_uploader(
    "Upload CSV File",
    type=["csv"]
)

if uploaded_file is not None:

    raw_df = pd.read_csv(uploaded_file, sep=";")

    st.subheader("Original Dataset")

    st.dataframe(raw_df.head())

    X = preprocess_data(raw_df)

    predictions = model.predict(X)

    probabilities = model.predict_proba(X)

    results = raw_df.copy()

    results["Prediction"] = predictions

    results["Prediction"] = results["Prediction"].map(
        {
            1: "PASS",
            0: "FAIL"
        }
    )

    results["Probability"] = probabilities.max(axis=1)

    st.subheader("Prediction Results")

    st.dataframe(results)

    st.subheader("Prediction Summary")

    summary = results["Prediction"].value_counts()

    st.write(summary)

    fig, ax = plt.subplots(figsize=(6,4))

    ax.bar(summary.index, summary.values)

    ax.set_title("Prediction Distribution")

    st.pyplot(fig)

    csv = results.to_csv(index=False).encode("utf-8")

    st.download_button(
        "Download Predictions",
        csv,
        "student_predictions.csv",
        "text/csv"
    )
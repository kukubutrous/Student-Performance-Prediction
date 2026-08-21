import streamlit as st
import pandas as pd
import joblib

model = joblib.load("../models/best_model.pkl")

st.title("Student Performance Prediction")

uploaded = st.file_uploader(
    "Upload Student CSV",
    type=["csv"]
)

if uploaded:

    df = pd.read_csv(uploaded)

    prediction = model.predict(df)

    df["Prediction"] = prediction

    df["Prediction"] = df["Prediction"].map({

        1:"PASS",

        0:"FAIL"

    })

    st.write(df)

    csv = df.to_csv(index=False)

    st.download_button(

        "Download Predictions",

        csv,

        "predictions.csv",

        "text/csv"

    )
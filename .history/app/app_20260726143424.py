import streamlit as st
import pandas as pd
import joblib
from utils import preprocess

# ----------------------------
# PAGE CONFIGURATION
# ----------------------------
st.set_page_config(
    page_title="Student Performance Prediction",
    page_icon="🎓",
    layout="wide"
)

# ----------------------------
# CUSTOM CSS
# ----------------------------
st.markdown("""
<style>

.main{
    background-color:#f7f9fc;
}

h1{
    color:#0f62fe;
}

.block-container{
    padding-top:2rem;
}

.stButton>button{
    width:100%;
    height:50px;
    font-size:18px;
    border-radius:10px;
    background:#0f62fe;
    color:white;
}

.result-pass{
    padding:20px;
    border-radius:10px;
    background:#d4edda;
    color:#155724;
    font-size:28px;
    font-weight:bold;
    text-align:center;
}

.result-fail{
    padding:20px;
    border-radius:10px;
    background:#f8d7da;
    color:#721c24;
    font-size:28px;
    font-weight:bold;
    text-align:center;
}

</style>
""", unsafe_allow_html=True)

# ----------------------------
# LOAD MODEL
# ----------------------------

model = joblib.load("../models/student_model.pkl")
scaler = joblib.load("../models/scaler.pkl")

# ----------------------------
# TITLE
# ----------------------------

st.title("🎓 Student Performance Prediction System")

st.write("""
Predict whether a student is likely to **PASS** or **FAIL**
using Machine Learning.
""")

st.divider()

left,right = st.columns(2)

# =====================================
# LEFT COLUMN
# =====================================

with left:

    st.subheader("👤 Personal Information")

    age = st.number_input(
        "Age",
        min_value=15,
        max_value=22,
        value=17
    )

    sex = st.selectbox(
        "Gender",
        ["F","M"]
    )

    address = st.selectbox(
        "Address",
        ["U","R"]
    )

    famsize = st.selectbox(
        "Family Size",
        ["GT3","LE3"]
    )

    guardian = st.selectbox(
        "Guardian",
        ["mother","father","other"]
    )

    st.subheader("👨‍👩‍👧 Parents")

    Medu = st.slider(
        "Mother Education",
        0,
        4,
        2
    )

    Fedu = st.slider(
        "Father Education",
        0,
        4,
        2
    )

    Mjob = st.selectbox(
        "Mother Job",
        [
            "at_home",
            "health",
            "other",
            "services",
            "teacher"
        ]
    )

    Fjob = st.selectbox(
        "Father Job",
        [
            "at_home",
            "health",
            "other",
            "services",
            "teacher"
        ]
    )

    famsup = st.selectbox(
        "Family Support",
        ["yes","no"]
    )

    nursery = st.selectbox(
        "Attended Nursery",
        ["yes","no"]
    )

    higher = st.selectbox(
        "Wants Higher Education",
        ["yes","no"]
    )

    internet = st.selectbox(
        "Internet Access",
        ["yes","no"]
    )

    romantic = st.selectbox(
        "Romantic Relationship",
        ["yes","no"]
    )
    
    # =====================================
# RIGHT COLUMN
# =====================================

with right:

    st.subheader("📚 Academic Information")

    studytime = st.slider(
        "Weekly Study Time",
        min_value=1,
        max_value=4,
        value=2
    )

    traveltime = st.slider(
        "Travel Time",
        min_value=1,
        max_value=4,
        value=1
    )

    failures = st.slider(
        "Previous Failures",
        min_value=0,
        max_value=4,
        value=0
    )

    absences = st.number_input(
        "Number of Absences",
        min_value=0,
        max_value=93,
        value=5
    )

    schoolsup = st.selectbox(
        "School Support",
        ["yes","no"]
    )

    paid = st.selectbox(
        "Extra Paid Classes",
        ["yes","no"]
    )

    activities = st.selectbox(
        "Extra-curricular Activities",
        ["yes","no"]
    )

    reason = st.selectbox(
        "Reason for Choosing School",
        [
            "course",
            "home",
            "other",
            "reputation"
        ]
    )

    st.subheader("🌍 Lifestyle")

    famrel = st.slider(
        "Family Relationship",
        1,
        5,
        4
    )

    freetime = st.slider(
        "Free Time",
        1,
        5,
        3
    )

    goout = st.slider(
        "Going Out With Friends",
        1,
        5,
        3
    )

    health = st.slider(
        "Health Status",
        1,
        5,
        5
    )

    Dalc = st.slider(
        "Workday Alcohol Consumption",
        1,
        5,
        1
    )

    Walc = st.slider(
        "Weekend Alcohol Consumption",
        1,
        5,
        2
    )

st.divider()

predict = st.button("🎯 Predict Student Performance")

if predict:

    student = {

        "absences": absences,
        "failures": failures,
        "goout": goout,
        "age": age,
        "Mjob": Mjob,
        "health": health,
        "Medu": Medu,
        "freetime": freetime,
        "Fedu": Fedu,
        "Fjob": Fjob,
        "Walc": Walc,
        "famrel": famrel,
        "studytime": studytime,
        "reason": reason,
        "guardian": guardian,
        "Dalc": Dalc,
        "schoolsup": schoolsup,
        "romantic": romantic,
        "traveltime": traveltime,
        "activities": activities,
        "paid": paid,
        "sex": sex,
        "famsize": famsize,
        "address": address,
        "famsup": famsup,
        "nursery": nursery,
        "higher": higher,
        "internet": internet
    }

    df = preprocess(student)

    scaled = scaler.transform(df)

    prediction = model.predict(scaled)[0]

    probability = model.predict_proba(scaled)[0]

    pass_probability = probability[1] * 100
    
    
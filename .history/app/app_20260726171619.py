import streamlit as st
import pandas as pd
import joblib
from pathlib import Path
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

# ----------------------------
# LOAD MODEL
# ----------------------------

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR.parent / "models"

model = joblib.load(MODEL_DIR / "best_model.pkl")
# scaler = joblib.load(MODEL_DIR / "scaler.pkl")

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

    # Preprocess the user's input
df = preprocess(student)

# Predict directly using the Random Forest model
prediction = model.predict(df)[0]

# Get prediction probabilities
probability = model.predict_proba(df)[0]

pass_probability = probability[1] * 100
    
st.divider()

st.subheader("📊 Prediction Results")

if prediction == 1:

        st.markdown(
            """
            <div class="result-pass">
            ✅ STUDENT IS LIKELY TO PASS
            </div>
            """,
            unsafe_allow_html=True
        )

        st.success(
            f"Probability of Passing: {pass_probability:.2f}%"
        )

        st.progress(min(pass_probability / 100, 1.0))

        st.markdown("### 📌 Interpretation")

        if pass_probability >= 90:

            st.info("""
The model predicts that the student has an **excellent chance of passing**.

The student's academic and personal characteristics are highly similar to students who successfully passed in the training dataset.
            """)

        elif pass_probability >= 75:

            st.info("""
The student has a **high probability of passing**.

Maintaining regular study habits and attendance should further improve performance.
            """)

        else:

            st.info("""
The student is predicted to pass, but the confidence is moderate.

Additional study time and reducing absences could improve performance.
            """)

else:

        st.markdown(
            """
            <div class="result-fail">
            ❌ STUDENT IS LIKELY TO FAIL
            </div>
            """,
            unsafe_allow_html=True
        )

        fail_probability = 100 - pass_probability

        st.error(
            f"Probability of Failing: {fail_probability:.2f}%"
        )

        st.progress(min(fail_probability / 100, 1.0))

        st.markdown("### 📌 Interpretation")

        st.warning("""
The model predicts that the student is at risk of failing.

Possible contributing factors include:

- High number of absences
- Previous academic failures
- Limited study time
- Poor lifestyle habits
- Low family or school support

Providing academic support early may improve the student's chances of success.
        """)

st.divider()

st.subheader("📋 Student Information Summary")

summary = pd.DataFrame({
        "Feature": list(student.keys()),
        "Value": list(student.values())
    })

    st.dataframe(
        summary,
        use_container_width=True,
        hide_index=True
    )

    st.download_button(
        label="📥 Download Student Information",
        data=summary.to_csv(index=False),
        file_name="student_information.csv",
        mime="text/csv"
    )

st.divider()

st.caption(
    """
Student Performance Prediction System

Developed using **Python**, **Streamlit**, **Scikit-learn**, and **Random Forest Classifier**.

This application predicts whether a student is likely to **PASS** or **FAIL** based on academic, demographic, family, and lifestyle factors.
"""
)
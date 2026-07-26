import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="Student Performance Prediction", page_icon="🎓", layout="centered")

st.title("🎓 Student Performance Prediction App")
st.write("Enter student metrics below to predict their final score or performance level.")

st.divider()

# --- INPUT SECTION ---
st.subheader("📋 Student Data Input")

col1, col2 = st.columns(2)

with col1:
    study_hours = st.number_input("Study Hours per Week", min_value=0.0, max_value=100.0, value=6.50, step=0.5)
    attendance = st.slider("Attendance Rate (%)", min_value=0, max_value=100, value=65)

with col2:
    previous_scores = st.number_input("Previous Exam Score (%)", min_value=0.0, max_value=100.0, value=66.00, step=1.0)
    tutoring_sessions = st.selectbox("Tutoring Sessions / Failures", [0, 1, 2, 3, 4, 5], index=0)

# --- PREDICTION SECTION ---
if st.button("🔮 Predict Performance", type="primary"):
    try:
        # 1. Load trained model
        model = joblib.load('models/student_model.pkl')
        
        # 2. Map UI inputs to the exact feature names the model expects
        input_data = pd.DataFrame({
            'studytime': [study_hours],
            'failures': [tutoring_sessions]
        })
        
        # 3. Predict outcome
        prediction = model.predict(input_data)[0]
        
        # 4. Map output to clean text
        outcome_text = "Pass (Good Performance) 🟢" if prediction == 1 else "Fail (Needs Attention) 🔴"
            
        st.success(f"🎯 **Predicted Final Outcome:** {outcome_text}")
        
        # Display submitted data summary table
        st.dataframe(input_data)

    except FileNotFoundError:
        st.error("⚠️ `student_model.pkl` not found in `models/`. Please run the notebook cell first.")
    except Exception as e:
        st.error(f"⚠️ Error calculating prediction: {e}")
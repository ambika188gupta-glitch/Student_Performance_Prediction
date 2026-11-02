
import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

st.set_page_config(
    page_title="Student Performance Predictor",
    page_icon="🎓",
    layout="centered"
)

st.title("🎓 Student Performance Prediction")
st.write("Predict student exam marks using Machine Learning.")

# Load model
MODEL_PATH = Path("models/student_model.pkl")

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

try:
    model = load_model()
except FileNotFoundError:
    st.error("Model not found. Please train the model first.")
    st.stop()

st.subheader("Enter Student Details")

study_hours = st.slider(
    "Daily Study Hours", 1.0, 10.0, 5.0
)

attendance = st.slider(
    "Attendance (%)", 50.0, 100.0, 80.0
)

previous_marks = st.slider(
    "Previous Marks", 30.0, 100.0, 70.0
)

sleep_hours = st.slider(
    "Sleep Hours", 4.0, 10.0, 7.0
)

if st.button("Predict Marks", type="primary"):

    input_data = pd.DataFrame([{
        "study_hours": study_hours,
        "attendance": attendance,
        "previous_marks": previous_marks,
        "sleep_hours": sleep_hours
    }])

    prediction = model.predict(input_data)[0]

    prediction = max(0, min(100, prediction))

    st.success(
        f"Predicted Final Marks: {prediction:.2f}/100"
    )

    st.info(
        "This is an educational prediction, not a guaranteed exam result."
    )

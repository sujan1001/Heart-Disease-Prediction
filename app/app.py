import pickle
import streamlit as st
import pandas as pd

with open("models/heart_disease_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("models/scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

with open("models/imputer.pkl", "rb") as f:
    imputer = pickle.load(f)


st.set_page_config(page_title="Heart Disease Prediction", page_icon=":heart:", layout="wide")

st.title("Heart Disease Prediction")


gender_dict = {
    "Female": 0,
    "Male": 1
}

chestpain_dict = {
    "Typical Angina": 0,
    "Atypical Angina": 1,
    "Non-Anginal Pain": 2,
    "Asymptomatic": 3
}

fbs_dict = {
    "<= 120 mg/dl": 0,
    "> 120 mg/dl": 1
}

ecg_dict = { 
     "Normal": 0,
     "ST-T Wave Abnormality": 1,
     "Left Ventricular Hypertrophy": 2
}

exercise_dict = {
    "No": 0,
    "Yes": 1
}

slope_dict = {
    "Upsloping": 0,
    "Flat": 1,
    "Downsloping": 2
}

st.subheader("Please enter the following information:")
col1, col2 = st.columns(2) 

with col1:
    age = st.number_input("Age", min_value=1, max_value=120, value=30)

    selected_gender = st.selectbox("Gender", options = list(gender_dict.keys()))
    gender = gender_dict[selected_gender]

    selected_cp = st.selectbox("Chest Pain Type", options = list(chestpain_dict.keys()))
    cp = chestpain_dict[selected_cp]

    restingBP = st.number_input(
        "Resting Blood Pressure (mmHg)",
        min_value=50,
        max_value=250,
        value=120
    )

    serumCholesterol = st.number_input(
        "Serum Cholesterol (mg/dl)", min_value=0, max_value=700, value=200)
    
    selected_fbs = st.selectbox("Fasting Blood Sugar", options = list(fbs_dict.keys()))
    fastingbloodsugar = fbs_dict[selected_fbs]

with col2:
    selected_ecg = st.selectbox("Resting ECG Results", options = list(ecg_dict.keys()))
    restecg = ecg_dict[selected_ecg]

    maxheartrate = st.number_input(
        "Maximum Heart Rate Achieved", min_value=50, max_value=250, value=150)

    selected_exercise = st.selectbox("Exercise Induced Angina", options = list(exercise_dict.keys()))
    exerciseangia  = exercise_dict[selected_exercise]

    oldpeak = st.number_input(
        "Oldpeak (ST depression induced by exercise relative to rest)",help="ST depression induced by exercise relative to rest", min_value=0.0, max_value=10.0, value=1.0, step=0.1)

    selected_slope = st.selectbox("Slope of the Peak Exercise ST Segment", options = list(slope_dict.keys()), help="Upsloping is usually normal, flat may suggest reduced blood flow, and downsloping is often associated with a higher likelihood of heart disease.")
    slope = slope_dict[selected_slope]
    
    noofmajorvessels = st.selectbox(
    "Number of Major Vessels (0-3) Colored by Fluoroscopy",help= "Number of major coronary vessels visible during fluoroscopy (0–3). More affected vessels may indicate more severe disease.",
    options=[0, 1, 2, 3]
)

if st.button("Predict"):
    input_data = pd.DataFrame({
        "age": [age],
        "gender": [gender],
        "chestpain": [cp],
        "restingBP": [restingBP],
        "serumcholestrol": [serumCholesterol],
        "fastingbloodsugar": [fastingbloodsugar],
        "restingrelectro": [restecg],
        "maxheartrate": [maxheartrate],
        "exerciseangia": [exerciseangia],
        "oldpeak": [oldpeak],
        "slope": [slope],
        "noofmajorvessels": [noofmajorvessels]
    })

    imputed_data = pd.DataFrame(
    imputer.transform(input_data),
    columns=input_data.columns
)

    numerical_cols = [
     "age",
     "restingBP",
     "serumcholestrol",
     "maxheartrate",
     "oldpeak"
]

    imputed_data[numerical_cols] = scaler.transform(
    imputed_data[numerical_cols]
)

    prediction = model.predict(imputed_data)[0]
    probability = model.predict_proba(imputed_data)[0]
    risk = probability[1]
    st.divider()
    if prediction == 1:
     st.error("High Risk of Heart Disease")
     st.write(f"Confidence: {probability[1]*100:.2f}%")
     st.write(f"Probability of Heart Disease: {risk*100:.2f}%")
     st.progress(risk)
     
    else:
     st.success("Low Risk of Heart Disease")
     st.write(f"Confidence: {(probability[0])*100:.2f}%")
     st.write(f"Probability of Heart Disease: {risk*100:.2f}%")
     st.progress(risk)

 


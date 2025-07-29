import streamlit as st
import numpy as np
import joblib

model = joblib.load("predicting_model.joblib")


st.title("Titanic Survival Prediction")


with st.form("prediction_form"):
    st.header("Passenger Details")

    # Input fields
    pclass = st.selectbox("Passenger Class", [1, 2, 3], index=2)
    sex = st.radio("Gender", ["Male", "Female"])
    age = st.slider("Age", 0.0, 100.0, 30.0)
    sibsp = st.slider("Number of Siblings/Spouses", 0, 8, 0)
    parch = st.slider("Number of Parents/Children", 0, 6, 0)
    embarked = st.selectbox("Port of Embarkation", ["S", "C", "Q"], index=0)

    submitted = st.form_submit_button("Predict Survival")


if submitted:
    sex_male = 1.0 if sex == "Male" else 0.0
    sex_female = 1.0 if sex == "Female" else 0.0

    embarked_c = 1.0 if embarked == "C" else 0.0
    embarked_s = 1.0 if embarked == "S" else 0.0
    embarked_q = 1.0 if embarked == "Q" else 0.0

    passId = 312
    fare = 262.3750

    input_data = np.array(
        [
            [
                passId,
                pclass,
                age,
                sibsp,
                parch,
                fare,
                embarked_c,
                embarked_s,
                embarked_q,
                sex_female,
                sex_male,
            ]
        ],
        dtype=object,
    )

    # Make prediction
    prediction = model.predict(input_data)[0]

    # Show result
    st.subheader("Prediction Result")
    if prediction == 1:
        st.success("This passenger would likely have survived!")
    else:
        st.error("This passenger would likely not have survived.")

    # Show raw prediction for debugging
    st.write(f"Model prediction value: {prediction}")

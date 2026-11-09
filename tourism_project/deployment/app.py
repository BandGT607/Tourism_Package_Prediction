import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="Wellness Tourism Predictor", layout="wide")
st.title("Wellness Tourism Package Predictor")
st.write("Enter customer details below to predict the likelihood of a package purchase.")

@st.cache_resource
def load_model():
    return joblib.load("model.joblib")

model = load_model()

with st.form("customer_data_form"):
    col1, col2 = st.columns(2)
    with col1:
        age = st.number_input("Age", min_value=18, max_value=100, value=30)
        city_tier = st.selectbox("City Tier", [1, 2, 3])
        occupation = st.selectbox("Occupation", ["Salaried", "Free Lancer", "Small Business", "Large Business"])
        gender = st.selectbox("Gender", ["Male", "Female"])
        monthly_income = st.number_input("Monthly Income", min_value=1000, value=20000)
        marital_status = st.selectbox("Marital Status", ["Single", "Married", "Divorced", "Unmarried"])
        designation = st.selectbox("Designation", ["Executive", "Manager", "VP", "AVP", "Senior Manager"])
        passport = st.selectbox("Passport (1=Yes, 0=No)", [0, 1])
        own_car = st.selectbox("Own Car (1=Yes, 0=No)", [0, 1])

    with col2:
        type_of_contact = st.selectbox("Type of Contact", ["Company Invited", "Self Inquiry"])
        duration_of_pitch = st.number_input("Duration of Pitch (minutes)", min_value=1, value=10)
        number_of_person_visiting = st.number_input("Number of Persons Visiting", min_value=1, value=2)
        number_of_children_visiting = st.number_input("Number of Children Visiting", min_value=0, value=0)
        preferred_property_star = st.selectbox("Preferred Property Star", [3.0, 4.0, 5.0])
        number_of_trips = st.number_input("Number of Trips", min_value=1, value=2)
        pitch_satisfaction_score = st.selectbox("Pitch Satisfaction Score", [1, 2, 3, 4, 5])
        product_pitched = st.selectbox("Product Pitched", ["Basic", "Standard", "Deluxe", "Super Deluxe", "King"])
        number_of_followups = st.number_input("Number of Followups", min_value=1, value=3)
    submit = st.form_submit_button("Predict Purchase Probability")

if submit:
    input_data = pd.DataFrame({
        "Age": [age], "TypeofContact": [type_of_contact], "CityTier": [city_tier],
        "DurationOfPitch": [duration_of_pitch], "Occupation": [occupation], "Gender": [gender],
        "NumberOfPersonVisiting": [number_of_person_visiting], "NumberOfFollowups": [number_of_followups],
        "ProductPitched": [product_pitched], "PreferredPropertyStar": [preferred_property_star],
        "MaritalStatus": [marital_status], "NumberOfTrips": [number_of_trips], "Passport": [passport],
        "PitchSatisfactionScore": [pitch_satisfaction_score], "OwnCar": [own_car],
        "NumberOfChildrenVisiting": [number_of_children_visiting], "Designation": [designation],
        "MonthlyIncome": [monthly_income]
    })
    
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]
    
    st.divider()
    if prediction == 1:
        st.success("### Prediction: High Likelihood to Purchase")
        st.write(f"**Confidence:** {probability:.1%}")
    else:
        st.error("### Prediction: Unlikely to Purchase")
        st.write(f"**Confidence:** {1 - probability:.1%}")
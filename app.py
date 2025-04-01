import streamlit as st
import joblib

# Load the pre-trained model (ensure this file is in your repo)
model = joblib.load("linear_model.pkl")

st.title("Real Estate Price Predictor")
st.write("Predict property price using multiple features.")

# User inputs for each feature
square_feet = st.number_input("Enter property square feet:", min_value=0.0, value=100.0)
num_bedrooms = st.number_input("Enter number of bedrooms:", min_value=0, value=3, step=1)
num_bathrooms = st.number_input("Enter number of bathrooms:", min_value=0, value=2, step=1)
num_floors = st.number_input("Enter number of floors:", min_value=0, value=1, step=1)
year_built = st.number_input("Enter year built:", min_value=1800, max_value=2025, value=2000, step=1)
has_garden = st.checkbox("Has Garden")
has_pool = st.checkbox("Has Pool")
garage_size = st.number_input("Enter garage size:", min_value=0, value=1, step=1)
location_score = st.number_input("Enter location score (0-10):", min_value=0.0, max_value=10.0, value=5.0)
distance_to_center = st.number_input("Enter distance to center (e.g., in miles):", min_value=0.0, value=5.0)

if st.button("Predict Price"):
    # Convert boolean checkboxes to integers if your model expects numeric values (1/0)
    has_garden_val = 1 if has_garden else 0
    has_pool_val = 1 if has_pool else 0
    
    # Create the feature vector in the same order as used during training
    features = [
        square_feet,
        num_bedrooms,
        num_bathrooms,
        num_floors,
        year_built,
        has_garden_val,
        has_pool_val,
        garage_size,
        location_score,
        distance_to_center
    ]
    
    # Perform prediction and display the result
    prediction = model.predict([features])[0]
    st.write(f"Predicted Price: ${prediction:,.2f}")

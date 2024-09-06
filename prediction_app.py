import numpy as np
import pickle
import streamlit as st
import pandas as pd
import os
from pathlib import Path
Country_Mapping = {
    'Afghanistan': 0, 'Angola': 1, 'Benin': 2, 'Botswana': 3, 'Burundi': 4, 'Cameroon': 5, 
    'Congo, DRC': 6, "Côte d'Ivoire": 7, 'Dominican Republic': 8, 'Ethiopia': 9, 'Ghana': 10, 
    'Guatemala': 11, 'Guyana': 12, 'Haiti': 13, 'Kenya': 14, 'Libya': 15, 'Malawi': 16, 
    'Mozambique': 17, 'Namibia': 18, 'Nigeria': 19, 'Rwanda': 20, 'Senegal': 21, 
    'South Africa': 22, 'South Sudan': 23, 'Swaziland': 24, 'Tanzania': 25, 'Togo': 26, 
    'Uganda': 27, 'Vietnam': 28, 'Zambia': 29, 'Zimbabwe': 30
}

Shipment_Mode_Mapping = {
    'Air': 0, 'Air Charter': 1, 'Ocean': 2, 'Truck': 3
}

Manufacturing_Site_Mapping = {
    'ABBVIE (Abbott) France': 0, 'ABBVIE (Abbott) Logis. UK': 1, "ABBVIE (Abbott) St. P'burg USA": 2, 
    'ABBVIE Ludwigshafen Germany': 3, 'Aspen-OSD, Port Elizabeth, SA': 4, 'Aurobindo Unit III, India': 5, 
    'Aurobindo Unit VII, IN': 6, 'BMS Evansville, US': 7, 'BMS Meymac, France': 8, 
    'Bristol-Myers Squibb Anagni IT': 9, 'Cipla Ltd A-42 MIDC Mahar. IN': 10, 'Cipla, Goa, India': 11, 
    'Cipla, Kurkumbh, India': 12, 'Cipla, Patalganga, India': 13, 
    'Emcure Plot No.P-2, I.T-B.T. Park, Phase II, MIDC, Hinjwadi, Pune, India': 14, 
    'GSK Barnard Castle UK': 15, 'GSK Mississauga (Canada)': 16, 'GSK Ware (UK)': 17, 
    'Gilead(Nycomed) Oranienburg DE': 18, 'Gland Pharma Ltd Pally Factory': 19, 
    'Gland Pharma, Hyderabad, IN': 20, 'Guilin OSD site, No 17, China': 21, 
    'Hetero Unit III Hyderabad IN': 22, 'Hetero, Jadcherla, unit 5, IN': 23, 
    'Ipca Dadra/Nagar Haveli IN': 24, 'Janssen Ortho LLC, Puerto Rico': 25, 
    'Janssen-Cilag, Latina, IT': 26, 'MSD Elkton USA': 27, 'MSD Manati, Puerto Rico, (USA)': 28, 
    'MSD Patheon, Canada': 29, 'MSD, Haarlem, NL': 30, 'Macleods Daman Plant INDIA': 31, 
    'Meditab (for Cipla) Daman IN': 32, 'Medochemie Factory A, CY': 33, 
    'Medopharm Malur Factory, INDIA': 34, 'Mepro Pharm Wadhwan Unit II': 35, 
    'Micro Labs Ltd. (Brown & Burk), India': 36, 'Micro Labs, Hosur, India': 37, 
    'Micro labs, Verna, Goa, India': 38, 'Mylan (formerly Matrix) Nashik': 39, 
    'Mylan,  H-12 & H-13, India': 40, 'Novartis Pharma AG, Switzerland': 41, 
    'Novartis Pharma Suffern, USA': 42, 'Ranbaxy per Shasun Pharma': 43, 
    'Ranbaxy per Shasun Pharma Ltd': 44, 'Ranbaxy, Paonta Shahib, India': 45, 
    'Remedica, Limassol, Cyprus': 46, 'Roche Basel': 47, 'Roche Madrid': 48, 
    'Strides, Bangalore, India.': 49, 
    'Weifa A.S., Hausmanngt. 6, P.O. Box 9113 GrÃ¸nland, 0133, Oslo, Norway': 50
}

# Caching the model loading to improve performance
@st.cache_resource
def load_model():
    model_path = Path.home() / "Desktop" / "PREDICTION MODEL" / "trained_model.sav"  # Use a relative path
    with open(model_path, 'rb') as file:
        loaded_model = pickle.load(file)
    return loaded_model

loaded_model = load_model()

def price_pre(input_data):
    rf_predictions = loaded_model.predict(input_data)
    return rf_predictions

def main():
    # Giving a title
    st.title("Shipping Price Prediction Web App")

    st.markdown("""
    ### Enter the details below to predict the shipping price:
    """)


    # Getting input data from user with appropriate widgets
    Unit_of_Measure = st.number_input("Number of Units of Measure", min_value=1, step=1)
    Line_item_Quantity = st.number_input("Enter Line Item Quantity", min_value=1, step=1)
    Pack_price = st.number_input("Enter Pack Price", min_value=0.0, step=1.0, format="%.2f")
    Unit_price = st.number_input("Enter Unit Price", min_value=0.0, step=1.0, format="%.2f")
    Weight = st.number_input("Enter Weight (Kilograms)", min_value=0.0, step=0.1, format="%.2f")
    
    country_label = st.selectbox("Select Country", options=list(Country_Mapping.keys()))
    Country_code = Country_Mapping[country_label]  # Get the encoded value

    shipping_mode_label = st.selectbox("Select Shipping Mode", options=list(Shipment_Mode_Mapping.keys()))
    shipping_mode = Shipment_Mode_Mapping[shipping_mode_label]  # Get the encoded value

    manufacturing_label = st.selectbox("Select Manufacturing Site", options=list(Manufacturing_Site_Mapping.keys()))
    manufacturing = Manufacturing_Site_Mapping[manufacturing_label] 

    # Placeholder for prediction output
    price_output = ''

    if st.button('Predict Price'):
        try:
            input_features = np.array([[Unit_of_Measure, Line_item_Quantity, Pack_price,
                                        Unit_price, Weight, Country_code, shipping_mode, manufacturing]])
            prediction = price_pre(input_features)
            # Assuming prediction is a NumPy array with a single value
            price_output = f"Estimated Shipping Price: ${prediction[0]:.2f}"
        except Exception as e:
            st.error(f"Error in prediction: {e}")

    if price_output:
        st.success(price_output)

if __name__ == '__main__':
    main()

import streamlit as st
import numpy as np
import pandas as pd
import pickle

# Load regression model
try:
    with open('Regression_copper_model.pkl', 'rb') as file:
        reg_model = pickle.load(file)
except Exception as e:
    st.error(f"Error loading regression model: {e}")

# Load classification model
try:
    with open("classification_copper_model.pkl", 'rb') as file1:
        classification_model = pickle.load(file1)
except Exception as e:
    st.error(f"Error loading classification model: {e}")

# Load the scaler
try:
    with open('scaler.pkl', 'rb') as file:
        loaded_scaler = pickle.load(file)
except Exception as e:
    st.error(f"Error loading scaler: {e}")

# Load the one-hot encoder
try:
    with open('one_hot_encoder.pkl', 'rb') as file:
        loaded_encoder = pickle.load(file)
except Exception as e:
    st.error(f"Error loading one-hot encoder: {e}")

# Style
st.markdown("""
<style>
    body {
        background-color: #fff8dc;
    }
    .stApp {
        background-color: #fff8dc;
    }
    div[data-testid="stSidebar"] {
        background-color: #d0e2f2;
    }
    div.stButton > button {
        background-color: red !important;
        color: white !important;
        border-radius: 8px;
        padding: 10px 20px;
        font-size: 16px;
        font-weight: bold;
        border: none;
    }
</style>
""", unsafe_allow_html=True)

st.title('Copper Modeling')
st.sidebar.title('Explore Sections')
selection = st.sidebar.radio('Go to', ['Overview', 'Selling Price', 'Status'])

# -------------------- Overview --------------------
if selection == 'Overview':
    st.header('Overview')
    st.image("D:/copper_model2/copper_images/overview.webp")
    st.markdown("""
    <div style='text-align: left; padding: 10px; font-size: 18px; line-height: 1.6; color: #333;'>
        <h3 style='color: red; text-align: center;'>Copper Modeling Overview</h3>
        <p>
            The Copper Modeling project focuses on analyzing and predicting copper-related metrics 
            to optimize industrial processes and decision-making. By leveraging advanced data analytics 
            and visualization techniques, the project aims to model critical parameters such as selling price, 
            production metrics, and market demand. 
        </p>
        <p>
            Using Python and Streamlit, an interactive web application will be developed, providing 
            stakeholders with real-time insights and dynamic visualizations.
        </p>
    </div>
    """, unsafe_allow_html=True)

# -------------------- Selling Price --------------------
elif selection == 'Selling Price':
    st.image("D:/copper_model2/copper_images/price.jpg")

    col1, col2 = st.columns(2)
    thickness_log = st.number_input("**THICKNESS (Log Value)**", format="%0.4f")

    with col1:
        country = st.slider('COUNTRY', 25, 113, 25)
        application = st.slider("APPLICATION", 2.0, 87.5, 2.0, 0.1)
        width = st.slider("WIDTH", 700.0, 1980.0, 700.0)
        product_ref = st.slider("PRODUCT_REF", 611728, 1722207579, 611728)
        quantity_tons_log = st.slider("QUANTITY_TONS (Log)", -0.3223, 6.9247, 0.3223)
        customer_log = st.slider("CUSTOMER (Log)", 17.2191, 17.2301, 17.2191)
        status = st.slider("STATUS", 0, 8, 0)

    with col2:
        item_date_day = st.slider("ITEM DATE - Day", 1, 31, 1)
        item_date_month = st.slider("ITEM DATE - Month", 1, 12, 1)
        item_date_year = st.slider("ITEM DATE - Year", 2020, 2021, 2020)
        delivery_date_day = st.slider("DELIVERY DATE - Day", 1, 31, 1)
        delivery_date_month = st.slider("DELIVERY DATE - Month", 1, 12, 1)
        delivery_date_year = st.slider("DELIVERY DATE - Year", 2020, 2022, 2020)
      

    item_type = st.selectbox("ITEM TYPE", ['Others', 'PL', 'S', 'W', 'WI'])
    encoded_item_type = loaded_encoder.transform([[item_type]])

    base_features = np.array([[quantity_tons_log, customer_log, country, status, application, thickness_log, 
                               width, product_ref, item_date_day, item_date_month, item_date_year, delivery_date_day,
                               delivery_date_month, delivery_date_year]])

    # Combine numerical and encoded categorical
    combined_features = np.concatenate((base_features, encoded_item_type), axis=1)

    # Apply scaling using the loaded scaler
    scaled_features = loaded_scaler.transform(combined_features)
    if st.button("Predict Selling Price"):
        try:
            predicted_price = reg_model.predict(scaled_features)[0]
            st.markdown(f'<h3 style="color: blue;">Predicted Selling Price: ${predicted_price:.2f}</h3>', unsafe_allow_html=True)
        except Exception as e:
            st.error(f"Prediction failed: {e}")

# -------------------- Status --------------------
elif selection == 'Status':
    st.image("D:/copper_model2/copper_images/istockphoto-1267692194-612x612.jpg")
    col3, col4 = st.columns(2)

    thickness_log = st.number_input("**THICKNESS (Log Value)**", format="%0.4f")

    with col3:
        country = st.slider('COUNTRY', 25, 113, 25)
        application = st.slider("APPLICATION", 2.0, 87.5, 2.0, 0.1)
        width = st.slider("WIDTH", 700.0, 1980.0, 700.0)
        product_ref = st.slider("PRODUCT_REF", 611728, 1722207579, 611728)
        quantity_tons_log = st.slider("QUANTITY_TONS (Log)", -0.3223, 6.9247, 0.3223)
        customer_log = st.slider("CUSTOMER (Log)", 17.2191, 17.2301, 17.2191)
        delivery_date_year = st.slider("DELIVERY DATE - Year", 2020, 2022, 2020)
    with col4:
        selling_price = st.slider("SELLING PRICE", 5.97503, 1500.0, 100.0)
        item_date_day = st.slider("ITEM DATE - Day", 1, 31, 1)
        item_date_month = st.slider("ITEM DATE - Month", 1, 12, 1)
        item_date_year = st.slider("ITEM DATE - Year", 2020, 2021, 2020)
        delivery_date_day = st.slider("DELIVERY DATE - Day", 1, 31, 1)
        delivery_date_month = st.slider("DELIVERY DATE - Month", 1, 12, 1)
       

    item_type = st.selectbox("ITEM TYPE", ['Others', 'PL', 'S', 'W', 'WI'], key="status_item_type")
    encoded_item_type = loaded_encoder.transform([[item_type]])

    base_features = np.array([[quantity_tons_log, customer_log, country, application, thickness_log,
                                width, product_ref, selling_price, item_date_day, item_date_month, 
                                item_date_year, delivery_date_day, delivery_date_month, delivery_date_year]])

    combined_features = np.concatenate((base_features, encoded_item_type), axis=1)

    # Apply scaling using the loaded scaler
    scaled_features = loaded_scaler.transform(combined_features)

    if st.button("Predict Status"):
        try:
            predicted_status = classification_model.predict(scaled_features)[0]
            if predicted_status == 1:
                
                st.markdown(f'<h3 style="color: green;"> WON</h3>', unsafe_allow_html=True) 
            else :
                st.markdown(f'<h3 style="color: red;"> LOSS </h3>', unsafe_allow_html=True) 
        except Exception as e:
            st.error(f"Prediction failed: {e}")

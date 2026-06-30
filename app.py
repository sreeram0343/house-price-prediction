import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import os
import matplotlib.pyplot as plt
import seaborn as sns
from src.preprocessing import engineer_features

# Set page config for premium styling
st.set_page_config(
    page_title="Real Estate AI Intelligence Platform",
    page_icon="🏡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern dark-themed aesthetics and glassmorphism cards
st.markdown("""
<style>
    /* Dark theme overrides and premium background */
    .stApp {
        background: linear-gradient(135deg, #121824 0%, #0A0F18 100%);
        color: #E2E8F0;
        font-family: 'Inter', sans-serif;
    }
    
    /* Title and Header customization */
    .main-title {
        background: linear-gradient(90deg, #60A5FA 0%, #3B82F6 50%, #2563EB 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800;
        font-size: 3rem;
        margin-bottom: 0.2rem;
        text-align: center;
    }
    .subtitle {
        color: #94A3B8;
        font-size: 1.2rem;
        text-align: center;
        margin-bottom: 2rem;
    }
    
    /* Card design with glassmorphism */
    .glass-card {
        background: rgba(30, 41, 59, 0.45);
        backdrop-filter: blur(12px);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 16px;
        padding: 24px;
        margin-bottom: 20px;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.3);
        transition: transform 0.2s ease-in-out;
    }
    .glass-card:hover {
        transform: translateY(-2px);
        border-color: rgba(96, 165, 250, 0.3);
    }
    
    /* Value displays */
    .price-display {
        font-size: 3.5rem;
        font-weight: 800;
        color: #10B981;
        text-align: center;
        margin: 15px 0;
        text-shadow: 0 0 20px rgba(16, 185, 129, 0.2);
    }
    
    .section-header {
        font-size: 1.3rem;
        font-weight: 700;
        color: #60A5FA;
        border-bottom: 2px solid rgba(96, 165, 250, 0.2);
        padding-bottom: 8px;
        margin-bottom: 15px;
    }
</style>
""", unsafe_allow_html=True)

# Helper function to load model artifacts
@st.cache_resource
def load_artifacts():
    try:
        model = joblib.load('models/best_model.pkl')
        scaler = joblib.load('models/scaler.pkl')
        with open('models/features.json', 'r') as f:
            features = json.load(f)
        with open('models/metrics.json', 'r') as f:
            metrics = json.load(f)
        return model, scaler, features, metrics
    except Exception as e:
        st.error(f"Error loading model artifacts: {e}. Please ensure the pipeline has been run successfully.")
        return None, None, None, None

model, scaler, feature_names, metrics = load_artifacts()

# App Header
st.markdown('<div class="main-title">🏡 Real-Estate AI Intelligence</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Phase 2: E2E Machine Learning Pipeline & Interactive Valuation Service</div>', unsafe_allow_html=True)

# Sidebar with model metrics and details
with st.sidebar:
    st.markdown('<div class="section-header">🧠 Model Performance</div>', unsafe_allow_html=True)
    if metrics:
        st.metric("Model Architecture", "XGBoost Regressor")
        st.metric("R² Score (Test Set)", f"{metrics['r2']*100:.2f}%")
        st.metric("Mean Absolute Error (MAE)", f"₹{metrics['mae']:,.2f}")
        st.metric("Mean Absolute % Error (MAPE)", f"{metrics['mape']:.2f}%")
    else:
        st.info("Run the training pipeline (`main.py`) to generate metrics.")
        
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="section-header">📂 Repository Resources</div>', unsafe_allow_html=True)
    st.markdown("- [preprocessing.py](file:///c:/Users/Thinkpad/Desktop/projects/house-price-prediction/src/preprocessing.py)")
    st.markdown("- [train.py](file:///c:/Users/Thinkpad/Desktop/projects/house-price-prediction/src/train.py)")
    st.markdown("- [evaluate.py](file:///c:/Users/Thinkpad/Desktop/projects/house-price-prediction/src/evaluate.py)")
    st.markdown("- [main.py](file:///c:/Users/Thinkpad/Desktop/projects/house-price-prediction/main.py)")
    st.markdown("- [README.md](file:///c:/Users/Thinkpad/Desktop/projects/house-price-prediction/README.md)")

# Check if model loaded successfully
if model is not None:
    # Set up columns for layout
    left_col, right_col = st.columns([2, 1.2])
    
    with left_col:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-header">🏠 Property Specifications</div>', unsafe_allow_html=True)
        
        # Grid layout for inputs
        col1, col2, col3 = st.columns(3)
        with col1:
            bedrooms = st.number_input("Bedrooms", min_value=1, max_value=33, value=3, step=1)
            bathrooms = st.number_input("Bathrooms", min_value=0.5, max_value=8.0, value=2.25, step=0.25)
            floors = st.number_input("Floors", min_value=1.0, max_value=4.0, value=1.5, step=0.5)
            views = st.slider("Number of Views", 0, 4, 0)
            
        with col2:
            living_area = st.number_input("Living Area (sqft)", min_value=200, max_value=15000, value=2000, step=50)
            lot_area = st.number_input("Lot Area (sqft)", min_value=200, max_value=100000, value=5000, step=100)
            basement_area = st.number_input("Basement Area (sqft)", min_value=0, max_value=5000, value=0, step=50)
            excl_basement = st.number_input("Living Area Excl. Basement (sqft)", min_value=200, max_value=15000, value=2000, step=50)
            
        with col3:
            condition = st.slider("House Condition (1-5)", 1, 5, 3)
            grade = st.slider("House Grade (1-13)", 1, 13, 7)
            waterfront = st.selectbox("Waterfront View?", ["No", "Yes"])
            waterfront_val = 1 if waterfront == "Yes" else 0
            
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-header">📍 Location & Neighborhood</div>', unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            latitude = st.number_input("Latitude", min_value=52.0, max_value=54.0, value=52.8064, format="%.5f")
            longitude = st.number_input("Longitude", min_value=-116.0, max_value=-112.0, value=-114.4210, format="%.5f")
            postal_code = st.number_input("Postal Code", min_value=122001, max_value=122080, value=122032)
            
        with col2:
            schools = st.slider("Nearby Schools", 1, 3, 2)
            airport_dist = st.slider("Distance from Airport (km)", 50, 80, 65)
            
        with col3:
            living_renov = st.number_input("Living Area (Renovated, sqft)", min_value=200, max_value=15000, value=2000, step=50)
            lot_renov = st.number_input("Lot Area (Renovated, sqft)", min_value=200, max_value=100000, value=5000, step=100)
            
        col1, col2 = st.columns(2)
        with col1:
            built_year = st.number_input("Year Built", min_value=1900, max_value=2016, value=1975)
        with col2:
            renovation_year = st.number_input("Year Renovated (0 if none)", min_value=0, max_value=2016, value=0)
            
        st.markdown('</div>', unsafe_allow_html=True)

    with right_col:
        # Prediction Output card
        st.markdown('<div class="glass-card" style="text-align: center;">', unsafe_allow_html=True)
        st.markdown('<h3 style="color:#60A5FA; font-weight:700; margin-bottom:10px;">Predicted Property Value</h3>', unsafe_allow_html=True)
        
        # Prepare inputs dictionary
        raw_inputs = {
            'number of bedrooms': [bedrooms],
            'number of bathrooms': [bathrooms],
            'living area': [living_area],
            'lot area': [lot_area],
            'number of floors': [floors],
            'waterfront present': [waterfront_val],
            'number of views': [views],
            'condition of the house': [condition],
            'grade of the house': [grade],
            'Area of the house(excluding basement)': [excl_basement],
            'Area of the basement': [basement_area],
            'Built Year': [built_year],
            'Renovation Year': [renovation_year],
            'Postal Code': [postal_code],
            'Lattitude': [latitude],
            'Longitude': [longitude],
            'living_area_renov': [living_renov],
            'lot_area_renov': [lot_renov],
            'Number of schools nearby': [schools],
            'Distance from the airport': [airport_dist]
        }
        
        # Convert to DataFrame
        raw_df = pd.DataFrame(raw_inputs)
        
        # Apply preprocessing & engineering
        engineered_df = engineer_features(raw_df)
        
        # Reorder features to match exact shape and order of training
        engineered_df = engineered_df[feature_names]
        
        # Scale
        scaled_inputs = scaler.transform(engineered_df)
        
        # Predict (log scale)
        log_pred = model.predict(scaled_inputs)
        
        # Inverse transform
        final_price = np.expm1(log_pred)[0]
        
        # Display formatted price
        st.markdown(f'<div class="price-display">₹{final_price:,.2f}</div>', unsafe_allow_html=True)
        
        # Additional valuation insights
        price_per_sqft = final_price / living_area if living_area > 0 else 0
        st.markdown(f"**Value per sqft:** ₹{price_per_sqft:,.2f}")
        st.markdown(f"**Airport Distance:** {airport_dist} km | **Condition Index:** {condition}/5")
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Plot location on Map
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-header">🗺️ Interactive Map Location</div>', unsafe_allow_html=True)
        # Convert longitude from anonymized range to Seattle GPS if needed, but st.map handles relative coordinates
        # Since Seattle is ~ 47.6, -122.3, the raw dataset latitude 52.8 and longitude -114.4 is in Alberta, Canada.
        # We will map it directly.
        map_df = pd.DataFrame({'lat': [latitude], 'lon': [longitude]})
        st.map(map_df, zoom=12)
        st.markdown('</div>', unsafe_allow_html=True)

    # Tabs for additional information
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    tab1, tab2 = st.tabs(["📊 Analytics & Model Insights", "ℹ️ Engineering & Feature Information"])
    
    with tab1:
        st.write("Below are the model diagnostics and feature importances extracted from the training runs:")
        col_plot1, col_plot2, col_plot3 = st.columns(3)
        with col_plot1:
            if os.path.exists('plots/feature_importance.png'):
                st.image('plots/feature_importance.png', caption="Feature Importances (XGBoost)", use_container_width=True)
        with col_plot2:
            if os.path.exists('plots/actual_vs_predicted.png'):
                st.image('plots/actual_vs_predicted.png', caption="Actual vs. Predicted Scatter Plot", use_container_width=True)
        with col_plot3:
            if os.path.exists('plots/residuals_distribution.png'):
                st.image('plots/residuals_distribution.png', caption="Residuals Distribution", use_container_width=True)
                
    with tab2:
        st.markdown("### Engineered Features Explanation")
        st.write("We added several highly predictive engineered features to boost performance:")
        st.markdown("""
        * **house_age**: Calculated as `2016 - Built Year` to capture age-related depreciation.
        * **is_renovated**: Binary flag (`0` or `1`) indicating whether the property underwent renovation.
        * **years_since_last_mod**: Measures the time elapsed since the property's structure was last touched (renovated or built).
        * **total_rooms**: Simple summation of bedrooms and bathrooms.
        * **bed_bath_ratio**: The ratio of bedrooms to bathrooms, capturing standard layouts vs cramped/spacious conditions.
        * **living_lot_ratio**: Proportion of living area relative to the land lot area.
        * **has_basement**: Binary flag checking if `Area of the basement > 0`.
        * **renovated_living_diff**: Captures changes in size between original living area and renovated living area.
        """)
    st.markdown('</div>', unsafe_allow_html=True)
else:
    st.warning("Please run `python main.py` first to train the model and serialize artifacts before running the dashboard.")

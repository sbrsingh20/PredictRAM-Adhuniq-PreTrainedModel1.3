import streamlit as st
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.exceptions import NotFittedError
from xgboost import XGBRegressor
import os

# Path to the result folder where pre-trained models are stored
RESULT_FOLDER = 'result/'

# Load a pre-trained model from a .pkl file in the result folder
def load_model(file_path):
    try:
        model = joblib.load(file_path)
        if isinstance(model, Pipeline):
            st.write("Loaded model is a pipeline.")
            # Check if the model inside the pipeline is fitted
            if hasattr(model.named_steps['model'], 'booster_'):
                st.write("Pipeline model is fitted.")
            else:
                st.error("Pipeline model is not fitted.")
        elif isinstance(model, XGBRegressor):
            st.write("Loaded model is a trained XGBRegressor.")
            if hasattr(model, 'booster_'):
                st.write("Model is fitted.")
            else:
                st.error("XGBRegressor model is not fitted.")
        else:
            st.error(f"Unexpected model type: {type(model)}")
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

# Check if the model inside the pipeline is fitted
def check_model_fitted(model):
    if isinstance(model, Pipeline):
        if 'model' in model.named_steps:
            inner_model = model.named_steps['model']
            if hasattr(inner_model, 'booster_'):  # XGBoost models have 'booster_' attribute when fitted
                return True
            else:
                return False
    return False

# Historical stock performance and predicted returns
def show_stock_performance(stock_data, predicted_returns):
    st.subheader("Historical Stock Performance and Predicted Returns")
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(stock_data['Close'], label="Historical Close Price", color='blue')
    ax.plot(stock_data.index, predicted_returns, label="Predicted Returns", color='red', linestyle='--')
    ax.set_xlabel('Date')
    ax.set_ylabel('Price / Predicted Returns')
    ax.set_title('Historical Stock Performance and Predicted Returns')
    ax.legend()
    st.pyplot(fig)

# Check and load the pre-trained model for the selected stock
def check_and_load_model(stock_name):
    model = None

    # Look for the pre-trained model in the result folder with specific name pattern
    model_filename = f"{stock_name}.xlsx_gdp_vix_stock_prediction_xgb_model.pkl"
    model_path = os.path.join(RESULT_FOLDER, model_filename)

    if os.path.exists(model_path):
        st.write(f"Loading pre-trained model for {stock_name}...")
        model = load_model(model_path)
        # Check if the model is fitted
        if model and not check_model_fitted(model):
            st.error(f"Model for {stock_name} is not fitted.")
            return None
    else:
        st.error(f"No pre-trained model found for {stock_name} in the result folder. Expected file: {model_filename}")
    
    return model

# Main Streamlit app
def main():
    st.title("Stock Prediction with Macroeconomic Parameters")
    uploaded_model = st.file_uploader("Upload pre-trained model", type="pkl")
    
    if uploaded_model is not None:
        model_pipeline = load_model(uploaded_model)
        
        if model_pipeline:
            # Display model parameters and evaluation
            available_stocks = ['ASHOKLEY', 'AJANTPHARM']  # Modify to include your available stocks
            selected_stocks = st.multiselect("Select stocks for prediction", available_stocks)
            
            if selected_stocks:
                st.subheader("Enter Macroeconomic Parameters")
                gdp = st.slider("GDP Growth (%)", min_value=-5, max_value=5, value=2)
                inflation = st.slider("Inflation (%)", min_value=0, max_value=10, value=2)
                interest_rate = st.slider("Interest Rate (%)", min_value=0, max_value=10, value=2)
                vix = st.slider("VIX Index", min_value=10, max_value=100, value=20)
                
                if st.button("Simulate and Predict"):
                    for stock_name in selected_stocks:
                        stock_file = f"stockdata/{stock_name}.xlsx"
                        
                        # Check if the stock data file exists
                        if os.path.exists(stock_file):
                            stock_data = pd.read_excel(stock_file, engine='openpyxl')
                            stock_data['Date'] = pd.to_datetime(stock_data['Date'])
                            stock_data.set_index('Date', inplace=True)

                            # Check and load the model for the selected stock
                            model = check_and_load_model(stock_name)
                            
                            if model:
                                input_data = np.array([[gdp, inflation, interest_rate, vix]])
                                
                                # Predict using the fitted model
                                if isinstance(model, Pipeline):
                                    inner_model = model.named_steps['model']
                                    predicted_returns = inner_model.predict(input_data)
                                    show_stock_performance(stock_data, predicted_returns)
                                elif isinstance(model, XGBRegressor):
                                    predicted_returns = model.predict(input_data)
                                    show_stock_performance(stock_data, predicted_returns)
                                else:
                                    st.error(f"Unexpected model type for {stock_name}.")
                        else:
                            st.error(f"Stock data file for {stock_name} does not exist at {stock_file}.")
        else:
            st.error("Failed to load the model. Please check the file format.")
    else:
        st.warning("Please upload a pre-trained model file.")

if __name__ == "__main__":
    main()

import streamlit as st
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.exceptions import NotFittedError
from xgboost import XGBRegressor
import os

# Load the overall model evaluation file
def load_model_evaluation(file_path):
    try:
        model = joblib.load(file_path)
        if isinstance(model, list):
            st.write("Loaded model is a list (perhaps multiple models).")
        elif isinstance(model, Pipeline):
            st.write("Loaded model is a pipeline.")
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

# Display model parameters for XGBRegressor or pipeline
def display_model_params(model):
    st.subheader("Model Parameters")
    if isinstance(model, Pipeline):
        if 'model' in model.named_steps:
            params = model.named_steps['model'].get_params()
            st.write(params)
        else:
            st.error("Pipeline does not have 'model' step.")
    elif isinstance(model, XGBRegressor):
        params = model.get_params()
        st.write(params)
    else:
        st.error("Model is not recognized.")

# Display model evaluation results
def display_evaluation_results(overall_results):
    st.subheader("Model Evaluation Results")
    if isinstance(overall_results, list):
        for result in overall_results:
            st.write(f"**Stock**: {result.get('stock', 'Unknown')}")
            st.write(f"**Accuracy (R-squared %)**: {result.get('accuracy', 0):.4f}%")
            st.write(f"**R-squared**: {result.get('r2_score', 0):.4f}")
            st.write(f"**Mean Squared Error**: {result.get('mean_squared_error', 0):.4f}")
            st.write("="*40)
    else:
        st.error("Evaluation results are not in the expected format.")

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

# Main Streamlit app
def main():
    st.title("Stock Prediction with Macroeconomic Parameters")
    uploaded_model = st.file_uploader("Upload pre-trained model", type="pkl")
    
    if uploaded_model is not None:
        model_pipeline = load_model_evaluation(uploaded_model)
        
        if model_pipeline:
            display_model_params(model_pipeline)
            overall_results = model_pipeline if isinstance(model_pipeline, list) else []
            display_evaluation_results(overall_results)
            
            available_stocks = [result['stock'] for result in overall_results] if overall_results else []
            selected_stocks = st.multiselect("Select stocks for prediction", available_stocks)
            
            if selected_stocks:
                st.subheader("Enter Macroeconomic Parameters")
                gdp = st.slider("GDP Growth (%)", min_value=-5, max_value=5, value=2)
                inflation = st.slider("Inflation (%)", min_value=0, max_value=10, value=2)
                interest_rate = st.slider("Interest Rate (%)", min_value=0, max_value=10, value=2)
                vix = st.slider("VIX Index", min_value=10, max_value=100, value=20)
                
                if st.button("Simulate and Predict"):
                    for stock_name in selected_stocks:
                        stock_result = next((result for result in overall_results if result['stock'] == stock_name), None)
                        
                        if stock_result:
                            model = stock_result.get('model')  # Ensure correct model is selected
                            
                            # Debug: Verify that the correct model is being selected
                            st.write(f"Selected model for {stock_name}: {type(model)}")
                            
                            if model:
                                # Ensure the file name has only one .xlsx extension
                                if not stock_name.endswith('.xlsx'):
                                    stock_name += '.xlsx'  # Add the extension only if it's not already present
                                
                                stock_file = f"stockdata/{stock_name}"
                                
                                # Debug: Check if the file exists
                                st.write(f"Looking for stock data file at: {stock_file}")
                                
                                if os.path.exists(stock_file):
                                    stock_data = pd.read_excel(stock_file, engine='openpyxl')
                                    stock_data['Date'] = pd.to_datetime(stock_data['Date'])
                                    stock_data.set_index('Date', inplace=True)

                                    input_data = np.array([[gdp, inflation, interest_rate, vix]])
                                    try:
                                        if isinstance(model, XGBRegressor) and hasattr(model, 'booster_'):
                                            st.write(f"Predicting with XGBRegressor for {stock_name}...")
                                            predicted_returns = model.predict(input_data)
                                            show_stock_performance(stock_data, predicted_returns)
                                        elif isinstance(model, Pipeline):
                                            if hasattr(model.named_steps['model'], 'booster_'):
                                                st.write(f"Predicting with pipeline model for {stock_name}...")
                                                predicted_returns = model.predict(input_data)
                                                show_stock_performance(stock_data, predicted_returns)
                                            else:
                                                st.error(f"Model for {stock_name} is not fitted.")
                                        else:
                                            st.error(f"Unexpected model type for {stock_name}.")
                                    except NotFittedError as e:
                                        st.error(f"Model for {stock_name} is not fitted: {e}")
                                else:
                                    st.error(f"Stock data file for {stock_name} does not exist at {stock_file}.")
                            else:
                                st.error(f"No model found for {stock_name}.")
                        else:
                            st.error(f"No model found for {stock_name}.")
        else:
            st.error("Failed to load the model. Please check the file format.")
    else:
        st.warning("Please upload a pre-trained model file.")

if __name__ == "__main__":
    main()

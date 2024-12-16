import streamlit as st
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline  # Import Pipeline
from sklearn.exceptions import NotFittedError
from xgboost import XGBRegressor

# Load the overall model evaluation file
def load_model_evaluation(file_path):
    try:
        # Load the pre-trained model (pipeline or model)
        model = joblib.load(file_path)
        
        # Check if the model is a pipeline or just a trained model
        if isinstance(model, list):
            st.write("Loaded model is a list (perhaps multiple models).")
            # You might need to handle this case specifically
        elif isinstance(model, Pipeline):
            st.write("Loaded model is a pipeline.")
            if hasattr(model.named_steps['model'], 'booster_'):
                st.write("Pipeline model is fitted.")
            else:
                st.error("Pipeline model is not fitted.")
        elif isinstance(model, XGBRegressor):
            st.write("Loaded model is a trained XGBRegressor (not in a pipeline).")
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
        params = model.named_steps['model'].get_params()
        st.write(params)
    elif isinstance(model, XGBRegressor):
        params = model.get_params()
        st.write(params)
    else:
        st.error("Model is not recognized.")

# Display model evaluation results
def display_evaluation_results(overall_results):
    st.subheader("Model Evaluation Results")
    for result in overall_results:
        st.write(f"**Stock**: {result['stock']}")
        st.write(f"**Accuracy (R-squared %)**: {result['accuracy']:.4f}%")
        st.write(f"**R-squared**: {result['r2_score']:.4f}")
        st.write(f"**Mean Squared Error**: {result['mean_squared_error']:.4f}")
        st.write("="*40)

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

    # Step 1: Upload the pre-trained model file
    uploaded_model = st.file_uploader("Upload pre-trained model", type="pkl")
    
    if uploaded_model is not None:
        # Load the overall model evaluation
        model_pipeline = load_model_evaluation(uploaded_model)
        
        if model_pipeline:
            # Step 2: Display model parameters and evaluation results
            display_model_params(model_pipeline)
            # Example of evaluation results (replace with actual loading logic for `overall_results`)
            overall_results = joblib.load(uploaded_model)  # Assuming this is the evaluation result
            display_evaluation_results(overall_results)
            
            # Step 3: Select multiple stocks for prediction
            available_stocks = [result['stock'] for result in overall_results]  # Assuming `overall_results` contains stock names
            selected_stocks = st.multiselect("Select stocks for prediction", available_stocks)

            if selected_stocks:
                # Step 4: Input macroeconomic parameters with sliders
                st.subheader("Enter Macroeconomic Parameters")
                
                gdp = st.slider("GDP Growth (%)", min_value=-5, max_value=5, value=2)
                inflation = st.slider("Inflation (%)", min_value=0, max_value=10, value=2)
                interest_rate = st.slider("Interest Rate (%)", min_value=0, max_value=10, value=2)
                vix = st.slider("VIX Index", min_value=10, max_value=100, value=20)

                # Step 5: Predict and simulate scenarios
                if st.button("Simulate and Predict"):
                    # For each selected stock, show historical performance and predictions
                    for stock_name in selected_stocks:
                        stock_result = next(result for result in overall_results if result['stock'] == stock_name)
                        model = stock_result['model']
                        
                        # Load historical stock data for selected stock
                        stock_data = pd.read_excel(f"stockdata/{stock_name}", engine='openpyxl')
                        stock_data['Date'] = pd.to_datetime(stock_data['Date'])
                        stock_data.set_index('Date', inplace=True)

                        # Prepare input data for prediction (macroeconomic parameters)
                        input_data = np.array([[gdp, inflation, interest_rate, vix]])
                        
                        try:
                            # Check if the model is fitted before making predictions
                            if isinstance(model, XGBRegressor) and hasattr(model, 'booster_'):
                                # Predict the stock returns using the pre-trained model
                                predicted_returns = model.predict(input_data)

                                # Show historical performance and predicted returns
                                show_stock_performance(stock_data, predicted_returns)
                            elif isinstance(model, Pipeline):
                                if hasattr(model.named_steps['model'], 'booster_'):
                                    predicted_returns = model.predict(input_data)
                                    show_stock_performance(stock_data, predicted_returns)
                                else:
                                    st.error(f"Model for {stock_name} is not fitted.")
                            else:
                                st.error(f"Unexpected model type for {stock_name}.")
                        except NotFittedError as e:
                            st.error(f"Model for {stock_name} is not fitted: {e}")

    else:
        st.warning("Please upload a pre-trained model file.")

if __name__ == "__main__":
    main()

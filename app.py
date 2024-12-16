import os
import pandas as pd
import joblib
import streamlit as st
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import xgboost as xgb

# Function to load stock data from a given stock name
def load_stock_data(stock_name):
    # Ensure correct path and extension
    file_path = f"stockdata/{stock_name}.xlsx"
    if os.path.exists(file_path):
        stock_data = pd.read_excel(file_path, engine='openpyxl')
        return stock_data
    else:
        st.error(f"File not found: {file_path}")
        return None

# Function to display model parameters
def display_model_params(model_pipeline):
    if isinstance(model_pipeline, Pipeline):
        # Extract model parameters
        params = model_pipeline.named_steps['model'].get_params()
        st.write("Model Parameters:")
        for param, value in params.items():
            st.write(f"{param}: {value}")
    else:
        st.write("Model is not recognized or does not have parameters.")

# Function to display evaluation results
def display_evaluation_results(overall_results):
    st.write("Model Evaluation Results:")
    for result in overall_results:
        st.write(f"Stock: {result['stock']}")
        st.write(f"Accuracy (R-squared %): {result['accuracy']:.4f}%")
        st.write(f"R-squared: {result['r2_score']:.4f}")
        st.write(f"Mean Squared Error: {result['mean_squared_error']:.4f}")
        st.write("=" * 40)

# Main function for Streamlit app
def main():
    st.title('Stock Prediction Using Macroeconomic Data')
    
    # File upload for pre-trained model
    uploaded_model = st.file_uploader("Upload pre-trained model", type=["pkl"])
    if uploaded_model is not None:
        # Load the model
        model_pipeline = joblib.load(uploaded_model)
        
        # Display model parameters
        display_model_params(model_pipeline)
        
        # Load the overall evaluation results (pkl file)
        overall_results_filename = 'results/overall_gdp_vix_xgb_model_evaluation.pkl'
        if os.path.exists(overall_results_filename):
            overall_results = joblib.load(overall_results_filename)
            display_evaluation_results(overall_results)
        else:
            st.error("No overall evaluation results file found.")
        
        # Stock selection for prediction
        st.subheader("Select Stocks for Prediction")
        stock_files = [file.replace('.xlsx', '') for file in os.listdir('stockdata') if file.endswith('.xlsx')]
        selected_stocks = st.multiselect("Choose stock(s)", stock_files)

        if selected_stocks:
            # Input for macroeconomic parameters
            st.subheader("Input Macroeconomic Parameters for Prediction")
            gdp = st.number_input("GDP", value=2.0)  # example default value
            inflation = st.number_input("Inflation", value=3.0)  # example default value
            interest_rate = st.number_input("Interest Rate", value=5.0)  # example default value
            vix_value = st.slider("VIX Value", 10, 40, 20)  # example VIX range from 10 to 40
            
            # Prepare input data for prediction
            input_data = pd.DataFrame({
                'GDP': [gdp] * len(selected_stocks),
                'Inflation': [inflation] * len(selected_stocks),
                'Interest Rate': [interest_rate] * len(selected_stocks),
                'VIX': [vix_value] * len(selected_stocks)
            })

            # Predict returns for selected stocks
            for stock_name in selected_stocks:
                stock_data = load_stock_data(stock_name)
                if stock_data is not None:
                    # Display historical stock performance
                    st.subheader(f"Historical Performance for {stock_name}")
                    st.line_chart(stock_data['Close'])

                    # Predict stock returns
                    predicted_returns = model_pipeline.predict(input_data)
                    stock_data['Predicted Returns'] = predicted_returns
                    st.write(f"Predicted Returns for {stock_name}:")
                    st.line_chart(stock_data[['Close', 'Predicted Returns']])

        else:
            st.write("Please select one or more stocks for prediction.")

if __name__ == "__main__":
    main()

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
        else:
            st.error("Failed to load the model. Please check the file format.")
    else:
        st.warning("Please upload a pre-trained model file.")

if __name__ == "__main__":
    main()

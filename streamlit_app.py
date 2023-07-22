import streamlit as st
import numpy as np
import pickle

# Load the trained model
with open('logreg_model.pkl', 'rb') as file:
    model = pickle.load(file)

# Define the logistic regression coefficients
coefficients = {
    "annual_inc": -0.3891,
    "loan_amnt": -1.5778,
    "int_rate": -1.7471,
    "const": 1.9908,
}

# Helper function to calculate the odds based on coefficients
def calculate_odds(inputs):
    odds = np.exp(np.dot(inputs, np.array(list(coefficients.values()))))
    return odds

# Helper function to calculate the probability of being in a positive class
def calculate_probability(inputs):
    odds = calculate_odds(inputs)
    probability = odds / (1 + odds)
    return probability

# Create the Streamlit app
def main():
    st.title("Logistic Regression Model Deployment")

    # Collect user inputs for the features
    st.header("Enter Feature Values")
    annual_income = st.number_input("Annual Income", value=50000)
    loan_amount = st.number_input("Loan Amount", value=10000)
    interest_rate = st.number_input("Interest Rate (%)", value=10.0)

    # Prepare the feature values for prediction
    input_values = np.array([annual_income, loan_amount, interest_rate, 1])

    # Calculate the probability of being in a positive class
    probability = calculate_probability(input_values)

    st.subheader("Model Prediction")
    st.write(f"The predicted probability of being in a positive class is: {probability:.2f}")

    st.subheader("Model Information")
    st.write("The model was trained using a logistic regression algorithm.")
    st.write("It predicts the probability of a binary outcome (positive or negative class).")
    st.write("The model uses features: Annual Income, Loan Amount, and Interest Rate.")
    st.write("The odds ratio shows the odds of a positive outcome for each unit increase in the feature.")
    st.write("An odds ratio greater than 1 indicates a positive effect, while less than 1 indicates a negative effect.")

if __name__ == "__main__":
    main()

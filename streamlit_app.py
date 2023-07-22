import streamlit as st
import pickle
import pandas as pd

# Load the trained logistic regression model
with open('logistic_regression.pkl', 'rb') as file:
    model = pickle.load(file)

# Streamlit app code
def main():
    st.title('Credit Scoring Prediction App')
    st.write('Enter the feature values and click the "Predict" button to get credit scoring prediction.')

    # Define the feature names and default values
    feature_names = [
        'Loan Amount (USD)',
        'Interest Rate (%)',
        'Annual Income (USD)',
        'Debt To Income (%)',
        'Total Account',
        'Initial List Status',
        'Last Payment Amount (USD)',
        'Payment Time in Month'
    ]
    default_values = [1000, 10, 50000, None, 10, 0, 500, 1]

    # Input fields for the features required for prediction
    feature_values = {}
    for feature_name, default_value in zip(feature_names, default_values):
        if feature_name == 'Debt To Income (%)':
            # Calculate Debt To Income (%) based on Loan Amount and Annual Income
            loan_amount = feature_values['Loan Amount (USD)'] if 'Loan Amount (USD)' in feature_values else default_values[0]
            annual_income = feature_values['Annual Income (USD)'] if 'Annual Income (USD)' in feature_values else default_values[2]
            feature_values['Debt To Income (%)'] = (loan_amount / annual_income) * 100 if annual_income != 0 else None
            st.number_input('Debt To Income (%)', value=feature_values['Debt To Income (%)'], step=0.01, key='debt_to_income')
        else:
            if feature_name == 'Initial List Status':
                # Display Initial List Status as a dropdown with 0 and 1 options
                feature_values[feature_name] = st.selectbox(feature_name, [0, 1], index=0)
            else:
                feature_values[feature_name] = st.number_input(feature_name, value=default_value, step=1)

    # Prepare the input data as a DataFrame with the same column names as in the training data
    input_data = pd.DataFrame(feature_values, index=[0])

    # Make predictions when the user clicks the "Predict" button
    if st.button('Predict'):
        # Perform any necessary preprocessing on the input data (e.g., scaling, encoding)
        # Make predictions using the loaded model
        prediction = model.predict(input_data)
        proba = model.predict_proba(input_data)[:, 1]

        # Display the prediction and probability
        st.write(f'Prediction: {"Good" if prediction[0] == 0 else "Bad"}')
        st.write(f'Probability of Default: {proba[0]:.4f}')

if __name__ == '__main__':
    main()

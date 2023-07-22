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

    # Define the original feature names and default values
    feature_names = [
        'loan_amnt',
        'int_rate',
        'annual_inc',
        'dti',
        'total_acc',
        'initial_list_status',
        'last_pymnt_amnt',
        'pymnt_time'
    ]
    default_values = [1000, 10, 50000, None, 10, 0, 500, 1]

    # Define the aliases for display
    aliases = {
        'loan_amnt': 'Loan Amount (USD)',
        'int_rate': 'Interest Rate (%)',
        'annual_inc': 'Annual Income (USD)',
        'dti': 'Debt To Income (%)',
        'total_acc': 'Total Account',
        'initial_list_status': 'Initial List Status',
        'last_pymnt_amnt': 'Last Payment Amount (USD)',
        'pymnt_time': 'Payment Time in Month'
    }

    # Input fields for the features required for prediction
    feature_values = {}
    for feature_name, default_value in zip(feature_names, default_values):
        if feature_name == 'dti':
            # Calculate Debt To Income (%) based on Loan Amount and Annual Income
            loan_amount = feature_values.get('loan_amnt', default_values[0])
            annual_income = feature_values.get('annual_inc', default_values[2])
            feature_values['dti'] = (loan_amount / annual_income) * 100 if annual_income != 0 else None
            st.number_input(aliases[feature_name], value=feature_values['dti'], step=0.01, key='debt_to_income')
        else:
            if feature_name == 'initial_list_status':
                # Display Initial List Status as a dropdown with 0 and 1 options
                feature_values[feature_name] = st.selectbox(aliases[feature_name], [0, 1], index=0)
            else:
                feature_values[feature_name] = st.number_input(aliases[feature_name], value=default_value, step=1)

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

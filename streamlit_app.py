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
    feature_names = ['Loan Amount (USD)', 'Interest Rate (%)', 'Annual Income (USD)']
    default_values = [1000, 10, 50000]

    # Input fields for the features required for prediction
    feature_values = {}
    for feature_name, default_value in zip(feature_names, default_values):
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

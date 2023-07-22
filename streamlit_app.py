import streamlit as st
import pickle
import pandas as pd

# Load the trained logistic regression model
with open('logreg_model.pkl', 'rb') as file:
    model = pickle.load(file)

# Streamlit app code
def main():
    st.title('Credit Scoring Prediction App')
    st.write('Enter the feature values and click the "Predict" button to get credit scoring prediction.')

    # Input fields for the features required for prediction
    loan_amnt_ = st.number_input('Loan Amount', value=100.0, step=1.0)
    int_rate_ = st.number_input('Interest Rate', value=10.0, step=0.1)
    annual_inc_ = st.number_input('Annual Income', value=50000.0, step=1000.0)

    # Prepare the input data as a DataFrame with the same column names as in the training data
    input_data = pd.DataFrame({
        'loan_amnt_': [loan_amnt_],
        'int_rate_': [int_rate_],
        'annual_inc_': [annual_inc_]
    })

    # Make predictions when the user clicks the "Predict" button
    if st.button('Predict'):
        # Perform any necessary preprocessing on the input data (e.g., scaling, encoding)
        # Make predictions using the loaded model
        prediction = model.predict(input_data)
        proba = model.predict_proba(input_data)[:, 1]

        # Display the prediction and probability
        st.write(f'Prediction: {"Low Default Risk" if prediction[0] == 0 else "High Default Risk"}')
        st.write(f'Probability of Default: {proba[0]:.4f}')

if __name__ == '__main__':
    main()

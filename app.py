import streamlit as st
import joblib
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report

# Load the trained logistic regression model
logreg = joblib.load('logistic_regression_model.pkl')

# Streamlit app code
def main():
    st.title('Logistic Regression Model App')
    st.write('Enter your input features and click the "Predict" button to get predictions.')

    # Input fields for the features required for prediction
    feature1 = st.number_input('Feature 1', value=0.0)
    feature2 = st.number_input('Feature 2', value=0.0)
    # Add more input fields for other features as needed

    # Prepare the input data as a NumPy array or Pandas DataFrame
    input_data = [[feature1, feature2]]
    
    # Make predictions when the user clicks the "Predict" button
    if st.button('Predict'):
        # Perform any necessary preprocessing on the input data (e.g., scaling, encoding)
        # Make predictions using the loaded model
        y_pred = logreg.predict(input_data)
        # Display the prediction
        st.write(f'Prediction: {y_pred[0]}')

if __name__ == '__main__':
    main()

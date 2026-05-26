import joblib
import pandas as pd

# Load Model
model = joblib.load('models/loan_model.pkl')

# Sample Customer Data (use raw column names expected by training)
data = {
    'Loan_ID': ['LOAN_SAMPLE'],
    'Gender': ['Male'],
    'Married': ['Yes'],
    'Dependents': ['0'],
    'Education': ['Graduate'],
    'Self_Employed': ['No'],
    'ApplicantIncome': [5000],
    'CoapplicantIncome': [0],
    'LoanAmount': [150.0],
    'Loan_Amount_Term': [360.0],
    'Credit_History': [1.0],
    'Property_Area': ['Urban'],
}

df = pd.DataFrame(data)

# Prediction
X = df.drop(columns=['Loan_ID'])
X_encoded = pd.get_dummies(X, dummy_na=True)

# Load training-time metadata so we can align encoded feature columns
bundle = model
if isinstance(bundle, dict):
    model = bundle['model']
    feature_columns = bundle['feature_columns']
else:
    feature_columns = None

# Align columns with the exact training-time dummy columns.
# Ensure we PASS a DataFrame with the same column set/order the model saw at fit-time.
if feature_columns is not None:
    X_aligned = X_encoded.reindex(columns=feature_columns, fill_value=0)
else:
    X_aligned = X_encoded

# If the stored feature list indicates the model expects more columns than our
# single-row encoding produced, they will be added by reindex above.
prediction = model.predict(X_aligned)




if prediction[0] == 1:
    print("Loan Approved")
else:
    print("Loan Rejected")
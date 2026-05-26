import pandas as pd

# Load Dataset
df = pd.read_csv('data/test_Y3wMUE5_7gLdaTN.csv')

print(df.head())

# Check Missing Values
print(df.isnull().sum())

# Fill Missing Values
df['LoanAmount'] = df['LoanAmount'].fillna(df['LoanAmount'].median())

df['Gender'] = df['Gender'].fillna(df['Gender'].mode()[0])

df['Married'] = df['Married'].fillna(df['Married'].mode()[0])

# Convert Categorical Data
df['Gender'] = df['Gender'].map({'Male':1, 'Female':0})

df['Married'] = df['Married'].map({'Yes':1, 'No':0})

df['Education'] = df['Education'].map({
    'Graduate':1,
    'Not Graduate':0
})

df['Loan_Status'] = df['Loan_Status'].map({
    'Y':1,
    'N':0
})

print(df.head())

# Save Cleaned Data
df.to_csv('data/train_u6lujuX_CVtuZ9i.csv', index=False)

print("Data Cleaning Completed")

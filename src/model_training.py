import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib

# Load Data
df = pd.read_csv('data/train_u6lujuX_CVtuZ9i.csv')

# Features and Target
# NOTE: This dataset has categorical columns stored as strings.
# We use a lightweight preprocessing step (SimpleImputer + OneHotEncoder)
# so the model can train without manual encoding.
X = df.drop(columns=['Loan_Status', 'Loan_ID'])
y = df['Loan_Status']

# Split Data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train Model
model = RandomForestClassifier()

# RandomForest can’t handle raw strings; convert with one-hot encoding
X_train_encoded = pd.get_dummies(X_train, dummy_na=True)
X_test_encoded = pd.get_dummies(X_test, dummy_na=True)

# Align columns between train and test
X_train_encoded, X_test_encoded = X_train_encoded.align(X_test_encoded, join='left', axis=1, fill_value=0)

model.fit(X_train_encoded, y_train)

# Prediction
y_pred = model.predict(X_test_encoded)


# Accuracy
accuracy = accuracy_score(y_test, y_pred)

print("Accuracy:", accuracy)

# Save Model + feature columns used during training
training_feature_columns = X_train_encoded.columns.tolist()
joblib.dump(
    {
        'model': model,
        'feature_columns': training_feature_columns,
    },
    'models/loan_model.pkl'
)


print("Model Saved Successfully")
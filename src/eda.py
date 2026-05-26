import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('data/train_u6lujuX_CVtuZ9i.csv')

# Loan Status Count
sns.countplot(x='Loan_Status', data=df)
plt.title("Loan Approval Count")
plt.show()

# Income Distribution
sns.histplot(df['ApplicantIncome'], bins=30)
plt.title("Applicant Income Distribution")
plt.show()

# Loan Amount Distribution
sns.histplot(df['LoanAmount'], bins=30)
plt.title("Loan Amount Distribution")
plt.show()

# Education vs Loan Status
sns.countplot(x='Education', hue='Loan_Status', data=df)
plt.title("Education vs Loan Status")
plt.show()
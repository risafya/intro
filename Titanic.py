import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt

# Load datasets
train_data = pd.read_csv(r"C:\Users\Administrator\AppData\python tugas\train.csv")
test_data = pd.read_csv(r"C:\Users\Administrator\AppData\python tugas\test.csv")

# Select features (excluding 'Name' and converting 'Sex' to numeric)
X = train_data[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare']]
y = train_data['Survived']

# Convert 'Sex' to numeric (Male = 0, Female = 1)
X['Sex'] = X['Sex'].map({'male': 0, 'female': 1})

# Fill missing values in 'Age' and 'Fare'
X['Age'].fillna(X['Age'].median(), inplace=True)
X['Fare'].fillna(X['Fare'].median(), inplace=True)

# Split into train and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Logistic Regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# Predict on validation set
y_pred = model.predict(X_val)

# Print accuracy score
accuracy = accuracy_score(y_val, y_pred)
print(f"Validation Accuracy: {accuracy:.4f}")

# Load the test dataset
test_data = pd.read_csv(r"C:\Users\Administrator\AppData\python tugas\test.csv")

# Apply the same preprocessing as train data
test_data['Sex'] = test_data['Sex'].map({'male': 0, 'female': 1})
test_data['Age'].fillna(test_data['Age'].median(), inplace=True)
test_data['Fare'].fillna(test_data['Fare'].median(), inplace=True)

# Select the same features used in training
X_test = test_data[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare']]

# Make predictions
test_data['Survived'] = model.predict(X_test)

# Create a submission file
submission = test_data[['PassengerId', 'Survived']]
submission.to_csv("submission.csv", index=False)

print("Submission file created successfully!")

# Load submission file
load_data = pd.read_csv("submission.csv")

# Plot the number of passengers who survived vs. not survived
plt.figure(figsize=(6, 4))
plt.bar(load_data["Survived"].value_counts().index, load_data["Survived"].value_counts().values, color=['red', 'green'])
plt.xticks([0, 1], ["Not Survived", "Survived"])
plt.xlabel("Survival Status")
plt.ylabel("Count")
plt.title("Survival Distribution in Predictions")
plt.show()


# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load the heart disease dataset
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"
column_names = [
    "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
    "thalach", "exang", "oldpeak", "slope", "ca", "thal", "target"
]
data = pd.read_csv(url, names=column_names, na_values="?")

# Handling missing values (if any)
data = data.dropna()

# Define features and target variable
X = data.drop(columns=['target'])
y = data['target']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a RandomForestClassifier (you can choose a different algorithm if you prefer)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")

# Assume you have a new person's details in the form of a dictionary
new_person = {
    "age": 60,
    "sex": 1,  # 1 for male, 0 for female
    "cp": 3,
    "trestbps": 145,
    "chol": 233,
    "fbs": 1,
    "restecg": 0,
    "thalach": 150,
    "exang": 0,
    "oldpeak": 2.3,
    "slope": 0,
    "ca": 0,
    "thal": 1
}

# Convert the dictionary to a pandas DataFrame
new_data = pd.DataFrame([new_person])

# Predict the heart condition for the new person
prediction = model.predict(new_data)
if prediction[0] == 1:
    result = "Person has heart disease"
else:
    result = "Person's heart condition is normal"

# Store data in a database (you'll need to set up a database first)
# Example using SQLite:
# import sqlite3
# conn = sqlite3.connect('heart_data.db')
# new_data.to_sql('person_data', conn, if_exists='append', index=False)

# Print the prediction result
print(result)

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

data = {
    'age': [48, 7, 62, 48, 51, 60, 68, 24, 52, 53],
    'bp': [80, 50, 80, 70, 80, 90, 70, 80, 100, 90],
    'sg': [1.02, 1.02, 1.01, 1.005, 1.01, 1.015, 1.01, 1.015, 1.015, 1.02],
    'al': [1, 4, 2, 4, 2, 3, 0, 2, 3, 2],
    'su': [0, 0, 3, 0, 0, 0, 0, 4, 0, 0],
    'rbc': [1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
    'pc': [0, 0, 0, 1, 0, 0, 0, 1, 1, 0],
    'pcc': [0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
    'ba': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    'bgr': [121, 146, 138, 70, 210, 138, 70, 484, 70, 138],
    'bu': [36, 18, 53, 56, 26, 25, 51, 44, 33, 48],
    'sc': [1.2, 0.8, 1.8, 3.8, 1.4, 1.1, 2.7, 1.3, 0.9, 1.2],
    'sod': [0, 0, 0, 111, 0, 142, 136, 130, 0, 0],
    'pot': [0.0, 0.0, 0.0, 2.5, 0.0, 3.2, 4.0, 3.7, 0.0, 0.0],
    'hemo': [15.4, 11.3, 9.6, 11.2, 11.6, 12.2, 12.4, 12.4, 10.8, 9.5],
    'pcv': [44, 38, 31, 32, 35, 39, 36, 44, 33, 29],
    'wbcc': [7800, 6000, 7500, 6700, 7300, 7800, 7100, 6900, 9600, 7800],
    'rbcc': [5.2, 4.0, 3.8, 3.9, 4.6, 4.4, 3.8, 5.0, 4.0, 4.0],
    'htn': [1, 1, 1, 1, 0, 1, 0, 1, 1, 1],
    'dm': [1, 1, 0, 0, 0, 1, 0, 0, 0, 1],
    'cad': [0, 0, 1, 1, 0, 0, 0, 0, 1, 1],
    'appet': [1, 1, 0, 1, 0, 1, 1, 1, 1, 1],
    'pe': [0, 0, 0, 1, 0, 1, 0, 0, 0, 0],
    'ane': [0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
    'class': [0, 0, 1, 1, 0, 0, 0, 0, 0, 0]
}

df = pd.DataFrame(data)
df.to_csv('kidney_disease.csv')

# Read the dataset
df = pd.read_csv('kidney_disease.csv')

# Check the unique values in 'class' column
print(df['class'].unique())  # This will help ensure that there are two distinct classes.

# Define features (X) and target (y)
X = df.drop('class', axis=1)
y = df['class']

# Scale the features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy*100}")
print(classification_report(y_test, y_pred, zero_division=1))

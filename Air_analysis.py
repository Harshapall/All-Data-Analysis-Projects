import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Load the dataset
df = pd.read_csv('air_data.csv')

# Step 2: Train the Model
X = df.drop(["Air Quality", "Date", "Unnamed: 0"], axis=1)  # Drop 'Unnamed: 0'
y = df["Air Quality"]

model = LogisticRegression(max_iter=1000)
model.fit(X, y)

# Print training accuracy
train_predictions = model.predict(X)
train_accuracy = accuracy_score(y, train_predictions)
print(f"Training Accuracy: {train_accuracy*100:.2f}%")

# Step 3: Make predictions on the new data
new_data = [
    {
        "Temperature (C)": 23,
        "Humidity (%)": 45,
        "CO2 Level (ppm)": 210,
        "Oxygen Level (%)": 20.5
    },
    {
        "Temperature (C)": 24,
        "Humidity (%)": 42,
        "CO2 Level (ppm)": 420,
        "Oxygen Level (%)": 20.2
    },
    {
        "Temperature (C)": 22,
        "Humidity (%)": 46,
        "CO2 Level (ppm)": 400,
        "Oxygen Level (%)": 20.7
    }
]

new_df = pd.DataFrame(new_data)

# Drop 'Unnamed: 0' if it exists
if 'Unnamed: 0' in new_df.columns:
    new_df = new_df.drop('Unnamed: 0', axis=1)

# Make predictions
predictions = model.predict(new_df)

# Print predictions
print("Predictions:")
for i, pred in enumerate(predictions):
    print(f"Data Point {i+1}: {'Normal' if pred == 1 else 'Abnormal'}")

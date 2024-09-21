import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler

# Loading the Data Model
df = pd.read_csv('water_potability.csv')
# Drop rows with missing values
df = df.dropna()

# Then proceed with the rest of your code using df_clean

X = df.drop('Potability', axis=1)
y = df['Potability']

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 5: Choose and Train the Model
model = LogisticRegression(max_iter=1000)  # Increased max_iter to ensure convergence
model.fit(x_train, y_train)

# Step 6: Evaluate the Model
y_pred = model.predict(x_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100}")
print(classification_report(y_test, y_pred, zero_division=1))

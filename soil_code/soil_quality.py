import mysql.connector
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score
import numpy as np

# Assuming you have a MySQL database set up with a table named 'soil_predictions'
# Columns: pH, organic_matter, nitrogen, phosphorus, potassium, texture, moisture, sand, silt, clay, crop_yield_prediction

# Connect to MySQL
conn = mysql.connector.connect(
    host='localhost',
    user='root',
    password='root',
    database='my_blocks'
)
cursor = conn.cursor()

# Sample dataset
data = {
    'pH': [6.5, 7.2, 6.8, 7.0, 6.2, 6.7, 7.5, 6.0, 6.3, 7.8],
    'organic_matter': [2.3, 1.8, 2.0, 2.5, 1.5, 2.2, 1.6, 2.8, 2.1, 1.4],
    'nitrogen': [30, 25, 28, 32, 20, 29, 22, 35, 27, 18],
    'phosphorus': [20, 15, 18, 22, 12, 19, 14, 25, 17, 10],
    'potassium': [40, 35, 38, 42, 30, 39, 32, 45, 37, 28],
    'texture': ['loam', 'sandy', 'clayey', 'silt loam', 'sandy loam', 'clayey loam', 'silt', 'sandy clay loam', 'loamy sand', 'silty clay'],
    'moisture': [60, 50, 55, 65, 45, 58, 70, 50, 40, 75],
    'sand': [30, 60, 25, 20, 70, 28, 15, 50, 80, 10],
    'silt': [40, 20, 35, 45, 15, 37, 55, 25, 10, 60],
    'clay': [30, 20, 40, 35, 15, 35, 30, 25, 10, 30],
    'crop_yield': [2000, 1800, 2200, 1900, 1600, 2100, 1800, 2300, 1500, 2000]
}

soil_df = pd.DataFrame(data)

# Display the first few rows of the DataFrame



# Define the features and target variable
features = ['pH', 'organic_matter', 'nitrogen', 'phosphorus', 'potassium', 'moisture', 'sand', 'silt', 'clay', 'texture']
target = 'crop_yield'

# Separate categorical and numerical features
categorical_features = ['texture']
numerical_features = list(set(features) - set(categorical_features))

# Apply one-hot encoding to categorical features
preprocessor = ColumnTransformer(
    transformers=[
        ('num', 'passthrough', numerical_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ]
)

# Extract features and target variable from the DataFrame
X = soil_df[features]
y = soil_df[target]

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Apply the preprocessor to the training and testing data
X_train_encoded = preprocessor.fit_transform(X_train)
X_test_encoded = preprocessor.transform(X_test)

# Train a linear regression model
model = LinearRegression()
model.fit(X_train_encoded, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test_encoded)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
print(f'\nMean Squared Error on Test Set: {mse}')

# Now, let's use the trained model to predict crop yields for new soil data
new_soil_data = pd.DataFrame([
    {'pH': 6.0, 'organic_matter': 2.0, 'nitrogen': 25, 'phosphorus': 15, 'potassium': 35, 'moisture': 55, 'sand': 30, 'silt': 35, 'clay': 35, 'texture': 'loam'},
    {'pH': 7.5, 'organic_matter': 1.2, 'nitrogen': 18, 'phosphorus': 10, 'potassium': 28, 'moisture': 70, 'sand': 20, 'silt': 45, 'clay': 35, 'texture': 'clayey'},
    {'pH': 6.8, 'organic_matter': 2.5, 'nitrogen': 30, 'phosphorus': 20, 'potassium': 40, 'moisture': 60, 'sand': 25, 'silt': 40, 'clay': 35, 'texture': 'silt loam'},
    {'pH': 7.2, 'organic_matter': 1.8, 'nitrogen': 28, 'phosphorus': 18, 'potassium': 38, 'moisture': 55, 'sand': 25, 'silt': 35, 'clay': 40, 'texture': 'clayey loam'},
    {'pH': 6.5, 'organic_matter': 2.3, 'nitrogen': 30, 'phosphorus': 20, 'potassium': 40, 'moisture': 60, 'sand': 30, 'silt': 40, 'clay': 30, 'texture': 'loam'},
])

# Apply the preprocessor to the new soil data
new_soil_data_encoded = preprocessor.transform(new_soil_data)

# Make predictions on new soil data
new_predictions = model.predict(new_soil_data_encoded)

# Display the predicted values
print('\nPredicted Crop Yields for New Soil Data:')

print(new_predictions)
# Convert numpy.int64 to native Python int before inserting into the database


# Store the predictions in the MySQL database
# Convert the predictions to native Python int before inserting into the database
new_predictions_list = [int(prediction) for prediction in new_predictions]

# Store the predictions in the MySQL database using executemany
new_data_points = new_soil_data.values.tolist()
for i, prediction in enumerate(new_predictions_list):
    new_data_point = new_data_points[i]
    new_data_point.append(prediction)
    cursor.execute("""
        INSERT INTO soil_predictions (pH,organic_matter,nitrogen,phosphorus,potassium,moisture,sand,silt,
clay,texture,crop_yield_prediction)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s,%s)
    """, new_data_point)

# Commit the changes and close the connection
conn.commit()
conn.close()

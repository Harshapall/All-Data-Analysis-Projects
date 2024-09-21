import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import mysql.connector

# Function to connect to MySQL database
def connect_to_database():
    return mysql.connector.connect(
        host="localhost",
        user="root",
        password="root",
        database="my_blocks",
        port = 3306
    )

# Function to create the table if it doesn't exist
def create_table(cursor):
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS urine_data (
            id INT AUTO_INCREMENT PRIMARY KEY,
            image_path VARCHAR(255) NOT NULL,
            label INT NOT NULL
        )
    """)

# Function to insert data into the MySQL database
def insert_data(cursor, image_path, label):
    cursor.execute("INSERT INTO urine_data (image_path, label) VALUES (%s, %s)", (image_path, label))

# Function for image preprocessing
def preprocess_image(image_path):
    image = cv2.imread(image_path)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    resized_image = cv2.resize(gray_image, (100, 100))
    flattened_image = resized_image.flatten()
    return flattened_image

# Function to predict using the trained model
def predict(model, image_path):
    image = preprocess_image(image_path)
    prediction = model.predict([image])
    return prediction[0]

# Sample data with image paths and labels
sample_data = [
    {"image_path": "E:\\MY_BLOCKS\\project_3\\image_2.png", "label": 0},
    {"image_path":"E:\\MY_BLOCKS\\project_3\\image_3.png", "label": 1}
    # Add more samples with image paths and labels
]

# Connect to the MySQL database
db_connection = connect_to_database()
cursor = db_connection.cursor()

# Create the table if it doesn't exist
create_table(cursor)

# Insert sample data into the MySQL database
for data in sample_data:
    image_path = data["image_path"]
    label = data["label"]
    insert_data(cursor, image_path, label)

# Commit the changes and close the database connection
db_connection.commit()
db_connection.close()

# Extract features and labels from the database
db_connection = connect_to_database()
cursor = db_connection.cursor()
cursor.execute("SELECT image_path, label FROM urine_data")
data_from_database = cursor.fetchall()

features = []
labels = []

for data in data_from_database:
    image_path = data[0]
    label = data[1]
    image = preprocess_image(image_path)
    features.append(image)
    labels.append(label)

db_connection.close()

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

# Train logistic regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")

# Example: Make predictions on a new image in real-time
new_image_path = "E:\\MY_BLOCKS\\project_3\\test_image.jpeg"
prediction = predict(model, new_image_path)
print(f"Prediction for {new_image_path}: {prediction}")
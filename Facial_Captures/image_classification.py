import os
import mysql.connector
from mysql.connector import Error
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input, decode_predictions
import numpy as np

# Load the MobileNetV2 model pre-trained on ImageNet data
model = MobileNetV2(weights='imagenet')

# Connect to MySQL database
def create_connection(host, user, password, database):
    try:
        connection = mysql.connector.connect(
            host=host,
            user=user,
            password=password,
            database=database
        )
        if connection.is_connected():
            print("Connected to MySQL database")
            return connection
    except Error as e:
        print(f"Error: {e}")
    return None

# Map ImageNet labels to custom categories
def map_to_custom_categories(predicted_label):
    sports_keywords = ["sports", "athlete", "game", "court"]
    accidents_keywords = ["accident", "collision", "crash", "emergency"]
    events_keywords = ["event", "concert", "festival", "celebration"]

    for keyword in sports_keywords:
        if keyword in predicted_label.lower():
            return "Sports"

    for keyword in accidents_keywords:
        if keyword in predicted_label.lower():
            return "Accidents"

    for keyword in events_keywords:
        if keyword in predicted_label.lower():
            return "Events"

    return "Other"

# Insert prediction results into the database
def insert_prediction(connection, image_path, image_name, predicted_label, confidence, category):
    try:
        cursor = connection.cursor()
        insert_query = "INSERT INTO predictions (image_path, image_name, predicted_label, confidence, category) VALUES (%s, %s, %s, %s, %s)"
        cursor.execute(insert_query, (image_path, image_name, predicted_label, confidence, category))
        connection.commit()
        print("Prediction results inserted into the database")
    except Error as e:
        print(f"Error: {e}")

# Classify image and insert results into the database
def classify_and_store(image_path):
    # Load and preprocess the image
    img = image.load_img(image_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)

    # Predict the category of the image
    predictions = model.predict(img_array)

    # Decode predictions
    decoded_predictions = decode_predictions(predictions, top=1)[0]
    predicted_label = decoded_predictions[0][1]
    confidence = float(decoded_predictions[0][2])

    # Get image name from the path
    image_name = os.path.basename(image_path)

    # Map ImageNet label to custom categories
    category = map_to_custom_categories(predicted_label)

    # Insert results into the database
    connection = create_connection(host='localhost',
                                   user='root',
                                   password='root',
                                   database='my_blocks')
    if connection:
        insert_prediction(connection, image_path, image_name, predicted_label, confidence, category)
        connection.close()

    # Print the results
    print(f"Image Name: {image_name}")
    print(f"Predicted Label: {predicted_label}")
    print(f"Confidence: {confidence:.2f}")
    print(f"Category: {category}")



# Example usage
image_path = "E:\\MY_BLOCKS\\Facial_Captures\\accident.jpg"
classify_and_store(image_path)
image_path = "E:\\MY_BLOCKS\\Facial_Captures\\event.jpg"
classify_and_store(image_path)
image_path = "E:\\MY_BLOCKS\\Facial_Captures\\sports.jpg"
classify_and_store(image_path)
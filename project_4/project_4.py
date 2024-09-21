import cv2
import time
import numpy as np
import mysql.connector

# Load the pre-trained Haar Cascade face detection model
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Initialize video capture
cap = cv2.VideoCapture('gym_video.mp4')  # Replace with your video file

# Connect to MySQL database
db = mysql.connector.connect(
    host="localhost",
    user="root",
    password="root",
    database="my_blocks"
)

cursor = db.cursor()

# Define a dictionary to store member information
members = {}

# Minimum distance to consider two faces as the same person
min_face_distance = 100

def calculate_face_distance(face1, face2):
    # Calculate the Euclidean distance between two face rectangles
    return np.sqrt(np.sum((np.array(face1) - np.array(face2)) ** 2))

def find_matching_member(face_rect):
    for key, value in members.items():
        # Calculate the distance between the detected face and the stored face for each member
        face_distance = calculate_face_distance(face_rect, value['face_rect'])
        if face_distance < min_face_distance:
            return key
    return None

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Convert frame to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Reset member IDs for each frame
    for key in list(members.keys()):
        members[key]['assigned'] = False

    for (x, y, w, h) in faces:
        # Draw rectangle around face
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

        # Check if the face matches an existing member
        matching_member_id = find_matching_member((x, y, x + w, y + h))

        if matching_member_id is not None:
            # Detected face is close to an existing member, assign the same ID
            members[matching_member_id]['assigned'] = True
            face_found = True

            # Get current time
            current_time = time.time()

            # Draw member ID and running time
            cv2.putText(frame, f'Member ID: {matching_member_id}', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            cv2.putText(frame, f'Running Time: {current_time - members[matching_member_id]["start_time"]:.2f} s', (x, y + h + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        else:
            # Face does not match any existing member, assign a new ID
            new_id = len(members) + 1
            members[new_id] = {'assigned': True, 'start_time': time.time(), 'face_rect': (x, y, x + w, y + h)}

            # Insert data into MySQL database
            insert_query = "INSERT INTO member_data (member_id, start_time, face_rect) VALUES (%s, FROM_UNIXTIME(%s), %s)"
            insert_data = (new_id, members[new_id]['start_time'], str(members[new_id]['face_rect']))
            cursor.execute(insert_query, insert_data)
            db.commit()

            # Draw member ID and running time
            cv2.putText(frame, f'Member ID: {new_id}', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            cv2.putText(frame, f'Running Time: 0.00 s', (x, y + h + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Display the frame
    cv2.imshow('Gym Video', frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Close the database connection
db.close()

cap.release()
cv2.destroyAllWindows()

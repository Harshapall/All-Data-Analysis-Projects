import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=Warning)

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

data = """Patient_ID,Age,Gender,Temperature,Symptoms,Preexisting_Conditions,Test_Result,Date_Sampled,City
11,37,Male,37.3,"Fever,Cough,Shortness of breath,Muscle aches,Headache",Diabetes,Positive,2023-11-12,New York
12,50,Female,36.6,"Fever,Cough,Headache",Hypertension,Positive,2023-11-11,Los Angeles
13,70,Male,38.1,"Cough,Shortness of breath,Fatigue,Headache",Obesity,Negative,2023-11-14,Chicago
14,65,Female,36.8,"Fever,Cough,Shortness of breath,Fatigue,Muscle aches,Headache",Asthma,Positive,2023-11-15,Houston
15,40,Non-binary,37.0,"Fever,Cough,Shortness of breath,Loss of taste/smell",Cancer,Positive,2023-11-13,Phoenix
16,60,Male,37.5,"Cough,Shortness of breath,Muscle aches",Heart Disease,Negative,2023-11-16,New York
17,48,Female,37.2,"Fever,Shortness of breath,Fatigue",Diabetes,Positive,2023-11-17,Los Angeles
18,75,Non-binary,36.7,"Cough,Shortness of breath,Loss of taste/smell",Asthma,Negative,2023-11-18,Chicago
19,55,Male,37.0,"Fever,Cough,Shortness of breath,Fatigue,Headache",Heart Disease,Positive,2023-11-19,Houston
20,44,Female,36.9,"Fever,Cough,Shortness of breath,Muscle aches",Obesity,Positive,2023-11-20,Phoenix
21,42,Female,36.9,"Fever,Cough,Shortness of breath",Hypertension,Positive,2023-11-21,New York
22,55,Male,37.3,"Fever,Cough,Shortness of breath,Muscle aches",Diabetes,Positive,2023-11-22,Los Angeles
23,48,Female,36.6,"Fever,Cough,Headache",Asthma,Positive,2023-11-23,Chicago
24,68,Male,38.0,"Cough,Shortness of breath,Fatigue",Heart Disease,Positive,2023-11-24,Houston
25,30,Non-binary,37.2,"Fever,Cough,Shortness of breath,Loss of taste/smell,Muscle aches",Obesity,Negative,2023-11-25,Phoenix
26,53,Female,37.4,"Cough,Shortness of breath,Muscle aches,Headache",Diabetes,Positive,2023-11-26,New York
27,35,Male,36.8,"Fever,Cough,Shortness of breath",Obesity,Positive,2023-11-27,Los Angeles
28,45,Non-binary,37.0,"Cough,Shortness of breath,Loss of taste/smell",Asthma,Negative,2023-11-28,Chicago
29,62,Female,37.1,"Fever,Cough,Shortness of breath,Fatigue",Diabetes,Positive,2023-11-29,Houston
30,40,Male,37.5,"Fever,Cough,Shortness of breath,Muscle aches",Asthma,Positive,2023-11-30,Phoenix
31,38,Non-binary,36.9,"Fever,Cough,Shortness of breath",Heart Disease,Positive,2023-12-01,New York
32,56,Male,37.2,"Cough,Shortness of breath,Muscle aches,Headache",Obesity,Positive,2023-12-02,Los Angeles
33,41,Female,36.7,"Fever,Cough,Headache",Diabetes,Positive,2023-12-03,Chicago
34,70,Male,38.1,"Cough,Shortness of breath,Fatigue,Headache",Heart Disease,Negative,2023-12-04,Houston
35,58,Female,36.8,"Fever,Cough,Shortness of breath,Fatigue,Muscle aches,Headache",Obesity,Positive,2023-12-05,Phoenix
36,47,Male,37.0,"Fever,Cough,Shortness of breath,Loss of taste/smell",Cancer,Positive,2023-12-06,New York
37,52,Non-binary,37.5,"Cough,Shortness of breath,Muscle aches",Asthma,Positive,2023-12-07,Los Angeles
38,37,Female,36.9,"Fever,Cough,Shortness of breath",Diabetes,Negative,2023-12-08,Chicago
39,63,Male,37.3,"Fever,Cough,Shortness of breath,Muscle aches",Heart Disease,Positive,2023-12-09,Houston
40,39,Non-binary,37.1,"Fever,Cough,Shortness of breath,Fatigue",Cancer,Positive,2023-12-10,Phoenix
41,59,Male,36.7,"Cough,Shortness of breath,Loss of taste/smell",Asthma,Positive,2023-12-11,New York
42,43,Female,36.9,"Fever,Cough,Headache",Diabetes,Positive,2023-12-12,Los Angeles
43,67,Male,38.2,"Cough,Shortness of breath,Fatigue",Heart Disease,Positive,2023-12-13,Chicago
44,32,Non-binary,37.4,"Fever,Cough,Shortness of breath,Loss of taste/smell,Muscle aches",Obesity,Positive,2023-12-14,Houston
45,51,Female,36.8,"Cough,Shortness of breath,Muscle aches,Headache",Cancer,Positive,2023-12-15,Phoenix
46,46,Male,36.5,"Fever,Cough,Shortness of breath",Asthma,Negative,2023-12-16,New York
47,36,Non-binary,37.2,"Fever,Cough,Shortness of breath,Muscle aches",Diabetes,Positive,2023-12-17,Los Angeles
48,54,Female,36.6,"Fever,Cough,Headache",Obesity,Positive,2023-12-18,Chicago
49,66,Male,38.0,"Cough,Shortness of breath,Fatigue,Headache",Heart Disease,Positive,2023-12-19,Houston
50,33,Non-binary,37.1,"Fever,Cough,Shortness of breath,Loss of taste/smell",Cancer,Positive,2023-12-20,Phoenix
"""

# Define the CSV file path
csv_file_path = 'advanced_infection_analysis_dataset.csv'

# Write data to CSV file
with open(csv_file_path, mode='w') as file:
    file.write(data)

print(f'Data saved to {csv_file_path}')

# Load the dataset
df = pd.read_csv('advanced_infection_analysis_dataset.csv')

# Preprocessing: Convert categorical variables into numerical form using one-hot encoding
df = pd.get_dummies(df, columns=['Gender', 'City'], drop_first=True)

# Define features (X) and target variable (y)
X = df[['Age', 'Temperature', 'Gender_Male', 'Gender_Non-binary', 'City_Los Angeles', 'City_New York', 'City_Phoenix']]
y = df['Test_Result']
# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the logistic regression model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

print(f'Accuracy: {accuracy*100:.2f}')
print(report)
sample_patient_data = [
    {
        'Age': 45,
        'Temperature': 37.1,
        'Gender_Male': 0,  # If Male, set to 1; otherwise, set to 0
        'Gender_Non-binary': 0,  # If Non-binary, set to 1; otherwise, set to 0
        'City_Los Angeles': 1,  # If from Los Angeles, set to 1; otherwise, set to 0
        'City_New York': 0,  # If from New York, set to 1; otherwise, set to 0
        'City_Phoenix': 0  # If from Phoenix, set to 1; otherwise, set to 0
    },
    {
        'Age': 30,
        'Temperature': 36.8,
        'Gender_Male': 1,
        'Gender_Non-binary': 0,
        'City_Los Angeles': 0,
        'City_New York': 1,
        'City_Phoenix': 0
    },
    {
        'Age': 55,
        'Temperature': 37.2,
        'Gender_Male': 0,
        'Gender_Non-binary': 1,
        'City_Los Angeles': 0,
        'City_New York': 0,
        'City_Phoenix': 1
    },
    {
        'Age': 40,
        'Temperature': 37.0,
        'Gender_Male': 0,
        'Gender_Non-binary': 0,
        'City_Los Angeles': 0,
        'City_New York': 1,
        'City_Phoenix': 0
    },
    {
        'Age': 35,
        'Temperature': 36.9,
        'Gender_Male': 1,
        'Gender_Non-binary': 0,
        'City_Los Angeles': 0,
        'City_New York': 0,
        'City_Phoenix': 1
    },
    {
        'Age': 55,
        'Temperature': 37.2,
        'Gender_Male': 0,
        'Gender_Non-binary': 1,
        'City_Los Angeles': 0,
        'City_New York': 0,
        'City_Phoenix': 1
    },{
    'Age': 48,
    'Temperature': 37.2,
    'Gender_Male': 0,
    'Gender_Non-binary': 0,
    'City_Los Angeles': 1,
    'City_New York': 0,
    'City_Phoenix': 0
},
    {
        'Age': 40,
        'Temperature': 37.0,
        'Gender_Male': 0,
        'Gender_Non-binary': 0,
        'City_Los Angeles': 0,
        'City_New York': 1,
        'City_Phoenix': 0
    },
    {
        'Age': 35,
        'Temperature': 36.9,
        'Gender_Male': 1,
        'Gender_Non-binary': 0,
        'City_Los Angeles': 0,
        'City_New York': 0,
        'City_Phoenix': 1
    },
    {
        'Age': 50,
        'Temperature': 37.5,
        'Gender_Male': 1,
        'Gender_Non-binary': 0,
        'City_Los Angeles': 1,
        'City_New York': 0,
        'City_Phoenix': 0
    },
    {
        'Age': 45,
        'Temperature': 37.1,
        'Gender_Male': 0,
        'Gender_Non-binary': 0,
        'City_Los Angeles': 0,
        'City_New York': 1,
        'City_Phoenix': 0
    }
]

# Create a DataFrame for prediction
sample_df = pd.DataFrame(sample_patient_data)

# Predict using the model
predicted_result = model.predict(sample_df)

# Convert the predictions to the original labels (Positive or Negative)
predicted_result_labels = ['Positive' if result == 1 else 'Negative' for result in predicted_result]

# Combine the sample data with the predicted results
sample_results = [{'Sample_Patient_Data': data, 'Predicted_Test_Result': result} for data, result in zip(sample_patient_data, predicted_result_labels)]

# Print the sample results
for idx, result in enumerate(sample_results):
    print(f'Sample {idx + 1}:')
    print(result)
    print('\n')
import mysql.connector

# Establish connection to your MySQL server
conn = mysql.connector.connect(
    host='localhost',
    user='root',
    password='',
    database='my_blocks'
)

cursor = conn.cursor()

# Assuming you have a table named 'infection_results' with columns:
# 'Age', 'Temperature', 'Gender_Male', 'Gender_Non-binary',
# 'City_Los Angeles', 'City_New York', 'City_Phoenix', 'Predicted_Test_Result'
for result in sample_results:
    sample_data = result['Sample_Patient_Data']
    predicted_result = result['Predicted_Test_Result']

    cursor.execute(
        "INSERT INTO infection_results (Age, Temperature, Gender_Male, Gender_Non_binary, "
        "City_Los_Angeles, City_New_York, City_Phoenix, Predicted_Test_Result) "
        "VALUES (%s, %s, %s, %s, %s, %s, %s, %s)",
        (sample_data['Age'], sample_data['Temperature'], sample_data['Gender_Male'],
         sample_data['Gender_Non-binary'], sample_data['City_Los Angeles'],
         sample_data['City_New York'], sample_data['City_Phoenix'], predicted_result)
    )

# Commit the changes and close the cursor and connection
conn.commit()
cursor.close()
conn.close()

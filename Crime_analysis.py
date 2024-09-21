import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

crime_data = [
    {
        "City": "City1",
        "Population": 100000,
        "Violent_Crimes": 500,
        "Property_Crimes": 2000,
        "Murders": 5,
        "Robberies": 50,
        "Burglaries": 200,
        "Thefts": 1500,
        "Arsons": 10
    },
    {
        "City": "City2",
        "Population": 150000,
        "Violent_Crimes": 700,
        "Property_Crimes": 2500,
        "Murders": 3,
        "Robberies": 70,
        "Burglaries": 180,
        "Thefts": 2000,
        "Arsons": 8
    },
    {
        "City": "City3",
        "Population": 80000,
        "Violent_Crimes": 300,
        "Property_Crimes": 1200,
        "Murders": 2,
        "Robberies": 30,
        "Burglaries": 100,
        "Thefts": 900,
        "Arsons": 5
    },
    {
        "City": "City4",
        "Population": 120000,
        "Violent_Crimes": 600,
        "Property_Crimes": 1800,
        "Murders": 4,
        "Robberies": 60,
        "Burglaries": 150,
        "Thefts": 1300,
        "Arsons": 7
    },
    {
        "City": "City5",
        "Population": 90000,
        "Violent_Crimes": 400,
        "Property_Crimes": 1500,
        "Murders": 3,
        "Robberies": 40,
        "Burglaries": 120,
        "Thefts": 1100,
        "Arsons": 6
    },
    {
        "City": "City6",
        "Population": 110000,
        "Violent_Crimes": 550,
        "Property_Crimes": 1700,
        "Murders": 5,
        "Robberies": 55,
        "Burglaries": 140,
        "Thefts": 1300,
        "Arsons": 8
    },
    {
        "City": "City7",
        "Population": 70000,
        "Violent_Crimes": 250,
        "Property_Crimes": 1000,
        "Murders": 1,
        "Robberies": 25,
        "Burglaries": 80,
        "Thefts": 900,
        "Arsons": 4
    },
    {
        "City": "City8",
        "Population": 95000,
        "Violent_Crimes": 450,
        "Property_Crimes": 1600,
        "Murders": 2,
        "Robberies": 45,
        "Burglaries": 130,
        "Thefts": 1200,
        "Arsons": 6
    },
    {
        "City": "City9",
        "Population": 85000,
        "Violent_Crimes": 350,
        "Property_Crimes": 1400,
        "Murders": 2,
        "Robberies": 35,
        "Burglaries": 110,
        "Thefts": 1000,
        "Arsons": 5
    },
    {
        "City": "City10",
        "Population": 105000,
        "Violent_Crimes": 500,
        "Property_Crimes": 1800,
        "Murders": 3,
        "Robberies": 50,
        "Burglaries": 140,
        "Thefts": 1600,
        "Arsons": 7
    },
    {
        "City": "City11",
        "Population": 75000,
        "Violent_Crimes": 300,
        "Property_Crimes": 1200,
        "Murders": 2,
        "Robberies": 30,
        "Burglaries": 90,
        "Thefts": 900,
        "Arsons": 4
    },
    {
        "City": "City12",
        "Population": 130000,
        "Violent_Crimes": 600,
        "Property_Crimes": 2000,
        "Murders": 4,
        "Robberies": 70,
        "Burglaries": 170,
        "Thefts": 1600,
        "Arsons": 9
    },
    {
        "City": "City13",
        "Population": 80000,
        "Violent_Crimes": 350,
        "Property_Crimes": 1400,
        "Murders": 3,
        "Robberies": 40,
        "Burglaries": 110,
        "Thefts": 1000,
        "Arsons": 5
    },
    {
        "City": "City14",
        "Population": 95000,
        "Violent_Crimes": 400,
        "Property_Crimes": 1600,
        "Murders": 3,
        "Robberies": 45,
        "Burglaries": 130,
        "Thefts": 1200,
        "Arsons": 6
    },
    {
        "City": "City15",
        "Population": 90000,
        "Violent_Crimes": 450,
        "Property_Crimes": 1800,
        "Murders": 3,
        "Robberies": 50,
        "Burglaries": 140,
        "Thefts": 1300,
        "Arsons": 7
    }
]

df=pd.DataFrame(crime_data)
df.to_csv("crime_data.csv")


# Define features and target variable
features = ['Population', 'Violent_Crimes', 'Property_Crimes', 'Murders', 'Robberies', 'Burglaries', 'Thefts', 'Arsons']
target = 'HighCrime'  # Assuming you have a binary variable indicating high or low crime rate

# Add a new column 'HighCrime' based on some threshold (for example, if Violent_Crimes > 500, HighCrime = 1, else 0)
df['HighCrime'] = (df['Violent_Crimes'] > 500).astype(int)

# Standardize the features
scaler = StandardScaler()
df[features] = scaler.fit_transform(df[features])

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(df[features], df[target], test_size=0.2, random_state=42)

# Initialize and train the logistic regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# Predict on the test set
y_pred = model.predict(X_test)

# Evaluate the model
from sklearn.metrics import accuracy_score, classification_report

accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

print(f'Accuracy: {accuracy}')
print(f'Classification Report:\n{report}')
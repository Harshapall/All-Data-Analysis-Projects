import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import mysql.connector

# Assuming 'number_of_employees' is your target variable
target = 'number_of_employees'
features = ['revenue', 'market_cap', 'average_wages', 'job_creation_rate']

# Create a dictionary for your dataset
data_set = {
    'Wipro': {
        'number_of_employees': 180000,
        'revenue': 900000000000,
        'market_cap': 4500000000000,
        'number_of_businesses_by_industry': {
            'IT Services': 1200,
            'Consulting': 500,
            'Software Development': 800
        },
        'average_wages': 90000,
        'job_creation_rate': 0.06,
        'business_formation_closure_rates': {
            'Formation': 0.025,
            'Closure': 0.015
        },
        'customer_satisfaction': 4.5,
        'employee_satisfaction': 4.2,
        'innovation_index': 75,
        'technology_adoption_rate': 0.8,
        'global_presence': True,
        'partnerships': ['Microsoft', 'Oracle', 'SAP']
    },
    'Tech Mahindra': {
        'number_of_employees': 150000,
        'revenue': 720000000000,
        'market_cap': 3000000000000,
        'number_of_businesses_by_industry': {
            'IT Services': 800,
            'BPO': 300,
            'Telecom': 200
        },
        'average_wages': 85000,
        'job_creation_rate': 0.05,
        'business_formation_closure_rates': {
            'Formation': 0.02,
            'Closure': 0.01
        },
        'customer_satisfaction': 4.4,
        'employee_satisfaction': 4.0,
        'innovation_index': 70,
        'technology_adoption_rate': 0.7,
        'global_presence': True,
        'partnerships': ['Cisco', 'Huawei', 'IBM']
    },
    'IBM': {
        'number_of_employees': 300000,
        'revenue': 1200000000000,
        'market_cap': 5000000000000,
        'number_of_businesses_by_industry': {
            'IT Services': 1500,
            'Consulting': 800,
            'Software Development': 1200
        },
        'average_wages': 95000,
        'job_creation_rate': 0.04,
        'business_formation_closure_rates': {
            'Formation': 0.018,
            'Closure': 0.012
        },
        'customer_satisfaction': 4.7,
        'employee_satisfaction': 4.3,
        'innovation_index': 80,
        'technology_adoption_rate': 0.75,
        'global_presence': True,
        'partnerships': ['Red Hat', 'Salesforce', 'VMware']
    },
    'Cyient': {
        'number_of_employees': 15000,
        'revenue': 80000000000,
        'market_cap': 150000000000,
        'number_of_businesses_by_industry': {
            'Engineering Services': 400,
            'Data Analytics': 150,
            'IT Solutions': 200
        },
        'average_wages': 75000,
        'job_creation_rate': 0.03,
        'business_formation_closure_rates': {
            'Formation': 0.012,
            'Closure': 0.008
        },
        'customer_satisfaction': 4.2,
        'employee_satisfaction': 3.8,
        'innovation_index': 65,
        'technology_adoption_rate': 0.7,
        'global_presence': False,
        'partnerships': ['Siemens', 'Airbus', 'Boeing']
    },
    'Softura': {
        'number_of_employees': 8000,
        'revenue': 50000000000,
        'market_cap': 100000000000,
        'number_of_businesses_by_industry': {
            'Software Development': 300,
            'Consulting': 150,
            'Digital Transformation': 100
        },
        'average_wages': 85000,
        'job_creation_rate': 0.04,
        'business_formation_closure_rates': {
            'Formation': 0.015,
            'Closure': 0.01
        },
        'customer_satisfaction': 4.3,
        'employee_satisfaction': 4.0,
        'innovation_index': 70,
        'technology_adoption_rate': 0.75,
        'global_presence': True,
        'partnerships': ['Microsoft', 'Amazon', 'Google']
    },
    'APITAVATech': {
        'number_of_employees': 12000,
        'revenue': 60000000000,
        'market_cap': 80000000000,
        'number_of_businesses_by_industry': {
            'IT Services': 400,
            'Cloud Solutions': 200,
            'Cybersecurity': 100
        },
        'average_wages': 90000,
        'job_creation_rate': 0.05,
        'business_formation_closure_rates': {
            'Formation': 0.018,
            'Closure': 0.012
        },
        'customer_satisfaction': 4.5,
        'employee_satisfaction': 4.2,
        'innovation_index': 75,
        'technology_adoption_rate': 0.8,
        'global_presence': True,
        'partnerships': ['Cisco', 'IBM', 'Fortinet']
    },
    'Dharani Info Technologies': {
        'number_of_employees': 10000,
        'revenue': 40000000000,
        'market_cap': 60000000000,
        'number_of_businesses_by_industry': {
            'IT Services': 300,
            'Digital Marketing': 100,
            'Web Development': 150
        },
        'average_wages': 80000,
        'job_creation_rate': 0.03,
        'business_formation_closure_rates': {
            'Formation': 0.01,
            'Closure': 0.008
        },
        'customer_satisfaction': 4.0,
        'employee_satisfaction': 3.7,
        'innovation_index': 60,
        'technology_adoption_rate': 0.6,
        'global_presence': False,
        'partnerships': ['Google', 'Facebook', 'HubSpot']
    },
    'IIC Technologies': {
        'number_of_employees': 5000,
        'revenue': 30000000000,
        'market_cap': 50000000000,
        'number_of_businesses_by_industry': {
            'Geospatial Solutions': 200,
            'Engineering Services': 100,
            'Marine Technology': 50
        },
        'average_wages': 70000,
        'job_creation_rate': 0.02,
        'business_formation_closure_rates': {
            'Formation': 0.008,
            'Closure': 0.005
        },
        'customer_satisfaction': 4.2,
        'employee_satisfaction': 3.9,
        'innovation_index': 65,
        'technology_adoption_rate': 0.7,
        'global_presence': True,
        'partnerships': ['Esri', 'Bentley', 'Hexagon']
    },
    'HCL Technologies': {
        'number_of_employees': 200000,
        'revenue': 1000000000000,
        'market_cap': 5500000000000,
        'number_of_businesses_by_industry': {
            'IT Services': 1600,
            'Infrastructure Services': 500,
            'Engineering Services': 700
        },
        'average_wages': 95000,
        'job_creation_rate': 0.07,
        'business_formation_closure_rates': {
            'Formation': 0.03,
            'Closure': 0.02
        },
        'customer_satisfaction': 4.6,
        'employee_satisfaction': 4.1,
        'innovation_index': 78,
        'technology_adoption_rate': 0.77,
        'global_presence': True,
        'partnerships': ['SAP', 'Microsoft', 'Salesforce']
    },
    'Infosys': {
        'number_of_employees': 250000,
        'revenue': 1100000000000,
        'market_cap': 6000000000000,
        'number_of_businesses_by_industry': {
            'IT Services': 1800,
            'Consulting': 700,
            'Software Development': 1000
        },
        'average_wages': 92000,
        'job_creation_rate': 0.05,
        'business_formation_closure_rates': {
            'Formation': 0.028,
            'Closure': 0.018
        },
        'customer_satisfaction': 4.8,
        'employee_satisfaction': 4.4,
        'innovation_index': 82,
        'technology_adoption_rate': 0.8,
        'global_presence': True,
        'partnerships': ['Oracle', 'Google', 'Amazon']
    },
    'Satyam Venture Engineering Services': {
        'number_of_employees': 18000,
        'revenue': 85000000000,
        'market_cap': 120000000000,
        'number_of_businesses_by_industry': {
            'Engineering Services': 600,
            'Product Design': 200,
            'Aerospace Solutions': 150
        },
        'average_wages': 78000,
        'job_creation_rate': 0.04,
        'business_formation_closure_rates': {
            'Formation': 0.015,
            'Closure': 0.01
        },
        'customer_satisfaction': 4.3,
        'employee_satisfaction': 4.0,
        'innovation_index': 70,
        'technology_adoption_rate': 0.75,
        'global_presence': False,
        'partnerships': ['Boeing', 'Airbus', 'Lockheed Martin']
    },
    'Sankhya Infotech': {
        'number_of_employees': 7000,
        'revenue': 60000000000,
        'market_cap': 90000000000,
        'number_of_businesses_by_industry': {
            'IT Services': 300,
            'Simulation Software': 100,
            'Aviation Solutions': 50
        },
        'average_wages': 82000,
        'job_creation_rate': 0.03,
        'business_formation_closure_rates': {
            'Formation': 0.012,
            'Closure': 0.008
        },
        'customer_satisfaction': 4.1,
        'employee_satisfaction': 3.7,
        'innovation_index': 65,
        'technology_adoption_rate': 0.7,
        'global_presence': False,
        'partnerships': ['Dassault Systèmes', 'ANSYS', 'Siemens']
    },
    'KTree Computer Solutions': {
        'number_of_employees': 8000,
        'revenue': 45000000000,
        'market_cap': 70000000000,
        'number_of_businesses_by_industry': {
            'Software Development': 250,
            'Web Development': 150,
            'Mobile App Development': 100
        },
        'average_wages': 80000,
        'job_creation_rate': 0.03,
        'business_formation_closure_rates': {
            'Formation': 0.01,
            'Closure': 0.008
        },
        'customer_satisfaction': 4.2,
        'employee_satisfaction': 3.9,
        'innovation_index': 65,
        'technology_adoption_rate': 0.7,
        'global_presence': False,
        'partnerships': ['Microsoft', 'AWS', 'Google']
    }
}


# Convert the dictionary to a Pandas DataFrame
data = pd.DataFrame.from_dict(data_set, orient='index')

# Extract 'number_of_businesses_by_industry' and convert it into separate columns
data = pd.concat([data.drop('number_of_businesses_by_industry', axis=1),
                  data['number_of_businesses_by_industry'].apply(pd.Series)], axis=1)

# Explore your data and perform any necessary preprocessing
# For example:
# - Handle missing values (if any)
# - Encode categorical variables (if any)
# - Scale numerical features (not required for Random Forest, but useful for some models)

# Split the data into training and testing sets
train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)

# Separate features and target variable in the training and testing sets
X_train = train_data[features]
y_train = train_data[target]
X_test = test_data[features]
y_test = test_data[target]

# Standardize features using StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Initialize and train the machine learning model (Random Forest Regressor in this case)
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test_scaled)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f'Mean Squared Error: {mse}')
print(f'R-squared: {r2}')

# Now, you can use this trained model to make predictions on new data
# For example, you can predict the number of employees for a new company
new_data = pd.DataFrame({
    'company_name': ['Tech', 'Grow', 'Mark', 'Brander', 'Policy', 'Marketer'],
    'revenue': [80000000000, 120000000000, 60000000000, 30000000000, 150000000000, 90000000000],
    'market_cap': [150000000000, 180000000000, 90000000000, 45000000000, 250000000000, 120000000000],
    'average_wages': [75000, 90000, 60000, 45000, 80000, 70000],
    'job_creation_rate': [0.03, 0.05, 0.02, 0.01, 0.04, 0.03]
})
# Store all details of a company in the MySQL database
new_data_features = new_data[['revenue', 'market_cap', 'average_wages', 'job_creation_rate']]
new_data_scaled = scaler.transform(new_data_features)

# Make predictions using the trained model
predicted_employees = model.predict(new_data_scaled)
print(f'Predicted Number of Employees: {predicted_employees}')

# Store all details of a company in the MySQL database
# Replace 'your_database', 'your_username', 'your_password', and 'your_table' with your actual values
db_connection = mysql.connector.connect(
    host="localhost",
    user="root",
    password="root",
    database="my_blocks"
)
companys = ['Tech', 'Grow', 'Mark', 'Brander', 'Policy', 'Marketer']
cursor = db_connection.cursor()

for i,name in enumerate(companys):
    # Insert all details into the database
    insert_query = f"INSERT INTO employe_data(company_name, revenue, market_cap, average_wages, job_creation_rate, predicted_employees) " \
                   f"VALUES ('{name}', {new_data['revenue'][i]}, {new_data['market_cap'][i]}, " \
                   f"{new_data['average_wages'][i]}, {new_data['job_creation_rate'][i]}, {predicted_employees[i]})"
    cursor.execute(insert_query)

# Commit the changes and close the connections
db_connection.commit()
cursor.close()
db_connection.close()

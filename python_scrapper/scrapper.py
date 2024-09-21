from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import re
import time
import mysql.connector

conn = mysql.connector.connect(
    host="localhost",
    user="root",
    password="",
    database="startup_company"
)

def get_contact_info(company_name):
    driver = webdriver.Chrome(
        executable_path="E:\\MY_BLOCKS\\python_scrapper\\chromedriver.exe")

    try:
        driver.get('https://www.google.com/')
        WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.NAME, 'q')))

        search_box = driver.find_element(By.NAME, 'q')
        search_box.send_keys(company_name + " contact information")
        search_box.submit()

        # Wait for the search results to load
        WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.CSS_SELECTOR, 'h3')))

        # Assuming the first result contains the contact information, click on it
        first_result = driver.find_element(By.CSS_SELECTOR, 'h3')
        driver.execute_script("arguments[0].click();", first_result)

        # Wait for the page to load
        time.sleep(5)  # You can adjust this as needed

        # Call the function to extract email and phone number
        extract_email_and_phone(driver.page_source, company_name)

    except Exception as e:
        print(f"Error: {e}")
    finally:
        driver.quit()


def extract_email_and_phone(page_source, company_name):
    # This is a basic regex pattern to match email addresses
    email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,7}\b'

    # This regex pattern matches phone numbers in a basic format
    phone_pattern = r'\b\d{3}[-.\s]?\d{3}[-.\s]?\d{4}\b'

    email = re.findall(email_pattern, page_source)
    phone = re.findall(phone_pattern, page_source)
    email = email[0] if email else "Not found"
    phone = phone[0] if phone else "Not found"
    name = company_name


    cursor = conn.cursor()
    cursor.execute("SELECT * FROM company_details")
    rows = cursor.fetchall()
    cursor.execute("INSERT into company_details (name, email, phone) values (%s, %s, %s)", (name, email, phone))
    conn.commit()
    print(f"Saved {len(rows)+1} company details")


startup_names = [
    "Tricog",
    "Milkbasket",
    "Mfine",
    "Country Delight",
    "Finbox",
    "Jumbotail",
    "Ninjacart",
    "PharmEasy",
    "Cars24",
    "CarDekho",
    "Ninjacart",
    "Bounce",
    "Licious"
]
print("Total 13 Companies Details Storing ... ... ... .. .. ")
for company_name in startup_names:
    get_contact_info(company_name)

cursor = conn.cursor()
cursor.execute("SELECT * FROM company_details")
rows = cursor.fetchall()
for row in rows:
    print('Company_Name :',row[1])
    print('Email :',row[2])
    print('Phone_number :',row[3])
cursor.close()
conn.close()

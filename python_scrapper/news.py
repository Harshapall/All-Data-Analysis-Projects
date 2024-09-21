import requests
from bs4 import BeautifulSoup
import mysql.connector
from datetime import datetime

# Define the URL to scrape
url = "https://www.edweek.org/"

# Database connection parameters
db_config = {
    "host": "localhost",
    "user": "root",
    "password": "",
    "database": "my_blocks"
}

# Connect to the MySQL database
conn = mysql.connector.connect(**db_config)

# Create a cursor object to interact with the database
cursor = conn.cursor()

# Define the INSERT SQL query
query = """
INSERT INTO news (DOC_VEND_ID, MEMBER_ID, DOC_TITL, DOC_DET, DOC_URL, DOC_CATEGRY, DOC_SDATE, DOC_EDATE, DOC_PUBDATE, portalid, parentportalid)
VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
"""

# Send an HTTP GET request to the URL
response = requests.get(url)

# Check if the request was successful (status code 200)
if response.status_code == 200:
    # Parse the HTML content of the page
    soup = BeautifulSoup(response.content, 'html.parser')

    # Find all elements with the specified classes
    promo_elements = soup.find_all('div', class_='m-promo__content')

    # Loop through each promo element
    for promo in promo_elements:
        # Extract the link from the 'a' tag within the 'm-promo__title' element
        link_element = promo.find('a', class_='m-promo__title')

        # Check if the link element exists
        if link_element:
            link = link_element['href']
            print(f"Processing link: {link}")  # Display the link being processed

            # Extract the title from the base URL
            base_title = promo.find('div', class_='m-promo__description')
            if base_title:
                article_title = base_title.text.strip()
            else:
                article_title = "Title not found"

            # Send an HTTP GET request to the extracted link
            inner_response = requests.get(link)

            # Check if the request to the inner link was successful
            if inner_response.status_code == 200:
                # Parse the HTML content of the inner page
                inner_soup = BeautifulSoup(inner_response.content, 'html.parser')

                # Find the content element
                content_element = inner_soup.find('div', class_='a-text Module')

                # Extract the first 3 or 4 <p> tags for content
                if content_element:
                    p_tags = content_element.find_all('p')
                    article_content = "\n".join([p.text.strip() for p in p_tags[:4]])  # Join the first 3 or 4 paragraphs
                else:
                    article_content = "Content not found"

                # Check if both title and content are not placeholders
                if article_title != "Title not found" and article_content != "Content not found":
                    # Define values for database insertion
                    DOC_VEND_ID = 108  # Example value, replace with appropriate data
                    emp_id =1127  # Example value, replace with appropriate data
                    category = "news"  # Example value, replace with appropriate data
                    current_datetime = datetime.now()  # Current date and time
                    portalid = 135559  # Example value, replace with appropriate data
                    parentportalid = 11058  # Example value, replace with appropriate data

                    # Execute the INSERT query with the extracted data
                    values = (
                        DOC_VEND_ID,
                        emp_id,
                        article_title,
                        article_content,
                        link,
                        category,
                        current_datetime,
                        current_datetime,
                        current_datetime,
                        portalid,
                        parentportalid
                    )

                    cursor.execute(query, values)
                    conn.commit()

                    print("Data inserted into the database.")
                else:
                    print("Title or content not found, skipping database insertion.")

            else:
                print(f"Failed to fetch inner page: {link}")

        else:
            print("Link not found in promo element.")

else:
    print(f"Failed to fetch the main page: {url}")

# Close the database connection
cursor.close()
conn.close()
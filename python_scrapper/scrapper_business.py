import requests
from bs4 import BeautifulSoup

def scrape_website(url):
    # Make an HTTP request to the website
    response = requests.get(url)

    # Check if the request was successful (status code 200)
    if response.status_code == 200:
        # Parse the HTML content of the page
        soup = BeautifulSoup(response.text, 'html.parser')

        # Find all <h3> tags on the page
        h3_tags = soup.find_all('h3')

        # Extract the text content of each <h3> tag and save it in a list
        h3_contents = [h3.get_text() for h3 in h3_tags]

        return h3_contents
    else:
        # Print an error message if the request was not successful
        print(f"Error: Unable to fetch the webpage. Status code: {response.status_code}")
        return None

# Example usage:
url = 'https://www.brandveda.in/blog/top-it-software-companies-in-vizag'
result = scrape_website(url)

if result:
    print("List of <h3> contents:")
    print(result)
    for content in result:
        print(content)

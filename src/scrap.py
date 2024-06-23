import requests
from bs4 import BeautifulSoup  
import json

# Website base URL and starting parameters 
base_url = "https://www.azhar.eg/fatwacenter/fatwa/ebadat/ArtMID/7985/ArticleID/"
start_id = 80000
desired_number_of_pages = 2000  # Adjust this as needed

# Function to extract data from a single page 
def extract_data_from_page(soup):
    

    title_element = soup.find("h1", class_="title")  
    answer_element = soup.find("div", class_="main_content")  
    
    articleTitle_element = soup.find("div", class_="articleTitle") # to get date from it
    
    
    # get similar links and titles
    similar_element = soup.find("div", id="dnn_P1_25_2", class_="col-md-4 spacingTop")

    if articleTitle_element and answer_element:
        date_element = articleTitle_element.find("div", class_="date")  
        title = title_element.text.strip() if title_element else ""
        answer = answer_element.text.strip()
        date = date_element.text.strip() if date_element else ""
        data = {"question": title, "answer": answer, "date": date}
        
        #get similar fatawa links
        if similar_element:
            sidebar_title_elements = similar_element.find_all("div", class_="sidebar__titleText")
            sidebar_link_elements = similar_element.find_all("a", class_="sidebar__link")

            # Create lists to store extracted titles and links
            sidebar_titles = []
            sidebar_links = []
            #print("sidebar_title_elements",sidebar_title_elements)
            #print("sidebar_link_elements",sidebar_link_elements)
            
            # Extract data from each element (assuming they have a one-to-one correspondence)
            for i in range(min(len(sidebar_title_elements), len(sidebar_link_elements))):
                sidebar_title = sidebar_title_elements[i].text.strip() if sidebar_title_elements[i] else ""
                sidebar_link = sidebar_link_elements[i].get("href") if sidebar_link_elements[i] else ""
                sidebar_titles.append(sidebar_title)
                sidebar_links.append(sidebar_link)
                print(sidebar_links)
                #print("sidebar_titles",sidebar_titles)

            # Add lists to the data dictionary
            data["sidebar_titles"] = sidebar_titles
            data["sidebar_links"] = sidebar_links

        return data
    else:
        print("Warning: Unable to extract data from this page.")
        return None

# Loop through pages, incrementing the ID
scraped_data = []  # List to store extracted data
for i in range(start_id, start_id + desired_number_of_pages):
    url = f"{base_url}{i}"

    # Make request and handle potential errors
    try:
        response = requests.get(url)
        response.raise_for_status()  # Raise an exception for non-200 status codes

        # Parse HTML content
        soup = BeautifulSoup(response.content, "html.parser")

        # Extract data from the page
        data = extract_data_from_page(soup)

        if data:
            scraped_data.append(data)
            print(f"Scraped data from page {i}")
    except requests.exceptions.RequestException as e:
        print(f"Error retrieving page {i}: {e}")

# Save scraped data to JSON file (replace with desired filename)
with open("fatwas.json", "w", encoding="utf-8") as outfile:
    json.dump(scraped_data, outfile, ensure_ascii=False)  # Preserve Arabic characters

print("Scraping completed. Data saved to fatwas.json")


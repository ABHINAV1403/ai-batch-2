import json
import os
import time
import csv
from datetime import datetime
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager
import google.generativeai as genai
from google.api_core import exceptions
from dotenv import load_dotenv
from bs4 import BeautifulSoup

load_dotenv()

print("GEMINI_API_KEY =", os.getenv("GEMINI_API_KEY"))
# Initialize API keys
current_key_index = 0
API_KEYS = [os.getenv("GEMINI_API_KEY")]
API_KEYS = [key for key in API_KEYS if key is not None]

current_dir = os.getcwd()

INPUT_CSV = r"C:\Users\lenovo\OneDrive\Desktop\infosys\ai-batch-2\Amazon\products.csv"
OUTPUT_CSV = r"C:\Users\lenovo\OneDrive\Desktop\infosys\ai-batch-2\Amazon\reviews.csv"

if not API_KEYS:
    raise ValueError("No API keys found. Please check your .env file.")

def get_current_api_key():
    global current_key_index
    return API_KEYS[current_key_index]

def switch_to_next_api_key():
    global current_key_index
    current_key_index += 1
    if current_key_index >= len(API_KEYS):
        raise ValueError("All API keys have been exhausted.")
    genai.configure(api_key=get_current_api_key())

def configure_gemini():
    while True:
        try:
            genai.configure(api_key=get_current_api_key())
            print(f"Connected to Gemini using API key")
            break
        except exceptions.ResourceExhausted:
            print(f"Quota exhausted for API key. Switching to next key.")
            switch_to_next_api_key()
        except Exception as e:
            print(f"Couldn't connect to Gemini: {e}")
            break

def setup_selenium_headless():
    chrome_options = Options()
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--headless")
    chrome_options.add_argument("--disable-dev-shm-usage")
    chrome_options.add_argument(
        "user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    )
    service = Service(ChromeDriverManager().install()) # Automatically fetch and use the correct ChromeDriver
    # service = Service(ChromeDriverManager(version="114.0.5735.90").install())  # Install specific version
    driver = webdriver.Chrome(service=service, options=chrome_options)
    return driver

def amazon_login(driver, review_url):
    try:
        driver.get(review_url)
        driver.get(review_url)
        time.sleep(5)
        try:
            driver.find_element(By.ID, "ap_email")  # Check if login page appears
            print("Type-1 Login page detected. Logging in...")
        except:
            print("Type-1 login page not detected.")
        try:
            driver.find_element(By.ID, "ap_login_form")  # Check if login page appears
            print("Type-2 Login page detected. Logging in...")
        except:
            print("Type-2 login page not detected.")

        USERNAME = os.getenv(f'AMAZON_USERNAME')  # Replace with your Amazon username
        PASSWORD = os.getenv(f'AMAZON_PASSWORD')  # Replace with your Amazon username

        try:
            WebDriverWait(driver, 5).until(EC.presence_of_element_located((By.ID, "ap_email")))
            driver.find_element(By.ID, "ap_email").send_keys(USERNAME, Keys.RETURN)
        except:
            print("not found, Type-1 sign in ")

        try:
            WebDriverWait(driver, 5).until(EC.presence_of_element_located((By.ID, "ap_login_form")))
            driver.find_element(By.ID, "ap_login_form").send_keys(USERNAME, Keys.RETURN)
        except:
            print("Not found, Type-2 sign in")

        try:
            WebDriverWait(driver, 5).until(EC.presence_of_element_located((By.ID, "ap_email_login")))
            driver.find_element(By.ID, "ap_email_login").send_keys(USERNAME, Keys.RETURN)
        except:
            print("Not found, Type-2 sign in tag")

        try:
            WebDriverWait(driver, 5).until(EC.presence_of_element_located((By.ID, "ap_password")))
            driver.find_element(By.ID, "ap_password").send_keys(PASSWORD, Keys.RETURN)
        except:
            print("Not found, password tag")

        time.sleep(5)  # Allow time for login to complete

        print("Logged in successfully!")
    except Exception as e:
        print(f"Error during login: {e}")
        driver.quit()

def fetch_review_page_html(driver, review_url):
    try:
        print(f"Loading URL: {review_url}")
        driver.get(review_url)
        time.sleep(5)  # Allow page to load
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        time.sleep(5)  # Wait for content to load
        return driver.page_source
    except Exception as e:
        print(f"Error fetching review page HTML: {e}")
        return None

def get_reviews(driver, review_url, output_file):
    html_content = fetch_review_page_html(driver, review_url)
    if not html_content:
        return None

    try:
        print("HTML content fetched successfully.")
        my_html = BeautifulSoup(html_content, 'html.parser')

        text = f"""
        {str(my_html)},

        Given the HTML content above, please extract all unique review data, ensuring to remove duplicates. For each product, structure the data in the following JSON format, and make sure to exclude HTML tags:

        {{
            "Product_Reviews": [
                {{
                    "Product_Name": "Product Name",
                    "Product_ASIN": "Product ASIN",
                    "Review_ASIN": "Review ASIN",
                    "Review_Title": "Review Title",
                    "Reviewer": "Reviewer Name",
                    "Review_Text": "Review Content",
                    "Review_Date": "Review Date (give in the format YYYY-MM-DD)",
                    "Rating": "Rating Value as Integer",
                    "Helpful_Votes": "Helpful Votes Count as Integer value or 0 if not available",
                    "Product_URL": "Product URL" (add https://amazon.in as prefix),
                    "Review_URL": "Review URL" (add https://amazon.in as prefix)
                }}
            ]
        }}

        Please ensure the following:
        1. Only include the reviews under the key "Product_Reviews". Do not include any other review data, such as data under a "reviews" key or other similar keys.
        2. Remove any HTML tags.
        3. Extract unique reviews only (no duplicates).
        4. Keep the data fields as specified.
        5. If no value is available for "Helpful_Votes", set it to NULL.
        6. Each review should be in a separate object within the "Product_Reviews" list.
        7. Return the result in a well-formed JSON object where the "Product_Reviews" key contains only the relevant reviews.

        Please format the output exactly as described above.
        """


        configure_gemini()
        model = genai.GenerativeModel(model_name="gemini-2.0-flash-exp")
        chat = model.start_chat(history=[])
        response = chat.send_message(text)

        cleaned_response = response.text.strip("json").strip("").strip()
        cleaned_response = cleaned_response.rstrip('```').rstrip()
        print(cleaned_response)

        reviews_data = json.loads(cleaned_response)  # Parse the response into JSON

        # Save reviews to CSV
        save_reviews_to_csv(reviews_data, output_file)

    except Exception as e:
        print(f"Error parsing reviews: {e}")
        return None
    
def save_reviews_to_csv(reviews_data, output_file):
    """
    Save the parsed reviews data to the output CSV file, ensuring that new data is appended.
    """
    try:
        # Check for keys that may hold the reviews data
        reviews = reviews_data.get("Product_Reviews", [])
        
        # If no reviews found, print a message and exit
        if not reviews:
            print("No reviews found in the data.")
            return
        
        # Process each review and prepare the rows for the CSV
        rows = []
        for review in reviews:
            row = [
                review.get("Product_Name", ""),
                review.get("Product_ASIN", ""),
                review.get("Review_ASIN", ""),
                review.get("Review_Title", ""),
                review.get("Reviewer", ""),
                review.get("Review_Text", ""),
                review.get("Review_Date", ""),
                review.get("Rating", ""),
                review.get("Helpful_Votes", "NULL"),
                review.get("Product_URL", ""),
                review.get("Review_URL", "")
            ]
            rows.append(row)

        # Add timestamp column for each review
        scraping_time = datetime.now().isoformat()  # Current timestamp
        for row in rows:
            row.append(scraping_time)

        # Check if the CSV file already exists
        file_exists = os.path.exists(output_file)

        # Open the CSV file in append mode
        with open(output_file, mode='a', newline='', encoding='utf-8') as file:
            writer = csv.writer(file, quoting=csv.QUOTE_ALL)  # Escape special characters

            # If the file doesn't exist, write the headers first
            if not file_exists:
                headers = [
                    "Product_Name", "Product_ASIN", "Review_ASIN", "Review_Title", "Reviewer",
                    "Review_Text", "Review_Date", "Rating", "Helpful_Votes", "Product_URL", "Review_URL", "Scraping_Time"
                ]
                writer.writerow(headers)

            # Write all rows (append mode)
            writer.writerows(rows)

        print(f"Reviews saved to {output_file} successfully.")

    except Exception as e:
        print(f"Error saving reviews to CSV: {e}")

def read_urls_from_csv(input_file):
    urls = []
    try:
        with open(input_file, mode='r', encoding="utf-8") as file:
            reader = csv.DictReader(file)
            for row in reader:
                # Support both old and new column names
                review_url = (
                    row.get("Reviews_URL") 
                    or row.get("Reviews_Link") 
                    or ""
                ).strip()
                product_url = (
                    row.get("Product_URL") 
                    or row.get("Product_Link") 
                    or ""
                ).strip()
                product_name = row.get("Product_Name", "").strip()
                product_asin = row.get("Product_ASIN", "").strip()

                if review_url and review_url.startswith("http"):
                    urls.append((review_url, product_name, product_asin, product_url))
                else:
                    print(f"⚠ No valid review URL for product: {product_name} ({product_asin})")

        return urls
    except Exception as e:
        print(f"Error reading the input file: {e}")
        return []

def process_reviews(input_csv, output_csv):
    urls = read_urls_from_csv(input_csv)
    if not urls:
        print("No URLs found in the input CSV!")
        return

    driver = setup_selenium_headless()

    try:
        # Log in to Amazon before processing reviews
        first_url = urls[0][0]  # Get the first review URL to access the login page
        amazon_login(driver, first_url)

        for review_url, product_name, product_id, product_url in urls:
            if not review_url.strip():  # Check for empty or whitespace-only URLs
                print(f"Skipping empty URL for product: {product_name} (ID: {product_id})")
                continue
            print(f"Processing URL: {review_url}")
            try:
                get_reviews(driver, review_url, output_csv)
            except Exception as e:
                print(f"Error processing URL {review_url}: {e}")
    finally:
        driver.quit()


# Start processing reviews
process_reviews(INPUT_CSV, OUTPUT_CSV)

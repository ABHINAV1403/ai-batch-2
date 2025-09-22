import os
import time
import json
import requests
import pandas as pd
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from datetime import datetime, timedelta

load_dotenv()
# Retrieve headers and cookies from the environment variables
HEADERS = json.loads(os.getenv("HEADERS"))
COOKIES = json.loads(os.getenv("COOKIES"))
BASE_URL = "https://www.amazon.in/s?k=headphones"

def fetch_content(url):
    """
    Fetch HTML content of the given URL.
    """
    try:
        response = requests.get(url, headers=HEADERS, cookies=COOKIES, timeout=10)
        response.raise_for_status()
        print(f"Successfully fetched page: {url}")
        soup = BeautifulSoup(response.content, "html.parser")
        return soup
    except requests.exceptions.RequestException as e:
        print(f"Error fetching page content: {e}")
        return None

def get_title(item):
    title = item.find("h2", class_="a-size-medium a-spacing-none a-color-base a-text-normal")
    return title.text.strip() if title else None

def get_brand(title_text):
    if title_text.startswith("soundcore"):
        return "Anker"
    return title_text.split()[0] if title_text else None

def get_price(item):
    discount_price = item.find("span", class_="a-price")
    return (
        discount_price.find("span", class_="a-offscreen").text.strip()
        if discount_price and discount_price.find("span", class_="a-offscreen")
        else None
    )

def get_mrp(item):
    base_price = item.find("div", class_="a-section aok-inline-block")
    return (
        base_price.find("span", class_="a-offscreen").text.strip()
        if base_price and base_price.find("span", class_="a-offscreen")
        else None
    )

def get_discount_percentage(item):
    discount = item.find("span", string=lambda text: text and "%" in text)
    return discount.text.strip().strip("()") if discount else None

def get_rating(item):
    rating = item.find("span", class_="a-icon-alt")
    return rating.text.strip() if rating else None

def get_reviews(item):
    reviews = item.find("span", class_="a-size-base s-underline-text")
    return reviews.text.strip() if reviews else None

def get_product_id(item):
    return item.get("data-asin", None)

def get_product_link(item):
    """
    Extract product link from the item.
    """
    link = item.find("a", class_="a-link-normal s-no-outline")
    return "https://www.amazon.in" + link["href"] if link and "href" in link.attrs else None

def get_reviews_link(product_link):
    """
    Extracts the reviews link from the given product link, adds a prefix
    for Amazon India, and returns the complete URL.

    Args:
        product_link: URL of the product page.

    Returns:
        The complete URL of the reviews page, or None if not found.
    """

    try:
        time.sleep(0.5)  # Introduce a delay to avoid overloading the server (adjust as needed)

        response = requests.get(product_link, headers=HEADERS, cookies=COOKIES)
        response.raise_for_status()

        soup = BeautifulSoup(response.content, 'html.parser')
        # Target the 'a' tag directly using class selector
        reviews_link_element = soup.select_one('a.a-link-emphasis[data-hook="see-all-reviews-link-foot"]')

        if reviews_link_element:
            href = reviews_link_element.get('href')
            if href:
                # Add prefix for Amazon India
                url = f"https://amazon.in{href}"
                return url
            else:
                return None
        else:
            return None

    except (requests.exceptions.RequestException, requests.exceptions.HTTPError) as e:
        print(f"Error fetching product page {product_link}: {e}")
        return None
    
def parse_product(item):
    """
    Extract details of a single product.
    """
    try:
        title_text = get_title(item)
        product_link = get_product_link(item)
        product_asin = get_product_id(item)

        return {
            "Product_Name": title_text,
            "Product_ASIN": product_asin,
            "Brand": get_brand(title_text),
            "Price": get_price(item),
            "MRP": get_mrp(item),
            "Discount": get_discount_percentage(item),
            "Stock_Status": "In Stock",   # or get_stock_status(item) if you wrote one
            "Rating": get_rating(item),
            "Reviews": get_reviews(item),
            "Seller": "Amazon.com, Inc",  # or get_seller(item)
            "Product_Link": product_link,
            "Reviews_Link": f"https://www.amazon.in/product-reviews/{product_asin}" if product_asin else None,
            "Scraped_At": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
    except Exception as e:
        print(f"Error parsing product: {e}")
        return None

def scrape_page(url):
    """
    Scrape all product details from a single page.
    """
    soup = fetch_content(url)
    if not soup:
        return [], None

    items = soup.find_all("div", {"data-component-type": "s-search-result"})
    products = []

    for item in items:
        product_data = parse_product(item)
        if product_data:
            products.append(product_data)

    # Find next page URL
    next_button = soup.find("a", class_="s-pagination-next")
    next_page_url = "https://www.amazon.in" + next_button["href"] if next_button and "href" in next_button.attrs else None

    return products, next_page_url

def scrape_within_time(base_url, max_time_minutes=5):
    """
    Scrape as many pages as possible within the given time limit.
    """
    all_products = []
    current_page = 1
    next_page_url = base_url
    end_time = datetime.now() + timedelta(minutes=max_time_minutes)

    try:
        print(f"Starting the scraping process for {max_time_minutes} minutes...")
        while next_page_url and datetime.now() < end_time:
            print(f"Scraping page {current_page}...")
            products, next_page_url = scrape_page(next_page_url)
            
            # If no products are found, end scraping
            if not products:
                print("No more products found. Stopping scraping.")
                break

            all_products.extend(products)
            print(f"Scraped {len(products)} products from page {current_page}.")
            current_page += 1
            time.sleep(2)  # Add delay to prevent being blocked by the server
            
            # Check the time condition after each iteration to stop when the time is up
            if datetime.now() >= end_time:
                print("Time limit reached. Stopping scraping.")
                break

    except Exception as e:
        print(f"An error occurred during scraping: {e}")
    
    print("Scraping finished.")
    return all_products

def save_to_csv(data, filename="products.csv"):
    """
    Save product data to a CSV file.
    """
    try:
        # Define the directory where you want to save the file (assuming the 'amazon/' folder already exists)
        directory = "Amazon"
        
        # Define the full file path to save the CSV file inside the 'amazon/' directory
        full_file_path = os.path.join(directory, filename)
        
        # Save the DataFrame to CSV
        df = pd.DataFrame(data)
        df.to_csv(full_file_path, index=False)
        
        print(f"Data successfully saved to {full_file_path}.")
    except Exception as e:
        print(f"Error saving data to CSV: {e}")


if __name__ == "__main__":
    max_scraping_time = 1  # Max time in minutes
    print("Starting the scraping process...")
    products = scrape_within_time(BASE_URL, max_time_minutes=max_scraping_time)

    # NOTE: For Amazon, we can scrape until Page-20, then there will be no more pages to scrape (EOP).
    # So the scraping will be finished within approx. 3 minutes.

    if products:
        save_to_csv(products)
        print(f"Scraped {len(products)} products.")
    else:
        print("No products found.")
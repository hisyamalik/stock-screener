import requests
from bs4 import BeautifulSoup

# Function to scrape stock symbols from IDX or other websites
def scrape_indonesian_stock_symbols(url):
    indonesian_stocks = []
    
    # Send a GET request to the URL
    response = requests.get(url)
    
    # Check if the request was successful
    if response.status_code == 200:
        # Parse the page content with BeautifulSoup
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Find the table or div where the stock tickers are listed
        table = soup.find('table', {'id': 'stock-table'})
        
        # Loop through the table rows to find the stock tickers
        for row in table.find_all('tr')[1:]:  # Skipping the header row
            columns = row.find_all('td')
            ticker = columns[0].text.strip() + ".JK"  # Append ".JK" for Yahoo Finance format
            indonesian_stocks.append(ticker)
        
        print(f"Scraped {len(indonesian_stocks)} stock symbols.")
    
    else:
        print(f"Failed to retrieve data from {url}. Status code: {response.status_code}")
    
    return indonesian_stocks

# Example of how to use the scraper
def main():
    # The URL of the webpage that lists Indonesian stock symbols
    url = 'https://finance.yahoo.com/quote/'  # Replace with actual URL
    
    # Scrape the stock symbols
    indonesian_stocks = scrape_indonesian_stock_symbols(url)
    
    # Output the scraped stock symbols
    print("Indonesian Stock Symbols:", indonesian_stocks)

if __name__ == "__main__":
    main()
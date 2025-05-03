# %%
#Do this in your terminal
import requests
import csv
from bs4 import BeautifulSoup

BASE_URL = "https://www.federalreserve.gov"

# Manually define the year range and corresponding URLs for SR letters from 1990 to 2024
MANUAL_YEAR_LINKS = {
    '1990': 'https://www.federalreserve.gov/supervisionreg/srletters/1990.htm',
    '1991': 'https://www.federalreserve.gov/supervisionreg/srletters/1991.htm',
    '1992': 'https://www.federalreserve.gov/supervisionreg/srletters/1992.htm',
    '1993': 'https://www.federalreserve.gov/supervisionreg/srletters/1993.htm',
    '1994': 'https://www.federalreserve.gov/supervisionreg/srletters/1994.htm',
    '1995': 'https://www.federalreserve.gov/supervisionreg/srletters/1995.htm',
    '1996': 'https://www.federalreserve.gov/supervisionreg/srletters/1996.htm',
    '1997': 'https://www.federalreserve.gov/supervisionreg/srletters/1997.htm',
    '1998': 'https://www.federalreserve.gov/supervisionreg/srletters/1998.htm',
    '1999': 'https://www.federalreserve.gov/supervisionreg/srletters/1999.htm',
    '2000': 'https://www.federalreserve.gov/supervisionreg/srletters/2000.htm',
    '2001': 'https://www.federalreserve.gov/supervisionreg/srletters/2001.htm',
    '2002': 'https://www.federalreserve.gov/supervisionreg/srletters/2002.htm',
    '2003': 'https://www.federalreserve.gov/supervisionreg/srletters/2003.htm',
    '2004': 'https://www.federalreserve.gov/supervisionreg/srletters/2004.htm',
    '2005': 'https://www.federalreserve.gov/supervisionreg/srletters/2005.htm',
    '2006': 'https://www.federalreserve.gov/supervisionreg/srletters/2006.htm',
    '2007': 'https://www.federalreserve.gov/supervisionreg/srletters/2007.htm',
    '2008': 'https://www.federalreserve.gov/supervisionreg/srletters/2008.htm',
    '2009': 'https://www.federalreserve.gov/supervisionreg/srletters/2009.htm',
    '2010': 'https://www.federalreserve.gov/supervisionreg/srletters/2010.htm',
    '2011': 'https://www.federalreserve.gov/supervisionreg/srletters/2011.htm',
    '2012': 'https://www.federalreserve.gov/supervisionreg/srletters/2012.htm',
    '2013': 'https://www.federalreserve.gov/supervisionreg/srletters/2013.htm',
    '2014': 'https://www.federalreserve.gov/supervisionreg/srletters/2014.htm',
    '2015': 'https://www.federalreserve.gov/supervisionreg/srletters/2015.htm',
    '2016': 'https://www.federalreserve.gov/supervisionreg/srletters/2016.htm',
    '2017': 'https://www.federalreserve.gov/supervisionreg/srletters/2017.htm',
    '2018': 'https://www.federalreserve.gov/supervisionreg/srletters/2018.htm',
    '2019': 'https://www.federalreserve.gov/supervisionreg/srletters/2019.htm',
    '2020': 'https://www.federalreserve.gov/supervisionreg/srletters/2020.htm',
    '2021': 'https://www.federalreserve.gov/supervisionreg/srletters/2021.htm',
    '2022': 'https://www.federalreserve.gov/supervisionreg/srletters/2022.htm',
    '2023': 'https://www.federalreserve.gov/supervisionreg/srletters/2023.htm',
    '2024': 'https://www.federalreserve.gov/supervisionreg/srletters/2024.htm',
}

def scrape_sr_letter_links(year_url):
    """Scrapes SR letter links from a given year's SR letters page."""
    response = requests.get(year_url)
    if response.status_code != 200:
        print(f"Failed to fetch SR letters for {year_url}")
        return []
    
    soup = BeautifulSoup(response.text, 'html.parser')
    
    sr_links = []
    for link in soup.find_all('a', href=True):
        # Check that the link is an SR letter (contains "SR" and ends with ".htm")
        if "SR" in link.text and link['href'].endswith(".htm"):
            href = link['href']
            full_link = href if href.startswith("http") else f"{BASE_URL}{href}"
            sr_links.append((link.text.strip(), full_link))
            print(f"Found SR Link: {link.text.strip()} -> {full_link}")  # Debugging output
    
    return sr_links

def scrape_all_years():
    """Scrapes SR letters for all manually defined years."""
    all_sr_letters = []
    for year, link in sorted(MANUAL_YEAR_LINKS.items(), reverse=True):  # Sort years from newest to oldest
        print(f"Scraping {year}...")
        sr_letters = scrape_sr_letter_links(link)
        for sr, link in sr_letters:
            all_sr_letters.append([year, sr, link])
    
    return all_sr_letters

def save_to_csv(sr_data, file_path):
    """Saves scraped SR letter data to a CSV file."""
    with open(file_path, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(["Year", "SR Letter Title", "Link"])  # Write header
        writer.writerows(sr_data)
    print(f"Data saved to {file_path}")

# Run the scraper
sr_data = scrape_all_years()

# Define the file path
csv_file_path = 'quarto-manuscript/data/sr_letters.csv'

# Save the data to CSV
save_to_csv(sr_data, csv_file_path)
"""
Assets for scraping and processing data for reports.
"""
from typing import Optional
from dagster import asset, Config, DailyPartitionsDefinition
import pandas as pd
from playwright.sync_api import sync_playwright
from bs4 import BeautifulSoup
import time
from datetime import datetime

class PerpetualPulseVolumeConfig(Config):
    url: str = "https://www.perpetualpulse.xyz/"
    headless: bool = True

partitions_def = DailyPartitionsDefinition(start_date="2024-07-01")

@asset(
    name="perpetual_pulse_daily_scrape",
    group_name="reports",
    description="Scraped daily volume from perpetualpulse.xyz for a single day.",
    partitions_def=partitions_def,
)
def perpetual_pulse_daily_scrape(context, config: PerpetualPulseVolumeConfig) -> pd.DataFrame:
    """
    Scrapes the perpetual protocol volume data from perpetualpulse.xyz for a single
    day, corresponding to the partition date.
    """
    partition_date_str = context.partition_key
    
    url = config.url
    
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=config.headless)
        page = browser.new_page()
        page.goto(url, wait_until="domcontentloaded", timeout=90000)
        
        first_row_selector = 'div[data-row-index="0"]'
        page.wait_for_selector(first_row_selector, timeout=60000)
        
        time.sleep(5) 
        
        content = page.content()
        browser.close()

    soup = BeautifulSoup(content, 'html.parser')
    
    rows = soup.find_all('div', class_='notion-table-view-row')
    if not rows:
        raise ValueError("Could not find any 'notion-table-view-row' divs. The site structure may have changed.")

    data = []
    for row in rows:
        exchange_cell = row.find('div', {'data-col-index': '0'})
        volume_cell = row.find('div', {'data-col-index': '1'})

        if exchange_cell and volume_cell:
            exchange = exchange_cell.get_text(strip=True)
            volume_str = volume_cell.get_text(strip=True)
            
            try:
                volume = float(volume_str.replace('$', '').replace(',', ''))
            except (ValueError, AttributeError):
                volume = None

            if exchange and volume is not None:
                data.append({'exchange': exchange, 'volume_24h': volume})
                
    if not data:
        raise ValueError("No data could be extracted from the table rows. The site structure may have changed.")
        
    df = pd.DataFrame(data)

    df['capture_date'] = datetime.strptime(partition_date_str, "%Y-%m-%d")
    df['capture_date'] = pd.to_datetime(df['capture_date'])

    return df

@asset(
    name="perpetual_pulse_volume",
    group_name="reports",
    description="A historical log of all daily perpetual pulse volumes, created by combining all daily scrapes."
)
def perpetual_pulse_volume(perpetual_pulse_daily_scrape: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Combines all daily scrapes into a single historical DataFrame. 
    The input `perpetual_pulse_daily_scrape` is a dictionary where keys are partition
    names (dates) and values are the DataFrames from each daily scrape.
    """
    if not perpetual_pulse_daily_scrape:
        return pd.DataFrame(columns=['exchange', 'volume_24h', 'capture_date'])
    
    all_dfs = list(perpetual_pulse_daily_scrape.values())
    combined_df = pd.concat(all_dfs)
    combined_df = combined_df.sort_values(by="capture_date").reset_index(drop=True)
    combined_df = combined_df.drop_duplicates(subset=['exchange', 'capture_date'], keep='last')
    return combined_df 
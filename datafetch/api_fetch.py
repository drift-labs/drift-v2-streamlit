"""
Fetch trades from the API

The API is documented at https://data.api.drift.trade/playground/
"""

import time
from datetime import timedelta

import pandas as pd
import requests
from streamlit import cache_data

URL_PREFIX = "https://data.api.drift.trade"


@cache_data(ttl=60 * 60 * 24)
def get_trades_for_range_pandas(market_symbol, start_date, end_date, page=1):
    print(f"Fetching trades for {market_symbol} from {start_date} to {end_date}")
    df = pd.DataFrame()
    all_trades = []
    current_date = start_date
    while current_date <= end_date:
        year = current_date.year
        month = current_date.month
        day = current_date.day
        url = f"{URL_PREFIX}/market/{market_symbol}/trades/{year}/{month:02}/{day:02}"
        print(f"==> {url}")
        try:
            response = requests.get(url, params={"page": page})
            response.raise_for_status()
            json = response.json()
            meta = json["meta"]
            df = pd.DataFrame(json["records"])
            while meta["nextPage"] is not None:
                pg = meta["nextPage"]
                response = requests.get(url, params={"page": pg})
                print("Page", str(pg))
                response.raise_for_status()
                json = response.json()
                df = pd.concat([df, pd.DataFrame(json["records"])], ignore_index=True)
                meta = json["meta"]
            time.sleep(0.1)
            all_trades.append(df)
        except requests.exceptions.RequestException as e:
            print(f"Error fetching data for {current_date}: {e}")
        except pd.errors.EmptyDataError:
            print(f"No data available for {current_date}")

        current_date += timedelta(days=1)

    if all_trades:
        df = pd.concat(all_trades, ignore_index=True)
    else:
        df = pd.DataFrame()

    df["ts"] = pd.to_datetime(df["ts"], unit="s")
    return df


@cache_data(ttl=60 * 60 * 24)
def get_trades_for_day(market_symbol, year, month, day, page=1):
    """
    Fetch trades for a specific market and date from the Drift API.

    Args:
        market_symbol (str): The market symbol (e.g., 'SOL-PERP')
        year (int): Year of the data
        month (int): Month of the data
        day (int): Day of the data
        page (int, optional): Starting page number. Defaults to 1.

    Returns:
        pd.DataFrame: DataFrame containing trade data
    """
    url = f"{URL_PREFIX}/market/{market_symbol}/trades/{year}/{month:02d}/{day:02d}"
    print(f"Fetching trades from: {url}")

    try:
        all_data = []
        current_page = page

        while True:
            print(f"Fetching page {current_page}")
            response = requests.get(url, params={"page": current_page})
            response.raise_for_status()
            json_data = response.json()
            meta = json_data["meta"]
            all_data.extend(json_data["records"])

            if meta["nextPage"] is None or current_page >= meta["totalPages"]:
                break

            current_page = meta["nextPage"]
            time.sleep(0.1)  # Be nice to the API

        df = pd.DataFrame(all_data)
        if not df.empty:
            df["ts"] = pd.to_datetime(df["ts"], unit="s")

        print(f"Retrieved {len(all_data)} records out of {meta.get('totalRecords', 'unknown')} total")
        return df

    except requests.exceptions.RequestException as e:
        print(f"Error fetching data: {e}")
        return pd.DataFrame()
    except Exception as e:
        print(f"Unexpected error: {e}")
        return pd.DataFrame()
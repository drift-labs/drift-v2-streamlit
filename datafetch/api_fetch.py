"""
Fetch trades from the API

The API is documented at https://data.api.drift.trade/playground/
"""

from datetime import timedelta
import time

import pandas as pd
import requests


URL_PREFIX = "https://data.api.drift.trade"


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

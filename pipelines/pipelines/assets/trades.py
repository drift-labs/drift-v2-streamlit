"""
Trades assets - Process and analyze trade data
"""

from dagster import asset, AutoMaterializePolicy
import pandas as pd
from ..partitions import daily_partitions

@asset(
    group_name="trades",
    description="Clean and basic enrichment of trade data",
    partitions_def=daily_partitions,
)
def clean_trades(raw_trades: pd.DataFrame) -> pd.DataFrame:
    """
    Basic cleaning of trade data:
    - Remove duplicates
    - Convert to proper numeric types
    """
    df = raw_trades.copy()
    
    if 'fillrecordid' in df.columns:
        df = df.drop_duplicates(subset=['fillrecordid'])
    
    numeric_cols = ['baseassetamountfilled', 'quoteassetamountfilled']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
        
    return df

@asset(
    group_name="trades", 
    description="Daily market index trade volume summary",
    partitions_def=daily_partitions,
    io_manager_key="csv_io_manager",
)
def daily_market_summary(clean_trades: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregates trade data by trade date and market index.
    """
    df = clean_trades.copy()
    df['trade_date'] = pd.to_datetime(df['ts'], unit='s').dt.date

    grouped = df.groupby(['trade_date', 'marketindex', 'markettype']).agg({
        'baseassetamountfilled': 'sum',
        'quoteassetamountfilled': 'sum',
        'fillrecordid': 'count' 
    }).rename(columns={'fillrecordid': 'trade_count'}).reset_index()

    return grouped
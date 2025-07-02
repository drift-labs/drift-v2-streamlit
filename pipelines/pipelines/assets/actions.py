"""
Trades assets - Process and analyze action data
"""

from dagster import asset
import pandas as pd


@asset(
    group_name="actions",
    description="Clean and basic enrichment of action data",
)
def clean_actions(raw_actions: pd.DataFrame) -> pd.DataFrame:
    return raw_actions

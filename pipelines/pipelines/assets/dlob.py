import pandas as pd


def get_oracle_prices_for_market(
    context, athena, athena_config, market_index: int, market_type: str
) -> pd.DataFrame:
    partition_date = context.partition_key
    year, month, day = partition_date.split("-")

    query = f"""
            SELECT oracle, ts FROM "{athena_config.database_name}"."{athena_config.dlob_snapshot_table_name}"
            WHERE marketindex = '{market_index}' AND markettype = '{market_type}' AND year = '{year}' AND month = '{month}' AND day = '{day}'
        """

    return athena.get_client().execute_query(query, fetch_results=True)

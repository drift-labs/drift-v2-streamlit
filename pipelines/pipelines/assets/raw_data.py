"""
Raw data assets - Extract data directly from Athena tables
"""

from dagster import asset
from ..resources import AthenaConfig, WrappedAthenaClientResource
from ..partitions import daily_partitions

@asset(
    group_name="raw_data",
    partitions_def=daily_partitions
)
def raw_trades(context, athena: WrappedAthenaClientResource, athena_config: AthenaConfig):
    partition_date = context.partition_key
    year, month, day = partition_date.split("-")

    query = f"""
        SELECT DISTINCT * FROM "{athena_config.database_name}"."{athena_config.trades_table_name}"
        WHERE year = '{year}' AND month = '{month}' AND day = '{day}'
    """
    return athena.get_client().execute_query(query, fetch_results=True)

"""
Raw data assets - Extract data directly from Athena tables
"""

from dagster import asset
from ..resources import AthenaConfig, WrappedAthenaClientResource


@asset(group_name="raw_data")
def raw_trades(
    context, athena: WrappedAthenaClientResource, athena_config: AthenaConfig
):
    partition_date = context.partition_key
    year, month, day = partition_date.split("-")

    query = f"""
        SELECT DISTINCT * FROM "{athena_config.database_name}"."{athena_config.trades_table_name}"
        WHERE year = '{year}' AND month = '{month}' AND day = '{day}'
    """
    return athena.get_client().execute_query(query, fetch_results=True)


@asset(group_name="raw_data")
def raw_trigger_orders(
    context, athena: WrappedAthenaClientResource, athena_config: AthenaConfig
):
    partition_date = context.partition_key
    year, month, day = partition_date.split("-")

    query = f"""
        SELECT DISTINCT * FROM "{athena_config.database_name}"."{athena_config.orders_table_name}"
        WHERE "order".orderType IN ('triggerMarket', 'triggerLimit') AND year = '{year}' AND month = '{month}' AND day = '{day}'
    """
    return athena.get_client().execute_query(query, fetch_results=True)


@asset(group_name="raw_data")
def raw_actions(
    context, athena: WrappedAthenaClientResource, athena_config: AthenaConfig
):
    partition_date = context.partition_key
    year, month, day = partition_date.split("-")

    # Ignore large makers to reduce the amount of data we retrieve
    query = f"""
        SELECT DISTINCT * FROM "{athena_config.database_name}"."{athena_config.actions_table_name}"
        WHERE COALESCE(makerorderid, 0) < 100000 AND year = '{year}' AND month = '{month}' AND day = '{day}'
    """
    return athena.get_client().execute_query(query, fetch_results=True)

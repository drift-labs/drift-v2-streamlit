"""
Order assets - Process and analyze order data
"""

from dagster import asset
import pandas as pd
import re
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from ..resources import AthenaConfig, WrappedAthenaClientResource
from ..partitions import daily_partitions
from ..assets.dlob import get_oracle_prices_for_market


def parse_order_string(order_str):
    """Convert stringified struct into a dictionary."""
    if not isinstance(order_str, str):
        return {}
    try:
        clean = order_str.strip("{}")
        pairs = re.split(r",\s*", clean)
        result = {}
        for pair in pairs:
            if "=" in pair:
                key, value = pair.split("=", 1)
                result[key.strip()] = value.strip()
        return result
    except Exception as e:
        print(f"Error parsing: {order_str} → {e}")
        return {}


def clean_orders_logic(df: pd.DataFrame) -> pd.DataFrame:
    if "order" in df.columns:
        parsed_orders = df["order"].apply(parse_order_string)
        order_normalized = pd.json_normalize(parsed_orders)
        order_normalized = order_normalized.add_prefix("order_")
        df_cleaned = pd.concat([df.drop(columns=["order"]), order_normalized], axis=1)
        return df_cleaned

    return df


def check_missed_orders_for_market(
    market_orders: pd.DataFrame, context, athena, athena_config
) -> list:
    """
    For a given market, fetch oracle prices and identify orders that should have been triggered.
    """
    missed_for_market = []
    if market_orders.empty:
        return missed_for_market

    market_index = market_orders["order_marketindex"].iloc[0]
    market_type = market_orders["order_markettype"].iloc[0]

    # Fetch oracle prices for the market
    dlob = get_oracle_prices_for_market(
        context, athena, athena_config, market_index, market_type
    )
    dlob["oracle_price"] = pd.to_numeric(dlob["oracle"], errors="coerce")
    dlob["timestamp"] = pd.to_numeric(dlob["ts"], errors="coerce")

    # Iterate over each unique order for the market
    for _, order_row in market_orders.iterrows():
        order_ts = pd.to_numeric(order_row["ts_x"], errors="coerce")
        trigger_price = pd.to_numeric(order_row["order_triggerprice"], errors="coerce")
        trigger_condition = order_row["order_triggercondition"]

        if pd.isna(trigger_price) or pd.isna(order_ts):
            continue

        # For canceled orders, we only want to look between the open and cancelled time
        if order_row["order_status"] == "canceled":
            cancellation_ts = pd.to_numeric(
                order_row["cancellation_ts"], errors="coerce"
            )
            if pd.isna(cancellation_ts):
                continue
            relevant_dlob = dlob[
                (dlob["timestamp"] > order_ts) & (dlob["timestamp"] < cancellation_ts)
            ]
        else:
            # For pending orders, consider all prices after the order was placed
            relevant_dlob = dlob[dlob["timestamp"] > order_ts]

        if relevant_dlob.empty:
            continue

        # Check if the trigger condition was met
        if trigger_condition == "above":
            triggering_dlob = relevant_dlob[
                relevant_dlob["oracle_price"] >= trigger_price
            ]
            if not triggering_dlob.empty:
                triggering_row = triggering_dlob.iloc[0]
                order_dict = order_row.to_dict()
                order_dict["triggering_oracle_price"] = triggering_row["oracle_price"]
                order_dict["triggering_timestamp"] = triggering_row["timestamp"]
                missed_for_market.append(order_dict)
        elif trigger_condition == "below":
            triggering_dlob = relevant_dlob[
                relevant_dlob["oracle_price"] <= trigger_price
            ]
            if not triggering_dlob.empty:
                triggering_row = triggering_dlob.iloc[0]
                order_dict = order_row.to_dict()
                order_dict["triggering_oracle_price"] = triggering_row["oracle_price"]
                order_dict["triggering_timestamp"] = triggering_row["timestamp"]
                missed_for_market.append(order_dict)

    return missed_for_market


@asset(
    group_name="orders",
    description="Clean and basic enrichment of trigger order data",
    partitions_def=daily_partitions,
)
def clean_trigger_orders(
    raw_trigger_orders: pd.DataFrame,
) -> pd.DataFrame:
    """
    Basic cleaning of order data of raw_trigger_orders
    """
    return clean_orders_logic(raw_trigger_orders)


@asset(
    group_name="orders",
    description="Combine the trigger orders with their actions and categorize by status",
    partitions_def=daily_partitions,
)
def trigger_orders_with_actions(
    clean_trigger_orders: pd.DataFrame,
    clean_actions: pd.DataFrame,
) -> pd.DataFrame:
    """
    Categorize all trigger orders as pending, canceled, or triggered
    """
    combined = clean_trigger_orders.merge(
        clean_actions,
        left_on=["user", "order_orderid"],
        right_on=["taker", "takerorderid"],
        how="left",
    )

    group_keys = [combined["user"], combined["order_orderid"]]
    is_triggered_group = (
        (combined["action"] == "trigger").groupby(group_keys).transform("any")
    )
    is_canceled_group = (
        (combined["action"] == "cancel").groupby(group_keys).transform("any")
    )

    conditions = [
        is_triggered_group,
        is_canceled_group,
    ]
    choices = ["triggered", "canceled"]
    combined["order_status"] = np.select(conditions, choices, default="pending")

    return combined


@asset(
    group_name="orders",
    description="Get pending trigger orders",
    partitions_def=daily_partitions,
)
def pending_trigger_orders(trigger_orders_with_actions: pd.DataFrame) -> pd.DataFrame:
    return trigger_orders_with_actions[
        trigger_orders_with_actions["order_status"] == "pending"
    ]


@asset(
    group_name="orders",
    description="Get canceled trigger orders",
    partitions_def=daily_partitions,
)
def canceled_trigger_orders(trigger_orders_with_actions: pd.DataFrame) -> pd.DataFrame:
    return trigger_orders_with_actions[
        trigger_orders_with_actions["order_status"] == "canceled"
    ]


@asset(
    group_name="orders",
    description="Identifies pending or canceled orders that would have been triggered by oracle prices.",
    partitions_def=daily_partitions,
)
def missed_trigger_orders(
    context,
    athena: WrappedAthenaClientResource,
    athena_config: AthenaConfig,
    pending_trigger_orders: pd.DataFrame,
    canceled_trigger_orders: pd.DataFrame,
) -> pd.DataFrame:
    """
    Determines which pending or canceled orders would have been triggered based on oracle prices.
    This helps identify orders that should have fired but didn't.
    """
    untriggered_orders = pd.concat(
        [pending_trigger_orders, canceled_trigger_orders], ignore_index=True
    )

    if untriggered_orders.empty:
        return pd.DataFrame(columns=untriggered_orders.columns)

    # Consolidate order data to have one row per unique order, with cancellation time if applicable
    # Get the base order data from the 'place' action
    placements = untriggered_orders[untriggered_orders["action"] == "place"].copy()

    # Get the cancellation timestamps from the 'cancel' action
    cancellations = untriggered_orders[untriggered_orders["action"] == "cancel"][
        ["user", "order_orderid", "ts_y"]
    ].copy()
    cancellations.rename(columns={"ts_y": "cancellation_ts"}, inplace=True)
    cancellations.drop_duplicates(subset=["user", "order_orderid"], inplace=True)

    # Merge cancellation time onto the base order data
    unique_orders = pd.merge(
        placements, cancellations, on=["user", "order_orderid"], how="left"
    )

    # Group orders by market
    grouped_by_market = unique_orders.groupby(["order_marketindex", "order_markettype"])

    missed = []
    with ThreadPoolExecutor() as executor:
        futures = {
            executor.submit(
                check_missed_orders_for_market,
                market_orders,
                context,
                athena,
                athena_config,
            ): market_name
            for market_name, market_orders in grouped_by_market
        }

        for future in as_completed(futures):
            missed.extend(future.result())

    if missed:
        return pd.DataFrame(missed)
    else:
        # Return a dataframe with the same columns as the input, even if empty
        return pd.DataFrame(columns=untriggered_orders.columns)


@asset(
    group_name="orders",
    description="Get triggered trigger orders",
    partitions_def=daily_partitions,
)
def triggered_trigger_orders(trigger_orders_with_actions: pd.DataFrame) -> pd.DataFrame:
    return trigger_orders_with_actions[
        trigger_orders_with_actions["order_status"] == "triggered"
    ]


@asset(
    group_name="orders",
    description="Get a list of triggered orders that were partially filled.",
    partitions_def=daily_partitions,
)
def partially_filled_triggered_orders(
    triggered_trigger_orders: pd.DataFrame, clean_trades: pd.DataFrame
) -> pd.DataFrame:
    """
    Provides a list of triggered orders that were partially filled by summing all trade fills.
    """
    analysis_df = triggered_trigger_orders.copy()

    total_fills = (
        clean_trades.groupby(["taker", "takerorderid"])["baseassetamountfilled"]
        .sum()
        .reset_index()
    )

    trade_lookup = total_fills.set_index(["taker", "takerorderid"])[
        "baseassetamountfilled"
    ]

    analysis_df_index = pd.MultiIndex.from_frame(analysis_df[["user", "order_orderid"]])

    analysis_df["baseassetamount_filled"] = analysis_df_index.map(trade_lookup).fillna(
        0
    )

    partially_filled = analysis_df[
        analysis_df["baseassetamount_filled"].astype(float)
        < analysis_df["order_baseassetamount"].astype(float)
    ]

    return partially_filled


@asset(
    group_name="orders",
    description="Analyze the fill status of triggered orders and provide a summary.",
    partitions_def=daily_partitions,
)
def triggered_order_summary(
    clean_trigger_orders: pd.DataFrame,
    pending_trigger_orders: pd.DataFrame,
    canceled_trigger_orders: pd.DataFrame,
    triggered_trigger_orders: pd.DataFrame,
    partially_filled_triggered_orders: pd.DataFrame,
    missed_trigger_orders: pd.DataFrame,
) -> pd.DataFrame:
    total_count = len(
        clean_trigger_orders.drop_duplicates(subset=["user", "order_orderid"])
    )
    pending_count = len(
        pending_trigger_orders.drop_duplicates(subset=["user", "order_orderid"])
    )
    cancel_count = len(
        canceled_trigger_orders.drop_duplicates(subset=["user", "order_orderid"])
    )
    total_triggered_count = len(
        triggered_trigger_orders.drop_duplicates(subset=["user", "order_orderid"])
    )
    triggered_partial_count = len(
        partially_filled_triggered_orders.drop_duplicates(
            subset=["user", "order_orderid"]
        )
    )

    triggered_filled_count = total_triggered_count - triggered_partial_count

    missed_count = len(
        missed_trigger_orders.drop_duplicates(subset=["user", "order_orderid"])
    )

    summary_data = {
        "total_count": total_count,
        "pending_count": pending_count,
        "cancelled_count": cancel_count,
        "triggered_count": total_triggered_count,
        "triggered_filled_count": triggered_filled_count,
        "triggered_partially_filled_count": triggered_partial_count,
        "missed_trigger_count": missed_count,
    }

    return pd.DataFrame(summary_data, index=[0])

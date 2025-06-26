"""
Order assets - Process and analyze order data
"""

from dagster import asset
import pandas as pd
import re
from ..partitions import daily_partitions


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


@asset(
    group_name="orders",
    description="Clean and basic enrichment of trigger order data",
    partitions_def=daily_partitions,
)
def clean_trigger_orders(
    raw_trigger_orders: pd.DataFrame = None,
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
    # Get the order IDs that have been triggered
    triggered_order_ids = clean_actions[clean_actions["action"] == "trigger"][
        "takerorderid"
    ].unique()

    # Get the order IDs that have been canceled
    canceled_order_ids = clean_actions[clean_actions["action"] == "cancel"][
        "takerorderid"
    ].unique()

    # Join all trigger orders with their actions
    combined = clean_trigger_orders.merge(
        clean_actions,
        left_on=["user", "order_orderid"],
        right_on=["taker", "takerorderid"],
        how="left",
    )

    # Add status column
    def get_order_status(order_id):
        if order_id in triggered_order_ids:
            return "triggered"
        elif order_id in canceled_order_ids:
            return "canceled"
        else:
            return "pending"

    combined["order_status"] = combined["order_orderid"].apply(get_order_status)

    return combined


@asset(
    group_name="orders",
    description="Get pending trigger orders",
    partitions_def=daily_partitions,
)
def pending_trigger_orders(trigger_orders_with_actions: pd.DataFrame) -> pd.DataFrame:
    # Determine if any of these should've been triggered depending on the oracle price
    return trigger_orders_with_actions[
        trigger_orders_with_actions["order_status"] == "pending"
    ]


@asset(
    group_name="orders",
    description="Get canceled trigger orders",
    partitions_def=daily_partitions,
)
def canceled_trigger_orders(trigger_orders_with_actions: pd.DataFrame) -> pd.DataFrame:
    # Determine if any of these should be triggered depending on the oracle price and were instead cancelled (potentially by the user because it did not trigger)
    return trigger_orders_with_actions[
        trigger_orders_with_actions["order_status"] == "canceled"
    ]


@asset(
    group_name="orders",
    description="Get triggered trigger orders",
    partitions_def=daily_partitions,
)
def triggered_trigger_orders(trigger_orders_with_actions: pd.DataFrame) -> pd.DataFrame:
    # Determine if the were completely filled
    # Determine if there are any triggered orders that were then cancelled
    # Determine if they were cancelled because they weren't filled
    # Determine if the were completely filled

    return trigger_orders_with_actions[
        trigger_orders_with_actions["order_status"] == "triggered"
    ]

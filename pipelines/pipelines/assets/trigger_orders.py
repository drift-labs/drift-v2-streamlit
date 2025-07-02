"""
Order assets - Process and analyze order data
"""

from dagster import asset, AssetIn, TimeWindowPartitionMapping, BackfillPolicy
import pandas as pd
import re
import numpy as np
from typing import Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
from ..resources import AthenaConfig, WrappedAthenaClientResource
from ..partitions import dlob_start_daily_partitions
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

    dlob = get_oracle_prices_for_market(
        context, athena, athena_config, market_index, market_type
    )
    dlob["oracle_price"] = pd.to_numeric(dlob["oracle"], errors="coerce")
    dlob["timestamp"] = pd.to_numeric(dlob["ts"], errors="coerce")

    for _, order_row in market_orders.iterrows():
        order_ts = pd.to_numeric(order_row["ts"], errors="coerce")
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
    group_name="trigger_orders",
    description="Clean and basic enrichment of trigger order data",
    partitions_def=dlob_start_daily_partitions,
)
def clean_trigger_orders(
    raw_trigger_orders: pd.DataFrame,
) -> pd.DataFrame:
    """
    Basic cleaning of order data of raw_trigger_orders
    """
    return clean_orders_logic(raw_trigger_orders)


@asset(
    group_name="trigger_orders",
    description="Calculates the daily state of all orders, carrying forward pending orders from the previous day.",
    partitions_def=dlob_start_daily_partitions,
    backfill_policy=BackfillPolicy.single_run(),
    ins={
        "rolling_order_state": AssetIn(
            "daily_order_state",
            partition_mapping=TimeWindowPartitionMapping(
                start_offset=-1, end_offset=-1
            ),
        ),
    },
)
def daily_order_state(
    context,
    clean_trigger_orders: pd.DataFrame,
    clean_actions: pd.DataFrame,
    rolling_order_state: Optional[pd.DataFrame],
) -> pd.DataFrame:
    """
    Categorize all trigger orders as pending, canceled, or triggered.
    This is a stateful asset that depends on its own output from the previous day
    to carry forward orders that were still pending.
    """
    all_orders = clean_trigger_orders.copy()

    if rolling_order_state is not None and not rolling_order_state.empty:
        previous_pending_orders = rolling_order_state[
            rolling_order_state["order_status"] == "pending"
        ].copy()

        if not previous_pending_orders.empty:
            context.log.info(
                f"Adding {len(previous_pending_orders)} pending orders from previous day"
            )

            order_columns = [
                col
                for col in clean_trigger_orders.columns
                if col in previous_pending_orders.columns
            ]

            previous_orders_clean = previous_pending_orders[order_columns].copy()

            previous_orders_clean["carried_forward"] = True
            all_orders["carried_forward"] = False

            all_orders = pd.concat(
                [all_orders, previous_orders_clean], ignore_index=True
            )

            context.log.info(
                f"Total orders for analysis: {len(all_orders)} (today: {len(clean_trigger_orders)}, carried forward: {len(previous_orders_clean)})"
            )
        else:
            context.log.info("No pending orders from previous day to carry forward.")
            all_orders["carried_forward"] = False
    else:
        context.log.info("No previous daily order state found. Starting fresh.")
        all_orders["carried_forward"] = False

    actions = clean_actions.copy()

    actions["user"] = actions["taker"].fillna(actions["maker"])
    actions["orderid"] = actions["takerorderid"].fillna(actions["makerorderid"])

    combined = pd.merge(
        all_orders,
        actions,
        left_on=["user", "order_orderid"],
        right_on=["user", "orderid"],
        how="left",
        suffixes=(None, "_action"),
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
    group_name="trigger_orders",
    description="Filters the daily order state to get only pending orders.",
    partitions_def=dlob_start_daily_partitions,
)
def pending_trigger_orders(daily_order_state: pd.DataFrame) -> pd.DataFrame:
    return daily_order_state[daily_order_state["order_status"] == "pending"]


@asset(
    group_name="trigger_orders",
    description="Filters the daily order state to get only canceled orders.",
    partitions_def=dlob_start_daily_partitions,
)
def canceled_trigger_orders(daily_order_state: pd.DataFrame) -> pd.DataFrame:
    return daily_order_state[daily_order_state["order_status"] == "canceled"]


@asset(
    group_name="trigger_orders",
    description="Identifies pending or canceled orders that would have been triggered by oracle prices.",
    partitions_def=dlob_start_daily_partitions,
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
        ["user", "order_orderid", "ts_action"]
    ].copy()
    cancellations.rename(columns={"ts_action": "cancellation_ts"}, inplace=True)
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
    group_name="trigger_orders",
    description="Filters the daily order state to get only triggered orders.",
    partitions_def=dlob_start_daily_partitions,
)
def triggered_trigger_orders(daily_order_state: pd.DataFrame) -> pd.DataFrame:
    return daily_order_state[daily_order_state["order_status"] == "triggered"]


@asset(
    group_name="trigger_orders",
    description="Get a list of triggered orders that were partially filled.",
    partitions_def=dlob_start_daily_partitions,
)
def partially_filled_triggered_orders(
    triggered_trigger_orders: pd.DataFrame, clean_trades: pd.DataFrame
) -> pd.DataFrame:
    """
    Provides a list of triggered orders that were partially filled by summing all trade fills.
    """
    analysis_df = triggered_trigger_orders.copy()

    # Consider both taker and maker fills to get the true total fill amount
    trades = clean_trades.copy()
    taker_fills = trades[["taker", "takerorderid", "baseassetamountfilled"]].rename(
        columns={"taker": "user", "takerorderid": "orderid"}
    )
    maker_fills = trades[["maker", "makerorderid", "baseassetamountfilled"]].rename(
        columns={"maker": "user", "makerorderid": "orderid"}
    )

    all_fills = pd.concat([taker_fills, maker_fills]).dropna(subset=["user", "orderid"])

    total_fills = (
        all_fills.groupby(["user", "orderid"])["baseassetamountfilled"]
        .sum()
        .reset_index()
    )

    trade_lookup = total_fills.set_index(["user", "orderid"])["baseassetamountfilled"]

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
    group_name="trigger_orders",
    description="Analyze the fill status of triggered orders and provide a summary.",
    partitions_def=dlob_start_daily_partitions,
    io_manager_key="csv_io_manager",
)
def triggered_order_summary(
    daily_order_state: pd.DataFrame,
    clean_trigger_orders: pd.DataFrame,
    pending_trigger_orders: pd.DataFrame,
    canceled_trigger_orders: pd.DataFrame,
    triggered_trigger_orders: pd.DataFrame,
    partially_filled_triggered_orders: pd.DataFrame,
    missed_trigger_orders: pd.DataFrame,
) -> pd.DataFrame:
    total_count = len(
        daily_order_state.drop_duplicates(subset=["user", "order_orderid"])
    )
    new_order_count = len(
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

    # Carried forward order analysis
    if "carried_forward" in daily_order_state.columns:
        carried_forward_orders = daily_order_state[
            daily_order_state["carried_forward"]
        ].drop_duplicates(subset=["user", "order_orderid"])
        carried_forward_count = len(carried_forward_orders)

        resolved_carried_forward_orders = carried_forward_orders[
            carried_forward_orders["order_status"].isin(["triggered", "canceled"])
        ]
        resolved_carried_forward_count = len(resolved_carried_forward_orders)
    else:
        carried_forward_count = 0
        resolved_carried_forward_count = 0

    summary_data = {
        "total_count": total_count,
        "new_order_count": new_order_count,
        "pending_count": pending_count,
        "cancelled_count": cancel_count,
        "triggered_count": total_triggered_count,
        "triggered_filled_count": triggered_filled_count,
        "triggered_partially_filled_count": triggered_partial_count,
        "missed_trigger_count": missed_count,
        "carried_forward_count": carried_forward_count,
        "resolved_carried_forward_count": resolved_carried_forward_count,
    }

    return pd.DataFrame(summary_data, index=[0])

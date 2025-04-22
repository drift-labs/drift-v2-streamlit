# old dashboard: for each user:
# - net deposits
# - all time pnl + fees
# - total volume
# - position over time

import datetime

import numpy as np
import pandas as pd
import plotly.graph_objs as go
import streamlit as st
from driftpy.accounts import get_state_account
from driftpy.constants.numeric_constants import BASE_PRECISION, QUOTE_PRECISION
from driftpy.constants.perp_markets import mainnet_perp_market_configs
from driftpy.constants.spot_markets import mainnet_spot_market_configs
from driftpy.drift_client import DriftClient
from solana.rpc.types import MemcmpOpts
from solders.pubkey import Pubkey

from datafetch.user_records import (
    get_user_deposits,
    get_user_funding,
    get_user_settle_pnls,
    get_user_trades,
    get_user_withdrawals,
)

pd.options.plotting.backend = "plotly"


def filter_dups(df):
    df = df.drop_duplicates(
        [
            "fillerReward",
            "baseAssetAmountFilled",
            "quoteAssetAmountFilled",
            "takerPnl",
            "makerPnl",
            "takerFee",
            "makerRebate",
            "refereeDiscount",
            "quoteAssetAmountSurplus",
            "takerOrderBaseAssetAmount",
            "takerOrderCumulativeBaseAssetAmountFilled",
            "takerOrderCumulativeQuoteAssetAmountFilled",
            "takerOrderFee",
            "makerOrderBaseAssetAmount",
            "makerOrderCumulativeBaseAssetAmountFilled",
            "makerOrderCumulativeQuoteAssetAmountFilled",
            "makerOrderFee",
            "oraclePrice",
            "makerFee",
            "txSig",
            "slot",
            "ts",
            "action",
            "actionExplanation",
            "marketIndex",
            "marketType",
            "filler",
            "fillRecordId",
            "taker",
            "takerOrderId",
            "takerOrderDirection",
            "maker",
            "makerOrderId",
            "makerOrderDirection",
            "spotFulfillmentMethodFee",
        ]
    ).reset_index(drop=True)
    return df


def process_trades_df(
    fairs_marked_out_df: pd.DataFrame, raw_trades_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Adds some columns to the market_trades_df for analysis in a vectorized way
    """
    # Select required columns
    filtered = raw_trades_df[
        [
            "filler",
            "fillRecordId",
            "taker",
            "takerOrderId",
            "takerOrderDirection",
            "takerFee",
            "maker",
            "makerOrderId",
            "makerOrderDirection",
            "makerFee",
            "baseAssetAmountFilled",
            "quoteAssetAmountFilled",
            "oraclePrice",
            "actionExplanation",
            "txSig",
            "slot",
        ]
    ].copy()

    # Vectorized operations
    filtered["makerBaseSigned"] = np.where(
        filtered["makerOrderDirection"] == "long",
        filtered["baseAssetAmountFilled"],
        filtered["baseAssetAmountFilled"] * -1,
    )
    filtered["makerQuoteSigned"] = np.where(
        filtered["makerOrderDirection"] == "long",
        -1 * filtered["quoteAssetAmountFilled"],
        filtered["quoteAssetAmountFilled"],
    )
    filtered["takerBaseSigned"] = np.where(
        filtered["takerOrderDirection"] == "long",
        filtered["baseAssetAmountFilled"],
        filtered["baseAssetAmountFilled"] * -1,
    )
    filtered["takerQuoteSigned"] = np.where(
        filtered["takerOrderDirection"] == "long",
        -1 * filtered["quoteAssetAmountFilled"],
        filtered["quoteAssetAmountFilled"],
    )
    filtered["fillPrice"] = (
        filtered["quoteAssetAmountFilled"] / filtered["baseAssetAmountFilled"]
    )
    filtered["isFillerMaker"] = filtered["filler"] == filtered["maker"]
    filtered["isFillerTaker"] = filtered["filler"] == filtered["taker"]
    filtered["makerOrderDirectionNum"] = np.where(
        filtered["makerOrderDirection"] == "long", 1, -1
    )
    filtered["takerOrderDirectionNum"] = np.where(
        filtered["takerOrderDirection"] == "long", 1, -1
    )

    # # Find closest times in fairs_marked_out_df for each trade
    # closest_times = fairs_marked_out_df.index.get_indexer(filtered.index, method='nearest')

    # # Add price_t0, price_t5, etc. from fairs_marked_out_df
    # for markout_period in markout_periods:
    #     filtered[f'price_{markout_period}'] = fairs_marked_out_df[f'price_{markout_period}'].iloc[closest_times].values

    #     # Calculate markouts
    #     filtered[f'maker_markout_{markout_period}'] = np.where(
    #         filtered['makerOrderDirectionNum'] == 1,
    #         filtered['baseAssetAmountFilled'] * (filtered[f'price_{markout_period}'] - filtered['fillPrice']),
    #         filtered['baseAssetAmountFilled'] * (filtered['fillPrice'] - filtered[f'price_{markout_period}'])
    #     )
    #     filtered[f'taker_markout_{markout_period}'] = np.where(
    #         filtered['takerOrderDirectionNum'] == 1,
    #         filtered['baseAssetAmountFilled'] * (filtered[f'price_{markout_period}'] - filtered['fillPrice']),
    #         filtered['baseAssetAmountFilled'] * (filtered['fillPrice'] - filtered[f'price_{markout_period}'])
    #     )

    return filtered


def render_trades_stats_for_user_account(processed_trades_df, filter_ua):
    if filter_ua is None:
        user_trades_df = processed_trades_df.loc[
            (processed_trades_df["maker"].isna())
            | (processed_trades_df["taker"].isna())
        ].copy()
    else:
        user_trades_df = processed_trades_df.loc[
            (processed_trades_df["maker"] == filter_ua)
            | (processed_trades_df["taker"] == filter_ua)
        ].copy()

    user_trades_df["isMaker"] = user_trades_df["maker"] == filter_ua
    user_trades_df["counterparty"] = np.where(
        user_trades_df["maker"] == filter_ua,
        user_trades_df["taker"],
        user_trades_df["maker"],
    )
    user_trades_df["user_direction"] = np.where(
        user_trades_df["maker"] == filter_ua,
        user_trades_df["makerOrderDirection"],
        user_trades_df["takerOrderDirection"],
    )

    user_trades_df["user_direction_num"] = np.where(
        user_trades_df["maker"] == filter_ua,
        user_trades_df["makerOrderDirectionNum"],
        user_trades_df["takerOrderDirectionNum"],
    )

    user_trades_df["user_fee_recv"] = np.where(
        user_trades_df["maker"] == filter_ua,
        -1 * user_trades_df["makerFee"],
        -1 * user_trades_df["takerFee"],
    )

    user_trades_df["user_base"] = np.where(
        user_trades_df["maker"] == filter_ua,
        user_trades_df["makerBaseSigned"],
        user_trades_df["takerBaseSigned"],
    )

    user_trades_df["user_quote"] = np.where(
        user_trades_df["maker"] == filter_ua,
        user_trades_df["makerQuoteSigned"],
        user_trades_df["takerQuoteSigned"],
    )

    # for markout_period in markout_periods:
    # 	user_trades_df[f'user_markout_{markout_period}'] = np.where(
    # 		user_trades_df['maker'] == filter_ua,
    # 		user_trades_df[f'maker_markout_{markout_period}'],
    # 		user_trades_df[f'taker_markout_{markout_period}'],
    # 	)

    user_trades_df["user_cum_base"] = user_trades_df[
        "user_base"
    ].cumsum()  # base_position
    user_trades_df["user_cum_base_prev"] = (
        user_trades_df["user_cum_base"].shift(1).fillna(0)
    )  # base_position_prev
    user_trades_df["user_cum_quote"] = user_trades_df["user_quote"].cumsum()

    # update types:
    # 0: flip pos
    # 1: increase pos
    # -1: decrease pos

    user_trades_df["position_update"] = 0
    user_trades_df["user_quote_entry_amount"] = 0.0
    user_trades_df["user_quote_breakeven_amount"] = 0.0
    user_trades_df["realized_pnl"] = 0.0

    for i in range(0, len(user_trades_df)):
        prev_row = user_trades_df.iloc[i - 1]
        current_row = user_trades_df.iloc[i]

        prev_quote_entry_amt = prev_row["user_quote_entry_amount"]
        prev_quote_breakeven_amt = prev_row["user_quote_breakeven_amount"]
        delta_base_amt = np.abs(current_row["user_base"])
        curr_base_amt = np.abs(prev_row["user_cum_base"])

        if current_row["user_cum_base"] * current_row["user_cum_base_prev"] < 0:
            # flipped direction
            user_trades_df.loc[user_trades_df.index[i], "position_update"] = 0
            # same for BE and entry
            new_quote = current_row["user_quote"] - (
                current_row["user_quote"] * curr_base_amt / delta_base_amt
            )
            user_trades_df.loc[user_trades_df.index[i], "user_quote_entry_amount"] = (
                new_quote
            )
            user_trades_df.loc[
                user_trades_df.index[i], "user_quote_breakeven_amount"
            ] = new_quote
            user_trades_df.loc[user_trades_df.index[i], "realized_pnl"] = prev_row[
                "user_quote_entry_amount"
            ] + (current_row["user_quote"] - new_quote)
        elif current_row["user_cum_base_prev"] == 0:
            # opening new position
            user_trades_df.loc[user_trades_df.index[i], "position_update"] = 1
            user_trades_df.loc[user_trades_df.index[i], "user_quote_entry_amount"] = (
                prev_quote_entry_amt + current_row["user_quote"]
            )
            user_trades_df.loc[
                user_trades_df.index[i], "user_quote_breakeven_amount"
            ] = prev_quote_breakeven_amt + current_row["user_quote"]
        else:
            if current_row["user_direction_num"] == np.sign(
                current_row["user_cum_base_prev"]
            ):
                # increase position
                user_trades_df.loc[user_trades_df.index[i], "position_update"] = 1
                user_trades_df.loc[
                    user_trades_df.index[i], "user_quote_entry_amount"
                ] = prev_quote_entry_amt + current_row["user_quote"]
                user_trades_df.loc[
                    user_trades_df.index[i], "user_quote_breakeven_amount"
                ] = prev_quote_breakeven_amt + current_row["user_quote"]
                user_trades_df.loc[user_trades_df.index[i], "realized_pnl"] = 0
            else:
                # decrease position
                user_trades_df.loc[user_trades_df.index[i], "position_update"] = -1
                new_quote_entry_amt = prev_quote_entry_amt - (
                    prev_quote_entry_amt * delta_base_amt / curr_base_amt
                )
                user_trades_df.loc[
                    user_trades_df.index[i], "user_quote_entry_amount"
                ] = new_quote_entry_amt
                user_trades_df.loc[
                    user_trades_df.index[i], "user_quote_breakeven_amount"
                ] = prev_quote_breakeven_amt - (
                    prev_quote_breakeven_amt * delta_base_amt / curr_base_amt
                )
                user_trades_df.loc[user_trades_df.index[i], "realized_pnl"] = (
                    prev_row["user_quote_entry_amount"]
                    - new_quote_entry_amt
                    + current_row["user_quote"]
                )

    user_trades_df["avg_price"] = np.abs(
        user_trades_df["user_quote_entry_amount"] / user_trades_df["user_cum_base"]
    )
    user_trades_df["user_cum_fee"] = user_trades_df["user_fee_recv"].cumsum()
    user_trades_df["user_cum_pnl"] = user_trades_df["realized_pnl"].cumsum()

    return user_trades_df


def plot_cumulative_pnl_for_user_account(user_trades_df, filter_ua):
    # Create a new figure for cumulative base and quote plots
    # fig_cumulative = make_subplots(rows=2, cols=1, shared_xaxes=True, subplot_titles=('Cumulative Base Asset', 'Cumulative PnL'))

    fig_1 = go.Figure()
    fig_1.update_layout(
        title_text=f"Cum. Base Amounts ({filter_ua})",
        height=400,
        width=1200,
        showlegend=True,
    )

    fig_2 = go.Figure()
    fig_2.update_layout(
        title_text=f"Cum. PnL({filter_ua})",
        height=400,
        width=1200,
        showlegend=True,
        barmode="overlay",
    )

    marker_size = 3

    fig_1.add_trace(
        go.Scatter(
            x=user_trades_df.index,
            y=user_trades_df["user_cum_base"],
            mode="lines+markers",
            name="Cumulative Base",
            line=dict(width=1),
            marker=dict(size=marker_size),
        ),
        # row=1, col=1
    )

    fig_2.add_trace(
        go.Scatter(
            x=user_trades_df.index,
            y=user_trades_df["user_cum_fee"],
            mode="lines+markers",
            name="Cumulative Fee Received",
            line=dict(width=1),
            marker=dict(size=marker_size),
        ),
        # row=2, col=1
    )
    fig_2.add_trace(
        go.Scatter(
            x=user_trades_df.index,
            y=user_trades_df["user_cum_pnl"],
            mode="lines+markers",
            name="Cumulative PnL",
            line=dict(width=1),
            marker=dict(size=marker_size),
        ),
        # row=2, col=1
    )

    st.plotly_chart(fig_1, use_container_width=True)
    st.plotly_chart(fig_2, use_container_width=True)


def calculate_performance(records_df: pd.DataFrame, selected_user_pk: Pubkey):
    """
    Processes a DataFrame of time-sorted user records (trades, deposits, etc.)
    to calculate PnL, position, and balance over time.

    Args:
        records_df: Combined DataFrame of user records sorted by timestamp.
        selected_user_pk: The public key of the user account being analyzed.

    Returns:
        A tuple containing DataFrames for:
        - pnl_summary (cumulative pnl types over time)
        - positions (position size per market over time)
        - balance (USDC balance over time)
    """

    if records_df.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    # Initialize state variables
    pnl_summary_rows = []
    position_rows = []
    balance_rows = []

    market_states = {}  # key: (marketType, marketIndex), value: {position_size, cost_basis, realized_pnl, funding_pnl, fees}
    current_usdc_balance = 0.0
    net_deposits = 0.0
    cumulative_fees = 0.0
    cumulative_funding_pnl = 0.0
    cumulative_realized_trading_pnl = 0.0

    # --- Precision Constants ---
    # Using imported BASE_PRECISION and QUOTE_PRECISION

    # --- Process records chronologically ---
    last_ts = None
    for index, record in records_df.iterrows():
        ts = record["ts"]
        record_type = record["record_type"]

        # Initialize market state if first time seeing it
        market_key = (
            record.get("marketType", "spot"),
            record.get("marketIndex", 0),
        )  # Default to spot 0 (USDC)
        if (
            market_key not in market_states
            and record_type != "deposit"
            and record_type != "withdrawal"
        ):
            market_states[market_key] = {
                "position_size": 0.0,
                "cost_basis": 0.0,
                "realized_pnl": 0.0,
                "funding_pnl": 0.0,
                "fees": 0.0,
            }

        # --- Handle specific record types ---
        if record_type == "deposit":
            try:
                amount_raw = pd.to_numeric(record.get("amount"), errors="coerce")
                if pd.isna(amount_raw):
                    print(
                        f"Warning: Skipping deposit record at {record.get('ts', 'N/A')} due to invalid amount (Tx: {record.get('txSig', 'N/A')})"
                    )
                    st.warning(
                        f"Skipping deposit record at {record.get('ts', 'N/A')} due to invalid amount (Tx: {record.get('txSig', 'N/A')})"
                    )
                    continue

                # TODO: Handle non-USDC deposits - requires marketIndex and precision lookup
                # Assuming USDC deposit for now
                amount = amount_raw / QUOTE_PRECISION
                current_usdc_balance += amount
                net_deposits += amount
                # print(f"{ts}: Deposit {amount:.2f}, New Balance: {current_usdc_balance:.2f}")
            except Exception as e:
                st.error(
                    f"Error processing deposit record at {record.get('ts', 'N/A')} (Tx: {record.get('txSig', 'N/A')}): {e}"
                )
                continue  # Skip problematic record

        elif record_type == "withdrawal":
            try:
                amount_raw = pd.to_numeric(record.get("amount"), errors="coerce")
                if pd.isna(amount_raw):
                    print(
                        f"Warning: Skipping withdrawal record at {record.get('ts', 'N/A')} due to invalid amount (Tx: {record.get('txSig', 'N/A')})"
                    )
                    st.warning(
                        f"Skipping withdrawal record at {record.get('ts', 'N/A')} due to invalid amount (Tx: {record.get('txSig', 'N/A')})"
                    )
                    continue

                # TODO: Handle non-USDC withdrawals
                # Assuming USDC withdrawal for now
                amount = amount_raw / QUOTE_PRECISION
                current_usdc_balance -= amount
                net_deposits -= amount
                # print(f"{ts}: Withdrawal {amount:.2f}, New Balance: {current_usdc_balance:.2f}")
            except Exception as e:
                st.error(
                    f"Error processing withdrawal record at {record.get('ts', 'N/A')} (Tx: {record.get('txSig', 'N/A')}): {e}"
                )
                continue  # Skip problematic record

        elif record_type == "settlePnl":
            # Add similar handling for pnl
            try:
                pnl_raw = pd.to_numeric(record.get("pnl"), errors="coerce")
                if pd.isna(pnl_raw):
                    print(
                        f"Warning: Skipping settlePnl record at {record.get('ts', 'N/A')} due to invalid pnl (Tx: {record.get('txSig', 'N/A')})"
                    )
                    st.warning(
                        f"Skipping settlePnl record at {record.get('ts', 'N/A')} due to invalid pnl (Tx: {record.get('txSig', 'N/A')})"
                    )
                    continue

                pnl = pnl_raw / QUOTE_PRECISION
                market_index = record.get("marketIndex")
                market_type = record.get("marketType")
                if market_index is None or market_type is None:
                    st.warning(
                        f"Skipping settlePnl record at {record.get('ts', 'N/A')} due to missing market info (Tx: {record.get('txSig', 'N/A')})"
                    )
                    continue
                m_key = (market_type, market_index)

                # Ensure market state exists
                if m_key not in market_states:
                    market_states[m_key] = {
                        "position_size": 0.0,
                        "cost_basis": 0.0,
                        "realized_pnl": 0.0,
                        "funding_pnl": 0.0,
                        "fees": 0.0,
                    }

                current_usdc_balance += pnl
                market_states[m_key]["realized_pnl"] += pnl
                cumulative_realized_trading_pnl += pnl
                # print(f"{ts}: SettlePnl ({market_type} {market_index}): {pnl:.2f}, New Balance: {current_usdc_balance:.2f}")
            except Exception as e:
                st.error(
                    f"Error processing settlePnl record at {record.get('ts', 'N/A')} (Tx: {record.get('txSig', 'N/A')}): {e}"
                )
                continue

        elif record_type == "funding":
            # Add similar handling for funding amount
            try:
                amount_raw = pd.to_numeric(record.get("amount"), errors="coerce")
                if pd.isna(amount_raw):
                    print(
                        f"Warning: Skipping funding record at {record.get('ts', 'N/A')} due to invalid amount (Tx: {record.get('txSig', 'N/A')})"
                    )
                    st.warning(
                        f"Skipping funding record at {record.get('ts', 'N/A')} due to invalid amount (Tx: {record.get('txSig', 'N/A')})"
                    )
                    continue

                amount = amount_raw / QUOTE_PRECISION
                market_index = record.get("marketIndex")
                market_type = "perp"  # Funding only applies to perps
                if market_index is None:
                    st.warning(
                        f"Skipping funding record at {record.get('ts', 'N/A')} due to missing market index (Tx: {record.get('txSig', 'N/A')})"
                    )
                    continue
                m_key = (market_type, market_index)

                # Ensure market state exists
                if m_key not in market_states:
                    market_states[m_key] = {
                        "position_size": 0.0,
                        "cost_basis": 0.0,
                        "realized_pnl": 0.0,
                        "funding_pnl": 0.0,
                        "fees": 0.0,
                    }

                current_usdc_balance += amount
                market_states[m_key]["funding_pnl"] += amount
                cumulative_funding_pnl += amount
                # print(f"{ts}: Funding ({market_type} {market_index}): {amount:.2f}, New Balance: {current_usdc_balance:.2f}")
            except Exception as e:
                st.error(
                    f"Error processing funding record at {record.get('ts', 'N/A')} (Tx: {record.get('txSig', 'N/A')}): {e}"
                )
                continue

        elif record_type == "trade":
            try:
                # Ensure values are numeric, default to 0 if missing/invalid
                # Use .get() to handle potentially missing columns gracefully
                base_filled_raw = pd.to_numeric(
                    record.get("baseAssetAmountFilled"), errors="coerce"
                )
                quote_filled_raw = pd.to_numeric(
                    record.get("quoteAssetAmountFilled"), errors="coerce"
                )
                taker_fee_raw = pd.to_numeric(record.get("takerFee"), errors="coerce")
                maker_fee_raw = pd.to_numeric(record.get("makerFee"), errors="coerce")

                # Skip if essential amounts are missing/invalid
                if pd.isna(base_filled_raw) or pd.isna(quote_filled_raw):
                    print(
                        f"Warning: Skipping trade record at {record.get('ts', 'N/A')} due to missing amount data (Tx: {record.get('txSig', 'N/A')})"
                    )
                    st.warning(
                        f"Skipping trade record at {record.get('ts', 'N/A')} due to missing amount data (Tx: {record.get('txSig', 'N/A')})"
                    )
                    continue

                # Using imported BASE_PRECISION directly
                current_base_precision = BASE_PRECISION

                base_filled = base_filled_raw / current_base_precision
                quote_filled = quote_filled_raw / QUOTE_PRECISION

                # Need market index/type for market_states key
                market_index = record.get("marketIndex")
                market_type = record.get("marketType")
                if market_index is None or market_type is None:
                    print(
                        f"Warning: Skipping trade record at {record.get('ts', 'N/A')} due to missing market info (Tx: {record.get('txSig', 'N/A')})"
                    )
                    st.warning(
                        f"Skipping trade record at {record.get('ts', 'N/A')} due to missing market info (Tx: {record.get('txSig', 'N/A')})"
                    )
                    continue

                m_key = (market_type, market_index)

                # Ensure market state exists before potentially updating fees or position
                if m_key not in market_states:
                    market_states[m_key] = {
                        "position_size": 0.0,
                        "cost_basis": 0.0,
                        "realized_pnl": 0.0,
                        "funding_pnl": 0.0,
                        "fees": 0.0,
                    }

                fee = 0.0
                base_change = 0.0
                quote_change = (
                    0.0  # Net quote change for this trade from user perspective
                )

                # Determine user's role and apply amounts correctly
                taker_pk = str(record.get("taker"))
                maker_pk = str(record.get("maker"))
                user_pk_str = str(selected_user_pk)

                if taker_pk == user_pk_str:
                    fee_raw = taker_fee_raw if not pd.isna(taker_fee_raw) else 0
                    fee = fee_raw / QUOTE_PRECISION
                    cumulative_fees += fee
                    market_states[m_key]["fees"] += fee
                    if record.get("takerOrderDirection") == "long":
                        base_change = base_filled
                        quote_change = -quote_filled  # Paid quote
                    else:  # short or unknown defaults to short calculation
                        base_change = -base_filled
                        quote_change = quote_filled  # Received quote

                elif maker_pk == user_pk_str:
                    fee_raw = maker_fee_raw if not pd.isna(maker_fee_raw) else 0
                    fee = (
                        fee_raw / QUOTE_PRECISION
                    )  # Maker fee might be rebate (negative)
                    cumulative_fees += fee
                    market_states[m_key]["fees"] += fee
                    if record.get("makerOrderDirection") == "long":
                        base_change = base_filled
                        quote_change = -quote_filled  # Paid quote
                    else:  # short or unknown defaults to short calculation
                        base_change = -base_filled
                        quote_change = quote_filled  # Received quote
                else:
                    # Should not happen if we fetched user-specific trades, but check anyway
                    print(
                        f"Warning: Trade record {record.get('txSig', 'N/A')} doesn't involve selected user {user_pk_str}"
                    )
                    continue

                # Update position size
                market_states[m_key]["position_size"] += base_change

                # Update balance (quote change from trade - fees)
                current_usdc_balance += quote_change - fee

                # Record position change
                position_rows.append(
                    {
                        "ts": record.get("ts"),  # Use .get for safety
                        "marketIndex": market_index,
                        "marketType": market_type,
                        "position_size": market_states[m_key]["position_size"],
                        "base_change": base_change,
                        "quote_change": quote_change,
                        "fee": fee,
                        "record_type": "trade",
                    }
                )

            except Exception as e:
                st.error(
                    f"Error processing trade record at {record.get('ts', 'N/A')} (Tx: {record.get('txSig', 'N/A')}): {e}"
                )
                # Optionally print the problematic record
                # print(f"Problematic trade record data: {record.to_dict()}")
                continue  # Skip problematic trade record

        # --- Record state after processing the record ---
        if (
            ts != last_ts
        ):  # Avoid duplicate entries if multiple records have the exact same timestamp
            # Record USDC balance and PnL summary
            balance_rows.append(
                {
                    "ts": ts,
                    "usdc_balance": current_usdc_balance,
                    "net_deposits": net_deposits,
                }
            )
            pnl_summary_rows.append(
                {
                    "ts": ts,
                    "realized_trading_pnl": cumulative_realized_trading_pnl,
                    "funding_pnl": cumulative_funding_pnl,
                    "fees": cumulative_fees,
                    "total_realized_pnl": cumulative_realized_trading_pnl
                    + cumulative_funding_pnl
                    - cumulative_fees,
                }
            )

            # Record position for all markets at this timestamp for plotting continuity
            for m_key, state in market_states.items():
                # Only record if position is non-zero or was non-zero previously (optional)
                # if state['position_size'] != 0:
                position_rows.append(
                    {
                        "ts": ts,
                        "marketIndex": m_key[1],
                        "marketType": m_key[0],
                        "position_size": state["position_size"],
                        "record_type": record_type,  # Mark what triggered this state log
                    }
                )

            last_ts = ts

    # --- Create output DataFrames ---
    pnl_summary_df = pd.DataFrame(pnl_summary_rows)
    positions_df = pd.DataFrame(position_rows)
    balance_df = pd.DataFrame(balance_rows)

    # Set timestamp as index for easier plotting
    if not pnl_summary_df.empty:
        pnl_summary_df = pnl_summary_df.set_index("ts")
    if not positions_df.empty:
        positions_df = positions_df.set_index(
            "ts"
        )  # Might need multi-index with market
    if not balance_df.empty:
        balance_df = balance_df.set_index("ts")

    return pnl_summary_df, positions_df, balance_df


def plot_performance_summary(
    pnl_summary_df: pd.DataFrame,
    positions_df: pd.DataFrame,
    balance_df: pd.DataFrame,
    selected_user_pk: Pubkey,
):
    """Plots the performance summary using Plotly."""

    st.subheader(f"Performance Analysis for {str(selected_user_pk)[:10]}...")

    if pnl_summary_df.empty and balance_df.empty and positions_df.empty:
        st.warning("No performance data to plot.")
        return

    # --- PnL Plot ---
    if not pnl_summary_df.empty:
        fig_pnl = go.Figure()
        fig_pnl.update_layout(
            title_text="Cumulative Realized PnL Components",
            height=400,
            # width=1200, # Use container width
            showlegend=True,
        )
        marker_size = 3
        line_width = 1

        fig_pnl.add_trace(
            go.Scatter(
                x=pnl_summary_df.index,
                y=pnl_summary_df["realized_trading_pnl"],
                mode="lines+markers",
                name="Trading PnL (from SettlePnl)",
                line=dict(width=line_width),
                marker=dict(size=marker_size),
            )
        )
        fig_pnl.add_trace(
            go.Scatter(
                x=pnl_summary_df.index,
                y=pnl_summary_df["funding_pnl"],
                mode="lines+markers",
                name="Funding PnL",
                line=dict(width=line_width),
                marker=dict(size=marker_size),
            )
        )
        fig_pnl.add_trace(
            go.Scatter(
                x=pnl_summary_df.index,
                y=-pnl_summary_df["fees"],  # Show fees as cost (negative)
                mode="lines+markers",
                name="Fees Paid",
                line=dict(width=line_width),
                marker=dict(size=marker_size),
            )
        )
        fig_pnl.add_trace(
            go.Scatter(
                x=pnl_summary_df.index,
                y=pnl_summary_df["total_realized_pnl"],
                mode="lines+markers",
                name="Total Realized PnL (Trading + Funding - Fees)",
                line=dict(width=line_width + 1, color="black"),
                marker=dict(size=marker_size),
            )
        )
        st.plotly_chart(fig_pnl, use_container_width=True)
    else:
        st.write("No PnL data available.")

    # --- Balance Plot ---
    if not balance_df.empty:
        fig_bal = go.Figure()
        fig_bal.update_layout(
            title_text="USDC Balance vs Net Deposits",
            height=400,
            showlegend=True,
        )
        fig_bal.add_trace(
            go.Scatter(
                x=balance_df.index,
                y=balance_df["usdc_balance"],
                mode="lines+markers",
                name="USDC Balance",
                line=dict(width=line_width),
                marker=dict(size=marker_size),
            )
        )
        fig_bal.add_trace(
            go.Scatter(
                x=balance_df.index,
                y=balance_df["net_deposits"],
                mode="lines+markers",
                name="Net Deposits",
                line=dict(width=line_width, dash="dash"),
                marker=dict(size=marker_size),
            )
        )
        st.plotly_chart(fig_bal, use_container_width=True)
    else:
        st.write("No balance data available.")

    # --- Position Plot ---
    if not positions_df.empty:
        # Need to pivot or filter for specific markets before plotting
        st.subheader("Positions Over Time (Raw Trade Events)")
        st.write("Plotting position changes requires filtering/pivoting by market.")

        # Example: Plot position for the first market found in trades
        if (
            "marketType" in positions_df.columns
            and "marketIndex" in positions_df.columns
        ):
            unique_markets = positions_df[
                ["marketType", "marketIndex"]
            ].drop_duplicates()
            st.write("Markets traded:", unique_markets)

            # Select a market to plot (e.g., the first one, or let user choose)
            if not unique_markets.empty:
                market_to_plot = unique_markets.iloc[0]
                market_type_filter = market_to_plot["marketType"]
                market_index_filter = market_to_plot["marketIndex"]

                st.write(
                    f"Plotting position for: {market_type_filter} market {market_index_filter}"
                )

                market_positions = positions_df[
                    (positions_df["marketType"] == market_type_filter)
                    & (positions_df["marketIndex"] == market_index_filter)
                ]

                if not market_positions.empty:
                    fig_pos = go.Figure()
                    fig_pos.update_layout(
                        title_text=f"Position Size ({market_type_filter} market {market_index_filter})",
                        height=400,
                        showlegend=True,
                    )
                    # Plotting cumulative position at the time of each trade
                    fig_pos.add_trace(
                        go.Scatter(
                            x=market_positions.index,
                            y=market_positions["position_size"],
                            mode="lines+markers",
                            name="Position Size (at trade time)",
                            line=dict(
                                shape="hv"
                            ),  # Use step chart for position changes
                            marker=dict(size=marker_size),
                        )
                    )
                    st.plotly_chart(fig_pos, use_container_width=True)
                else:
                    st.write("No position data for the selected market.")
            else:
                st.write("No specific markets identified in trade data.")
        else:
            st.write("Market type/index columns missing in position data.")

    else:
        st.write("No position data available.")


async def show_user_perf_for_authority(
    dc: DriftClient,
    user_authority: str,
    market_symbol: str,
    start_date: datetime.date,
    end_date: datetime.date,
):
    user_authority_pk = Pubkey.from_string(user_authority)

    st.write("Fetching user accounts...")
    users = await dc.program.account["User"].all(
        filters=[
            MemcmpOpts(offset=0, bytes="TfwwBiNJtao"),  # User discriminator
            MemcmpOpts(offset=8, bytes=bytes(user_authority_pk)),  # Authority
        ]
    )

    if not users:
        st.error(f"No user accounts found for authority: {user_authority}")
        return

    users.sort(key=lambda x: x.account.sub_account_id)
    user_names = [
        f"{bytes(x.account.name).decode('utf-8', errors='ignore').strip() or 'Unnamed'} (SubID: {x.account.sub_account_id}, PK: {str(x.public_key)[:6]}...)"
        for x in users
    ]

    # Default to the first user if list is not empty
    default_index = (
        0 if user_names else -1
    )  # Handle case where user_names might be empty after filtering/processing

    user_selected_name = st.selectbox(
        "Select subaccount:",
        user_names,
        index=default_index,
        key=f"subaccount_selector_{user_authority}",  # Unique key
    )

    if (
        not user_selected_name
    ):  # Handle case where selectbox is empty or selection is cleared
        st.warning("Please select a subaccount.")
        return

    selected_user_index = user_names.index(user_selected_name)
    selected_user_account_info = users[selected_user_index]
    selected_user_pk = selected_user_account_info.public_key
    # selected_user_account = selected_user_account_info.account

    st.write(f"Selected User Account: {selected_user_pk}")
    # print("selected user account data:", selected_user_account)
    # print("market symbol filter (currently unused in combined view):", market_symbol)
    # print("start date:", start_date)
    # print("end date:", end_date)

    # --- Fetch all relevant records ---
    st.write(
        f"Fetching records for {selected_user_pk} from {start_date} to {end_date}..."
    )
    trades_df = get_user_trades(selected_user_pk, start_date, end_date)
    settle_pnls_df = get_user_settle_pnls(selected_user_pk, start_date, end_date)
    deposits_df = get_user_deposits(selected_user_pk, start_date, end_date)
    withdrawals_df = get_user_withdrawals(selected_user_pk, start_date, end_date)
    funding_df = get_user_funding(selected_user_pk, start_date, end_date)
    # liquidations_df = get_user_liquidations(selected_user_pk, start_date, end_date)

    # --- Combine and sort records ---
    records = []
    if not trades_df.empty:
        trades_df["record_type"] = "trade"
        records.append(trades_df)
    if not settle_pnls_df.empty:
        settle_pnls_df["record_type"] = "settlePnl"
        records.append(settle_pnls_df)
    if not deposits_df.empty:
        deposits_df["record_type"] = "deposit"
        # Standardize amount column if needed (example: deposits might have 'amount')
        # deposits_df['amount'] = deposits_df['amount'] / (10**6) # Assuming USDC
        records.append(deposits_df)
    if not withdrawals_df.empty:
        withdrawals_df["record_type"] = "withdrawal"
        # withdrawals_df['amount'] = withdrawals_df['amount'] / (10**6)
        records.append(withdrawals_df)
    if not funding_df.empty:
        funding_df["record_type"] = "funding"
        # funding_df['amount'] = funding_df['amount'] / (10**6) # Funding is in quote
        records.append(funding_df)
    # if not liquidations_df.empty:
    #     liquidations_df['record_type'] = 'liquidation'
    #     records.append(liquidations_df)

    if not records:
        st.warning("No records found for the selected user and date range.")
        return

    all_records_df = pd.concat(records, ignore_index=True)
    all_records_df = all_records_df.sort_values("ts").reset_index(drop=True)

    # --- Display Raw Data (Temporary) ---
    st.subheader("Combined User Records (Raw)")
    st.dataframe(all_records_df)

    # --- Call the new processing function ---
    pnl_summary_df, positions_df, balance_df = calculate_performance(
        all_records_df, selected_user_pk
    )

    # st.subheader("Performance Summary")
    # st.dataframe(pnl_summary_df)

    # st.subheader("Positions Over Time (Trades Only)")
    # st.dataframe(positions_df) # Needs filtering/pivoting for plotting per market

    # st.subheader("USDC Balance Over Time")
    # st.dataframe(balance_df)

    # --- Call the new plotting function ---
    plot_performance_summary(pnl_summary_df, positions_df, balance_df, selected_user_pk)


async def show_user_perf(clearing_house: DriftClient):
    # print("loading frens")
    # frens = get_loaded_auths()
    # print("frens loaded", frens)

    # st.write('query string:', frens)

    state = await get_state_account(clearing_house.program)
    ch = clearing_house

    # start = timeit.default_timer()
    # every_user_stats = await ch.program.account["UserStats"].all()
    # end = timeit.default_timer()
    # authorities = sorted([str(x.account.authority) for x in every_user_stats])
    authority0, mol0, mol2, market_symbol0 = st.columns([10, 3, 3, 3])

    authority = authority0.text_input(
        "authority", value="GXyE3Snk3pPYX4Nz9QRVBrnBfbJRTAQYxuy5DRdnebAn"
    )
    perp_markets = [m.symbol for m in mainnet_perp_market_configs]
    spot_markets = [m.symbol for m in mainnet_spot_market_configs]

    markets = []
    for perp in perp_markets:
        markets.append(perp)
        base_asset = perp.replace("-PERP", "")
        if base_asset in spot_markets:
            markets.append(base_asset)

    for spot in spot_markets:
        if spot not in markets:
            markets.append(spot)
    market_symbol = market_symbol0.selectbox(
        "market symbol", markets, index=markets.index("SOL-PERP")
    )
    # if len(frens) == 0:
    #     user_authorities = authority0.selectbox(
    #         "user authorities",
    #         authorities,
    #         0,
    #         # on_change=ccc
    #         # frens
    #     )
    #     # user_authorities = ['4FbQvke11D4EdHVsCD3xej2Pncp4LFMTXWJUXv7irxTj']
    #     user_authorities = [user_authorities]
    # else:
    #     user_authorities = authority0.selectbox(
    #         "user authorities",
    #         frens,
    #         0,
    #         # on_change=ccc
    #         # frens
    #     )
    #     user_authorities = frens
    # print(user_authorities)
    # user_authority = user_authorities[0]

    # await chu.set_cache()
    # cache = chu.CACHE

    # user_stats_pk = get_user_stats_account_public_key(ch.program_id, user_authority_pk)
    # all_user_stats = await ch.program.account['UserStats'].fetch(user_stats_pk)
    # user_stats = [
    #     x.account
    #     for x in every_user_stats
    #     if str(x.account.authority) in user_authorities
    # ]
    # user_stats = []
    # all_summarys = []
    # balances = []
    # positions = []
    # url = 'https://drift-historical-data.s3.eu-west-1.amazonaws.com/program/dRiftyHA39MWEi3m9aunc5MzRF1JYuBsbn6VPcn33UH/'
    # url += 'user/%s/trades/%s/%s'
    # userAccountKeys = []
    # user_stat = None
    # for user_authority in user_authorities:
    #     user_authority_pk = Pubkey.from_string(user_authority)
    #     user_stats_pk = get_user_stats_account_public_key(ch.program_id, user_authority_pk)
    #     user_stat = await ch.program.account['UserStats'].fetch(user_stats_pk)
    #     for sub_id in range(user_stat.number_of_sub_accounts_created):
    #         user_account_pk = get_user_account_public_key(
    #             clearing_house.program_id, user_authority_pk, sub_id
    #         )
    #         userAccountKeys.append(user_account_pk)
    # st.write(
    #     "Authority owned Drift User Accounts:",
    # )
    # uak_df = pd.DataFrame(userAccountKeys, columns=["userAccountPublicKey"])
    # uak_df.index = ["subaccount_" + str(x) for x in uak_df.index]
    # st.dataframe(uak_df.T)

    lastest_date = pd.to_datetime(datetime.datetime.now(), utc=True)
    start_date = mol0.date_input(
        "start date:",
        lastest_date - datetime.timedelta(days=1),
        min_value=datetime.datetime(2022, 11, 4),
        max_value=lastest_date,
    )  # (datetime.datetime.now(tzInfo)))
    end_date = mol2.date_input(
        "end date:",
        lastest_date,
        min_value=datetime.datetime(2022, 11, 4),
        max_value=lastest_date,
    )  # (datetime.datetime.now(tzInfo)))
    # dates = pd.date_range(start_date, end_date)

    await show_user_perf_for_authority(
        clearing_house, authority, market_symbol, start_date, end_date
    )
    # for user_authority in user_authorities:
    # st.markdown('user stats')
    # st.dataframe(pd.DataFrame([x for x in user_stats]).T)

    # else:
    #     st.text('not found')

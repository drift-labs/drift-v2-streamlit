import datetime

import numpy as np
import pandas as pd
import pytz
import streamlit as st
from driftpy.constants.numeric_constants import QUOTE_PRECISION
from driftpy.constants.perp_markets import PerpMarketConfig, mainnet_perp_market_configs
from driftpy.constants.spot_markets import SpotMarketConfig, mainnet_spot_market_configs
from driftpy.drift_client import DriftClient
from driftpy.types import MarketType
from solders.pubkey import Pubkey

from datafetch.api_fetch import get_user_trades_for_month
from tabs.tradeflow import dedupdf


async def counterparty_analysis_page(drift_client: DriftClient):
    st.title("Trade Counterparty Analysis")

    tzInfo = pytz.timezone("UTC")

    # Input fields
    col_input1, col_input2, col_input3 = st.columns(3)
    user_account_key_str = col_input1.text_input(
        "Enter User Account Public Key:",
        value="BM5Kvgpz2XLez2XhfNNeZaXWtHH6NASYw1P74NwEB4sL",
        help="The public key of the Drift user account to analyze.",
    )
    try:
        user_account_key = (
            Pubkey.from_string(user_account_key_str) if user_account_key_str else None
        )
    except ValueError:
        st.error("Invalid Public Key format.")
        user_account_key = None
        return

    perp_markets = [m.symbol for m in mainnet_perp_market_configs]
    spot_markets = [m.symbol for m in mainnet_spot_market_configs]
    all_markets = sorted(perp_markets + spot_markets)
    default_market = "ETH-PERP"
    default_market_index = (
        all_markets.index(default_market) if default_market in all_markets else 0
    )

    market_selection_type = col_input2.radio(
        "Market Selection:", ["All Markets", "Select Market"], index=1
    )

    selected_markets = []
    if market_selection_type == "Select Market":
        market = st.selectbox("Select Market:", all_markets, index=default_market_index)
        if market:
            selected_markets = [market]
    else:
        selected_markets = all_markets

    date_range = col_input3.date_input(
        "Select Date Range:",
        value=(
            datetime.datetime.now(tzInfo) - datetime.timedelta(days=4),
            datetime.datetime.now(tzInfo),
        ),
        min_value=datetime.datetime(2022, 11, 4, tzinfo=tzInfo),
        max_value=datetime.datetime.now(tzInfo) + datetime.timedelta(days=1),
        help="Select the start and end date (UTC) for the analysis.",
    )

    if len(date_range) != 2:
        st.warning("Please select a valid date range (start and end date).")
        return

    start_date, end_date = date_range
    start_datetime = datetime.datetime.combine(
        start_date, datetime.time.min, tzinfo=tzInfo
    )
    end_datetime = datetime.datetime.combine(end_date, datetime.time.max, tzinfo=tzInfo)

    if not user_account_key:
        st.info("Please enter a User Account Public Key to begin analysis.")
        return

    if not selected_markets:
        st.info("Please select at least one market or 'All Markets'.")
        return

    st.write(f"Analyzing trades for user: `{user_account_key}`")
    st.write(f"Date Range: `{start_datetime}` to `{end_datetime}`")
    st.write(f"Markets: `{', '.join(selected_markets)}`")

    all_trades_data = {}
    user_trades_all_markets_df = pd.DataFrame()

    progress_bar = st.progress(0)
    status_text = st.empty()

    if user_account_key_str:
        status_text.text(
            f"Fetching all trades for user {user_account_key_str} via user API endpoint..."
        )
        start_year = start_datetime.year
        start_month = start_datetime.month
        end_year = end_datetime.year
        end_month = end_datetime.month

        current_year = start_year
        current_month = start_month
        user_monthly_trades = []

        total_months = (end_year - start_year) * 12 + (end_month - start_month) + 1
        month_count = 0

        while (current_year < end_year) or (
            current_year == end_year and current_month <= end_month
        ):
            month_count += 1
            status_text.text(
                f"Fetching user trades for {current_year}-{current_month:02d}... ({month_count}/{total_months})"
            )
            try:
                monthly_data = get_user_trades_for_month(
                    user_account_key_str, current_year, current_month
                )
                if monthly_data is not None and not monthly_data.empty:
                    if "ts" in monthly_data.columns:
                        if not pd.api.types.is_datetime64_any_dtype(monthly_data["ts"]):
                            monthly_data["ts"] = pd.to_datetime(
                                monthly_data["ts"], unit="s", utc=True
                            )
                        elif monthly_data["ts"].dt.tz is None:
                            monthly_data["ts"] = monthly_data["ts"].dt.tz_localize(
                                "UTC"
                            )
                        else:
                            monthly_data["ts"] = monthly_data["ts"].dt.tz_convert("UTC")
                    user_monthly_trades.append(monthly_data)
                # Avoid printing success messages here to prevent clutter
            except Exception as e:
                st.error(
                    f"Error fetching user trades for {current_year}-{current_month:02d}: {e}"
                )
                # Optionally break or continue based on error tolerance

            if total_months > 0:
                progress_bar.progress(month_count / total_months)

            current_month += 1
            if current_month > 12:
                current_month = 1
                current_year += 1

        if user_monthly_trades:
            user_trades_all_markets_df = pd.concat(
                user_monthly_trades, ignore_index=True
            )
            user_trades_all_markets_df = user_trades_all_markets_df[
                (user_trades_all_markets_df["ts"] >= start_datetime)
                & (user_trades_all_markets_df["ts"] <= end_datetime)
            ]
            print(
                f"Fetched {len(user_trades_all_markets_df)} total user trades in date range via user endpoint."
            )
            if user_trades_all_markets_df.empty:
                st.warning(
                    "User trades fetched, but none fall within the specified start/end datetimes."
                )
        else:
            print("No user trades fetched via user endpoint.")
        status_text.text("User trade fetching complete. Filtering selected markets...")
        progress_bar.empty()
        progress_bar = st.progress(0)
    else:
        st.error("User account key is required.")
        return

    market_configs = mainnet_perp_market_configs + mainnet_spot_market_configs
    for i, market_symbol in enumerate(selected_markets):
        market_status_text = (
            f"Filtering for {market_symbol}... ({i + 1}/{len(selected_markets)})"
        )
        status_text.text(market_status_text)
        market_data = pd.DataFrame()

        try:
            if not user_trades_all_markets_df.empty:
                market_config: PerpMarketConfig | SpotMarketConfig | None = next(
                    (m for m in market_configs if m.symbol == market_symbol), None
                )
                if market_config:
                    market_index = market_config.market_index
                    if isinstance(market_config, PerpMarketConfig):
                        market_type_str = "perp"
                    elif isinstance(market_config, SpotMarketConfig):
                        market_type_str = "spot"
                    else:
                        st.warning(f"Unknown market config type for {market_symbol}")
                        market_type_str = None

                    if market_type_str:
                        market_data = user_trades_all_markets_df[
                            (user_trades_all_markets_df["marketIndex"] == market_index)
                            & (
                                user_trades_all_markets_df["marketType"]
                                .astype(str)
                                .str.lower()
                                == market_type_str
                            )
                        ].copy()
                        print(
                            f"Filtered {len(market_data)} user trades for {market_symbol} (Index: {market_index}, Type: {market_type_str})"
                        )
                    else:
                        market_data = pd.DataFrame()

                else:
                    st.warning(
                        f"Market configuration not found for {market_symbol}. Cannot filter user trades."
                    )
            else:
                print(
                    f"Skipping market {market_symbol} as no user trades were pre-fetched."
                )

            if not market_data.empty:
                if "marketSymbol" not in market_data.columns:
                    market_data["marketSymbol"] = market_symbol
                all_trades_data[market_symbol] = market_data

        except Exception as e:
            st.error(f"Error processing data for {market_symbol}: {e}")
            print(f"Error processing {market_symbol}: {e}")
        progress_bar.progress((i + 1) / len(selected_markets))

    status_text.text("Data filtering complete.")
    progress_bar.empty()

    if not all_trades_data:
        st.warning(
            "No trade data found for the selected user, markets, and date range after processing."
        )
        return

    combined_df = pd.concat(all_trades_data.values(), ignore_index=True)

    if combined_df.empty:
        st.warning("No trade data found after combining markets.")
        return

    numeric_cols = [
        "fillerReward",
        "baseAssetAmountFilled",
        "quoteAssetAmountFilled",
        "quoteAssetAmountSurplus",
        "takerOrderBaseAssetAmount",
        "takerOrderCumulativeBaseAssetAmountFilled",
        "takerOrderCumulativeQuoteAssetAmountFilled",
        "makerOrderBaseAssetAmount",
        "makerOrderCumulativeBaseAssetAmountFilled",
        "makerOrderCumulativeQuoteAssetAmountFilled",
        "oraclePrice",
        "makerFee",
        "takerFee",
        "markPrice",
    ]
    st.write("Converting numeric columns...")
    for col in numeric_cols:
        if col in combined_df.columns:
            combined_df[col] = pd.to_numeric(combined_df[col], errors="coerce")

    essential_numeric = [
        "baseAssetAmountFilled",
        "quoteAssetAmountFilled",
        "oraclePrice",
    ]
    combined_df.dropna(subset=essential_numeric, inplace=True)
    if combined_df.empty:
        st.warning("No valid trade data remaining after handling numeric conversions.")
        return

    if "ts" not in combined_df.columns or not pd.api.types.is_datetime64_any_dtype(
        combined_df["ts"]
    ):
        st.error(
            "Timestamp column 'ts' is missing or not in datetime format after combining data."
        )
        return
    if combined_df["ts"].dt.tz is None:
        combined_df["ts"] = combined_df["ts"].dt.tz_localize("UTC")
    else:
        combined_df["ts"] = combined_df["ts"].dt.tz_convert("UTC")

    st.write(
        f"Filtering {len(combined_df)} combined trades for user {user_account_key}..."
    )
    user_trades = combined_df[
        (combined_df["taker"] == str(user_account_key))
        | (combined_df["maker"] == str(user_account_key))
    ].copy()

    if user_trades.empty:
        st.warning(
            f"No trades found involving user `{user_account_key}` in the final combined dataset."
        )
        return

    st.subheader(f"Trades Involving User: {user_account_key}")
    st.write(
        f"Found {len(user_trades)} trades involving the user across selected markets."
    )

    user_trades["role"] = "unknown"
    user_trades.loc[user_trades["taker"] == str(user_account_key), "role"] = "taker"
    user_trades.loc[user_trades["maker"] == str(user_account_key), "role"] = "maker"

    user_trades["counterparty"] = ""
    user_trades.loc[user_trades["role"] == "taker", "counterparty"] = user_trades[
        "maker"
    ]
    user_trades.loc[user_trades["role"] == "maker", "counterparty"] = user_trades[
        "taker"
    ]

    user_trades["counterparty"] = user_trades["counterparty"].fillna("vAMM")

    st.dataframe(
        user_trades[
            [
                "ts",
                "marketSymbol",
                "marketType",
                "role",
                "counterparty",
                "baseAssetAmountFilled",
                "quoteAssetAmountFilled",
                "oraclePrice",
                "takerFee",
                "makerFee",
            ]
        ]
    )

    st.subheader("Performance Analysis")
    lookahead = st.slider(
        "Markout Lookahead (seconds):",
        0,
        3600,
        60,
        10,
        help="Seconds lookahead to check PnL after the trade execution.",
    )

    results_by_market = {}
    processed_trades_list = []

    st.write("Calculating markouts per market...")
    analysis_progress = st.progress(0)
    analysis_status = st.empty()

    unique_markets_in_data = user_trades["marketSymbol"].unique()

    for idx, market_sym in enumerate(unique_markets_in_data):
        analysis_status.text(
            f"Analyzing markouts for {market_sym}... ({idx + 1}/{len(unique_markets_in_data)})"
        )
        market_trades = user_trades[user_trades["marketSymbol"] == market_sym].copy()

        if market_trades.empty:
            analysis_progress.progress((idx + 1) / len(unique_markets_in_data))
            continue

        required_cols = [
            "ts",
            "oraclePrice",
            "quoteAssetAmountFilled",
            "baseAssetAmountFilled",
            "takerOrderDirection",
            "role",
        ]
        if not all(col in market_trades.columns for col in required_cols):
            st.warning(
                f"Skipping markout for {market_sym}: Missing required columns ({required_cols})."
            )
            analysis_progress.progress((idx + 1) / len(unique_markets_in_data))
            continue
        if not all(
            pd.api.types.is_numeric_dtype(market_trades[col])
            for col in [
                "oraclePrice",
                "quoteAssetAmountFilled",
                "baseAssetAmountFilled",
            ]
        ):
            st.warning(
                f"Skipping markout for {market_sym}: Essential columns are not numeric."
            )
            analysis_progress.progress((idx + 1) / len(unique_markets_in_data))
            continue

        market_trades = market_trades.sort_values("ts")
        market_trades = market_trades.reset_index(drop=True)

        if (
            "ts" not in market_trades.columns
            or not pd.api.types.is_datetime64_any_dtype(market_trades["ts"])
        ):
            st.error(f"Timestamp issue for {market_sym}, cannot create oracle series.")
            analysis_progress.progress((idx + 1) / len(unique_markets_in_data))
            continue

        try:
            oracle_series = (
                user_trades[user_trades["marketSymbol"] == market_sym]
                .set_index("ts")["oraclePrice"]
                .sort_index()
            )
            oracle_series = oracle_series.dropna()
            if oracle_series.index.has_duplicates:
                oracle_series = oracle_series.groupby(oracle_series.index).mean()

        except Exception as e:
            st.error(f"Error creating oracle price series for {market_sym}: {e}")
            analysis_progress.progress((idx + 1) / len(unique_markets_in_data))
            continue

        market_trades["markPrice"] = np.where(
            market_trades["baseAssetAmountFilled"] != 0,
            market_trades["quoteAssetAmountFilled"]
            / market_trades["baseAssetAmountFilled"],
            np.nan,
        )
        market_trades.dropna(subset=["markPrice"], inplace=True)

        market_trades["user_direction"] = "unknown"
        market_trades.loc[
            (market_trades["role"] == "taker")
            & (market_trades["takerOrderDirection"] == "long"),
            "user_direction",
        ] = "long"
        market_trades.loc[
            (market_trades["role"] == "taker")
            & (market_trades["takerOrderDirection"] == "short"),
            "user_direction",
        ] = "short"
        market_trades.loc[
            (market_trades["role"] == "maker")
            & (market_trades["takerOrderDirection"] == "long"),
            "user_direction",
        ] = "short"
        market_trades.loc[
            (market_trades["role"] == "maker")
            & (market_trades["takerOrderDirection"] == "short"),
            "user_direction",
        ] = "long"

        lookahead_delta = pd.Timedelta(seconds=lookahead)
        future_timestamps = market_trades["ts"] + lookahead_delta

        oracle_lookup_values = oracle_series.reindex(
            future_timestamps, method="ffill"
        ).values
        market_trades["oraclePrice_lookahead"] = oracle_lookup_values

        direction_mult = np.where(market_trades["user_direction"] == "long", 1, -1)

        market_trades["markout_pnl"] = (
            (market_trades["oraclePrice_lookahead"] - market_trades["markPrice"])
            * market_trades["baseAssetAmountFilled"].abs()
            * direction_mult
        )

        market_trades["execution_pnl"] = (
            (market_trades["markPrice"] - market_trades["oraclePrice"])
            * market_trades["baseAssetAmountFilled"].abs()
            * -direction_mult
        )

        market_trades["takerFeeNum"] = pd.to_numeric(
            market_trades["takerFee"], errors="coerce"
        ).fillna(0)
        market_trades["makerFeeNum"] = pd.to_numeric(
            market_trades["makerFee"], errors="coerce"
        ).fillna(0)
        market_trades["trade_fee"] = 0.0
        market_trades.loc[market_trades["role"] == "taker", "trade_fee"] = (
            market_trades["takerFeeNum"] / QUOTE_PRECISION
        )
        market_trades.loc[market_trades["role"] == "maker", "trade_fee"] = (
            market_trades["makerFeeNum"] / QUOTE_PRECISION
        )

        market_trades["net_markout_pnl"] = (
            market_trades["markout_pnl"].fillna(0) - market_trades["trade_fee"]
        )
        market_trades["net_execution_pnl"] = (
            market_trades["execution_pnl"].fillna(0) - market_trades["trade_fee"]
        )

        processed_trades_list.append(market_trades)
        analysis_progress.progress((idx + 1) / len(unique_markets_in_data))

    analysis_status.text("Markout calculations complete.")
    analysis_progress.empty()

    if not processed_trades_list:
        st.warning("No trades could be processed for markout analysis.")
        with st.expander("Show User Trade Data (Before Markout Attempt)"):
            st.dataframe(user_trades)
        return

    user_trades_processed = pd.concat(processed_trades_list, ignore_index=True)

    st.dataframe(
        user_trades_processed[
            [
                "ts",
                "marketSymbol",
                "role",
                "counterparty",
                "user_direction",
                "markPrice",
                "oraclePrice",
                "oraclePrice_lookahead",
                "execution_pnl",
                "markout_pnl",
                "trade_fee",
                "net_markout_pnl",
                "net_execution_pnl",
            ]
        ]
    )

    st.subheader("Analysis by Counterparty")

    # Keep the original counterparty string for linking later
    user_trades_processed["counterparty_raw"] = user_trades_processed["counterparty"]

    counterparty_summary = (
        user_trades_processed.groupby("counterparty_raw")  # Group by raw pubkey
        .agg(
            total_trades=("ts", "count"),
            taker_trades=("role", lambda x: (x == "taker").sum()),
            maker_trades=("role", lambda x: (x == "maker").sum()),
            total_volume_usd=(
                "quoteAssetAmountFilled",
                lambda x: x.abs().sum() / QUOTE_PRECISION,
            ),
            total_execution_pnl=("execution_pnl", "sum"),
            avg_execution_pnl_per_trade=("execution_pnl", "mean"),
            total_markout_pnl=("markout_pnl", "sum"),
            avg_markout_pnl_per_trade=("markout_pnl", "mean"),
            total_fees=("trade_fee", "sum"),
            total_net_markout_pnl=("net_markout_pnl", "sum"),
            total_net_execution_pnl=("net_execution_pnl", "sum"),
        )
        .sort_values("total_volume_usd", ascending=False)
        .reset_index()  # Bring counterparty_raw into columns
    )

    def make_clickable(pubkey_str):
        if pubkey_str == "vAMM":
            return "vAMM"
        try:
            if (
                isinstance(pubkey_str, str)
                and len(pubkey_str) > 30
                and len(pubkey_str) < 50
                and pubkey_str.isalnum()
            ):
                return f"{pubkey_str}"
            else:
                return f"{pubkey_str} (Unknown Format)"
        except Exception:
            return f"{pubkey_str} (Error)"

    # Create a display column with links
    counterparty_summary["Counterparty"] = counterparty_summary[
        "counterparty_raw"
    ].apply(make_clickable)

    st.write(f"Analyzed {len(counterparty_summary)} unique counterparties.")

    # Define columns to display and their order
    display_columns = [
        "Counterparty",
        "total_trades",
        "taker_trades",
        "maker_trades",
        "total_volume_usd",
        "total_execution_pnl",
        "avg_execution_pnl_per_trade",
        "total_markout_pnl",
        "avg_markout_pnl_per_trade",
        "total_fees",
        "total_net_markout_pnl",
        "total_net_execution_pnl",
        # 'counterparty_raw' # Optionally keep this if needed
    ]

    # Use st.dataframe for interactivity
    st.dataframe(
        counterparty_summary[display_columns],
        column_config={
            "Counterparty": st.column_config.TextColumn(
                "Counterparty", help="Counterparty (vAMM or User Account)"
            ),
            "total_volume_usd": st.column_config.NumberColumn(
                "Total Volume ($)", format="$%.2f"
            ),
            "total_execution_pnl": st.column_config.NumberColumn(
                "Total Exec PnL ($)", format="$%.2f"
            ),
            "avg_execution_pnl_per_trade": st.column_config.NumberColumn(
                "Avg Exec PnL ($)", format="$%.4f"
            ),
            "total_markout_pnl": st.column_config.NumberColumn(
                "Total Markout PnL ($)", format="$%.2f"
            ),
            "avg_markout_pnl_per_trade": st.column_config.NumberColumn(
                "Avg Markout PnL ($)", format="$%.4f"
            ),
            "total_fees": st.column_config.NumberColumn(
                "Total Fees ($)", format="$%.2f"
            ),
            "total_net_markout_pnl": st.column_config.NumberColumn(
                "Net Markout PnL ($)", format="$%.2f"
            ),
            "total_net_execution_pnl": st.column_config.NumberColumn(
                "Net Exec PnL ($)", format="$%.2f"
            ),
        },
        hide_index=True,
    )

    with st.expander("Show Processed User Trade Data (with Markouts)"):
        st.dataframe(user_trades_processed)

    with st.expander("Show Raw Combined Fetched Data (Before User Filter)"):
        st.dataframe(combined_df)

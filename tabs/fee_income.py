import asyncio
import datetime
from datetime import timedelta

import httpx
import pandas as pd
import streamlit as st
from driftpy.constants.perp_markets import mainnet_perp_market_configs
from driftpy.drift_client import DriftClient

pd.options.plotting.backend = "plotly"

URL_PREFIX = "https://data.api.drift.trade"


async def fetch_day_trades(
    client: httpx.AsyncClient, market_symbol: str, date: datetime.date, page: int = 1
) -> pd.DataFrame:
    """Asynchronously fetches all trades for a specific market and day, handling pagination."""
    all_records = []
    year = date.year
    month = date.month
    day = date.day
    url = f"{URL_PREFIX}/market/{market_symbol}/trades/{year}/{month:02}/{day:02}"

    try:
        current_page = page
        while True:
            response = await client.get(url, params={"page": current_page})
            response.raise_for_status()
            json_data = response.json()
            meta = json_data["meta"]
            records = json_data.get("records", [])
            if not records:  # Stop if no records are returned
                break
            all_records.extend(records)

            if meta.get("nextPage") is None:
                break  # Exit loop if no nextPage key or value is None
            current_page = meta["nextPage"]
            print(
                f"Fetching page {current_page} for {market_symbol} on {date}"
            )  # Optional: keep for debug
            await asyncio.sleep(0.1)  # Be nice to the API

        if all_records:
            return pd.DataFrame(all_records)
        else:
            return pd.DataFrame()  # Return empty DataFrame if no records found

    except httpx.RequestError as e:
        print(f"Error fetching data for {date}: {e}")
        return pd.DataFrame()  # Return empty DataFrame on error
    except Exception as e:  # Catch potential JSON errors or other issues
        print(f"Unexpected error for {date}: {e}")
        return pd.DataFrame()


async def get_trades_for_range_pandas(
    market_symbol, start_date, end_date, status_container=None
) -> pd.DataFrame:
    """Fetches trades asynchronously for a date range."""
    all_trades_list = []
    date_list = [
        start_date + timedelta(days=x) for x in range((end_date - start_date).days + 1)
    ]
    total_days = len(date_list)

    async with httpx.AsyncClient(timeout=30.0) as client:
        tasks = []
        for date in date_list:
            tasks.append(fetch_day_trades(client, market_symbol, date))

        for i, future in enumerate(asyncio.as_completed(tasks)):
            if status_container:
                progress = (i + 1) / total_days * 100
                status_container.update(
                    label=f"Fetching trades for {market_symbol}... ({i + 1}/{total_days} days processed, {progress:.1f}%)"
                )

            try:
                daily_df = await future
                if not daily_df.empty:
                    all_trades_list.append(daily_df)
            except Exception as e:
                print(f"Error processing future for a day: {e}")

    if not all_trades_list:
        return pd.DataFrame()  # Return empty if all requests failed or yielded no data

    df = pd.concat(all_trades_list, ignore_index=True)
    if df.empty:
        return df

    if "ts" in df.columns:
        df["ts"] = pd.to_datetime(df["ts"], unit="s")
        df = df.sort_values(by="ts").reset_index(drop=True)
        df = add_fee_ratio(df)
    else:
        print("Warning: 'ts' column not found in combined trade data.")
        return df

    return df


def add_fee_ratio(df):
    """
    Analyzes trade fee ratios accounting for both base and quote asset amounts
    """
    df["quoteAssetAmountFilled"] = pd.to_numeric(df["quoteAssetAmountFilled"])
    df["baseAssetAmountFilled"] = pd.to_numeric(df["baseAssetAmountFilled"])
    df["takerFee"] = pd.to_numeric(df["takerFee"])

    df["price"] = df["quoteAssetAmountFilled"] / df["baseAssetAmountFilled"]
    df["trade_value"] = abs(df["baseAssetAmountFilled"] * df["price"])
    df["fee_ratio"] = abs(df["takerFee"] / df["trade_value"])
    return df


def get_fee_tier_trades(trades):
    """
    Clusters trades into fee ratio groups:
    - High leverage: > 2.5 bps (0.00025)
    - Tier 1: = 2.5 bps
    - Tier 2-4: 0.75 bps < x < 2.5 bps
    - VIP: = 0.75 bps (0.0000075)
    """
    high_leverage = trades[
        (trades["fee_ratio"] > 0.000251)
        & (trades["actionExplanation"] != "liquidation")
    ]

    tier_1 = trades[
        (trades["fee_ratio"] == 0.00025)
        | (trades["fee_ratio"] == 0.000251)
        & (trades["actionExplanation"] != "liquidation")
    ]

    tier_2_4 = trades[
        (trades["fee_ratio"] > 0.0000751)
        & (trades["fee_ratio"] < 0.000249)
        & (trades["actionExplanation"] != "liquidation")
    ]

    vip = trades[
        (trades["fee_ratio"] < 0.0000751)
        & (trades["fee_ratio"] > 0.000071)
        & (trades["actionExplanation"] != "liquidation")
    ]

    other_small = trades[
        (trades["fee_ratio"] < 0.000071)
        & (trades["actionExplanation"] != "liquidation")
    ]

    return {
        "high_leverage": high_leverage,
        "tier_1": tier_1,
        "tier_2_4": tier_2_4,
        "vip": vip,
        "other_small": other_small,
    }


def summarize_trading_data(trades):
    summary = {
        "total_volume_base": abs(trades["baseAssetAmountFilled"]).sum(),
        "total_volume_quote": abs(trades["quoteAssetAmountFilled"]).sum(),
        "total_fees": abs(trades["takerFee"]).sum(),
        "trade_count": len(trades),
        "avg_price": (
            trades["quoteAssetAmountFilled"] / trades["baseAssetAmountFilled"]
        ).mean(),
    }
    summary["volume_base_pct"] = (
        summary["total_volume_base"] / abs(trades["baseAssetAmountFilled"]).sum() * 100
    )
    summary["volume_quote_pct"] = (
        summary["total_volume_quote"]
        / abs(trades["quoteAssetAmountFilled"]).sum()
        * 100
    )

    return summary


def from_summary(summary):
    st.metric("Total Volume Base", f"{summary['total_volume_base']:,.2f}")
    st.metric("Total Volume Quote", f"{summary['total_volume_quote']:,.2f}")
    st.metric("Taker Fees", f"{summary['total_fees']:,.2f}")


def display_fee_tier_metrics(fee_tier_data, tier_name):
    st.metric(tier_name, len(fee_tier_data))
    summary = summarize_trading_data(fee_tier_data)
    from_summary(summary)


async def fee_income_page(ch: DriftClient):
    market = st.selectbox(
        "Select market",
        options=[
            {"market_index": c.market_index, "symbol": c.symbol}
            for c in mainnet_perp_market_configs
        ],
        index=0,
        format_func=lambda x: f"({x['market_index']}) {x['symbol']}",
    )

    today = datetime.datetime.now()
    date_range = st.date_input(
        "Select date range",
        (today, today),
        max_value=today,
        format="MM.DD.YYYY",
        help="For a single day, select the same date for both the start and end.",
    )

    if date_range and len(date_range) == 2:
        start_date, end_date = date_range
        if start_date > end_date:
            st.warning("Start date cannot be after end date.")
            st.stop()

        with st.status(
            f"Fetching trade data for {market['symbol']} ({start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')})...",
            expanded=True,
        ) as status:
            trades = await get_trades_for_range_pandas(
                market["symbol"], start_date, end_date, status_container=status
            )
            status.update(label="Trade data fetched!", state="complete", expanded=False)

        if trades.empty:
            st.warning(
                f"No trade data found for {market['symbol']} between {start_date.strftime('%Y-%m-%d')} and {end_date.strftime('%Y-%m-%d')}."
            )
            st.stop()  # Stop execution if no data

        fee_tiers = get_fee_tier_trades(trades)

        tier_configs = [
            ("High Leverage Trades (> 2.5 bps)", "high_leverage"),
            ("Tier 1 Trades (2.5 bps)", "tier_1"),
            ("Tier 2-4 Trades (0.75 bps < x < 2.5 bps)", "tier_2_4"),
            ("VIP Trades (0.75 bps)", "vip"),
        ]
        cols = st.columns(4)
        for col, (tier_name, tier_key) in zip(cols, tier_configs):
            with col:
                display_fee_tier_metrics(fee_tiers[tier_key], tier_name)
    else:
        st.info("Please select a valid date range.")
        st.stop()

    with st.expander("Show Fee Tier Trades"):
        tab1, tab2, tab3, tab4, tab5 = st.tabs(
            ["High Leverage", "Tier 1", "Tier 2-4", "VIP", "Other Small"]
        )

        with tab1:
            st.dataframe(
                fee_tiers["high_leverage"],
                column_config={
                    "fee_ratio": st.column_config.NumberColumn(format="%.6f")
                },
            )
        with tab2:
            st.dataframe(
                fee_tiers["tier_1"],
                column_config={
                    "fee_ratio": st.column_config.NumberColumn(format="%.6f")
                },
            )
        with tab3:
            st.dataframe(
                fee_tiers["tier_2_4"],
                column_config={
                    "fee_ratio": st.column_config.NumberColumn(format="%.6f")
                },
            )
        with tab4:
            st.dataframe(
                fee_tiers["vip"],
                column_config={
                    "fee_ratio": st.column_config.NumberColumn(format="%.7f")
                },
            )
        with tab5:
            st.dataframe(
                fee_tiers["other_small"],
                column_config={
                    "fee_ratio": st.column_config.NumberColumn(format="%.7f")
                },
            )

    with st.expander("Show takers per fee tier"):
        tab1, tab2, tab3, tab4, tab5 = st.tabs(
            ["High Leverage", "Tier 1", "Tier 2-4", "VIP", "Other Small"]
        )

        with tab1:
            st.write(
                f"There are {len(fee_tiers['high_leverage']['taker'].unique())} unique high leverage takers"
            )
            st.dataframe(fee_tiers["high_leverage"]["taker"].value_counts())
        with tab2:
            st.write(
                f"There are {len(fee_tiers['tier_1']['taker'].unique())} unique tier 1 takers"
            )
            st.dataframe(fee_tiers["tier_1"]["taker"].value_counts())
        with tab3:
            st.write(
                f"There are {len(fee_tiers['tier_2_4']['taker'].unique())} unique tier 2-4 takers"
            )
            st.dataframe(fee_tiers["tier_2_4"]["taker"].value_counts())
        with tab4:
            st.write(
                f"There are {len(fee_tiers['vip']['taker'].unique())} unique VIP takers"
            )
            st.dataframe(fee_tiers["vip"]["taker"].value_counts())
        with tab5:
            st.write(
                f"There are {len(fee_tiers['other_small']['taker'].unique())} unique other small takers"
            )
            st.dataframe(fee_tiers["other_small"]["taker"].value_counts())

    st.write("---")
    cols = st.columns(3)
    with cols[0]:
        st.metric("All trades", len(trades))
        from_summary(summarize_trading_data(trades))
    with cols[1]:
        trades_just_liquidations = trades[trades["actionExplanation"] == "liquidation"]
        st.metric("Liquidations only", len(trades_just_liquidations))
        from_summary(summarize_trading_data(trades_just_liquidations))
    with cols[2]:
        all_trades_minus_liquidations = trades[
            trades["actionExplanation"] != "liquidation"
        ]
        st.metric("All trades minus liquidations", len(all_trades_minus_liquidations))
        from_summary(summarize_trading_data(all_trades_minus_liquidations))

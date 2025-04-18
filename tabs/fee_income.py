import datetime
import time
from datetime import timedelta

import pandas as pd
import requests
import streamlit as st
from driftpy.constants.perp_markets import mainnet_perp_market_configs
from driftpy.drift_client import DriftClient

pd.options.plotting.backend = "plotly"

URL_PREFIX = "https://data.api.drift.trade"


def get_trades_for_range_pandas(market_symbol, start_date, end_date, page=1):
    df = pd.DataFrame()
    all_trades = []
    current_date = start_date
    while current_date <= end_date:
        year = current_date.year
        month = current_date.month
        day = current_date.day
        url = f"{URL_PREFIX}/market/{market_symbol}/trades/{year}/{month:02}/{day:02}"
        try:
            response = requests.get(url, params={"page": page})
            response.raise_for_status()
            json = response.json()
            meta = json["meta"]
            df = pd.DataFrame(json["records"])
            while meta["nextPage"] is not None:
                pg = meta["nextPage"]
                response = requests.get(url, params={"page": pg})
                print("Page", str(pg))
                response.raise_for_status()
                json = response.json()
                df = pd.concat([df, pd.DataFrame(json["records"])], ignore_index=True)
                meta = json["meta"]
            time.sleep(0.1)
            all_trades.append(df)
        except requests.exceptions.RequestException as e:
            print(f"Error fetching data for {current_date}: {e}")
        except pd.errors.EmptyDataError:
            print(f"No data available for {current_date}")

        current_date += timedelta(days=1)

    if all_trades:
        df = pd.concat(all_trades, ignore_index=True)
    else:
        df = pd.DataFrame()

    df["ts"] = pd.to_datetime(df["ts"], unit="s")
    df = add_fee_ratio(df)
    return df


def get_trades_for_day_pandas(market_symbol, day):
    return get_trades_for_range_pandas(market_symbol, day, day)


def add_fee_ratio(df):
    """
    Analyzes trade fee ratios accounting for both base and quote asset amounts
    """
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
        mainnet_perp_market_configs,
        index=0,
        format_func=lambda x: f"({x.market_index}) {x.symbol}",
    )

    if "range_check" not in st.session_state:
        st.session_state["range_check"] = False
    range_check = st.session_state["range_check"]
    day = st.date_input(
        "Select day of trades to analyze",
        datetime.datetime.now(),
        disabled=range_check,
    )
    date_range = None

    st.checkbox("Select range of time instead?", key="range_check")
    if st.session_state["range_check"]:
        today = datetime.datetime.now()
        last_week = today - datetime.timedelta(days=7)
        date_range = st.date_input(
            "Select range of trades to analyze",
            (last_week, today),
            max_value=today,
            format="MM.DD.YYYY",
        )

    if st.session_state["range_check"]:
        trades = get_trades_for_range_pandas(
            market.symbol, date_range[0], date_range[1]
        )
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
        trades = get_trades_for_day_pandas(market.symbol, day)
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

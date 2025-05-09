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


def get_fee_tier_trades(trades, market_symbol: str):
    """
    Clusters trades into fee ratio groups based on specified fee ratio ranges,
    adjusting boundaries for specific markets.
    Tiers (standard, before market adjustment):
    - High Leverage: fr > 0.0010 (10 BPS)
    - Tier 1: 0.0009 < fr <= 0.0010 (9-10 BPS)
    - Tier 2: 0.0008 < fr <= 0.0009 (8-9 BPS)
    - Tier 3: 0.0007 < fr <= 0.0008 (7-8 BPS)
    - Tier 4: 0.0006 < fr <= 0.0007 (6-7 BPS)
    - Tier 5: 0.0003 < fr <= 0.0006 (3-6 BPS)
    - VIP:    0 < fr <= 0.0003 (0-3 BPS)
    Returns a tuple: (tier_frames, histogram_bounds_bps, tier_display_configs_data, tab_titles_data).
    """
    non_liq_trades = trades[trades["actionExplanation"] != "liquidation"].copy()

    all_keys = [
        "high_leverage",
        "tier_1",
        "tier_2",
        "tier_3",
        "tier_4",
        "tier_5",
        "vip",
        "other",
    ]
    tier_frames = {
        key: pd.DataFrame(columns=non_liq_trades.columns) for key in all_keys
    }
    standard_bounds = {
        "tier1_upper": 0.0010 + 0.00005,
        "tier2_upper": 0.0009 + 0.00005,
        "tier3_upper": 0.0008 + 0.00005,
        "tier4_upper": 0.0007 + 0.00005,
        "tier5_upper": 0.0006 + 0.00005,
        "vip_upper": 0.0003 + 0.00005,
    }

    special_markets = ["SOL-PERP", "ETH-PERP", "BTC-PERP"]
    bounds_multiplier = 1.0
    if market_symbol in special_markets:
        bounds_multiplier = 0.25
    current_bounds = {k: v * bounds_multiplier for k, v in standard_bounds.items()}

    b = {k: v * 10000 for k, v in current_bounds.items()}

    histogram_bounds_bps = {
        "HL": b["tier1_upper"],
        "T1": b["tier2_upper"],
        "T2": b["tier3_upper"],
        "T3": b["tier4_upper"],
        "T4": b["tier5_upper"],
        "VIP": b["vip_upper"],
        "Other": 0.09,
    }

    tier_display_configs_data = [
        (f"High Leverage (> {b['tier1_upper']:.2f} BPS)", "high_leverage"),
        (f"Tier 1 ({b['tier2_upper']:.2f} - {b['tier1_upper']:.2f} BPS)", "tier_1"),
        (f"Tier 2 ({b['tier3_upper']:.2f} - {b['tier2_upper']:.2f} BPS)", "tier_2"),
        (f"Tier 3 ({b['tier4_upper']:.2f} - {b['tier3_upper']:.2f} BPS)", "tier_3"),
        (f"Tier 4 ({b['tier5_upper']:.2f} - {b['tier4_upper']:.2f} BPS)", "tier_4"),
        (f"Tier 5 ({b['vip_upper']:.2f} - {b['tier5_upper']:.2f} BPS)", "tier_5"),
        (f"VIP Trades (0 - {b['vip_upper']:.2f} BPS & >0)", "vip"),
        (
            "Other Non-Liquidation Trades",
            "other",
        ),
    ]
    tab_titles_data = [
        f"HL (>{b['tier1_upper']:.2f} BPS)",
        f"T1 ({b['tier2_upper']:.2f}-{b['tier1_upper']:.2f} BPS)",
        f"T2 ({b['tier3_upper']:.2f}-{b['tier2_upper']:.2f} BPS)",
        f"T3 ({b['tier4_upper']:.2f}-{b['tier3_upper']:.2f} BPS)",
        f"T4 ({b['tier5_upper']:.2f}-{b['tier4_upper']:.2f} BPS)",
        f"T5 ({b['vip_upper']:.2f}-{b['tier5_upper']:.2f} BPS)",
        f"VIP (0-{b['vip_upper']:.2f} BPS)",
        "Other",
    ]

    if non_liq_trades.empty:
        return (
            tier_frames,
            histogram_bounds_bps,
            tier_display_configs_data,
            tab_titles_data,
        )

    remaining_df = non_liq_trades.copy()

    tier_frames["high_leverage"] = remaining_df[
        remaining_df["fee_ratio"] > current_bounds["tier1_upper"]
    ]
    remaining_df = remaining_df[
        ~(remaining_df["fee_ratio"] > current_bounds["tier1_upper"])
    ]

    tier_frames["tier_1"] = remaining_df[
        (remaining_df["fee_ratio"] > current_bounds["tier2_upper"])
        & (remaining_df["fee_ratio"] <= current_bounds["tier1_upper"])
    ]
    remaining_df = remaining_df[
        ~(
            (remaining_df["fee_ratio"] > current_bounds["tier2_upper"])
            & (remaining_df["fee_ratio"] <= current_bounds["tier1_upper"])
        )
    ]

    tier_frames["tier_2"] = remaining_df[
        (remaining_df["fee_ratio"] > current_bounds["tier3_upper"])
        & (remaining_df["fee_ratio"] <= current_bounds["tier2_upper"])
    ]
    remaining_df = remaining_df[
        ~(
            (remaining_df["fee_ratio"] > current_bounds["tier3_upper"])
            & (remaining_df["fee_ratio"] <= current_bounds["tier2_upper"])
        )
    ]

    tier_frames["tier_3"] = remaining_df[
        (remaining_df["fee_ratio"] > current_bounds["tier4_upper"])
        & (remaining_df["fee_ratio"] <= current_bounds["tier3_upper"])
    ]
    remaining_df = remaining_df[
        ~(
            (remaining_df["fee_ratio"] > current_bounds["tier4_upper"])
            & (remaining_df["fee_ratio"] <= current_bounds["tier3_upper"])
        )
    ]

    tier_frames["tier_4"] = remaining_df[
        (remaining_df["fee_ratio"] > current_bounds["tier5_upper"])
        & (remaining_df["fee_ratio"] <= current_bounds["tier4_upper"])
    ]
    remaining_df = remaining_df[
        ~(
            (remaining_df["fee_ratio"] > current_bounds["tier5_upper"])
            & (remaining_df["fee_ratio"] <= current_bounds["tier4_upper"])
        )
    ]

    tier_frames["tier_5"] = remaining_df[
        (remaining_df["fee_ratio"] > current_bounds["vip_upper"])
        & (remaining_df["fee_ratio"] <= current_bounds["tier5_upper"])
    ]
    remaining_df = remaining_df[
        ~(
            (remaining_df["fee_ratio"] > current_bounds["vip_upper"])
            & (remaining_df["fee_ratio"] <= current_bounds["tier5_upper"])
        )
    ]

    tier_frames["vip"] = remaining_df[
        (remaining_df["fee_ratio"] > 0)
        & (remaining_df["fee_ratio"] <= current_bounds["vip_upper"])
    ]
    remaining_df = remaining_df[
        ~(
            (remaining_df["fee_ratio"] > 0)
            & (remaining_df["fee_ratio"] <= current_bounds["vip_upper"])
        )
    ]

    # 'Other' tier captures non-liquidation trades with zero fee_ratio not caught by other tiers
    tier_frames["other"] = remaining_df[remaining_df["fee_ratio"] == 0]

    return tier_frames, histogram_bounds_bps, tier_display_configs_data, tab_titles_data


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
    market_options = [
        {"market_index": c.market_index, "symbol": c.symbol}
        for c in mainnet_perp_market_configs
    ]

    if not market_options:
        st.warning("No markets available for selection.")
        st.stop()

    market = st.selectbox(
        "Select market",
        options=market_options,
        index=0,
        format_func=lambda x: f"({x['market_index']}) {x['symbol']}",
    )

    special_markets = ["SOL-PERP", "ETH-PERP", "BTC-PERP"]
    if market["symbol"] in special_markets:
        st.info(
            f"Note: Fee tiers for {market['symbol']} are adjusted (75% lower than standard)."
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
            st.stop()

        fee_tiers, histogram_bounds_bps, tier_display_configs, tab_titles = (
            get_fee_tier_trades(trades, market["symbol"])
        )

        st.subheader("Fee Ratio Distribution (Excluding Liquidations)")
        non_liquidation_trades = trades[trades["actionExplanation"] != "liquidation"]
        if (
            not non_liquidation_trades.empty
            and "fee_ratio" in non_liquidation_trades.columns
        ):
            bins = 200
            with st.expander("Adjust graph", expanded=False):
                bins = st.slider(
                    "Number of bins", min_value=100, max_value=300, value=200
                )
            try:
                fee_ratio_bps = non_liquidation_trades["fee_ratio"] * 10000
                fig_dist = fee_ratio_bps.plot(
                    kind="hist",
                    title="Distribution of Taker Fee Ratios (BPS)",
                    labels={"value": "Fee Ratio (BPS)", "count": "Number of Trades"},
                    nbins=bins,
                )
                fig_dist.update_layout(yaxis_type="log")

                tier_thresholds_bps = histogram_bounds_bps

                line_colors = [
                    "red",
                    "orange",
                    "gold",
                    "green",
                    "blue",
                    "purple",
                ]
                color_idx = 0
                for label, threshold_bps in tier_thresholds_bps.items():
                    fig_dist.add_vline(
                        x=threshold_bps,
                        line_width=2,
                        line_dash="dash",
                        line_color=line_colors[color_idx % len(line_colors)],
                        annotation_text=f"{label}",
                        annotation_position="top right",
                    )
                    color_idx += 1

                st.plotly_chart(fig_dist, use_container_width=True)
            except Exception as e:
                st.error(f"Could not generate fee ratio distribution plot: {e}")
        elif non_liquidation_trades.empty:
            st.info("No non-liquidation trades to display fee ratio distribution.")
        else:
            st.warning("'fee_ratio' column not found, cannot display distribution.")

        st.write("---")

        with st.expander("Fee Tier Summary Statistics", expanded=True):
            num_metrics = len(tier_display_configs)
            metrics_per_row = 3
            for i in range(0, num_metrics, metrics_per_row):
                num_cols_for_this_row = min(metrics_per_row, num_metrics - i)
                cols = st.columns(num_cols_for_this_row)

                for j in range(num_cols_for_this_row):
                    if (i + j) < num_metrics:
                        tier_name_template, tier_key = tier_display_configs[i + j]
                        with cols[j]:
                            display_fee_tier_metrics(
                                fee_tiers[tier_key], tier_name_template
                            )

        with st.spinner("Preparing detailed trade views..."):
            with st.expander("Show Fee Tier Trades"):
                created_tabs = st.tabs(tab_titles)
                for i, tier_key_data in enumerate(tier_display_configs):
                    tier_key = tier_key_data[1]
                    with created_tabs[i]:
                        st.dataframe(
                            fee_tiers[tier_key],
                            column_config={
                                "fee_ratio": st.column_config.NumberColumn(
                                    format="%.6f"
                                )
                            },
                        )

            with st.expander("Show takers per fee tier"):
                created_tabs_takers = st.tabs(tab_titles)
                for i, tier_key_data in enumerate(tier_display_configs):
                    tier_key = tier_key_data[1]
                    with created_tabs_takers[i]:
                        unique_takers = (
                            fee_tiers[tier_key]["taker"].unique()
                            if not fee_tiers[tier_key].empty
                            else []
                        )
                        st.write(
                            f"There are {len(unique_takers)} unique takers in {tab_titles[i]}"
                        )
                        if not fee_tiers[tier_key].empty:
                            st.dataframe(fee_tiers[tier_key]["taker"].value_counts())
                        else:
                            st.write("No takers in this tier.")

            st.write("---")
            cols_summary = st.columns(3)
            with cols_summary[0]:
                st.metric("All trades", len(trades))
                from_summary(summarize_trading_data(trades))
            with cols_summary[1]:
                trades_just_liquidations = trades[
                    trades["actionExplanation"] == "liquidation"
                ]
                st.metric("Liquidations only", len(trades_just_liquidations))
                from_summary(summarize_trading_data(trades_just_liquidations))
            with cols_summary[2]:
                all_trades_minus_liquidations = trades[
                    trades["actionExplanation"] != "liquidation"
                ]
                st.metric(
                    "All trades minus liquidations", len(all_trades_minus_liquidations)
                )
                from_summary(summarize_trading_data(all_trades_minus_liquidations))

    else:
        st.info("Please select a valid date range.")
        st.stop()

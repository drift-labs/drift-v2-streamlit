import datetime
import logging
from concurrent.futures import ThreadPoolExecutor

import pandas as pd
import plotly.express as px
import pytz
import streamlit as st
from driftpy.constants.perp_markets import mainnet_perp_market_configs
from streamlit.runtime.scriptrunner import add_script_run_ctx

from datafetch.api_fetch import get_trades_for_range_pandas


def get_trades_for_range_pandas_with_toast(market_symbol, start_date, end_date):
    df = get_trades_for_range_pandas(market_symbol, start_date, end_date)
    st.toast(f"Fetched {len(df)} trades for {market_symbol}")
    return df


def get_trades_for_range_multiple_markets(
    selected_markets: list[str], start_date, end_date
):
    # this is to disable an annoying warning in the console that streamlit has when calling functions in threadspool
    # read about it: https://discuss.streamlit.io/t/warning-for-missing-scriptruncontext/83893/15
    for name, logger_item in logging.root.manager.loggerDict.items():
        if "streamlit" in name:
            logger_item.disabled = True

    with st.spinner("Fetching trade data..."):
        markets_data = {}
        api_tasks = []

        for i, market in enumerate(selected_markets):
            api_tasks.append(
                {"market": market, "start_date": start_date, "end_date": end_date}
            )
        with ThreadPoolExecutor(max_workers=8) as executor:
            futures = []
            for task in api_tasks:
                future = executor.submit(
                    get_trades_for_range_pandas_with_toast,
                    task["market"],
                    task["start_date"],
                    task["end_date"],
                )
                add_script_run_ctx(future)
                futures.append((future, task))

        failed_requests = []
        for future, task in futures:
            try:
                raw_records = future.result()
                st.toast(f"Fetched {task['market']} data ✅")
                market = task["market"]
                markets_data[market] = raw_records
            except Exception as e:
                failed_requests.append(f"{task['market']}: {e}")

        if failed_requests:
            st.warning("Some API requests failed:\n" + "\n".join(failed_requests[:5]))
            if len(failed_requests) > 5:
                st.warning(f"... and {len(failed_requests) - 5} more failures")

        for name, logger_item in logging.root.manager.loggerDict.items():
            if "streamlit" in name:
                logger_item.disabled = False

        return markets_data


def vamm_stats_page():
    st.title("vAMM Statistics")
    st.markdown("Analysis of volume filled by the virtual AMM")

    tzInfo = pytz.timezone("UTC")

    col1, col2, col3 = st.columns(3)

    date_mode = col1.radio(
        "Date Selection:",
        ["Single Date", "Date Range"],
        help="Choose between analyzing a single day or a date range",
    )

    if date_mode == "Single Date":
        date = col2.date_input(
            "Select date:",
            min_value=datetime.datetime(2023, 11, 4),
            max_value=datetime.datetime.now(tzInfo),
            value=datetime.datetime.now(tzInfo) - datetime.timedelta(days=1),
            help="UTC timezone",
        )
        start_date = end_date = date
    else:
        start_date = col2.date_input(
            "Start date:",
            value=datetime.datetime.now(tzInfo) - datetime.timedelta(days=7),
            min_value=datetime.datetime(2023, 11, 4),
            max_value=datetime.datetime.now(tzInfo) - datetime.timedelta(days=1),
            help="UTC timezone",
        )
        end_date = col3.date_input(
            "End date:",
            min_value=start_date,
            max_value=datetime.datetime.now(tzInfo),
            value=datetime.datetime.now(tzInfo),
            help="UTC timezone",
        )

    perp_markets = [m.symbol for m in mainnet_perp_market_configs]
    markets = perp_markets

    if date_mode == "Single Date":
        selected_markets = col3.multiselect(
            "Select markets (empty = all):",
            markets,
            default=[],
            help="Leave empty to analyze all markets",
        )
    else:
        selected_markets = st.multiselect(
            "Select markets (empty = all):",
            markets,
            default=[],
            help="Leave empty to analyze all markets",
        )

    if not selected_markets:
        selected_markets = markets

    if st.button("Load Data"):
        if date_mode == "Date Range":
            date_range_days = (end_date - start_date).days + 1
            st.info(
                f"Fetching data for {date_range_days} days from {start_date} to {end_date}"
            )
        else:
            st.info(f"Fetching data for {start_date}")

        markets_data = get_trades_for_range_multiple_markets(
            selected_markets, start_date, end_date
        )

        if markets_data:
            if date_mode == "Date Range":
                st.header(f"vAMM Volume Analysis ({start_date} to {end_date})")
            else:
                st.header(f"vAMM Volume Analysis ({start_date})")

            market_stats = []
            total_volume = 0
            total_vamm_volume = 0

            for market, data in markets_data.items():
                data["quoteAssetAmountFilled"] = pd.to_numeric(
                    data["quoteAssetAmountFilled"], errors="coerce"
                ).fillna(0)

                market_volume = data["quoteAssetAmountFilled"].sum()
                total_volume += market_volume

                vamm_trades = data[
                    ((data["maker"].isnull()) | (data["maker"] == ""))
                    | ((data["taker"].isnull()) | (data["taker"] == ""))
                ]

                vamm_volume = vamm_trades["quoteAssetAmountFilled"].sum()
                total_vamm_volume += vamm_volume

                vamm_percentage = (
                    (vamm_volume / market_volume * 100) if market_volume > 0 else 0
                )

                market_stats.append(
                    {
                        "Market": market,
                        "Total Volume ($)": f"{market_volume:,.2f}",
                        "vAMM Volume ($)": f"{vamm_volume:,.2f}",
                        "vAMM %": f"{vamm_percentage:.2f}%",
                        "Total Trades": len(data),
                        "vAMM Trades": len(vamm_trades),
                    }
                )

            col1, col2, col3 = st.columns(3)
            aggregate_vamm_percentage = (
                (total_vamm_volume / total_volume * 100) if total_volume > 0 else 0
            )

            col1.metric("Total Volume", f"${total_volume:,.2f}")
            col2.metric(
                "vAMM Volume",
                f"${total_vamm_volume:,.2f}",
                f"{aggregate_vamm_percentage:.2f}% of total",
            )
            col3.metric("Markets Analyzed", len(markets_data))

            st.subheader("Per-Market Breakdown")
            stats_df = pd.DataFrame(market_stats)
            st.dataframe(stats_df, use_container_width=True)

            st.subheader("vAMM Role Analysis")

            role_tabs = st.tabs(["vAMM as Maker", "vAMM as Taker", "Combined Analysis"])

            with role_tabs[0]:
                st.write("**vAMM as Maker** (trades where maker field is null/empty)")
                maker_stats = []
                for market, data in markets_data.items():
                    data["quoteAssetAmountFilled"] = pd.to_numeric(
                        data["quoteAssetAmountFilled"], errors="coerce"
                    ).fillna(0)

                    vamm_maker_trades = data[
                        (data["maker"].isnull()) | (data["maker"] == "")
                    ]
                    market_volume = data["quoteAssetAmountFilled"].sum()
                    vamm_maker_volume = vamm_maker_trades[
                        "quoteAssetAmountFilled"
                    ].sum()
                    percentage = (
                        (vamm_maker_volume / market_volume * 100)
                        if market_volume > 0
                        else 0
                    )

                    maker_stats.append(
                        {
                            "Market": market,
                            "vAMM Maker Volume ($)": f"{vamm_maker_volume:,.2f}",
                            "% of Market": f"{percentage:.2f}%",
                            "Trades": len(vamm_maker_trades),
                        }
                    )

                st.dataframe(pd.DataFrame(maker_stats), use_container_width=True)

            with role_tabs[1]:
                st.write("**vAMM as Taker** (trades where taker field is null/empty)")
                taker_stats = []
                for market, data in markets_data.items():
                    data["quoteAssetAmountFilled"] = pd.to_numeric(
                        data["quoteAssetAmountFilled"], errors="coerce"
                    ).fillna(0)

                    vamm_taker_trades = data[
                        (data["taker"].isnull()) | (data["taker"] == "")
                    ]
                    market_volume = data["quoteAssetAmountFilled"].sum()
                    vamm_taker_volume = vamm_taker_trades[
                        "quoteAssetAmountFilled"
                    ].sum()
                    percentage = (
                        (vamm_taker_volume / market_volume * 100)
                        if market_volume > 0
                        else 0
                    )

                    taker_stats.append(
                        {
                            "Market": market,
                            "vAMM Taker Volume ($)": f"{vamm_taker_volume:,.2f}",
                            "% of Market": f"{percentage:.2f}%",
                            "Trades": len(vamm_taker_trades),
                        }
                    )

                st.dataframe(pd.DataFrame(taker_stats), use_container_width=True)

            with role_tabs[2]:
                st.write("**Combined vAMM Analysis**")

                viz_data = []
                for market, data in markets_data.items():
                    data["quoteAssetAmountFilled"] = pd.to_numeric(
                        data["quoteAssetAmountFilled"], errors="coerce"
                    ).fillna(0)

                    market_volume = data["quoteAssetAmountFilled"].sum()

                    vamm_maker_volume = data[
                        (data["maker"].isnull()) | (data["maker"] == "")
                    ]["quoteAssetAmountFilled"].sum()

                    vamm_taker_volume = data[
                        (data["taker"].isnull()) | (data["taker"] == "")
                    ]["quoteAssetAmountFilled"].sum()

                    human_volume = market_volume - vamm_maker_volume - vamm_taker_volume

                    viz_data.append(
                        {
                            "Market": market,
                            "Human Trading": human_volume,
                            "vAMM as Maker": vamm_maker_volume,
                            "vAMM as Taker": vamm_taker_volume,
                        }
                    )

                viz_df = pd.DataFrame(viz_data).set_index("Market")

                st.bar_chart(viz_df)

            st.subheader("Daily vAMM Percentage Trend (combined maker and taker)")

            daily_vamm_records_list = []
            all_market_dfs_for_aggregate = []
            blocktime_column_found_overall = True

            for market_name, original_df in markets_data.items():
                if original_df.empty:
                    if market_name in selected_markets:
                        pass
                    continue

                df = original_df.copy()

                df["quoteAssetAmountFilled"] = pd.to_numeric(
                    df["quoteAssetAmountFilled"], errors="coerce"
                ).fillna(0)

                if "ts" not in df.columns:
                    if blocktime_column_found_overall:
                        st.error(
                            "Timestamp column 'ts' not found in some market data. Daily time series may be incomplete or missing for those markets."
                        )
                    blocktime_column_found_overall = False
                    if market_name in selected_markets:
                        chart_s_date = pd.to_datetime(start_date)
                        chart_e_date = pd.to_datetime(end_date)
                        for date_obj in pd.date_range(
                            start=chart_s_date, end=chart_e_date, freq="D"
                        ):
                            daily_vamm_records_list.append(
                                {
                                    "Date": date_obj,
                                    "Market": market_name,
                                    "VammPercentage": 0.0,
                                }
                            )
                    continue

                df["timestamp"] = pd.to_datetime(df["ts"], unit="s")
                all_market_dfs_for_aggregate.append(df.copy())
                if not df.empty:
                    daily_groups = df.groupby(df["timestamp"].dt.date)

                    for date_obj, group in daily_groups:
                        market_volume_day = group["quoteAssetAmountFilled"].sum()
                        vamm_trades_day = group[
                            ((group["maker"].isnull()) | (group["maker"] == ""))
                            | ((group["taker"].isnull()) | (group["taker"] == ""))
                        ]
                        vamm_volume_day = vamm_trades_day[
                            "quoteAssetAmountFilled"
                        ].sum()

                        vamm_percentage_day = (
                            (vamm_volume_day / market_volume_day * 100)
                            if market_volume_day > 0
                            else 0.0
                        )

                        daily_vamm_records_list.append(
                            {
                                "Date": pd.to_datetime(date_obj),
                                "Market": market_name,
                                "VammPercentage": vamm_percentage_day,
                            }
                        )

            if all_market_dfs_for_aggregate:
                combined_df = pd.concat(all_market_dfs_for_aggregate)
                if not combined_df.empty and "timestamp" in combined_df.columns:
                    daily_groups_combined = combined_df.groupby(
                        combined_df["timestamp"].dt.date
                    )

                    for date_obj, group in daily_groups_combined:
                        total_volume_day = group["quoteAssetAmountFilled"].sum()

                        vamm_trades_day_combined = group[
                            ((group["maker"].isnull()) | (group["maker"] == ""))
                            | ((group["taker"].isnull()) | (group["taker"] == ""))
                        ]
                        total_vamm_volume_day = vamm_trades_day_combined[
                            "quoteAssetAmountFilled"
                        ].sum()

                        aggregate_vamm_percentage_day = (
                            (total_vamm_volume_day / total_volume_day * 100)
                            if total_volume_day > 0
                            else 0.0
                        )

                        daily_vamm_records_list.append(
                            {
                                "Date": pd.to_datetime(date_obj),
                                "Market": "Aggregate",
                                "VammPercentage": aggregate_vamm_percentage_day,
                            }
                        )
                elif not blocktime_column_found_overall:
                    chart_s_date = pd.to_datetime(start_date)
                    chart_e_date = pd.to_datetime(end_date)
                    for date_obj in pd.date_range(
                        start=chart_s_date, end=chart_e_date, freq="D"
                    ):
                        daily_vamm_records_list.append(
                            {
                                "Date": date_obj,
                                "Market": "Aggregate",
                                "VammPercentage": 0.0,
                            }
                        )

            if daily_vamm_records_list:
                time_series_plot_df = pd.DataFrame(daily_vamm_records_list)

                chart_start_date = pd.to_datetime(start_date)
                chart_end_date = pd.to_datetime(end_date)
                all_chart_dates = pd.date_range(
                    start=chart_start_date, end=chart_end_date, freq="D"
                )

                pivot_df = time_series_plot_df.pivot_table(
                    index="Date", columns="Market", values="VammPercentage"
                )

                pivot_df = pivot_df.reindex(all_chart_dates, fill_value=0.0)

                final_columns_for_chart = []
                for market_name in selected_markets:
                    if market_name not in pivot_df.columns:
                        pivot_df[market_name] = 0.0
                    final_columns_for_chart.append(market_name)

                if "Aggregate" not in pivot_df.columns:
                    pivot_df["Aggregate"] = 0.0
                if "Aggregate" not in final_columns_for_chart:
                    final_columns_for_chart.append("Aggregate")

                cols_to_plot = [
                    col for col in final_columns_for_chart if col in pivot_df.columns
                ]
                other_existing_cols = [
                    col for col in pivot_df.columns if col not in cols_to_plot
                ]
                pivot_df = pivot_df[cols_to_plot + other_existing_cols]
                pivot_df = pivot_df.sort_index()

                if not pivot_df.empty:
                    fig = px.line(pivot_df, title="Daily vAMM Percentage (%)")
                    fig.update_layout(
                        xaxis_title="Date", yaxis_title="vAMM Percentage (%)"
                    )
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info(
                        "No data to display for the daily vAMM percentage trend after processing."
                    )
            elif not markets_data:  # No initial data at all
                st.info("No market data loaded, skipping daily trend chart.")
            elif (
                not blocktime_column_found_overall and not all_market_dfs_for_aggregate
            ):
                st.warning(
                    "No time series data generated as 'ts' column was missing in all relevant market data."
                )
            else:  # Some other case, e.g. all data filtered out by date, or no trades
                st.info(
                    "No trade data found for the selected markets/date range to generate a daily trend chart."
                )

                st.subheader("Export Data")
                csv = stats_df.to_csv(index=False)

                if date_mode == "Date Range":
                    filename = f"vamm_stats_{start_date}_to_{end_date}.csv"
                else:
                    filename = f"vamm_stats_{start_date}.csv"

                st.download_button(
                    label="Download CSV",
                    data=csv,
                    file_name=filename,
                    mime="text/csv",
                )
        else:
            st.error("No data found for the selected markets and date.")


if __name__ == "__main__":
    vamm_stats_page()

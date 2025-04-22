# TODO: Make sure these numbers are correct? Match the numbers from the website?
import datetime
import logging

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st
from driftpy.accounts import get_state_account
from driftpy.constants.perp_markets import (
    mainnet_perp_market_configs,
)
from driftpy.drift_client import DriftClient

pd.options.plotting.backend = "plotly"


async def funding_history(ch: DriftClient, env: str):
    try:
        state = await get_state_account(ch.program)
        market_options = list(range(state.number_of_markets))

        m1, m2, m3 = st.columns(3)

        selected_market_index = m1.selectbox(
            "Select Perp Market:",
            options=market_options,
            format_func=lambda x: f"{mainnet_perp_market_configs[x].symbol} ({x})",
            index=0,
            key="market_select",
        )

        if selected_market_index is None:
            st.warning("Please select a market.")
            return

        market_config = mainnet_perp_market_configs[selected_market_index]
        market_symbol = market_config.symbol

        if "end_date" not in st.session_state:
            st.session_state.end_date = datetime.datetime.now(
                datetime.timezone.utc
            ).date()
        if "start_date" not in st.session_state:
            st.session_state.start_date = (
                st.session_state.end_date - datetime.timedelta(days=7)
            )

        end_date = m3.date_input(
            "End Date (UTC):", value=st.session_state.end_date, key="end_date_input"
        )
        start_date = m2.date_input(
            "Start Date (UTC):",
            value=st.session_state.start_date,
            key="start_date_input",
        )

        st.session_state.end_date = end_date
        st.session_state.start_date = start_date

        all_records = []
        current_date = start_date
        total_days = (end_date - start_date).days + 1
        progress_bar = st.progress(0)
        status_text = st.empty()

        with st.spinner(f"Fetching funding rate data for {market_symbol}..."):
            for i, day_offset in enumerate(range(total_days)):
                fetch_date = start_date + datetime.timedelta(days=day_offset)
                url = f"https://data.api.drift.trade/market/{market_symbol}/fundingRates/{fetch_date.year}/{fetch_date.month:02d}/{fetch_date.day:02d}"
                status_text.text(
                    f"Fetching data for: {fetch_date.strftime('%Y-%m-%d')} ({i + 1}/{total_days})"
                )

                try:
                    response = requests.get(url, timeout=10)
                    response.raise_for_status()
                    data = response.json()

                    if data.get("success") and data.get("records"):
                        all_records.extend(data["records"])

                except requests.exceptions.Timeout:
                    st.error(
                        f"Request timed out for {fetch_date.strftime('%Y-%m-%d')}. Skipping."
                    )
                    logging.warning(f"Request timed out for {url}")
                except requests.exceptions.HTTPError as http_err:
                    if response.status_code == 404:
                        logging.warning(
                            f"No data found (404) for {market_symbol} on {fetch_date.strftime('%Y-%m-%d')}. URL: {url}"
                        )
                        status_text.text(
                            f"No data found for: {fetch_date.strftime('%Y-%m-%d')}"
                        )
                    else:
                        st.error(
                            f"HTTP error fetching data for {fetch_date.strftime('%Y-%m-%d')}: {http_err}"
                        )
                        logging.error(
                            f"HTTP error fetching data from {url}: {http_err}"
                        )
                except requests.exceptions.RequestException as req_err:
                    st.error(
                        f"Request error fetching data for {fetch_date.strftime('%Y-%m-%d')}: {req_err}"
                    )
                    logging.error(f"Request error fetching data from {url}: {req_err}")
                    break
                except requests.exceptions.JSONDecodeError as json_err:
                    st.error(
                        f"Error decoding JSON response for {fetch_date.strftime('%Y-%m-%d')}: {json_err}"
                    )
                    logging.error(
                        f"JSON decode error for {url}: {json_err} - Response text: {response.text[:200]}"
                    )
                except Exception as e:
                    st.error(
                        f"An unexpected error occurred fetching data for {fetch_date.strftime('%Y-%m-%d')}: {str(e)}"
                    )
                    logging.exception(f"Unexpected error fetching data from {url}")
                    break

                progress_bar.progress((i + 1) / total_days)

        status_text.text("Data fetching complete.")
        progress_bar.empty()

        if not all_records:
            st.warning(
                f"No funding rate data found for {market_symbol} in the selected period ({start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')})."
            )
            return

        df = pd.DataFrame(all_records)
        numeric_fields = [
            "ts",
            "slot",
            "marketIndex",
            "fundingRate",
            "fundingRateLong",
            "fundingRateShort",
            "cumulativeFundingRateLong",
            "cumulativeFundingRateShort",
            "oraclePriceTwap",
            "markPriceTwap",
            "periodRevenue",
            "baseAssetAmountWithAmm",
            "baseAssetAmountWithUnsettledLp",
        ]
        if "txSigIndex" in df.columns:
            numeric_fields.append("txSigIndex")

        for field in numeric_fields:
            if field in df.columns:
                df[field] = pd.to_numeric(df[field], errors="coerce")
        df["datetime"] = pd.to_datetime(df["ts"], unit="s", utc=True)

        essential_cols = [
            "datetime",
            "fundingRate",
            "oraclePriceTwap",
            "markPriceTwap",
            "cumulativeFundingRateLong",
            "cumulativeFundingRateShort",
        ]
        df.dropna(subset=essential_cols, inplace=True)

        df = df.sort_values("datetime", ascending=True).reset_index(drop=True)
        if df.empty:
            st.warning(
                "Dataframe is empty after processing and cleaning. Check data quality or date range."
            )
            return

        tab1, tab2, tab3 = st.tabs(
            ["📈 Funding Rates", "📊 Price TWAPs", "📋 Raw Data & Stats"]
        )

        with tab1:
            st.subheader(f"{market_symbol} Funding Rate Analysis")
            r1, r2 = st.columns(2)
            hor = r1.radio(
                "Rate Time Horizon:",
                ["hourly", "daily", "annual"],
                index=0,
                horizontal=True,
                key="rate_horizon",
            )
            iscum = r2.checkbox(
                "Show Actual Cumulative Funding Rates",
                value=False,
                key="show_cumulative",
            )

            rate_df = df[
                [
                    "datetime",
                    "fundingRate",
                    "cumulativeFundingRateLong",
                    "cumulativeFundingRateShort",
                ]
            ].copy()

            rate_df["funding_rate_hourly_pct"] = rate_df["fundingRate"]

            display_rate = rate_df["funding_rate_hourly_pct"].copy()
            time_label_suffix = "(hourly %)"
            if hor == "daily":
                display_rate *= 24
                time_label_suffix = "(daily %)"
            elif hor == "annual":
                display_rate *= 24 * 365.25
                time_label_suffix = "(annualized %)"

            fig_funding = go.Figure()

            fig_funding.add_trace(
                go.Scatter(
                    x=np.array(rate_df["datetime"]),
                    y=display_rate,
                    mode="lines",
                    name=f"Funding Rate {time_label_suffix}",
                    line=dict(color="royalblue"),
                )
            )

            if iscum:
                rate_df["cum_long_pct"] = rate_df["cumulativeFundingRateLong"]
                rate_df["cum_short_pct"] = rate_df["cumulativeFundingRateShort"]

                fig_funding.add_trace(
                    go.Scatter(
                        x=np.array(rate_df["datetime"]),
                        y=rate_df["cum_long_pct"],
                        mode="lines",
                        name="Cumulative Long Funding Rate (%)",
                        line=dict(dash="dash", color="green"),
                        yaxis="y2",
                    )
                )
                fig_funding.add_trace(
                    go.Scatter(
                        x=np.array(rate_df["datetime"]),
                        y=rate_df["cum_short_pct"],
                        mode="lines",
                        name="Cumulative Short Funding Rate (%)",
                        line=dict(dash="dot", color="red"),
                        yaxis="y2",
                    )
                )

            fig_funding.update_layout(
                title=f"{market_symbol} Funding Rate",
                xaxis_title="Date / Time (UTC)",
                yaxis_title=f"Funding Rate {time_label_suffix}",
                yaxis=dict(title=f"Funding Rate {time_label_suffix}", side="left"),
                yaxis2=dict(
                    title="Cumulative Funding Rate (%)",
                    overlaying="y",
                    side="right",
                    showgrid=False,
                )
                if iscum
                else {},
                legend=dict(
                    orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1
                ),
                hovermode="x unified",
                height=500,
            )

            st.plotly_chart(fig_funding, use_container_width=True)

            st.markdown("#### Funding Rate Data Points")
            display_cols_funding = ["datetime", "funding_rate_hourly_pct"]
            col_rename_map_funding = {
                "datetime": "Timestamp (UTC)",
                "funding_rate_hourly_pct": "Hourly Rate (%)",
            }
            if iscum:
                display_cols_funding.extend(["cum_long_pct", "cum_short_pct"])
                col_rename_map_funding["cum_long_pct"] = "Cumulative Long (%)"
                col_rename_map_funding["cum_short_pct"] = "Cumulative Short (%)"

            st.dataframe(
                rate_df[display_cols_funding]
                .rename(columns=col_rename_map_funding)
                .sort_values("Timestamp (UTC)", ascending=False),
                use_container_width=True,
                hide_index=True,
                column_config={
                    "Timestamp (UTC)": st.column_config.DatetimeColumn(
                        format="YYYY-MM-DD HH:mm:ss"
                    ),
                    "Hourly Rate (%)": st.column_config.NumberColumn(format="%.6f%%"),
                    "Cumulative Long (%)": st.column_config.NumberColumn(
                        format="%.4f%%"
                    ),
                    "Cumulative Short (%)": st.column_config.NumberColumn(
                        format="%.4f%%"
                    ),
                },
            )

        with tab2:
            st.subheader(f"{market_symbol} Price TWAP Analysis")
            plot_type = st.radio(
                "Plot Type:",
                [
                    "Price Levels",
                    "Price Difference",
                    "Implied Hourly Rate (%)",
                ],
                horizontal=True,
                key="twap_plot_type",
            )

            twap_df = df[["datetime", "oraclePriceTwap", "markPriceTwap"]].copy()

            twap_df["price_diff"] = (
                twap_df["markPriceTwap"] - twap_df["oraclePriceTwap"]
            )
            twap_df["implied_hourly_funding_rate_pct"] = (
                twap_df["markPriceTwap"] - twap_df["oraclePriceTwap"]
            ) / twap_df["oraclePriceTwap"].replace(0, pd.NA)
            twap_df["actual_funding_rate_pct"] = df["fundingRate"]

            fig_twap = go.Figure()

            if plot_type == "Price Levels":
                fig_twap.add_trace(
                    go.Scatter(
                        x=np.array(twap_df["datetime"]),
                        y=twap_df["oraclePriceTwap"],
                        mode="lines",
                        name="Oracle TWAP",
                        line=dict(color="orange"),
                    )
                )
                fig_twap.add_trace(
                    go.Scatter(
                        x=np.array(twap_df["datetime"]),
                        y=twap_df["markPriceTwap"],
                        mode="lines",
                        name="Mark TWAP",
                        line=dict(color="purple"),
                    )
                )
                y_title = "Price (USD)"
            elif plot_type == "Price Difference":
                fig_twap.add_trace(
                    go.Scatter(
                        x=np.array(twap_df["datetime"]),
                        y=twap_df["price_diff"],
                        mode="lines",
                        name="Mark TWAP - Oracle TWAP",
                        line=dict(color="teal"),
                    )
                )
                y_title = "Price Difference (Mark - Oracle)"
            elif plot_type == "Implied Hourly Rate (%)":
                fig_twap.add_trace(
                    go.Scatter(
                        x=np.array(twap_df["datetime"]),
                        y=twap_df["implied_hourly_funding_rate_pct"],
                        mode="lines",
                        name="Implied Hourly Rate (%)",
                        line=dict(color="brown"),
                    )
                )
                fig_twap.add_trace(
                    go.Scatter(
                        x=np.array(twap_df["datetime"]),
                        y=twap_df["actual_funding_rate_pct"],
                        mode="lines",
                        name="Actual Hourly Rate (%)",
                        line=dict(color="gray", dash="dot"),
                    )
                )
                y_title = "Hourly Funding Rate (%)"

            fig_twap.update_layout(
                title=f"{market_symbol} TWAP Analysis - {plot_type}",
                xaxis_title="Date / Time (UTC)",
                yaxis_title=y_title,
                legend=dict(
                    orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1
                ),
                hovermode="x unified",
                height=500,
            )

            st.plotly_chart(fig_twap, use_container_width=True)

            st.markdown("#### TWAP Data Points")
            display_cols_twap = [
                "datetime",
                "oraclePriceTwap",
                "markPriceTwap",
                "price_diff",
                "implied_hourly_funding_rate_pct",
            ]
            col_rename_map_twap = {
                "datetime": "Timestamp (UTC)",
                "oraclePriceTwap": "Oracle TWAP",
                "markPriceTwap": "Mark TWAP",
                "price_diff": "Difference (Mark-Oracle)",
                "implied_hourly_funding_rate_pct": "Implied Hourly Rate (%)",
            }

            st.dataframe(
                twap_df[display_cols_twap]
                .rename(columns=col_rename_map_twap)
                .sort_values("Timestamp (UTC)", ascending=False),
                use_container_width=True,
                hide_index=True,
                column_config={
                    "Timestamp (UTC)": st.column_config.DatetimeColumn(
                        format="YYYY-MM-DD HH:mm:ss"
                    ),
                    "Oracle TWAP": st.column_config.NumberColumn(format="$%.4f"),
                    "Mark TWAP": st.column_config.NumberColumn(format="$%.4f"),
                    "Difference (Mark-Oracle)": st.column_config.NumberColumn(
                        format="$%.4f"
                    ),
                    "Implied Hourly Rate (%)": st.column_config.NumberColumn(
                        format="%.6f%%"
                    ),
                },
            )

        with tab3:
            st.subheader("Summary Statistics")
            if not df.empty:
                col1, col2, col3 = st.columns(3)

                avg_hourly_rate_pct = (
                    df["fundingRate"].mean()
                    if pd.notna(df["fundingRate"].mean())
                    else 0
                )
                total_revenue = (
                    df["periodRevenue"].sum()
                    if pd.notna(df["periodRevenue"].sum())
                    else 0
                )
                latest_record = df.iloc[-1]

                with col1:
                    st.metric(
                        "Average Hourly Funding Rate", f"{avg_hourly_rate_pct:.6f}%"
                    )
                    st.metric(
                        "Latest Hourly Funding Rate",
                        f"{latest_record['fundingRate']:.6f}%",
                    )
                with col2:
                    st.metric(
                        "Latest Oracle Price TWAP",
                        f"${latest_record['oraclePriceTwap']:,.4f}",
                    )
                    st.metric(
                        "Latest Mark Price TWAP",
                        f"${latest_record['markPriceTwap']:,.4f}",
                    )
                with col3:
                    st.metric(
                        "Total Period Revenue (Sum)",
                        f"${total_revenue:,.2f}",
                    )
                    st.metric("Number of Funding Periods", f"{len(df):,}")

            else:
                st.info("No data available to calculate summary statistics.")

            st.subheader("Raw Data")
            st.dataframe(
                df.sort_values("datetime", ascending=False),
                use_container_width=True,
                height=500,
            )

    except Exception as e:
        st.exception(e)
        logging.exception(
            "An error occurred in the funding_history Streamlit function."
        )

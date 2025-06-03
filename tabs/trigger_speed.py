from datetime import datetime, timedelta

import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st
from driftpy.constants.perp_markets import mainnet_perp_market_configs

BASE_API_URL = "https://data-staging.api.drift.trade/"


def trigger_speed_analysis():
    st.write("# Trigger Speed Analysis")
    top_col1, top_col2 = st.columns(2)
    with top_col1:
        start_date = st.date_input(
            "Start Date",
            value=datetime.now().date() - timedelta(days=14),
            key="trigger_start_date",
        )
    with top_col2:
        end_date = st.date_input(
            "End Date",
            value=datetime.now().date() - timedelta(days=1),
            key="trigger_end_date",
        )

    perp_markets = [m.symbol for m in mainnet_perp_market_configs]
    selected_market = st.selectbox(
        "Select Market to Analyze:",
        options=perp_markets,
        index=0,
        key="trigger_market_select",
    )

    toggle_use_p99 = st.toggle(
        "Use P99 instead of true maximum for box plots",
        value=True,
        key="trigger_use_p99",
    )

    if selected_market and start_date and end_date:
        if start_date > end_date:
            st.error("Start date must be before end date.")
            return

        start_ts = int(datetime.combine(start_date, datetime.min.time()).timestamp())
        end_ts = int(datetime.combine(end_date, datetime.max.time()).timestamp())

        data_df = fetch_trigger_speed_data(start_ts, end_ts, selected_market)

        if data_df is None or data_df.empty:
            st.warning(
                "No data available for the selected criteria. Try a different date range or market."
            )
            return

        st.write("## Trigger Order Analytics")

        data_df["ts"] = pd.to_numeric(data_df["ts"], errors="coerce")
        data_df.dropna(subset=["ts"], inplace=True)
        data_df["datetime"] = pd.to_datetime(data_df["ts"], unit="s")
        numeric_cols = [
            "slotsToFillP10",
            "slotsToFillP25",
            "slotsToFillP50",
            "slotsToFillP75",
            "slotsToFillP99",
            "slotsToFillMax",
            "slotsToFillAvg",
            "fillVsTriggerMin",
            "fillVsTriggerMax",
            "fillVsTriggerAvg",
            "fillVsTriggerP10",
            "fillVsTriggerP25",
            "fillVsTriggerP50",
            "fillVsTriggerP75",
            "fillVsTriggerP99",
            "triggerMarketCount",
            "triggerLimitCount",
            "totalTriggeredOrders",
            "reduceOnlyCount",
            "nonReduceOnlyCount",
        ]
        for col in numeric_cols:
            if col in data_df.columns:
                data_df[col] = pd.to_numeric(data_df[col], errors="coerce")

        fillvstrigger_cols = [
            "fillVsTriggerMin",
            "fillVsTriggerMax",
            "fillVsTriggerAvg",
            "fillVsTriggerP10",
            "fillVsTriggerP25",
            "fillVsTriggerP50",
            "fillVsTriggerP75",
            "fillVsTriggerP99",
        ]
        for col in fillvstrigger_cols:
            if col in data_df.columns:
                data_df[col] = pd.to_numeric(data_df[col], errors="coerce") * 10000

        st.write(
            "### Overall Summary Statistics (Calculated from hourly/daily aggregates)"
        )
        if not data_df.empty:
            avg_p50 = (
                data_df["slotsToFillP50"].mean()
                if "slotsToFillP50" in data_df
                else float("nan")
            )
            avg_p99 = (
                data_df["slotsToFillP99"].mean()
                if "slotsToFillP99" in data_df
                else float("nan")
            )
            avg_fill_vs_trigger = (
                data_df["fillVsTriggerAvg"].mean()
                if "fillVsTriggerAvg" in data_df
                else float("nan")
            )
            total_triggered = (
                data_df["totalTriggeredOrders"].sum()
                if "totalTriggeredOrders" in data_df
                else float("nan")
            )

            col1, col2, col3, col4 = st.columns(4)
            col1.metric(
                "Avg P50 Slots to Fill",
                f"{avg_p50:.2f}" if not pd.isna(avg_p50) else "N/A",
            )
            col2.metric(
                "Avg P99 Slots to Fill",
                f"{avg_p99:.2f}" if not pd.isna(avg_p99) else "N/A",
            )
            col3.metric(
                "Avg Fill vs Trigger Price (bps)",
                f"{avg_fill_vs_trigger:.1f}"
                if not pd.isna(avg_fill_vs_trigger)
                else "N/A",
            )
            col4.metric(
                "Total Triggered Orders in Period",
                f"{total_triggered:,.0f}" if not pd.isna(total_triggered) else "N/A",
            )
        else:
            st.info("No data to display for summary statistics.")

        st.write("### Daily Distribution Visualizations")
        st.write("#### Slots to Fill Distribution")
        fig_slots_box = go.Figure()

        for i, row in data_df.iterrows():
            if not pd.isna(row.get("slotsToFillP10")) and not pd.isna(
                row.get("slotsToFillP99")
            ):
                fig_slots_box.add_trace(
                    go.Box(
                        x=[row["datetime"].strftime("%Y-%m-%d")],
                        q1=[row["slotsToFillP25"]],
                        median=[row["slotsToFillP50"]],
                        q3=[row["slotsToFillP75"]],
                        lowerfence=[row["slotsToFillP10"]],
                        upperfence=[
                            row["slotsToFillP99"]
                            if toggle_use_p99
                            else row["slotsToFillMax"]
                        ],
                        mean=[row.get("slotsToFillAvg", row["slotsToFillP50"])],
                        name=row["datetime"].strftime("%Y-%m-%d"),
                        showlegend=False,
                        boxpoints=False,
                    )
                )

        fig_slots_box.update_layout(
            title="Daily Distribution of Slots to Fill",
            xaxis_title="Date",
            yaxis_title="Slots to Fill",
            showlegend=False,
        )
        st.plotly_chart(fig_slots_box, use_container_width=True)

        st.write("#### Fill vs Trigger Price Distribution (Basis Points)")
        fig_fillvstrigger_box = go.Figure()

        for i, row in data_df.iterrows():
            if not pd.isna(row.get("fillVsTriggerP10")) and not pd.isna(
                row.get("fillVsTriggerP99")
            ):
                fig_fillvstrigger_box.add_trace(
                    go.Box(
                        x=[row["datetime"].strftime("%Y-%m-%d")],
                        q1=[row["fillVsTriggerP25"]],
                        median=[row["fillVsTriggerP50"]],
                        q3=[row["fillVsTriggerP75"]],
                        lowerfence=[row["fillVsTriggerP10"]],
                        upperfence=[
                            row["fillVsTriggerP99"]
                            if toggle_use_p99
                            else row["fillVsTriggerMax"]
                        ],
                        mean=[row.get("fillVsTriggerAvg", row["fillVsTriggerP50"])],
                        name=row["datetime"].strftime("%Y-%m-%d"),
                        showlegend=False,
                        boxpoints=False,
                    )
                )

        fig_fillvstrigger_box.update_layout(
            title="Daily Distribution of Fill vs Trigger Price Difference",
            xaxis_title="Date",
            yaxis_title="Price Difference (Basis Points)",
            showlegend=False,
        )
        st.plotly_chart(fig_fillvstrigger_box, use_container_width=True)

        st.write("#### Order Counts Over Time")
        count_cols_to_plot = [
            "triggerMarketCount",
            "triggerLimitCount",
            "totalTriggeredOrders",
            "reduceOnlyCount",
            "nonReduceOnlyCount",
        ]
        fig_counts = go.Figure()
        for c_col in count_cols_to_plot:
            if c_col in data_df.columns and not data_df[c_col].isnull().all():
                fig_counts.add_trace(
                    go.Scatter(
                        x=data_df["datetime"],
                        y=data_df[c_col],
                        mode="lines+markers",
                        name=c_col,
                    )
                )

        fig_counts.update_layout(
            title="Trigger Order Counts",
            xaxis_title="Time",
            yaxis_title="Count",
            legend_title="Order Types/Stats",
        )
        st.plotly_chart(fig_counts, use_container_width=True)

        with st.expander("📊 Legacy Time Series Plots", expanded=False):
            st.write("### Slots to Fill Over Time")
            fig_slots = go.Figure()
            percentiles_to_plot = [
                "slotsToFillP10",
                "slotsToFillP50",
                "slotsToFillP99",
                "slotsToFillAvg",
            ]
            for p_col in percentiles_to_plot:
                if p_col in data_df.columns and not data_df[p_col].isnull().all():
                    fig_slots.add_trace(
                        go.Scatter(
                            x=data_df["datetime"],
                            y=data_df[p_col],
                            mode="lines+markers",
                            name=p_col.replace("slotsToFill", ""),
                        )
                    )

            fig_slots.update_layout(
                title="Slots to Fill (Trigger to Fill/Processed)",
                xaxis_title="Time",
                yaxis_title="Slots to Fill",
                legend_title="Percentiles/Avg",
            )
            st.plotly_chart(fig_slots, use_container_width=True)

            st.write("### Fill vs Trigger Delay Over Time")
            fig_fill_vs_trigger = go.Figure()
            fill_vs_trigger_percentiles_to_plot = [
                "fillVsTriggerP10",
                "fillVsTriggerP50",
                "fillVsTriggerP99",
                "fillVsTriggerAvg",
            ]
            for p_col in fill_vs_trigger_percentiles_to_plot:
                if p_col in data_df.columns and not data_df[p_col].isnull().all():
                    fig_fill_vs_trigger.add_trace(
                        go.Scatter(
                            x=data_df["datetime"],
                            y=data_df[p_col],
                            mode="lines+markers",
                            name=p_col.replace("fillVsTrigger", ""),
                        )
                    )

            fig_fill_vs_trigger.update_layout(
                title="Fill vs Trigger Price Difference Over Time",
                xaxis_title="Time",
                yaxis_title="Price Difference (Basis Points)",
                legend_title="Percentiles/Avg",
            )
            st.plotly_chart(fig_fill_vs_trigger, use_container_width=True)

        st.write("### Raw Data Records")
        st.dataframe(data_df)


@st.cache_data(ttl=300)
def fetch_trigger_speed_data(start_ts, end_ts, market_symbol, limit=100):
    api_url = f"{BASE_API_URL}/stats/{market_symbol}/analytics/triggerOrderFill"
    params = {
        "startTs": start_ts,
        "endTs": end_ts,
        "limit": limit,
    }

    records_df = pd.DataFrame()

    try:
        print(f"Requesting data from: {api_url} with params: {params}")  # For debugging
        response = requests.get(api_url, params=params, timeout=10)  # Added timeout
        response.raise_for_status()
        data = response.json()
        print(f"Received response: {data}")  # For debugging

        if data.get("success") and "records" in data:
            records_df = pd.DataFrame(data["records"])
            if records_df.empty:
                st.info("API returned success but no records for the selected period.")
                return pd.DataFrame()

            if "ts" in records_df.columns:
                records_df["ts"] = pd.to_numeric(records_df["ts"], errors="coerce")
                records_df.sort_values("ts", inplace=True)

            return records_df
        else:
            st.error(
                f"API request was not successful or missing 'records'. Full response: {data}"
            )
            return pd.DataFrame()
    except requests.exceptions.HTTPError as http_err:
        st.error(
            f"HTTP error occurred: {http_err} - Status Code: {response.status_code} - Response: {response.text}"
        )
        return pd.DataFrame()
    except requests.exceptions.RequestException as e:
        st.error(f"Error fetching data from API: {e}")
        return pd.DataFrame()
    except ValueError as e:
        st.error(
            f"Error decoding JSON response: {e}. Response text: {response.text if 'response' in locals() else 'N/A'}"
        )
        return pd.DataFrame()

from datetime import datetime, timedelta
import timeit

import os
from typing import Literal

import boto3
from boto3.dynamodb.conditions import Key
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from driftpy.constants.perp_markets import mainnet_perp_market_configs

# Remove the BASE_API_URL since we're using DynamoDB now
# BASE_API_URL = "https://data-staging.api.drift.trade/"

# Configuration
DYNAMODB_TABLE_NAME = "staging-analytics"
DYNAMODB_REGION = "eu-west-1"
AWS_PROFILE = os.getenv("AWS_PROFILE")

# Initialize DynamoDB client
@st.cache_resource
def get_dynamodb_client():
    try:
        # Use AWS SSO profile
        session = boto3.Session(region_name=DYNAMODB_REGION, profile_name=AWS_PROFILE)
        return session.resource('dynamodb', region_name=DYNAMODB_REGION)
    except Exception as e:
        st.error(f"Failed to initialize DynamoDB client with profile '{AWS_PROFILE}': {e}")
        st.error("Make sure you've run: `aws sso login --profile drift-non-prod`")
        return None

def get_trigger_order_fill_pk(
        market: str,
        order_type: Literal['triggerMarket', 'triggerLimit', 'all'],
        cohort: Literal['0', '1000', '10000', '100000'] = '0') -> str:
    """Generate the partition key for trigger order fill stats"""
    return f"ANALYTICS#TRIGGER_ORDER_FILL#{market}#{order_type}#{cohort}"

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
    
    col1, col2 = st.columns(2)
    with col1:
        selected_market = st.selectbox(
            "Select Market to Analyze:",
            options=perp_markets,
            index=0,
            key="trigger_market_select",
        )
    
    with col2:
        order_type = st.selectbox(
            "Select Order Type:",
            options=["all", "triggerMarket", "triggerLimit"],
            index=1,
            key="trigger_order_type_select",
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

        # Fetch data for all cohorts
        all_cohorts = ["0", "1000", "10000", "100000"]
        cohort_data = {}
        
        for cohort in all_cohorts:
            data_df = fetch_trigger_speed_data_dynamodb(start_ts, end_ts, selected_market, order_type, cohort)
            if data_df is not None and not data_df.empty:
                cohort_data[cohort] = data_df

        if not cohort_data:
            st.warning(
                "No data available for the selected criteria. Try a different date range or market."
            )
            return

        st.write("## Trigger Order Analytics")

        # Process data for all cohorts
        processed_cohort_data = {}
        for cohort, data_df in cohort_data.items():
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
            
            processed_cohort_data[cohort] = data_df

        st.write(
            "### Overall Summary Statistics (Calculated from hourly/daily aggregates)"
        )
        
        # Display summary for each cohort
        cohort_items = processed_cohort_data.items()
        for i, (cohort, data_df) in enumerate(cohort_items):
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

                if i < len(cohort_items) - 1:
                    next_cohort = f" - {list(cohort_items)[i+1][0]}"
                else:
                    next_cohort = "+"
                st.write(f"#### Cohort {cohort}{next_cohort}")
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

        st.write("### Daily Distribution Visualizations")
        
        # Define colors for each cohort
        cohort_colors = {
            "0": "blue",
            "1000": "red", 
            "10000": "green",
            "100000": "orange"
        }
        
        fig_slots_box = go.Figure()

        for cohort, data_df in processed_cohort_data.items():
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
                            name=f"Cohort {cohort}",
                            legendgroup=f"cohort_{cohort}",
                            showlegend=i == 0,  # Only show legend for first trace of each cohort
                            boxpoints=False,
                            marker_color=cohort_colors.get(cohort, 'blue'),
                            line_color=cohort_colors.get(cohort, 'blue')
                        )
                    )

        fig_slots_box.update_layout(
            title="Slots from Trigger to Complete Fill (All Cohorts)",
            xaxis_title="Date",
            yaxis_title="Slots to Fill",
            showlegend=True,
        )
        st.plotly_chart(fig_slots_box, use_container_width=True)

        st.write("#### Order Avg Fill vs Trigger Price (Basis Points)")
        fig_fillvstrigger_box = go.Figure()

        for cohort, data_df in processed_cohort_data.items():
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
                            name=f"Cohort {cohort}",
                            legendgroup=f"cohort_{cohort}",
                            showlegend=i == 0,  # Only show legend for first trace of each cohort
                            boxpoints=False,
                            marker_color=cohort_colors.get(cohort, 'red'),
                            line_color=cohort_colors.get(cohort, 'red')
                        )
                    )

        fig_fillvstrigger_box.update_layout(
            title="Daily Distribution of Fill vs Trigger Price Difference (All Cohorts)",
            xaxis_title="Date",
            yaxis_title="Price Difference (Basis Points)",
            showlegend=True,
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
        
        for cohort, data_df in processed_cohort_data.items():
            for c_col in count_cols_to_plot:
                if c_col in data_df.columns and not data_df[c_col].isnull().all():
                    fig_counts.add_trace(
                        go.Scatter(
                            x=data_df["datetime"],
                            y=data_df[c_col],
                            mode="lines+markers",
                            name=f"{c_col} - Cohort {cohort}",
                            line=dict(color=cohort_colors.get(cohort, 'blue')),
                            legendgroup=f"cohort_{cohort}",
                        )
                    )

        fig_counts.update_layout(
            title="Trigger Order Counts (All Cohorts)",
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
            
            for cohort, data_df in processed_cohort_data.items():
                for p_col in percentiles_to_plot:
                    if p_col in data_df.columns and not data_df[p_col].isnull().all():
                        fig_slots.add_trace(
                            go.Scatter(
                                x=data_df["datetime"],
                                y=data_df[p_col],
                                mode="lines+markers",
                                name=f"{p_col.replace('slotsToFill', '')} - Cohort {cohort}",
                                line=dict(color=cohort_colors.get(cohort, 'blue')),
                                legendgroup=f"cohort_{cohort}",
                            )
                        )

            fig_slots.update_layout(
                title="Slots to Fill (Trigger to Fill/Processed) - All Cohorts",
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
            
            for cohort, data_df in processed_cohort_data.items():
                for p_col in fill_vs_trigger_percentiles_to_plot:
                    if p_col in data_df.columns and not data_df[p_col].isnull().all():
                        fig_fill_vs_trigger.add_trace(
                            go.Scatter(
                                x=data_df["datetime"],
                                y=data_df[p_col],
                                mode="lines+markers",
                                name=f"{p_col.replace('fillVsTrigger', '')} - Cohort {cohort}",
                                line=dict(color=cohort_colors.get(cohort, 'red')),
                                legendgroup=f"cohort_{cohort}",
                            )
                        )

            fig_fill_vs_trigger.update_layout(
                title="Fill vs Trigger Price Difference Over Time (All Cohorts)",
                xaxis_title="Time",
                yaxis_title="Price Difference (Basis Points)",
                legend_title="Percentiles/Avg",
            )
            st.plotly_chart(fig_fill_vs_trigger, use_container_width=True)

        with st.expander("📊 Debugging"):
            st.write("#### Raw Data Records")
            # Show raw data for each cohort in separate tabs
            if len(processed_cohort_data) > 1:
                tab_names = [f"Cohort {cohort}" for cohort in processed_cohort_data.keys()]
                tabs = st.tabs(tab_names)
                for i, (cohort, data_df) in enumerate(processed_cohort_data.items()):
                    with tabs[i]:
                        st.dataframe(data_df)
            else:
                # If only one cohort has data, show it directly
                for cohort, data_df in processed_cohort_data.items():
                    st.write(f"**Cohort {cohort}**")
                    st.dataframe(data_df)


@st.cache_data(ttl=300)
def fetch_trigger_speed_data_dynamodb(start_ts, end_ts, market_symbol, order_type, cohort='0'):
    """
    Fetch trigger speed data from DynamoDB
    
    Args:
        start_ts: Start timestamp (Unix seconds)
        end_ts: End timestamp (Unix seconds) 
        market_symbol: Market symbol (e.g., 'SOL-PERP')
        order_type: 'triggerMarket', 'triggerLimit', or 'all'
        cohort: Cohort identifier (default '0')
        limit: Maximum number of records to return
    """
    try:
        dynamodb = get_dynamodb_client()
        
        # Check if client initialization failed
        if dynamodb is None:
            st.error("Cannot proceed without valid DynamoDB connection.")
            return pd.DataFrame()
            
        table = dynamodb.Table(DYNAMODB_TABLE_NAME)
        
        pk = get_trigger_order_fill_pk(market_symbol, order_type, cohort)
        
        # Debug information
        
        start_time = timeit.default_timer()
        
        # Collect all items with pagination
        all_items = []
        last_evaluated_key = None
        page_count = 0
        
        while True:
            page_count += 1
            query_params = {
                'KeyConditionExpression': Key('pk').eq(pk) & Key('sk').between(str(start_ts), str(end_ts)),
                'ScanIndexForward': True
            }
            
            if last_evaluated_key:
                query_params['ExclusiveStartKey'] = last_evaluated_key
            
            response = table.query(**query_params)
            
            items = response.get('Items', [])
            all_items.extend(items)
            
            # Check if there are more items to fetch
            last_evaluated_key = response.get('LastEvaluatedKey')
            if not last_evaluated_key:
                break
        
        # Debugging metadata
        # total_time = timeit.default_timer() - start_time
        # st.write(f"### `{pk}`")
        # st.write(f"- SK range: `{start_ts}` to `{end_ts}`")
        # st.write(f"- Total items: {len(all_items)}")
        # st.write(f"- Pages fetched: {page_count}")
        # st.write(f"- Query time: {total_time:.2f}s")
        # st.write(f"- Response size estimate: ~{len(str(all_items)) / 1024:.1f} KB")
        
        if not all_items:
            st.warning(f"No records found in DynamoDB for PK: {pk} in the specified time range.")
            return pd.DataFrame()
        
        # Convert to DataFrame
        records_df = pd.DataFrame(all_items)
        
        # Convert sk back to ts for compatibility with existing code
        if 'sk' in records_df.columns:
            records_df['ts'] = pd.to_numeric(records_df['sk'], errors='coerce')
            records_df.sort_values('ts', inplace=True)
        
        return records_df
        
    except Exception as e:
        st.error(f"❌ Error fetching data from DynamoDB: {str(e)}")
        st.error(f"Table: {DYNAMODB_TABLE_NAME}, Region: {DYNAMODB_REGION}")
        return pd.DataFrame()

from datetime import datetime, timedelta

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import numpy as np
from driftpy.constants.perp_markets import mainnet_perp_market_configs

# Import DynamoDB utilities
from utils.dynamodb_client import (
    fetch_trigger_speed_data_dynamodb,
    fetch_liquidity_source_data_dynamodb
)

all_cohorts = ["0", "1000", "10000", "100000"]
cohort_colors = {
    "0": "blue",
    "1000": "red", 
    "10000": "green",
    "100000": "orange"
}

def trigger_speed_analysis():
    st.write("# Trigger Speed Analysis")
    top_col1, top_col2 = st.columns(2)
    with top_col1:
        start_date = st.date_input(
            "Start Date",
            value=datetime.now().date() - timedelta(weeks=15),
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
        value=False,
        key="trigger_use_p99",
    )

    if selected_market and start_date and end_date:
        if start_date > end_date:
            st.error("Start date must be before end date.")
            return

        start_ts = int(datetime.combine(start_date, datetime.min.time()).timestamp())
        end_ts = int(datetime.combine(end_date, datetime.max.time()).timestamp())

        # Fetch data for all cohorts
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

        (trigger_limit_liquidity_source_data, trigger_market_liquidity_source_data) = render_liquidity_source_analysis(start_ts, end_ts, selected_market, order_type)

        st.write("### Daily Distribution Visualizations")

        render_summary_stats(processed_cohort_data)
        
        render_trigger_latency(toggle_use_p99, processed_cohort_data)

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
            st.write("#### Trigger speed cohort data")
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

            if not trigger_limit_liquidity_source_data.empty:
                st.write("#### Trigger Limit Liquidity Source Data")
                st.dataframe(trigger_limit_liquidity_source_data)

            if not trigger_market_liquidity_source_data.empty:
                st.write("#### Trigger Market Liquidity Source Data")
                st.dataframe(trigger_market_liquidity_source_data)


def render_summary_stats(processed_cohort_data):
    with st.expander("📊 Overall Summary Statistics", expanded=False):
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


def render_trigger_latency(toggle_use_p99, processed_cohort_data):
    # Add selectbox for choosing which trend lines to display
    st.write("#### Slots from Trigger to Complete Fill (Trigger Latency)")
    
    col1, col2 = st.columns(2)
    
    with col1:
        trend_line_options = {
            "All": ["slotsToFillP25", "slotsToFillP50", "slotsToFillP99", "slotsToFillAvg"],
            "P25 only": ["slotsToFillP25"],
            "P50 (Median) only": ["slotsToFillP50"], 
            "P99 only": ["slotsToFillP99"],
            "Average only": ["slotsToFillAvg"],
            "None": []
        }
        
        selected_trend_option = st.selectbox(
            "Select trend lines to display:",
            options=list(trend_line_options.keys()),
            index=3,
            key="trigger_latency_trend_select"
        )
    
    with col2:
        smoothing_window = st.selectbox(
            "Trend line smoothing:",
            options=[("No smoothing", 1), ("Light (3-day)", 3), ("Medium (7-day)", 7), ("Heavy (14-day)", 14)],
            index=2,  # Default to Medium (7-day)
            key="trigger_latency_smoothing_select",
            format_func=lambda x: x[0]
        )[1]
    
    selected_percentiles = trend_line_options[selected_trend_option]
    
    fig_slots_box = go.Figure()

    # First, add the box plots
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

    # Add trend lines for selected percentiles only
    percentiles_config = {
        "slotsToFillP25": ("P25", "dash"),
        "slotsToFillP50": ("P50 (Median)", "solid"),
        "slotsToFillP99": ("P99", "dot"),
        "slotsToFillAvg": ("Average", "dashdot")
    }
    
    for cohort, data_df in processed_cohort_data.items():
        if len(data_df) > 1:  # Only add trend lines if we have multiple data points
            cohort_color = cohort_colors.get(cohort, 'blue')
            
            for percentile_col in selected_percentiles:
                if percentile_col in percentiles_config and percentile_col in data_df.columns and not data_df[percentile_col].isnull().all():
                    percentile_name, line_style = percentiles_config[percentile_col]
                    
                    # Sort by datetime to ensure proper line connection
                    sorted_df = data_df.sort_values('datetime').copy()
                    
                    # Apply smoothing if requested
                    y_values = sorted_df[percentile_col]
                    if smoothing_window > 1 and len(sorted_df) >= smoothing_window:
                        # Apply rolling average smoothing
                        y_values = sorted_df[percentile_col].rolling(
                            window=smoothing_window, 
                            center=True, 
                            min_periods=1
                        ).mean()
                    
                    fig_slots_box.add_trace(
                        go.Scatter(
                            x=sorted_df["datetime"],
                            y=y_values,
                            mode="lines",
                            name=f"{percentile_name} Trend - Cohort {cohort}",
                            line=dict(
                                color=cohort_color,
                                width=3,  # Slightly thicker for smoothed lines
                                dash=line_style,
                                shape='spline',  # Spline interpolation for smoother curves
                                smoothing=1.3  # Additional smoothing parameter
                            ),
                            legendgroup=f"cohort_{cohort}_trend",
                            showlegend=True,
                            opacity=0.9,  # Slightly more opaque for smoothed lines
                            hovertemplate=f"<b>{percentile_name} Trend - Cohort {cohort}</b><br>" +
                                        "Date: %{x}<br>" +
                                        "Smoothed Slots: %{y:.2f}<br>" +
                                        f"({'Smoothed' if smoothing_window > 1 else 'Raw'} data)<extra></extra>"
                        )
                    )

    # Update title based on selected trend lines and smoothing
    title_suffix = ""
    if selected_trend_option != "None":
        if selected_trend_option == "All":
            title_suffix = " with Trend Lines"
        else:
            title_suffix = f" with {selected_trend_option} Trend Lines"
        
        if smoothing_window > 1:
            title_suffix += f" (Smoothed)"
    
    fig_slots_box.update_layout(
        title=f"Slots from Trigger to Complete Fill (All Cohorts){title_suffix}",
        xaxis_title="Date",
        yaxis_title="Slots to Fill",
        showlegend=True,
        legend=dict(
            orientation="v",
            yanchor="top",
            y=1,
            xanchor="left",
            x=1.01
        ),
        hovermode='x unified'
    )
    
    # Calculate and display slopes for selected trend lines
    if selected_trend_option != "None" and len(processed_cohort_data) > 0:
        st.write("**Trend Analysis (Slots per Day):**")
        
        # Calculate slopes for each cohort and selected percentile
        slope_data = {}
        for cohort, data_df in processed_cohort_data.items():
            if len(data_df) > 1:
                sorted_df = data_df.sort_values('datetime').copy()
                
                for percentile_col in selected_percentiles:
                    if percentile_col in sorted_df.columns and not sorted_df[percentile_col].isnull().all():
                        # Calculate slope using linear regression
                        x_days = np.arange(len(sorted_df))  # Days from start
                        y_values = sorted_df[percentile_col].dropna()
                        x_days = x_days[:len(y_values)]  # Match lengths if there are NaN values
                        
                        if len(y_values) > 1:
                            slope, _ = np.polyfit(x_days, y_values, 1)
                            
                            percentile_name = percentile_col.replace('slotsToFill', '')
                            key = f"{percentile_name} - Cohort {cohort}"
                            slope_data[key] = slope
        
        # Display slopes in columns
        if slope_data:
            cols = st.columns(min(len(slope_data), 4))
            for i, (key, slope) in enumerate(slope_data.items()):
                col_idx = i % len(cols)
                with cols[col_idx]:
                    # Determine color based on slope direction
                    delta_color = "normal"
                    if slope > 0.01:
                        delta_color = "inverse"  # Red for increasing (bad)
                    elif slope < -0.01:
                        delta_color = "normal"  # Green for decreasing (good)
                    
                    st.metric(
                        label=key,
                        value=f"{slope:.3f}",
                        delta=f"{'↑' if slope > 0 else '↓'} {'Worsening' if slope > 0 else 'Improving'}",
                        delta_color=delta_color
                    )
    
    st.plotly_chart(fig_slots_box, use_container_width=True)

    st.write("#### Order Avg Fill vs Trigger Price (Basis Points)")
    
    # Add selectbox for choosing which trend lines to display for fill vs trigger
    col1, col2 = st.columns(2)
    
    with col1:
        fillvstrigger_trend_line_options = {
            "All": ["fillVsTriggerP25", "fillVsTriggerP50", "fillVsTriggerP99", "fillVsTriggerAvg"],
            "P25 only": ["fillVsTriggerP25"],
            "P50 (Median) only": ["fillVsTriggerP50"], 
            "P99 only": ["fillVsTriggerP99"],
            "Average only": ["fillVsTriggerAvg"],
            "None": []
        }
        
        selected_fillvstrigger_trend_option = st.selectbox(
            "Select trend lines to display:",
            options=list(fillvstrigger_trend_line_options.keys()),
            index=3,
            key="fillvstrigger_trend_select"
        )
    
    with col2:
        fillvstrigger_smoothing_window = st.selectbox(
            "Trend line smoothing:",
            options=[("No smoothing", 1), ("Light (3-day)", 3), ("Medium (7-day)", 7), ("Heavy (14-day)", 14)],
            index=2,  # Default to Medium (7-day)
            key="fillvstrigger_smoothing_select",
            format_func=lambda x: x[0]
        )[1]
    
    selected_fillvstrigger_percentiles = fillvstrigger_trend_line_options[selected_fillvstrigger_trend_option]
    
    fig_fillvstrigger_box = go.Figure()

    # First, add the box plots
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

    # Add trend lines for selected percentiles only
    fillvstrigger_percentiles_config = {
        "fillVsTriggerP25": ("P25", "dash"),
        "fillVsTriggerP50": ("P50 (Median)", "solid"),
        "fillVsTriggerP99": ("P99", "dot"),
        "fillVsTriggerAvg": ("Average", "dashdot")
    }
    
    for cohort, data_df in processed_cohort_data.items():
        if len(data_df) > 1:  # Only add trend lines if we have multiple data points
            cohort_color = cohort_colors.get(cohort, 'red')
            
            for percentile_col in selected_fillvstrigger_percentiles:
                if percentile_col in fillvstrigger_percentiles_config and percentile_col in data_df.columns and not data_df[percentile_col].isnull().all():
                    percentile_name, line_style = fillvstrigger_percentiles_config[percentile_col]
                    
                    # Sort by datetime to ensure proper line connection
                    sorted_df = data_df.sort_values('datetime').copy()
                    
                    # Apply smoothing if requested
                    y_values = sorted_df[percentile_col]
                    if fillvstrigger_smoothing_window > 1 and len(sorted_df) >= fillvstrigger_smoothing_window:
                        # Apply rolling average smoothing
                        y_values = sorted_df[percentile_col].rolling(
                            window=fillvstrigger_smoothing_window, 
                            center=True, 
                            min_periods=1
                        ).mean()
                    
                    fig_fillvstrigger_box.add_trace(
                        go.Scatter(
                            x=sorted_df["datetime"],
                            y=y_values,
                            mode="lines",
                            name=f"{percentile_name} Trend - Cohort {cohort}",
                            line=dict(
                                color=cohort_color,
                                width=3,  # Slightly thicker for smoothed lines
                                dash=line_style,
                                shape='spline',  # Spline interpolation for smoother curves
                                smoothing=1.3  # Additional smoothing parameter
                            ),
                            legendgroup=f"cohort_{cohort}_fillvstrigger_trend",
                            showlegend=True,
                            opacity=0.9,  # Slightly more opaque for smoothed lines
                            hovertemplate=f"<b>{percentile_name} Trend - Cohort {cohort}</b><br>" +
                                        "Date: %{x}<br>" +
                                        "Smoothed Basis Points: %{y:.2f}<br>" +
                                        f"({'Smoothed' if fillvstrigger_smoothing_window > 1 else 'Raw'} data)<extra></extra>"
                        )
                    )

    # Update title based on selected trend lines and smoothing
    fillvstrigger_title_suffix = ""
    if selected_fillvstrigger_trend_option != "None":
        if selected_fillvstrigger_trend_option == "All":
            fillvstrigger_title_suffix = " with Trend Lines"
        else:
            fillvstrigger_title_suffix = f" with {selected_fillvstrigger_trend_option} Trend Lines"
        
        if fillvstrigger_smoothing_window > 1:
            fillvstrigger_title_suffix += f" (Smoothed)"

    fig_fillvstrigger_box.update_layout(
        title=f"Daily Distribution of Fill vs Trigger Price Difference (All Cohorts){fillvstrigger_title_suffix}",
        xaxis_title="Date",
        yaxis_title="Price Difference (Basis Points)",
        showlegend=True,
        legend=dict(
            orientation="v",
            yanchor="top",
            y=1,
            xanchor="left",
            x=1.01
        ),
        hovermode='x unified'
    )
    
    # Calculate and display slopes for selected fill vs trigger trend lines
    if selected_fillvstrigger_trend_option != "None" and len(processed_cohort_data) > 0:
        st.write("**Trend Analysis (Basis Points per Day):**")
        
        # Calculate slopes for each cohort and selected percentile
        fillvstrigger_slope_data = {}
        for cohort, data_df in processed_cohort_data.items():
            if len(data_df) > 1:
                sorted_df = data_df.sort_values('datetime').copy()
                
                for percentile_col in selected_fillvstrigger_percentiles:
                    if percentile_col in sorted_df.columns and not sorted_df[percentile_col].isnull().all():
                        # Calculate slope using linear regression
                        x_days = np.arange(len(sorted_df))  # Days from start
                        y_values = sorted_df[percentile_col].dropna()
                        x_days = x_days[:len(y_values)]  # Match lengths if there are NaN values
                        
                        if len(y_values) > 1:
                            slope, _ = np.polyfit(x_days, y_values, 1)
                            
                            percentile_name = percentile_col.replace('fillVsTrigger', '')
                            key = f"{percentile_name} - Cohort {cohort}"
                            fillvstrigger_slope_data[key] = slope
        
        # Display slopes in columns
        if fillvstrigger_slope_data:
            cols = st.columns(min(len(fillvstrigger_slope_data), 4))
            for i, (key, slope) in enumerate(fillvstrigger_slope_data.items()):
                col_idx = i % len(cols)
                with cols[col_idx]:
                    # Determine color based on slope direction
                    # For fill vs trigger price difference, negative slope could be good or bad depending on context
                    # Generally, we want this difference to be small and stable
                    delta_color = "normal"
                    abs_slope = abs(slope)
                    if abs_slope > 1.0:  # More than 1 basis point change per day
                        delta_color = "inverse" if slope > 0 else "normal"
                    
                    # Create more nuanced delta text
                    if abs_slope < 0.1:
                        trend_text = "Stable"
                        delta_color = "normal"
                    elif slope > 0:
                        trend_text = "↑ Increasing"
                        delta_color = "inverse"
                    else:
                        trend_text = "↓ Decreasing"
                        delta_color = "normal"
                    
                    st.metric(
                        label=key,
                        value=f"{slope:.2f}",
                        delta=trend_text,
                        delta_color=delta_color
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

def render_liquidity_source_analysis(start_ts, end_ts, selected_market, order_type):
    """
    Render liquidity source analysis charts for trigger orders
    
    Args:
        start_ts: Start timestamp (Unix seconds)
        end_ts: End timestamp (Unix seconds) 
        selected_market: Market symbol (e.g., 'SOL-PERP')
        order_type: Order type filter ('all', 'triggerLimit', 'triggerMarket')
    """
    st.write("### Liquidity Source Analysis")
    
    # Determine which data to fetch based on order_type
    trigger_limit_liquidity_source_data = pd.DataFrame()
    trigger_market_liquidity_source_data = pd.DataFrame()
    
    if order_type in ['all', 'triggerLimit']:
        trigger_limit_liquidity_source_data = fetch_liquidity_source_data_dynamodb(
            start_ts, end_ts, selected_market, cohort='all', taker_order_type='triggerLimit', bit_flag='all'
        )
    
    if order_type in ['all', 'triggerMarket']:
        trigger_market_liquidity_source_data = fetch_liquidity_source_data_dynamodb(
            start_ts, end_ts, selected_market, cohort='all', taker_order_type='triggerMarket', bit_flag='all'
        )
    
    def process_liquidity_data(df, title_prefix):
        if df.empty:
            st.warning(f"No {title_prefix.lower()} liquidity source data available for the selected period.")
            return
        
        # Convert timestamp and numeric columns
        df_processed = df.copy()
        df_processed["ts"] = pd.to_numeric(df_processed["ts"], errors="coerce")
        df_processed.dropna(subset=["ts"], inplace=True)
        df_processed["datetime"] = pd.to_datetime(df_processed["ts"], unit="s")
        
        # Convert liquidity source columns to numeric
        liquidity_cols = [
            "totalAmm", "totalAmmJit", "totalMatch", "totalMatchJit",
            "totalAmmJitLpSplit", "totalLpJit", "totalSerum", "totalPhoenix"
        ]
        for col in liquidity_cols:
            if col in df_processed.columns:
                df_processed[col] = pd.to_numeric(df_processed[col], errors="coerce")
        
        # Sort by datetime
        df_processed = df_processed.sort_values('datetime')
        
        # Focus on main liquidity sources for the chart
        main_sources = ["totalAmm", "totalAmmJit", "totalMatch", "totalMatchJit"]
        available_sources = [col for col in main_sources if col in df_processed.columns and not df_processed[col].isnull().all()]
        
        if not available_sources:
            st.warning(f"No valid liquidity source data found for {title_prefix.lower()} orders.")
            return
        
        # Calculate total volume for each time point
        df_processed['total_volume'] = df_processed[available_sources].sum(axis=1)
        
        # Calculate percentages for each source
        for col in available_sources:
            df_processed[f'{col}_pct'] = (df_processed[col] / df_processed['total_volume'] * 100).fillna(0)
        
        # Create stacked area chart with absolute values and total volume line
        fig_amm_match = go.Figure()
        
        # Define colors and labels
        source_config = {
            "totalAmm": {"color": "rgba(255, 99, 132, 0.8)", "name": "AMM"},
            "totalAmmJit": {"color": "rgba(255, 159, 164, 0.8)", "name": "AMM JIT"},
            "totalMatch": {"color": "rgba(54, 162, 235, 0.8)", "name": "Match"},
            "totalMatchJit": {"color": "rgba(116, 185, 255, 0.8)", "name": "Match JIT"}
        }
        
        # Add stacked area traces first
        for col in available_sources:
            config = source_config.get(col, {"color": "rgba(128, 128, 128, 0.8)", "name": col})
            
            # Create custom hover text with both absolute and percentage values
            hover_text = [
                f"<b>{config['name']}</b><br>" +
                f"Volume: ${abs_val:,.0f}<br>" +
                f"Percentage: {pct:.1f}%<br>"
                for date, abs_val, pct, total in zip(
                    df_processed['datetime'],
                    df_processed[col],
                    df_processed[f'{col}_pct'],
                    df_processed['total_volume']
                )
            ]
            
            fig_amm_match.add_trace(go.Scatter(
                x=df_processed['datetime'],
                y=df_processed[col],
                fill='tonexty',
                mode='none',
                name=config['name'],
                fillcolor=config['color'],
                line=dict(color=config['color'].replace('0.8', '1.0'), width=0),
                hovertemplate='%{text}<extra></extra>',
                text=hover_text,
                stackgroup='one'
            ))
        
        # Add total volume line on top
        fig_amm_match.add_trace(go.Scatter(
            x=df_processed['datetime'],
            y=df_processed['total_volume'],
            mode='lines+markers',
            name='Total Volume',
            line=dict(color='rgba(147, 112, 219, 0.9)', width=3, dash='solid'),
            marker=dict(size=6, color='rgba(147, 112, 219, 0.9)'),
            hovertemplate='<b>Total Volume</b><br>Date: %{x}<br>Volume: $%{y:,.0f}<extra></extra>',
            yaxis='y1'
        ))
        
        fig_amm_match.update_layout(
            title=f"{title_prefix} Orders: Liquidity Sources Breakdown with Total Volume",
            xaxis_title="Date",
            yaxis_title="Volume ($)",
            yaxis=dict(tickformat='$,.0f'),
            hovermode='x unified',
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            height=500
        )
        
        st.plotly_chart(fig_amm_match, use_container_width=True)
        
        # Display summary statistics
        with st.expander(f"📊 {title_prefix} Liquidity Source Summary Statistics", expanded=False):
            if not df_processed.empty:
                st.write("**Average Distribution:**")
                summary_data = {}
                for col in available_sources:
                    avg_pct = df_processed[f'{col}_pct'].mean()
                    min_pct = df_processed[f'{col}_pct'].min()
                    max_pct = df_processed[f'{col}_pct'].max()
                    avg_abs = df_processed[col].mean()
                    min_abs = df_processed[col].min()
                    max_abs = df_processed[col].max()
                    total_abs = df_processed[col].sum()
                    config = source_config.get(col, {"name": col})
                    summary_data[config['name']] = {
                        'Min %': f"{min_pct:.1f}%",
                        'Max %': f"{max_pct:.1f}%",
                        'Average %': f"{avg_pct:.1f}%",
                        'Min Volume': f"${min_abs:,.0f}",
                        'Max Volume': f"${max_abs:,.0f}",
                        'Average Volume': f"${avg_abs:,.0f}",
                        'Total Volume': f"${total_abs:,.0f}"
                    }
                
                summary_df = pd.DataFrame(summary_data).T
                st.dataframe(summary_df)
                
                st.write("**Total Volume Over Time:**")
                total_volume_stats = {
                    'Min': f"${df_processed['total_volume'].min():,.0f}",
                    'Max': f"${df_processed['total_volume'].max():,.0f}",
                    'Average': f"${df_processed['total_volume'].mean():,.0f}",
                    'Total': f"${df_processed['total_volume'].sum():,.0f}"
                }
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Min Daily", total_volume_stats['Min'])
                col2.metric("Max Daily", total_volume_stats['Max'])
                col3.metric("Average Daily", total_volume_stats['Average'])
                col4.metric("Total Period", total_volume_stats['Total'])
    
    # Process and chart based on order_type
    if order_type in ['all', 'triggerMarket'] and not trigger_market_liquidity_source_data.empty:
        process_liquidity_data(trigger_market_liquidity_source_data, "Trigger Market")
    
    if order_type in ['all', 'triggerLimit'] and not trigger_limit_liquidity_source_data.empty:
        process_liquidity_data(trigger_limit_liquidity_source_data, "Trigger Limit")

    return (trigger_limit_liquidity_source_data, trigger_market_liquidity_source_data)
from datetime import datetime as dt
from datetime import timedelta

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from driftpy.constants.perp_markets import mainnet_perp_market_configs
from driftpy.drift_client import (
    DriftClient,
)
from plotly.subplots import make_subplots

# Import DynamoDB utilities
from utils.dynamodb_client import (
    fetch_fill_metrics_data_dynamodb,
    fetch_liquidity_source_data_dynamodb
)

cohort_colors = {
    "0-1k": "#1f77b4",      # Blue
    "1k-10k": "#ff7f0e",    # Orange  
    "10k-100k": "#2ca02c",  # Green
    "100k+": "#d62728"      # Red
}
cohort_order = ["0-1k", "1k-10k", "10k-100k", "100k+"]
cohorts = ["0", "1000", "10000", "100000"]
cohort_labels = {"0": "0-1k", "1000": "1k-10k", "10000": "10k-100k", "100000": "100k+"}

order_type_colors = {
    "market": "#636EFA",
    "limit": "#EF553B",
    "oracle": "#00CC96",
    "triggerMarket": "#AB63FA",
    "triggerLimit": "#FFA15A"
}

direction_colors = {
    "long": "#00B050",  # Traditional green
    "short": "#EF553B"  # Red
}

bit_flag_colors = {
    "0": "#1f77b4",  # Non-Swift (Blue)
    "1": "#AB63FA",  # Swift (Purple)
    "both": "#AB63FA"  # Mixed (Purple)
}

# Liquidity source configurations
liquidity_source_colors = {
    "totalAmm": "rgba(255, 99, 132, 0.8)",
    "totalAmmJit": "rgba(255, 159, 164, 0.8)", 
    "totalMatch": "rgba(54, 162, 235, 0.8)",
    "totalMatchJit": "rgba(116, 185, 255, 0.8)",
    "countAmm": "rgba(255, 99, 132, 0.8)",
    "countAmmJit": "rgba(255, 159, 164, 0.8)",
    "countMatch": "rgba(54, 162, 235, 0.8)",
    "countMatchJit": "rgba(116, 185, 255, 0.8)",
    "jit_sources": "rgba(255, 159, 64, 0.8)",
    "non_jit_sources": "rgba(75, 192, 192, 0.8)",
    "amm_sources": "rgba(153, 102, 255, 0.8)",
    "dlob_sources": "rgba(255, 206, 86, 0.8)"
}

swift_type_colors = {
    "0": "#1f77b4",  # Non-Swift (Blue)
    "1": "#AB63FA"   # Swift (Purple)
}

# Cache the data fetching function to prevent unnecessary API calls
@st.cache_data(ttl=300)  # Cache for 5 minutes
def cached_fetch_fill_quality_data(start_ts, end_ts, selected_market, cohort, taker_order_type, taker_order_direction, bit_flag):
    """Cached wrapper for fetch_fill_metrics_data_dynamodb"""
    return fetch_fill_metrics_data_dynamodb(
        start_ts, end_ts, selected_market, 
        cohort, taker_order_type, 
        taker_order_direction, bit_flag
    )

# Cache the data processing function
@st.cache_data
def cached_process_fill_quality_data(data_hash, data):
    """Cached wrapper for process_fill_quality_data"""
    return process_fill_quality_data(data)

def get_data_hash(data):
    """Generate a hash for the data to use as cache key"""
    if data is None or data.empty:
        return "empty"
    return str(hash(str(data.values.tobytes()) + str(data.columns.tolist())))


def get_liquidity_source_config_and_processing(grouping_mode, volume_units="Dollars"):
    """
    Returns source configuration and processing logic based on grouping mode and volume units
    """
    # Determine which columns to use based on volume units
    if volume_units == "Counts":
        base_sources = ["countAmm", "countAmmJit", "countMatch", "countMatchJit"]
        all_liquidity_cols = [
            "countAmm", "countAmmJit", "countMatch", "countMatchJit",
            "countAmmJitLpSplit", "countLpJit", "countSerum", "countPhoenix"
        ]
    else:
        base_sources = ["totalAmm", "totalAmmJit", "totalMatch", "totalMatchJit"]
        all_liquidity_cols = [
            "totalAmm", "totalAmmJit", "totalMatch", "totalMatchJit",
            "totalAmmJitLpSplit", "totalLpJit", "totalSerum", "totalPhoenix"
        ]
    
    if grouping_mode == "Individual Sources":
        # Show all 4 sources separately
        main_sources = base_sources
        source_config = {
            base_sources[0]: {"color": "rgba(255, 99, 132, 0.8)", "name": "AMM"},
            base_sources[1]: {"color": "rgba(255, 159, 164, 0.8)", "name": "AMM JIT"},
            base_sources[2]: {"color": "rgba(54, 162, 235, 0.8)", "name": "Match"},
            base_sources[3]: {"color": "rgba(116, 185, 255, 0.8)", "name": "Match JIT"}
        }
        return main_sources, source_config, lambda df, sources: df, all_liquidity_cols
        
    elif grouping_mode == "JIT vs Non-JIT":
        # Group by JIT type
        main_sources = ["jit_sources", "non_jit_sources"]
        source_config = {
            "jit_sources": {"color": "rgba(255, 159, 64, 0.8)", "name": "JIT Sources"},
            "non_jit_sources": {"color": "rgba(75, 192, 192, 0.8)", "name": "Non-JIT Sources"}
        }
        
        def process_jit_grouping(df, sources):
            df_copy = df.copy()
            # Combine JIT sources
            jit_cols = [base_sources[1], base_sources[3]]  # AmmJit, MatchJit
            available_jit = [col for col in jit_cols if col in df_copy.columns]
            if available_jit:
                df_copy["jit_sources"] = df_copy[available_jit].sum(axis=1)
            else:
                df_copy["jit_sources"] = 0
                
            # Combine Non-JIT sources  
            non_jit_cols = [base_sources[0], base_sources[2]]  # Amm, Match
            available_non_jit = [col for col in non_jit_cols if col in df_copy.columns]
            if available_non_jit:
                df_copy["non_jit_sources"] = df_copy[available_non_jit].sum(axis=1)
            else:
                df_copy["non_jit_sources"] = 0
                
            return df_copy
            
        return main_sources, source_config, process_jit_grouping, all_liquidity_cols
        
    elif grouping_mode == "AMM vs DLOB":
        # Group by liquidity type
        main_sources = ["amm_sources", "dlob_sources"]
        source_config = {
            "amm_sources": {"color": "rgba(153, 102, 255, 0.8)", "name": "AMM Sources"},
            "dlob_sources": {"color": "rgba(255, 206, 86, 0.8)", "name": "DLOB Sources"}
        }
        
        def process_amm_dlob_grouping(df, sources):
            df_copy = df.copy()
            # Combine AMM sources
            amm_cols = [base_sources[0], base_sources[1]]  # Amm, AmmJit
            available_amm = [col for col in amm_cols if col in df_copy.columns]
            if available_amm:
                df_copy["amm_sources"] = df_copy[available_amm].sum(axis=1)
            else:
                df_copy["amm_sources"] = 0
                
            # Combine DLOB sources
            dlob_cols = [base_sources[2], base_sources[3]]  # Match, MatchJit
            available_dlob = [col for col in dlob_cols if col in df_copy.columns]
            if available_dlob:
                df_copy["dlob_sources"] = df_copy[available_dlob].sum(axis=1)
            else:
                df_copy["dlob_sources"] = 0
                
            return df_copy
            
        return main_sources, source_config, process_amm_dlob_grouping, all_liquidity_cols


def process_liquidity_source_data(data, grouping_mode, volume_units):
    """Process raw liquidity source data from DynamoDB"""
    if data is None or data.empty:
        return None, None, None
    
    df = data.copy()
    
    # Convert timestamp
    df["ts"] = pd.to_numeric(df["ts"], errors="coerce")
    df.dropna(subset=["ts"], inplace=True)
    df["datetime"] = pd.to_datetime(df["ts"], unit="s")
    
    # Get source configuration and processing based on grouping mode and volume units
    main_sources, source_config, process_func, liquidity_cols = get_liquidity_source_config_and_processing(grouping_mode, volume_units)
    
    # Convert liquidity source columns to numeric
    for col in liquidity_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    
    # Sort by datetime
    df = df.sort_values('datetime')
    
    # Apply processing function to combine sources if needed
    df = process_func(df, main_sources)
    
    # Focus on the sources defined by the grouping mode
    available_sources = [col for col in main_sources if col in df.columns and not df[col].isnull().all()]
    
    if not available_sources:
        return None, None, None
    
    # Calculate total volume for each time point
    df['total_volume'] = df[available_sources].sum(axis=1)
    
    # Calculate percentages for each source
    for col in available_sources:
        df[f'{col}_pct'] = (df[col] / df['total_volume'] * 100).fillna(0)
    
    return df, available_sources, source_config


@st.cache_data(ttl=300)  # Cache for 5 minutes
def cached_fetch_liquidity_source_data(start_ts, end_ts, selected_market, cohort="all", taker_order_type="all", bit_flag="all"):
    """Cached wrapper for fetch_liquidity_source_data_dynamodb"""
    return fetch_liquidity_source_data_dynamodb(
        start_ts, end_ts, selected_market, 
        cohort, taker_order_type, bit_flag
    )


async def fill_quality_analysis(clearinghouse: DriftClient):
    st.write(
        "Metrics on Fill Quality. Includes auction progress (time into the `auction_duction` before filling), fill price vs oracle price, and metrics on sources of liquidity."
    )

    # Date selection
    top_col1, top_col2, top_col3 = st.columns(3)
    with top_col1:
        start_date = st.date_input(
            "Start Date", value=dt.now().date() - timedelta(weeks=15)
        )
    with top_col2:
        end_date = st.date_input("End Date", value=dt.now().date() - timedelta(days=1))
    
    with top_col3:
        perp_markets = [m.symbol for m in mainnet_perp_market_configs]
        selected_market = st.selectbox(
            "Select Market to Analyze:",
            options=perp_markets,
            index=0,
        )

    # Metric customization settings
    st.write("#### ⚙️ Display Settings")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        display_mode = st.radio(
            "Fill Quality - Display Units:",
            options=["Percentiles", "Count"],
            index=0,
            help="""
            - **Percentiles**: Show P10, P25, P50, P75, P99 distributions
            - **Count**: Show number of fills
            """
        )

    with col2:
        liquidity_grouping_mode = st.radio(
            "Liquidity Sources - Grouping:",
            options=["Individual Sources", "JIT vs Non-JIT", "AMM vs DLOB"],
            index=0,
            help="""
            - **Individual Sources**: Show all 4 sources separately (AMM, AMM JIT, Match, Match JIT)
            - **JIT vs Non-JIT**: Group by JIT type (JIT Sources vs Non-JIT Sources)  
            - **AMM vs DLOB**: Group by entity type (AMM vs Users DLOB)
            """
        )
    
    with col3:
        liquidity_volume_units = st.radio(
            "Liquidity Sources - Volume Units:",
            options=["Dollars", "Counts"],
            index=0,
            help="""
            - **Dollars**: Display volume in dollar amounts
            - **Counts**: Display volume as number of individual fills
            """
        )

    with col4:
        liquidity_display_mode = st.radio(
            "Liquidity Sources - Display Mode:",
            options=["Absolute Values", "Percentage Values"],
            index=1,
            help="""
            - **Absolute Values**: Show raw volume/count values
            - **Percentage Values**: Show percentage breakdown (adds up to 100%)
            """
        )

    # Convert dates to timestamps
    start_ts = int(dt.combine(start_date, dt.min.time()).timestamp())
    end_ts = int(dt.combine(end_date, dt.max.time()).timestamp())

    # Create tabs for different analysis views
    tab_overview, tab_cohort, tab_order_type, tab_direction, tab_swift = st.tabs([
        "Overview", "By Cohort", "By Order Type", "By Direction", "By Swift"
    ])

    with tab_overview:
        render_fill_quality_overview(start_ts, end_ts, selected_market, display_mode, 
                                   liquidity_grouping_mode, liquidity_volume_units, liquidity_display_mode)
    
    with tab_cohort:
        render_fill_quality_by_cohort(start_ts, end_ts, selected_market, display_mode,
                                    liquidity_grouping_mode, liquidity_volume_units, liquidity_display_mode)
    
    with tab_order_type:
        render_fill_quality_by_order_type(start_ts, end_ts, selected_market, display_mode,
                                        liquidity_grouping_mode, liquidity_volume_units, liquidity_display_mode)
    
    with tab_direction:
        render_fill_quality_by_direction(start_ts, end_ts, selected_market, display_mode,
                                       liquidity_grouping_mode, liquidity_volume_units, liquidity_display_mode)
    
    with tab_swift:
        render_fill_quality_by_swift(start_ts, end_ts, selected_market, display_mode,
                                   liquidity_grouping_mode, liquidity_volume_units, liquidity_display_mode)


def get_metric_config(metric_type):
    """Get metric configuration based on selected metric type"""
    if metric_type == "Auction Progress":
        return {
            "base_col": "auctionProgress",
            "title": "Auction Progress",
            "y_axis": "Auction Progress (%)",
            "format": "{:.3f}",
            "hover_format": ":.3f",
            "unit": "",
            "description": "How far through the auction when filled"
        }
    elif metric_type == "Fill vs Oracle ($)":
        return {
            "base_col": "fillVsOracleAbs",
            "title": "Fill vs Oracle (Absolute)",
            "y_axis": "Price Difference ($)",
            "format": "${:.4f}",
            "hover_format": ":$.4f",
            "unit": "$",
            "description": "Absolute difference between fill price and oracle price"
        }
    elif metric_type == "Fill vs Oracle (BPS)":
        return {
            "base_col": "fillVsOracleAbsBps",
            "title": "Fill vs Oracle (BPS)",
            "y_axis": "Basis Points",
            "format": "{:.2f} bps",
            "hover_format": ":.2f",
            "unit": " bps",
            "description": "Basis points difference between fill price and oracle price"
        }


def render_fill_quality_overview(start_ts, end_ts, selected_market, display_mode, 
                                   liquidity_grouping_mode, liquidity_volume_units, liquidity_display_mode):
    """Render overview of fill quality across all segments"""
    st.write("## Overall Fill Quality Summary")
    
    # Add metric selection for this tab
    available_metrics = ["Auction Progress", "Fill vs Oracle ($)", "Fill vs Oracle (BPS)"]
    selected_metrics = st.multiselect(
        "Select Metrics to Display:",
        options=available_metrics,
        default=["Auction Progress", "Fill vs Oracle (BPS)"],
        help="""
        - **Auction Progress**: How far through the auction when filled
        - **Fill vs Oracle ($)**: Absolute dollar difference between fill and oracle price  
        - **Fill vs Oracle (BPS)**: Basis points difference between fill and oracle price
        """,
        key="overview_metrics"
    )
    
    # Check if any metrics are selected
    if not selected_metrics:
        st.warning("Please select at least one metric to display.")
        return
    
    # Fetch aggregated data (all cohorts, order types, etc.)
    fill_quality_data = cached_fetch_fill_quality_data(
        start_ts, end_ts, selected_market, 
        cohort="all", taker_order_type="all", 
        taker_order_direction="all", bit_flag="all"
    )
    
    if fill_quality_data is None or fill_quality_data.empty:
        st.warning("No fill quality data available for the selected period.")
        return
    
    # Process the data
    df_processed = cached_process_fill_quality_data(get_data_hash(fill_quality_data), fill_quality_data)
    
    if df_processed.empty:
        st.warning("No valid fill quality data found after processing.")
        return
    
    # Display selected metric types
    metric_types = selected_metrics
    
    # Display all three metric types
    for metric_type in metric_types:
        # Get metric configuration
        metric_config = get_metric_config(metric_type)
        
        # Create time series chart
        fig = create_fill_quality_timeseries(df_processed, metric_config, display_mode, "Overall")
        st.plotly_chart(fig, use_container_width=True)
    
    # Add liquidity source analysis
    st.write("---")
    st.write("### 📊 Liquidity Source Analysis - Overall")
    
    # Fetch liquidity source data
    liquidity_data = cached_fetch_liquidity_source_data(
        start_ts, end_ts, selected_market,
        cohort="all", taker_order_type="all",
        bit_flag="all"
    )
    
    if liquidity_data is not None and not liquidity_data.empty:
        df_liquidity, available_sources, source_config = process_liquidity_source_data(
            liquidity_data, liquidity_grouping_mode, liquidity_volume_units
        )
        
        if df_liquidity is not None:
            fig_liquidity = create_liquidity_source_chart(
                df_liquidity, available_sources, source_config, 
                liquidity_grouping_mode, liquidity_volume_units, liquidity_display_mode, "Overall"
            )
            if fig_liquidity:
                st.plotly_chart(fig_liquidity, use_container_width=True)
        else:
            st.warning("No valid liquidity source data found for overall analysis.")
    else:
        st.warning("No liquidity source data available for the selected period.")


def render_fill_quality_by_cohort(start_ts, end_ts, selected_market, display_mode,
                                    liquidity_grouping_mode, liquidity_volume_units, liquidity_display_mode):
    """Render fill quality analysis by order size cohort"""
    st.write("## Fill Quality by Order Size Cohort")
    
    # Add metric selection for this tab
    available_metrics = ["Auction Progress", "Fill vs Oracle ($)", "Fill vs Oracle (BPS)"]
    selected_metrics = st.multiselect(
        "Select Metrics to Display:",
        options=available_metrics,
        default=["Auction Progress", "Fill vs Oracle (BPS)"],
        help="""
        - **Auction Progress**: How far through the auction when filled
        - **Fill vs Oracle ($)**: Absolute dollar difference between fill and oracle price  
        - **Fill vs Oracle (BPS)**: Basis points difference between fill and oracle price
        """,
        key="cohort_metrics"
    )
    
    # Check if any metrics are selected
    if not selected_metrics:
        st.warning("Please select at least one metric to display.")
        return
    
    # Fetch data for each cohort
    cohort_data = {}
    for cohort in cohorts:
        data = cached_fetch_fill_quality_data(
            start_ts, end_ts, selected_market,
            cohort=cohort, taker_order_type="all",
            taker_order_direction="all", bit_flag="all"
        )
        if data is not None and not data.empty:
            cohort_data[cohort] = cached_process_fill_quality_data(get_data_hash(data), data)
    
    if not cohort_data:
        st.warning("No fill quality data available for any cohort in the selected period.")
        return
    
    # Display selected metric types
    metric_types = selected_metrics
    
    # Add box plot controls once per tab (outside the metric loop)
    if display_mode == "Percentiles":
        st.write("### Box Plot Controls (applies to all metrics)")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            toggle_use_p99 = st.toggle(
                "Use P99 instead of true maximum for box plots",
                value=True,
                key="cohort_use_p99",
            )
        
        with col2:
            trend_line_options = {
                "All": ["P25", "P50", "P99", "Avg"],
                "P25 only": ["P25"],
                "P50 (Median) only": ["P50"], 
                "P99 only": ["P99"],
                "Average only": ["Avg"],
                "None": []
            }
            
            selected_trend_option = st.selectbox(
                "Select trend lines to display:",
                options=list(trend_line_options.keys()),
                index=2,
                key="cohort_trend_select"
            )
        
        with col3:
            smoothing_window = st.selectbox(
                "Trend line smoothing:",
                options=[("No smoothing", 1), ("Light (3-day)", 3), ("Medium (7-day)", 7), ("Heavy (14-day)", 14)],
                index=1,
                key="cohort_smoothing_select",
                format_func=lambda x: x[0]
            )[1]
        
        selected_percentiles_base = trend_line_options[selected_trend_option]
    
    for metric_type in metric_types:
        # Get metric configuration
        metric_config = get_metric_config(metric_type)
        
        # Cohort selection for analysis
        selected_cohort_data = {cohort: cohort_data[cohort] for cohort in cohorts if cohort in cohort_data}
        
        if not selected_cohort_data:
            st.warning("No data available for the selected cohorts.")
            continue
        
        if display_mode == "Percentiles":
            # Convert generic percentile names to metric-specific column names
            base_col = metric_config['base_col']
            selected_percentiles = [f"{base_col}{p}" for p in selected_percentiles_base]
            
            st.write(f"### {metric_config['title']} - Daily Distribution Visualization (Box Plots with Trend Lines)")
            fig_box = create_fill_quality_box_plot(selected_cohort_data, metric_config, display_mode, toggle_use_p99, selected_percentiles, smoothing_window)
            if fig_box:
                st.plotly_chart(fig_box, use_container_width=True)
        
        elif display_mode == "Count":
            st.write(f"### {metric_config['title']} - Combined Cohort Count Analysis")
            fig_combined = create_cohort_count_comparison_chart(selected_cohort_data, metric_config)
            st.plotly_chart(fig_combined, use_container_width=True)
    
    # Add liquidity source analysis
    st.write("---")
    st.write("### 📊 Liquidity Source Analysis - By Cohort")
    
    # Fetch liquidity source data for each cohort
    cohort_liquidity_data = {}
    for cohort in cohorts:
        liquidity_data = cached_fetch_liquidity_source_data(
            start_ts, end_ts, selected_market,
            cohort=cohort, taker_order_type="all",
            bit_flag="all"
        )
        if liquidity_data is not None and not liquidity_data.empty:
            cohort_liquidity_data[cohort] = liquidity_data
    
    if cohort_liquidity_data:
        # Create comparison chart showing total volume across cohorts
        cohort_colors_by_lb = {
            "0": "#636EFA", "1000": "#EF553B", "10000": "#00CC96", "100000": "#AB63FA"
        }
        
        # Determine if we're showing percentages or absolute values
        use_percentages = liquidity_display_mode == "Percentage Values"
        
        # Determine formatting based on volume units and display mode
        if use_percentages:
            y_axis_title = "Percentage (%)"
            hover_total_format = "%{y:.1f}%"
            axis_format = ".1f"
            display_mode_text = "Percentage"
        elif liquidity_volume_units == "Counts":
            y_axis_title = "Count"
            hover_total_format = "%{y:,.0f}"
            axis_format = ",.0f"
            display_mode_text = "Absolute"
        else:
            y_axis_title = "Total Volume ($)"
            hover_total_format = "$%{y:,.0f}"
            axis_format = "$,.0f"
            display_mode_text = "Absolute"
        
        fig_comparison = go.Figure()
        
        # If using percentages, we need to calculate the total across all cohorts for each time point
        if use_percentages:
            # First, collect all data points and calculate totals
            all_cohort_data = {}
            for cohort in cohorts:
                if cohort in cohort_liquidity_data:
                    df_liquidity, available_sources, source_config = process_liquidity_source_data(
                        cohort_liquidity_data[cohort], liquidity_grouping_mode, liquidity_volume_units
                    )
                    if df_liquidity is not None:
                        all_cohort_data[cohort] = df_liquidity
            
            # Calculate percentages relative to total across all cohorts
            if all_cohort_data:
                # Create a combined dataset to calculate total volume per time point
                combined_volumes = {}
                for cohort, df in all_cohort_data.items():
                    for _, row in df.iterrows():
                        date_key = row['datetime']
                        if date_key not in combined_volumes:
                            combined_volumes[date_key] = 0
                        combined_volumes[date_key] += row['total_volume']
                
                # Now create percentage traces
                for cohort in cohorts:
                    if cohort in all_cohort_data:
                        df_liquidity = all_cohort_data[cohort]
                        cohort_label = cohort_labels[cohort]
                        
                        # Calculate percentage of total volume for each time point
                        percentage_values = []
                        absolute_values = []
                        for _, row in df_liquidity.iterrows():
                            date_key = row['datetime']
                            total_volume = combined_volumes.get(date_key, 1)  # Avoid division by zero
                            percentage = (row['total_volume'] / total_volume * 100) if total_volume > 0 else 0
                            percentage_values.append(percentage)
                            absolute_values.append(row['total_volume'])
                        
                        # Create hover text with both percentage and absolute values
                        if liquidity_volume_units == "Counts":
                            abs_format = "{:,.0f}"
                        else:
                            abs_format = "${:,.0f}"
                        
                        hover_text = [
                            f"<b>Cohort {cohort_label}</b><br>Date: {date}<br>Percentage: {pct:.1f}%<br>Volume: {abs_format.format(abs_val)}"
                            for date, pct, abs_val in zip(
                                df_liquidity['datetime'].dt.strftime('%Y-%m-%d'),
                                percentage_values,
                                absolute_values
                            )
                        ]
                        
                        # Helper function to convert hex to rgba
                        def hex_to_rgba(hex_color, alpha=0.8):
                            """Convert hex color to rgba format with specified alpha"""
                            hex_color = hex_color.lstrip('#')
                            if len(hex_color) == 6:
                                r = int(hex_color[0:2], 16)
                                g = int(hex_color[2:4], 16)
                                b = int(hex_color[4:6], 16)
                                return f"rgba({r}, {g}, {b}, {alpha})"
                            return hex_color  # fallback to original if not valid hex
                        
                        cohort_color = cohort_colors_by_lb.get(cohort, "#636EFA")
                        fill_color = hex_to_rgba(cohort_color, 0.8)
                        
                        fig_comparison.add_trace(go.Scatter(
                            x=df_liquidity['datetime'],
                            y=percentage_values,
                            fill='tonexty',
                            mode='none',
                            name=f'Cohort {cohort_label}',
                            fillcolor=fill_color,
                            line=dict(color=cohort_color, width=0),
                            hovertemplate='%{text}<extra></extra>',
                            text=hover_text,
                            stackgroup='one'
                        ))
        else:
            # Absolute values mode (existing logic)
            for cohort in cohorts:
                if cohort in cohort_liquidity_data:
                    df_liquidity, available_sources, source_config = process_liquidity_source_data(
                        cohort_liquidity_data[cohort], liquidity_grouping_mode, liquidity_volume_units
                    )
                    
                    if df_liquidity is not None:
                        cohort_label = cohort_labels[cohort]
                        fig_comparison.add_trace(go.Scatter(
                            x=df_liquidity['datetime'],
                            y=df_liquidity['total_volume'],
                            mode='lines+markers',
                            name=f'Cohort {cohort_label}',
                            line=dict(color=cohort_colors_by_lb.get(cohort, "#636EFA"), width=2),
                            marker=dict(size=4),
                            hovertemplate=f'<b>Cohort {cohort_label}</b><br>Date: %{{x}}<br>Volume: {hover_total_format}<extra></extra>'
                        ))
        
        fig_comparison.update_layout(
            title=f"Liquidity Source Volume Comparison Across Cohorts ({liquidity_volume_units} - {display_mode_text})",
            xaxis_title="Date",
            yaxis_title=y_axis_title,
            yaxis=dict(tickformat=axis_format),
            hovermode='x unified',
            height=400
        )
        
        # For percentage mode, ensure y-axis goes from 0 to 100
        if use_percentages:
            fig_comparison.update_layout(yaxis=dict(range=[0, 100], tickformat=".1f", ticksuffix="%"))
        
        st.plotly_chart(fig_comparison, use_container_width=True)
        
        # Show individual cohort charts - change default to all cohorts
        selected_cohorts = st.multiselect(
            "Select Cohorts for Detailed Liquidity Analysis", 
            cohorts, 
            default=cohorts,  # Changed from cohorts[:2] to cohorts to show all by default
            format_func=lambda x: f"Cohort {cohort_labels[x]}"
        )
        
        for cohort in selected_cohorts:
            if cohort in cohort_liquidity_data:
                cohort_label = cohort_labels[cohort]
                
                df_liquidity, available_sources, source_config = process_liquidity_source_data(
                    cohort_liquidity_data[cohort], liquidity_grouping_mode, liquidity_volume_units
                )
                
                if df_liquidity is not None:
                    fig_liquidity = create_liquidity_source_chart(
                        df_liquidity, available_sources, source_config, 
                        liquidity_grouping_mode, liquidity_volume_units, liquidity_display_mode, f"Cohort {cohort_label}"
                    )
                    if fig_liquidity:
                        st.plotly_chart(fig_liquidity, use_container_width=True)
    else:
        st.warning("No liquidity source data available for any cohort.")


@st.fragment
def render_fill_quality_by_order_type(start_ts, end_ts, selected_market, display_mode,
                                        liquidity_grouping_mode, liquidity_volume_units, liquidity_display_mode):
    """Render fill quality analysis by order type"""
    st.write("## Fill Quality by Order Type")
    
    # Add metric selection for this tab
    available_metrics = ["Auction Progress", "Fill vs Oracle ($)", "Fill vs Oracle (BPS)"]
    selected_metrics = st.multiselect(
        "Select Metrics to Display:",
        options=available_metrics,
        default=["Auction Progress", "Fill vs Oracle (BPS)"],
        help="""
        - **Auction Progress**: How far through the auction when filled
        - **Fill vs Oracle ($)**: Absolute dollar difference between fill and oracle price  
        - **Fill vs Oracle (BPS)**: Basis points difference between fill and oracle price
        """,
        key="order_type_metrics"
    )
    
    # Check if any metrics are selected
    if not selected_metrics:
        st.warning("Please select at least one metric to display.")
        return
    
    order_types = ['market', 'oracle']#, 'limit', 'triggerMarket', 'triggerLimit']
    
    # Fetch data for each order type
    order_type_data = {}
    for order_type in order_types:
        data = cached_fetch_fill_quality_data(
            start_ts, end_ts, selected_market,
            cohort="all", taker_order_type=order_type,
            taker_order_direction="all", bit_flag="all"
        )
        if data is not None and not data.empty:
            order_type_data[order_type] = cached_process_fill_quality_data(get_data_hash(data), data)
    
    if not order_type_data:
        st.warning("No fill quality data available for any order type in the selected period.")
        return
    
    # Display selected metric types
    metric_types = selected_metrics
    
    # Add box plot controls once per tab (outside the metric loop)
    if display_mode == "Percentiles":
        st.write("### Box Plot Controls (applies to all metrics)")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            toggle_use_p99 = st.toggle(
                "Use P99 instead of true maximum for box plots",
                value=True,
                key="order_type_use_p99",
            )
        
        with col2:
            trend_line_options = {
                "All": ["P25", "P50", "P99", "Avg"],
                "P25 only": ["P25"],
                "P50 (Median) only": ["P50"], 
                "P99 only": ["P99"],
                "Average only": ["Avg"],
                "None": []
            }
            
            selected_trend_option = st.selectbox(
                "Select trend lines to display:",
                options=list(trend_line_options.keys()),
                index=2,
                key="order_type_trend_select"
            )
        
        with col3:
            smoothing_window = st.selectbox(
                "Trend line smoothing:",
                options=[("No smoothing", 1), ("Light (3-day)", 3), ("Medium (7-day)", 7), ("Heavy (14-day)", 14)],
                index=1,
                key="order_type_smoothing_select",
                format_func=lambda x: x[0]
            )[1]
        
        selected_percentiles_base = trend_line_options[selected_trend_option]
    
    for metric_type in metric_types:
        # Get metric configuration
        metric_config = get_metric_config(metric_type)
        
        if display_mode == "Percentiles":
            # Convert generic percentile names to metric-specific column names
            base_col = metric_config['base_col']
            selected_percentiles = [f"{base_col}{p}" for p in selected_percentiles_base]
            
            st.write(f"### {metric_config['title']} - Daily Distribution Visualization (Box Plots with Trend Lines)")
            fig_box = create_order_type_box_plot(order_type_data, metric_config, display_mode, toggle_use_p99, selected_percentiles, smoothing_window)
            if fig_box:
                st.plotly_chart(fig_box, use_container_width=True)
        
        elif display_mode == "Count":
            st.write(f"### {metric_config['title']} - Order Type Count Comparison")
            fig_comparison = create_order_type_comparison_chart(order_type_data, metric_config, display_mode)
            st.plotly_chart(fig_comparison, use_container_width=True)
    
    # Add liquidity source analysis
    st.write("---")
    st.write("### 📊 Liquidity Source Analysis - By Order Type")
    
    # Fetch liquidity source data for each order type
    order_type_liquidity_data = {}
    for order_type in order_types:
        liquidity_data = cached_fetch_liquidity_source_data(
            start_ts, end_ts, selected_market,
            cohort="all", taker_order_type=order_type,
            bit_flag="all"
        )
        if liquidity_data is not None and not liquidity_data.empty:
            order_type_liquidity_data[order_type] = liquidity_data
    
    if order_type_liquidity_data:
        # Show individual order type charts
        for order_type in order_types:
            if order_type in order_type_liquidity_data:
                df_liquidity, available_sources, source_config = process_liquidity_source_data(
                    order_type_liquidity_data[order_type], liquidity_grouping_mode, liquidity_volume_units
                )
                
                if df_liquidity is not None:
                    fig_liquidity = create_liquidity_source_chart(
                        df_liquidity, available_sources, source_config, 
                        liquidity_grouping_mode, liquidity_volume_units, liquidity_display_mode, f"{order_type.title()} Orders"
                    )
                    if fig_liquidity:
                        st.plotly_chart(fig_liquidity, use_container_width=True)
    else:
        st.warning("No liquidity source data available for any order type.")


@st.fragment
def render_fill_quality_by_direction(start_ts, end_ts, selected_market, display_mode,
                                       liquidity_grouping_mode, liquidity_volume_units, liquidity_display_mode):
    """Render fill quality analysis by order direction"""
    st.write("## Fill Quality by Order Direction")
    
    # Add metric selection for this tab
    available_metrics = ["Auction Progress", "Fill vs Oracle ($)", "Fill vs Oracle (BPS)"]
    selected_metrics = st.multiselect(
        "Select Metrics to Display:",
        options=available_metrics,
        default=["Auction Progress", "Fill vs Oracle (BPS)"],
        help="""
        - **Auction Progress**: How far through the auction when filled
        - **Fill vs Oracle ($)**: Absolute dollar difference between fill and oracle price  
        - **Fill vs Oracle (BPS)**: Basis points difference between fill and oracle price
        """,
        key="direction_metrics"
    )
    
    # Check if any metrics are selected
    if not selected_metrics:
        st.warning("Please select at least one metric to display.")
        return
    
    # Fetch data for each direction
    direction_data = {}
    for direction in ['long', 'short']:
        data = cached_fetch_fill_quality_data(
            start_ts, end_ts, selected_market,
            cohort="all", taker_order_type="all",
            taker_order_direction=direction, bit_flag="all"
        )
        if data is not None and not data.empty:
            direction_data[direction] = cached_process_fill_quality_data(get_data_hash(data), data)
    
    if not direction_data:
        st.warning("No fill quality data available for any direction in the selected period.")
        return
    
    # Display selected metric types
    metric_types = selected_metrics
    
    # Add box plot controls once per tab (outside the metric loop)
    if display_mode == "Percentiles":
        st.write("### Box Plot Controls (applies to all metrics)")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            toggle_use_p99 = st.toggle(
                "Use P99 instead of true maximum for box plots",
                value=True,
                key="direction_use_p99",
            )
        
        with col2:
            trend_line_options = {
                "All": ["P25", "P50", "P99", "Avg"],
                "P25 only": ["P25"],
                "P50 (Median) only": ["P50"], 
                "P99 only": ["P99"],
                "Average only": ["Avg"],
                "None": []
            }
            
            selected_trend_option = st.selectbox(
                "Select trend lines to display:",
                options=list(trend_line_options.keys()),
                index=2,
                key="direction_trend_select"
            )
        
        with col3:
            smoothing_window = st.selectbox(
                "Trend line smoothing:",
                options=[("No smoothing", 1), ("Light (3-day)", 3), ("Medium (7-day)", 7), ("Heavy (14-day)", 14)],
                index=1,
                key="direction_smoothing_select",
                format_func=lambda x: x[0]
            )[1]
        
        selected_percentiles_base = trend_line_options[selected_trend_option]
    
    for metric_type in metric_types:
        # Get metric configuration
        metric_config = get_metric_config(metric_type)
        
        if display_mode == "Percentiles":
            # Convert generic percentile names to metric-specific column names
            base_col = metric_config['base_col']
            selected_percentiles = [f"{base_col}{p}" for p in selected_percentiles_base]
            
            st.write(f"### {metric_config['title']} - Daily Distribution Visualization (Box Plots with Trend Lines)")
            fig_box = create_direction_box_plot(direction_data, metric_config, display_mode, toggle_use_p99, selected_percentiles, smoothing_window)
            if fig_box:
                st.plotly_chart(fig_box, use_container_width=True)
        
        elif display_mode == "Count":
            st.write(f"### {metric_config['title']} - Direction Count Comparison")
            fig_comparison = create_direction_comparison_chart(direction_data, metric_config, display_mode)
            st.plotly_chart(fig_comparison, use_container_width=True)


@st.fragment
def render_fill_quality_by_swift(start_ts, end_ts, selected_market, display_mode,
                                   liquidity_grouping_mode, liquidity_volume_units, liquidity_display_mode):
    """Render fill quality analysis by Swift flag"""
    st.write("## Fill Quality by Swift Flag")
    
    # Add metric selection for this tab
    available_metrics = ["Auction Progress", "Fill vs Oracle ($)", "Fill vs Oracle (BPS)"]
    selected_metrics = st.multiselect(
        "Select Metrics to Display:",
        options=available_metrics,
        default=["Auction Progress", "Fill vs Oracle (BPS)"],
        help="""
        - **Auction Progress**: How far through the auction when filled
        - **Fill vs Oracle ($)**: Absolute dollar difference between fill and oracle price  
        - **Fill vs Oracle (BPS)**: Basis points difference between fill and oracle price
        """,
        key="swift_metrics"
    )
    
    # Check if any metrics are selected
    if not selected_metrics:
        st.warning("Please select at least one metric to display.")
        return
    
    swift_types = ['0', '1']
    swift_labels = {'0': 'Non-Swift', '1': 'Swift'}
    # selected_swift_types = st.multiselect("Select Swift Types to Analyze", swift_types, default=swift_types, format_func=lambda x: swift_labels[x])
    selected_swift_types = swift_types
    
    # Fetch data for each swift type
    swift_data = {}
    for swift_type in selected_swift_types:
        data = cached_fetch_fill_quality_data(
            start_ts, end_ts, selected_market,
            cohort="all", taker_order_type="all",
            taker_order_direction="all", bit_flag=swift_type
        )
        if data is not None and not data.empty:
            swift_data[swift_type] = cached_process_fill_quality_data(get_data_hash(data), data)
    
    if not swift_data:
        st.warning("No fill quality data available for any swift type in the selected period.")
        return
    
    # Display selected metric types
    metric_types = selected_metrics
    
    # Add box plot controls once per tab (outside the metric loop)
    if display_mode == "Percentiles":
        st.write("### Box Plot Controls (applies to all metrics)")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            toggle_use_p99 = st.toggle(
                "Use P99 instead of true maximum for box plots",
                value=True,
                key="swift_use_p99",
            )
        
        with col2:
            trend_line_options = {
                "All": ["P25", "P50", "P99", "Avg"],
                "P25 only": ["P25"],
                "P50 (Median) only": ["P50"], 
                "P99 only": ["P99"],
                "Average only": ["Avg"],
                "None": []
            }
            
            selected_trend_option = st.selectbox(
                "Select trend lines to display:",
                options=list(trend_line_options.keys()),
                index=2,
                key="swift_trend_select"
            )
        
        with col3:
            smoothing_window = st.selectbox(
                "Trend line smoothing:",
                options=[("No smoothing", 1), ("Light (3-day)", 3), ("Medium (7-day)", 7), ("Heavy (14-day)", 14)],
                index=1,
                key="swift_smoothing_select",
                format_func=lambda x: x[0]
            )[1]
        
        selected_percentiles_base = trend_line_options[selected_trend_option]
    
    for metric_type in metric_types:
        # Get metric configuration
        metric_config = get_metric_config(metric_type)
        
        if display_mode == "Percentiles":
            # Convert generic percentile names to metric-specific column names
            base_col = metric_config['base_col']
            selected_percentiles = [f"{base_col}{p}" for p in selected_percentiles_base]
            
            st.write(f"### {metric_config['title']} - Daily Distribution Visualization (Box Plots with Trend Lines)")
            fig_box = create_swift_box_plot(swift_data, metric_config, display_mode, toggle_use_p99, selected_percentiles, smoothing_window)
            if fig_box:
                st.plotly_chart(fig_box, use_container_width=True)
        
        else:
            # Create comparison chart for other display modes
            st.write(f"### {metric_config['title']} - Swift Comparison")
            fig_comparison = create_swift_comparison_chart(swift_data, metric_config, display_mode)
            st.plotly_chart(fig_comparison, use_container_width=True)
    
    # Add liquidity source analysis
    st.write("---")
    st.write("### 📊 Liquidity Source Analysis - By Swift Flag")
    
    swift_labels = {'0': 'Non-Swift', '1': 'Swift'}
    
    # Fetch liquidity source data for each swift type
    swift_liquidity_data = {}
    for swift_type in ['0', '1']:
        liquidity_data = cached_fetch_liquidity_source_data(
            start_ts, end_ts, selected_market,
            cohort="all", taker_order_type="all",
            bit_flag=swift_type
        )
        if liquidity_data is not None and not liquidity_data.empty:
            swift_liquidity_data[swift_type] = liquidity_data
    
    if swift_liquidity_data:
        # Show individual swift type charts
        for swift_type in ['0', '1']:
            if swift_type in swift_liquidity_data:
                swift_label = swift_labels[swift_type]
                
                df_liquidity, available_sources, source_config = process_liquidity_source_data(
                    swift_liquidity_data[swift_type], liquidity_grouping_mode, liquidity_volume_units
                )
                
                if df_liquidity is not None:
                    fig_liquidity = create_liquidity_source_chart(
                        df_liquidity, available_sources, source_config, 
                        liquidity_grouping_mode, liquidity_volume_units, liquidity_display_mode, f"{swift_label} Orders"
                    )
                    if fig_liquidity:
                        st.plotly_chart(fig_liquidity, use_container_width=True)
    else:
        st.warning("No liquidity source data available for any swift type.")


def process_fill_quality_data(data):
    """Process raw fill quality data from DynamoDB"""
    if data is None or data.empty:
        return pd.DataFrame()
    
    df = data.copy()
    
    # Convert timestamp
    df["ts"] = pd.to_numeric(df["ts"], errors="coerce")
    df.dropna(subset=["ts"], inplace=True)
    df["datetime"] = pd.to_datetime(df["ts"], unit="s")
    
    # Convert all numeric columns
    numeric_columns = [
        "auctionProgressCount", "auctionProgressMin", "auctionProgressMax", "auctionProgressAvg",
        "auctionProgressP10", "auctionProgressP25", "auctionProgressP50", "auctionProgressP75", "auctionProgressP99",
        "fillVsOracleAbsMin", "fillVsOracleAbsMax", "fillVsOracleAbsAvg",
        "fillVsOracleAbsP10", "fillVsOracleAbsP25", "fillVsOracleAbsP50", "fillVsOracleAbsP75", "fillVsOracleAbsP99",
        "fillVsOracleAbsBpsMin", "fillVsOracleAbsBpsMax", "fillVsOracleAbsBpsAvg",
        "fillVsOracleAbsBpsP10", "fillVsOracleAbsBpsP25", "fillVsOracleAbsBpsP50", "fillVsOracleAbsBpsP75", "fillVsOracleAbsBpsP99"
    ]
    
    for col in numeric_columns:
        if col in df.columns:
            if "auctionProgress" in col:
                df[col] = pd.to_numeric(df[col], errors="coerce") * 100
            else:
                df[col] = pd.to_numeric(df[col], errors="coerce")
    
    # Sort by datetime
    df = df.sort_values('datetime')
    
    return df


def display_summary_metrics(df, metric_config):
    """Display summary metrics in a grid layout"""
    if df.empty:
        st.warning("No data available for summary metrics.")
        return
    
    base_col = metric_config["base_col"]
    format_str = metric_config["format"]
    
    # Calculate summary statistics
    count_total = df[f"{base_col}Count"].sum() if f"{base_col}Count" in df.columns else 0
    avg_value = df[f"{base_col}Avg"].mean() if f"{base_col}Avg" in df.columns else 0
    p50_value = df[f"{base_col}P50"].mean() if f"{base_col}P50" in df.columns else 0
    p90_value = df[f"{base_col}P90"].mean() if f"{base_col}P90" in df.columns else 0
    p99_value = df[f"{base_col}P99"].mean() if f"{base_col}P99" in df.columns else 0
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric("Total Count", f"{int(count_total):,}")
    with col2:
        st.metric("Avg Value", format_str.format(avg_value))
    with col3:
        st.metric("Avg P50", format_str.format(p50_value))
    with col4:
        st.metric("Avg P90", format_str.format(p90_value))
    with col5:
        st.metric("Avg P99", format_str.format(p99_value))


def create_fill_quality_timeseries(df, metric_config, display_mode, title_prefix):
    """Create time series chart for fill quality"""
    fig = go.Figure()
    
    base_col = metric_config["base_col"]
    y_axis_title = metric_config["y_axis"]
    hover_format = metric_config["hover_format"]
    
    if display_mode == "Percentiles":
        percentiles = ["P10", "P25", "P50", "P75", "P99"]
        colors = ["#00CC96", "#636EFA", "#FFA15A", "#EF553B", "#AB63FA"]
        
        for i, percentile in enumerate(percentiles):
            col_name = f"{base_col}{percentile}"
            if col_name in df.columns:
                fig.add_trace(go.Scatter(
                    x=df['datetime'],
                    y=df[col_name],
                    mode='lines+markers',
                    name=f"{percentile}",
                    line=dict(color=colors[i], width=2),
                    marker=dict(size=4),
                    hovertemplate=f'<b>{percentile}</b><br>Date: %{{x}}<br>Value: %{{y{hover_format}}}<extra></extra>'
                ))

        fig.add_trace(go.Scatter(
            x=df['datetime'],
            y=df[f"{base_col}Avg"],
            mode='lines',
            name="Average",
            line=dict(color="#636EFA", width=2, dash="dash"),
            hovertemplate=f'<b>Average</b><br>Date: %{{x}}<br>Value: %{{y{hover_format}}}<extra></extra>'
        ))
    
    elif display_mode == "Count":
        col_name = f"{base_col}Count"
        if col_name in df.columns:
            fig.add_trace(go.Scatter(
                x=df['datetime'],
                y=df[col_name],
                mode='lines+markers',
                name="Count",
                line=dict(color="#00CC96", width=3),
                marker=dict(size=6),
                hovertemplate=f'<b>Count</b><br>Date: %{{x}}<br>Value: %{{y:,.0f}}<extra></extra>'
            ))
            y_axis_title = "Number of Fills"
    
    fig.update_layout(
        title=f"{title_prefix}: {metric_config['title']} ({display_mode})",
        xaxis_title="Date",
        yaxis_title=y_axis_title,
        hovermode='x unified',
        height=500
    )
    
    return fig


def create_fill_quality_distribution(df, metric_config, title_prefix):
    """Create distribution box plot for fill quality"""
    fig = go.Figure()
    
    base_col = metric_config["base_col"]
    percentiles = ["P10", "P25", "P50", "P75", "P99"]
    
    for percentile in percentiles:
        col_name = f"{base_col}{percentile}"
        if col_name in df.columns and not df[col_name].dropna().empty:
            fig.add_trace(go.Box(
                y=df[col_name].dropna(),
                name=percentile,
                boxpoints=False,
            ))
    
    fig.update_layout(
        title=f"{title_prefix}: {metric_config['title']} Distribution",
        yaxis_title=metric_config['y_axis'],
        height=400
    )
    
    return fig


def create_order_type_comparison_chart(order_type_data, metric_config, display_mode):
    """Create comparison chart across order types"""
    fig = go.Figure()
    
    base_col = metric_config["base_col"]
    y_axis_title = metric_config["y_axis"]
    hover_format = metric_config["hover_format"]
    
    if display_mode == "Percentiles":
        col_name = f"{base_col}P50"
        for order_type, df in order_type_data.items():
            if col_name in df.columns:
                fig.add_trace(go.Scatter(
                    x=df['datetime'],
                    y=df[col_name],
                    mode='lines+markers',
                    name=f"{order_type.title()}",
                    line=dict(color=order_type_colors.get(order_type, "#636EFA"), width=2),
                    marker=dict(size=4),
                    hovertemplate=f'<b>{order_type.title()}</b><br>Date: %{{x}}<br>P50: %{{y{hover_format}}}<extra></extra>'
                ))
    
    elif display_mode == "Count":
        col_name = f"{base_col}Count"
        for order_type, df in order_type_data.items():
            if col_name in df.columns:
                fig.add_trace(go.Scatter(
                    x=df['datetime'],
                    y=df[col_name],
                    mode='lines+markers',
                    name=f"{order_type.title()}",
                    line=dict(color=order_type_colors.get(order_type, "#636EFA"), width=2),
                    marker=dict(size=4),
                    hovertemplate=f'<b>{order_type.title()}</b><br>Date: %{{x}}<br>Count: %{{y:,.0f}}<extra></extra>'
                ))
        y_axis_title = "Number of Fills"
    
    fig.update_layout(
        title=f"Order Type Comparison: {metric_config['title']} ({display_mode})",
        xaxis_title="Date",
        yaxis_title=y_axis_title,
        hovermode='x unified',
        height=500
    )
    
    return fig


def create_direction_comparison_chart(direction_data, metric_config, display_mode):
    """Create comparison chart across order directions"""
    fig = go.Figure()
    
    base_col = metric_config["base_col"]
    y_axis_title = metric_config["y_axis"]
    hover_format = metric_config["hover_format"]
    
    if display_mode == "Percentiles":
        col_name = f"{base_col}P50"
        for direction, df in direction_data.items():
            if col_name in df.columns:
                fig.add_trace(go.Scatter(
                    x=df['datetime'],
                    y=df[col_name],
                    mode='lines+markers',
                    name=f"{direction.title()}",
                    line=dict(color=direction_colors.get(direction, "#636EFA"), width=2),
                    marker=dict(size=4),
                    hovertemplate=f'<b>{direction.title()}</b><br>Date: %{{x}}<br>P50: %{{y{hover_format}}}<extra></extra>'
                ))
    
    elif display_mode == "Average":
        col_name = f"{base_col}Avg"
        for direction, df in direction_data.items():
            if col_name in df.columns:
                fig.add_trace(go.Scatter(
                    x=df['datetime'],
                    y=df[col_name],
                    mode='lines+markers',
                    name=f"{direction.title()}",
                    line=dict(color=direction_colors.get(direction, "#636EFA"), width=2),
                    marker=dict(size=4),
                    hovertemplate=f'<b>{direction.title()}</b><br>Date: %{{x}}<br>Avg: %{{y{hover_format}}}<extra></extra>'
                ))
    
    elif display_mode == "Count":
        col_name = f"{base_col}Count"
        for direction, df in direction_data.items():
            if col_name in df.columns:
                fig.add_trace(go.Scatter(
                    x=df['datetime'],
                    y=df[col_name],
                    mode='lines+markers',
                    name=f"{direction.title()}",
                    line=dict(color=direction_colors.get(direction, "#636EFA"), width=2),
                    marker=dict(size=4),
                    hovertemplate=f'<b>{direction.title()}</b><br>Date: %{{x}}<br>Count: %{{y:,.0f}}<extra></extra>'
                ))
        y_axis_title = "Number of Fills"
    
    fig.update_layout(
        title=f"Direction Comparison: {metric_config['title']} ({display_mode})",
        xaxis_title="Date",
        yaxis_title=y_axis_title,
        hovermode='x unified',
        height=500
    )
    
    return fig


def create_swift_comparison_chart(swift_data, metric_config, display_mode):
    """Create comparison chart across swift flags"""
    fig = go.Figure()
    
    swift_labels = {'0': 'Non-Swift', '1': 'Swift'}
    base_col = metric_config["base_col"]
    y_axis_title = metric_config["y_axis"]
    hover_format = metric_config["hover_format"]
    
    if display_mode == "Percentiles":
        col_name = f"{base_col}P50"
        for swift_type, df in swift_data.items():
            if col_name in df.columns:
                swift_label = swift_labels[swift_type]
                fig.add_trace(go.Scatter(
                    x=df['datetime'],
                    y=df[col_name],
                    mode='lines+markers',
                    name=swift_label,
                    line=dict(color=bit_flag_colors.get(swift_type, "#636EFA"), width=2),
                    marker=dict(size=4),
                    hovertemplate=f'<b>{swift_label}</b><br>Date: %{{x}}<br>P50: %{{y{hover_format}}}<extra></extra>'
                ))
    
    elif display_mode == "Average":
        col_name = f"{base_col}Avg"
        for swift_type, df in swift_data.items():
            if col_name in df.columns:
                swift_label = swift_labels[swift_type]
                fig.add_trace(go.Scatter(
                    x=df['datetime'],
                    y=df[col_name],
                    mode='lines+markers',
                    name=swift_label,
                    line=dict(color=bit_flag_colors.get(swift_type, "#636EFA"), width=2),
                    marker=dict(size=4),
                    hovertemplate=f'<b>{swift_label}</b><br>Date: %{{x}}<br>Avg: %{{y{hover_format}}}<extra></extra>'
                ))
    
    elif display_mode == "Count":
        col_name = f"{base_col}Count"
        for swift_type, df in swift_data.items():
            if col_name in df.columns:
                swift_label = swift_labels[swift_type]
                fig.add_trace(go.Scatter(
                    x=df['datetime'],
                    y=df[col_name],
                    mode='lines+markers',
                    name=swift_label,
                    line=dict(color=bit_flag_colors.get(swift_type, "#636EFA"), width=2),
                    marker=dict(size=4),
                    hovertemplate=f'<b>{swift_label}</b><br>Date: %{{x}}<br>Count: %{{y:,.0f}}<extra></extra>'
                ))
        y_axis_title = "Number of Fills"
    
    fig.update_layout(
        title=f"Swift Comparison: {metric_config['title']} ({display_mode})",
        xaxis_title="Date",
        yaxis_title=y_axis_title,
        hovermode='x unified',
        height=500
    )
    
    return fig


def create_generic_box_plot_with_trends(data_dict, metric_config, display_mode, toggle_use_p99, 
                                      color_mapping, label_mapping, title_prefix, widget_key_prefix,
                                      selected_percentiles=[], smoothing_window=1):
    """Create box plots with trend lines for any data grouping (cohorts, order types, etc.)"""
    
    base_col = metric_config["base_col"]
    y_axis_title = metric_config["y_axis"]
    hover_format = metric_config["hover_format"]
    title = metric_config["title"]
    
    if display_mode != "Percentiles":
        st.warning("Box plots are only available in Percentiles mode.")
        return None
    
    # Check if we have the required percentile columns
    required_percentiles = [f"{base_col}P10", f"{base_col}P25", f"{base_col}P50", f"{base_col}P75", f"{base_col}P99"]
    max_col = f"{base_col}Max" if f"{base_col}Max" in next(iter(data_dict.values())).columns else None
    avg_col = f"{base_col}Avg" if f"{base_col}Avg" in next(iter(data_dict.values())).columns else None
    
    fig_box = go.Figure()
    
    # First, add the box plots
    for data_key, data_df in data_dict.items():
        display_label = label_mapping.get(data_key, data_key) if label_mapping else data_key.title()
        color = color_mapping.get(data_key, "#636EFA") if color_mapping else "#636EFA"
        
        for i, row in data_df.iterrows():
            # Check if we have the minimum required data for a box plot
            if (not pd.isna(row.get(f"{base_col}P10")) and 
                not pd.isna(row.get(f"{base_col}P99")) and
                not pd.isna(row.get(f"{base_col}P50"))):
                
                fig_box.add_trace(
                    go.Box(
                        x=[row["datetime"].strftime("%Y-%m-%d")],
                        q1=[row.get(f"{base_col}P25", row.get(f"{base_col}P50", 0))],
                        median=[row.get(f"{base_col}P50", 0)],
                        q3=[row.get(f"{base_col}P75", row.get(f"{base_col}P50", 0))],
                        lowerfence=[row.get(f"{base_col}P10", row.get(f"{base_col}P50", 0))],
                        upperfence=[
                            row.get(f"{base_col}P99", row.get(f"{base_col}P50", 0))
                            if toggle_use_p99 or max_col is None
                            else row.get(max_col, row.get(f"{base_col}P99", row.get(f"{base_col}P50", 0)))
                        ],
                        mean=[row.get(avg_col, row.get(f"{base_col}P50", 0))],
                        name=display_label,
                        legendgroup=f"{widget_key_prefix}_{data_key}",
                        showlegend=i == 0,  # Only show legend for first trace of each group
                        boxpoints=False,
                        marker_color=color,
                        line_color=color
                    )
                )

    # Add trend lines for selected percentiles only
    percentiles_config = {
        f"{base_col}P25": ("P25", "dash"),
        f"{base_col}P50": ("P50 (Median)", "solid"),
        f"{base_col}P99": ("P99", "dot"),
        f"{base_col}Avg": ("Average", "dashdot")
    }
    
    for data_key, data_df in data_dict.items():
        if len(data_df) > 1:  # Only add trend lines if we have multiple data points
            display_label = label_mapping.get(data_key, data_key) if label_mapping else data_key.title()
            color = color_mapping.get(data_key, "#636EFA") if color_mapping else "#636EFA"
            
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
                    
                    fig_box.add_trace(
                        go.Scatter(
                            x=sorted_df["datetime"],
                            y=y_values,
                            mode="lines",
                            name=f"{percentile_name} Trend - {display_label}",
                            line=dict(
                                color=color,
                                width=3,  # Slightly thicker for smoothed lines
                                dash=line_style,
                                shape='spline',  # Spline interpolation for smoother curves
                                smoothing=1.3  # Additional smoothing parameter
                            ),
                            legendgroup=f"{widget_key_prefix}_{data_key}_trend",
                            showlegend=True,
                            opacity=0.9,  # Slightly more opaque for smoothed lines
                            hovertemplate=f"<b>{percentile_name} Trend - {display_label}</b><br>" +
                                        "Date: %{x}<br>" +
                                        f"Smoothed Value: %{{y{hover_format}}}<br>" +
                                        f"({'Smoothed' if smoothing_window > 1 else 'Raw'} data)<extra></extra>"
                        )
                    )

    # Update title based on selected trend lines and smoothing
    title_suffix = ""
    if selected_percentiles:
        if len(selected_percentiles) > 1:
            title_suffix = " with Trend Lines"
        else:
            # Find which percentile is selected for single trend line
            percentile_names = {f"{base_col}P25": "P25", f"{base_col}P50": "P50", f"{base_col}P99": "P99", f"{base_col}Avg": "Average"}
            for col in selected_percentiles:
                if col in percentile_names:
                    title_suffix = f" with {percentile_names[col]} Trend Line"
                    break
        
        if smoothing_window > 1:
            title_suffix += f" (Smoothed)"
    
    fig_box.update_layout(
        title=f"Daily Distribution of {title} ({title_prefix}){title_suffix}",
        xaxis_title="Date",
        yaxis_title=y_axis_title,
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
    if selected_percentiles and len(data_dict) > 0:
        st.write("**Trend Analysis (Change per Day):**")
        
        # Calculate slopes for each data group and selected percentile
        slope_data = {}
        for data_key, data_df in data_dict.items():
            if len(data_df) > 1:
                sorted_df = data_df.sort_values('datetime').copy()
                display_label = label_mapping.get(data_key, data_key) if label_mapping else data_key.title()
                
                for percentile_col in selected_percentiles:
                    if percentile_col in sorted_df.columns and not sorted_df[percentile_col].isnull().all():
                        # Calculate slope using linear regression
                        x_days = np.arange(len(sorted_df))  # Days from start
                        y_values = sorted_df[percentile_col].dropna()
                        x_days = x_days[:len(y_values)]  # Match lengths if there are NaN values
                        
                        if len(y_values) > 1:
                            slope, _ = np.polyfit(x_days, y_values, 1)
                            
                            percentile_name = percentile_col.replace(base_col, '')
                            key = f"{percentile_name} - {display_label}"
                            slope_data[key] = slope
        
        # Display slopes in columns
        if slope_data:
            cols = st.columns(min(len(slope_data), 4))
            for i, (key, slope) in enumerate(slope_data.items()):
                col_idx = i % len(cols)
                with cols[col_idx]:
                    # Determine color based on slope direction and metric type
                    delta_color = "normal"
                    
                    # For auction progress, lower is generally better (faster fills)
                    # For fill vs oracle metrics, smaller differences are generally better
                    if "auctionProgress" in base_col:
                        if slope > 0.001:  # More than 0.1% increase per day
                            delta_color = "inverse"  # Red for worsening (slower)
                        elif slope < -0.001:  # More than 0.1% decrease per day
                            delta_color = "normal"  # Green for improving (faster)
                    else:  # Fill vs oracle metrics
                        if abs(slope) > 0.1:  # Significant change
                            delta_color = "inverse" if slope > 0 else "normal"
                    
                    # Create trend text
                    if abs(slope) < 0.001:
                        trend_text = "Stable"
                        delta_color = "normal"
                    elif slope > 0:
                        trend_text = "↑ Increasing"
                    else:
                        trend_text = "↓ Decreasing"
                    
                    # Format slope based on metric type
                    if "auctionProgress" in base_col:
                        slope_display = f"{slope*100:.3f}%"
                    elif "Bps" in base_col:
                        slope_display = f"{slope:.2f} bps"
                    else:
                        slope_display = f"{slope:.4f}"
                    
                    st.metric(
                        label=key,
                        value=slope_display,
                        delta=trend_text,
                        delta_color=delta_color
                    )
    
    return fig_box


def create_fill_quality_box_plot(cohort_data, metric_config, display_mode, toggle_use_p99=False, selected_percentiles=[], smoothing_window=1):
    """Create box plots for fill quality across cohorts"""
    # Create proper color mapping from cohort keys to colors
    cohort_color_mapping = {}
    for cohort in cohorts:
        if cohort in cohort_labels:
            cohort_color_mapping[cohort] = cohort_colors[cohort_labels[cohort]]
    
    return create_generic_box_plot_with_trends(
        data_dict=cohort_data,
        metric_config=metric_config,
        display_mode=display_mode,
        toggle_use_p99=toggle_use_p99,
        color_mapping=cohort_color_mapping,
        label_mapping={cohort: f"Cohort {cohort_labels[cohort]}" for cohort in cohorts},
        title_prefix="All Cohorts",
        widget_key_prefix="cohort",
        selected_percentiles=selected_percentiles,
        smoothing_window=smoothing_window
    )


def create_order_type_box_plot(order_type_data, metric_config, display_mode, toggle_use_p99=False, selected_percentiles=[], smoothing_window=1):
    """Create box plots for fill quality across order types"""
    return create_generic_box_plot_with_trends(
        data_dict=order_type_data,
        metric_config=metric_config,
        display_mode=display_mode,
        toggle_use_p99=toggle_use_p99,
        color_mapping=order_type_colors,
        label_mapping=None,  # Will use title case of keys
        title_prefix="All Order Types",
        widget_key_prefix="order_type",
        selected_percentiles=selected_percentiles,
        smoothing_window=smoothing_window
    )


def create_direction_box_plot(direction_data, metric_config, display_mode, toggle_use_p99=False, selected_percentiles=[], smoothing_window=1):
    """Create box plots for fill quality across order directions"""
    return create_generic_box_plot_with_trends(
        data_dict=direction_data,
        metric_config=metric_config,
        display_mode=display_mode,
        toggle_use_p99=toggle_use_p99,
        color_mapping=direction_colors,
        label_mapping=None,  # Will use title case of keys
        title_prefix="All Directions",
        widget_key_prefix="direction",
        selected_percentiles=selected_percentiles,
        smoothing_window=smoothing_window
    )


def create_swift_box_plot(swift_data, metric_config, display_mode, toggle_use_p99=False, selected_percentiles=[], smoothing_window=1):
    """Create box plots for fill quality across swift flags"""
    swift_labels = {'0': 'Non-Swift', '1': 'Swift'}
    return create_generic_box_plot_with_trends(
        data_dict=swift_data,
        metric_config=metric_config,
        display_mode=display_mode,
        toggle_use_p99=toggle_use_p99,
        color_mapping=bit_flag_colors,
        label_mapping=swift_labels,
        title_prefix="All Swift Types",
        widget_key_prefix="swift",
        selected_percentiles=selected_percentiles,
        smoothing_window=smoothing_window
    )


def create_cohort_count_comparison_chart(cohort_data, metric_config):
    """Create combined count comparison chart across cohorts"""
    fig = go.Figure()
    
    base_col = metric_config["base_col"]
    col_name = f"{base_col}Count"
    
    for cohort, df in cohort_data.items():
        if col_name in df.columns:
            cohort_label = cohort_labels[cohort]
            cohort_color = cohort_colors[cohort_label]
            
            fig.add_trace(go.Scatter(
                x=df['datetime'],
                y=df[col_name],
                mode='lines+markers',
                name=f"Cohort {cohort_label}",
                line=dict(color=cohort_color, width=2),
                marker=dict(size=4),
                hovertemplate=f'<b>Cohort {cohort_label}</b><br>Date: %{{x}}<br>Count: %{{y:,.0f}}<extra></extra>'
            ))
    
    fig.update_layout(
        title=f"Cohort Comparison: {metric_config['title']} (Count)",
        xaxis_title="Date",
        yaxis_title="Number of Fills",
        hovermode='x unified',
        height=500
    )
    
    return fig


def create_liquidity_source_chart(df_processed, available_sources, source_config, grouping_mode, volume_units, liquidity_display_mode, title_prefix=""):
    """Create stacked area chart for liquidity sources"""
    if df_processed is None or available_sources is None:
        return None
    
    # Determine if we're showing percentages or absolute values
    use_percentages = liquidity_display_mode == "Percentage Values"
    
    # Determine formatting based on volume units and display mode
    if use_percentages:
        value_format = "{:.1f}%"
        axis_format = ".1f"
        hover_value_format = "{:.1f}%"
        y_axis_title = "Percentage (%)"
        hover_total_format = "%{y:.1f}%"
        data_suffix = "_pct"
    elif volume_units == "Counts":
        value_format = "{:,.0f}"
        axis_format = ",.0f"
        hover_value_format = "{:,.0f}"
        y_axis_title = "Count"
        hover_total_format = "%{y:,.0f}"
        data_suffix = ""
    else:
        value_format = "${:,.0f}"
        axis_format = "$,.0f"
        hover_value_format = "${:,.0f}"
        y_axis_title = "Volume ($)"
        hover_total_format = "$%{y:,.0f}"
        data_suffix = ""
    
    fig = go.Figure()
    
    # Add stacked area traces
    for col in available_sources:
        config = source_config.get(col, {"color": "rgba(128, 128, 128, 0.8)", "name": col})
        
        # Determine which data to use based on display mode
        if use_percentages:
            y_data = df_processed[f'{col}_pct']
            # For percentages, we want to show both the percentage and the absolute value in hover
            if volume_units == "Counts":
                abs_format = "{:,.0f}"
            else:
                abs_format = "${:,.0f}"
            
            hover_text = [
                f"<b>{config['name']}</b><br>" +
                f"Percentage: {pct:.1f}%<br>" +
                f"Volume: {abs_format.format(abs_val)}<br>"
                for abs_val, pct in zip(
                    df_processed[col],
                    df_processed[f'{col}_pct']
                )
            ]
        else:
            y_data = df_processed[col]
            # Create custom hover text for absolute values
            hover_text = [
                f"<b>{config['name']}</b><br>" +
                f"Volume: {hover_value_format.format(abs_val)}<br>" +
                f"Percentage: {pct:.1f}%<br>"
                for abs_val, pct in zip(
                    df_processed[col],
                    df_processed[f'{col}_pct']
                )
            ]
        
        fig.add_trace(go.Scatter(
            x=df_processed['datetime'],
            y=y_data,
            fill='tonexty',
            mode='none',
            name=config['name'],
            fillcolor=config['color'],
            line=dict(color=config['color'].replace('0.8', '1.0'), width=0),
            hovertemplate='%{text}<extra></extra>',
            text=hover_text,
            stackgroup='one'
        ))
    
    # Add total line only for absolute values mode
    if not use_percentages:
        fig.add_trace(go.Scatter(
            x=df_processed['datetime'],
            y=df_processed['total_volume'],
            mode='lines+markers',
            name='Total Volume',
            line=dict(color='rgba(75, 0, 130, 0.9)', width=3, dash='solid'),
            marker=dict(size=6, color='rgba(75, 0, 130, 0.9)'),
            hovertemplate=f'<b>Total Volume</b><br>Date: %{{x}}<br>Volume: {hover_total_format}<extra></extra>',
        ))
    
    # Update title to reflect display mode
    display_mode_text = "Percentage" if use_percentages else "Absolute"
    title = f"Liquidity Sources Breakdown ({grouping_mode} - {volume_units} - {display_mode_text})"
    if title_prefix:
        title = f"{title_prefix}: {title}"
    
    fig.update_layout(
        title=title,
        xaxis_title="Date",
        yaxis_title=y_axis_title,
        yaxis=dict(tickformat=axis_format),
        hovermode='x unified',
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        height=400
    )
    
    # For percentage mode, ensure y-axis goes from 0 to 100
    if use_percentages:
        fig.update_layout(yaxis=dict(range=[0, 100], tickformat=".1f", ticksuffix="%"))
    
    return fig
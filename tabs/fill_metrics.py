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
    fetch_fill_metrics_data_dynamodb
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

# Cache the data fetching function to prevent unnecessary API calls
@st.cache_data(ttl=300)  # Cache for 5 minutes
def cached_fetch_fill_metrics_data(start_ts, end_ts, selected_market, cohort, taker_order_type, taker_order_direction, bit_flag):
    """Cached wrapper for fetch_fill_metrics_data_dynamodb"""
    return fetch_fill_metrics_data_dynamodb(
        start_ts, end_ts, selected_market, 
        cohort, taker_order_type, 
        taker_order_direction, bit_flag
    )

# Cache the data processing function
@st.cache_data
def cached_process_fill_metrics_data(data_hash, data):
    """Cached wrapper for process_fill_metrics_data"""
    return process_fill_metrics_data(data)

def get_data_hash(data):
    """Generate a hash for the data to use as cache key"""
    if data is None or data.empty:
        return "empty"
    return str(hash(str(data.values.tobytes()) + str(data.columns.tolist())))

async def fill_metrics_analysis(clearinghouse: DriftClient):
    st.write("# Auction Fill Analysis")
    st.write(
        "Analyze various fill quality metrics "
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

    # Metric selection
    col1, col2 = st.columns(2)
    with col1:
        st.write("### Primary Metric")
        metric_type = st.radio(
            "Choose primary metric to analyze:",
            options=["Auction Progress", "Fill vs Oracle ($)", "Fill vs Oracle (BPS)"],
            index=0,
            help="""
            - **Auction Progress**: How far into the auction until an order was fully filled (0-1, where 1 = end of auction)
            - **Fill vs Oracle (Absolute)**: Absolute difference between fill price and oracle price
            - **Fill vs Oracle (BPS)**: Basis points difference between fill price and oracle price
            """
        )

    with col2:
        st.write("### Display Units")
        display_mode = st.radio(
            "Choose display units:",
            options=["Percentiles", "Count"],
            index=0,
            help="""
            - **Percentiles**: Show P10, P25, P50, P75, P99 distributions
            - **Count**: Show number of fills
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
        render_fill_metrics_overview(start_ts, end_ts, selected_market, metric_type, display_mode)
    
    with tab_cohort:
        render_fill_metrics_by_cohort(start_ts, end_ts, selected_market, metric_type, display_mode)
    
    with tab_order_type:
        render_fill_metrics_by_order_type(start_ts, end_ts, selected_market, metric_type, display_mode)
    
    with tab_direction:
        render_fill_metrics_by_direction(start_ts, end_ts, selected_market, metric_type, display_mode)
    
    with tab_swift:
        render_fill_metrics_by_swift(start_ts, end_ts, selected_market, metric_type, display_mode)


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
    elif metric_type == "Fill vs Oracle (Absolute)":
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


def render_fill_metrics_overview(start_ts, end_ts, selected_market, metric_type, display_mode):
    """Render overview of fill metrics across all segments"""
    st.write("## Overall Fill Metrics Summary")
    
    # Fetch aggregated data (all cohorts, order types, etc.)
    fill_metrics_data = cached_fetch_fill_metrics_data(
        start_ts, end_ts, selected_market, 
        cohort="all", taker_order_type="all", 
        taker_order_direction="all", bit_flag="all"
    )
    
    if fill_metrics_data is None or fill_metrics_data.empty:
        st.warning("No fill metrics data available for the selected period.")
        return
    
    # Process the data
    df_processed = cached_process_fill_metrics_data(get_data_hash(fill_metrics_data), fill_metrics_data)
    
    if df_processed.empty:
        st.warning("No valid fill metrics data found after processing.")
        return
    
    # Get metric configuration
    metric_config = get_metric_config(metric_type)
    
    # Display summary metrics
    display_summary_metrics(df_processed, metric_config)
    
    # Create time series chart
    fig = create_fill_metrics_timeseries(df_processed, metric_config, display_mode, "Overall")
    st.plotly_chart(fig, use_container_width=True)
    
    # Show distribution chart
    if display_mode == "Percentiles":
        fig_dist = create_fill_metrics_distribution(df_processed, metric_config, "Overall")
        st.plotly_chart(fig_dist, use_container_width=True)


def render_fill_metrics_by_cohort(start_ts, end_ts, selected_market, metric_type, display_mode):
    """Render fill metrics analysis by order size cohort"""
    st.write("## Fill Metrics by Order Size Cohort")
    
    # Fetch data for each cohort
    cohort_data = {}
    for cohort in cohorts:
        data = cached_fetch_fill_metrics_data(
            start_ts, end_ts, selected_market,
            cohort=cohort, taker_order_type="all",
            taker_order_direction="all", bit_flag="all"
        )
        if data is not None and not data.empty:
            cohort_data[cohort] = cached_process_fill_metrics_data(get_data_hash(data), data)
    
    if not cohort_data:
        st.warning("No fill metrics data available for any cohort in the selected period.")
        return
    
    # Get metric configuration
    metric_config = get_metric_config(metric_type)
    
    # Cohort selection for analysis
    selected_cohort_data = {cohort: cohort_data[cohort] for cohort in cohorts if cohort in cohort_data}
    
    if not selected_cohort_data:
        st.warning("No data available for the selected cohorts.")
        return
    
    if display_mode == "Percentiles":
        col1, col2, col3 = st.columns(3)

        with col1:
            toggle_use_p99 = st.toggle(
                "Use P99 instead of true maximum for box plots",
                value=True,
                key=f"{metric_config['base_col']}_use_p99",
            )
        
        with col2:
            trend_line_options = {
                "All": [f"{metric_config['base_col']}P25", f"{metric_config['base_col']}P50", f"{metric_config['base_col']}P99", f"{metric_config['base_col']}Avg"],
                "P25 only": [f"{metric_config['base_col']}P25"],
                "P50 (Median) only": [f"{metric_config['base_col']}P50"], 
                "P99 only": [f"{metric_config['base_col']}P99"],
                "Average only": [f"{metric_config['base_col']}Avg"],
                "None": []
            }
            
            selected_trend_option = st.selectbox(
                "Select trend lines to display:",
                options=list(trend_line_options.keys()),
                index=2,
                key=f"cohort_{metric_config['base_col']}_trend_select"
            )
        
        with col3:
            smoothing_window = st.selectbox(
                "Trend line smoothing:",
                options=[("No smoothing", 1), ("Light (3-day)", 3), ("Medium (7-day)", 7), ("Heavy (14-day)", 14)],
                index=1,
                key=f"cohort_{metric_config['base_col']}_smoothing_select",
                format_func=lambda x: x[0]
            )[1]
        
        selected_percentiles = trend_line_options[selected_trend_option]
        
        st.write("### Daily Distribution Visualization (Box Plots with Trend Lines)")
        fig_box = create_fill_metrics_box_plot(selected_cohort_data, metric_config, display_mode, toggle_use_p99, selected_percentiles, smoothing_window)
        if fig_box:
            st.plotly_chart(fig_box, use_container_width=True)
    
    elif display_mode == "Count":
        st.write("### Combined Cohort Count Analysis")
        fig_combined = create_cohort_count_comparison_chart(selected_cohort_data, metric_config)
        st.plotly_chart(fig_combined, use_container_width=True)


@st.fragment
def render_fill_metrics_by_order_type(start_ts, end_ts, selected_market, metric_type, display_mode):
    """Render fill metrics analysis by order type"""
    st.write("## Fill Metrics by Order Type")
    
    order_types = ['market', 'oracle']#, 'limit', 'triggerMarket', 'triggerLimit']
    
    # Fetch data for each order type
    order_type_data = {}
    for order_type in order_types:
        data = cached_fetch_fill_metrics_data(
            start_ts, end_ts, selected_market,
            cohort="all", taker_order_type=order_type,
            taker_order_direction="all", bit_flag="all"
        )
        if data is not None and not data.empty:
            order_type_data[order_type] = cached_process_fill_metrics_data(get_data_hash(data), data)
    
    if not order_type_data:
        st.warning("No fill metrics data available for any order type in the selected period.")
        return
    
    # Get metric configuration
    metric_config = get_metric_config(metric_type)
    
    if display_mode == "Percentiles":
        col1, col2, col3 = st.columns(3)
        
        with col1:
            toggle_use_p99 = st.toggle(
                "Use P99 instead of true maximum for box plots",
                value=True,
                key=f"{metric_config['base_col']}_order_type_use_p99",
            )
        
        with col2:
            trend_line_options = {
                "All": [f"{metric_config['base_col']}P25", f"{metric_config['base_col']}P50", f"{metric_config['base_col']}P99", f"{metric_config['base_col']}Avg"],
                "P25 only": [f"{metric_config['base_col']}P25"],
                "P50 (Median) only": [f"{metric_config['base_col']}P50"], 
                "P99 only": [f"{metric_config['base_col']}P99"],
                "Average only": [f"{metric_config['base_col']}Avg"],
                "None": []
            }
            
            selected_trend_option = st.selectbox(
                "Select trend lines to display:",
                options=list(trend_line_options.keys()),
                index=2,
                key=f"order_type_{metric_config['base_col']}_trend_select"
            )
        
        with col3:
            smoothing_window = st.selectbox(
                "Trend line smoothing:",
                options=[("No smoothing", 1), ("Light (3-day)", 3), ("Medium (7-day)", 7), ("Heavy (14-day)", 14)],
                index=1,
                key=f"order_type_{metric_config['base_col']}_smoothing_select",
                format_func=lambda x: x[0]
            )[1]
        
        selected_percentiles = trend_line_options[selected_trend_option]
        
        st.write("### Daily Distribution Visualization (Box Plots with Trend Lines)")
        fig_box = create_order_type_box_plot(order_type_data, metric_config, display_mode, toggle_use_p99, selected_percentiles, smoothing_window)
        if fig_box:
            st.plotly_chart(fig_box, use_container_width=True)
    
    elif display_mode == "Count":
        st.write("### Order Type Count Comparison")
        fig_comparison = create_order_type_comparison_chart(order_type_data, metric_config, display_mode)
        st.plotly_chart(fig_comparison, use_container_width=True)


@st.fragment
def render_fill_metrics_by_direction(start_ts, end_ts, selected_market, metric_type, display_mode):
    """Render fill metrics analysis by order direction"""
    st.write("## Fill Metrics by Order Direction")
    
    # Fetch data for each direction
    direction_data = {}
    for direction in ['long', 'short']:
        data = cached_fetch_fill_metrics_data(
            start_ts, end_ts, selected_market,
            cohort="all", taker_order_type="all",
            taker_order_direction=direction, bit_flag="all"
        )
        if data is not None and not data.empty:
            direction_data[direction] = cached_process_fill_metrics_data(get_data_hash(data), data)
    
    if not direction_data:
        st.warning("No fill metrics data available for any direction in the selected period.")
        return
    
    # Get metric configuration
    metric_config = get_metric_config(metric_type)
    
    if display_mode == "Percentiles":
        col1, col2, col3 = st.columns(3)
        
        with col1:
            toggle_use_p99 = st.toggle(
                "Use P99 instead of true maximum for box plots",
                value=True,
                key=f"{metric_config['base_col']}_direction_use_p99",
            )
        
        with col2:
            trend_line_options = {
                "All": [f"{metric_config['base_col']}P25", f"{metric_config['base_col']}P50", f"{metric_config['base_col']}P99", f"{metric_config['base_col']}Avg"],
                "P25 only": [f"{metric_config['base_col']}P25"],
                "P50 (Median) only": [f"{metric_config['base_col']}P50"], 
                "P99 only": [f"{metric_config['base_col']}P99"],
                "Average only": [f"{metric_config['base_col']}Avg"],
                "None": []
            }
            
            selected_trend_option = st.selectbox(
                "Select trend lines to display:",
                options=list(trend_line_options.keys()),
                index=2,
                key=f"direction_{metric_config['base_col']}_trend_select"
            )
        
        with col3:
            smoothing_window = st.selectbox(
                "Trend line smoothing:",
                options=[("No smoothing", 1), ("Light (3-day)", 3), ("Medium (7-day)", 7), ("Heavy (14-day)", 14)],
                index=1,
                key=f"direction_{metric_config['base_col']}_smoothing_select",
                format_func=lambda x: x[0]
            )[1]
        
        selected_percentiles = trend_line_options[selected_trend_option]
        
        st.write("### Daily Distribution Visualization (Box Plots with Trend Lines)")
        fig_box = create_direction_box_plot(direction_data, metric_config, display_mode, toggle_use_p99, selected_percentiles, smoothing_window)
        if fig_box:
            st.plotly_chart(fig_box, use_container_width=True)
    
    elif display_mode == "Count":
        st.write("### Direction Count Comparison")
        fig_comparison = create_direction_comparison_chart(direction_data, metric_config, display_mode)
        st.plotly_chart(fig_comparison, use_container_width=True)


@st.fragment
def render_fill_metrics_by_swift(start_ts, end_ts, selected_market, metric_type, display_mode):
    """Render fill metrics analysis by Swift flag"""
    st.write("## Fill Metrics by Swift Flag")
    
    swift_types = ['0', '1']
    swift_labels = {'0': 'Non-Swift', '1': 'Swift'}
    # selected_swift_types = st.multiselect("Select Swift Types to Analyze", swift_types, default=swift_types, format_func=lambda x: swift_labels[x])
    selected_swift_types = swift_types
    
    # Fetch data for each swift type
    swift_data = {}
    for swift_type in selected_swift_types:
        data = cached_fetch_fill_metrics_data(
            start_ts, end_ts, selected_market,
            cohort="all", taker_order_type="all",
            taker_order_direction="all", bit_flag=swift_type
        )
        if data is not None and not data.empty:
            swift_data[swift_type] = cached_process_fill_metrics_data(get_data_hash(data), data)
    
    if not swift_data:
        st.warning("No fill metrics data available for any swift type in the selected period.")
        return
    
    # Get metric configuration
    metric_config = get_metric_config(metric_type)
    
    if display_mode == "Percentiles":
        # Add controls in columns
        col1, col2, col3 = st.columns(3)
        
        with col1:
            toggle_use_p99 = st.toggle(
                "Use P99 instead of true maximum for box plots",
                value=True,
                key=f"{metric_config['base_col']}_swift_use_p99",
            )
        
        with col2:
            trend_line_options = {
                "All": [f"{metric_config['base_col']}P25", f"{metric_config['base_col']}P50", f"{metric_config['base_col']}P99", f"{metric_config['base_col']}Avg"],
                "P25 only": [f"{metric_config['base_col']}P25"],
                "P50 (Median) only": [f"{metric_config['base_col']}P50"], 
                "P99 only": [f"{metric_config['base_col']}P99"],
                "Average only": [f"{metric_config['base_col']}Avg"],
                "None": []
            }
            
            selected_trend_option = st.selectbox(
                "Select trend lines to display:",
                options=list(trend_line_options.keys()),
                index=2,
                key=f"swift_{metric_config['base_col']}_trend_select"
            )
        
        with col3:
            smoothing_window = st.selectbox(
                "Trend line smoothing:",
                options=[("No smoothing", 1), ("Light (3-day)", 3), ("Medium (7-day)", 7), ("Heavy (14-day)", 14)],
                index=1,
                key=f"swift_{metric_config['base_col']}_smoothing_select",
                format_func=lambda x: x[0]
            )[1]
        
        selected_percentiles = trend_line_options[selected_trend_option]
        
        st.write("### Daily Distribution Visualization (Box Plots with Trend Lines)")
        fig_box = create_swift_box_plot(swift_data, metric_config, display_mode, toggle_use_p99, selected_percentiles, smoothing_window)
        if fig_box:
            st.plotly_chart(fig_box, use_container_width=True)
    
    else:
        # Create comparison chart for other display modes
        fig_comparison = create_swift_comparison_chart(swift_data, metric_config, display_mode)
        st.plotly_chart(fig_comparison, use_container_width=True)


def process_fill_metrics_data(data):
    """Process raw fill metrics data from DynamoDB"""
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


def create_fill_metrics_timeseries(df, metric_config, display_mode, title_prefix):
    """Create time series chart for fill metrics"""
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


def create_fill_metrics_distribution(df, metric_config, title_prefix):
    """Create distribution box plot for fill metrics"""
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


def create_fill_metrics_box_plot(cohort_data, metric_config, display_mode, toggle_use_p99=False, selected_percentiles=[], smoothing_window=1):
    """Create box plots for fill metrics across cohorts"""
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
    """Create box plots for fill metrics across order types"""
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
    """Create box plots for fill metrics across order directions"""
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
    """Create box plots for fill metrics across swift flags"""
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
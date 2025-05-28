from datetime import datetime as dt
from datetime import timedelta

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st
from driftpy.drift_client import (
    DriftClient,
)
from plotly.subplots import make_subplots

from constants import get_all_markets


async def fill_speed_analysis(clearinghouse: DriftClient):
    st.write("# Fill Speed Analysis")
    st.write(
        "Analyze order fill times and speeds across different markets and conditions"
    )

    if "date_filter" not in st.session_state:
        st.session_state.date_filter = None
    if "performance_filter" not in st.session_state:
        st.session_state.performance_filter = None
    if "share_y_axes_individual_cohorts" not in st.session_state:
        st.session_state.share_y_axes_individual_cohorts = False

    top_col1, top_col2, top_col3 = st.columns([2, 2, 1])
    with top_col1:
        start_date = st.date_input(
            "Start Date", value=dt.now().date() - timedelta(days=7)
        )
    with top_col2:
        end_date = st.date_input("End Date", value=dt.now().date() - timedelta(days=1))

    market_options = get_all_markets()
    selected_market = st.selectbox(
        "Select Market to Analyze:",
        options=market_options,
        index=0,
    )

    fill_data = await fetch_fill_speed_data(start_date, end_date, selected_market)

    if fill_data is None or fill_data.empty:
        st.error("No fill speed data available for the selected time range and markets")
        return

    fill_data["datetime"] = pd.to_datetime(fill_data["ts"], unit="s")
    filtered_data = apply_filters(fill_data)

    if st.session_state.date_filter or st.session_state.performance_filter:
        filter_info = []
        if st.session_state.date_filter:
            try:
                start_str = pd.to_datetime(st.session_state.date_filter[0]).strftime(
                    "%Y-%m-%d"
                )
                end_str = pd.to_datetime(st.session_state.date_filter[1]).strftime(
                    "%Y-%m-%d"
                )
                filter_info.append(f"Date: {start_str} to {end_str}")
            except Exception:
                filter_info.append("Date filter active")
        if st.session_state.performance_filter:
            filter_info.append(
                f"P90: {st.session_state.performance_filter[0]:.2f}slots to {st.session_state.performance_filter[1]:.2f}slots"
            )

        st.info(
            f"🔍 Active filters: {' | '.join(filter_info)} ({len(filtered_data)}/{len(fill_data)} data points)"
        )

    st.write("## Overall Fill Speed Metrics (All Selected Data)")

    num_days_with_data = 0
    if not filtered_data.empty:
        num_days_with_data = filtered_data["datetime"].dt.normalize().nunique()

    overall_metrics_row1_col1, overall_metrics_row1_col2 = st.columns(2)
    with overall_metrics_row1_col1:
        st.metric("Number of Days with Data", f"{num_days_with_data}")
    with overall_metrics_row1_col2:
        if not filtered_data.empty:
            avg_median = filtered_data["p50"].mean()
            st.metric("Avg Median Fill Time", f"{avg_median:.2f} slots")
        else:
            st.metric("Avg Median Fill Time", "N/A")

    overall_metrics_row2_col1, overall_metrics_row2_col2, overall_metrics_row2_col3 = (
        st.columns(3)
    )
    with overall_metrics_row2_col1:
        if not filtered_data.empty:
            avg_p90 = filtered_data["p90"].mean()
            st.metric("Avg P90 Fill Time", f"{avg_p90:.2f} slots")
        else:
            st.metric("Avg P90 Fill Time", "N/A")
    with overall_metrics_row2_col2:
        if not filtered_data.empty:
            avg_p99 = filtered_data["p99"].mean()
            st.metric("Avg P99 Fill Time", f"{avg_p99:.2f} slots")
        else:
            st.metric("Avg P99 Fill Time", "N/A")
    with overall_metrics_row2_col3:
        if not filtered_data.empty:
            fast_fills_pct = (filtered_data["p80"] < 1.0).mean() * 100
            st.metric("Fast Periods (P80<1 slot)", f"{fast_fills_pct:.1f}%")
        else:
            st.metric("Fast Periods (P80<1 slot)", "N/A")

    cohort_pxx_col1, cohort_pxx_col2 = st.columns([4, 2])
    with cohort_pxx_col1:
        st.write("#### Fill Time by Order Size Cohort (Select Display Mode)")
        percentile_options = [f"p{x}" for x in [10, 20, 30, 40, 50, 60, 70, 80, 90, 99]]
        display_mode_options = ["Distribution Ribbon (P10-P90)"] + percentile_options

        selected_display_mode = st.selectbox(
            "Select Display Mode:",
            options=display_mode_options,
            index=10,  # Default to p99 (1 for ribbon + 9 for p10-p90)
            key="cohort_display_mode_selector",
        )

        fig_all_cohorts_comparison = create_all_cohorts_comparison_chart(
            filtered_data, selected_display_mode
        )

    with cohort_pxx_col2:
        if st.button(
            "🔄 Reset Time Filter",
            key="reset_cohort_comparison_time_filter",
            help="Clear date filter from this chart",
        ):
            st.session_state.date_filter = None
            st.rerun()
        st.write("**Instructions:**")
        st.write("• Drag to select date range (will affect all charts)")
        st.write(
            "• Hit reset to go back to the date range selected in the top controls"
        )

    all_cohorts_selection = st.plotly_chart(
        fig_all_cohorts_comparison,
        use_container_width=True,
        key="all_cohorts_comparison_chart",
        on_select="rerun",
    )
    if all_cohorts_selection and all_cohorts_selection.selection:
        if all_cohorts_selection.selection.points:
            handle_timeseries_selection(
                all_cohorts_selection.selection.points, filtered_data
            )
        elif (
            hasattr(all_cohorts_selection.selection, "box")
            and all_cohorts_selection.selection.box
        ):
            handle_timeseries_box_selection(
                all_cohorts_selection.selection.box, filtered_data
            )
        elif (
            hasattr(all_cohorts_selection.selection, "range")
            and all_cohorts_selection.selection.range
        ):
            handle_timeseries_range_selection(
                all_cohorts_selection.selection.range, filtered_data
            )

    st.write("#### Fill Speed Percentiles by Individual Cohort (Line Charts)")
    st.session_state.share_y_axes_individual_cohorts = st.toggle(
        "🔗 Share Y-Axis Across Cohort Line Charts",
        value=st.session_state.share_y_axes_individual_cohorts,
        key="toggle_share_y_cohort_lines",
    )
    fig_individual_cohorts = create_individual_cohort_subplots(
        filtered_data, share_y_axes=st.session_state.share_y_axes_individual_cohorts
    )
    st.plotly_chart(
        fig_individual_cohorts,
        use_container_width=True,
        key="individual_cohorts_subplots",
    )

    st.write("#### Fill Speed Distribution by Individual Cohort (Heatmaps)")
    cohort_heatmap_col1, cohort_heatmap_col2 = st.columns([4, 1])  # For layout control
    with cohort_heatmap_col2:
        selected_cohort_heatmap_colorscale = st.selectbox(
            "Color Scale for Cohort Heatmaps:",
            options=["Viridis", "Plasma", "Inferno", "RdYlBu_r", "RdBu_r"],
            index=0,
            key="cohort_heatmap_colorscale_selector",
            help="Select a common color scale for the 2x2 cohort heatmaps.",
        )

    with cohort_heatmap_col1:
        fig_individual_cohort_heatmaps = create_individual_cohort_heatmap_subplots(
            filtered_data, selected_cohort_heatmap_colorscale
        )
        st.plotly_chart(
            fig_individual_cohort_heatmaps,
            use_container_width=True,
            key="individual_cohort_heatmaps",
        )

    st.write("#### Key Metrics by Cohort")
    cohort_order = ["0-1k", "1k-10k", "10k-100k", "100k+"]

    cohort_metrics_row1_cols = st.columns(2)
    cohort_metrics_row2_cols = st.columns(2)

    cohort_col_map = {
        0: cohort_metrics_row1_cols[0],
        1: cohort_metrics_row1_cols[1],
        2: cohort_metrics_row2_cols[0],
        3: cohort_metrics_row2_cols[1],
    }

    for i, cohort_name in enumerate(cohort_order):
        with cohort_col_map[i]:
            st.markdown(f"##### Cohort: {cohort_name}")
            cohort_specific_data = filtered_data[filtered_data["cohort"] == cohort_name]
            if not cohort_specific_data.empty:
                num_days_cohort = (
                    cohort_specific_data["datetime"].dt.normalize().nunique()
                )
                avg_median_cohort = cohort_specific_data["p50"].mean()
                avg_p90_cohort = cohort_specific_data["p90"].mean()
                avg_p99_cohort = cohort_specific_data["p99"].mean()
                fast_fills_pct_cohort = (cohort_specific_data["p80"] < 1.0).mean() * 100

                st.metric("Number of Days", f"{num_days_cohort}")
                st.metric("Avg Median", f"{avg_median_cohort:.2f} slots")
                st.metric("Avg P90", f"{avg_p90_cohort:.2f} slots")
                st.metric("Avg P99", f"{avg_p99_cohort:.2f} slots")
                st.metric("Fast P80<1 slot", f"{fast_fills_pct_cohort:.1f}%")
            else:
                st.metric("Number of Days", "N/A")
                st.metric("Avg Median", "N/A")
                st.metric("Avg P90", "N/A")
                st.metric("Avg P99", "N/A")
                st.metric("Fast P80<1 slot", "N/A")
    st.markdown("<br>", unsafe_allow_html=True)  # Add some space

    st.write("## Fill Speed Percentiles Over Time (Combined Average)")

    overall_perf_col1, overall_perf_col2 = st.columns([5, 1])
    with overall_perf_col2:
        if st.button(
            "🔄 Reset Time Filter",
            key="reset_overall_timeseries",
            help="Clear date range filter from this chart",
        ):
            st.session_state.date_filter = None
            st.rerun()
        st.write("**Instructions:**")
        st.write("• Drag to select date range")
        st.write("• Double-click to zoom")

    with overall_perf_col1:
        daily_avg_data = (
            filtered_data.groupby(pd.Grouper(key="datetime", freq="D"))[
                ["p10", "p20", "p30", "p40", "p50", "p60", "p70", "p80", "p90", "p99"]
            ]
            .mean()
            .reset_index()
        )

        fig_timeseries = create_timeseries_chart(daily_avg_data)  # Pass aggregated data
        timeseries_selection = st.plotly_chart(
            fig_timeseries,
            use_container_width=True,
            key="timeseries_chart",
            on_select="rerun",
        )

    if timeseries_selection and timeseries_selection.selection:
        with st.expander("Debug: Overall Selection Data", expanded=False):
            st.write("Selection object:", timeseries_selection.selection)

        if timeseries_selection.selection.points:
            handle_timeseries_selection(
                timeseries_selection.selection.points, daily_avg_data
            )
        elif (
            hasattr(timeseries_selection.selection, "box")
            and timeseries_selection.selection.box
        ):
            handle_timeseries_box_selection(
                timeseries_selection.selection.box, daily_avg_data
            )
        elif (
            hasattr(timeseries_selection.selection, "range")
            and timeseries_selection.selection.range
        ):
            handle_timeseries_range_selection(
                timeseries_selection.selection.range, daily_avg_data
            )

    st.write("## Daily Average Fill Speed Distribution Heatmap")
    heatmap_col1, heatmap_col2, heatmap_col3 = st.columns([4, 1, 1])
    with heatmap_col2:
        color_scale = st.selectbox(
            "Heatmap Color Scale:",
            options=["Viridis", "Plasma", "Inferno", "RdYlBu_r", "RdBu_r"],
            index=0,
            key="heatmap_color_scale",
            help="Select color scale for the heatmap",
        )
        st.write("**Heatmap shows (daily avg across cohorts):**")
        st.write("• Y-axis: Avg Percentiles")
        st.write("• X-axis: Time (dates)")
        st.write("• Color: Fill time values (slots)")
    with heatmap_col3:
        if st.button(
            "🔄 Reset Heatmap Filter",
            key="reset_heatmap_time_filter",
            help="Clear date filter from heatmap interaction",
        ):
            st.session_state.date_filter = None
            st.rerun()

    with heatmap_col1:
        heatmap_display_data = create_heatmap_data(filtered_data)
        fig_heatmap = create_interactive_heatmap(
            heatmap_display_data, None, color_scale
        )
        heatmap_selection = st.plotly_chart(
            fig_heatmap,
            use_container_width=True,
            key="heatmap_chart_main",
            on_select="rerun",
        )
        if heatmap_selection and heatmap_selection.selection:
            if heatmap_selection.selection.points:
                handle_heatmap_selection(
                    heatmap_selection.selection.points, heatmap_display_data
                )


def apply_filters(data):
    """Apply session state filters to the data"""
    filtered = data.copy()

    if st.session_state.date_filter:
        start_date, end_date = st.session_state.date_filter
        filtered = filtered[
            (filtered["datetime"] >= start_date) & (filtered["datetime"] <= end_date)
        ]

    if st.session_state.performance_filter:
        min_p90, max_p90 = st.session_state.performance_filter
        filtered = filtered[(filtered["p90"] >= min_p90) & (filtered["p90"] <= max_p90)]

    return filtered


def create_heatmap_data(data):
    data = data.copy()
    data["date"] = data["datetime"].dt.date

    percentile_cols = [
        "p10",
        "p20",
        "p30",
        "p40",
        "p50",
        "p60",
        "p70",
        "p80",
        "p90",
        "p99",
    ]

    heatmap_df = data.groupby("date")[percentile_cols].mean().reset_index()

    heatmap_df["datetime_temp"] = pd.to_datetime(heatmap_df["date"])
    heatmap_df["day_of_week"] = heatmap_df["datetime_temp"].dt.day_name()

    return heatmap_df


def create_interactive_heatmap(data, metric, color_scale):
    """Create interactive heatmap with percentiles on Y-axis and time on X-axis
    Assumes input `data` is already aggregated daily with necessary percentile columns.
    """
    # `data` is now heatmap_df, which is pre-aggregated daily
    percentile_cols = [
        "p10",
        "p20",
        "p30",
        "p40",
        "p50",
        "p60",
        "p70",
        "p80",
        "p90",
        "p99",
    ]

    # Create matrix for heatmap
    # Dates are already unique in heatmap_df from groupby('date')
    dates = sorted(data["date"].unique())

    # Build the Z matrix (percentiles x dates)
    z_matrix = []
    for percentile in percentile_cols:
        # For each percentile, get the row from heatmap_df that corresponds to it
        # Since heatmap_df is date-indexed, and columns are p10,p20..p99, we can directly use them
        row_values = data.set_index("date")[percentile].reindex(dates).values
        z_matrix.append(row_values)

    # Create custom hover text
    hover_text = []
    for i, percentile_label in enumerate(percentile_cols):
        hover_row = []
        for j, date_val in enumerate(dates):
            value = z_matrix[i][j]
            if not np.isnan(value):
                hover_row.append(
                    f"Date: {date_val}<br>Avg Percentile: {percentile_label.upper()}<br>Fill Time: {value:.3f} slots"
                )
            else:
                hover_row.append(
                    f"Date: {date_val}<br>Avg Percentile: {percentile_label.upper()}<br>No data"
                )
        hover_text.append(hover_row)

    fig = go.Figure(
        data=go.Heatmap(
            z=z_matrix,
            x=[str(date) for date in dates],
            y=[p.upper() for p in percentile_cols],
            colorscale=color_scale,
            hoverongaps=False,
            hovertemplate="%{text}<extra></extra>",
            text=hover_text,
            colorbar=dict(title="Avg Fill Time (slots)"),
        )
    )

    fig.update_layout(
        title="Daily Average Fill Speed Distribution Heatmap (All Cohorts)",
        xaxis_title="Date",
        yaxis_title="Average Percentile (All Cohorts)",
        height=500,
        xaxis=dict(tickangle=45),
        dragmode="select",
        selectdirection="h",
    )

    return fig


def create_timeseries_chart(data):
    """Create interactive timeseries chart"""
    fig = go.Figure()

    percentiles = ["p10", "p50", "p90", "p99"]
    colors = ["green", "blue", "orange", "red"]

    for p, color in zip(percentiles, colors):
        fig.add_trace(
            go.Scatter(
                x=data["datetime"],
                y=data[p],
                mode="lines+markers",
                name=f"{p.upper()} Fill Time",
                line=dict(color=color),
                opacity=0.8,
                marker=dict(size=4),
            )
        )

    fig.update_layout(
        title="Fill Speed Percentiles Over Time (Drag to select date range)",
        xaxis_title="Time",
        yaxis_title="Fill Time (slots)",
        height=500,
        hovermode="x unified",
        dragmode="select",
        selectdirection="h",
    )

    return fig


def create_box_plot(data):
    """Create interactive box plot"""
    fig = go.Figure()

    percentiles = ["p10", "p50", "p90", "p99"]

    for p in percentiles:
        fig.add_trace(
            go.Box(
                y=data[p],
                name=p.upper(),
                boxpoints="outliers",
                marker=dict(size=4),
                line=dict(width=2),
            )
        )

    fig.update_layout(
        title="Fill Time Distribution by Percentile (Click outliers to filter)",
        yaxis_title="Fill Time (slots)",
        height=400,
        dragmode="select",
    )

    return fig


def create_trends_chart(data):
    """Create performance trends chart"""
    window = min(7, len(data) // 4)  # 7 day window or 1/4 of data

    if window > 1:
        data = data.copy()
        data["p50_rolling"] = data["p50"].rolling(window=window, center=True).mean()
        data["p90_rolling"] = data["p90"].rolling(window=window, center=True).mean()

        fig = make_subplots(
            rows=2,
            cols=1,
            subplot_titles=("Median Fill Time Trend", "P90 Fill Time Trend"),
            vertical_spacing=0.1,
        )

        # Median trend
        fig.add_trace(
            go.Scatter(
                x=data["datetime"],
                y=data["p50"],
                mode="markers",
                name="P50 Raw",
                opacity=0.3,
                marker=dict(size=3),
            ),
            row=1,
            col=1,
        )

        fig.add_trace(
            go.Scatter(
                x=data["datetime"],
                y=data["p50_rolling"],
                mode="lines",
                name="P50 Trend",
                line=dict(width=3),
            ),
            row=1,
            col=1,
        )

        # P90 trend
        fig.add_trace(
            go.Scatter(
                x=data["datetime"],
                y=data["p90"],
                mode="markers",
                name="P90 Raw",
                opacity=0.3,
                marker=dict(size=3),
            ),
            row=2,
            col=1,
        )

        fig.add_trace(
            go.Scatter(
                x=data["datetime"],
                y=data["p90_rolling"],
                mode="lines",
                name="P90 Trend",
                line=dict(width=3),
            ),
            row=2,
            col=1,
        )

        fig.update_layout(height=600, showlegend=True)
        fig.update_yaxes(title_text="Fill Time (slots)", row=1, col=1)
        fig.update_yaxes(title_text="Fill Time (slots)", row=2, col=1)
        fig.update_xaxes(title_text="Time", row=2, col=1)

        return fig
    else:
        # Fallback for small datasets
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=data["datetime"], y=data["p50"], mode="lines+markers", name="P50"
            )
        )
        fig.update_layout(title="Fill Time Trend", height=400)
        return fig


def handle_heatmap_selection(points, data):
    """Handle heatmap selection for filtering"""
    if not points:
        return

    try:
        # Extract selected dates from heatmap
        selected_dates = []
        for point in points:
            if isinstance(point, dict) and "x" in point:
                date_str = point["x"]
                if date_str:
                    selected_dates.append(pd.to_datetime(date_str).date())

        if selected_dates:
            min_date = min(selected_dates)
            max_date = max(selected_dates)
            st.session_state.date_filter = (
                pd.Timestamp(min_date),
                pd.Timestamp(max_date),
            )
            st.rerun()
    except Exception as e:
        st.error(f"Error processing heatmap selection: {e}")


def handle_timeseries_selection(points, data):
    """Handle timeseries selection for filtering"""
    if not points:
        return

    try:
        # Extract selected date range
        selected_dates = []
        for point in points:
            if isinstance(point, dict) and "x" in point:
                date_str = point["x"]
                if date_str:
                    selected_dates.append(pd.to_datetime(date_str))

        if selected_dates:
            min_date = min(selected_dates)
            max_date = max(selected_dates)
            st.session_state.date_filter = (min_date, max_date)
            st.rerun()
    except Exception as e:
        st.error(f"Error processing timeseries selection: {e}")


def handle_box_selection(points, data):
    """Handle box plot selection for filtering"""
    if not points:
        return

    try:
        # Extract selected performance range
        selected_values = []
        for point in points:
            if isinstance(point, dict) and "y" in point:
                value = point["y"]
                if value is not None:
                    selected_values.append(float(value))

        if selected_values:
            min_val = min(selected_values)
            max_val = max(selected_values)
            st.session_state.performance_filter = (min_val, max_val)
            st.rerun()
    except Exception as e:
        st.error(f"Error processing box plot selection: {e}")


def handle_timeseries_box_selection(box_data, data):
    """Handle timeseries box selection for date range filtering"""
    if not box_data:
        return

    try:
        if hasattr(box_data, "x") and len(box_data.x) >= 2:
            x_range = box_data.x
            min_date = pd.to_datetime(min(x_range))
            max_date = pd.to_datetime(max(x_range))
            st.session_state.date_filter = (min_date, max_date)
            st.rerun()
        elif hasattr(box_data, "range") and "x" in box_data.range:
            x_range = box_data.range["x"]
            min_date = pd.to_datetime(x_range[0])
            max_date = pd.to_datetime(x_range[1])
            st.session_state.date_filter = (min_date, max_date)
            st.rerun()
    except Exception as e:
        st.error(f"Error processing box selection: {e}")


def handle_timeseries_range_selection(range_data, data):
    """Handle timeseries range selection for date range filtering"""
    if not range_data:
        return

    try:
        # Range selection provides x0, x1 coordinates
        if hasattr(range_data, "x") and len(range_data.x) >= 2:
            x_range = range_data.x
            min_date = pd.to_datetime(min(x_range))
            max_date = pd.to_datetime(max(x_range))
            st.session_state.date_filter = (min_date, max_date)
            st.rerun()
        elif hasattr(range_data, "range") and "x" in range_data.range:
            x_range = range_data.range["x"]
            min_date = pd.to_datetime(x_range[0])
            max_date = pd.to_datetime(x_range[1])
            st.session_state.date_filter = (min_date, max_date)
            st.rerun()
    except Exception as e:
        st.error(f"Error processing range selection: {e}")


@st.cache_data(ttl=600)  # Cache for 10 minutes
def _fetch_single_cohort_data(api_url: str, params: dict, cohort_display_name: str):
    """Helper function to fetch data for a single cohort from the API. Cached."""
    import urllib.parse

    url = f"{api_url}?{urllib.parse.urlencode(params)}"
    print(url)
    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        api_data = response.json()
        print(len(api_data["records"]))

        if not api_data.get("success") or not api_data.get("records"):
            st.warning(
                f"API returned no successful records for cohort {cohort_display_name}."
            )
            return []
        return api_data["records"]
    except requests.exceptions.RequestException as e:
        st.error(f"API request failed for cohort {cohort_display_name}: {e}")
        return []
    except Exception as e:
        st.error(f"Error processing data from API: {e}")
        return pd.DataFrame()


async def fetch_fill_speed_data(start_date, end_date, selected_market):
    """Fetch and process fill speed data from the new API format."""
    api_cohort_to_display_cohort = {
        "0": "0-1k",
        "1000": "1k-10k",
        "10000": "10k-100k",
        "100000": "100k+",
    }
    api_cohort_values_to_query = ["0", "1000", "10000", "100000"]
    percentile_cols = [f"p{p}" for p in [10, 20, 30, 40, 50, 60, 70, 80, 90, 99]]
    all_processed_data = []

    market_to_query = selected_market

    try:
        for cohort_value in api_cohort_values_to_query:
            all_records_for_this_cohort = []
            current_chunk_start_date = start_date

            api_url_base = f"https://data-staging.api.drift.trade/stats/{market_to_query}/analytics/auctionLatency/D/{cohort_value}"
            cohort_display_name = api_cohort_to_display_cohort.get(
                cohort_value, cohort_value
            )

            while current_chunk_start_date <= end_date:
                chunk_end_date = current_chunk_start_date + timedelta(
                    days=99
                )  # Max 100 days per chunk
                if chunk_end_date > end_date:
                    chunk_end_date = end_date

                chunk_start_ts = int(
                    dt.combine(current_chunk_start_date, dt.min.time()).timestamp()
                )
                chunk_end_ts = int(
                    dt.combine(chunk_end_date, dt.max.time()).timestamp()
                )

                params = {
                    "startTs": chunk_start_ts,
                    "endTs": chunk_end_ts,
                    "limit": 100,  # API limit
                }

                chunk_raw_records = _fetch_single_cohort_data(
                    api_url_base, params, cohort_display_name
                )

                if chunk_raw_records:
                    all_records_for_this_cohort.extend(chunk_raw_records)

                current_chunk_start_date = chunk_end_date + timedelta(days=1)

            raw_records = all_records_for_this_cohort

            if not raw_records:
                continue

            for record in raw_records:
                transformed_record = {}
                transformed_record["ts"] = record["ts"]
                transformed_record["datetime"] = pd.to_datetime(
                    record["ts"], unit="s"
                ).replace(hour=12, minute=0, second=0, microsecond=0)
                transformed_record["cohort"] = api_cohort_to_display_cohort.get(
                    str(record["cohort"]), "Unknown"
                )

                for p_col in percentile_cols:
                    transformed_record[p_col] = (
                        float(record[p_col])
                        if record.get(p_col) is not None and record[p_col] != "NaN"
                        else np.nan
                    )
                all_processed_data.append(transformed_record)

        if not all_processed_data:
            st.info(
                "No records returned from the API for the selected criteria after checking all cohorts."
            )
            return pd.DataFrame()

        df = pd.DataFrame(all_processed_data)
        return df

    except (
        Exception
    ) as e:  # General exception handler for fetch_fill_speed_data orchestration
        st.error(f"An unexpected error occurred while fetching or processing data: {e}")
        return pd.DataFrame()


def create_all_cohorts_comparison_chart(data, selection_mode):
    """Create a timeseries chart comparing cohorts: either a selected percentile or a P10-P90 distribution ribbon with P50 line."""
    fig = go.Figure()
    cohort_order = ["0-1k", "1k-10k", "10k-100k", "100k+"]
    try:
        colors = ["#636EFA", "#EF553B", "#00CC96", "#AB63FA", "#FFA15A"]
    except Exception:
        colors = [
            "#1f77b4",
            "#ff7f0e",
            "#2ca02c",
            "#d62728",
            "#9467bd",
        ]

    if selection_mode == "Distribution Ribbon (P10-P90)":
        required_cols = ["p10", "p50", "p90"]
        title_text = "P10-P90 Fill Time Distribution with P50 Median Across Cohorts"
        yaxis_title_text = "Fill Time (slots)"
        for col in required_cols:
            if col not in data.columns:
                st.error(f"Required percentile '{col}' for ribbon not found in data.")
                return fig

        for idx, cohort in enumerate(cohort_order):
            cohort_data = data[data["cohort"] == cohort].sort_values(by="datetime")
            color = colors[idx % len(colors)]
            if not cohort_data.empty:
                fig.add_trace(
                    go.Scatter(
                        x=cohort_data["datetime"],
                        y=cohort_data["p90"],
                        mode="lines",
                        line=dict(width=0.5, color=color),
                        legendgroup=cohort,
                        name=f"P90 - {cohort}",
                        showlegend=False,
                    )
                )
                fig.add_trace(
                    go.Scatter(
                        x=cohort_data["datetime"],
                        y=cohort_data["p10"],
                        mode="lines",
                        line=dict(width=0.5, color=color),
                        fill="tonexty",
                        fillcolor=f"rgba({int(color[1:3], 16)},{int(color[3:5], 16)},{int(color[5:7], 16)},0.2)",
                        legendgroup=cohort,
                        name=f"P10-P90 - {cohort}",
                        showlegend=True,
                    )
                )
                fig.add_trace(
                    go.Scatter(
                        x=cohort_data["datetime"],
                        y=cohort_data["p50"],
                        mode="lines",
                        line=dict(color=color, width=2.5, dash="dash"),
                        name=f"P50 - {cohort}",
                        legendgroup=cohort,
                        showlegend=False,
                    )
                )
    else:
        percentile_to_plot = selection_mode
        if percentile_to_plot not in data.columns:
            st.error(f"Selected percentile '{percentile_to_plot}' not found in data.")
            return fig

        title_text = f"{percentile_to_plot.upper()} Fill Time Across Order Size Cohorts"
        yaxis_title_text = f"{percentile_to_plot.upper()} Fill Time (slots)"

        for idx, cohort in enumerate(cohort_order):
            cohort_data = data[data["cohort"] == cohort].sort_values(by="datetime")
            color = colors[idx % len(colors)]
            if not cohort_data.empty:
                fig.add_trace(
                    go.Scatter(
                        x=cohort_data["datetime"],
                        y=cohort_data[percentile_to_plot],
                        mode="lines+markers",
                        name=f"{percentile_to_plot.upper()} - {cohort}",
                        line=dict(color=color, width=2),
                        marker=dict(size=4),
                        opacity=0.8,
                    )
                )

    fig.update_layout(
        title=title_text + " (Drag to select date range)",
        xaxis_title="Time",
        yaxis_title=yaxis_title_text,
        height=500,
        hovermode="x unified",
        dragmode="select",
        selectdirection="h",
        legend_title_text="Cohort / Distribution",
    )
    return fig


def create_individual_cohort_subplots(data, share_y_axes=False):
    """Create a 2x2 grid of timeseries charts, one for each cohort."""
    cohort_order = ["0-1k", "1k-10k", "10k-100k", "100k+"]
    percentiles_to_plot = [f"p{x}" for x in [10, 20, 30, 40, 50, 60, 70, 80, 90, 99]]
    percentiles_to_plot.reverse()  # p99 will be plotted first

    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=[f"Cohort: {c}" for c in cohort_order],
        shared_xaxes=True,
        shared_yaxes=share_y_axes,
        vertical_spacing=0.1,
    )

    for i, cohort in enumerate(cohort_order):
        cohort_data = data[data["cohort"] == cohort].sort_values(by="datetime")
        row_idx = (i // 2) + 1
        col_idx = (i % 2) + 1

        if not cohort_data.empty:
            for p_col in percentiles_to_plot:
                fig.add_trace(
                    go.Scatter(
                        x=cohort_data["datetime"],
                        y=cohort_data[p_col],
                        name=f"{p_col.upper()} - {cohort}",
                        mode="lines",
                        legendgroup=p_col.upper(),
                        showlegend=(i == 0),
                    ),
                    row=row_idx,
                    col=col_idx,
                )
        fig.update_yaxes(title_text="Fill Time (slots)", row=row_idx, col=col_idx)

    if share_y_axes:
        global_max_val = 0
        if not data.empty:
            all_percentile_values = []
            for p_col in percentiles_to_plot:
                if p_col in data.columns:
                    all_percentile_values.extend(data[p_col].dropna().tolist())

            if all_percentile_values:
                global_max_val = max(all_percentile_values)

        padded_global_max_val = global_max_val * 1.05
        if padded_global_max_val == 0:
            padded_global_max_val = 1

        fig.update_yaxes(range=[0, padded_global_max_val])
    else:
        fig.update_yaxes(autorange=True)

    fig.update_layout(
        title="Fill Speed Percentiles by Order Size Cohort",
        height=700,
        hovermode="x unified",
        dragmode="select",
        selectdirection="h",
        legend_title_text="Percentile",
    )
    return fig


def create_individual_cohort_heatmap_subplots(data, common_color_scale):
    """Create a 2x2 grid of heatmaps, one for each cohort, showing percentile distributions over time."""
    cohort_order = ["0-1k", "1k-10k", "10k-100k", "100k+"]
    percentile_cols = [
        "p10",
        "p20",
        "p30",
        "p40",
        "p50",
        "p60",
        "p70",
        "p80",
        "p90",
        "p99",
    ]

    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=[f"Heatmap - Cohort: {c}" for c in cohort_order],
        shared_xaxes=True,
        shared_yaxes=True,
        vertical_spacing=0.05,
        horizontal_spacing=0.03,
    )

    for i, cohort in enumerate(cohort_order):
        cohort_data = data[data["cohort"] == cohort].copy()
        row_idx = (i // 2) + 1
        col_idx = (i % 2) + 1

        if not cohort_data.empty:
            cohort_data["date"] = cohort_data["datetime"].dt.date
            dates = sorted(cohort_data["date"].unique())

            z_matrix = []
            for p_col in percentile_cols:
                row_values = cohort_data.set_index("date")[p_col].reindex(dates).values
                z_matrix.append(row_values)

            hover_text = []
            for r_idx, percentile_label in enumerate(percentile_cols):
                hover_row = []
                for c_idx, date_val in enumerate(dates):
                    value = z_matrix[r_idx][c_idx]
                    text = f"Date: {date_val}<br>Cohort: {cohort}<br>Percentile: {percentile_label.upper()}<br>"
                    text += (
                        f"Fill Time: {value:.3f} slots"
                        if not np.isnan(value)
                        else "No data"
                    )
                    hover_row.append(text)
                hover_text.append(hover_row)

            fig.add_trace(
                go.Heatmap(
                    z=z_matrix,
                    x=[str(date) for date in dates],
                    y=[p.upper() for p in percentile_cols],
                    colorscale=common_color_scale,
                    hoverongaps=False,
                    hovertemplate="%{text}<extra></extra>",
                    text=hover_text,
                    colorbar_len=0.4,
                    colorbar_y=1.05 - (row_idx - 1) * 0.55,
                    colorbar_x=1.02 + (col_idx - 1) * 0.0,
                ),
                row=row_idx,
                col=col_idx,
            )

        fig.update_xaxes(tickangle=45, row=row_idx, col=col_idx)
        if col_idx == 1:
            fig.update_yaxes(title_text="Percentile", row=row_idx, col=col_idx)

    fig.update_layout(
        title_text="Fill Speed Distribution Heatmaps by Order Size Cohort",
        height=800,
    )
    fig.update_traces(showscale=True)

    return fig

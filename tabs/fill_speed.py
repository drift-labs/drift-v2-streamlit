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
    fetch_fill_speed_data_dynamodb
)

cohort_colors = {
    "0-1k": "blue",
    "1k-10k": "red", 
    "10k-100k": "green",
    "100k+": "orange"
}
cohort_order = ["0-1k", "1k-10k", "10k-100k", "100k+"]
cohorts = ["0", "1000", "10000", "100000"]
cohort_labels = {"0": "0-1k", "1000": "1k-10k", "10000": "10k-100k", "100000": "100k+"}

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
            "Start Date", value=dt.now().date() - timedelta(weeks=15)
        )
    with top_col2:
        end_date = st.date_input("End Date", value=dt.now().date() - timedelta(days=1))

    perp_markets = [m.symbol for m in mainnet_perp_market_configs]
    selected_market = st.selectbox(
        "Select Market to Analyze:",
        options=perp_markets,
        index=0,
    )

    st.write("## Order Type Analysis")

    fill_data = await fetch_fill_speed_data(start_date, end_date, selected_market, "all")
    swift_fill_data = await fetch_fill_speed_data(start_date, end_date, selected_market, "swift")
    non_swift_fill_data = await fetch_fill_speed_data(start_date, end_date, selected_market, "normal")

    if fill_data is None or fill_data.empty:
        st.error("No fill speed data available for the selected time range and markets")
        return

    fill_data["datetime"] = pd.to_datetime(fill_data["ts"], unit="s")
    swift_fill_data["datetime"] = pd.to_datetime(swift_fill_data["ts"], unit="s")
    non_swift_fill_data["datetime"] = pd.to_datetime(non_swift_fill_data["ts"], unit="s")
    filtered_data = apply_filters(fill_data)

    if (
        st.session_state.date_filter
        or st.session_state.performance_filter
    ):
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
                f"P90: {st.session_state.performance_filter[0]:.2f}% auction duration to {st.session_state.performance_filter[1]:.2f}% auction duration"
            )

        st.info(
            f"🔍 Active filters: {' | '.join(filter_info)} ({len(filtered_data)}/{len(fill_data)} data points)"
        )

    st.write("## Overall Fill Speed Metrics (All Selected Data)")

    if not filtered_data.empty:
        num_days_with_data = filtered_data["datetime"].dt.normalize().nunique()
        min_date = filtered_data["datetime"].min().strftime("%Y-%m-%d")
        max_date = filtered_data["datetime"].max().strftime("%Y-%m-%d")
        st.info(
            f"📊 Data available from {min_date} to {max_date} ({num_days_with_data} days)"
        )
    else:
        st.warning("No data available after filtering")

    overall_metrics_row1_col1, overall_metrics_row1_col2 = st.columns(2)
    with overall_metrics_row1_col1:
        st.metric("Number of Days", f"{num_days_with_data}")
        # Clean and convert count column to numeric before summing (defensive programming)
        if "count" in fill_data.columns:
            count_numeric = pd.to_numeric(fill_data["count"], errors='coerce')
            # Filter out unreasonably large values (likely corrupted data)
            count_numeric = count_numeric.where(count_numeric <= 10000000, np.nan)
            total_samples = count_numeric.sum()
            if pd.isna(total_samples):
                total_samples = 0
            st.metric("Number of samples in period", f"{int(total_samples):,}")
        else:
            st.metric("Number of samples in period", "N/A")
    with overall_metrics_row1_col2:
        if not filtered_data.empty and "p50" in filtered_data.columns:
            # Ensure p50 is numeric before calculating mean
            p50_numeric = pd.to_numeric(filtered_data["p50"], errors='coerce')
            avg_median = p50_numeric.mean()
            st.metric("Avg Median Fill Time", f"{avg_median:.2f}% auction duration")
        else:
            st.metric("Avg Median Fill Time", "N/A")

    overall_metrics_row2_col1, overall_metrics_row2_col2 = st.columns(2)
    with overall_metrics_row2_col1:
        if not filtered_data.empty and "p90" in filtered_data.columns:
            # Ensure p90 is numeric before calculating mean
            p90_numeric = pd.to_numeric(filtered_data["p90"], errors='coerce')
            avg_p90 = p90_numeric.mean()
            st.metric("Avg P90 Fill Time", f"{avg_p90:.2f}% auction duration")
        else:
            st.metric("Avg P90 Fill Time", "N/A")
    with overall_metrics_row2_col2:
        if not filtered_data.empty and "p99" in filtered_data.columns:
            # Ensure p99 is numeric before calculating mean
            p99_numeric = pd.to_numeric(filtered_data["p99"], errors='coerce')
            avg_p99 = p99_numeric.mean()
            st.metric("Avg P99 Fill Time", f"{avg_p99:.2f}% auction duration")
        else:
            st.metric("Avg P99 Fill Time", "N/A")


    st.write(
        "#### Daily Fill Speed Distribution by Individual Cohort (Detailed Box Plots)"
    )
    
    if not filtered_data.empty:
        plot_fill_speed_box_plot_over_time(non_swift_fill_data, "Daily Fill Speed Distribution (Normal)")
        plot_fill_speed_box_plot_over_time(swift_fill_data, "Daily Fill Speed Distribution (Swift)")
    else:
        st.info("No data available to display individual cohort daily box plots.")

    cohort_pxx_col1, cohort_pxx_col2 = st.columns([4, 2])
    with cohort_pxx_col1:
        st.write("#### Fill Time by Order Size Cohort (Select Display Mode)")
        percentile_options = [
            f"p{x}" for x in [10, 20, 25, 30, 40, 50, 60, 70, 75, 80, 90, 99]
        ]
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
    with st.expander(
        "Show/Hide Fill Speed Percentiles by Individual Cohort (Line Charts)",
        expanded=False,
    ):
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
            index=3,
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

    # --- Individual Cohort Box Plot Subplots ---
    st.write("#### Fill Speed Distribution by Individual Cohort (Box Plots)")
    if "share_y_axes_individual_cohort_boxes" not in st.session_state:
        st.session_state.share_y_axes_individual_cohort_boxes = False

    st.session_state.share_y_axes_individual_cohort_boxes = st.toggle(
        "🔗 Share Y-Axis Across Cohort Box Plots",
        value=st.session_state.share_y_axes_individual_cohort_boxes,
        key="toggle_share_y_cohort_boxes",
    )
    if not filtered_data.empty:
        fig_individual_cohort_boxes = create_individual_cohort_box_subplots(
            filtered_data,
            share_y_axes=st.session_state.share_y_axes_individual_cohort_boxes,
        )
        st.plotly_chart(
            fig_individual_cohort_boxes,
            use_container_width=True,
            key="individual_cohort_box_subplots",
        )
    else:
        st.info("No data available to display individual cohort box plots.")


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
            st.markdown(f"## Cohort: {cohort_name}")
            cohort_specific_data = filtered_data[filtered_data["cohort"] == cohort_name]
            if not cohort_specific_data.empty:
                num_days_cohort = (
                    cohort_specific_data["datetime"].dt.normalize().nunique()
                )
                
                # Safe calculations with column existence checks and numeric conversion
                if "p50" in cohort_specific_data.columns:
                    p50_numeric = pd.to_numeric(cohort_specific_data["p50"], errors='coerce')
                    avg_median_cohort = p50_numeric.mean()
                else:
                    avg_median_cohort = None
                    
                if "p90" in cohort_specific_data.columns:
                    p90_numeric = pd.to_numeric(cohort_specific_data["p90"], errors='coerce')
                    avg_p90_cohort = p90_numeric.mean()
                else:
                    avg_p90_cohort = None
                    
                if "p99" in cohort_specific_data.columns:
                    p99_numeric = pd.to_numeric(cohort_specific_data["p99"], errors='coerce')
                    avg_p99_cohort = p99_numeric.mean()
                else:
                    avg_p99_cohort = None
                    
                if "p80" in cohort_specific_data.columns:
                    p80_numeric = pd.to_numeric(cohort_specific_data["p80"], errors='coerce')
                    fast_fills_pct_cohort = (p80_numeric < 1.0).mean() * 100
                else:
                    fast_fills_pct_cohort = None

                st.metric("Number of Days", f"{num_days_cohort}")
                st.metric("Avg Median", f"{avg_median_cohort:.2f}% auction duration" if avg_median_cohort is not None else "N/A")
                st.metric("Avg P90", f"{avg_p90_cohort:.2f}% auction duration" if avg_p90_cohort is not None else "N/A")
                st.metric("Avg P99", f"{avg_p99_cohort:.2f}% auction duration" if avg_p99_cohort is not None else "N/A")
                st.metric(
                    "Fast P80<1% auction duration", f"{fast_fills_pct_cohort:.1f}%" if fast_fills_pct_cohort is not None else "N/A"
                )
                # Clean and convert count column to numeric before summing (defensive programming)
                if "count" in cohort_specific_data.columns:
                    count_numeric = pd.to_numeric(cohort_specific_data["count"], errors='coerce')
                    # Filter out unreasonably large values (likely corrupted data)
                    count_numeric = count_numeric.where(count_numeric <= 10000000, np.nan)
                    cohort_total_samples = count_numeric.sum()
                    if pd.isna(cohort_total_samples):
                        cohort_total_samples = 0
                    st.metric("Number of samples in period", f"{int(cohort_total_samples):,}")
                else:
                    st.metric("Number of samples in period", "N/A")
            else:
                st.metric("Number of Days", "N/A")
                st.metric("Avg Median", "N/A")
                st.metric("Avg P90", "N/A")
                st.metric("Avg P99", "N/A")
                st.metric("Fast P80<1% auction duration", "N/A")
                st.metric("Number of samples in period", "N/A")
            st.divider()

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
        # Define the columns we want to aggregate - only include those that exist and are numeric
        desired_cols = [
            "p10", "p20", "p25", "p30", "p40", "p50", "p60", "p70", "p75", "p80", "p90", "p99",
            "min", "max", "count"
        ]
        
        # Filter to only include columns that exist in the dataframe
        available_cols = [col for col in desired_cols if col in filtered_data.columns]
        
        if not available_cols:
            st.error("No numeric columns available for aggregation")
            return
        
        # More robust data cleaning and conversion
        numeric_data = filtered_data.copy()
        
        # Debug: Show data types before conversion
        with st.expander("Debug: Data Types Before Conversion", expanded=False):
            st.write("Column data types:")
            for col in available_cols:
                if col in numeric_data.columns:
                    st.write(f"- {col}: {numeric_data[col].dtype}")
                    sample_values = numeric_data[col].dropna().head(5).tolist()
                    st.write(f"  Sample values: {sample_values}")
        
        # Convert each column individually with more aggressive cleaning
        cleaned_cols = []
        for col in available_cols:
            try:
                # First, handle any obvious string issues
                series = numeric_data[col].astype(str)
                
                # Remove any non-numeric characters except decimal points and negative signs
                series = series.str.replace(r'[^\d.-]', '', regex=True)
                
                # Convert to numeric, coercing errors to NaN
                numeric_series = pd.to_numeric(series, errors='coerce')
                
                # For percentile columns, filter out unreasonably large values
                if col.startswith('p') and col[1:].isdigit():
                    numeric_series = numeric_series.where(numeric_series <= 10000, np.nan)
                elif col == 'count':
                    numeric_series = numeric_series.where(numeric_series <= 1000000, np.nan)
                
                # Only keep the column if we have some valid numeric values
                if not numeric_series.dropna().empty:
                    numeric_data[col] = numeric_series
                    cleaned_cols.append(col)
            except Exception:
                # Skip problematic columns
                continue
        
        # Update available_cols to only include successfully cleaned columns
        available_cols = cleaned_cols
        
        if not available_cols:
            st.error("No columns could be successfully converted to numeric format")
            return
        
        # Debug: Show data types after conversion
        with st.expander("Debug: Data Types After Conversion", expanded=False):
            st.write("Successfully converted columns:")
            for col in available_cols:
                st.write(f"- {col}: {numeric_data[col].dtype}")
                valid_count = numeric_data[col].dropna().shape[0]
                total_count = numeric_data[col].shape[0]
                st.write(f"  Valid values: {valid_count}/{total_count}")
        
        # Final safety check: ensure we still have the datetime column for grouping
        if "datetime" not in numeric_data.columns:
            st.error("datetime column is missing - cannot perform time-based aggregation")
            return
        
        try:
            # Group by date and calculate mean for numeric columns only
            daily_avg_data = (
                numeric_data.groupby(pd.Grouper(key="datetime", freq="D"))[available_cols]
                .mean()
                .reset_index()
            )
        except Exception as e:
            st.error(f"Error during aggregation: {e}")
            st.write("Attempting alternative aggregation method...")
            
            # Alternative approach: aggregate manually
            numeric_data['date'] = numeric_data['datetime'].dt.date
            daily_avg_data = numeric_data.groupby('date')[available_cols].mean().reset_index()
            daily_avg_data['datetime'] = pd.to_datetime(daily_avg_data['date'])
            daily_avg_data = daily_avg_data.drop('date', axis=1)
        
        if daily_avg_data.empty:
            st.error("No data available after aggregation")
            return

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
        st.write("• Color: Fill time values (% auction duration)")
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
            handle_heatmap_selection(
                heatmap_selection.selection.points, heatmap_display_data
            )

    # --- Box Plot for Overall Daily Averaged Percentiles ---
    st.write("## Overall Percentile Distribution (Box Plot)")
    if not daily_avg_data.empty:
        overall_box_col1, _ = st.columns([5, 1])
        with overall_box_col1:
            fig_overall_box = create_box_plot(daily_avg_data)
            st.plotly_chart(
                fig_overall_box,
                use_container_width=True,
                key="overall_box_plot_chart",
            )
    else:
        st.info(
            "Not enough daily average data to display the overall percentile distribution box plot."
        )

    # --- Daily Box Plots using Min/Max data ---
    st.write("## Daily Fill Speed Distribution (Enhanced Box Plots)")
    if (
        not daily_avg_data.empty
        and "min" in daily_avg_data.columns
        and "max" in daily_avg_data.columns
    ):
        daily_box_col1, daily_box_col2 = st.columns([5, 1])
        with daily_box_col2:
            st.write("**Enhanced Box Plots:**")
            st.write("• Uses actual min/max values")
            st.write("• P25/P75 as quartiles")
            st.write("• P50 as median")
            st.write("• Shows count in hover")

        with daily_box_col1:
            fig_daily_boxes = create_daily_box_plots(daily_avg_data)
            st.plotly_chart(
                fig_daily_boxes,
                use_container_width=True,
                key="daily_box_plots_chart",
            )
    else:
        st.info("Min/max data not available for enhanced daily box plots.")


def plot_fill_speed_box_plot_over_time(filtered_data, title):
    required_cols_for_daily_box = [
        "min",
        "p25",
        "p50",
        "p75",
        "max",
        "count",
        "datetime",
        "cohort",
    ]
    # Also check for p10 and p99 which are needed for the box plots
    if "p10" in filtered_data.columns:
        required_cols_for_daily_box.append("p10")
    if "p99" in filtered_data.columns:
        required_cols_for_daily_box.append("p99")

    if "datetime" not in filtered_data.columns:
        filtered_data["datetime"] = pd.to_datetime(filtered_data["ts"], unit="s")
        
    # if all(col in filtered_data.columns for col in required_cols_for_daily_box):
    if True:
        # Define colors for each cohort
        
        # Create overlaid box plots for fill speed
        fig_fill_speed_box = go.Figure()
        
        # Group data by cohort and create box plots for each day
        for cohort in cohort_order:
            cohort_data = filtered_data[filtered_data["cohort"] == cohort]
            
            for i, (_, row) in enumerate(cohort_data.iterrows()):
                if not pd.isna(row.get("p10")) and not pd.isna(row.get("p99")):
                    fig_fill_speed_box.add_trace(
                        go.Box(
                            x=[row["datetime"].strftime("%Y-%m-%d")],
                            q1=[row.get("p25", row.get("p50", 0))],
                            median=[row.get("p50", 0)],
                            q3=[row.get("p75", row.get("p50", 0))],
                            lowerfence=[row.get("p10", row.get("p50", 0))],
                            upperfence=[row.get("p99", row.get("p50", 0))],
                            mean=[row.get("p50", 0)],  # Use p50 as mean approximation
                            name=f"Cohort {cohort}",
                            legendgroup=f"cohort_{cohort}",
                            showlegend=i == 0,  # Only show legend for first trace of each cohort
                            boxpoints=False,
                            marker_color=cohort_colors.get(cohort, 'blue'),
                            line_color=cohort_colors.get(cohort, 'blue')
                        )
                    )

        fig_fill_speed_box.update_layout(
            title=title,
            xaxis_title="Date",
            yaxis_title="Fill Time (% auction duration)",
            showlegend=True,
        )
        st.plotly_chart(fig_fill_speed_box, use_container_width=True)
    else:
        st.info(
            "Required data columns (min, p25, p50, p75, max, count, p10, p99) not available for daily cohort box plots."
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
        f"p{p}" for p in [10, 20, 25, 30, 40, 50, 60, 70, 75, 80, 90, 99]
    ]
    
    # Filter to only include columns that exist in the dataframe
    available_percentile_cols = [col for col in percentile_cols if col in data.columns]
    
    if not available_percentile_cols:
        # Return empty dataframe with expected structure if no percentile columns found
        return pd.DataFrame(columns=['date'] + percentile_cols)
    
    # More robust numeric conversion
    numeric_data = data.copy()
    cleaned_cols = []
    
    for col in available_percentile_cols:
        try:
            # Convert to string first, then clean and convert to numeric
            series = numeric_data[col].astype(str)
            series = series.str.replace(r'[^\d.-]', '', regex=True)
            numeric_series = pd.to_numeric(series, errors='coerce')
            
            # For percentile columns, filter out unreasonably large values
            if col.startswith('p') and col[1:].isdigit():
                numeric_series = numeric_series.where(numeric_series <= 10000, np.nan)
            elif col == 'count':
                numeric_series = numeric_series.where(numeric_series <= 1000000, np.nan)
            
            if not numeric_series.dropna().empty:
                numeric_data[col] = numeric_series
                cleaned_cols.append(col)
        except Exception:
            # Skip problematic columns
            continue
    
    if not cleaned_cols:
        return pd.DataFrame(columns=['date'] + percentile_cols)

    try:
        heatmap_df = numeric_data.groupby("date")[cleaned_cols].mean().reset_index()
    except Exception:
        # Return empty dataframe if aggregation fails
        return pd.DataFrame(columns=['date'] + percentile_cols)

    heatmap_df["datetime_temp"] = pd.to_datetime(heatmap_df["date"])
    heatmap_df["day_of_week"] = heatmap_df["datetime_temp"].dt.day_name()

    return heatmap_df


def create_interactive_heatmap(data, metric, color_scale):
    """Create interactive heatmap with percentiles on Y-axis and time on X-axis
    Assumes input `data` is already aggregated daily with necessary percentile columns.
    """
    percentile_cols = [
        f"p{p}" for p in [10, 20, 25, 30, 40, 50, 60, 70, 75, 80, 90, 99]
    ]

    dates = sorted(data["date"].unique())

    z_matrix = []
    for percentile in percentile_cols:
        row_values = data.set_index("date")[percentile].reindex(dates).values
        z_matrix.append(row_values)

    hover_text = []
    for i, percentile_label in enumerate(percentile_cols):
        hover_row = []
        for j, date_val in enumerate(dates):
            value = z_matrix[i][j]
            if not np.isnan(value):
                hover_row.append(
                    f"Date: {date_val}<br>Avg Percentile: {percentile_label.upper()}<br>Fill Time: {value:.3f}% auction duration"
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
            colorbar=dict(title="Avg Fill Time (% auction duration)"),
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

    # Fix datetime conversion for plotly
    if "datetime" in data.columns:
        datetime_values = pd.to_datetime(data["datetime"]).dt.to_pydatetime()
    else:
        datetime_values = data["datetime"]

    for p, color in zip(percentiles, colors):
        fig.add_trace(
            go.Scatter(
                x=datetime_values,
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
        yaxis_title="Fill Time (% auction duration)",
        height=500,
        hovermode="x unified",
        dragmode="select",
        selectdirection="h",
    )

    return fig


def create_box_plot(data):
    fig = go.Figure()

    percentile_cols = sorted(
        [col for col in data.columns if col.startswith("p") and col[1:].isdigit()]
    )

    if not percentile_cols:
        st.warning("No percentile columns (p10-p99) found in data for the box plot.")
        return fig

    for p in percentile_cols:
        if not data[p].dropna().empty:
            fig.add_trace(
                go.Box(
                    y=data[p].dropna(),
                    name=p.upper(),
                    boxpoints="outliers",
                    legendgroup=p.upper(),
                    showlegend=(p == percentile_cols[0]),
                ),
            )

    fig.update_layout(
        title="Fill Time Distribution by Percentile (Click outliers to filter)",
        yaxis_title="Fill Time (% auction duration)",
        height=400,
        dragmode="select",
    )

    return fig


def create_trends_chart(data):
    """Create performance trends chart"""
    window = min(7, len(data) // 4)

    if window > 1:
        data = data.copy()
        
        # Safe rolling mean calculations with column existence checks
        if "p50" in data.columns:
            p50_numeric = pd.to_numeric(data["p50"], errors='coerce')
            data["p50_rolling"] = p50_numeric.rolling(window=window, center=True).mean()
        else:
            data["p50_rolling"] = None
            
        if "p90" in data.columns:
            p90_numeric = pd.to_numeric(data["p90"], errors='coerce')
            data["p90_rolling"] = p90_numeric.rolling(window=window, center=True).mean()
        else:
            data["p90_rolling"] = None

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
        fig.update_yaxes(title_text="Fill Time (% auction duration)", row=1, col=1)
        fig.update_yaxes(title_text="Fill Time (% auction duration)", row=2, col=1)
        fig.update_xaxes(title_text="Time", row=2, col=1)

        return fig
    else:
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


async def fetch_fill_speed_data(start_date, end_date, selected_market, order_type):
    """Fetch and process fill speed data from DynamoDB."""
    
    start_ts = int(dt.combine(start_date, dt.min.time()).timestamp())
    end_ts = int(dt.combine(end_date, dt.max.time()).timestamp())
    
    # Fetch data for all cohorts
    all_cohorts = ["0", "1000", "10000", "100000"]
    cohort_data = {}
    
    for cohort in all_cohorts:
        data_df = fetch_fill_speed_data_dynamodb(
            start_ts, end_ts, selected_market, order_type, cohort
        )
        if data_df is not None and not data_df.empty:
            cohort_data[cohort] = data_df

    if not cohort_data:
        st.warning(
            "No data available for the selected criteria. Try a different date range or market."
        )
        return pd.DataFrame()

    # Combine all cohort data
    all_processed_data = []
    for cohort, data_df in cohort_data.items():
        all_processed_data.append(data_df)
    
    if not all_processed_data:
        return pd.DataFrame()
    
    # Combine all DataFrames
    combined_df = pd.concat(all_processed_data, ignore_index=True)
    
    return combined_df


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
        yaxis_title_text = "Fill Time (% auction duration)"
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
        yaxis_title_text = (
            f"{percentile_to_plot.upper()} Fill Time (% auction duration)"
        )

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
    percentiles_to_plot = [
        f"p{x}" for x in [10, 20, 25, 30, 40, 50, 60, 70, 75, 80, 90, 99]
    ]
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
        fig.update_yaxes(
            title_text="Fill Time (% auction duration)", row=row_idx, col=col_idx
        )

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
        f"p{p}" for p in [10, 20, 25, 30, 40, 50, 60, 70, 75, 80, 90, 99]
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

            # Filter to only include columns that exist in the dataframe
            available_percentile_cols = [col for col in percentile_cols if col in cohort_data.columns]
            
            if not available_percentile_cols:
                continue  # Skip this cohort if no percentile columns are available
            
            # More robust numeric conversion
            numeric_cohort_data = cohort_data.copy()
            cleaned_cols = []
            
            for col in available_percentile_cols:
                try:
                    # Convert to string first, then clean and convert to numeric
                    series = numeric_cohort_data[col].astype(str)
                    series = series.str.replace(r'[^\d.-]', '', regex=True)
                    numeric_series = pd.to_numeric(series, errors='coerce')
                    
                    # For percentile columns, filter out unreasonably large values
                    if col.startswith('p') and col[1:].isdigit():
                        numeric_series = numeric_series.where(numeric_series <= 10000, np.nan)
                    elif col == 'count':
                        numeric_series = numeric_series.where(numeric_series <= 1000000, np.nan)
                    
                    if not numeric_series.dropna().empty:
                        numeric_cohort_data[col] = numeric_series
                        cleaned_cols.append(col)
                except Exception:
                    # Skip problematic columns
                    continue
            
            if not cleaned_cols:
                continue  # Skip this cohort if no valid columns

            # Aggregate by date to handle duplicate dates (from different order types)
            try:
                cohort_data_agg = (
                    numeric_cohort_data.groupby("date")[cleaned_cols].mean().reset_index()
                )
            except Exception:
                continue  # Skip this cohort if aggregation fails

            dates = sorted(cohort_data_agg["date"].unique())

            z_matrix = []
            for p_col in cleaned_cols:
                row_values = (
                    cohort_data_agg.set_index("date")[p_col].reindex(dates).values
                )
                z_matrix.append(row_values)

            hover_text = []
            for r_idx, percentile_label in enumerate(cleaned_cols):
                hover_row = []
                for c_idx, date_val in enumerate(dates):
                    value = z_matrix[r_idx][c_idx]
                    text = f"Date: {date_val}<br>Cohort: {cohort}<br>Percentile: {percentile_label.upper()}<br>"
                    text += (
                        f"Fill Time: {value:.3f}% auction duration"
                        if not np.isnan(value)
                        else "No data"
                    )
                    hover_row.append(text)
                hover_text.append(hover_row)

            fig.add_trace(
                go.Heatmap(
                    z=z_matrix,
                    x=[str(date) for date in dates],
                    y=[p.upper() for p in cleaned_cols],
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


def create_individual_cohort_box_subplots(data, share_y_axes=False):
    """Create a 2x2 grid of box plots, one for each cohort, showing distributions of key percentiles."""
    cohort_order = ["0-1k", "1k-10k", "10k-100k", "100k+"]
    percentiles_to_plot = sorted(
        [col for col in data.columns if col.startswith("p") and col[1:].isdigit()]
    )

    if not percentiles_to_plot:
        fig = go.Figure()
        fig.update_layout(
            title_text="No percentile data available for cohort box plots"
        )
        return fig

    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=[f"Box Plot - Cohort: {c}" for c in cohort_order],
        shared_xaxes=True,
        shared_yaxes=share_y_axes,
        vertical_spacing=0.15,
        horizontal_spacing=0.05,
    )

    all_y_values_for_shared_axis = []

    for i, cohort in enumerate(cohort_order):
        cohort_data = data[data["cohort"] == cohort]
        row_idx = (i // 2) + 1
        col_idx = (i % 2) + 1

        if not cohort_data.empty:
            for p_col in percentiles_to_plot:
                if (
                    p_col in cohort_data.columns
                    and not cohort_data[p_col].dropna().empty
                ):
                    valid_y_data = cohort_data[p_col].dropna()
                    fig.add_trace(
                        go.Box(
                            y=valid_y_data,
                            name=p_col.upper(),
                            boxpoints="outliers",
                            legendgroup=p_col.upper(),
                            showlegend=(i == 0),
                        ),
                        row=row_idx,
                        col=col_idx,
                    )
                    if share_y_axes:
                        all_y_values_for_shared_axis.extend(valid_y_data.tolist())

        fig.update_yaxes(
            title_text="Fill Time (% auction duration)", row=row_idx, col=col_idx
        )

    if share_y_axes and all_y_values_for_shared_axis:
        global_y_min = min(all_y_values_for_shared_axis)
        global_y_max = max(all_y_values_for_shared_axis)
        padded_min = global_y_min * 0.95 if global_y_min > 0 else global_y_min * 1.05
        padded_max = global_y_max * 1.05
        if padded_min == padded_max:
            padded_min -= 1
            padded_max += 1
        if padded_min > padded_max:
            padded_min, padded_max = padded_max, padded_min

        fig.update_yaxes(range=[padded_min, padded_max])
    else:
        fig.update_yaxes(autorange=True)

    fig.update_layout(
        title_text="Fill Speed Distribution by Order Size Cohort (Box Plots)",
        height=800,
        legend_title_text="Percentile",
    )
    return fig


def create_daily_box_plots(data):
    """Create daily box plots using min, max, and percentile data from the new schema"""
    fig = go.Figure()

    data_sorted = data.sort_values("datetime")
    for _, row in data_sorted.iterrows():
        if pd.isna(row.get("min")) or pd.isna(row.get("max")):
            continue

        date_str = row["datetime"].strftime("%Y-%m-%d")

        fig.add_trace(
            go.Box(
                x=[date_str],
                q1=[row.get("p25", np.nan)],  # First quartile
                median=[row.get("p50", np.nan)],  # Median
                q3=[row.get("p75", np.nan)],  # Third quartile
                lowerfence=[row.get("min", np.nan)],  # Minimum
                upperfence=[row.get("max", np.nan)],  # Maximum
                name=date_str,
                showlegend=False,
                boxpoints=False,
                hovertemplate=(
                    f"Date: {date_str}<br>"
                    f"Min: {row.get('min', 'N/A'):.2f}%<br>"
                    f"P25: {row.get('p25', 'N/A'):.2f}%<br>"
                    f"Median: {row.get('p50', 'N/A'):.2f}%<br>"
                    f"P75: {row.get('p75', 'N/A'):.2f}%<br>"
                    f"Max: {row.get('max', 'N/A'):.2f}%<br>"
                    f"Count: {row.get('count', 'N/A')}<extra></extra>"
                ),
            )
        )

    fig.update_layout(
        title="Daily Fill Speed Distribution (Box Plots with Min/Max)",
        xaxis_title="Date",
        yaxis_title="Fill Time (% auction duration)",
        height=500,
        xaxis=dict(tickangle=45),
    )

    return fig


def create_swift_vs_normal_comparison_chart(
    swift_data, normal_data, comparison_percentiles
):
    """Create a comparison chart between swift and normal order types."""
    fig = go.Figure()

    if swift_data.empty and normal_data.empty:
        fig.update_layout(title="No data available for comparison")
        return fig

    # Aggregate data by date and calculate averages across cohorts
    def prepare_data(data, order_type_name):
        if data.empty:
            return pd.DataFrame()

        # Filter comparison_percentiles to only include those that exist in the data
        available_percentiles = [col for col in comparison_percentiles if col in data.columns]
        
        if not available_percentiles:
            return pd.DataFrame()
        
        # More robust numeric conversion
        numeric_data = data.copy()
        cleaned_cols = []
        
        for col in available_percentiles:
            try:
                # Convert to string first, then clean and convert to numeric
                series = numeric_data[col].astype(str)
                series = series.str.replace(r'[^\d.-]', '', regex=True)
                numeric_series = pd.to_numeric(series, errors='coerce')
                
                # For percentile columns, filter out unreasonably large values
                if col.startswith('p') and col[1:].isdigit():
                    numeric_series = numeric_series.where(numeric_series <= 10000, np.nan)
                elif col == 'count':
                    numeric_series = numeric_series.where(numeric_series <= 1000000, np.nan)
                
                if not numeric_series.dropna().empty:
                    numeric_data[col] = numeric_series
                    cleaned_cols.append(col)
            except Exception:
                # Skip problematic columns
                continue
        
        if not cleaned_cols:
            return pd.DataFrame()

        # Group by date and calculate average across all cohorts for each percentile
        try:
            daily_avg = (
                numeric_data.groupby(pd.Grouper(key="datetime", freq="D"))[cleaned_cols]
                .mean()
                .reset_index()
            )
            daily_avg["order_type"] = order_type_name
            return daily_avg
        except Exception:
            return pd.DataFrame()

    swift_daily = prepare_data(swift_data, "Swift")
    normal_daily = prepare_data(normal_data, "Normal")

    # Colors for swift vs normal
    swift_color = "#00CC96"  # Green
    normal_color = "#EF553B"  # Red

    # Add traces for each percentile
    for i, percentile in enumerate(comparison_percentiles):
        line_style = dict(width=2)
        if percentile == "p50":
            line_style["dash"] = "solid"
        elif percentile in ["p90", "p99"]:
            line_style["dash"] = "dash"
        else:
            line_style["dash"] = "dot"

        # Swift traces
        if not swift_daily.empty and percentile in swift_daily.columns:
            fig.add_trace(
                go.Scatter(
                    x=swift_daily["datetime"],
                    y=swift_daily[percentile],
                    mode="lines+markers",
                    name=f"Swift {percentile.upper()}",
                    line=dict(color=swift_color, **line_style),
                    marker=dict(color=swift_color, size=4),
                    legendgroup="Swift",
                    opacity=0.8,
                )
            )

        # Normal traces
        if not normal_daily.empty and percentile in normal_daily.columns:
            fig.add_trace(
                go.Scatter(
                    x=normal_daily["datetime"],
                    y=normal_daily[percentile],
                    mode="lines+markers",
                    name=f"Normal {percentile.upper()}",
                    line=dict(color=normal_color, **line_style),
                    marker=dict(color=normal_color, size=4),
                    legendgroup="Normal",
                    opacity=0.8,
                )
            )

    fig.update_layout(
        title="Swift vs Normal Order Type Fill Speed Comparison",
        xaxis_title="Date",
        yaxis_title="Fill Time (% auction duration)",
        height=500,
        hovermode="x unified",
        legend=dict(groupclick="toggleitem", title="Order Type & Percentile"),
        dragmode="select",
        selectdirection="h",
    )

    # Add annotation explaining the chart
    if not swift_daily.empty or not normal_daily.empty:
        fig.add_annotation(
            text="Lower values = faster fills | Solid=P50, Dash=P90/P99, Dot=Others",
            xref="paper",
            yref="paper",
            x=0.5,
            y=1.05,
            showarrow=False,
            font=dict(size=10),
            xanchor="center",
        )

    return fig
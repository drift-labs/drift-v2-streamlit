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

# Import DynamoDB utilities
from utils.dynamodb_client import (
    fetch_liquidity_source_data_dynamodb
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


async def liquidity_source_analysis(clearinghouse: DriftClient):
    st.write("# Liquidity Source Analysis")
    st.write(
        "Analyze liquidity sources across different markets and order size cohorts"
    )

    col1, col2 = st.columns(2)
    with col1:
        # Add liquidity source grouping selection
        st.write("### Liquidity Source Grouping")
        grouping_mode = st.radio(
            "Choose how to display liquidity sources:",
            options=["Individual Sources", "JIT vs Non-JIT", "AMM vs DLOB"],
            index=0,
            help="""
            - **Individual Sources**: Show all 4 sources separately (AMM, AMM JIT, Match, Match JIT)
            - **JIT vs Non-JIT**: Group by JIT type (JIT Sources vs Non-JIT Sources)  
            - **AMM vs DLOB**: Group by entity type (AMM vs Users DLOB)
            """
        )

    with col2:
        # Add volume units selection
        st.write("### Volume Units")
        volume_units = st.radio(
            "Choose volume measurement:",
            options=["Dollars", "Counts"],
            index=0,
            help="""
            - **Dollars**: Display volume in dollar amounts
            - **Counts**: Display volume as number of individual fills
            """
        )

    top_col1, top_col2 = st.columns([2, 2])
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

    render_liquidity_source_analysis(start_date, end_date, selected_market, grouping_mode, volume_units)


def get_source_config_and_processing(grouping_mode, volume_units="Dollars"):
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


def render_liquidity_source_analysis_total(start_ts, end_ts, selected_market, grouping_mode, volume_units):
    liquidity_source_data = fetch_liquidity_source_data_dynamodb(start_ts, end_ts, selected_market)
    
    if liquidity_source_data is None or liquidity_source_data.empty:
        st.warning("No liquidity source data available for the selected period.")
        return
    
    # Process the data
    df_processed = liquidity_source_data.copy()
    df_processed["ts"] = pd.to_numeric(df_processed["ts"], errors="coerce")
    df_processed.dropna(subset=["ts"], inplace=True)
    df_processed["datetime"] = pd.to_datetime(df_processed["ts"], unit="s")
    
    # Get source configuration and processing based on grouping mode and volume units
    main_sources, source_config, process_func, liquidity_cols = get_source_config_and_processing(grouping_mode, volume_units)
    
    # Convert liquidity source columns to numeric
    for col in liquidity_cols:
        if col in df_processed.columns:
            df_processed[col] = pd.to_numeric(df_processed[col], errors="coerce")
    
    # Sort by datetime
    df_processed = df_processed.sort_values('datetime')
    
    # Apply processing function to combine sources if needed
    df_processed = process_func(df_processed, main_sources)
    
    # Focus on the sources defined by the grouping mode
    available_sources = [col for col in main_sources if col in df_processed.columns and not df_processed[col].isnull().all()]
    
    if not available_sources:
        st.warning("No valid liquidity source data found.")
        return
    
    # Calculate total volume for each time point
    df_processed['total_volume'] = df_processed[available_sources].sum(axis=1)
    
    # Calculate percentages for each source
    for col in available_sources:
        df_processed[f'{col}_pct'] = (df_processed[col] / df_processed['total_volume'] * 100).fillna(0)
    
    # Determine formatting based on volume units
    if volume_units == "Counts":
        value_format = "{:,.0f}"
        axis_format = ",.0f"
        hover_value_format = "{:,.0f}"
        unit_label = ""
        y_axis_title = "Count"
        hover_total_format = "%{y:,.0f}"
    else:
        value_format = "${:,.0f}"
        axis_format = "$,.0f"
        hover_value_format = "${:,.0f}"
        unit_label = "$"
        y_axis_title = "Volume ($)"
        hover_total_format = "$%{y:,.0f}"
    
    # Create stacked area chart
    fig_total = go.Figure()
    
    # Add stacked area traces
    for col in available_sources:
        config = source_config.get(col, {"color": "rgba(128, 128, 128, 0.8)", "name": col})
        
        # Create custom hover text
        hover_text = [
            f"<b>{config['name']}</b><br>" +
            f"Volume: {hover_value_format.format(abs_val)}<br>" +
            f"Percentage: {pct:.1f}%<br>"
            for abs_val, pct in zip(
                df_processed[col],
                df_processed[f'{col}_pct']
            )
        ]
        
        fig_total.add_trace(go.Scatter(
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
    
    # Add total volume line
    fig_total.add_trace(go.Scatter(
        x=df_processed['datetime'],
        y=df_processed['total_volume'],
        mode='lines+markers',
        name='Total Volume',
        line=dict(color='rgba(75, 0, 130, 0.9)', width=3, dash='solid'),
        marker=dict(size=6, color='rgba(75, 0, 130, 0.9)'),
        hovertemplate=f'<b>Total Volume</b><br>Date: %{{x}}<br>Volume: {hover_total_format}<extra></extra>',
    ))
    
    fig_total.update_layout(
        title=f"Total Fill Volume: Liquidity Sources Breakdown ({grouping_mode} - {volume_units}) - {selected_market}",
        xaxis_title="Date",
        yaxis_title=y_axis_title,
        yaxis=dict(tickformat=axis_format),
        hovermode='x unified',
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        height=500
    )
    
    st.plotly_chart(fig_total, use_container_width=True)
    
    # Show summary statistics in an expander
    with st.expander("📊 Total Fill Volume Summary Statistics", expanded=False):
        st.write("**Average Distribution:**")
        summary_data = {}
        for col in available_sources:
            avg_pct = df_processed[f'{col}_pct'].mean()
            min_pct = df_processed[f'{col}_pct'].min()
            max_pct = df_processed[f'{col}_pct'].max()
            avg_abs = df_processed[col].mean()
            total_abs = df_processed[col].sum()
            config = source_config.get(col, {"name": col})
            summary_data[config['name']] = {
                'Min %': f"{min_pct:.1f}%",
                'Max %': f"{max_pct:.1f}%", 
                'Average %': f"{avg_pct:.1f}%",
                'Average Volume': value_format.format(avg_abs),
                'Total Volume': value_format.format(total_abs)
            }
        
        summary_df = pd.DataFrame(summary_data).T
        st.dataframe(summary_df)
        
        # Total volume metrics
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Min Daily", value_format.format(df_processed['total_volume'].min()))
        col2.metric("Max Daily", value_format.format(df_processed['total_volume'].max()))
        col3.metric("Average Daily", value_format.format(df_processed['total_volume'].mean()))
        col4.metric("Total Period", value_format.format(df_processed['total_volume'].sum()))
    
    # Show raw data in an expander
    with st.expander("📋 Raw Data", expanded=False):
        st.dataframe(df_processed)


def render_liquidity_source_analysis_by_cohort(start_ts, end_ts, selected_market, grouping_mode, volume_units):
    # Fetch data for all cohorts
    cohort_liquidity_data = {}
    for cohort in cohorts:
        data = fetch_liquidity_source_data_dynamodb(start_ts, end_ts, selected_market, cohort)
        if data is not None and not data.empty:
            cohort_liquidity_data[cohort] = data
    
    if not cohort_liquidity_data:
        st.warning("No liquidity source data available for any cohort in the selected period.")
        return
    
    # Process and plot data for each cohort
    def process_cohort_liquidity_data(df):
        if df.empty:
            return None
        
        # Convert timestamp and numeric columns
        df_processed = df.copy()
        df_processed["ts"] = pd.to_numeric(df_processed["ts"], errors="coerce")
        df_processed.dropna(subset=["ts"], inplace=True)
        df_processed["datetime"] = pd.to_datetime(df_processed["ts"], unit="s")
        
        # Get source configuration and processing based on grouping mode and volume units
        main_sources, source_config, process_func, liquidity_cols = get_source_config_and_processing(grouping_mode, volume_units)
        
        # Convert liquidity source columns to numeric
        for col in liquidity_cols:
            if col in df_processed.columns:
                df_processed[col] = pd.to_numeric(df_processed[col], errors="coerce")
        
        # Sort by datetime
        df_processed = df_processed.sort_values('datetime')
        
        # Apply processing function to combine sources if needed
        df_processed = process_func(df_processed, main_sources)
        
        # Focus on the sources defined by the grouping mode
        available_sources = [col for col in main_sources if col in df_processed.columns and not df_processed[col].isnull().all()]
        
        if not available_sources:
            return None
        
        # Calculate total volume for each time point
        df_processed['total_volume'] = df_processed[available_sources].sum(axis=1)
        
        # Calculate percentages for each source
        for col in available_sources:
            df_processed[f'{col}_pct'] = (df_processed[col] / df_processed['total_volume'] * 100).fillna(0)
        
        return df_processed, available_sources
    
    # Define colors for cohorts
    cohort_colors_by_lb = {
        "0": "#636EFA",      # Blue
        "1000": "#EF553B",   # Red
        "10000": "#00CC96",  # Green
        "100000": "#AB63FA"  # Purple
    }

    selected_cohorts = st.multiselect("Select Cohorts to Analyze", cohorts, default=cohorts)

    # Determine formatting based on volume units
    if volume_units == "Counts":
        value_format = "{:,.0f}"
        axis_format = ",.0f"
        hover_value_format = "{:,.0f}"
        unit_label = ""
        y_axis_title = "Count"
        hover_total_format = "%{y:,.0f}"
    else:
        value_format = "${:,.0f}"
        axis_format = "$,.0f"
        hover_value_format = "${:,.0f}"
        unit_label = "$"
        y_axis_title = "Total Volume ($)"
        hover_total_format = "$%{y:,.0f}"

    # Create comparative chart showing all cohorts together
    fig_comparison = go.Figure()
    
    for cohort in cohorts:
        if cohort in cohort_liquidity_data:
            result = process_cohort_liquidity_data(cohort_liquidity_data[cohort])
            
            if result is not None:
                df_processed, _ = result
                fig_comparison.add_trace(go.Scatter(
                    x=df_processed['datetime'],
                    y=df_processed['total_volume'],
                    mode='lines+markers',
                    name=f'Cohort {cohort_labels[cohort]}',
                    line=dict(color=cohort_colors_by_lb.get(cohort, "#636EFA"), width=2),
                    marker=dict(size=4),
                    hovertemplate=f'<b>Cohort {cohort_labels[cohort]}</b><br>Date: %{{x}}<br>Volume: {hover_total_format}<extra></extra>'
                ))
    
    fig_comparison.update_layout(
        title=f"Total Fill Volume Comparison Across Cohorts ({volume_units})",
        xaxis_title="Date",
        yaxis_title=y_axis_title,
        yaxis=dict(tickformat=axis_format),
        hovermode='x unified',
        height=500
    )
    
    st.plotly_chart(fig_comparison, use_container_width=True)
    
    # Create individual charts for each cohort
    for cohort in selected_cohorts:
        if cohort in cohort_liquidity_data:
            cohort_label = cohort_labels[cohort]
            cohort_color = cohort_colors_by_lb.get(cohort, "#636EFA")
            
            st.write(f"##### Cohort {cohort_label} Liquidity Sources")
            
            result = process_cohort_liquidity_data(cohort_liquidity_data[cohort])
            
            if result is None:
                st.warning(f"No valid liquidity source data for cohort {cohort_label}")
                continue
            
            df_processed, available_sources = result
            
            # Create stacked area chart
            fig_cohort = go.Figure()
            
            # Get source configuration based on grouping mode
            _, source_config, _, _ = get_source_config_and_processing(grouping_mode, volume_units)
            
            # Add stacked area traces
            for col in available_sources:
                config = source_config.get(col, {"color": "rgba(128, 128, 128, 0.8)", "name": col})
                
                # Create custom hover text
                hover_text = [
                    f"<b>{config['name']}</b><br>" +
                    f"Volume: {hover_value_format.format(abs_val)}<br>" +
                    f"Percentage: {pct:.1f}%<br>"
                    for abs_val, pct in zip(
                        df_processed[col],
                        df_processed[f'{col}_pct']
                    )
                ]
                
                fig_cohort.add_trace(go.Scatter(
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
            
            # Add total volume line
            fig_cohort.add_trace(go.Scatter(
                x=df_processed['datetime'],
                y=df_processed['total_volume'],
                mode='lines+markers',
                name='Total Volume',
                line=dict(color=cohort_color, width=3, dash='solid'),
                marker=dict(size=6, color=cohort_color),
                hovertemplate=f'<b>Total Volume</b><br>Date: %{{x}}<br>Volume: {hover_total_format}<extra></extra>',
            ))
            
            fig_cohort.update_layout(
                title=f"Cohort {cohort_label}: Liquidity Sources Breakdown ({grouping_mode} - {volume_units})",
                xaxis_title="Date",
                yaxis_title=y_axis_title.replace("Total ", ""),
                yaxis=dict(tickformat=axis_format),
                hovermode='x unified',
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                height=400
            )
            
            st.plotly_chart(fig_cohort, use_container_width=True)
            
            # Show summary statistics in an expander
            with st.expander(f"📊 Cohort {cohort_label} Summary Statistics", expanded=False):
                st.write("**Average Distribution:**")
                summary_data = {}
                for col in available_sources:
                    avg_pct = df_processed[f'{col}_pct'].mean()
                    min_pct = df_processed[f'{col}_pct'].min()
                    max_pct = df_processed[f'{col}_pct'].max()
                    avg_abs = df_processed[col].mean()
                    total_abs = df_processed[col].sum()
                    config = source_config.get(col, {"name": col})
                    summary_data[config['name']] = {
                        'Min %': f"{min_pct:.1f}%",
                        'Max %': f"{max_pct:.1f}%", 
                        'Average %': f"{avg_pct:.1f}%",
                        'Average Volume': value_format.format(avg_abs),
                        'Total Volume': value_format.format(total_abs)
                    }
                
                summary_df = pd.DataFrame(summary_data).T
                st.dataframe(summary_df)
                
                # Total volume metrics
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Min Daily", value_format.format(df_processed['total_volume'].min()))
                col2.metric("Max Daily", value_format.format(df_processed['total_volume'].max()))
                col3.metric("Average Daily", value_format.format(df_processed['total_volume'].mean()))
                col4.metric("Total Period", value_format.format(df_processed['total_volume'].sum()))


def render_liquidity_source_analysis_by_order_type(start_ts, end_ts, selected_market, grouping_mode, volume_units):
    order_types_all = [
        'market',
        'limit',
        'oracle',
        'triggerMarket',
        'triggerLimit',
    ]
    selected_order_types = st.multiselect("Select Order Types to Analyze", order_types_all, default=order_types_all)
    
    # Define colors for order types
    order_type_colors = {
        "market": "#636EFA",      # Blue
        "limit": "#EF553B",       # Red
        "oracle": "#00CC96",      # Green
        "triggerMarket": "#AB63FA", # Purple
        "triggerLimit": "#FFA15A"   # Orange
    }
    
    # Determine formatting based on volume units
    if volume_units == "Counts":
        value_format = "{:,.0f}"
        axis_format = ",.0f"
        hover_value_format = "{:,.0f}"
        unit_label = ""
        y_axis_title = "Count"
        hover_total_format = "%{y:,.0f}"
    else:
        value_format = "${:,.0f}"
        axis_format = "$,.0f"
        hover_value_format = "${:,.0f}"
        unit_label = "$"
        y_axis_title = "Total Volume ($)"
        hover_total_format = "$%{y:,.0f}"
    
    # Create comparative chart showing all order types together
    fig_comparison = go.Figure()
    order_type_data = {}
    
    for order_type in selected_order_types:
        liquidity_source_data = fetch_liquidity_source_data_dynamodb(start_ts, end_ts, selected_market, taker_order_type=order_type)
        
        if liquidity_source_data is None or liquidity_source_data.empty:
            st.warning(f"No liquidity source data available for {order_type}")
            continue
        
        # Process the data
        df_processed = liquidity_source_data.copy()
        df_processed["ts"] = pd.to_numeric(df_processed["ts"], errors="coerce")
        df_processed.dropna(subset=["ts"], inplace=True)
        df_processed["datetime"] = pd.to_datetime(df_processed["ts"], unit="s")
        
        # Get source configuration and processing based on grouping mode and volume units
        main_sources, source_config, process_func, liquidity_cols = get_source_config_and_processing(grouping_mode, volume_units)
        
        # Convert liquidity source columns to numeric
        for col in liquidity_cols:
            if col in df_processed.columns:
                df_processed[col] = pd.to_numeric(df_processed[col], errors="coerce")
        
        # Sort by datetime
        df_processed = df_processed.sort_values('datetime')
        
        # Apply processing function to combine sources if needed
        df_processed = process_func(df_processed, main_sources)
        
        # Focus on the sources defined by the grouping mode
        available_sources = [col for col in main_sources if col in df_processed.columns and not df_processed[col].isnull().all()]
        
        if available_sources:
            # Calculate total volume for each time point
            df_processed['total_volume'] = df_processed[available_sources].sum(axis=1)
            order_type_data[order_type] = df_processed
            
            # Add to comparison chart
            fig_comparison.add_trace(go.Scatter(
                x=df_processed['datetime'],
                y=df_processed['total_volume'],
                mode='lines+markers',
                name=f'{order_type.title()} Orders',
                line=dict(color=order_type_colors.get(order_type, "#636EFA"), width=2),
                marker=dict(size=4),
                hovertemplate=f'<b>{order_type.title()} Orders</b><br>Date: %{{x}}<br>Volume: {hover_total_format}<extra></extra>'
            ))
    
    if order_type_data:
        fig_comparison.update_layout(
            title=f"Total Fill Volume Comparison Across Order Types ({grouping_mode} - {volume_units})",
            xaxis_title="Date",
            yaxis_title=y_axis_title,
            yaxis=dict(tickformat=axis_format),
            hovermode='x unified',
            height=500
        )
        
        st.plotly_chart(fig_comparison, use_container_width=True)
    
    # Create individual charts for each order type
    for order_type in selected_order_types:
        if order_type not in order_type_data:
            continue
            
        df_processed = order_type_data[order_type]
        order_type_color = order_type_colors.get(order_type, "#636EFA")
        
        # Get source configuration and processing based on grouping mode
        main_sources, source_config, process_func, _ = get_source_config_and_processing(grouping_mode, volume_units)
        
        # Apply processing function to combine sources if needed (already done, but get available sources)
        available_sources = [col for col in main_sources if col in df_processed.columns and not df_processed[col].isnull().all()]
        
        if not available_sources:
            st.warning(f"No valid liquidity source data for {order_type} orders")
            continue
        
        # Calculate percentages for each source
        for col in available_sources:
            df_processed[f'{col}_pct'] = (df_processed[col] / df_processed['total_volume'] * 100).fillna(0)
        
        # Create stacked area chart
        fig_order_type = go.Figure()
        
        # Add stacked area traces
        for col in available_sources:
            config = source_config.get(col, {"color": "rgba(128, 128, 128, 0.8)", "name": col})
            
            # Create custom hover text
            hover_text = [
                f"<b>{config['name']}</b><br>" +
                f"Volume: {hover_value_format.format(abs_val)}<br>" +
                f"Percentage: {pct:.1f}%<br>"
                for abs_val, pct in zip(
                    df_processed[col],
                    df_processed[f'{col}_pct']
                )
            ]
            
            fig_order_type.add_trace(go.Scatter(
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
        
        # Add total volume line
        fig_order_type.add_trace(go.Scatter(
            x=df_processed['datetime'],
            y=df_processed['total_volume'],
            mode='lines+markers',
            name='Total Volume',
            line=dict(color=order_type_color, width=3, dash='solid'),
            marker=dict(size=6, color=order_type_color),
            hovertemplate=f'<b>Total Volume</b><br>Date: %{{x}}<br>Volume: {hover_total_format}<extra></extra>',
        ))
        
        fig_order_type.update_layout(
            title=f"{order_type.title()} Orders: Liquidity Sources Breakdown ({grouping_mode} - {volume_units})",
            xaxis_title="Date",
            yaxis_title=y_axis_title.replace("Total ", ""),
            yaxis=dict(tickformat=axis_format),
            hovermode='x unified',
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            height=400
        )
        
        st.plotly_chart(fig_order_type, use_container_width=True)
        
        # Show summary statistics in an expander
        with st.expander(f"📊 {order_type.title()} Orders Summary Statistics", expanded=False):
            st.write("**Average Distribution:**")
            summary_data = {}
            for col in available_sources:
                avg_pct = df_processed[f'{col}_pct'].mean()
                min_pct = df_processed[f'{col}_pct'].min()
                max_pct = df_processed[f'{col}_pct'].max()
                avg_abs = df_processed[col].mean()
                total_abs = df_processed[col].sum()
                config = source_config.get(col, {"name": col})
                summary_data[config['name']] = {
                    'Min %': f"{min_pct:.1f}%",
                    'Max %': f"{max_pct:.1f}%", 
                    'Average %': f"{avg_pct:.1f}%",
                    'Average Volume': value_format.format(avg_abs),
                    'Total Volume': value_format.format(total_abs)
                }
            
            summary_df = pd.DataFrame(summary_data).T
            st.dataframe(summary_df)
            
            # Total volume metrics
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Min Daily", value_format.format(df_processed['total_volume'].min()))
            col2.metric("Max Daily", value_format.format(df_processed['total_volume'].max()))
            col3.metric("Average Daily", value_format.format(df_processed['total_volume'].mean()))
            col4.metric("Total Period", value_format.format(df_processed['total_volume'].sum()))


def render_liquidity_source_analysis_by_swift(start_ts, end_ts, selected_market, grouping_mode, volume_units):
    swift_types = ['0', '1']  # 0 = non-swift, 1 = swift
    swift_labels = {'0': 'Non-Swift', '1': 'Swift'}
    selected_swift_types = st.multiselect("Select Swift Types to Analyze", swift_types, default=swift_types, format_func=lambda x: swift_labels[x])
    
    # Define colors for swift types
    swift_type_colors = {
        "0": "#EF553B",       # Red for Non-Swift
        "1": "#00CC96",       # Green for Swift
    }
    
    # Determine formatting based on volume units
    if volume_units == "Counts":
        value_format = "{:,.0f}"
        axis_format = ",.0f"
        hover_value_format = "{:,.0f}"
        unit_label = ""
        y_axis_title = "Count"
        hover_total_format = "%{y:,.0f}"
    else:
        value_format = "${:,.0f}"
        axis_format = "$,.0f"
        hover_value_format = "${:,.0f}"
        unit_label = "$"
        y_axis_title = "Total Volume ($)"
        hover_total_format = "$%{y:,.0f}"
    
    # Create comparative chart showing both swift types together
    fig_comparison = go.Figure()
    swift_type_data = {}
    
    for swift_type in selected_swift_types:
        liquidity_source_data = fetch_liquidity_source_data_dynamodb(start_ts, end_ts, selected_market, bit_flag=swift_type)
        
        if liquidity_source_data is None or liquidity_source_data.empty:
            st.warning(f"No liquidity source data available for {swift_labels[swift_type]}")
            continue
        
        # Process the data
        df_processed = liquidity_source_data.copy()
        df_processed["ts"] = pd.to_numeric(df_processed["ts"], errors="coerce")
        df_processed.dropna(subset=["ts"], inplace=True)
        df_processed["datetime"] = pd.to_datetime(df_processed["ts"], unit="s")
        
        # Get source configuration and processing based on grouping mode and volume units
        main_sources, source_config, process_func, liquidity_cols = get_source_config_and_processing(grouping_mode, volume_units)
        
        # Convert liquidity source columns to numeric
        for col in liquidity_cols:
            if col in df_processed.columns:
                df_processed[col] = pd.to_numeric(df_processed[col], errors="coerce")
        
        # Sort by datetime
        df_processed = df_processed.sort_values('datetime')
        
        # Apply processing function to combine sources if needed
        df_processed = process_func(df_processed, main_sources)
        
        # Focus on the sources defined by the grouping mode
        available_sources = [col for col in main_sources if col in df_processed.columns and not df_processed[col].isnull().all()]
        
        if available_sources:
            # Calculate total volume for each time point
            df_processed['total_volume'] = df_processed[available_sources].sum(axis=1)
            swift_type_data[swift_type] = df_processed
            
            # Add to comparison chart
            fig_comparison.add_trace(go.Scatter(
                x=df_processed['datetime'],
                y=df_processed['total_volume'],
                mode='lines+markers',
                name=f'{swift_labels[swift_type]} Orders',
                line=dict(color=swift_type_colors.get(swift_type, "#636EFA"), width=2),
                marker=dict(size=4),
                hovertemplate=f'<b>{swift_labels[swift_type]} Orders</b><br>Date: %{{x}}<br>Volume: {hover_total_format}<extra></extra>'
            ))
    
    if swift_type_data:
        fig_comparison.update_layout(
            title=f"Total Fill Volume Comparison: Swift vs Non-Swift ({grouping_mode} - {volume_units})",
            xaxis_title="Date",
            yaxis_title=y_axis_title,
            yaxis=dict(tickformat=axis_format),
            hovermode='x unified',
            height=500
        )
        
        st.plotly_chart(fig_comparison, use_container_width=True)
    
    # Create individual charts for each swift type
    for swift_type in selected_swift_types:
        if swift_type not in swift_type_data:
            continue
            
        df_processed = swift_type_data[swift_type]
        swift_type_color = swift_type_colors.get(swift_type, "#636EFA")
        swift_label = swift_labels[swift_type]
        
        # Get source configuration and processing based on grouping mode
        main_sources, source_config, process_func, _ = get_source_config_and_processing(grouping_mode, volume_units)
        
        # Apply processing function to combine sources if needed (already done, but get available sources)
        available_sources = [col for col in main_sources if col in df_processed.columns and not df_processed[col].isnull().all()]
        
        if not available_sources:
            st.warning(f"No valid liquidity source data for {swift_label} orders")
            continue
        
        # Calculate percentages for each source
        for col in available_sources:
            df_processed[f'{col}_pct'] = (df_processed[col] / df_processed['total_volume'] * 100).fillna(0)
        
        # Create stacked area chart
        fig_swift_type = go.Figure()
        
        # Add stacked area traces
        for col in available_sources:
            config = source_config.get(col, {"color": "rgba(128, 128, 128, 0.8)", "name": col})
            
            # Create custom hover text
            hover_text = [
                f"<b>{config['name']}</b><br>" +
                f"Volume: {hover_value_format.format(abs_val)}<br>" +
                f"Percentage: {pct:.1f}%<br>"
                for abs_val, pct in zip(
                    df_processed[col],
                    df_processed[f'{col}_pct']
                )
            ]
            
            fig_swift_type.add_trace(go.Scatter(
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
        
        # Add total volume line
        fig_swift_type.add_trace(go.Scatter(
            x=df_processed['datetime'],
            y=df_processed['total_volume'],
            mode='lines+markers',
            name='Total Volume',
            line=dict(color=swift_type_color, width=3, dash='solid'),
            marker=dict(size=6, color=swift_type_color),
            hovertemplate=f'<b>Total Volume</b><br>Date: %{{x}}<br>Volume: {hover_total_format}<extra></extra>',
        ))
        
        fig_swift_type.update_layout(
            title=f"{swift_label} Orders: Liquidity Sources Breakdown ({grouping_mode} - {volume_units})",
            xaxis_title="Date",
            yaxis_title=y_axis_title.replace("Total ", ""),
            yaxis=dict(tickformat=axis_format),
            hovermode='x unified',
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            height=400
        )
        
        st.plotly_chart(fig_swift_type, use_container_width=True)
        
        # Show summary statistics in an expander
        with st.expander(f"📊 {swift_label} Orders Summary Statistics", expanded=False):
            st.write("**Average Distribution:**")
            summary_data = {}
            for col in available_sources:
                avg_pct = df_processed[f'{col}_pct'].mean()
                min_pct = df_processed[f'{col}_pct'].min()
                max_pct = df_processed[f'{col}_pct'].max()
                avg_abs = df_processed[col].mean()
                total_abs = df_processed[col].sum()
                config = source_config.get(col, {"name": col})
                summary_data[config['name']] = {
                    'Min %': f"{min_pct:.1f}%",
                    'Max %': f"{max_pct:.1f}%", 
                    'Average %': f"{avg_pct:.1f}%",
                    'Average Volume': value_format.format(avg_abs),
                    'Total Volume': value_format.format(total_abs)
                }
            
            summary_df = pd.DataFrame(summary_data).T
            st.dataframe(summary_df)
            
            # Total volume metrics
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Min Daily", value_format.format(df_processed['total_volume'].min()))
            col2.metric("Max Daily", value_format.format(df_processed['total_volume'].max()))
            col3.metric("Average Daily", value_format.format(df_processed['total_volume'].mean()))
            col4.metric("Total Period", value_format.format(df_processed['total_volume'].sum()))


def render_liquidity_source_analysis(start_date, end_date, selected_market, grouping_mode, volume_units):
    tab_total, tab_cohort, tab_order_type, tab_swift = st.tabs(["Total", "By Cohort", "By Order Type", "By Swift"])
    
    # Convert dates to timestamps
    start_ts = int(dt.combine(start_date, dt.min.time()).timestamp())
    end_ts = int(dt.combine(end_date, dt.max.time()).timestamp())

    with tab_total:
        render_liquidity_source_analysis_total(start_ts, end_ts, selected_market, grouping_mode, volume_units)
    with tab_cohort:
        render_liquidity_source_analysis_by_cohort(start_ts, end_ts, selected_market, grouping_mode, volume_units)
    with tab_order_type:
        render_liquidity_source_analysis_by_order_type(start_ts, end_ts, selected_market, grouping_mode, volume_units)
    with tab_swift:
        render_liquidity_source_analysis_by_swift(start_ts, end_ts, selected_market, grouping_mode, volume_units) 
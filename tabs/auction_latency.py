import os
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import List

from datafetch.api_fetch import get_trades_for_day

import streamlit as st
from constants import ALL_MARKET_NAMES

from datetime import datetime as dt, timedelta, timezone

import pandas as pd
import numpy as np
import json
import requests

from driftpy.drift_client import (
	DriftClient,
)


# set pandas width and max columns
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 150)

hide_streamlit_style = """
			<style>
			#MainMenu {visibility: hidden;}
			footer {visibility: hidden;}
			</style>
			"""
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

async def auction_latency(clearinghouse: DriftClient):
	# Read local CSV file for auction latency data
	try:
		auction_latency_df_original = pd.read_csv('may_15_auction_latency.csv')
		st.write(f"Columns in loaded CSV: {auction_latency_df_original.columns.tolist()}") # Diagnostic print

		st.write("### Percentage of Trades Filled Within Auction")

		default_excluded_users = [
			"GGbgZLcBWDKP1Trhh5vzs9hX3TxDFZPt3oaJRnKwVeB6", "45LQKTnrEcKWMCz2ECbT9RczNuDvb5Gr2JjZj1e3uJBF",
			"6YhekEW8rLpnCWty86hHck54yE5UJbpAWQqot1d8uYMG", "BEaUJkvtRDjcwb5nYeqT6WXX1uUMfDZu4y73zjKPu6nV",
			"Er1RDT6hK3ELGrw3e2sdWEpL3fGpiqbDUmrktULcfoHT", "37fQT3ycEU4HeCVe7h3pYAPZJXdCXC9gNFsfjGx4Ryxq",
			"5DxkTJDA1fx7zLyerYYFsyQnYEdgp8AdHuS8St8FfPW1", "BQ4Cvv34cTGGDGsnUzhFGkx5jFykBBYuDynCZ6e8agWG",
			"5ADqAWZ6q6i49GiR5CGc5Qg2ZWH5Es4MrLgNsWyeFUj8"
		]

		# Ensure all default users are present in the options
		# Combine unique takers and makers for options, handling potential NAs
		available_takers_makers = list(pd.concat([
		    auction_latency_df_original['taker'].dropna(),
		    auction_latency_df_original['maker'].dropna()
		]).unique())
		valid_default_excluded_users = [user for user in default_excluded_users if user in available_takers_makers]

		excluded_users = st.multiselect(
			"Exclude users (taker or maker):", # Reverted label
			options=available_takers_makers, 
			default=valid_default_excluded_users
		)

		filter_non_auction_fills = st.checkbox("Filter out fills outside of auction (auction_progress > 1)", value=True)
		if filter_non_auction_fills:
			auction_latency_df = auction_latency_df_original[auction_latency_df_original['auction_progress'] <= 1].copy()
		else:
			auction_latency_df = auction_latency_df_original.copy()

		# Apply user exclusion filter to auction_latency_df
		if excluded_users:
			auction_latency_df = auction_latency_df[
			    (~auction_latency_df['taker'].astype(str).isin(excluded_users)) &
			    (~auction_latency_df['maker'].astype(str).isin(excluded_users))
			].copy() # Use .copy() to ensure it's a new DataFrame

		# Add a selection box to filter for a specific maker
		unique_makers = ["All"] + sorted(list(auction_latency_df['maker'].dropna().unique()))
		selected_maker = st.selectbox(
			"Filter for maker:",
			options=unique_makers,
			index=0  # Default to "All"
		)

		# Apply maker filter if a specific maker is selected
		if selected_maker != "All":
			auction_latency_df = auction_latency_df[auction_latency_df['maker'] == selected_maker].copy()

		# Calculate summary statistics (now based on filtered auction_latency_df)
		summary_stats = auction_latency_df.groupby(['marketindex', 'markettype'])['auction_progress'].describe().reset_index()
		
		# Sort by count in descending order
		summary_stats = summary_stats.sort_values(by='count', ascending=False).reset_index(drop=True)

		# Calculate percentage of trades filled within auction
		within_auction_df = auction_latency_df.copy() # auction_latency_df is already filtered by users

		# The user exclusion filter previously here on within_auction_df is now redundant
		# as auction_latency_df (from which within_auction_df is copied) is already filtered.
		
		within_auction_df['filled_within_auction'] = within_auction_df['auction_progress'] <= 1
		
		filled_within_auction_summary = within_auction_df.groupby(['marketindex', 'markettype'])['filled_within_auction'].agg(['sum', 'count']).reset_index()
		filled_within_auction_summary['percentage_filled_within_auction'] = (filled_within_auction_summary['sum'] / filled_within_auction_summary['count']) * 100
		filled_within_auction_summary = filled_within_auction_summary.rename(columns={'sum': 'trades_within_auction', 'count': 'total_trades'})
		filled_within_auction_summary = filled_within_auction_summary.sort_values(by='total_trades', ascending=False).reset_index(drop=True)
		st.dataframe(filled_within_auction_summary[['marketindex', 'markettype', 'trades_within_auction', 'total_trades', 'percentage_filled_within_auction']])

		# Add a selection column and select top 10 by default
		summary_stats['Select to Plot'] = False
		top_n = min(10, len(summary_stats))
		summary_stats.loc[:top_n-1, 'Select to Plot'] = True
		
		st.write(f"Distribution of auction progress for each market `(fill_slot - order_slot)/auction_duration`. Based on fills on May 15, 2025")
		st.write(f"Summary statistics (filtering out auction only: {filter_non_auction_fills}):")
		
		col = st.columns(2)
		with col[0]: 
			edited_summary_stats = st.data_editor(
				summary_stats,
				column_config={
					"Select to Plot": st.column_config.CheckboxColumn(
						"Select to Plot",
						default=False, # Default is overridden by direct assignment above
					)
				},
				hide_index=True,
			)

			selected_markets = edited_summary_stats[edited_summary_stats['Select to Plot']]
		
		with col[1]:
			# Add a selector for x-axis scale
			x_axis_scale_options = ['linear', 'log']
			selected_x_axis_scale = st.selectbox("Select X-axis scale for histograms:", x_axis_scale_options, index=0)

			# Add a selector for y-axis scale
			y_axis_scale_options = ['linear', 'log']
			selected_y_axis_scale = st.selectbox("Select Y-axis scale for histograms:", y_axis_scale_options, index=0)


		# Plot histograms of auction_progress for selected markets
		st.write(f"### Auction Progress Histograms for Selected Markets, {selected_x_axis_scale.capitalize()} Scale X-axis), ({selected_y_axis_scale.capitalize()} Scale Y-axis")
		
		if not selected_markets.empty:
			selected_groups_data = []
			for _, row in selected_markets.iterrows():
				market_idx = row['marketindex']
				market_t = row['markettype']
				group_df = auction_latency_df[
					(auction_latency_df['marketindex'] == market_idx) & 
					(auction_latency_df['markettype'] == market_t)
				]
				selected_groups_data.append(((market_idx, market_t), group_df))

			num_selected_groups = len(selected_groups_data)

			if num_selected_groups > 0:
				cols = int(np.ceil(np.sqrt(num_selected_groups)))
				rows = int(np.ceil(num_selected_groups / cols))

				fig = make_subplots(rows=rows, cols=cols, subplot_titles=[f"Market {idx} ({m_type})" for ((idx, m_type), _) in selected_groups_data])

				row_num = 1
				col_num = 1
				for ((market_index, market_type), group_df) in selected_groups_data:
					fig.add_trace(
						go.Histogram(x=group_df['auction_progress'], name=f"Market {market_index} ({market_type})"),
						row=row_num, col=col_num
					)
					# Update y-axis to the selected scale for the current subplot
					fig.update_yaxes(type=selected_y_axis_scale, row=row_num, col=col_num)
					fig.update_xaxes(type=selected_x_axis_scale, row=row_num, col=col_num)

					col_num += 1
					if col_num > cols:
						col_num = 1
						row_num += 1
				
				fig.update_layout(height=300 * rows, showlegend=False)
				st.plotly_chart(fig, use_container_width=True)
			else:
				st.write("No markets selected to plot histograms.")
		else:
			st.write("No markets selected or available to plot histograms.")

		with st.expander("Raw Data"):
			st.write("### Auction Latency Data")
			st.dataframe(auction_latency_df)

	except FileNotFoundError:
		st.error("Could not find may_15_auction_latency.csv file")
		return
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
		auction_latency_df = pd.read_csv('may_15_auction_latency.csv')

		# Calculate summary statistics
		st.write("### Auction Progress Summary Statistics")
		summary_stats = auction_latency_df.groupby(['marketindex', 'markettype'])['auction_progress'].describe().reset_index()
		
		# Sort by count in descending order
		summary_stats = summary_stats.sort_values(by='count', ascending=False).reset_index(drop=True)

		# Add a selection column and select top 10 by default
		summary_stats['Select to Plot'] = False
		top_n = min(10, len(summary_stats))
		summary_stats.loc[:top_n-1, 'Select to Plot'] = True
		
		st.write("Distribution of auction progress for each market `(fill_slot - order_slot)/auction_duration`. Based on fills on May 15, 2025")
		st.write("Select markets from the table below to plot the distributions:")
		
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
			selected_x_axis_scale = st.selectbox("Select X-axis scale for histograms:", x_axis_scale_options, index=0) # Default to linear

			# Add a selector for y-axis scale
			y_axis_scale_options = ['linear', 'log']
			selected_y_axis_scale = st.selectbox("Select Y-axis scale for histograms:", y_axis_scale_options, index=1) # Default to log


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
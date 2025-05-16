import os
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import List, Optional

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

import time
import timeit
from solders.signature import Signature


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


markout_periods = ['t0', 't5', 't10', 't30', 't60']
markout_colors = ['#FF0000', '#00FF00', '#0000FF', '#FFA500', '#800080']


def process_trades_df(raw_trades_df: pd.DataFrame) -> pd.DataFrame:
	'''
	Adds some columns to the market_trades_df for analysis in a vectorized way
	'''
	# Select required columns
	filtered = raw_trades_df[[
		'ts', 'filler', 'fillRecordId', 'taker', 'takerOrderId', 'takerOrderDirection', 'takerFee',
		'maker', 'makerOrderId', 'makerOrderDirection', 'makerFee', 'makerRebate', 'referrerReward',
		'baseAssetAmountFilled', 'quoteAssetAmountFilled', 'oraclePrice', 'actionExplanation', 'txSig',
		'txSigIndex', 'slot', 'fillerReward', 'quoteAssetAmountSurplus', 'takerOrderBaseAssetAmount',
		'takerOrderCumulativeBaseAssetAmountFilled', 'takerOrderCumulativeQuoteAssetAmountFilled',
		'makerOrderBaseAssetAmount', 'makerOrderCumulativeBaseAssetAmountFilled',
		'makerOrderCumulativeQuoteAssetAmountFilled', 'action', 'marketIndex', 'marketType',
		'spotFulfillmentMethodFee', 'marketFilter', 'symbol'
	]].copy()

	# Convert numeric columns to float
	numeric_columns = [
		'takerFee', 'makerFee', 'baseAssetAmountFilled',
		'quoteAssetAmountFilled', 'oraclePrice', 'slot'
	]
	for col in numeric_columns:
		filtered[col] = pd.to_numeric(filtered[col], errors='coerce')

	# Vectorized operations
	filtered['makerBaseSigned'] = np.where(filtered['makerOrderDirection'] == 'long',
										   filtered['baseAssetAmountFilled'],
										   filtered['baseAssetAmountFilled'] * -1)
	filtered['makerQuoteSigned'] = np.where(filtered['makerOrderDirection'] == 'long',
											-1 * filtered['quoteAssetAmountFilled'],
											filtered['quoteAssetAmountFilled'])
	filtered['takerBaseSigned'] = np.where(filtered['takerOrderDirection'] == 'long',
										   filtered['baseAssetAmountFilled'],
										   filtered['baseAssetAmountFilled'] * -1)
	filtered['takerQuoteSigned'] = np.where(filtered['takerOrderDirection'] == 'long',
											-1 * filtered['quoteAssetAmountFilled'],
											filtered['quoteAssetAmountFilled'])
	filtered['fillPrice'] = filtered['quoteAssetAmountFilled'] / filtered['baseAssetAmountFilled']
	filtered['isFillerMaker'] = filtered['filler'] == filtered['maker']
	filtered['isFillerTaker'] = filtered['filler'] == filtered['taker']
	filtered['makerOrderDirectionNum'] = np.where(filtered['makerOrderDirection'] == 'long', 1, -1)
	filtered['takerOrderDirectionNum'] = np.where(filtered['takerOrderDirection'] == 'long', 1, -1)

	# Add markout calculations
	filtered['makerPremium'] = (filtered['fillPrice'] - filtered['oraclePrice']) * filtered['makerOrderDirectionNum']
	filtered['takerPremium'] = (filtered['fillPrice'] - filtered['oraclePrice']) * filtered['takerOrderDirectionNum']
	filtered['makerPremiumDollar'] = filtered['makerPremium'] * filtered['baseAssetAmountFilled']
	filtered['takerPremiumDollar'] = filtered['takerPremium'] * filtered['baseAssetAmountFilled']

	# Calculate future premiums for different lookahead periods
	for period in markout_periods:
		seconds = int(period[1:])  # t0 will be 0 seconds naturally
		# Create a copy of the dataframe with shifted timestamps
		filtered = filtered.sort_values('ts', ascending=True).reset_index(drop=True)
		future_df = filtered.copy()
		future_df['ts'] = future_df['ts'] - pd.Timedelta(seconds=seconds)

		# Merge with future dataframe to get future oracle prices
		merged_df = pd.merge_asof(
			filtered,
			future_df[['ts', 'oraclePrice']],
			on='ts',
			direction='forward',
			allow_exact_matches=True,
			suffixes=('', '_future')
		)

		filtered[f'oraclePrice_{period}'] = merged_df['oraclePrice_future']

		# Calculate markout premium
		filtered[f'makerPremium{period}'] = (filtered['fillPrice'] - filtered[f'oraclePrice_{period}']) * filtered['makerOrderDirectionNum'] * -1
		filtered[f'makerPremium{period}Dollar'] = filtered[f'makerPremium{period}'] * filtered['baseAssetAmountFilled']
		filtered[f'takerPremium{period}'] = (filtered['fillPrice'] - filtered[f'oraclePrice_{period}']) * filtered['takerOrderDirectionNum'] * -1
		filtered[f'takerPremium{period}Dollar'] = filtered[f'takerPremium{period}'] * filtered['baseAssetAmountFilled']

	# Convert timestamp to datetime in UTC timezone
	filtered['datetime'] = pd.to_datetime(filtered['ts'], unit='s').dt.tz_localize('UTC')
	filtered = filtered.set_index('datetime')
	filtered = filtered.sort_index()

	return filtered


def render_trades_stats_for_user_account(processed_trades_df, filter_ua):
	'''
	Plots and prints some stats for the user_account
	'''
	if filter_ua == 'vAMM':
		user_trades_df = processed_trades_df.loc[
			(processed_trades_df['maker'].isna()) | (processed_trades_df['taker'].isna())
		].copy()
		filter_ua = None
	else:
		user_trades_df = processed_trades_df.loc[
			(processed_trades_df['maker'] == filter_ua) | (processed_trades_df['taker'] == filter_ua)
		].copy()


	user_trades_df['isMaker'] = user_trades_df['maker'] == filter_ua
	user_trades_df['counterparty'] = np.where(
		user_trades_df['maker'] == filter_ua,
		user_trades_df['taker'],
		user_trades_df['maker']
	)
	user_trades_df['user'] = np.where(
		user_trades_df['isMaker'],
		user_trades_df['maker'],
		user_trades_df['taker']
	)
	user_trades_df['user_direction'] = np.where(
		user_trades_df['maker'] == filter_ua,
		user_trades_df['makerOrderDirection'],
		user_trades_df['takerOrderDirection']
	)

	user_trades_df['user_direction_num'] = np.where(
		user_trades_df['maker'] == filter_ua,
		user_trades_df['makerOrderDirectionNum'],
		user_trades_df['takerOrderDirectionNum']
	)

	user_trades_df['user_fee_recv'] = np.where(
		user_trades_df['maker'] == filter_ua,
		-1 * user_trades_df['makerFee'],
		-1 * user_trades_df['takerFee'],
	)

	user_trades_df['user_base'] = np.where(
		user_trades_df['maker'] == filter_ua,
		user_trades_df['makerBaseSigned'],
		user_trades_df['takerBaseSigned'],
	)

	user_trades_df['user_quote'] = np.where(
		user_trades_df['maker'] == filter_ua,
		user_trades_df['makerQuoteSigned'],
		user_trades_df['takerQuoteSigned'],
	)

	user_trades_df['userPremium'] = np.where(
		user_trades_df['maker'] == filter_ua,
		user_trades_df['makerPremium'],
		user_trades_df['takerPremium'],
	)

	user_trades_df['userPremiumDollar'] = np.where(
		user_trades_df['maker'] == filter_ua,
		user_trades_df['makerPremiumDollar'],
		user_trades_df['takerPremiumDollar'],
	)

	for period in markout_periods:
		user_trades_df[f'userPremium{period}'] = np.where(
			user_trades_df['maker'] == filter_ua,
			user_trades_df[f'makerPremium{period}'],
			user_trades_df[f'takerPremium{period}'],
		)
		user_trades_df[f'userPremium{period}Dollar'] = np.where(
			user_trades_df['maker'] == filter_ua,
			user_trades_df[f'makerPremium{period}Dollar'],
			user_trades_df[f'takerPremium{period}Dollar'],
		)

	user_trades_df['user_cum_base'] = user_trades_df['user_base'].cumsum() # base_position
	user_trades_df['user_cum_base_prev'] = user_trades_df['user_cum_base'].shift(1).fillna(0) # base_position_prev
	user_trades_df['user_cum_quote'] = user_trades_df['user_quote'].cumsum()

	# update types:
	# 0: flip pos
	# 1: increase pos
	# -1: decrease pos

	user_trades_df['position_update'] = 0
	user_trades_df['user_quote_entry_amount'] = 0.0
	user_trades_df['user_quote_breakeven_amount'] = 0.0
	user_trades_df['realized_pnl'] = 0.0

	for i in range(0, len(user_trades_df)):
		prev_row = user_trades_df.iloc[i - 1]
		current_row = user_trades_df.iloc[i]

		prev_quote_entry_amt = prev_row['user_quote_entry_amount']
		prev_quote_breakeven_amt = prev_row['user_quote_breakeven_amount']
		delta_base_amt = np.abs(current_row['user_base'])
		curr_base_amt = np.abs(prev_row['user_cum_base'])

		if current_row['user_cum_base'] * current_row['user_cum_base_prev'] < 0:
			# flipped direction
			user_trades_df.loc[user_trades_df.index[i], 'position_update'] = 0
			# same for BE and entry
			new_quote = current_row['user_quote'] - (current_row['user_quote'] * curr_base_amt / delta_base_amt)
			user_trades_df.loc[user_trades_df.index[i], 'user_quote_entry_amount'] = new_quote
			user_trades_df.loc[user_trades_df.index[i], 'user_quote_breakeven_amount'] = new_quote
			user_trades_df.loc[user_trades_df.index[i], 'realized_pnl'] = prev_row['user_quote_entry_amount'] + (current_row['user_quote'] - new_quote)
		elif current_row['user_cum_base_prev'] == 0:
			# opening new position
			user_trades_df.loc[user_trades_df.index[i], 'position_update'] = 1
			user_trades_df.loc[user_trades_df.index[i], 'user_quote_entry_amount'] = prev_quote_entry_amt + current_row['user_quote']
			user_trades_df.loc[user_trades_df.index[i], 'user_quote_breakeven_amount'] = prev_quote_breakeven_amt + current_row['user_quote']
		else:
			if current_row['user_direction_num'] == np.sign(current_row['user_cum_base_prev']):
				# increase position
				user_trades_df.loc[user_trades_df.index[i], 'position_update'] = 1
				user_trades_df.loc[user_trades_df.index[i], 'user_quote_entry_amount'] = prev_quote_entry_amt + current_row['user_quote']
				user_trades_df.loc[user_trades_df.index[i], 'user_quote_breakeven_amount'] = prev_quote_breakeven_amt + current_row['user_quote']
				user_trades_df.loc[user_trades_df.index[i], 'realized_pnl'] = 0
			else:
				# decrease position
				user_trades_df.loc[user_trades_df.index[i], 'position_update'] = -1
				new_quote_entry_amt = prev_quote_entry_amt - (prev_quote_entry_amt * delta_base_amt / curr_base_amt)
				user_trades_df.loc[user_trades_df.index[i], 'user_quote_entry_amount'] = new_quote_entry_amt
				user_trades_df.loc[user_trades_df.index[i], 'user_quote_breakeven_amount'] = prev_quote_breakeven_amt - (prev_quote_breakeven_amt * delta_base_amt / curr_base_amt)
				user_trades_df.loc[user_trades_df.index[i], 'realized_pnl'] = prev_row['user_quote_entry_amount'] - new_quote_entry_amt + current_row['user_quote']

	user_trades_df['avg_price'] = np.abs(user_trades_df['user_quote_entry_amount'] / user_trades_df['user_cum_base'])
	user_trades_df['user_cum_fee'] = user_trades_df['user_fee_recv'].cumsum()
	user_trades_df['user_cum_pnl'] = user_trades_df['realized_pnl'].cumsum()

	return user_trades_df


def plot_cumulative_pnl_for_user_account(user_trades_df, filter_ua):
	# First figure with time series
	fig1 = make_subplots(rows=4, cols=1, shared_xaxes=True,
		subplot_titles=('Trade Analysis', 'Cumulative Base Asset', 'Cumulative PnL', 'Markout Analysis'),
		vertical_spacing=0.05,
		row_heights=[1, 1, 1, 1],
		specs=[
			[{"secondary_y": True}],
			[{}],
			[{}],
			[{}]
		]
	)

	fig1.update_layout(
		height=1500,
		showlegend=True,
		# legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
	)

	# Add scatter plot for trades with color based on direction
	long_trades = user_trades_df[user_trades_df['user_direction'] == 'long']
	short_trades = user_trades_df[user_trades_df['user_direction'] == 'short']

	trade_marker_size = 5
	opacity = 0.75

	# Add trade visualization traces to first subplot
	fig1.add_trace(
		go.Scatter(x=long_trades.index, y=long_trades['fillPrice'],
			mode='markers+lines', name='Long Trades',
			marker=dict(size=trade_marker_size, opacity=opacity, color='#00AA3F'),
			line=dict(width=0.5, color='#00AA3F'),
			hoverinfo='text',
			hoverlabel=dict(bgcolor='rgba(0,255,127,0.8)'),
			hovertext=[f"Time: {idx}<br>Price: {price:.2f}<br>Size: {size:.4f}"
				for idx, price, size in zip(long_trades.index, long_trades['fillPrice'], abs(long_trades['user_base']))]),
		row=1, col=1, secondary_y=False
	)

	fig1.add_trace(
		go.Scatter(x=short_trades.index, y=short_trades['fillPrice'],
			mode='markers+lines', name='Short Trades',
			marker=dict(size=trade_marker_size, opacity=opacity, color='#CC0000'),
			line=dict(width=0.5, color='#CC0000'),
			hoverinfo='text',
			hoverlabel=dict(bgcolor='rgba(255,107,107,0.8)'),
			hovertext=[f"Time: {idx}<br>Price: {price:.2f}<br>Size: {size:.4f}"
				for idx, price, size in zip(short_trades.index, short_trades['fillPrice'], abs(short_trades['user_base']))]),
		row=1, col=1, secondary_y=False
	)

	fig1.add_trace(
		go.Scatter(x=user_trades_df.index, y=user_trades_df['oraclePrice'],
			mode='markers', name='Oracle',
			marker=dict(size=trade_marker_size, opacity=opacity, color='#00BFFF'),
			hoverinfo='text',
			hoverlabel=dict(bgcolor='rgba(0,191,255,0.8)'),
			hovertext=[f"Time: {idx}<br>Price: {price:.2f}"
				for idx, price in zip(user_trades_df.index, user_trades_df['oraclePrice'])]),
		row=1, col=1, secondary_y=False
	)

	# Add average price curve
	fig1.add_trace(
		go.Bar(x=long_trades.index, y=abs(long_trades['user_base']),
			name='Long Trade Size',
			marker=dict(color='#00FF7F', opacity=0.5),
			showlegend=True),
		row=1, col=1, secondary_y=True
	)

	fig1.add_trace(
		go.Bar(x=short_trades.index, y=abs(short_trades['user_base']),
			name='Short Trade Size',
			marker=dict(color='#FF6B6B', opacity=0.5),
			showlegend=True),
		row=1, col=1, secondary_y=True
	)

	marker_size = 3

	# Add cumulative base asset trace to second subplot
	fig1.add_trace(
		go.Scatter(x=user_trades_df.index, y=user_trades_df['user_cum_base'],
			mode='lines+markers', name='Cumulative Base',
			line=dict(width=1), marker=dict(size=marker_size)),
		row=2, col=1
	)

	# Add PnL traces to third subplot
	fig1.add_trace(
		go.Scatter(x=user_trades_df.index, y=user_trades_df['user_cum_fee'],
			mode='lines+markers', name='Cumulative Fee Received',
			line=dict(width=1), marker=dict(size=marker_size),
			visible='legendonly'),
		row=3, col=1
	)

	fig1.add_trace(
		go.Scatter(x=user_trades_df.index, y=user_trades_df['user_cum_pnl'],
			mode='lines+markers', name='Cumulative PnL',
			line=dict(width=1), marker=dict(size=marker_size),
			visible='legendonly'),
		row=3, col=1
	)

	fig1.add_trace(
		go.Scatter(x=user_trades_df.index, y=user_trades_df['user_cum_pnl'] - user_trades_df['user_cum_fee'],
			mode='lines+markers', name='Cumulative PnL + Fee',
			line=dict(width=1), marker=dict(size=marker_size)),
		row=3, col=1
	)

	# Add markout traces to fourth subplot
	for i, period in enumerate(markout_periods):
		# Filter out NaN values before creating histogram
		premium_data = user_trades_df[f'userPremium{period}'].dropna()

		# Only create histogram if we have valid data
		if len(premium_data) > 0:
			# Add histogram
			fig1.add_trace(
				go.Scatter(x=user_trades_df.index,
						  y=user_trades_df[f'userPremium{period}Dollar'].cumsum(),
						  mode='lines+markers',
						  name=f'Markout {period}',
						  line=dict(width=1, color=markout_colors[i]),
						  marker=dict(size=marker_size)),
				row=4, col=1
			)

	# Second figure for histograms
	fig2 = make_subplots(rows=1, cols=len(markout_periods),
		subplot_titles=[f'Markout {period}' for period in markout_periods],
		horizontal_spacing=0.05
	)


	for i, period in enumerate(markout_periods):
		# Filter out NaN values before creating histogram
		premium_data = user_trades_df[f'userPremium{period}'].dropna()

		# Only create histogram if we have valid data
		if len(premium_data) > 0:
			# Add histogram
			fig2.add_trace(
				go.Histogram(
					x=premium_data,  # Use filtered data
					name=f'Markout {period}',
					nbinsx=100,
					marker_color=markout_colors[i],
					opacity=0.5,
					showlegend=False
				),
				row=1, col=i+1
			)

			# Calculate metrics on filtered data
			mean_val = premium_data.mean()
			std_val = premium_data.std()
			median_val = premium_data.median()
			skew_val = premium_data.skew()

			# Get histogram data to find max y value
			hist, bins = np.histogram(premium_data, bins=100, density=False)
			max_y = np.max(hist) if len(hist) > 0 else 0
			max_x = np.max(bins) if len(bins) > 0 else 0

			# Add metrics as annotations
			fig2.add_annotation(
				text=f"Mean: {mean_val:.4f}<br>Std: {std_val:.4f}<br>Median: {median_val:.4f}<br>Skew: {skew_val:.4f}",
				xref=f"x{i+1}",
				yref=f"y{i+1}",
				x=max_x,
				y=max_y,
				showarrow=False,
				align="right",
				row=1,
				col=i+1
			)
		else:
			# Add empty subplot with message if no valid data
			fig2.add_annotation(
				text="No valid data",
				xref=f"x{i+1}",
				yref=f"y{i+1}",
				x=0.5,
				y=0.5,
				showarrow=False,
				row=1,
				col=i+1
			)

	fig2.update_layout(
		title="Markout Distributions",
		height=400,
		showlegend=False,
		xaxis_title="Premium",
		yaxis_title="Frequency",
		plot_bgcolor='white',  # Light mode background
		paper_bgcolor='white'
	)

	# Update all x and y axes to have consistent ranges
	for i in range(1, len(markout_periods) + 1):
		fig2.update_xaxes(title_text="Premium", row=1, col=i)
		fig2.update_yaxes(title_text="Frequency", row=1, col=i)
		# Set light mode grid
		fig2.update_xaxes(showgrid=True, gridwidth=1, gridcolor='LightGrey', row=1, col=i)
		fig2.update_yaxes(showgrid=True, gridwidth=1, gridcolor='LightGrey', row=1, col=i)

	# Display both figures
	st.plotly_chart(fig1, use_container_width=True)
	st.plotly_chart(fig2, use_container_width=True)

	# Display metrics in columns below the plot
	with st.expander("Markout Statistics"):
		# Create a dataframe with markout statistics
		markout_stats = pd.DataFrame()
		for period in markout_periods:
			markout_stats[period] = [
				user_trades_df[f'userPremium{period}'].mean(),
				user_trades_df[f'userPremium{period}'].std(),
				user_trades_df[f'userPremium{period}'].median(),
				user_trades_df[f'userPremium{period}'].skew()
			]

		markout_stats.index = ['Mean', 'Std', 'Median', 'Skew']
		markout_stats = markout_stats.round(4)

		st.dataframe(markout_stats)

	# Add counterparty analysis expander
	with st.expander("Counterparty Analysis"):
		# Calculate fill volumes by counterparty
		counterparty_volumes = user_trades_df.groupby('counterparty')['user_quote'].apply(lambda x: abs(x).sum()).reset_index()
		counterparty_volumes.columns = ['Counterparty', 'Fill Volume']

		# Calculate total volume and percentages
		total_volume = counterparty_volumes['Fill Volume'].sum()
		counterparty_volumes['Percentage'] = (counterparty_volumes['Fill Volume'] / total_volume * 100).round(2)

		# Sort by volume
		counterparty_volumes = counterparty_volumes.sort_values('Fill Volume', ascending=False)

		# Add total row
		total_row = pd.DataFrame({
			'Counterparty': ['TOTAL'],
			'Fill Volume': [total_volume],
			'Percentage': [100.00]
		})
		counterparty_volumes = pd.concat([counterparty_volumes, total_row], ignore_index=True)

		# Format numbers
		counterparty_volumes['Fill Volume'] = counterparty_volumes['Fill Volume'].round(2)

		# Display the table
		st.dataframe(counterparty_volumes)

		# Add pie chart for counterparty distribution
		fig_counterparty = go.Figure(data=[go.Pie(
			labels=counterparty_volumes['Counterparty'].iloc[:-1],  # Exclude total row
			values=counterparty_volumes['Fill Volume'].iloc[:-1],
			hole=.3
		)])
		fig_counterparty.update_layout(
			title="Distribution of Fill Volumes by Counterparty",
			height=400,
			showlegend=True
		)
		st.plotly_chart(fig_counterparty, use_container_width=True)

		# Add fill methods breakdown by counterparty
		st.write("### Fill Methods by Counterparty")

		# Calculate fill methods by counterparty
		fill_methods = user_trades_df.groupby(['counterparty', 'spotFulfillmentMethodFee'])['user_quote'].apply(lambda x: abs(x).sum()).reset_index()
		fill_methods.columns = ['Counterparty', 'Fill Method', 'Volume']

		# Calculate percentages within each counterparty
		total_by_counterparty = fill_methods.groupby('Counterparty')['Volume'].transform('sum')
		fill_methods['Percentage'] = (fill_methods['Volume'] / total_by_counterparty * 100).round(2)

		# Sort by volume within each counterparty
		fill_methods = fill_methods.sort_values(['Counterparty', 'Volume'], ascending=[True, False])

		# Create pivot table for volumes
		volume_pivot = fill_methods.pivot(index='Counterparty', columns='Fill Method', values='Volume').fillna(0)

		# Create pivot table for percentages
		pct_pivot = fill_methods.pivot(index='Counterparty', columns='Fill Method', values='Percentage').fillna(0)

		# Combine the two pivots
		combined_pivot = pd.DataFrame()
		for method in volume_pivot.columns:
			combined_pivot[f"{method} (Volume)"] = volume_pivot[method].round(2)
			combined_pivot[f"{method} (%)"] = pct_pivot[method].round(2)

		# Add total volume column
		combined_pivot['Total Volume'] = volume_pivot.sum(axis=1).round(2)

		# Sort by total volume
		combined_pivot = combined_pivot.sort_values('Total Volume', ascending=False)

		# Add total row
		total_row = pd.Series({
			'Total Volume': combined_pivot['Total Volume'].sum()
		})
		for method in volume_pivot.columns:
			total_row[f"{method} (Volume)"] = volume_pivot[method].sum()
			total_row[f"{method} (%)"] = 100.0 if volume_pivot[method].sum() > 0 else 0.0

		combined_pivot.loc['TOTAL'] = total_row

		st.dataframe(combined_pivot)

		# Add stacked bar chart for fill methods
		fig_methods = go.Figure()

		# Get unique fill methods
		methods = fill_methods['Fill Method'].unique()

		for method in methods:
			method_data = fill_methods[fill_methods['Fill Method'] == method]
			fig_methods.add_trace(go.Bar(
				name=method,
				x=method_data['Counterparty'],
				y=method_data['Volume']
			))

		fig_methods.update_layout(
			title="Fill Methods Distribution by Counterparty",
			barmode='stack',
			height=400,
			showlegend=True
		)
		st.plotly_chart(fig_methods, use_container_width=True)


'''
resp from jito:
[
  {
    "bundle_id": "0ab6f798a0f85945b69b55f1b88573df58711af7f25d2a34edfab47e46153d7c"
  }
]
'''
def get_jito_bundle_id(tx_sig: str) -> Optional[str]:
	try:
		url = f"https://bundles.jito.wtf/api/v1/bundles/transaction/{tx_sig}"
		response = requests.get(url)
		data = response.json()
		if 'error' in data:
			return None
		if len(data) > 0:
			return data[0]['bundle_id']
	except Exception as e:
		st.error(f"Error getting jito bundle id: {e}")
		return None


'''
resp from jito:
[
  {
    "bundleId": "e2c30cc1cd9c3ed11dcd23ee0123ee801654f2782541886aa1f5c50b537aa298",
    "slot": 340282457,
    "validator": "CW9C7HBwAMgqNdXkNgFg9Ujr3edR2Ab9ymEuQnVacd1A",
    "tippers": [
      "5GMbJEjRJvQNiGRJzKud3ncTw11Efumpa6VQTJeGogbP"
    ],
    "landedTipLamports": 10000,
    "landedCu": 274241,
    "blockIndex": 1617,
    "timestamp": "2025-05-15T23:59:59+00:00",
    "txSignatures": [
      "57BxAwFoPV6MPk3P2kkj3zXL6SJXwxzYBcUb5vv2vtGmH5HgdhDVHL7cxNtgeerGy35ZR5zDhAS7Y7hQ8mFCkac"
    ]
  }
]
'''
def get_jito_bundle_data(bundle_id: str) -> Optional[dict]:
	try:
		url = f"https://bundles.jito.wtf/api/v1/bundles/bundle/{bundle_id}"
		response = requests.get(url)
		data = response.json()
		if 'error' in data:
			return None
		if len(data) > 0:
			return data[0]
		return None
	except Exception as e:
		st.error(f"Error getting jito bundle data: {e}")
		return None


async def maker_tx_landing_analysis(clearinghouse: DriftClient):
	cols = st.columns(2)
	with cols[0]:
		date_picker = st.date_input("Select a date")
	with cols[1]:
		market_symbol = st.selectbox("Select a market symbol", ALL_MARKET_NAMES, index=None)

	if market_symbol is None:
		st.write("Select a market first")
		return

	# Create a session key based on market symbol and date
	session_key = f"{market_symbol}-{date_picker.year}-{date_picker.month}-{date_picker.day}"

	# Only fetch data if a date is selected
	if date_picker:
		# Check if data is already in session
		if session_key in st.session_state:
			market_trades_df = st.session_state[session_key]
		else:
			try:
				market_trades_df = get_trades_for_day(market_symbol, date_picker.year, date_picker.month, date_picker.day)
				# Save to session state
				st.session_state[session_key] = market_trades_df
			except Exception as e:
				st.error(f"Error loading {market_symbol} trades for {date_picker} from s3, try a different date: {e}")
				return
		market_trades_df = market_trades_df.sort_values(by='ts', ascending=False)
		all_tx_sigs = market_trades_df['txSig'].unique()
		st.write(f"Total unique transactions for the day: {len(all_tx_sigs)}")

		cols_batch = st.columns(2)
		with cols_batch[0]:
			batch_size = st.number_input("Number of transactions to process (Batch Size)", min_value=1, max_value=len(all_tx_sigs), value=min(50, len(all_tx_sigs)), step=10)
		with cols_batch[1]:
			offset = st.number_input("Transaction offset", min_value=0, max_value=len(all_tx_sigs) - batch_size, value=0, step=10)

		tx_sigs_to_process = all_tx_sigs[offset : offset + batch_size]

		if not tx_sigs_to_process.size:
			st.warning("No transactions selected with the current offset and batch size.")
			return
		
		st.write(f"Processing {len(tx_sigs_to_process)} transactions from offset {offset} (most recent first).")

		connection = clearinghouse.program.provider.connection

		tx_done = 0
		total_txs_in_batch = len(tx_sigs_to_process)
		progress_bar = st.progress(0)
		status_text = st.empty()
		start_total_time = timeit.default_timer()

		for tx_sig in tx_sigs_to_process:
			start_iter_time = timeit.default_timer()
			# st.write(tx_sig)
			tx = await connection.get_transaction(Signature.from_string(tx_sig), 'json', None, 0)
			# st.json(tx.to_json(), expanded=False)

			market_trades_df.loc[market_trades_df['txSig'] == tx_sig, 'txFeePaid'] = tx.value.transaction.meta.fee
			market_trades_df.loc[market_trades_df['txSig'] == tx_sig, 'txComputeUnitsConsumed'] = tx.value.transaction.meta.compute_units_consumed
			# market_trades_df.loc[market_trades_df['txSig'] == tx_sig, 'txSigner'] = str(tx.value.transaction.transaction.account_keys[0])

			jito_bundle_id = get_jito_bundle_id(tx_sig)
			market_trades_df.loc[market_trades_df['txSig'] == tx_sig, 'jitoBundleId'] = jito_bundle_id

			# st.write(jito_bundle_id)
			if jito_bundle_id is not None:
				bundle_data = get_jito_bundle_data(jito_bundle_id)
				if bundle_data is not None:
					market_trades_df.loc[market_trades_df['txSig'] == tx_sig, 'jitoLandedTipLamports'] = bundle_data['landedTipLamports']
					market_trades_df.loc[market_trades_df['txSig'] == tx_sig, 'jitoLandedCu'] = bundle_data['landedCu']
					market_trades_df.loc[market_trades_df['txSig'] == tx_sig, 'jitoblockIndex'] = bundle_data['blockIndex']
					market_trades_df.loc[market_trades_df['txSig'] == tx_sig, 'jitoTippers'] = json.dumps(bundle_data['tippers'])
					market_trades_df.loc[market_trades_df['txSig'] == tx_sig, 'jitoValidator'] = bundle_data['validator']
					# st.json(bundle_data, expanded=False)

			iter_time = timeit.default_timer() - start_iter_time
			
			tx_done += 1
			progress_bar.progress(tx_done / total_txs_in_batch)

			elapsed_total_time = timeit.default_timer() - start_total_time
			avg_time_per_tx = elapsed_total_time / tx_done
			remaining_txs_in_batch = total_txs_in_batch - tx_done
			estimated_remaining_time_seconds = avg_time_per_tx * remaining_txs_in_batch
			
			status_text.caption(f"Processed {tx_done}/{total_txs_in_batch} transactions in this batch. " 
								 f"Avg time/tx: {avg_time_per_tx:.2f}s. "
								 f"Est. time remaining for batch: {timedelta(seconds=int(estimated_remaining_time_seconds))}")
			
			# time.sleep(1) # Maintained original sleep, consider if this is for rate limiting or can be removed/adjusted
			# if tx_done > 10: # Maintained original break condition for testing
			# 	break
		
		
		
		progress_bar.empty()
		status_text.empty()

		# --- NEW SECTION FOR BATCH STATISTICS ---
		# Filter the main dataframe for the transactions that were just processed
		processed_mask = market_trades_df['txSig'].isin(tx_sigs_to_process)
		# Contains all trades for the processed tx_sigs
		processed_trades_in_batch_df = market_trades_df[processed_mask].copy() 

		st.subheader(f"Statistics for Processed Batch (Offset: {offset}, Size: {len(tx_sigs_to_process)})")

		if processed_trades_in_batch_df.empty:
			st.write("No transaction data processed in this batch to display statistics.")
		else:
			# Get unique transaction data from the processed batch
			# These columns are per-transaction, so drop_duplicates on txSig is appropriate
			unique_tx_in_batch_df = processed_trades_in_batch_df.drop_duplicates(subset=['txSig']).copy()

			num_tx_processed_in_batch = len(unique_tx_in_batch_df)
			
			# Jito transactions count
			num_jito_tx = unique_tx_in_batch_df['jitoBundleId'].notna().sum()

			st.write(f"- Unique transactions processed in this batch: {num_tx_processed_in_batch}")
			st.write(f"- Jito transactions in this batch: {num_jito_tx} ({num_jito_tx*100.0/num_tx_processed_in_batch if num_tx_processed_in_batch > 0 else 0:.2f}%)")

			# Prepare data for histograms using unique_tx_in_batch_df
			compute_units = pd.to_numeric(unique_tx_in_batch_df['txComputeUnitsConsumed'], errors='coerce').dropna()
			fee_paid = pd.to_numeric(unique_tx_in_batch_df['txFeePaid'], errors='coerce').dropna()
			
			# For Jito tips, filter for Jito transactions first
			jito_tx_data = unique_tx_in_batch_df[unique_tx_in_batch_df['jitoBundleId'].notna()]
			jito_tips = pd.to_numeric(jito_tx_data['jitoLandedTipLamports'], errors='coerce').dropna()
			
			# Combined fee (txFeePaid + jitoLandedTipLamports or 0)
			unique_tx_in_batch_df['txFeePaidNumeric'] = pd.to_numeric(unique_tx_in_batch_df['txFeePaid'], errors='coerce').fillna(0)
			unique_tx_in_batch_df['jitoLandedTipLamportsNumeric'] = pd.to_numeric(unique_tx_in_batch_df['jitoLandedTipLamports'], errors='coerce').fillna(0)
			
			unique_tx_in_batch_df['totalEffectiveFee'] = unique_tx_in_batch_df['txFeePaidNumeric'] + unique_tx_in_batch_df['jitoLandedTipLamportsNumeric']
			total_effective_fee = unique_tx_in_batch_df['totalEffectiveFee'].dropna()

			# Plotting using subplots
			fig_hist = make_subplots(
				rows=1, cols=4, 
				subplot_titles=(
					"txComputeUnitsConsumed", 
					"txFeePaid (lamports)", 
					"jitoLandedTipLamports (Jito Txs)", 
					"Total Effective Fee (lamports)"
				),
				horizontal_spacing=0.05 # Adjusted for 1x4 layout
			)

			# Histogram 1: txComputeUnitsConsumed
			if not compute_units.empty:
				fig_hist.add_trace(go.Histogram(x=compute_units, name="Compute Units"), row=1, col=1)
			else:
				fig_hist.add_annotation(text="No data", xref="x domain", yref="y domain", x=0.5, y=0.5, showarrow=False, row=1, col=1)
			fig_hist.update_xaxes(title_text="Compute Units", row=1, col=1)
			fig_hist.update_yaxes(title_text="Frequency", row=1, col=1)

			# Histogram 2: txFeePaid
			if not fee_paid.empty:
				fig_hist.add_trace(go.Histogram(x=fee_paid, name="Fee Paid"), row=1, col=2)
			else:
				fig_hist.add_annotation(text="No data", xref="x domain", yref="y domain", x=0.5, y=0.5, showarrow=False, row=1, col=2)
			fig_hist.update_xaxes(title_text="Fee Paid (lamports)", row=1, col=2)
			fig_hist.update_yaxes(title_text="Frequency", row=1, col=2)

			# Histogram 3: jitoLandedTipLamports
			if not jito_tips.empty:
				fig_hist.add_trace(go.Histogram(x=jito_tips, name="Jito Tip"), row=1, col=3)
			else:
				fig_hist.add_annotation(text="No Jito tips data", xref="x domain", yref="y domain", x=0.5, y=0.5, showarrow=False, row=1, col=3)
			fig_hist.update_xaxes(title_text="Jito Tip (lamports)", row=1, col=3)
			fig_hist.update_yaxes(title_text="Frequency", row=1, col=3)

			# Histogram 4: totalEffectiveFee
			if not total_effective_fee.empty:
				fig_hist.add_trace(go.Histogram(x=total_effective_fee, name="Total Effective Fee"), row=1, col=4)
			else:
				fig_hist.add_annotation(text="No data", xref="x domain", yref="y domain", x=0.5, y=0.5, showarrow=False, row=1, col=4)
			fig_hist.update_xaxes(title_text="Total Effective Fee (lamports)", row=1, col=4)
			fig_hist.update_yaxes(title_text="Frequency", row=1, col=4)

			fig_hist.update_layout(
				height=400, # Adjusted height for 1x4 layout
				showlegend=False,
				title_text="Batch Transaction Metrics Overview",
				plot_bgcolor='white',
				paper_bgcolor='white'
			)
			fig_hist.update_xaxes(showgrid=True, gridwidth=1, gridcolor='LightGrey')
			fig_hist.update_yaxes(showgrid=True, gridwidth=1, gridcolor='LightGrey')

			st.plotly_chart(fig_hist, use_container_width=True)

		# --- END OF NEW SECTION ---

		st.write(f"# market_trades_df")
		st.write(market_trades_df.sort_values(by='ts', ascending=False))

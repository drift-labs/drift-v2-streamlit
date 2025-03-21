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


markout_periods = ['t0', 't5', 't10', 't30', 't60']


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
	colors = ['#FF0000', '#00FF00', '#0000FF', '#FFA500', '#800080']
	for i, period in enumerate(markout_periods):
		fig1.add_trace(
			go.Scatter(x=user_trades_df.index,
					  y=user_trades_df[f'userPremium{period}Dollar'].cumsum(),
					  mode='lines+markers',
					  name=f'Markout {period}',
					  line=dict(width=1, color=colors[i]),
					  marker=dict(size=marker_size)),
			row=4, col=1
		)

	# Second figure for histograms
	fig2 = make_subplots(rows=1, cols=len(markout_periods),
		subplot_titles=[f'Markout {period}' for period in markout_periods],
		horizontal_spacing=0.05
	)

	colors = ['#FF0000', '#00FF00', '#0000FF', '#FFA500', '#800080']

	for i, period in enumerate(markout_periods):
		# Add histogram
		fig2.add_trace(
			go.Histogram(
				x=user_trades_df[f'userPremium{period}'],
				name=f'Markout {period}',
				nbinsx=100,
				# histnorm='probability',
				# histnorm='probability density',
				marker_color=colors[i],
				opacity=0.5,
				showlegend=False
			),
			row=1, col=i+1
		)

		# Add vertical line at x=0
		fig2.add_vline(x=0, line_dash="dash", line_color="black", row=1, col=i+1)

		# Calculate metrics
		mean_val = user_trades_df[f'userPremium{period}'].mean()
		std_val = user_trades_df[f'userPremium{period}'].std()
		median_val = user_trades_df[f'userPremium{period}'].median()
		skew_val = user_trades_df[f'userPremium{period}'].skew()

		# Get histogram data to find max y value
		hist, bins = np.histogram(user_trades_df[f'userPremium{period}'], bins=100, density=False)
		max_y = np.max(hist)
		max_x = np.max(bins)

		# Add metrics as annotations in top right
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


async def post_trade_analysis(clearinghouse: DriftClient):
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

		processed_trades_df = process_trades_df(market_trades_df)

		# Extract unique user accounts from the data
		unique_makers = set(market_trades_df['maker'].dropna().unique())
		unique_takers = set(market_trades_df['taker'].dropna().unique())
		unique_accounts = list(unique_makers.union(unique_takers)) + ['vAMM']

		# Let user select from the accounts found in the data
		filtered_uas = st.multiselect("Select user accounts", unique_accounts)

		for filtered_ua in filtered_uas:
			st.write(f"# User Account: {filtered_ua}")
			users_trades = render_trades_stats_for_user_account(processed_trades_df, filtered_ua)

			# Create columns for filter and distribution
			filter_col, dist_col, action_col= st.columns([1, 1, 1])

			with filter_col:
				# Add actionExplanation filter
				action_explanations = users_trades['actionExplanation'].unique()
				selected_action = st.radio("Filter by Action", ['All'] + list(action_explanations), key=f"action_{filtered_ua}")
				if selected_action != 'All':
					users_trades = users_trades[users_trades['actionExplanation'] == selected_action]

			# Calculate distribution of quote values by action
			action_quotes = users_trades.groupby('actionExplanation')['user_quote'].apply(lambda x: abs(x).sum()).reset_index()
			action_quotes.columns = ['Action', 'Total Quote Value']
			action_quotes['Total Quote %'] = action_quotes['Total Quote Value'] / action_quotes['Total Quote Value'].sum() * 100

			with dist_col:

				# Create pie chart
				fig_pie = go.Figure(data=[go.Pie(
					labels=action_quotes['Action'],
					values=action_quotes['Total Quote Value'],
					hole=.3
				)])
				fig_pie.update_layout(
					title="Distribution of Fill Volumeby Action",
					height=300,
					showlegend=True
				)
				st.plotly_chart(fig_pie, use_container_width=True)

			with action_col:
				# Show the data in a table below the pie chart
				st.dataframe(action_quotes)

			plot_cumulative_pnl_for_user_account(users_trades, filtered_ua)

		with st.expander("Show raw data"):
			st.write(f"# market_trades_df")
			st.write(market_trades_df)
			st.write(f"# processed_trades_df")
			st.write(processed_trades_df)
			if filtered_uas:
				st.write(f"# users_trades")
				st.write(users_trades)
				st.write(f"# users_trades minified")

				columns = [
					'ts', 'user_direction', 'user_base', 'user_cum_base',
					'realized_pnl', 'user_cum_pnl', 'user', 'counterparty',
					'fillPrice', 'oraclePrice'
				]
				[columns.append(f'oraclePrice_{period}') for period in markout_periods]
				columns.extend([
					'userPremium', 'userPremiumDollar',

				])
				[columns.append(f'userPremium{period}Dollar') for period in markout_periods]

				users_trades_minified = users_trades[columns]
				st.write(users_trades_minified)

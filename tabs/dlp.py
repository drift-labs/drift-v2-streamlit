
from typing import Tuple
from driftpy.constants.numeric_constants import *
from driftpy.types import PerpMarketAccount

from driftpy.math.amm import calculate_spread
PERCENTAGE_PRECISION = 10**6
DEFAULT_REVENUE_SINCE_LAST_FUNDING_SPREAD_RETREAT = 100

def cap_to_max_spread(
    long_spread: int,
    short_spread: int,
    max_spread: int
) -> Tuple[int, int]:
    total_spread = long_spread + short_spread

    if total_spread > max_spread:
        if long_spread > short_spread:
            long_spread = (long_spread * max_spread + total_spread - 1) // total_spread
            short_spread = max_spread - long_spread
        else:
            short_spread = (short_spread * max_spread + total_spread - 1) // total_spread
            long_spread = max_spread - short_spread

    new_total_spread = long_spread + short_spread

    assert new_total_spread <= max_spread, f"new_total_spread({new_total_spread}) > max_spread({max_spread})"

    return long_spread, short_spread

def calculate_spread_revenue_retreat_amount(
    base_spread: int,
    max_spread: int,
    net_revenue_since_last_funding: int
) -> int:
    revenue_retreat_amount = 0

    if net_revenue_since_last_funding < DEFAULT_REVENUE_SINCE_LAST_FUNDING_SPREAD_RETREAT:
        max_retreat = max_spread // 10

        if net_revenue_since_last_funding >= DEFAULT_REVENUE_SINCE_LAST_FUNDING_SPREAD_RETREAT * 1000:
            revenue_retreat_amount = min(
                max_retreat,
                (base_spread * abs(net_revenue_since_last_funding)) // abs(DEFAULT_REVENUE_SINCE_LAST_FUNDING_SPREAD_RETREAT)
            )
        else:
            revenue_retreat_amount = max_retreat

    return revenue_retreat_amount

def calculate_spread_inventory_scale(
    base_asset_amount_with_amm: int,
    base_asset_reserve: int,
    min_base_asset_reserve: int,
    max_base_asset_reserve: int,
    directional_spread: int,
    max_spread: int
) -> int:
    if base_asset_amount_with_amm == 0:
        return BID_ASK_SPREAD_PRECISION

    amm_inventory_pct = calculate_inventory_liquidity_ratio(
        base_asset_amount_with_amm,
        base_asset_reserve,
        min_base_asset_reserve,
        max_base_asset_reserve
    )

    inventory_scale_max = max(MAX_BID_ASK_INVENTORY_SKEW_FACTOR, max_spread * BID_ASK_SPREAD_PRECISION // max(directional_spread, 1))

    inventory_scale_capped = min(
        inventory_scale_max,
        BID_ASK_SPREAD_PRECISION + (inventory_scale_max * amm_inventory_pct // PERCENTAGE_PRECISION)
    )

    return inventory_scale_capped

def calculate_spread_leverage_scale(
    quote_asset_reserve: int,
    terminal_quote_asset_reserve: int,
    peg_multiplier: int,
    base_asset_amount_with_amm: int,
    reserve_price: int,
    total_fee_minus_distributions: int
) -> int:
    AMM_TIMES_PEG_TO_QUOTE_PRECISION_RATIO_I128 = 1e9 * 1e3 / 1e6
    AMM_TO_QUOTE_PRECISION_RATIO_I128 = 1e3
    net_base_asset_value = (
        (quote_asset_reserve - terminal_quote_asset_reserve) *
        peg_multiplier *
        AMM_TIMES_PEG_TO_QUOTE_PRECISION_RATIO_I128 // AMM_TO_QUOTE_PRECISION_RATIO_I128
    )

    local_base_asset_value = (
        base_asset_amount_with_amm *
        reserve_price *
        AMM_TO_QUOTE_PRECISION_RATIO_I128 *
        PRICE_PRECISION // AMM_TO_QUOTE_PRECISION_RATIO_I128
    )

    effective_leverage = max(0, local_base_asset_value - net_base_asset_value) * BID_ASK_SPREAD_PRECISION // (max(0, total_fee_minus_distributions) + 1)

    effective_leverage_capped = min(
        MAX_BID_ASK_INVENTORY_SKEW_FACTOR,
        BID_ASK_SPREAD_PRECISION + int(max(0, effective_leverage) + 1)
    )

    return effective_leverage_capped

def calculate_long_short_vol_spread(
    last_oracle_conf_pct: int,
    reserve_price: int,
    mark_std: int,
    oracle_std: int,
    long_intensity_volume: int,
    short_intensity_volume: int,
    volume_24h: int
) -> Tuple[int, int]:
    market_avg_std_pct = (oracle_std + mark_std) * PERCENTAGE_PRECISION // (2 * reserve_price)

    vol_spread = max(last_oracle_conf_pct, market_avg_std_pct // 2)

    factor_clamp_min = PERCENTAGE_PRECISION // 100  # .01
    factor_clamp_max = 16 * PERCENTAGE_PRECISION // 10  # 1.6

    long_vol_spread_factor = max(
        factor_clamp_min,
        min(factor_clamp_max, long_intensity_volume * PERCENTAGE_PRECISION // max(volume_24h, 1))
    )
    short_vol_spread_factor = max(
        factor_clamp_min,
        min(factor_clamp_max, short_intensity_volume * PERCENTAGE_PRECISION // max(volume_24h, 1))
    )

    return (
        max(last_oracle_conf_pct, (vol_spread * long_vol_spread_factor) // PERCENTAGE_PRECISION),
        max(last_oracle_conf_pct, (vol_spread * short_vol_spread_factor) // PERCENTAGE_PRECISION)
    )


def _calculate_market_open_bids_asks(base_asset_reserve, min_base_asset_reserve, max_base_asset_reserve):
    bids = max_base_asset_reserve-base_asset_reserve
    asks = base_asset_reserve - min_base_asset_reserve
    return bids,asks

def calculate_inventory_liquidity_ratio(
    base_asset_amount_with_amm: int,
    base_asset_reserve: int,
    min_base_asset_reserve: int,
    max_base_asset_reserve: int
) -> int:
    max_bids, max_asks = _calculate_market_open_bids_asks(base_asset_reserve, min_base_asset_reserve, max_base_asset_reserve)

    min_side_liquidity = min(max_bids, abs(max_asks))

    if abs(base_asset_amount_with_amm) < min_side_liquidity:
        amm_inventory_pct = (abs(base_asset_amount_with_amm) * PERCENTAGE_PRECISION) // max(min_side_liquidity, 1)
        amm_inventory_pct = min(amm_inventory_pct, PERCENTAGE_PRECISION)
    else:
        amm_inventory_pct = PERCENTAGE_PRECISION  # 100%

    return amm_inventory_pct

def calculate_spread_local(
    base_spread: int,
    last_oracle_reserve_price_spread_pct: int,
    last_oracle_conf_pct: int,
    max_spread: int,
    quote_asset_reserve: int,
    terminal_quote_asset_reserve: int,
    peg_multiplier: int,
    base_asset_amount_with_amm: int,
    reserve_price: int,
    total_fee_minus_distributions: int,
    net_revenue_since_last_funding: int,
    base_asset_reserve: int,
    min_base_asset_reserve: int,
    max_base_asset_reserve: int,
    mark_std: int,
    oracle_std: int,
    long_intensity_volume: int,
    short_intensity_volume: int,
    volume_24h: int
) -> Tuple[int, int]:
    
    long_vol_spread, short_vol_spread = calculate_long_short_vol_spread(
        last_oracle_conf_pct,
        reserve_price,
        mark_std,
        oracle_std,
        long_intensity_volume,
        short_intensity_volume,
        volume_24h
    )

    long_spread = max(base_spread // 2, long_vol_spread)
    short_spread = max(base_spread // 2, short_vol_spread)

    max_target_spread = max(max_spread, abs(last_oracle_reserve_price_spread_pct))

    if last_oracle_reserve_price_spread_pct < 0:
        long_spread = max(long_spread, abs(last_oracle_reserve_price_spread_pct) + long_vol_spread)
    elif last_oracle_reserve_price_spread_pct > 0:
        short_spread = max(short_spread, abs(last_oracle_reserve_price_spread_pct) + short_vol_spread)

    inventory_scale_capped = calculate_spread_inventory_scale(
        base_asset_amount_with_amm,
        base_asset_reserve,
        min_base_asset_reserve,
        max_base_asset_reserve,
        long_spread if base_asset_amount_with_amm > 0 else short_spread,
        max_target_spread
    )

    if base_asset_amount_with_amm > 0:
        long_spread = (long_spread * inventory_scale_capped) // BID_ASK_SPREAD_PRECISION
    elif base_asset_amount_with_amm < 0:
        short_spread = (short_spread * inventory_scale_capped) // BID_ASK_SPREAD_PRECISION

    if total_fee_minus_distributions <= 0:
        long_spread = (long_spread * DEFAULT_LARGE_BID_ASK_FACTOR) // BID_ASK_SPREAD_PRECISION
        short_spread = (short_spread * DEFAULT_LARGE_BID_ASK_FACTOR) // BID_ASK_SPREAD_PRECISION
    else:
        effective_leverage_capped = calculate_spread_leverage_scale(
            quote_asset_reserve,
            terminal_quote_asset_reserve,
            peg_multiplier,
            base_asset_amount_with_amm,
            reserve_price,
            total_fee_minus_distributions
        )

        if base_asset_amount_with_amm > 0:
            long_spread = (long_spread * effective_leverage_capped) // BID_ASK_SPREAD_PRECISION
        elif base_asset_amount_with_amm < 0:
            short_spread = (short_spread * effective_leverage_capped) // BID_ASK_SPREAD_PRECISION

    revenue_retreat_amount = calculate_spread_revenue_retreat_amount(
        base_spread,
        max_target_spread,
        net_revenue_since_last_funding
    )

    if revenue_retreat_amount != 0:
        if base_asset_amount_with_amm > 0:
            long_spread += revenue_retreat_amount
            short_spread += revenue_retreat_amount // 2
        elif base_asset_amount_with_amm < 0:
            long_spread += revenue_retreat_amount // 2
            short_spread += revenue_retreat_amount
        else:
            long_spread += revenue_retreat_amount // 2
            short_spread += revenue_retreat_amount // 2

    long_spread, short_spread = cap_to_max_spread(long_spread, short_spread, max_target_spread)

    return int(long_spread), int(short_spread)




from datetime import datetime as dt
import sys
from tokenize import tabsize
import driftpy
import pandas as pd
import numpy as np
from driftpy.accounts.oracle import *
from constants import ALL_MARKET_NAMES

import plotly.express as px

pd.options.plotting.backend = "plotly"

# from driftpy.constants.config import configs
from anchorpy import Provider, Wallet
from solders.keypair import Keypair
from solana.rpc.async_api import AsyncClient
from driftpy.drift_client import DriftClient
from driftpy.drift_user import DriftUser
from driftpy.accounts import (
    get_perp_market_account,
    get_spot_market_account,
    get_user_account,
    get_state_account,
)
from driftpy.constants.numeric_constants import *
from driftpy.drift_user import get_token_amount
import os
import json
import streamlit as st
from driftpy.types import MarginRequirementType
from driftpy.constants.spot_markets import devnet_spot_market_configs, SpotMarketConfig
from driftpy.constants.perp_markets import devnet_perp_market_configs, PerpMarketConfig
from driftpy.addresses import *
from dataclasses import dataclass
from solders.pubkey import Pubkey
from helpers import serialize_perp_market_2, serialize_spot_market
from anchorpy import EventParser
import asyncio
import time
from enum import Enum
from driftpy.math.margin import MarginCategory, calculate_asset_weight
import datetime


def calculate_market_open_bid_ask(base_asset_reserve, min_base_asset_reserve, max_base_asset_reserve, step_size=None):
    # open orders
    if min_base_asset_reserve < base_asset_reserve:
        open_asks = (base_asset_reserve - min_base_asset_reserve) * -1

        if step_size and abs(open_asks) // 2 < step_size:
            open_asks = 0
    else:
        open_asks = 0

    if max_base_asset_reserve > base_asset_reserve:
        open_bids = max_base_asset_reserve - base_asset_reserve

        if step_size and open_bids // 2 < step_size:
            open_bids = 0
    else:
        open_bids = 0

    return open_bids, open_asks

def calculate_inventory_liquidity_ratio(base_asset_amount_with_amm, base_asset_reserve, min_base_asset_reserve, max_base_asset_reserve):
    # inventory skew
    open_bids, open_asks = calculate_market_open_bid_ask(base_asset_reserve, min_base_asset_reserve, max_base_asset_reserve)

    min_side_liquidity = min(abs(open_bids), abs(open_asks))

    inventory_scale_bn = min(
        base_asset_amount_with_amm * PERCENTAGE_PRECISION // max(min_side_liquidity, 1),
        PERCENTAGE_PRECISION
    )
    return inventory_scale_bn


def clamp(value, min_value, max_value):
        return max(min(value, max_value), min_value)



def calculate_reservation_price_offset(
    reserve_price,
    last_24h_avg_funding_rate,
    liquidity_fraction,
    oracle_twap_fast,
    mark_twap_fast,
    oracle_twap_slow,
    mark_twap_slow,
    max_offset_pct
):
    offset = 0
    # max_offset_pct = (1e6 / 400) 
    # base_inventory_threshold = min_order_size * 5
    # calculate quote denominated market premium

    max_offset_in_price = int(max_offset_pct * reserve_price / PERCENTAGE_PRECISION)

    # Calculate quote denominated market premium
    mark_premium_minute = clamp(mark_twap_fast - oracle_twap_fast, -max_offset_in_price, max_offset_in_price)
    mark_premium_hour = clamp(mark_twap_slow - oracle_twap_slow, -max_offset_in_price, max_offset_in_price)

    # convert last_24h_avg_funding_rate to quote denominated premium
    mark_premium_day = clamp(last_24h_avg_funding_rate / FUNDING_RATE_BUFFER * 24, -max_offset_in_price, max_offset_in_price)

    mark_premium_avg = (mark_premium_day + mark_premium_hour + mark_premium_minute ) / 3
    

    mark_offset = mark_premium_avg * PRICE_PRECISION / reserve_price
    inv_offset = liquidity_fraction * max_offset_pct / PERCENTAGE_PRECISION
    offset = mark_offset + inv_offset

    if np.sign(inv_offset) != np.sign(mark_offset):
        offset = 0

    clamped_offset = clamp(offset, -max_offset_in_price, max_offset_in_price)

    return clamped_offset


async def dlp(ch: DriftClient):
    tabs = st.tabs(['overview', ])
    await ch.account_subscriber.update_cache()

    with tabs[0]:
        dlp_aum = st.number_input('DLP AUM', 0, value=10_000_000)

        num_m = ch.get_state_account().number_of_markets
        res = []

        # Convert to value terms assuming prices (fetch or hardcode for now)
        sol_price = 140  # You can update to real-time later
        btc_price = 85000
        eth_price = 2000
        for i in range(num_m):
            market: PerpMarketAccount = ch.get_perp_market_account(i)
            op = market.amm.historical_oracle_data.last_oracle_price/1e6

            vamm_base_inventory = -market.amm.base_asset_amount_with_amm / 1e9
            vamm_quote_inventory = vamm_base_inventory * op

            sol_wgt = 0
            btc_wgt = 0
            eth_wgt = 0
            if market.market_index == 0:
                sol_wgt = 1
                sol_price = op
            elif market.market_index == 1:
                btc_wgt = 1
                btc_price = op
            elif market.market_index == 2:
                eth_wgt = 1
                eth_price = op
            elif market.market_index in [6, 5, 15, 16, 17]:
                eth_wgt = .6
                sol_wgt = .2
                btc_wgt = .2
            elif 'BET' in bytes(market.name).decode('utf-8'):
                sol_wgt = 0
            else:
                sol_wgt = .4
                btc_wgt = .4
                eth_wgt = .2

            res.append((
                bytes(market.name).decode('utf-8'),
                vamm_base_inventory,
                vamm_quote_inventory,
                sol_wgt,
                btc_wgt,
                eth_wgt
            ))

        df = pd.DataFrame(res, columns=[
            'market',
            'vamm_base_inventory',
            'vamm_quote_inventory',
            'SOL',
            'BTC',
            'ETH',
        ])

        st.write(sol_price, btc_price, eth_price)

        # Calculate target hedge by flipping sign of inventory * weight
        df['SOL_target'] = -df['vamm_quote_inventory'] * df['SOL']/sol_price
        df['BTC_target'] = -df['vamm_quote_inventory'] * df['BTC']/btc_price
        df['ETH_target'] = -df['vamm_quote_inventory'] * df['ETH']/eth_price

        st.write("Market Data with Target Hedge:")
        st.write(df)

        # Aggregate target hedge amounts
        total_sol = df['SOL_target'].sum()
        total_btc = df['BTC_target'].sum()
        total_eth = df['ETH_target'].sum()


        sol_val = total_sol * sol_price
        btc_val = total_btc * btc_price
        eth_val = total_eth * eth_price

        total_val = sol_val + btc_val + eth_val
        usdc_val = dlp_aum - total_val

        pct_sol = sol_val / dlp_aum * 100
        pct_btc = btc_val / dlp_aum * 100
        pct_eth = eth_val / dlp_aum * 100
        pct_usdc = usdc_val / dlp_aum * 100

        final_df = pd.DataFrame([{
            'Asset': 'SOL', 'Target Amount': total_sol, 'Value': sol_val, 'Target %': pct_sol
        }, {
            'Asset': 'BTC', 'Target Amount': total_btc, 'Value': btc_val, 'Target %': pct_btc
        }, {
            'Asset': 'ETH', 'Target Amount': total_eth, 'Value': eth_val, 'Target %': pct_eth
        }, {
            'Asset': 'USDC', 'Target Amount': usdc_val / 1, 'Value': usdc_val, 'Target %': pct_usdc
        }])

        st.write("Target Hedge Summary:")
        st.write(final_df)


        # Clamp target hedge amounts at 0 (no negative positions)
        df['SOL_target_clamped'] = np.clip(df['SOL_target'], 0, None)
        df['BTC_target_clamped'] = np.clip(df['BTC_target'], 0, None)
        df['ETH_target_clamped'] = np.clip(df['ETH_target'], 0, None)

        st.write("Clamped Market Data with Target Hedge (no shorts):")
        st.write(df)

       # Step 1: Unclamped targets (already computed)
        sol_amt = df['SOL_target'].sum()
        btc_amt = df['BTC_target'].sum()
        eth_amt = df['ETH_target'].sum()

        # Step 2: Convert to value
        sol_val = sol_amt * sol_price
        btc_val = btc_amt * btc_price
        eth_val = eth_amt * eth_price

        # Step 3: Total hedge value
        total_target_val = max(sol_val,0) + max(btc_val,0) + max(eth_val,0)

        # Step 4: If over AUM, scale down
        scaling_factor = 1.0
        if total_target_val > dlp_aum:
            scaling_factor = dlp_aum / total_target_val

        # Step 5: Apply scaling and clamp to 0 (no shorts)
        sol_val_adj = max(sol_val * scaling_factor, 0)
        btc_val_adj = max(btc_val * scaling_factor, 0)
        eth_val_adj = max(eth_val * scaling_factor, 0)

        # Step 6: Remaining in USDC
        usdc_val_adj = np.clip(dlp_aum - (sol_val_adj + btc_val_adj + eth_val_adj), 0, 1e12)

        # Step 7: Convert back to base units
        sol_amt_adj = sol_val_adj / sol_price
        btc_amt_adj = btc_val_adj / btc_price
        eth_amt_adj = eth_val_adj / eth_price

        # Step 8: Percentages
        pct_sol = sol_val_adj / dlp_aum * 100
        pct_btc = btc_val_adj / dlp_aum * 100
        pct_eth = eth_val_adj / dlp_aum * 100
        pct_usdc = usdc_val_adj / dlp_aum * 100

        final_df_clamped = pd.DataFrame([{
            'Asset': 'SOL', 'Target Amount': sol_amt_adj, 'Value': sol_val_adj, 'Target %': pct_sol
        }, {
            'Asset': 'BTC', 'Target Amount': btc_amt_adj, 'Value': btc_val_adj, 'Target %': pct_btc
        }, {
            'Asset': 'ETH', 'Target Amount': eth_amt_adj, 'Value': eth_val_adj, 'Target %': pct_eth
        }, {
            'Asset': 'USDC', 'Target Amount': usdc_val_adj, 'Value': usdc_val_adj, 'Target %': pct_usdc
        }])

        st.write("Clamped Target Hedge Summary (No Shorts, Fits AUM):")
        st.write(final_df_clamped)
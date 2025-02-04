import sys
from tokenize import tabsize

import driftpy
import numpy as np
import pandas as pd

pd.options.plotting.backend = "plotly"
import asyncio
import json
import os
from dataclasses import dataclass
from glob import glob

import streamlit as st
from driftpy.decode.utils import decode_name
from anchorpy import EventParser, Provider, Wallet
from driftpy.accounts import (
    get_perp_market_account,
    get_spot_market_account,
    get_state_account,
    get_user_account,
)
from driftpy.constants.numeric_constants import *
from driftpy.constants.perp_markets import PerpMarketConfig, devnet_perp_market_configs
from driftpy.constants.spot_markets import SpotMarketConfig, devnet_spot_market_configs
from driftpy.drift_client import DriftClient
from driftpy.drift_user import DriftUser
from driftpy.math.margin import (
    MarginCategory,
    calculate_asset_weight,
    calculate_liability_weight,
    calculate_market_margin_ratio,
)
from solana.rpc.async_api import AsyncClient
from solders.keypair import Keypair
from solders.pubkey import Pubkey

from helpers import serialize_perp_market_2, serialize_spot_market

async def imf_page(clearing_house: DriftClient):
    tabs = st.tabs(["Overview", "Calculator"])
    ch = clearing_house
    if (
        ch.account_subscriber.cache is None
        or ch.account_subscriber.cache.get("state", None) is None
    ):
        await ch.account_subscriber.update_cache()
    state = ch.get_state_account()

    with tabs[0]:
        st.header("IMF Market Overview")
        
        col1, col2 = st.columns([1, 2])
        with col1:
            not_size = st.number_input("Notional Size ($):", 
                                     value=100_000, 
                                     min_value=1000,
                                     help="Reference position size for IMF calculation")
            weight_type = st.radio("Weight Type:", 
                                 ["Liability (Borrow)", "Asset (Collateral)"],
                                 help="Calculate IMF effect on borrowing vs collateral capacity")

        # Prepare data for both perp and spot markets
        overview = {}
        for i in range(state.number_of_markets):
            market = ch.get_perp_market_account(i)
            nom = bytes(market.name).decode("utf-8").strip('\x00')
            size = (
                not_size
                * BASE_PRECISION
                / (
                    market.amm.historical_oracle_data.last_oracle_price
                    / PRICE_PRECISION
                )
            )
            res = [
                calculate_market_margin_ratio(market, 0, MarginCategory.INITIAL),
                calculate_market_margin_ratio(market, size, MarginCategory.INITIAL),
                calculate_market_margin_ratio(market, 0, MarginCategory.MAINTENANCE),
                calculate_market_margin_ratio(market, size, MarginCategory.MAINTENANCE),
            ]
            res = [1 / (x / MARGIN_PRECISION) for x in res]
            res = [market.imf_factor / 1e6] + res
            overview[nom] = res

        overview2 = {}
        for i in range(state.number_of_spot_markets):
            market = ch.get_spot_market_account(i)
            nom = decode_name(market.name)
            oracle_price = market.historical_oracle_data.last_oracle_price
            size = not_size * (10**market.decimals) / (oracle_price / PRICE_PRECISION)
            
            if weight_type == "Asset (Collateral)":
                res = [
                    calculate_asset_weight(0, oracle_price, market, MarginCategory.INITIAL),
                    calculate_asset_weight(size, oracle_price, market, MarginCategory.INITIAL),
                    calculate_asset_weight(size, oracle_price, market, MarginCategory.MAINTENANCE),
                ]
                res = [x / MARGIN_PRECISION for x in res]
            else:
                res = [
                    calculate_liability_weight(0, market, MarginCategory.INITIAL),
                    calculate_liability_weight(size,  market, MarginCategory.INITIAL),
                    calculate_liability_weight(size,  market, MarginCategory.MAINTENANCE),
                ]
                res = [1 / (x / MARGIN_PRECISION) for x in res]
            
            res = [market.imf_factor / 1e6] + res
            overview2[nom] = res

        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Perpetual Markets")
            df_perp = pd.DataFrame(
                overview,
                index=[
                    "IMF Factor",
                    "Initial Leverage",
                    "Initial Leverage (w/Size)",
                    "Maint. Leverage",
                    "Maint. Leverage (w/Size)",
                ],
            ).T.reset_index()
            df_perp.columns = ['Market'] + list(df_perp.columns[1:])
            st.dataframe(df_perp, use_container_width=True)

        with col2:
            st.subheader("Spot Markets")
            weight_label = "Weight" if weight_type == "Asset (Collateral)" else "Leverage"
            df_spot = pd.DataFrame(
                overview2,
                index=[
                    "IMF Factor",
                    f"Initial {weight_label}",
                    f"Initial {weight_label} (w/Size)",
                    f"Maint. {weight_label}",
                ],
            ).T.reset_index()
            df_spot.columns = ['Market'] + list(df_spot.columns[1:])
            st.dataframe(df_spot, use_container_width=True)

    with tabs[1]:
        st.header("IMF Impact Calculator")
        
        col1, col2, col3 = st.columns([1, 1, 1])
        
        with col1:
            mode = st.selectbox("Market Type:", ["Custom", "Perpetual", "Spot"])
            weight_type = st.radio("Weight Type:", 
                                 ["Liability (Borrow)", "Asset (Collateral)"],
                                 help="Calculate IMF effect on borrowing vs collateral")
        
        market_idx = 0
        oracle_px = 1.0
        
        with col2:
            if mode != "Custom":
                n = state.number_of_markets
                start_idx = 0
                if mode == "Spot":
                    start_idx = 1
                    n = state.number_of_spot_markets
                market_idx = st.selectbox("Market:", range(start_idx, n), 0)
                
                if mode == "Perpetual":
                    market = ch.get_perp_market_account(market_idx)
                    if weight_type == "Liability (Borrow)":
                        init_weight = market.margin_ratio_initial / 1e4
                        maint_weight = market.margin_ratio_maintenance / 1e4
                    else:
                        init_weight = 1 - (market.margin_ratio_initial / 1e4)
                        maint_weight = 1 - (market.margin_ratio_maintenance / 1e4)
                    imf = market.imf_factor / 1e6
                    oracle_px = market.amm.historical_oracle_data.last_oracle_price / 1e6
                else:
                    market = ch.get_spot_market_account(market_idx)
                    if weight_type == "Liability (Borrow)":
                        init_weight = market.initial_liability_weight / 1e4 - 1
                        maint_weight = market.maintenance_liability_weight / 1e4 - 1
                    else:
                        init_weight = market.initial_asset_weight / 1e4
                        maint_weight = market.maintenance_asset_weight / 1e4
                    imf = market.imf_factor / 1e6
                    oracle_px = market.historical_oracle_data.last_oracle_price / 1e6
                
                st.write(f"Selected Market: {decode_name(market.name)}")
                
                max_imf = 0.01 if mode == "Perpetual" else 0.5
                imf = st.slider("IMF Factor", 0.0, max_imf, imf, step=0.00005)
                st.write(f"Current IMF = {imf:.5f}")
            
            else:
                init_weight = st.slider("Initial Weight", 0.0, 1.0, 0.2, step=0.01)
                maint_weight = st.slider("Maintenance Weight", 0.0, 1.0, 0.1, step=0.01)
                imf = st.slider("IMF Factor", 0.0, 0.5, 0.001, step=0.0001)
                oracle_px = st.number_input("Oracle Price", value=1.0, min_value=0.0)
        
        with col3:
            base = st.number_input("Position Size (Base Units)", value=1.0)
            notional = oracle_px * base
            st.write(f"Notional Value: ${notional:,.2f}")
            
            # Calculate effective weight
            base_weight = init_weight
            if weight_type == "Liability (Borrow)":
                if imf != 0:
                    base_weight = base_weight * 0.8  # Discount when IMF active
                
                ddsize = np.sqrt(np.abs(base))
                effective_weight = max(init_weight, base_weight + imf * ddsize)
            else:
                ddsize = np.sqrt(np.abs(base))
                effective_weight = min(init_weight, (1.1) - imf * ddsize)
            
            if weight_type == "Asset (Collateral)":
                st.write(f"Base Asset Weight: {init_weight:.3f}")
                st.write(f"Effective Asset Weight: {effective_weight:.3f}")
                leverage_label = "Collateral Value Multiplier"
            else:
                st.write(f"Base Liability Weight: {init_weight:.3f}")
                st.write(f"Effective Liability Weight: {effective_weight:.3f}")
                leverage_label = "Max Leverage"
            
            st.write(f"{leverage_label}: {(1/effective_weight if weight_type == 'Liability (Borrow)' else effective_weight):.2f}x")
        
        st.subheader("IMF Impact Curve")
        
        if oracle_px != 0:
            max_size = max(10000 / oracle_px, base * 2)
        else:
            max_size = base * 2
            
        step = int(max(1000, max_size / 1000))
        index = np.linspace(0, max_size, step)
        
        def calc_effective_weight(weight, imf, size, is_liability=True):
            base_weight = weight

            if is_liability:
                if imf != 0:
                    base_weight = weight * 0.8
                dd = np.sqrt(np.abs(size))
                effective = max(weight, base_weight + imf * dd)
                return 1/effective if is_liability else effective
            else:
                dd = np.sqrt(np.abs(size))
                effective = min(weight, (1.1) - imf * dd)
                return effective
        
        is_liability = weight_type == "Liability (Borrow)"
        df = pd.Series([calc_effective_weight(init_weight, imf, x, is_liability) for x in index])
        df.index = index
        
        df2 = pd.Series([calc_effective_weight(maint_weight, imf, x, is_liability) for x in index])
        df2.index = index
        
        imf_df = pd.concat({"Initial": df, "Maintenance": df2}, axis=1)
        imf_df.index *= oracle_px
        
        fig = imf_df.plot(kind="line")
        fig.update_layout(
            title=f"IMF Impact on {'Leverage' if is_liability else 'Asset Weight'}",
            xaxis_title="Position Notional ($)",
            yaxis_title=leverage_label,
        )
        
        st.plotly_chart(fig, use_container_width=True)
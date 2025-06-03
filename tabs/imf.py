import asyncio
import json
import os
import sys
from dataclasses import dataclass
from glob import glob
from tokenize import tabsize

import driftpy
import numpy as np
import pandas as pd
import streamlit as st
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
from driftpy.decode.utils import decode_name
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

pd.options.plotting.backend = "plotly"


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
            not_size = st.number_input(
                "Notional Size ($):",
                value=100_000,
                min_value=1000,
                help="Reference position size for IMF calculation",
            )
            weight_type = st.radio(
                "Weight Type:",
                ["Liability (Borrow)", "Asset (Collateral)"],
                help="Calculate IMF effect on borrowing vs collateral capacity",
            )
            hlm_toggle_overview = st.toggle(
                "High Leverage Mode",
                key="overview_hlm_toggle",
                value=False,
                help="Enable High Leverage Mode (HLM) for Perpetual markets. Uses the market's specific High Leverage Initial/Maintenance margin ratios if available. Does not affect Spot markets.",
            )

        overview = {}

        for i in range(state.number_of_markets):
            market = ch.get_perp_market_account(i)
            nom = bytes(market.name).decode("utf-8").strip("\x00")
            size = (
                not_size
                * BASE_PRECISION
                / (
                    market.amm.historical_oracle_data.last_oracle_price
                    / PRICE_PRECISION
                )
            )
            res = [
                calculate_market_margin_ratio(
                    market,
                    0,
                    MarginCategory.INITIAL,
                    user_high_leverage_mode=hlm_toggle_overview,
                ),
                calculate_market_margin_ratio(
                    market,
                    size,
                    MarginCategory.INITIAL,
                    user_high_leverage_mode=hlm_toggle_overview,
                ),
                calculate_market_margin_ratio(
                    market,
                    0,
                    MarginCategory.MAINTENANCE,
                    user_high_leverage_mode=hlm_toggle_overview,
                ),
                calculate_market_margin_ratio(
                    market,
                    size,
                    MarginCategory.MAINTENANCE,
                    user_high_leverage_mode=hlm_toggle_overview,
                ),
            ]
            res = [1 / (x / MARGIN_PRECISION) if x > 0 else float("inf") for x in res]
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
                    calculate_asset_weight(
                        0, oracle_price, market, MarginCategory.INITIAL
                    ),
                    calculate_asset_weight(
                        size, oracle_price, market, MarginCategory.INITIAL
                    ),
                    calculate_asset_weight(
                        size, oracle_price, market, MarginCategory.MAINTENANCE
                    ),
                ]
                res = [x / MARGIN_PRECISION for x in res]
            else:
                res = [
                    calculate_liability_weight(0, market, MarginCategory.INITIAL),
                    calculate_liability_weight(size, market, MarginCategory.INITIAL),
                    calculate_liability_weight(
                        size, market, MarginCategory.MAINTENANCE
                    ),
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
            df_perp.columns = ["Market"] + list(df_perp.columns[1:])
            st.dataframe(df_perp, use_container_width=True)

        with col2:
            st.subheader("Spot Markets")
            weight_label = (
                "Weight" if weight_type == "Asset (Collateral)" else "Leverage"
            )
            df_spot = pd.DataFrame(
                overview2,
                index=[
                    "IMF Factor",
                    f"Initial {weight_label}",
                    f"Initial {weight_label} (w/Size)",
                    f"Maint. {weight_label}",
                ],
            ).T.reset_index()
            df_spot.columns = ["Market"] + list(df_spot.columns[1:])
            st.dataframe(df_spot, use_container_width=True)

    with tabs[1]:
        st.header("IMF Impact Calculator")

        col1, col2, col3 = st.columns([1, 1, 1])

        with col1:
            mode = st.selectbox("Market Type:", ["Custom", "Perpetual", "Spot"])
            weight_type = st.radio(
                "Weight Type:",
                ["Liability (Borrow)", "Asset (Collateral)"],
                help="Calculate IMF effect on borrowing vs collateral",
            )
            hlm_toggle_calculator = st.toggle(
                "High Leverage Mode",
                key="calculator_hlm_toggle",
                value=False,
                help="Enable High Leverage Mode (HLM) for Perpetual markets. Uses the market's specific High Leverage Initial/Maintenance margin ratios if available. Does not affect Spot markets.",
            )

        market_idx = 0
        oracle_px = 1.0

        with col2:
            if mode != "Custom":
                n = state.number_of_markets
                start_idx = 0
                if mode == "Spot":
                    start_idx = 1
                    n = state.number_of_spot_markets

                market_options = range(start_idx, n)
                if not market_options and n > 0:
                    st.warning(
                        f"No {mode} markets available with current indexing. Adjust start_idx or check market availability."
                    )
                elif n == 0:
                    st.warning(f"No {mode} markets available at all.")
                    st.stop()

                market_idx_display_name_map = {}
                actual_market_indices = []

                if mode == "Perpetual":
                    for i in range(state.number_of_markets):
                        m = ch.get_perp_market_account(i)
                        market_idx_display_name_map[i] = f"{i}: {decode_name(m.name)}"
                        actual_market_indices.append(i)
                    market_options = actual_market_indices

                elif mode == "Spot":
                    for i in range(state.number_of_spot_markets):
                        m = ch.get_spot_market_account(i)
                        market_idx_display_name_map[i] = f"{i}: {decode_name(m.name)}"
                        actual_market_indices.append(i)
                    market_options = actual_market_indices

                if not market_options:
                    st.warning(f"No {mode} markets could be listed.")
                    st.stop()

                selected_market_idx = st.selectbox(
                    "Market:",
                    options=list(market_options),
                    format_func=lambda x: market_idx_display_name_map.get(x, str(x)),
                    index=0,
                )

                standard_init_weight = 0.0
                standard_maint_weight = 0.0
                market_name_decoded = "N/A"

                if mode == "Perpetual":
                    market = ch.get_perp_market_account(selected_market_idx)
                    market_name_decoded = decode_name(market.name)

                    base_initial_mr = (
                        calculate_market_margin_ratio(
                            market,
                            0,
                            MarginCategory.INITIAL,
                            user_high_leverage_mode=hlm_toggle_calculator,
                        )
                        / MARGIN_PRECISION
                    )
                    base_maint_mr = (
                        calculate_market_margin_ratio(
                            market,
                            0,
                            MarginCategory.MAINTENANCE,
                            user_high_leverage_mode=hlm_toggle_calculator,
                        )
                        / MARGIN_PRECISION
                    )

                    if weight_type == "Liability (Borrow)":
                        standard_init_weight = base_initial_mr
                        standard_maint_weight = base_maint_mr
                    else:  # Asset (representing (1 - margin ratio))
                        standard_init_weight = 1 - base_initial_mr
                        standard_maint_weight = 1 - base_maint_mr

                    imf = market.imf_factor / 1e6
                    oracle_px = (
                        market.amm.historical_oracle_data.last_oracle_price
                        / PRICE_PRECISION
                    )
                else:  # Spot
                    market = ch.get_spot_market_account(selected_market_idx)
                    market_name_decoded = decode_name(market.name)

                    actual_initial_asset_weight = (
                        calculate_asset_weight(
                            0,
                            market.historical_oracle_data.last_oracle_price,
                            market,
                            MarginCategory.INITIAL,
                        )
                        / MARGIN_PRECISION
                    )
                    actual_maint_asset_weight = (
                        calculate_asset_weight(
                            0,
                            market.historical_oracle_data.last_oracle_price,
                            market,
                            MarginCategory.MAINTENANCE,
                        )
                        / MARGIN_PRECISION
                    )

                    actual_initial_liability_weight = (
                        calculate_liability_weight(0, market, MarginCategory.INITIAL)
                        / MARGIN_PRECISION
                    )
                    actual_maint_liability_weight = (
                        calculate_liability_weight(
                            0, market, MarginCategory.MAINTENANCE
                        )
                        / MARGIN_PRECISION
                    )

                    if weight_type == "Liability (Borrow)":
                        standard_init_weight = actual_initial_liability_weight
                        standard_maint_weight = actual_maint_liability_weight
                    else:  # Asset
                        standard_init_weight = actual_initial_asset_weight
                        standard_maint_weight = actual_maint_asset_weight

                    imf = market.imf_factor / 1e6
                    oracle_px = (
                        market.historical_oracle_data.last_oracle_price
                        / PRICE_PRECISION
                    )

                init_weight = standard_init_weight
                maint_weight = standard_maint_weight

                st.write(f"Selected Market: {market_name_decoded}")
                disable_imf_slider = mode == "Spot"
                imf_display_value = imf

                if disable_imf_slider:
                    st.write(
                        f"Market IMF Factor: {imf_display_value:.5f} (intrinsic to spot market weights)"
                    )
                else:
                    max_imf_slider_val = 0.01
                    imf = st.slider(
                        "IMF Factor (Perp)",
                        0.0,
                        max_imf_slider_val,
                        imf,
                        step=0.00005,
                        format="%.5f",
                        help="Adjustable for perpetual markets. For spot, IMF is intrinsic.",
                    )

                if mode == "Perpetual" and hlm_toggle_calculator:
                    st.info(
                        "High Leverage Mode (HLM) is ON for Perpetual market calculations."
                    )
                elif mode == "Spot" and hlm_toggle_calculator:
                    st.warning(
                        "High Leverage Mode (HLM) toggle is ON but does NOT apply to Spot markets."
                    )

            else:  # Custom mode
                init_weight = st.slider(
                    "Initial Weight/Ratio", 0.0, 2.0, 0.2, step=0.01
                )
                maint_weight = st.slider(
                    "Maintenance Weight/Ratio", 0.0, 2.0, 0.1, step=0.01
                )
                imf = st.slider(
                    "IMF Factor", 0.0, 0.5, 0.001, step=0.0001, format="%.4f"
                )
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
                effective_weight = min(init_weight, (init_weight * 1.05) - imf * ddsize)

            if weight_type == "Asset (Collateral)":
                st.write(f"Base Asset Weight: {init_weight:.3f}")
                st.write(f"Effective Asset Weight: {effective_weight:.3f}")
                leverage_label = "Collateral Value Multiplier"
            else:
                st.write(f"Base Liability Weight: {init_weight:.3f}")
                st.write(f"Effective Liability Weight: {effective_weight:.3f}")
                leverage_label = "Max Leverage"

            st.write(
                f"{leverage_label}: {(1 / effective_weight if weight_type == 'Liability (Borrow)' else effective_weight):.2f}x"
            )

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
                return 1 / effective if is_liability else effective
            else:
                dd = np.sqrt(np.abs(size))
                effective = min(weight, (1.1) - imf * dd)
                return effective

        is_liability = weight_type == "Liability (Borrow)"
        df = pd.Series(
            [calc_effective_weight(init_weight, imf, x, is_liability) for x in index]
        )
        df.index = index

        df2_series_data = []
        for x_val in index:
            val = calc_effective_weight(maint_weight, imf, x_val, is_liability)
            df2_series_data.append(val)

        df2 = pd.Series(df2_series_data)
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

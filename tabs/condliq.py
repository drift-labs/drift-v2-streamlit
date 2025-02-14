import datetime
import sys
from tokenize import tabsize

import driftpy
import numpy as np
import pandas as pd
import requests
from driftpy.accounts.oracle import *

pd.options.plotting.backend = "plotly"

# from driftpy.constants.config import configs
import asyncio
import json
import os
import time
from dataclasses import dataclass
from enum import Enum

import plotly.graph_objs as go
import streamlit as st
from anchorpy import EventParser, Provider, Wallet
from driftpy.accounts import (
    get_perp_market_account,
    get_spot_market_account,
    get_state_account,
    get_user_account,
)
from driftpy.addresses import *
from driftpy.constants.numeric_constants import *
from driftpy.constants.perp_markets import PerpMarketConfig, devnet_perp_market_configs
from driftpy.constants.spot_markets import SpotMarketConfig, devnet_spot_market_configs
from driftpy.drift_client import DriftClient
from driftpy.drift_user import DriftUser, get_token_amount
from driftpy.math.margin import MarginCategory, calculate_asset_weight
from driftpy.types import (
    MarginRequirementType,
    PerpPosition,
    SpotPosition,
    UserAccount,
    UserStatsAccount,
)
from solana.rpc.async_api import AsyncClient
from solders.keypair import Keypair
from solders.pubkey import Pubkey

from datafetch.snapshot_fetch import load_user_snapshot
from helpers import serialize_perp_market_2, serialize_spot_market


async def condliqcheck(clearing_house: DriftClient):
    # connection = clearing_house.program.provider.connection
    class UserAccountEncoder(json.JSONEncoder):
        def default(self, obj):
            # st.write(type(obj))
            # st.write(str(type(obj)))
            if "Position" in str(type(obj)) or "Order" in str(type(obj)):
                return obj.__dict__
            elif isinstance(obj, Pubkey):
                return str(obj)
            else:
                return str(obj)
            return super().default(obj)

    s1, s2, s3 = st.columns([2, 1, 2])
    inp = s1.text_input(
        "user account:",
    )
    mode = s2.radio("mode:", ["live"])
    commit_hash = "main"
    if len(inp) > 5:
        # st.write(inp)
        # st.write(Pubkey.from_string(str(inp)))

        if mode == "live":
            user_pk = Pubkey.from_string(str(inp))
            # st.write(user_pk)
            user: UserAccount = await clearing_house.program.account["User"].fetch(
                user_pk
            )
            # st.json(json.dumps(user.__dict__, cls=UserAccountEncoder))

            user_authority = user.authority
            st.write("authority:", user_authority)

            user_stats_pk = get_user_stats_account_public_key(
                clearing_house.program_id, user_authority
            )

            userstats: UserStatsAccount = await clearing_house.program.account[
                "UserStats"
            ].fetch(user_stats_pk)

            def is_labeled_for_cond(user, user_stats):
                is_t = False
                if user.next_order_id > 3000:
                    st.write("high next_order_id:", user.next_order_id)
                    is_t = True

                if user_stats.number_of_sub_accounts_created > 10:
                    st.write(
                        "high number_of_sub_accounts_created:",
                        user_stats.number_of_sub_accounts_created,
                    )
                    is_t = True

                if user_stats.disable_update_perp_bid_ask_twap:
                    st.write(
                        "user_incorrectly_update_twap:",
                        user_stats.disable_update_perp_bid_ask_twap,
                    )
                    is_t = True
                return is_t

            is_t1 = is_labeled_for_cond(user, userstats)
            if not is_t1:
                st.write("User is not labeled informed flow")
            else:
                st.write(
                    "User is labeled informed flow, can only atomically take at conditional prices posted by protected makers"
                )
            return 0
            # st.json(json.dumps(userstats.__dict__, cls=UserAccountEncoder))

            st.header("perp positions")
            dff = pd.concat(
                [
                    pd.DataFrame(pos.__dict__, index=[0])
                    for i, pos in enumerate(user.perp_positions)
                ],
                axis=0,
            )
            # print(dff.columns)
            dff = dff[
                [
                    # 'authority', 'name',
                    "lp_shares",
                    #     'last_active_slot', 'public_key',
                    # 'last_add_perp_lp_shares_ts',
                    "market_index",
                    #    'position_index',
                    "last_cumulative_funding_rate",
                    "base_asset_amount",
                    "quote_asset_amount",
                    "quote_break_even_amount",
                    "quote_entry_amount",
                    "last_base_asset_amount_per_lp",
                    "last_quote_asset_amount_per_lp",
                    "remainder_base_asset_amount",
                    "open_orders",
                ]
            ]
            for col in [
                "lp_shares",
                "last_base_asset_amount_per_lp",
                "base_asset_amount",
                "remainder_base_asset_amount",
            ]:
                dff[col] /= 1e9
            for col in [
                "quote_asset_amount",
                "quote_break_even_amount",
                "quote_entry_amount",
                "last_quote_asset_amount_per_lp",
            ]:
                dff[col] /= 1e6

            # st.write('perp market lp info:')
            # a0, a1, a2, a3 = st.columns(4)
            # mi = a0.selectbox('market index:', range(0, state.number_of_markets), 0)

            # st.write(perp_market.amm)
            # bapl = perp_market.amm.base_asset_amount_per_lp/1e9
            # qapl = perp_market.amm.quote_asset_amount_per_lp/1e6
            # baawul = perp_market.amm.base_asset_amount_with_unsettled_lp/1e9

            # a1.metric('base asset amount per lp:', bapl)
            # a2.metric('quote asset amount per lp:', qapl)
            # a3.metric('unsettled base asset amount with lp:', baawul)

            # cols = (st.multiselect('columns:', ))
            # dff = dff[cols]
            st.write("raw lp positions")
            st.dataframe(dff)

            await clearing_house.account_subscriber.update_cache()

            def get_wouldbe_lp_settle(row):
                def standardize_base_amount(amount, step):
                    remainder = amount % step
                    standard = amount - remainder
                    return standard, remainder

                mi = row["market_index"]
                pm = clearing_house.account_subscriber.cache["perp_markets"][
                    int(mi)
                ].data
                delta_baapl = (
                    pm.amm.base_asset_amount_per_lp / 1e9
                    - row["last_base_asset_amount_per_lp"]
                )
                delta_qaapl = (
                    pm.amm.quote_asset_amount_per_lp / 1e6
                    - row["last_quote_asset_amount_per_lp"]
                )

                delta_baa = delta_baapl * row["lp_shares"]
                delta_qaa = delta_qaapl * row["lp_shares"]

                standard_baa, remainder_baa = standardize_base_amount(
                    delta_baa, pm.amm.order_step_size / 1e9
                )

                row["remainder_base_asset_amount"] += remainder_baa
                row["base_asset_amount"] += (
                    standard_baa + row["remainder_base_asset_amount"]
                )
                row["quote_asset_amount"] += delta_qaa
                row["quote_entry_amount"] += delta_qaa
                row["quote_break_even_amount"] += delta_qaa
                row["entry_price"] = (
                    -row["quote_entry_amount"] / row["base_asset_amount"]
                )

                return row

            newd = dff.apply(get_wouldbe_lp_settle, axis=1)
            st.write(newd)

        else:
            user, ff = load_user_snapshot(str(inp), commit_hash)
            st.write(ff)
            dd = user.set_index(user.columns[0]).to_json()
            st.json(dd)
        # st.write(user.__dict__['spot_positions'])
        # st.json(user)

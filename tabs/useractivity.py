import datetime
import sys
from tokenize import tabsize

import driftpy
import numpy as np
import pandas as pd
from anchorpy.provider import Signature
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
from driftpy.types import MarginRequirementType
from solana.rpc.async_api import AsyncClient
from solders.keypair import Keypair
from solders.pubkey import Pubkey

from datafetch.transaction_fetch import transaction_history_for_account


async def show_user_activity(clearing_house: DriftClient):
    # connection = clearing_house.program.provider.connection
    addycol, rpc_overcol, limcol, mlimcol, pagecol = st.columns([4, 2, 1, 1, 1])
    rpc_override = rpc_overcol.text_input(
        "rpc override:", "https://api.mainnet-beta.solana.com"
    )
    connection = AsyncClient(rpc_override)

    addy = addycol.text_input(
        "userAccount:", value="CJfc9nPHgrZohPWEsBnvYJeTs3LxBA5j8ZoZuZPugSQb"
    )
    limit = limcol.number_input("limit:", 1, 1000, 1000)
    MAX_LIMIT = mlimcol.number_input("max limit:", 1, None, value=5000)
    before_sig = pagecol.text_input("before sig:", value="")
    before_sig1 = None
    if before_sig != "":
        before_sig1 = str(before_sig)

    tabs = st.tabs(["heatmap", "dataframe", "transaction details"])

    res2 = await transaction_history_for_account(
        connection, Pubkey.from_string(addy), before_sig1, limit, MAX_LIMIT
    )
    t = pd.DataFrame(res2)
    if "blockTime" not in t.columns:
        st.write(t)
        return 0

    t["date"] = pd.to_datetime(t["blockTime"] * 1e9)
    t["day"] = t["date"].apply(lambda x: x.date())

    with tabs[1]:
        st.dataframe(t)

    with tabs[2]:
        parser = EventParser(
            clearing_house.program.program_id, clearing_house.program.coder
        )
        all_sigs = t["signature"].values.tolist()
        c1, c2 = st.columns([1, 8])
        ra = c1.radio("run all:", [True, False], index=1)
        signature = c2.selectbox("tx signatures:", all_sigs)
        st.write(t[t.signature == signature])

        theset = all_sigs if ra else [signature]
        idx = 0
        txs = []
        sigs = []
        # try:
        while idx < len(theset):
            sig = t["signature"].values[idx]
            ff = "transactions/" + sig + ".json"

            if not os.path.exists(ff) or not ra:
                transaction_got = await connection.get_transaction(
                    Signature.from_string(sig)
                )

                if ra:
                    os.makedirs("transactions", exist_ok=True)
                    with open(ff, "w") as f:
                        json.dump(transaction_got, f)
            else:
                with open(ff, "r") as f:
                    transaction_got = json.load(f)
            txs.append(transaction_got)
            sigs.append(sig)

            st.json(transaction_got, expanded=False)
            idx += 1
        # except Exception as e:
        #     st.warning('rpc failed: '+str(e))

        # txs = [transaction_got]
        # sigs = all_sigs[:idx]
        logs = {}
        for tx, sig in zip(txs, sigs):

            def call_b(evt):
                logs[sig] = logs.get(sig, []) + [evt]

            # likely rate limited
            tx1 = json.loads(tx.to_json())
            if "result" not in tx1:
                st.write(tx1["error"])
                break
            parser.parse_logs(tx1["result"]["meta"]["logMessages"], call_b)
        st.write(logs)
        # st.write(transaction_got['result']['meta']['logMessages'])

    tgrp = t.groupby("day").count()["blockTime"]
    weeks = (
        pd.to_datetime(tgrp.reset_index().iloc[:, 0])
        .dt.to_period("W-SUN")
        .dt.start_time
    )  # tgrp.reset_index().iloc[:,0].apply(lambda x: (datetime.datetime(x.isocalendar()[1]))
    dow = pd.to_datetime(tgrp.reset_index().iloc[:, 0]).apply(lambda x: (x.dayofweek))

    with tabs[0]:
        dates = tgrp.reset_index().iloc[:, 0].values.tolist()
        fig = go.Figure(
            data=go.Heatmap(
                z=tgrp,
                x=weeks,
                y=dow,
                text=["transactions on " + str(x) for x in dates],
                hoverinfo="z+text",
                colorscale="greens",
            )
        )
        layout = go.Layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            grid=None,
            xaxis_showgrid=False,
            yaxis_showgrid=False,
        )
        fig.update_layout(layout)
        fig["layout"]["yaxis"]["autorange"] = "reversed"
        num_tx = int(sum(tgrp.tolist()))
        first_day = dates[0]
        last_day = dates[-1]
        st.write(
            str(num_tx)
            + " transactions from "
            + str(first_day)
            + " to "
            + str(last_day)
        )
        st.plotly_chart(fig, use_container_width=True)

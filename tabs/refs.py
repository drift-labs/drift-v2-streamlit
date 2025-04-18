import sys
from tokenize import tabsize

import driftpy
import numpy as np
import pandas as pd

pd.options.plotting.backend = "plotly"

# from driftpy.constants.config import configs
import asyncio
import json
import os
from dataclasses import dataclass

import requests
import streamlit as st
from aiocache import Cache, cached
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
from driftpy.drift_user import get_token_amount
from driftpy.types import InsuranceFundStakeAccount, SpotMarketAccount
from solana.rpc.async_api import AsyncClient
from solders.keypair import Keypair
from solders.pubkey import Pubkey


async def ref_page(ch: DriftClient):
    # state = await get_state_account(ch.program)
    all_refs_stats = await ch.program.account["ReferrerName"].all()
    with st.expander("ref accounts"):
        st.write(all_refs_stats)
    df = pd.DataFrame([x.account.__dict__ for x in all_refs_stats])
    df.name = df.name.apply(lambda x: bytes(x).decode("utf-8", errors="ignore"))
    st.dataframe(df)

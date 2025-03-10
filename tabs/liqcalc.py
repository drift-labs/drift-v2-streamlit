import asyncio
import json
import streamlit as st
import pandas as pd
from driftpy.drift_client import DriftClient
from driftpy.drift_user import DriftUser
from driftpy.constants.perp_markets import mainnet_perp_market_configs
from driftpy.constants.spot_markets import SpotMarketConfig
from driftpy.drift_client import AccountSubscriptionConfig
from driftpy.accounts import get_state_account
from driftpy.accounts.oracle import *
from driftpy.types import MarginRequirementType, SpotPosition, PerpPosition, UserAccount, UserStatsAccount
from solders.pubkey import Pubkey
import copy

async def liqcalc(clearing_house: DriftClient):
    pd.options.plotting.backend = "plotly"

    class UserAccountEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, (SpotPosition, PerpPosition)):
                return obj.__dict__
            elif isinstance(obj, Pubkey):
                return str(obj)
            else:
                return super().default(obj)

    user_pk = st.text_input('user account:')
    mode = st.radio('mode:', ['raw', 'sdk'], index=1)

    if len(user_pk) > 5:
        user_pk = Pubkey.from_string(str(user_pk))
        SUB_ID = 0  # TODO: Set appropriate subscription ID

        if mode == 'sdk':
            balances = []
            positions = []

            chu_sub = DriftUser(
                clearing_house,
                user_public_key=user_pk,
                account_subscription=AccountSubscriptionConfig("cached"),
            )

            if clearing_house.account_subscriber.cache is None or \
            clearing_house.account_subscriber.cache.get('state') is None:
                st.write('updating caches')
                await clearing_house.account_subscriber.update_cache()
                await chu_sub.account_subscriber.update_cache()
                st.write('updated caches')

            user_authority = str(chu_sub.get_user_account().authority)

            pc1 = st.number_input('sol % change', -100, 100, value=0, step=None)
            if pc1 != 0:
                cache1 = copy.deepcopy(clearing_house.account_subscriber.cache)

                new_oracles_dat = {}

                for i,(key, val) in enumerate(cache1['oracle_price_data'].items()):
                    new_oracles_dat[key] = copy.deepcopy(val)
                    config_acct = mainnet_perp_market_configs[0] #todo
                    if key == str(config_acct.oracle):
                        new_oracles_dat[key].data.price *= (1 + pc1/100)
                cache1['oracle_price_data'] = new_oracles_dat
                chu_sub.account_subscriber.cache = cache1

            if st.button('Submit'):
                for spot_pos in chu_sub.get_active_spot_positions():
                    if spot_pos is not None:
                        spot_market_index = spot_pos.market_index
                        spot = chu_sub.get_spot_market_account(spot_market_index)
                        tokens = chu_sub.get_token_amount(spot_market_index)
                        market_name = "".join(map(chr, spot.name)).strip(" ")

                        if tokens >= 0:
                            dd = {
                                "name": market_name,
                                "tokens": tokens / (10 ** spot.decimals),
                                "net usd value": chu_sub.get_spot_asset_value(
                                    tokens, False, spot, None
                                ) / 1e6,
                                "cur price": chu_sub.get_oracle_data_for_spot_market(spot_market_index) / 1e6,
                                "liq price": chu_sub.get_spot_liq_price(spot_market_index),
                            }
                        else:
                            dd = {
                                "name": market_name,
                                "tokens": tokens / (10 ** spot.decimals),
                                "net usd value": chu_sub.get_spot_liability_value(
                                    tokens, False, spot, None, None
                                ) / 1e6,
                                "cur price": chu_sub.get_oracle_data_for_spot_market(spot_market_index) / 1e6,
                                "liq price": chu_sub.get_spot_liq_price(spot_market_index) / 1e6,
                            }

                        balances.append(dd)

                for pos in chu_sub.get_active_perp_positions():
                    perp_market_index = pos.market_index
                    if pos is not None:
                        perp = chu_sub.get_perp_market_account(perp_market_index)
                        market_name = "".join(map(chr, perp.name)).strip(" ")

                        dd = {
                            "authority": user_authority + "-" + str(SUB_ID),
                            "name": market_name,
                            "base": pos.base_asset_amount / 1e9,
                            "liq price": chu_sub.get_perp_liq_price(perp_market_index) / 1e6,
                        }

                        positions.append({**pos.__dict__, **dd})

                st.markdown("assets/liabilities")
                spotcol, perpcol = st.columns([1, 1])
                bal_df = pd.DataFrame(balances).T
                bal_df.columns = ["spot" + str(x) for x in bal_df.columns]
                pos_df = pd.DataFrame(positions).T
                pos_df.columns = ["perp" + str(x) for x in pos_df.columns]
                spotcol.dataframe(bal_df.T, use_container_width=True)
                perpcol.dataframe(pos_df.T, use_container_width=True)
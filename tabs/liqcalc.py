import copy
import logging
import time

import pandas as pd
import streamlit as st
from driftpy.account_subscription_config import AccountSubscriptionConfig
from driftpy.accounts.cache import (
    CachedDriftClientAccountSubscriber,
    CachedUserAccountSubscriber,
)
from driftpy.accounts.ws import mainnet_perp_market_configs
from driftpy.addresses import get_user_account_public_key
from driftpy.drift_client import DriftClient
from driftpy.drift_user import DriftUser
from driftpy.math.spot_balance import StrictOraclePrice
from solders.pubkey import Pubkey

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def get_all_subaccounts(drift_client: DriftClient, authority: Pubkey):
    subaccounts = []
    for sub_id in range(7):
        user_account_pk = get_user_account_public_key(
            drift_client.program_id, authority, sub_id
        )
        try:
            user = DriftUser(
                drift_client,
                user_public_key=user_account_pk,
                account_subscription=AccountSubscriptionConfig("cached"),
            )
            if not isinstance(user.account_subscriber, CachedUserAccountSubscriber):
                raise Exception("User account subscriber is not cached")
            await user.account_subscriber.update_cache()

            subaccounts.append(
                {
                    "pubkey": user_account_pk,
                    "sub_id": sub_id,
                    "name": bytes(user.get_user_account().name)
                    .decode("utf-8")
                    .strip("\x00"),
                }
            )
        except Exception:
            continue
    return subaccounts


async def is_subaccount(pubkey: Pubkey, drift_client: DriftClient):
    try:
        user = DriftUser(
            drift_client,
            user_public_key=pubkey,
            account_subscription=AccountSubscriptionConfig("cached"),
        )
        if not isinstance(user.account_subscriber, CachedUserAccountSubscriber):
            raise Exception("User account subscriber is not cached")
        await user.account_subscriber.update_cache()
        if user.get_user_account().authority == pubkey:
            return False
        return user
    except Exception:
        return False


async def liqcalc(drift_client: DriftClient):
    pd.options.plotting.backend = "plotly"
    logger.info("Starting liqcalc")

    authority_str = st.text_input("Authority or User Account:")

    if len(authority_str) <= 5:
        st.write("Enter an authority or user account to get started")
        return

    account = await is_subaccount(Pubkey.from_string(authority_str), drift_client)
    if account:
        st.write(
            f"This account is a subaccount, instead please use `{account.get_user_account().authority}` and select this subaccount."
        )
        return
    try:
        authority = Pubkey.from_string(authority_str)
        subaccounts = await get_all_subaccounts(drift_client, authority)
        if not subaccounts:
            st.error("No subaccounts found for this authority")
            return

        options = [
            f"{sub['sub_id']}: {sub['name']} ({sub['pubkey']})" for sub in subaccounts
        ]
        selected = st.selectbox(
            "Select subaccount:",
            options,
            format_func=lambda x: x,
        )
        if not selected:
            return
        user_pubkey = Pubkey.from_string(selected.split("(")[1].rstrip(")"))

    except Exception:
        user_pubkey = Pubkey.from_string(authority_str)

    mode = st.radio("Mode:", ["raw", "sdk"], index=1)
    if mode != "sdk":
        return

    try:
        user = DriftUser(
            drift_client,
            user_public_key=user_pubkey,
            account_subscription=AccountSubscriptionConfig("cached"),
        )
        if not isinstance(user.account_subscriber, CachedUserAccountSubscriber):
            raise Exception("User account subscriber is not cached")

        await user.account_subscriber.update_cache()

    except Exception as e:
        logger.error(f"Error getting user account: {e}")
        st.write("This account has the following subaccounts:")
        subaccounts = await get_all_subaccounts(drift_client, user_pubkey)
        for subaccount in subaccounts:
            st.markdown(
                f"Subaccount {subaccount['sub_id']} :green[{subaccount['name']}]  👉 {subaccount['pubkey']}",
            )
        return

    st.write("Authority:", user.get_user_account().authority)

    if not isinstance(
        drift_client.account_subscriber, CachedDriftClientAccountSubscriber
    ):
        raise Exception("Clearing house account subscriber is not cached")

    user_authority = str(user.get_user_account().authority)

    st.markdown(
        "### Price Shock",
        help="This section allows you to manipulate the oracle price of selected PERP markets.",
    )
    market_config_map = mainnet_perp_market_configs
    market_names = [market.symbol for market in market_config_map]

    selected_markets = st.multiselect(
        "Select which PERP markets to manipulate the oracle price:",
        market_names,
        default=[market_names[0]] if market_names else [],
        help="Pick the markets whose oracle price you want to shift.",
    )

    price_change_map = {}
    for mkt_name in selected_markets:
        price_change_map[mkt_name] = st.number_input(
            f"{mkt_name} Price Change (%)", value=0, step=1, format="%d"
        )

    show_balance_adjustments = st.checkbox("Adjust balances?", value=False)

    balance_changes = {}
    if show_balance_adjustments:
        st.markdown(
            "### Balance Adjustments",
            help="This section allows you to manipulate the balance of your spot holdings.",
        )

        await drift_client.account_subscriber.update_cache()
        active_spot_positions = user.get_active_spot_positions()
        spot_markets = {
            pos.market_index: user.get_spot_market_account(pos.market_index)
            for pos in active_spot_positions
        }

        for market_idx, spot_market in spot_markets.items():
            if not spot_market:
                continue
            market_name = "".join(map(chr, spot_market.name)).strip()
            current_balance = user.get_token_amount(market_idx) / (
                10**spot_market.decimals
            )

            col1, col2 = st.columns([3, 2])
            with col1:
                st.text(f"Current {market_name} balance: {current_balance:.4f}")
            with col2:
                balance_changes[market_idx] = st.number_input(
                    f"{market_name} Balance Change",
                    value=0.0,
                    step=0.1,
                    format="%.4f",
                    key=f"balance_{market_idx}",
                )

    if st.button("Submit"):
        st.write("Updating caches and calculating positions...")

        await drift_client.account_subscriber.update_cache()
        await user.account_subscriber.update_cache()

        original_cache = drift_client.account_subscriber.cache
        manipulated_cache = copy.deepcopy(original_cache)
        oracle_data = manipulated_cache["oracle_price_data"]

        for mkt_name, pct_change in price_change_map.items():
            cfg = next(
                (
                    market
                    for market in mainnet_perp_market_configs
                    if market.symbol == mkt_name
                ),
                None,
            )
            if not cfg:
                st.error(f"Market {mkt_name} not found")
                return
            oracle_key = str(cfg.oracle)
            if oracle_key in oracle_data:
                old_price = oracle_data[oracle_key].price
                new_price = old_price * (1 + pct_change / 100.0)
                oracle_data[oracle_key].price = new_price
                logger.info(
                    f"Applied {pct_change}% change to {mkt_name}, "
                    f"price: {old_price} -> {new_price}"
                )

        try:
            drift_client.account_subscriber.cache = manipulated_cache

            st.markdown("### Spot Positions")
            spot_positions = []
            for spot_pos in user.get_active_spot_positions():
                market_idx = spot_pos.market_index
                spot_market = user.get_spot_market_account(market_idx)

                token_amount = user.get_token_amount(market_idx)
                if show_balance_adjustments and market_idx in balance_changes:
                    delta = int(
                        balance_changes[market_idx] * (10**spot_market.decimals)
                    )
                    token_amount += delta

                market_name = "".join(map(chr, spot_market.name)).strip()

                oracle_price = user.get_oracle_data_for_spot_market(market_idx)
                if not oracle_price:
                    continue

                strict_oracle_price = StrictOraclePrice(oracle_price.price, None)

                modified_position = copy.deepcopy(spot_pos)
                modified_position.scaled_balance = token_amount

                position = {
                    "Market": market_name,
                    "Current Balance": token_amount / (10**spot_market.decimals),
                    "Cur. Price": oracle_price.price / 1e6,
                    "Liq. Price": user.get_spot_liq_price(
                        market_idx, modified_position.scaled_balance
                    )
                    / 1e6,
                }

                if show_balance_adjustments:
                    position["Delta Applied"] = balance_changes.get(market_idx, 0)

                if token_amount >= 0:
                    val = user.get_spot_asset_value(
                        token_amount, strict_oracle_price, spot_market, None
                    )
                else:
                    val = user.get_spot_liability_value(
                        token_amount, strict_oracle_price, spot_market, None, None
                    )
                position["Net USD Value"] = val / 1e6

                spot_positions.append(position)

            if spot_positions:
                spot_df = pd.DataFrame(spot_positions)

                total_value = spot_df["Net USD Value"].sum()

                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Portfolio Value", f"${total_value:,.2f}")
                with col2:
                    st.metric(
                        "Total Assets",
                        f"${spot_df[spot_df['Net USD Value'] > 0]['Net USD Value'].sum():,.2f}",
                    )
                with col3:
                    st.metric(
                        "Total Liabilities",
                        f"${abs(spot_df[spot_df['Net USD Value'] < 0]['Net USD Value'].sum()):,.2f}",
                    )

                st.dataframe(
                    spot_df.style.format(
                        {
                            "Current Balance": "{:.4f}",
                            "Delta Applied": "{:.4f}",
                            "Cur. Price": "${:.2f}",
                            "Liq. Price": "${:.2f}",
                            "Net USD Value": "${:.2f}",
                        }
                    ),
                    use_container_width=True,
                )
            else:
                st.write("No active spot positions.")

            st.markdown("### Perp Positions")
            perp_positions = []
            for perp_pos in user.get_active_perp_positions():
                if not perp_pos:
                    continue

                market_idx = perp_pos.market_index
                perp_market = user.get_perp_market_account(market_idx)
                market_name = "".join(map(chr, perp_market.name)).strip()
                liq_price = user.get_perp_liq_price(market_idx)
                if not liq_price:
                    continue

                position = {
                    "Authority": f"{user_authority}-0",
                    "Market": market_name,
                    "Base (1e9)": perp_pos.base_asset_amount / 1e9,
                    "Liq. Price": liq_price / 1e6,
                }
                perp_positions.append(position)

            if perp_positions:
                perp_df = pd.DataFrame(perp_positions)
                st.dataframe(perp_df, use_container_width=True)
            else:
                st.write("No active perp positions.")

            st.success("Calculation complete!")

        finally:
            drift_client.account_subscriber.cache = original_cache

    else:
        st.info("Adjust parameters and click 'Submit' to see liquidation calculations.")

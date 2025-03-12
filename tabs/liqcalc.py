import copy
import logging

import pandas as pd
import streamlit as st
from driftpy.account_subscription_config import AccountSubscriptionConfig
from driftpy.accounts.cache import CachedUserAccountSubscriber
from driftpy.addresses import get_user_account_public_key
from driftpy.constants.numeric_constants import QUOTE_PRECISION
from driftpy.drift_client import DriftClient
from driftpy.drift_user import DriftUser
from driftpy.math.spot_balance import StrictOraclePrice
from driftpy.oracles.oracle_id import get_oracle_id
from driftpy.types import SpotPosition
from solders.pubkey import Pubkey

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_market_name(market):
    """Extract human-readable market name from bytes."""
    return bytes(market.name).decode("utf-8").strip("\x00")


async def get_all_subaccounts(drift_client: DriftClient, authority: Pubkey):
    """Find all subaccounts for a given authority."""
    subaccounts = []
    for sub_id in range(10):
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
    """Check if a public key belongs to a subaccount."""
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
    """Main liquidation calculator function."""
    st.title("Drift Liquidation Calculator")

    if "clearing_house_cache" not in st.session_state:
        st.session_state.clearing_house_cache = None
    if "original_token_amounts" not in st.session_state:
        st.session_state.original_token_amounts = {}

    authority_str = st.text_input(
        "Authority or User Account:",
        help="Enter an authority address or subaccount address",
    )

    if not authority_str or len(authority_str) <= 5:
        st.write("Enter an authority or user account to get started")
        return

    try:
        input_pubkey = Pubkey.from_string(authority_str.strip())
    except Exception:
        st.error("Invalid address format")
        return

    account = await is_subaccount(input_pubkey, drift_client)
    if account:
        st.warning(
            f"This address is a subaccount. Please use the authority address `{account.get_user_account().authority}` instead."
        )
        return

    authority_changed = (
        "current_authority" not in st.session_state
        or st.session_state.current_authority != authority_str
    )

    if authority_changed:
        st.session_state.current_authority = authority_str
        st.session_state.clearing_house_cache = None
        st.session_state.user_and_slot = None
        st.session_state.subaccount = None
        st.session_state.original_token_amounts = {}

    authority = Pubkey.from_string(authority_str)

    with st.spinner("Fetching subaccounts..."):
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
        help="Select which subaccount to analyze",
    )

    if not selected:
        return

    user_pubkey = Pubkey.from_string(selected.split("(")[1].rstrip(")"))
    selected_sub_id = int(selected.split(":")[0])

    if st.session_state.clearing_house_cache is None:
        with st.spinner("Initializing market data... (this may take a minute...)"):
            await drift_client.account_subscriber.update_cache()
            st.session_state.clearing_house_cache = copy.deepcopy(
                drift_client.account_subscriber.cache
            )

    drift_client.account_subscriber.cache = copy.deepcopy(
        st.session_state.clearing_house_cache
    )

    subaccount_changed = (
        "subaccount" not in st.session_state
        or st.session_state.subaccount != selected_sub_id
    )

    user = DriftUser(
        drift_client,
        user_public_key=user_pubkey,
        account_subscription=AccountSubscriptionConfig("cached"),
    )

    if not isinstance(user.account_subscriber, CachedUserAccountSubscriber):
        raise Exception("User account subscriber is not cached")

    if subaccount_changed:
        with st.spinner("Loading subaccount data..."):
            await user.account_subscriber.update_cache()
            st.session_state.subaccount = selected_sub_id
            st.session_state.user_and_slot = copy.deepcopy(
                user.account_subscriber.user_and_slot
            )

            for key in list(st.session_state.keys()):
                if key.startswith("price_") or key.startswith("balance_"):
                    del st.session_state[key]

            if not st.session_state.user_and_slot:
                st.info("No data found for this subaccount")
                return
    else:
        user.account_subscriber.user_and_slot = copy.deepcopy(
            st.session_state.user_and_slot
        )

    spot_positions = user.get_active_spot_positions()
    perp_positions = user.get_active_perp_positions()

    if not spot_positions and not perp_positions:
        st.info("No active positions found for this account")
        return

    adjustment_mode = st.radio(
        "Adjustment Mode",
        ["Value", "Percentage"],
        horizontal=True,
        key="adjustment_mode",
        help="Choose whether to adjust by direct value or percentage change",
    )

    oracle_markets = {}

    for pos in perp_positions:
        market = user.get_perp_market_account(pos.market_index)
        oracle_key = get_oracle_id(market.amm.oracle, market.amm.oracle_source)
        if oracle_key not in oracle_markets:
            oracle_markets[oracle_key] = []
        oracle_markets[oracle_key].append(
            {
                "type": "perp",
                "market": market,
                "name": get_market_name(market),
                "index": pos.market_index,
            }
        )

    max_spot_decimals = 0
    for pos in spot_positions:
        market = user.get_spot_market_account(pos.market_index)
        oracle_key = get_oracle_id(market.oracle, market.oracle_source)
        if oracle_key not in oracle_markets:
            oracle_markets[oracle_key] = []
        oracle_markets[oracle_key].append(
            {
                "type": "spot",
                "market": market,
                "name": get_market_name(market),
                "index": pos.market_index,
            }
        )
        max_spot_decimals = max(max_spot_decimals, market.decimals)

    st.subheader(
        "Oracle Price Adjustments",
        help="Adjust oracle prices to see impact on liquidation thresholds",
    )

    price_changes = {}
    price_cols = st.columns(3)

    for i, (oracle_key, markets) in enumerate(oracle_markets.items()):
        market_names = [m["name"] for m in markets]
        market_str = " / ".join(market_names)

        market = markets[0]["market"]
        if markets[0]["type"] == "perp":
            oracle_data = user.get_oracle_data_for_perp_market(markets[0]["index"])
        else:
            oracle_data = user.get_oracle_data_for_spot_market(markets[0]["index"])

        initial_price = float(oracle_data.price) / QUOTE_PRECISION

        value_key = f"price_value_{oracle_key}"
        pct_key = f"price_pct_change_{oracle_key}"
        col = price_cols[i % 3]

        if adjustment_mode == "Value":
            if value_key not in st.session_state:
                if pct_key in st.session_state:
                    st.session_state[value_key] = initial_price * (
                        1 + st.session_state[pct_key] / 100
                    )
                else:
                    st.session_state[value_key] = initial_price

            new_price = col.number_input(
                f"{market_str} price ($)",
                min_value=0.0,
                value=st.session_state[value_key],
                step=0.01,
                key=value_key,
                format="%.6f",
            )
            price_changes[oracle_key] = {"price": new_price}

            if new_price != initial_price:
                st.session_state[pct_key] = (
                    (new_price - initial_price) / initial_price
                ) * 100

        else:
            if pct_key not in st.session_state:
                if value_key in st.session_state:
                    st.session_state[pct_key] = (
                        (st.session_state[value_key] - initial_price) / initial_price
                    ) * 100
                else:
                    st.session_state[pct_key] = 0.0

            pct_change = col.number_input(
                f"{market_str} price change (%)",
                value=st.session_state[pct_key],
                min_value=-99.0,
                max_value=1000.0,
                step=1.0,
                key=pct_key,
                format="%.2f",
            )

            new_price = initial_price * (1 + pct_change / 100)
            price_changes[oracle_key] = {"price": new_price}

            st.session_state[value_key] = new_price

    st.subheader(
        "Balance Adjustments",
        help="Adjust spot token balances to see effect on account health",
    )

    collateral_changes = {}
    collateral_cols = st.columns(3)

    for i, pos in enumerate(spot_positions):
        market = user.get_spot_market_account(pos.market_index)
        tokens = user.get_token_amount(pos.market_index)
        initial_ui_tokens = tokens / (10**market.decimals)

        st.session_state.original_token_amounts[pos.market_index] = tokens

        value_key = f"balance_value_{pos.market_index}"
        pct_key = f"balance_pct_change_{pos.market_index}"
        col = collateral_cols[i % 3]

        if adjustment_mode == "Value":
            if value_key not in st.session_state:
                if pct_key in st.session_state:
                    st.session_state[value_key] = initial_ui_tokens * (
                        1 + st.session_state[pct_key] / 100
                    )
                else:
                    st.session_state[value_key] = initial_ui_tokens

            new_balance = col.number_input(
                f"{get_market_name(market)} balance",
                value=st.session_state[value_key],
                step=10 ** (-market.decimals),
                key=value_key,
                format=f"%.{market.decimals}f",
            )

            collateral_changes[pos.market_index] = {
                "balance": new_balance,
                "decimals": market.decimals,
            }

            if new_balance != initial_ui_tokens and initial_ui_tokens != 0:
                st.session_state[pct_key] = (
                    (new_balance - initial_ui_tokens) / initial_ui_tokens
                ) * 100
            else:
                st.session_state[pct_key] = 0

        else:
            if pct_key not in st.session_state:
                if value_key in st.session_state and initial_ui_tokens != 0:
                    st.session_state[pct_key] = (
                        (st.session_state[value_key] - initial_ui_tokens)
                        / initial_ui_tokens
                    ) * 100
                else:
                    st.session_state[pct_key] = 0.0

            pct_change = col.number_input(
                f"{get_market_name(market)} balance change (%)",
                value=st.session_state[pct_key],
                min_value=-100.0,
                step=1.0,
                key=pct_key,
                format="%.2f",
            )

            new_balance = initial_ui_tokens * (1 + pct_change / 100)
            collateral_changes[pos.market_index] = {
                "balance": new_balance,
                "decimals": market.decimals,
            }

            st.session_state[value_key] = new_balance

    if price_changes:
        cache = copy.deepcopy(drift_client.account_subscriber.cache)
        oracle_price_data = cache["oracle_price_data"]

        for oracle_key, change_info in price_changes.items():
            if oracle_key in oracle_price_data:
                oracle_data = oracle_price_data[oracle_key]
                if oracle_data and hasattr(oracle_data, "data"):
                    original_price = oracle_data.data.price
                    new_price = int(change_info["price"] * QUOTE_PRECISION)
                    oracle_data.data.price = new_price
                    logger.info(
                        f"Applied price change to {oracle_key}, "
                        f"price: {original_price / QUOTE_PRECISION:.6f} -> {new_price / QUOTE_PRECISION:.6f}"
                    )

        drift_client.account_subscriber.cache = cache

    if collateral_changes:
        new_positions = []
        for pos in user.get_user_account().spot_positions:
            if pos.market_index in collateral_changes:
                change_info = collateral_changes[pos.market_index]
                original_tokens = st.session_state.original_token_amounts[
                    pos.market_index
                ]
                new_tokens = int(
                    change_info["balance"] * (10 ** change_info["decimals"])
                )

                if original_tokens != 0:
                    new_scaled_balance = int(
                        pos.scaled_balance * (new_tokens / original_tokens)
                    )
                else:
                    new_scaled_balance = new_tokens

                new_position = SpotPosition(
                    scaled_balance=new_scaled_balance,
                    open_bids=pos.open_bids,
                    open_asks=pos.open_asks,
                    cumulative_deposits=pos.cumulative_deposits,
                    market_index=pos.market_index,
                    balance_type=pos.balance_type,
                    open_orders=pos.open_orders,
                    padding=pos.padding,
                )
                new_positions.append(new_position)

                logger.info(
                    f"Applied balance change to {pos.market_index}, "
                    f"balance: {original_tokens / (10 ** change_info['decimals']):.6f} -> "
                    f"{new_tokens / (10 ** change_info['decimals']):.6f}"
                )
            else:
                new_positions.append(pos)

        user.account_subscriber.user_and_slot.data.spot_positions = new_positions

    with st.spinner("Calculating liquidation thresholds..."):
        spot_data = []
        for pos in spot_positions:
            market = user.get_spot_market_account(pos.market_index)
            tokens = user.get_token_amount(pos.market_index)
            oracle_price_data = user.get_oracle_data_for_spot_market(pos.market_index)

            if oracle_price_data:
                oracle_price = float(oracle_price_data.price) / QUOTE_PRECISION
                balance = tokens / (10**market.decimals)
                liq_price = (
                    float(user.get_spot_liq_price(pos.market_index)) / QUOTE_PRECISION
                )

                strict_oracle_price = StrictOraclePrice(oracle_price_data.price, None)
                if tokens >= 0:
                    val = user.get_spot_asset_value(
                        tokens, strict_oracle_price, market, None
                    )
                else:
                    val = user.get_spot_liability_value(
                        tokens, strict_oracle_price, market, None, None
                    )

                net_value = val / QUOTE_PRECISION

                spot_data.append(
                    {
                        "Name": get_market_name(market),
                        "Balance": balance,
                        "Price ($)": oracle_price,
                        "Net Value ($)": net_value,
                        "Liquidation Price ($)": liq_price,
                    }
                )

        perp_data = []
        for pos in perp_positions:
            market = user.get_perp_market_account(pos.market_index)
            oracle_price_data = user.get_oracle_data_for_perp_market(pos.market_index)

            if oracle_price_data:
                oracle_price = float(oracle_price_data.price) / QUOTE_PRECISION
                base_size = pos.base_asset_amount / 1e9
                notional = base_size * oracle_price
                liq_price = (
                    float(user.get_perp_liq_price(pos.market_index)) / QUOTE_PRECISION
                )

                perp_data.append(
                    {
                        "Name": get_market_name(market),
                        "Base Size": base_size,
                        "Price ($)": oracle_price,
                        "Notional ($)": notional,
                        "Liquidation Price ($)": liq_price,
                    }
                )

    st.subheader("Position Analysis")
    result_cols = st.columns([1, 1])

    with result_cols[0]:
        if spot_data:
            st.markdown("#### Spot Positions")
            spot_df = pd.DataFrame(spot_data)

            total_assets = sum([v for v in spot_df["Net Value ($)"] if v > 0])
            total_liabilities = sum([abs(v) for v in spot_df["Net Value ($)"] if v < 0])

            metrics_cols = st.columns(2)
            metrics_cols[0].metric("Total Assets", f"${total_assets:,.2f}")
            metrics_cols[1].metric("Total Liabilities", f"${total_liabilities:,.2f}")

            st.dataframe(
                spot_df,
                column_config={
                    "Name": st.column_config.TextColumn("Market"),
                    "Balance": st.column_config.NumberColumn(
                        format=f"%.{max_spot_decimals}f"
                    ),
                    "Price ($)": st.column_config.NumberColumn(
                        "Price ($)", format="$%.2f"
                    ),
                    "Net Value ($)": st.column_config.NumberColumn(
                        "Net Value ($)", format="$%.2f"
                    ),
                    "Liquidation Price ($)": st.column_config.NumberColumn(
                        "Liq. Price ($)", format="$%.2f"
                    ),
                },
                use_container_width=True,
            )
        else:
            st.info("No spot positions to display")

    with result_cols[1]:
        if perp_data:
            st.markdown("#### Perp Positions")
            perp_df = pd.DataFrame(perp_data)

            total_long_notional = sum([v for v in perp_df["Notional ($)"] if v > 0])
            total_short_notional = sum(
                [abs(v) for v in perp_df["Notional ($)"] if v < 0]
            )

            metrics_cols = st.columns(2)
            metrics_cols[0].metric("Total Long", f"${total_long_notional:,.2f}")
            metrics_cols[1].metric("Total Short", f"${total_short_notional:,.2f}")

            st.dataframe(
                perp_df,
                column_config={
                    "Name": st.column_config.TextColumn("Market"),
                    "Base Size": st.column_config.NumberColumn(format="%.9f"),
                    "Price ($)": st.column_config.NumberColumn(
                        "Price ($)", format="$%.2f"
                    ),
                    "Notional ($)": st.column_config.NumberColumn(
                        "Notional ($)", format="$%.2f"
                    ),
                    "Liquidation Price ($)": st.column_config.NumberColumn(
                        "Liq. Price ($)", format="$%.2f"
                    ),
                },
                use_container_width=True,
            )
        else:
            st.info("No perp positions to display")

    health = user.get_health()
    health_color = "normal"
    if health < 20:
        health_color = "off"
    elif health < 50:
        health_color = "error"

    st.metric("Account Health", f"{health:.2f}%", delta_color=health_color)

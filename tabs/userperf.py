# old dashboard: for each user:
# - net deposits
# - all time pnl + fees
# - total volume
# - position over time

import datetime

import numpy as np
import pandas as pd
import plotly.graph_objs as go
import streamlit as st
from driftpy.accounts import (
    UserAccount,
    get_state_account,
)
from driftpy.accounts.oracle import *
from driftpy.addresses import *
from driftpy.constants.numeric_constants import *
from driftpy.drift_client import DriftClient
from solana.rpc.types import MemcmpOpts
from solders.pubkey import Pubkey

from constants import ALL_MARKET_NAMES
from tabs.fee_income import get_trades_for_range_pandas

pd.options.plotting.backend = "plotly"

markout_periods = ["t0", "t5", "t10", "t30", "t60"]


def load_trade_history(market_symbol, start_date, end_date):
    df = get_trades_for_range_pandas(market_symbol, start_date, end_date)
    return df


def filter_dups(df):
    df = df.drop_duplicates(
        [
            "fillerReward",
            "baseAssetAmountFilled",
            "quoteAssetAmountFilled",
            "takerPnl",
            "makerPnl",
            "takerFee",
            "makerRebate",
            "refereeDiscount",
            "quoteAssetAmountSurplus",
            "takerOrderBaseAssetAmount",
            "takerOrderCumulativeBaseAssetAmountFilled",
            "takerOrderCumulativeQuoteAssetAmountFilled",
            "takerOrderFee",
            "makerOrderBaseAssetAmount",
            "makerOrderCumulativeBaseAssetAmountFilled",
            "makerOrderCumulativeQuoteAssetAmountFilled",
            "makerOrderFee",
            "oraclePrice",
            "makerFee",
            "txSig",
            "slot",
            "ts",
            "action",
            "actionExplanation",
            "marketIndex",
            "marketType",
            "filler",
            "fillRecordId",
            "taker",
            "takerOrderId",
            "takerOrderDirection",
            "maker",
            "makerOrderId",
            "makerOrderDirection",
            "spotFulfillmentMethodFee",
        ]
    ).reset_index(drop=True)
    return df


def process_trades_df(
    fairs_marked_out_df: pd.DataFrame, raw_trades_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Adds some columns to the market_trades_df for analysis in a vectorized way
    """
    # Select required columns
    filtered = raw_trades_df[
        [
            "filler",
            "fillRecordId",
            "taker",
            "takerOrderId",
            "takerOrderDirection",
            "takerFee",
            "maker",
            "makerOrderId",
            "makerOrderDirection",
            "makerFee",
            "baseAssetAmountFilled",
            "quoteAssetAmountFilled",
            "oraclePrice",
            "actionExplanation",
            "txSig",
            "slot",
        ]
    ].copy()

    # Vectorized operations
    filtered["makerBaseSigned"] = np.where(
        filtered["makerOrderDirection"] == "long",
        filtered["baseAssetAmountFilled"],
        filtered["baseAssetAmountFilled"] * -1,
    )
    filtered["makerQuoteSigned"] = np.where(
        filtered["makerOrderDirection"] == "long",
        -1 * filtered["quoteAssetAmountFilled"],
        filtered["quoteAssetAmountFilled"],
    )
    filtered["takerBaseSigned"] = np.where(
        filtered["takerOrderDirection"] == "long",
        filtered["baseAssetAmountFilled"],
        filtered["baseAssetAmountFilled"] * -1,
    )
    filtered["takerQuoteSigned"] = np.where(
        filtered["takerOrderDirection"] == "long",
        -1 * filtered["quoteAssetAmountFilled"],
        filtered["quoteAssetAmountFilled"],
    )
    filtered["fillPrice"] = (
        filtered["quoteAssetAmountFilled"] / filtered["baseAssetAmountFilled"]
    )
    filtered["isFillerMaker"] = filtered["filler"] == filtered["maker"]
    filtered["isFillerTaker"] = filtered["filler"] == filtered["taker"]
    filtered["makerOrderDirectionNum"] = np.where(
        filtered["makerOrderDirection"] == "long", 1, -1
    )
    filtered["takerOrderDirectionNum"] = np.where(
        filtered["takerOrderDirection"] == "long", 1, -1
    )

    # # Find closest times in fairs_marked_out_df for each trade
    # closest_times = fairs_marked_out_df.index.get_indexer(filtered.index, method='nearest')

    # # Add price_t0, price_t5, etc. from fairs_marked_out_df
    # for markout_period in markout_periods:
    #     filtered[f'price_{markout_period}'] = fairs_marked_out_df[f'price_{markout_period}'].iloc[closest_times].values

    #     # Calculate markouts
    #     filtered[f'maker_markout_{markout_period}'] = np.where(
    #         filtered['makerOrderDirectionNum'] == 1,
    #         filtered['baseAssetAmountFilled'] * (filtered[f'price_{markout_period}'] - filtered['fillPrice']),
    #         filtered['baseAssetAmountFilled'] * (filtered['fillPrice'] - filtered[f'price_{markout_period}'])
    #     )
    #     filtered[f'taker_markout_{markout_period}'] = np.where(
    #         filtered['takerOrderDirectionNum'] == 1,
    #         filtered['baseAssetAmountFilled'] * (filtered[f'price_{markout_period}'] - filtered['fillPrice']),
    #         filtered['baseAssetAmountFilled'] * (filtered['fillPrice'] - filtered[f'price_{markout_period}'])
    #     )

    return filtered


def render_trades_stats_for_user_account(processed_trades_df, filter_ua):
    if filter_ua is None:
        user_trades_df = processed_trades_df.loc[
            (processed_trades_df["maker"].isna())
            | (processed_trades_df["taker"].isna())
        ].copy()
    else:
        user_trades_df = processed_trades_df.loc[
            (processed_trades_df["maker"] == filter_ua)
            | (processed_trades_df["taker"] == filter_ua)
        ].copy()

    user_trades_df["isMaker"] = user_trades_df["maker"] == filter_ua
    user_trades_df["counterparty"] = np.where(
        user_trades_df["maker"] == filter_ua,
        user_trades_df["taker"],
        user_trades_df["maker"],
    )
    user_trades_df["user_direction"] = np.where(
        user_trades_df["maker"] == filter_ua,
        user_trades_df["makerOrderDirection"],
        user_trades_df["takerOrderDirection"],
    )

    user_trades_df["user_direction_num"] = np.where(
        user_trades_df["maker"] == filter_ua,
        user_trades_df["makerOrderDirectionNum"],
        user_trades_df["takerOrderDirectionNum"],
    )

    user_trades_df["user_fee_recv"] = np.where(
        user_trades_df["maker"] == filter_ua,
        -1 * user_trades_df["makerFee"],
        -1 * user_trades_df["takerFee"],
    )

    user_trades_df["user_base"] = np.where(
        user_trades_df["maker"] == filter_ua,
        user_trades_df["makerBaseSigned"],
        user_trades_df["takerBaseSigned"],
    )

    user_trades_df["user_quote"] = np.where(
        user_trades_df["maker"] == filter_ua,
        user_trades_df["makerQuoteSigned"],
        user_trades_df["takerQuoteSigned"],
    )

    # for markout_period in markout_periods:
    # 	user_trades_df[f'user_markout_{markout_period}'] = np.where(
    # 		user_trades_df['maker'] == filter_ua,
    # 		user_trades_df[f'maker_markout_{markout_period}'],
    # 		user_trades_df[f'taker_markout_{markout_period}'],
    # 	)

    user_trades_df["user_cum_base"] = user_trades_df[
        "user_base"
    ].cumsum()  # base_position
    user_trades_df["user_cum_base_prev"] = (
        user_trades_df["user_cum_base"].shift(1).fillna(0)
    )  # base_position_prev
    user_trades_df["user_cum_quote"] = user_trades_df["user_quote"].cumsum()

    # update types:
    # 0: flip pos
    # 1: increase pos
    # -1: decrease pos

    user_trades_df["position_update"] = 0
    user_trades_df["user_quote_entry_amount"] = 0.0
    user_trades_df["user_quote_breakeven_amount"] = 0.0
    user_trades_df["realized_pnl"] = 0.0

    for i in range(0, len(user_trades_df)):
        prev_row = user_trades_df.iloc[i - 1]
        current_row = user_trades_df.iloc[i]

        prev_quote_entry_amt = prev_row["user_quote_entry_amount"]
        prev_quote_breakeven_amt = prev_row["user_quote_breakeven_amount"]
        delta_base_amt = np.abs(current_row["user_base"])
        curr_base_amt = np.abs(prev_row["user_cum_base"])

        if current_row["user_cum_base"] * current_row["user_cum_base_prev"] < 0:
            # flipped direction
            user_trades_df.loc[user_trades_df.index[i], "position_update"] = 0
            # same for BE and entry
            new_quote = current_row["user_quote"] - (
                current_row["user_quote"] * curr_base_amt / delta_base_amt
            )
            user_trades_df.loc[user_trades_df.index[i], "user_quote_entry_amount"] = (
                new_quote
            )
            user_trades_df.loc[
                user_trades_df.index[i], "user_quote_breakeven_amount"
            ] = new_quote
            user_trades_df.loc[user_trades_df.index[i], "realized_pnl"] = prev_row[
                "user_quote_entry_amount"
            ] + (current_row["user_quote"] - new_quote)
        elif current_row["user_cum_base_prev"] == 0:
            # opening new position
            user_trades_df.loc[user_trades_df.index[i], "position_update"] = 1
            user_trades_df.loc[user_trades_df.index[i], "user_quote_entry_amount"] = (
                prev_quote_entry_amt + current_row["user_quote"]
            )
            user_trades_df.loc[
                user_trades_df.index[i], "user_quote_breakeven_amount"
            ] = prev_quote_breakeven_amt + current_row["user_quote"]
        else:
            if current_row["user_direction_num"] == np.sign(
                current_row["user_cum_base_prev"]
            ):
                # increase position
                user_trades_df.loc[user_trades_df.index[i], "position_update"] = 1
                user_trades_df.loc[
                    user_trades_df.index[i], "user_quote_entry_amount"
                ] = prev_quote_entry_amt + current_row["user_quote"]
                user_trades_df.loc[
                    user_trades_df.index[i], "user_quote_breakeven_amount"
                ] = prev_quote_breakeven_amt + current_row["user_quote"]
                user_trades_df.loc[user_trades_df.index[i], "realized_pnl"] = 0
            else:
                # decrease position
                user_trades_df.loc[user_trades_df.index[i], "position_update"] = -1
                new_quote_entry_amt = prev_quote_entry_amt - (
                    prev_quote_entry_amt * delta_base_amt / curr_base_amt
                )
                user_trades_df.loc[
                    user_trades_df.index[i], "user_quote_entry_amount"
                ] = new_quote_entry_amt
                user_trades_df.loc[
                    user_trades_df.index[i], "user_quote_breakeven_amount"
                ] = prev_quote_breakeven_amt - (
                    prev_quote_breakeven_amt * delta_base_amt / curr_base_amt
                )
                user_trades_df.loc[user_trades_df.index[i], "realized_pnl"] = (
                    prev_row["user_quote_entry_amount"]
                    - new_quote_entry_amt
                    + current_row["user_quote"]
                )

    user_trades_df["avg_price"] = np.abs(
        user_trades_df["user_quote_entry_amount"] / user_trades_df["user_cum_base"]
    )
    user_trades_df["user_cum_fee"] = user_trades_df["user_fee_recv"].cumsum()
    user_trades_df["user_cum_pnl"] = user_trades_df["realized_pnl"].cumsum()

    return user_trades_df


def plot_cumulative_pnl_for_user_account(user_trades_df, filter_ua):
    # Create a new figure for cumulative base and quote plots
    # fig_cumulative = make_subplots(rows=2, cols=1, shared_xaxes=True, subplot_titles=('Cumulative Base Asset', 'Cumulative PnL'))

    fig_1 = go.Figure()
    fig_1.update_layout(
        title_text=f"Cum. Base Amounts ({filter_ua})",
        height=400,
        width=1200,
        showlegend=True,
    )

    fig_2 = go.Figure()
    fig_2.update_layout(
        title_text=f"Cum. PnL({filter_ua})",
        height=400,
        width=1200,
        showlegend=True,
        barmode="overlay",
    )

    marker_size = 3

    fig_1.add_trace(
        go.Scatter(
            x=user_trades_df.index,
            y=user_trades_df["user_cum_base"],
            mode="lines+markers",
            name="Cumulative Base",
            line=dict(width=1),
            marker=dict(size=marker_size),
        ),
        # row=1, col=1
    )

    fig_2.add_trace(
        go.Scatter(
            x=user_trades_df.index,
            y=user_trades_df["user_cum_fee"],
            mode="lines+markers",
            name="Cumulative Fee Received",
            line=dict(width=1),
            marker=dict(size=marker_size),
        ),
        # row=2, col=1
    )
    fig_2.add_trace(
        go.Scatter(
            x=user_trades_df.index,
            y=user_trades_df["user_cum_pnl"],
            mode="lines+markers",
            name="Cumulative PnL",
            line=dict(width=1),
            marker=dict(size=marker_size),
        ),
        # row=2, col=1
    )

    st.plotly_chart(fig_1, use_container_width=True)
    st.plotly_chart(fig_2, use_container_width=True)


async def show_user_perf_for_authority(
    dc: DriftClient,
    user_authority: str,
    market_symbol: str,
    start_date: datetime.date,
    end_date: datetime.date,
):
    user_authority_pk = Pubkey.from_string(user_authority)

    user_stats_pk = get_user_stats_account_public_key(dc.program_id, user_authority_pk)
    user_stat = await dc.program.account["UserStats"].fetch(user_stats_pk)

    users = await dc.program.account["User"].all(
        filters=[
            MemcmpOpts(offset=0, bytes="TfwwBiNJtao"),
            MemcmpOpts(offset=8, bytes=user_authority),
        ]
    )

    user_names = [
        bytes(x.account.name).decode("utf-8", errors="ignore").strip()
        + f" ({x.account.sub_account_id})"
        for x in users
    ]
    user_selected = st.selectbox("select subaccount:", user_names)
    print("selected:", user_selected)
    selected_user: UserAccount = users[user_names.index(user_selected)].account
    selected_user_pk = users[user_names.index(user_selected)].public_key
    print("selected user:", str(selected_user_pk))
    print("selected user:", selected_user.perp_positions)
    print("market symbol:", market_symbol)
    print("start date:", start_date)
    print("end date:", end_date)

    market_trades_df = load_trade_history(market_symbol, start_date, end_date)
    st.write(market_trades_df)
    processed_trades_df = process_trades_df(None, market_trades_df)
    print(selected_user_pk)
    user_trades = render_trades_stats_for_user_account(
        processed_trades_df, selected_user_pk
    )
    st.write(user_trades)
    plot_cumulative_pnl_for_user_account(user_trades, selected_user_pk)
    # chu = DriftUser(
    #     dc,
    #     user_public_key=selected_user.public_key,
    #     account_subscription=AccountSubscriptionConfig("cached"),
    # )
    # await chu.drift_client.account_subscriber.update_cache()
    # await chu.account_subscriber.update_cache()

    # user_acct = chu.get_user_account()
    # nom = bytes(user_acct.name).decode("utf-8")
    # st.write('"' + nom + '"')


async def show_user_perf(clearing_house: DriftClient):
    # print("loading frens")
    # frens = get_loaded_auths()
    # print("frens loaded", frens)

    # st.write('query string:', frens)

    state = await get_state_account(clearing_house.program)
    ch = clearing_house

    # start = timeit.default_timer()
    # every_user_stats = await ch.program.account["UserStats"].all()
    # end = timeit.default_timer()
    # authorities = sorted([str(x.account.authority) for x in every_user_stats])
    authority0, mol0, mol2, market_symbol0 = st.columns([10, 3, 3, 3])

    authority = authority0.text_input(
        "authority", value="GXyE3Snk3pPYX4Nz9QRVBrnBfbJRTAQYxuy5DRdnebAn"
    )
    market_symbol = market_symbol0.selectbox(
        "market symbol", ALL_MARKET_NAMES, index=ALL_MARKET_NAMES.index("SOL-PERP")
    )
    # if len(frens) == 0:
    #     user_authorities = authority0.selectbox(
    #         "user authorities",
    #         authorities,
    #         0,
    #         # on_change=ccc
    #         # frens
    #     )
    #     # user_authorities = ['4FbQvke11D4EdHVsCD3xej2Pncp4LFMTXWJUXv7irxTj']
    #     user_authorities = [user_authorities]
    # else:
    #     user_authorities = authority0.selectbox(
    #         "user authorities",
    #         frens,
    #         0,
    #         # on_change=ccc
    #         # frens
    #     )
    #     user_authorities = frens
    # print(user_authorities)
    # user_authority = user_authorities[0]

    # await chu.set_cache()
    # cache = chu.CACHE

    # user_stats_pk = get_user_stats_account_public_key(ch.program_id, user_authority_pk)
    # all_user_stats = await ch.program.account['UserStats'].fetch(user_stats_pk)
    # user_stats = [
    #     x.account
    #     for x in every_user_stats
    #     if str(x.account.authority) in user_authorities
    # ]
    # user_stats = []
    # all_summarys = []
    # balances = []
    # positions = []
    # url = 'https://drift-historical-data.s3.eu-west-1.amazonaws.com/program/dRiftyHA39MWEi3m9aunc5MzRF1JYuBsbn6VPcn33UH/'
    # url += 'user/%s/trades/%s/%s'
    # userAccountKeys = []
    # user_stat = None
    # for user_authority in user_authorities:
    #     user_authority_pk = Pubkey.from_string(user_authority)
    #     user_stats_pk = get_user_stats_account_public_key(ch.program_id, user_authority_pk)
    #     user_stat = await ch.program.account['UserStats'].fetch(user_stats_pk)
    #     for sub_id in range(user_stat.number_of_sub_accounts_created):
    #         user_account_pk = get_user_account_public_key(
    #             clearing_house.program_id, user_authority_pk, sub_id
    #         )
    #         userAccountKeys.append(user_account_pk)
    # st.write(
    #     "Authority owned Drift User Accounts:",
    # )
    # uak_df = pd.DataFrame(userAccountKeys, columns=["userAccountPublicKey"])
    # uak_df.index = ["subaccount_" + str(x) for x in uak_df.index]
    # st.dataframe(uak_df.T)

    lastest_date = pd.to_datetime(datetime.datetime.now(), utc=True)
    start_date = mol0.date_input(
        "start date:",
        lastest_date - datetime.timedelta(days=1),
        min_value=datetime.datetime(2022, 11, 4),
        max_value=lastest_date,
    )  # (datetime.datetime.now(tzInfo)))
    end_date = mol2.date_input(
        "end date:",
        lastest_date,
        min_value=datetime.datetime(2022, 11, 4),
        max_value=lastest_date,
    )  # (datetime.datetime.now(tzInfo)))
    # dates = pd.date_range(start_date, end_date)

    await show_user_perf_for_authority(
        clearing_house, authority, market_symbol, start_date, end_date
    )
    # for user_authority in user_authorities:
    # st.markdown('user stats')
    # st.dataframe(pd.DataFrame([x for x in user_stats]).T)

    # else:
    #     st.text('not found')

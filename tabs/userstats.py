import os
import time

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from driftpy.drift_client import DriftClient
from solders.keypair import Keypair

from datafetch.transaction_fetch import transaction_history_for_account

pd.options.plotting.backend = "plotly"


async def show_user_stats(clearing_house: DriftClient):
    ch = clearing_house

    if "user_stats_df" not in st.session_state:
        all_user_stats = await ch.program.account["UserStats"].all()
        kp = Keypair()
        ch = DriftClient(ch.program, kp)

        df_rr = pd.DataFrame([x.account.__dict__ for x in all_user_stats])
        fees_df = pd.DataFrame([x.account.fees.__dict__ for x in all_user_stats])

        df = pd.concat([df_rr, fees_df], axis=1)
        for x in df.columns:
            if x in [
                "taker_volume30d",
                "maker_volume30d",
                "filler_volume30d",
                "total_fee_paid",
                "total_fee_rebate",
                "total_referrer_reward",
                "current_epoch_referrer_reward",
                "if_staked_quote_asset_amount",
            ]:
                df[x] /= 1e6

        current_ts = time.time()
        df["last_trade_seconds_ago"] = int(current_ts) - df[
            ["last_taker_volume30d_ts", "last_maker_volume30d_ts"]
        ].max(axis=1).astype(int)
        df["last_fill_seconds_ago"] = int(current_ts) - df[
            "last_filler_volume30d_ts"
        ].astype(int)

        volume_scale = (1 - df["last_trade_seconds_ago"] / (60 * 60 * 24 * 30)).apply(
            lambda x: max(0, x)
        )
        fill_scale = (1 - df["last_fill_seconds_ago"] / (60 * 60 * 24 * 30)).apply(
            lambda x: max(0, x)
        )

        df["filler_volume30d_calc"] = (
            df[["filler_volume30d"]].sum(axis=1).mul(fill_scale, axis=0)
        )
        df["taker_volume30d_calc"] = (
            df[["taker_volume30d"]].sum(axis=1).mul(volume_scale, axis=0)
        )
        df["maker_volume30d_calc"] = (
            df[["maker_volume30d"]].sum(axis=1).mul(volume_scale, axis=0)
        )
        df["total_30d_volume_calc"] = (
            df[["taker_volume30d", "maker_volume30d"]]
            .sum(axis=1)
            .mul(volume_scale, axis=0)
        )
        df["authority"] = df["authority"].astype(str)
        df["referrer"] = df["referrer"].astype(str)

        df = (
            df[
                [
                    "authority",
                    "total_30d_volume_calc",
                    "taker_volume30d_calc",
                    "maker_volume30d_calc",
                    "last_trade_seconds_ago",
                    "taker_volume30d",
                    "maker_volume30d",
                    "filler_volume30d_calc",
                    "filler_volume30d",
                    "total_fee_paid",
                    "total_fee_rebate",
                    "number_of_sub_accounts",
                    "if_staked_quote_asset_amount",
                    "referrer",
                    "total_referrer_reward",
                    "current_epoch_referrer_reward",
                    "last_fill_seconds_ago",
                    "last_taker_volume30d_ts",
                    "last_maker_volume30d_ts",
                    "last_filler_volume30d_ts",
                ]
            ]
            .sort_values("last_trade_seconds_ago")
            .reset_index(drop=True)
        )

        st.session_state.user_stats_df = df

    df = st.session_state.user_stats_df

    tab_names = ["Volume", "New Signups", "Refferals", "Fillers", "Clusters"]
    if "userstats_active_tab" not in st.session_state:
        st.session_state.userstats_active_tab = tab_names[0]

    selected_tab = st.segmented_control(
        "Select View", options=tab_names, default=st.session_state.userstats_active_tab
    )

    st.session_state.userstats_active_tab = selected_tab

    if selected_tab == "Volume":
        render_volume_tab(df)
    elif selected_tab == "New Signups":
        await render_new_signups_tab(df, clearing_house)
    elif selected_tab == "Refferals":
        render_referrals_tab(df)
    elif selected_tab == "Fillers":
        render_fillers_tab(df)
    elif selected_tab == "Clusters":
        render_clusters_tab(df)


def render_volume_tab(df):
    pie1, pie2 = st.columns(2)

    net_vamm_maker_volume = (
        df["taker_volume30d_calc"].sum() - df["maker_volume30d_calc"].sum()
    )

    other = pd.DataFrame(
        df.sort_values("taker_volume30d_calc", ascending=False).iloc[10:].sum(axis=0)
    ).T
    other["authority"] = "Other"
    dfmin = pd.concat(
        [df.sort_values("taker_volume30d_calc", ascending=False).head(10), other],
        axis=0,
    )
    dfmin["authority"] = dfmin["authority"].apply(
        lambda x: str(x)[:4] + "..." + str(x)[-4:] if x != "Other" else x
    )

    fig = px.pie(
        dfmin,
        values="taker_volume30d_calc",
        names="authority",
        title="30D Taker Volume Breakdown ("
        + str(int(df["taker_volume30d_calc"].pipe(np.sign).sum()))
        + " unique)",
        hover_data=["taker_volume30d_calc"],
    )
    pie1.plotly_chart(fig)

    other = pd.DataFrame(
        df.sort_values("maker_volume30d_calc", ascending=False).iloc[10:].sum(axis=0)
    ).T
    if net_vamm_maker_volume > 0:
        vamm = other.copy()
        vamm.maker_volume30d_calc = net_vamm_maker_volume
        other = pd.concat([other, vamm])
    other["authority"] = ["Other", "vAMM"]
    dfmin = pd.concat(
        [df.sort_values("maker_volume30d_calc", ascending=False).head(10), other],
        axis=0,
    )
    dfmin["authority"] = dfmin["authority"].apply(
        lambda x: str(x)[:4] + "..." + str(x)[-4:] if x not in ["Other", "vAMM"] else x
    )

    fig = px.pie(
        dfmin,
        values="maker_volume30d_calc",
        names="authority",
        title="30D Maker Volume Breakdown ("
        + str(int(df["maker_volume30d_calc"].pipe(np.sign).sum()))
        + " unique)",
        hover_data=["maker_volume30d_calc"],
    )
    pie2.plotly_chart(fig)

    col1, col2, col3 = st.columns(3)
    col1.metric(
        "30D User Taker Volume",
        str(np.round(df["taker_volume30d_calc"].sum() / 1e6, 2)) + "M",
        str(int(df["taker_volume30d_calc"].pipe(np.sign).sum())) + " unique",
    )
    col2.metric(
        "30D User Maker Volume",
        str(np.round(df["maker_volume30d_calc"].sum() / 1e6, 2)) + "M",
        str(int(df["maker_volume30d_calc"].pipe(np.sign).sum())) + " unique",
    )
    col3.metric(
        "30D vAMM Volume (Net Maker)",
        str(np.round(net_vamm_maker_volume / 1e6, 2)) + "M",
    )

    st.dataframe(df)
    df2download2 = df.to_csv(escapechar='"').encode("utf-8")
    st.download_button(
        label="user stats full [bytes=" + str(len(df)) + "]",
        data=df2download2,
        file_name="userstats_full.csv",
        mime="text/csv",
    )

    dd = (
        df.set_index("last_trade_seconds_ago")[
            [
                "taker_volume30d",
                "maker_volume30d",
            ]
        ]
        .pipe(np.sign)
        .cumsum()
        .loc[: 60 * 60 * 24]
    )
    dd.index /= 3600
    active_users_past_24hrs = int(dd.values[-1].max())
    st.plotly_chart(
        dd.plot(title="# user active over past 24hr = " + str(active_users_past_24hrs))
    )


async def render_new_signups_tab(df, clearing_house):
    dfx = df.set_index("authority")
    df2 = dfx[dfx["filler_volume30d"] == 0]
    dd = df2[
        [
            "last_filler_volume30d_ts",
            "last_taker_volume30d_ts",
            "last_maker_volume30d_ts",
        ]
    ].min(axis=1)
    tt = (df2[["last_fill_seconds_ago"]] / 60).round(1)
    totalfee = df2["total_fee_paid"]
    res1 = pd.concat([dd, tt, totalfee], axis=1)
    res1["status"] = "trader"
    res1.columns = ["min date", "minutes_ago", "total_fee_paid", "status"]

    df2 = dfx[dfx["filler_volume30d"] != 0]
    dd = df2[
        [
            "last_filler_volume30d_ts",
            "last_taker_volume30d_ts",
            "last_maker_volume30d_ts",
        ]
    ].min(axis=1)
    tt = (df2[["last_fill_seconds_ago"]] / 60).round(1)
    res2 = pd.concat([dd, tt], axis=1)
    res2["status"] = "filler"

    res2.columns = ["min date", "minutes_ago", "status"]

    df2 = pd.concat([res1, res2])

    df2["min date"] = pd.to_datetime(df2["min date"] * 1e9)
    df2 = df2[df2["minutes_ago"] <= 60 * 24]

    df2 = df2.sort_values("min date", ascending=False)
    unumbers = [len(df2) - (x) for x in range(len(df2))]
    df2["number"] = unumbers
    st.dataframe(df2)

    filename = "user_signup_funding_source2.csv"

    if os.path.exists(filename):
        existing_data = pd.read_csv(filename, index_col=0)
    else:
        existing_data = pd.DataFrame(
            columns=["blockTime", "sourceAddress", "sourceTransactionSignature"]
        )

    connection = clearing_house.program.provider.connection
    limit = 1000
    max_limit = limit * 20

    o1, o2 = st.columns([1, 5])
    dorun = o1.radio(
        "run fundingSource fetch:", [True, False], index=1, horizontal=True
    )

    if len(existing_data):
        o2.dataframe(existing_data.groupby("sourceAddress").count())

    if dorun:
        # Loop over df2.index
        for addy1 in df2.index:
            # Check if this index is already in the existing data
            if addy1 in existing_data.index:
                continue

            # Fetch transaction history for the account
            txns = await transaction_history_for_account(
                connection, addy1, None, limit, max_limit
            )

            if len(txns) < max_limit and len(txns):
                lastsig = txns[-1]["signature"]
                transaction_got = await connection.get_transaction(lastsig)
                st.write(lastsig)
                st.json(transaction_got, expanded=False)
                if "result" not in transaction_got:
                    st.write(transaction_got)
                else:
                    blocktime = transaction_got["result"]["blockTime"]
                    source_addy = transaction_got["result"]["transaction"]["message"][
                        "accountKeys"
                    ][0]

                    # Create a DataFrame with the new data and append it to the existing data
                    dat = pd.DataFrame(
                        {
                            "blockTime": [blocktime],
                            "sourceAddress": [source_addy],
                            "sourceTransactionSignature": [lastsig],
                        },
                        index=[addy1],
                    )
                    existing_data = existing_data.append(dat)

                # Write the updated data to the CSV file
                existing_data.to_csv(filename)

            else:
                st.error("too many txns")

    setting2, setting1 = st.columns(2)
    ss = setting1.radio("y scale:", ["log", "linear"], horizontal=True)
    ss2 = setting2.radio("data:", ["by day", "by hour"], horizontal=True)

    field = ["total_fee_paid"]

    dd = df2.set_index("min date")[field]
    if field == ["minutes_ago"]:
        dd = dd.pipe(np.sign)
    dd.columns = ["new user creations"]
    dd = dd[dd.columns[0]]

    if ss2 == "by day":
        dd2 = dd.resample("1D").sum()
        st.plotly_chart(dd2.plot(log_y="log" in ss))
    elif ss2 == "by hour":
        dd.index = dd.index.hour
        dd = dd.reset_index().groupby("date").sum()
        dd = dd.sort_index()
        st.plotly_chart(dd.plot(log_y="log" in ss))
    else:
        st.plotly_chart(dd.plot(log_y="log" in ss))


def render_referrals_tab(df):
    st.write("referral leaderboard")
    oo = df.groupby("referrer")
    tt = oo.count().iloc[:, 0:1]
    tt2 = oo.authority.agg(list)
    tt2.columns = ["referees"]
    tt.columns = ["number referred"]
    val = df.loc[
        df.authority.isin(tt.index),
        ["authority", "total_referrer_reward", "current_epoch_referrer_reward"],
    ]
    val = val.set_index("authority")
    tt = pd.concat([tt, val, tt2], axis=1)
    tt = tt.loc[[x for x in tt.index if x != "11111111111111111111111111111111"]]
    tt = tt.sort_values("number referred", ascending=False)
    st.dataframe(tt)


def render_fillers_tab(df):
    df2 = df[df.filler_volume30d > 0]
    df2 = df2.set_index("authority")[
        ["filler_volume30d_calc", "last_fill_seconds_ago"]
    ].sort_values(by="filler_volume30d_calc", ascending=False)
    col1, col2, col3, col4 = st.columns(4)
    col1.metric(
        "unique fillers (last 10 minutes)",
        int(len(df2[df2.last_fill_seconds_ago < 60 * 10])),
    )
    col2.metric(
        "unique fillers (last 24 hours)",
        int(len(df2[df2.last_fill_seconds_ago < 60 * 60 * 24])),
    )

    col3.metric(
        "unique fillers (last 30 days)",
        int(df2["filler_volume30d_calc"].pipe(np.sign).sum()),
    )
    col4.metric(
        "unique fillers (since inception)",
        int(len(df2)),
    )

    st.write("filler leaderboard")
    st.dataframe(df2)


def render_clusters_tab(df):
    st.write(df["total_fee_paid"].sum(), df["total_fee_rebate"].sum())
    dff = df[df["total_30d_volume_calc"] > 0].sort_values("total_30d_volume_calc")
    dd = dff[
        [
            "total_30d_volume_calc",
            "taker_volume30d_calc",
            "if_staked_quote_asset_amount",
        ]
    ].pipe(np.sqrt)
    fig = dd.plot(kind="scatter")

    st.write(df.corr())
    st.plotly_chart(fig)

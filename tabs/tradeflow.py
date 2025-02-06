import datetime

import pandas as pd

pd.options.plotting.backend = "plotly"
import numpy as np
import pytz
import streamlit as st

from constants import ALL_MARKET_NAMES
from datafetch.api_fetch import get_trades_for_range_pandas
from datafetch.s3_fetch import load_s3_trades_data


def dedupdf(all_markets, market_name, lookahead=60):
    df1 = all_markets.copy()
    df1["date"] = pd.to_datetime(df1["ts"])
    if "maker" not in df1.columns:
        df1["maker"] = np.nan
    df1 = df1.drop_duplicates(
        [
            "fillerReward",
            "baseAssetAmountFilled",
            "quoteAssetAmountFilled",
            "quoteAssetAmountSurplus",
            "takerOrderBaseAssetAmount",
            "takerOrderCumulativeBaseAssetAmountFilled",
            "takerOrderCumulativeQuoteAssetAmountFilled",
            "makerOrderBaseAssetAmount",
            "makerOrderCumulativeBaseAssetAmountFilled",
            "makerOrderCumulativeQuoteAssetAmountFilled",
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
            "date",
        ]
    ).reset_index(drop=True)
    oracle_series = pd.to_numeric(df1.groupby("ts")["oraclePrice"].last())

    df1["quoteAssetAmountFilled"] = pd.to_numeric(df1["quoteAssetAmountFilled"])
    df1["baseAssetAmountFilled"] = pd.to_numeric(df1["baseAssetAmountFilled"])
    df1["oraclePrice"] = pd.to_numeric(df1["oraclePrice"])
    df1["markPrice"] = df1["quoteAssetAmountFilled"] / df1["baseAssetAmountFilled"]
    df1["buyPrice"] = np.nan
    df1["sellPrice"] = np.nan
    df1["buyPrice"] = df1.loc[
        df1[df1["takerOrderDirection"] == "long"].index, "markPrice"
    ]
    df1["sellPrice"] = df1.loc[df1["takerOrderDirection"] == "short", "markPrice"]

    df1["takerPremium"] = (
        pd.to_numeric(df1["markPrice"]) - pd.to_numeric(df1["oraclePrice"])
    ) * (2 * (df1["takerOrderDirection"] == "long") - 1)
    df1["takerPremiumDollar"] = pd.to_numeric(df1["takerPremium"]) * pd.to_numeric(
        df1["baseAssetAmountFilled"]
    )
    df1["takerPremiumNextMinute"] = (
        df1["markPrice"]
        - df1["ts"].apply(
            lambda x: oracle_series.loc[x + pd.Timedelta(seconds=lookahead) :]
            .head(1)
            .max()
        )
    ) * (2 * (df1["takerOrderDirection"] == "long") - 1)
    df1["takerPremiumNextMinuteDollar"] = (
        df1["takerPremiumNextMinute"] * df1["baseAssetAmountFilled"]
    )

    return df1


@st.cache_data
def convert_df(df):
    # IMPORTANT: Cache the conversion to prevent computation on every rerun
    return df.to_csv().encode("utf-8")


def trade_flow_analysis():
    tzInfo = pytz.timezone("UTC")

    modecol, col1, col2 = st.columns([2, 3, 3])
    selection = modecol.radio("mode:", ["summary", "per-market"], index=1)
    data_source = modecol.radio("data source:", ["api", "s3"], index=0)

    markets = ALL_MARKET_NAMES
    market = None
    if selection == "per-market":
        market = col1.selectbox("select market:", markets)
        market_selected = [market]
    else:
        market_selected = markets
    date = col2.date_input(
        "select date:",
        min_value=datetime.datetime(2022, 11, 4),
        max_value=(datetime.datetime.now(tzInfo)),
        help="UTC timezone",
    )
    if data_source == "s3":
        markets_data = load_s3_trades_data(market_selected, date, date)
        markets_data[market]["ts"] = pd.to_datetime(markets_data[market]["ts"] * 1e9)
    else:
        markets_data = {}
        for market in market_selected:
            data = get_trades_for_range_pandas(market, date, date)
            markets_data[market] = data

    # Display each market's data in separate expanders
    for market, data in markets_data.items():
        with st.expander(f"Data for {market}"):
            st.dataframe(data)

    solect1, solect2, solect3 = st.columns(3)
    lookahead = solect1.slider(
        "markout lookahead:",
        0,
        3600,
        60,
        10,
        help="second lookhead to check if trader predicted price action",
    )
    user_type = solect2.radio(
        "user type:",
        ["taker", "maker", "vAMM"],
        0,
        help="examine this user type (vAMM means only examine vAMM flow)",
    )
    botcutoff = solect3.number_input(
        "retail order count cutoff:",
        0,
        value=3000,
        help="users have more than this many orders are classified as 'bot' instead of retail",
    )
    # solperp = pd.concat(
    #     [dedupdf(markets_data, nom, lookahead) for nom in market_selected]
    # )
    solperp = pd.concat(
        [dedupdf(markets_data[nom], nom, lookahead) for nom in market_selected]
    )

    def show_analysis_tables(nom, user_accounts, user_type):
        def make_clickable(link):
            # target _blank to open new window
            # extract clickable text to display for your link
            text = link.split("=")[1]
            ts = text[:4] + "..." + text[-4:]
            return f'<a target="_blank" href="{link}">{ts}</a>'

        st.write(nom, len(user_accounts))
        with st.expander("users"):
            tt = pd.DataFrame(
                solperp[solperp[user_type].isin(user_accounts)]
                .groupby(user_type)[
                    [
                        "takerFee",
                        "quoteAssetAmountFilled",
                        "takerPremium",
                        "takerPremiumNextMinute",
                        "takerPremiumDollar",
                    ]
                ]
                .agg(
                    {
                        "takerFee": "count",
                        "quoteAssetAmountFilled": "sum",
                        "takerPremium": "sum",
                        "takerPremiumNextMinute": "sum",
                        "takerPremiumDollar": "sum",
                    }
                )
                .sort_values(by="quoteAssetAmountFilled", ascending=False)
            )
            tt.index = tt.index
            tt.columns = [
                "count",
                "volume",
                "takerPremiumTotal",
                "takerPremiumNextMinuteTotal",
                "takerUSDPremiumSum",
            ]
            tt["Markout"] = -(
                tt["takerPremiumNextMinuteTotal"] - tt["takerPremiumTotal"]
            )
            tt["userAccount"] = [
                "https://app.drift.trade?userAccount=" + str(x) for x in tt.index
            ]
            tt["userAccount"] = tt["userAccount"].apply(make_clickable)
            zol1, zol2 = st.columns([1, 4])
            zol2.dataframe(
                tt.drop(["userAccount"], axis=1),
                height=len(tt) * 60,
                use_container_width=True,
            )
            zol1.write(
                tt[["userAccount", "volume"]]
                .reset_index(drop=True)
                .to_html(index=False, escape=False),
                unsafe_allow_html=True,
            )

        df1 = (
            solperp[solperp[user_type].isin(user_accounts)]
            .groupby("takerOrderDirection")
            .agg(
                {
                    "takerOrderDirection": "count",
                    "quoteAssetAmountFilled": "sum",
                    "takerPremium": "mean",
                    "takerPremiumNextMinute": "mean",
                    "takerPremiumDollar": "sum",
                    "takerFee": lambda x: pd.to_numeric(x).sum(),
                    "makerFee": lambda x: pd.to_numeric(x).sum(),
                }
            )
        )
        df1.columns = [
            "count",
            "volume",
            "takerPricePremiumMean",
            "takerPricePremiumNextMinuteMean",
            "takerUSDPremiumSum",
            "takerFeeSum",
            "makerFeeSum",
        ]
        df1["Markout"] = -(
            df1["takerPricePremiumNextMinuteMean"] - df1["takerPricePremiumMean"]
        )

        df2 = (
            solperp[solperp[user_type].isin(user_accounts)]
            .groupby("actionExplanation")
            .agg(
                {
                    "fillerReward": "count",
                    "quoteAssetAmountFilled": "sum",
                    "takerPremium": "mean",
                    "takerPremiumNextMinute": "mean",
                    "takerPremiumDollar": "sum",
                    "takerFee": lambda x: pd.to_numeric(x).sum(),
                    "makerFee": lambda x: pd.to_numeric(x).sum(),
                }
            )
        )
        df2.columns = [
            "count",
            "volume",
            "takerPricePremiumMean",
            "takerPricePremiumNextMinuteMean",
            "takerUSDPremiumSum",
            "takerFeeSum",
            "makerFeeSum",
        ]
        df2["Markout"] = -(
            df2["takerPricePremiumNextMinuteMean"] - df2["takerPricePremiumMean"]
        )

        st.dataframe(df1)
        st.dataframe(df2)

    if user_type != "vAMM":
        zol1, zol2, zol3 = st.columns(3)

        takerfee = pd.to_numeric(solperp["takerFee"]).sum()
        makerfee = pd.to_numeric(solperp["makerFee"]).sum()
        fillerfee = pd.to_numeric(solperp["fillerReward"]).sum()

        showprem = zol3.radio("show premiums in plot", [True, False], 1)
        showmarkoutfees = zol3.radio(
            "show spread vs oracle w/ markout", [True, False], 1
        )
        liq_only = zol3.radio("liquidations only", [True, False], 1)

        pol1, pol2 = st.columns(2)
        if liq_only:
            solperp = solperp[solperp['actionExplanation'].str.contains('liquidation', case=False, na=False)]
        
        trade_df = solperp.set_index("date").sort_index()[
            ["buyPrice", "oraclePrice", "sellPrice"]
        ]
        if showprem:
            trade_df["buyPremium"] = (
                trade_df["buyPrice"] - trade_df["oraclePrice"]
            ).ffill()
            trade_df["sellPremium"] = (
                trade_df["oraclePrice"] - trade_df["sellPrice"]
            ).ffill()
        fig = trade_df.plot().update_layout(
            title="fills vs oracle", xaxis_title="date", yaxis_title="price"
        )

        pol1.plotly_chart(fig)

        retail_takers = (
            solperp[
                pd.to_numeric(solperp[user_type + "OrderId"], errors="coerce")
                < botcutoff
            ][user_type]
            .dropna()
            .unique()
        )
        bot_takers = (
            solperp[
                pd.to_numeric(solperp[user_type + "OrderId"], errors="coerce")
                >= botcutoff
            ][user_type]
            .dropna()
            .unique()
        )

        if showmarkoutfees:
            solperp["markout_pnl_1MIN"] = -(
                solperp["baseAssetAmountFilled"] * (solperp["takerPremiumNextMinute"])
            )
        else:
            solperp["markout_pnl_1MIN"] = -(
                solperp["baseAssetAmountFilled"]
                * (solperp["takerPremiumNextMinute"] - solperp["takerPremium"])
            )

        retail_trades = solperp[solperp[user_type].isin(retail_takers)]
        retail_volume = retail_trades["quoteAssetAmountFilled"].sum().round(2)
        total_volume = solperp["quoteAssetAmountFilled"].sum().round(2)
        num_trades = len(solperp)
        unique_user_types = len(solperp[user_type].unique())

        tt = market if market is not None else ""
        zol1.metric(
            f"{tt} Retail Volume:",
            f"${retail_volume:,}",
            f"${total_volume:,} over {num_trades} trades among {unique_user_types} unique {user_type}s",
        )
        zol2.metric(
            f"{tt} Fees:",
            f"${takerfee:,.2f}",
            f"${-makerfee:,.2f} in liq/maker rebates, ${fillerfee:,.2f} in filler rewards",
        )

        if liq_only:
            takerfee_liq_only = pd.to_numeric(solperp["takerFee"]).sum()
            makerfee_liq_only = pd.to_numeric(solperp["makerFee"]).sum()
            fillerfee_liq_only = pd.to_numeric(solperp["fillerReward"]).sum()
            zol2.metric(
                f"{tt} Liquidations Only Taker Fees:",
                f"${takerfee_liq_only:,.2f}",
            )
            zol2.metric(
                f"{tt} Liquidations Only Maker Fees:",
                f"${makerfee_liq_only:,.2f}",
            )
            zol2.metric(
                f"{tt} Liquidations Only Filler Rewards:",
                f"${fillerfee_liq_only:,.2f}",
            )



        st.code(
            f"Note: bot classfication is for users who have placed {botcutoff}+ orders in a single market"
        )

        markoutdf = (
            pd.concat(
                {
                    "retailMarkout1MIN:": retail_trades.set_index("date")
                    .sort_index()["markout_pnl_1MIN"]
                    .resample("min")
                    .last(),
                    "botMarkout1MIN": solperp[solperp[user_type].isin(bot_takers)]
                    .set_index("date")
                    .sort_index()["markout_pnl_1MIN"]
                    .resample("min")
                    .last(),
                },
                axis=1,
            )
            .cumsum()
            .ffill()
        )
        fig2 = markoutdf.plot().update_layout(
            title="markout", xaxis_title="date", yaxis_title="PnL (USD)"
        )
        pol2.plotly_chart(fig2)

        show_analysis_tables("retail " + user_type + "s:", retail_takers, user_type)
        show_analysis_tables("bot " + user_type + "s:", bot_takers, user_type)
    else:
        vamm_maker_trades = solperp[solperp["maker"].isnull()]
        s1, s2, s3 = st.columns(3)
        unit = s1.radio("unit:", ["$", "Price"], horizontal=True)
        field = s2.radio(
            "fields", ["execOnly", "timeOnly", "all"], index=2, horizontal=True
        )
        bdown = s3.radio("breakdown:", ["by action", "all"], index=1, horizontal=True)

        if unit == "$" and field == "all":
            vamm_maker_trades["ff"] = vamm_maker_trades["takerPremiumNextMinuteDollar"]
        elif unit == "$" and field == "timeOnly":
            vamm_maker_trades["ff"] = (
                vamm_maker_trades["takerPremiumNextMinuteDollar"]
                - vamm_maker_trades["takerPremiumDollar"]
            )
        elif unit == "$" and field == "execOnly":
            vamm_maker_trades["ff"] = vamm_maker_trades["takerPremiumDollar"]
        elif unit == "Price" and field == "all":
            vamm_maker_trades["ff"] = vamm_maker_trades["takerPremiumNextMinute"]
        elif unit == "Price" and field == "timeOnly":
            vamm_maker_trades["ff"] = (
                vamm_maker_trades["takerPremiumNextMinute"]
                - vamm_maker_trades["takerPremium"]
            )
        elif unit == "Price" and field == "execOnly":
            vamm_maker_trades["ff"] = vamm_maker_trades["takerPremium"]

        st.metric(
            "vAMM volume:",
            "$" + f"{vamm_maker_trades['quoteAssetAmountFilled'].sum().round(2):,}",
            f"{vamm_maker_trades['quoteAssetAmountFilled'].sum().round(2) / solperp['quoteAssetAmountFilled'].sum().round(2) * 100:,.2f}% of maker volume",
        )
        tabs = st.tabs(["plot", "table"])

        vamm_maker_trades = vamm_maker_trades.set_index("ts")

        with tabs[1]:
            st.write(vamm_maker_trades)
        with tabs[0]:
            fig = None
            vamm_maker_trades.index = pd.to_datetime(
                (vamm_maker_trades.index.astype(str).astype(float) * 1e9).astype(int)
            )
            if bdown == "by action":
                vamm_maker_trades = vamm_maker_trades.pivot_table(
                    index="ts", columns="actionExplanation", values="ff", aggfunc="sum"
                )
                # st.write(vamm_maker_trades)
                # vamm_maker_trades['Markout'] = -(vamm_maker_trades['takerPremium'] - vamm_maker_trades['takerPremiumNextMinute'])
                fig = (-vamm_maker_trades.fillna(0).cumsum()).plot()
            else:
                fig = (-vamm_maker_trades["ff"].cumsum()).plot()

            fig.update_layout(
                title="Taker Markouts vs vAMM",
                xaxis_title="ts",
                yaxis_title="Markout " + "(" + unit + ")",
            )

            st.plotly_chart(fig, use_container_width=True)

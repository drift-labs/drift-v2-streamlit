import asyncio
import datetime
import os

import pandas as pd
import streamlit as st
from anchorpy import Provider, Wallet
from dotenv import load_dotenv
from driftpy.constants.config import configs
from driftpy.drift_client import AccountSubscriptionConfig, DriftClient
from solana.rpc.async_api import AsyncClient
from solders.keypair import Keypair

from tabs.acred_stats import show_acred_stats
from tabs.amplify_stats import show_amplify_stats
from tabs.api import show_api
from tabs.competition import competitions
from tabs.counterparty_analysis import counterparty_analysis_page
from tabs.dlp import dlp
from tabs.drift_buyback import drift_buyback_dashboard
from tabs.fee_income import fee_income_page
from tabs.fees import fee_page
from tabs.fill_quality import fill_quality_analysis
from tabs.funding_history import funding_history
from tabs.gpt import gpt_page
from tabs.if_stakers import insurance_fund_page
from tabs.imf import imf_page
from tabs.liqcalc import liqcalc
from tabs.liquidity import mm_page
from tabs.logs import log_page
from tabs.maker_tx_landing_analysis import maker_tx_landing_analysis
from tabs.mm_program import mm_program_page
from tabs.openbookv2 import tab_openbookv2
from tabs.orders import orders_page
from tabs.overview_markets import show_overview_markets
from tabs.perpLP import perp_lp_page
from tabs.pid import show_pid_positions
from tabs.post_trade_analysis import post_trade_analysis
from tabs.refs import ref_page
from tabs.simulations import sim_page
from tabs.superstake import super_stake
from tabs.tradeflow import trade_flow_analysis
from tabs.trigger_speed import trigger_speed_analysis
from tabs.useractivity import show_user_activity
from tabs.userdataraw import userdataraw
from tabs.userhealth import show_user_health
from tabs.usermap import usermap_page
from tabs.userperf import show_user_perf
from tabs.users_in_market import users_in_market_page
from tabs.userstats import show_user_stats
from tabs.userstatus import userstatus_page
from tabs.uservolume import show_user_volume
from tabs.vamm import vamm
from tabs.vamm_stats import vamm_stats_page
from tabs.vaults import vaults

# import ssl
# import urllib.request
# ssl._create_default_https_context = ssl._create_unverified_context


load_dotenv()

try:
    from tabs.backtester import backtester_page
except ImportError:
    print("backtester not found")
    pass


def main():
    current_time = datetime.datetime.now().strftime("%Y/%m/%d %H:%M:%S")
    if st.sidebar.button("Clear Cache"):
        st.cache_data.clear()

    env = st.sidebar.radio("env", ("mainnet-beta", "devnet"))

    url = "🤫"
    if env == "devnet":
        url = "https://api." + env + ".solana.com"
    else:
        url = "🤫"

    rpc = st.sidebar.text_input("rpc", url)
    if env == "mainnet-beta" and (rpc == "🤫" or rpc == ""):
        rpc = os.environ["ANCHOR_PROVIDER_URL"]

    def query_string_callback():
        st.query_params.from_dict({"tab": st.session_state.query_key})

    query_p = st.query_params
    query_tab = query_p.get("tab", "Welcome")

    tab_options = (
        "Welcome",
        "DRIFT Smart Acquisition",
        "Overview-Markets",
        "Overview-Users",
        "Simulations",
        "Logs",
        "Fee-Income",
        "User-Performance",
        "Counterparty Analysis",
        "Fee-Schedule",
        "Insurance-Fund",
        "Perp-LPs",
        "User-Stats",
        "DLOB",
        "Refs",
        "MM",
        "Trade Flow",
        "Post Trade Analysis",
        "Maker Tx Landing Analysis",
        "Drift-GPT",
        "IMF",
        "Network",
        "User-Status",
        "Config",
        "API",
        "SuperStake",
        "vAMM",
        "vAMM Stats",
        "DLP",
        "FundingHistory",
        "UserDataRaw",
        "Vaults",
        "DriftDraw",
        "Openbookv2",
        "UserMap",
        "UsersInMarket",
        "Backtester",
        "MM (legacy)",
        "Liquidation Calculator",
        "Amplify Stats",
        "ACRED Stats",
        "Fill Quality",
        "Trigger Speed",
    )

    to_remove = [
        "User-Activity",
        "User-Performance",
        "User-Health",
        "User-Volume",
    ]

    query_index = 0
    for idx, x in enumerate(tab_options):
        if x.lower() == query_tab.lower():
            query_index = idx

    tab = st.sidebar.radio(
        "Select Tab:",
        tab_options,
        query_index,
        on_change=query_string_callback,
        key="query_key",
    )

    if env == "mainnet-beta":
        config = configs["mainnet"]
    else:
        config = configs[env]

    kp = Keypair()  # random wallet
    wallet = Wallet(kp)
    connection = AsyncClient(rpc)
    provider = Provider(connection, wallet)
    clearing_house: DriftClient = DriftClient(
        provider.connection,
        provider.wallet,
        env.split("-")[0],
        account_subscription=AccountSubscriptionConfig("cached"),
    )
    # st.write(clearing_house.__dict__.keys())
    clearing_house.time = current_time

    st.title(f"Drift v2: {tab}")

    if tab.lower() == "welcome":
        st.warning(
            "DISCLAIMER: INFORMATIONAL PURPOSES ONLY. USE EXPERIMENTAL SOFTWARE AT YOUR OWN RISK."
        )
        st.markdown(
            "## welcome to the [Drift v2](app.drift.trade) streamlit dashboard!"
        )
        st.metric(
            "protocol has been live for:",
            str(
                int(
                    (
                        datetime.datetime.now() - pd.to_datetime("2022-11-05")
                    ).total_seconds()
                    / (60 * 60 * 24)
                )
            )
            + " days",
        )
        st.markdown(
            """
        On this site, did you know you can...
        - explore all perp/spot market stats in [Overview](/?tab=Overview).
        - track the Insurance Fund and balances in [Insurance Fund](/?tab=Insurance-Fund).
        - inspect historical price impact and market maker leaderboards in [MM](/?tab=MM).
        - view a taker trader breakdown in [Trade Flow](/?tab=Trade-Flow).
        - talk to an AI at [Drift-GPT](/?tab=Drift-GPT).


        this entire website is [open source](https://github.com/0xbigz/drift-v2-streamlit) and you can run it locally:
        ```
        git clone https://github.com/0xbigz/drift-v2-streamlit.git
        cd drift-v2-streamlit/

        python3.10 -m venv venv
        source venv/bin/activate
        pip install -r requirements.txt

        streamlit run app.py
        ```

        (this app uses streamlit==1.27, see `requirements.txt`)
        """
        )

    elif tab.lower() == "drift smart acquisition":
        loop = asyncio.new_event_loop()
        loop.run_until_complete(drift_buyback_dashboard(clearing_house))

    elif tab.lower() == "overview-markets":
        loop = asyncio.new_event_loop()
        loop.run_until_complete(show_overview_markets(clearing_house))

    elif tab.lower() == "overview-users":
        loop = asyncio.new_event_loop()
        loop.run_until_complete(show_pid_positions(clearing_house))

    elif tab.lower() == "config":
        with st.expander(f"pid={clearing_house.program_id} config"):
            st.json(config.__dict__)

        if os.path.exists("gpt_database.csv"):
            gpt_database = pd.read_csv("gpt_database.csv").to_csv().encode("utf-8")
            st.download_button(
                label="share browser statistics [bytes=" + str(len(gpt_database)) + "]",
                data=gpt_database,
                file_name="gpt_database.csv",
                mime="text/csv",
            )

    elif tab.lower() == "logs":
        loop = asyncio.new_event_loop()
        loop.run_until_complete(log_page(rpc, clearing_house))

    elif tab.lower() == "simulations":
        sim_page()

    elif tab.lower() == "fee-schedule":
        loop = asyncio.new_event_loop()
        loop.run_until_complete(fee_page(clearing_house))

    elif tab.lower() == "fee-income":
        loop = asyncio.new_event_loop()
        loop.run_until_complete(fee_income_page(clearing_house))

    elif tab.lower() == "user-performance":
        loop = asyncio.new_event_loop()
        loop.run_until_complete(show_user_perf(clearing_house))

    elif tab.lower() == "counterparty analysis":
        loop = asyncio.new_event_loop()
        loop.run_until_complete(counterparty_analysis_page(clearing_house))

    elif tab.lower() == "refs":
        loop = asyncio.new_event_loop()
        loop.run_until_complete(ref_page(clearing_house))

    elif tab.lower() == "insurance-fund":
        loop = asyncio.new_event_loop()
        loop.run_until_complete(insurance_fund_page(clearing_house, env))

    elif tab.lower() == "user-status":
        loop = asyncio.new_event_loop()
        loop.run_until_complete(userstatus_page(clearing_house))
    elif tab.lower() == "perp-lps":
        loop = asyncio.new_event_loop()
        loop.run_until_complete(perp_lp_page(clearing_house, env))
    elif tab.lower() == "user-stats":
        loop = asyncio.new_event_loop()
        loop.run_until_complete(show_user_stats(clearing_house))

    elif tab.lower() == "api":
        loop = asyncio.new_event_loop()
        loop.run_until_complete(show_api(clearing_house))

    elif tab.lower() == "dlob":
        orders_page(clearing_house)

    elif tab.lower() == "user-volume":
        loop = asyncio.new_event_loop()
        loop.run_until_complete(show_user_volume(clearing_house))
    elif tab.lower() == "vamm":
        loop = asyncio.new_event_loop()
        loop.run_until_complete(vamm(clearing_house))
    elif tab.lower() == "vamm stats":
        vamm_stats_page()
    elif tab.lower() == "dlp":
        loop = asyncio.new_event_loop()
        loop.run_until_complete(dlp(clearing_house))
    elif tab.lower() == "mm":
        loop = asyncio.new_event_loop()
        loop.run_until_complete(mm_program_page(clearing_house, env))
    elif tab.lower() == "mm (legacy)":
        loop = asyncio.new_event_loop()
        loop.run_until_complete(mm_page(clearing_house))
    elif tab.lower() == "trade flow":
        loop = asyncio.new_event_loop()
        loop.run_until_complete(trade_flow_analysis(clearing_house))
    elif tab.lower() == "post trade analysis":
        loop = asyncio.new_event_loop()
        loop.run_until_complete(post_trade_analysis(clearing_house))
    elif tab.lower() == "maker tx landing analysis":
        loop = asyncio.new_event_loop()
        loop.run_until_complete(maker_tx_landing_analysis(clearing_house))
    elif tab.lower() == "imf":
        loop = asyncio.new_event_loop()
        loop.run_until_complete(imf_page(clearing_house))
    elif tab.lower() == "social":
        repo = "https://github.com/drift-labs/protocol-v2"
        st.markdown(
            "["
            + repo
            + "]("
            + repo
            + ") | [@driftprotocol](https://twitter.com/@driftprotocol)"
        )

        tweets = {
            "cindy": "https://twitter.com/cindyleowtt/status/1569713537454579712",
            "0xNineteen": "https://twitter.com/0xNineteen/status/1571926865681711104",
        }
        st.header("twitter:")
        st.table(pd.Series(tweets))

    # elif tab.lower() == "platyperps":
    #     loop = asyncio.new_event_loop()
    #     loop.run_until_complete(show_platyperps(clearing_house))

    elif tab.lower() == "drift-gpt":
        loop = asyncio.new_event_loop()
        loop.run_until_complete(gpt_page(clearing_house))

    elif tab.lower() == "superstake":
        loop = asyncio.new_event_loop()
        loop.run_until_complete(super_stake(clearing_house))
    elif tab.lower() == "userdataraw":
        loop = asyncio.new_event_loop()
        loop.run_until_complete(userdataraw(clearing_house))
    elif tab.lower() == "liquidation calculator":
        loop = asyncio.new_event_loop()
        loop.run_until_complete(liqcalc(clearing_house))
    elif tab.lower() == "fundinghistory":
        loop = asyncio.new_event_loop()
        loop.run_until_complete(funding_history(clearing_house, env))
    elif tab.lower() == "vaults":
        loop = asyncio.new_event_loop()
        loop.run_until_complete(vaults(clearing_house, env))
    elif tab.lower() == "driftdraw":
        loop = asyncio.new_event_loop()
        loop.run_until_complete(competitions(clearing_house, env))
    elif tab.lower() == "openbookv2":
        loop = asyncio.new_event_loop()
        loop.run_until_complete(tab_openbookv2(clearing_house, env))
    elif tab.lower() == "usermap":
        # loop = asyncio.new_event_loop()
        usermap_page(clearing_house, env)
    elif tab.lower() == "usersinmarket":
        loop = asyncio.new_event_loop()
        loop.run_until_complete(users_in_market_page(clearing_house, env))
        # users_in_market_page
    elif tab.lower() == "backtester":
        try:
            backtester_page(clearing_house, env)
        except ImportError:
            print("backtester not found")
            pass
    elif tab.lower() == "amplify stats":
        loop = asyncio.new_event_loop()
        loop.run_until_complete(show_amplify_stats(clearing_house))
    elif tab.lower() == "acred stats":
        loop = asyncio.new_event_loop()
        loop.run_until_complete(show_acred_stats(clearing_house))
    elif tab.lower() == "fill quality":
        loop = asyncio.new_event_loop()
        loop.run_until_complete(fill_quality_analysis(clearing_house))
    elif tab.lower() == "trigger speed":
        trigger_speed_analysis()
    elif tab.lower() == "user-activity":
        loop = asyncio.new_event_loop()
        loop.run_until_complete(show_user_activity(clearing_house))
    hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
            """
    st.markdown(hide_streamlit_style, unsafe_allow_html=True)


if __name__ == "__main__":
    try:
        image_path = os.path.join(os.path.dirname(__file__), "images/drift-light.png")
        st.logo(image=image_path)
        st.set_page_config(
            page_title="Drift Analytics",
            layout="wide",
            page_icon="📈",
        )
    except Exception as e:
        print(e)
        pass

    main()

import pandas as pd
import plotly.express as px
import streamlit as st
from driftpy.constants.numeric_constants import (
    FUNDING_RATE_BUFFER,
    PERCENTAGE_PRECISION,
)
from driftpy.drift_client import DriftClient

from datafetch.transaction_fetch import load_token_balance
from helpers import serialize_perp_market_2, serialize_spot_market

pd.options.plotting.backend = "plotly"


async def show_overview_markets(clearing_house: DriftClient):
    ch = clearing_house
    await ch.account_subscriber.update_cache()
    state = ch.get_state_account()

    with st.expander("state"):
        st.json(state.__dict__)

    num_perp_markets = state.number_of_markets
    num_spot_markets = state.number_of_spot_markets

    cat_tabs = st.tabs(
        ["overview", "perp (%i)" % num_perp_markets, "spot (%i)" % num_spot_markets]
    )
    usdc_market = ch.get_spot_market_account(0)

    with cat_tabs[0]:
        po_deposits = 0
        po_position_notional = 0
        po_position_notional_net = 0
        est_funding_dol = 0
        dd1 = {}
        dd2 = {}
        for market_index in range(state.number_of_markets):
            perp_i = ch.get_perp_market_account(market_index)
            market_name = "".join(map(chr, perp_i.name)).strip(" ")

            fee_pool = (
                perp_i.amm.fee_pool.scaled_balance
                * usdc_market.cumulative_deposit_interest
                / 1e10
            ) / (1e9)
            pnl_pool = (
                perp_i.pnl_pool.scaled_balance
                * usdc_market.cumulative_deposit_interest
                / 1e10
            ) / (1e9)
            po_deposits += fee_pool + pnl_pool

            otwap = perp_i.amm.historical_oracle_data.last_oracle_price_twap
            market_price_spread = perp_i.amm.last_mark_price_twap - otwap
            funding_offset = otwap / 5000  # ~ 7% annual premium
            pred_fund = ((market_price_spread + funding_offset) / otwap) / (
                3600.0 / perp_i.amm.funding_period * 24
            )

            fundings = [
                x * 100 * 365.25 * 24
                for x in (
                    pred_fund,
                    perp_i.amm.last_funding_rate / otwap / FUNDING_RATE_BUFFER,
                    perp_i.amm.last24h_avg_funding_rate / otwap / FUNDING_RATE_BUFFER,
                )
            ]

            dd1[market_name] = fundings

            po_position_notional_i = (
                (
                    perp_i.amm.base_asset_amount_with_amm
                    + perp_i.amm.base_asset_amount_with_unsettled_lp
                )
                / 1e9
                * otwap
                / 1e6
            )
            po_position_notional += abs(po_position_notional_i)
            po_position_notional_net += po_position_notional_i
            est_funding_dol += pred_fund * po_position_notional_i

        total_borrow_fee_annualized = 0
        for market_index in range(state.number_of_spot_markets):
            market = ch.get_spot_market_account(market_index)
            market_name = "".join(map(chr, market.name)).strip(" ")
            deposits = (
                market.deposit_balance
                * market.cumulative_deposit_interest
                / 1e10
                / (1e9)
            )
            borrows = (
                market.borrow_balance * market.cumulative_borrow_interest / 1e10 / (1e9)
            )

            utilization = borrows / (deposits + 1e-12) * 100
            opt_util = market.optimal_utilization / PERCENTAGE_PRECISION * 100
            opt_borrow = market.optimal_borrow_rate / PERCENTAGE_PRECISION
            max_borrow = market.max_borrow_rate / PERCENTAGE_PRECISION

            bor_ir_curve = [
                opt_borrow * (100 / opt_util) * x / 100
                if x <= opt_util
                else ((max_borrow - opt_borrow) * (100 / (100 - opt_util)))
                * (x - opt_util)
                / 100
                + opt_borrow
                for x in [utilization]
            ]

            borrow_fee_annualized = (
                borrows * bor_ir_curve[0] * market.insurance_fund.total_factor / 1e6
            )
            borrow_fee_notional_annualized = (
                borrow_fee_annualized
                * market.historical_oracle_data.last_oracle_price
                / 1e6
            )

            total_borrow_fee_annualized += borrow_fee_notional_annualized

            dep_ir_curve = [
                ir * utilization * (1 - market.insurance_fund.total_factor / 1e6) / 100
                for idx, ir in enumerate(bor_ir_curve)
            ]
            if market_index == 0:
                usdc_dep_rate = dep_ir_curve[0]
            dd2[market_name] = (
                dep_ir_curve[0] * 100,
                bor_ir_curve[0] * 100,
                utilization,
                borrow_fee_notional_annualized,
            )
        s1, s2 = st.columns(2)
        s1.write(
            pd.DataFrame(
                dd1,
                index=[
                    "predicted funding rate",
                    "last funding rate",
                    "24h avg funding rate",
                ],
            ).T
        )
        s2.write(
            pd.DataFrame(
                dd2,
                index=[
                    "deposit rate",
                    "borrow rate",
                    "utilization",
                    "annualized_borrow_revenue",
                ],
            ).T
        )

        with st.expander("All Perp Markets", expanded=True):
            markets = ch.get_perp_market_accounts()
            a = pd.concat(
                [pd.DataFrame(serialize_perp_market_2(x)).T for x in markets], axis=1
            )
            a.columns = [m.market_index for m in markets]
            st.write(a)

        with st.expander("All Spot Markets", expanded=True):
            markets = ch.get_spot_market_accounts()
            a = pd.concat(
                [pd.DataFrame(serialize_spot_market(x)).T for x in markets], axis=1
            )
            a.columns = [m.market_index for m in markets]
            st.write(a)

    with cat_tabs[1]:
        tabs = st.tabs([str(x) for x in range(num_perp_markets)])
        for market_index, tab in enumerate(tabs):
            market_index = int(market_index)
            with tab:
                market = ch.get_perp_market_account(market_index)
                market_name = "".join(map(chr, market.name)).strip(" ")

                with st.expander(
                    "Perp"
                    + " market market_index="
                    + str(market_index)
                    + " "
                    + market_name
                ):
                    mdf = serialize_perp_market_2(market).T
                    st.dataframe(mdf)

                st.text(
                    f"vAMM Liquidity (bids= {(market.amm.max_base_asset_reserve - market.amm.base_asset_reserve) / 1e9} | asks={(market.amm.base_asset_reserve - market.amm.min_base_asset_reserve) / 1e9})"
                )
                t0, t1, t2 = st.columns([1, 1, 5])
                dir = t0.selectbox(
                    "direction:", ["buy", "sell"], key="selectbox-" + str(market_index)
                )
                ba = t1.number_input(
                    "base amount:", value=1, key="numin-" + str(market_index)
                )
                bid_price = (
                    market.amm.bid_quote_asset_reserve
                    / market.amm.bid_base_asset_reserve
                    * market.amm.peg_multiplier
                    / 1e6
                )
                ask_price = (
                    market.amm.ask_quote_asset_reserve
                    / market.amm.ask_base_asset_reserve
                    * market.amm.peg_multiplier
                    / 1e6
                )

                def px_impact(dir, ba):
                    f = ba / (market.amm.base_asset_reserve / 1e9)
                    if dir == "buy":
                        pct_impact = (1 / ((1 - f) ** 2) - 1) * 100
                    else:
                        pct_impact = (1 - 1 / ((1 + f) ** 2)) * 100
                    return pct_impact

                px_impact = px_impact(dir, ba)
                price = (
                    (ask_price * (1 + px_impact / 100))
                    if dir == "buy"
                    else (bid_price * (1 - px_impact / 100))
                )
                t2.text(f"vAMM stats: \n px={price} \n px_impact={px_impact}%")

                rev_pool = (
                    usdc_market.revenue_pool.scaled_balance
                    * usdc_market.cumulative_deposit_interest
                    / 1e10
                    / (1e9)
                )

                fee_pool = (
                    market.amm.fee_pool.scaled_balance
                    * usdc_market.cumulative_deposit_interest
                    / 1e10
                ) / (1e9)
                pnl_pool = (
                    market.pnl_pool.scaled_balance
                    * usdc_market.cumulative_deposit_interest
                    / 1e10
                ) / (1e9)
                excess_pnl = (
                    fee_pool
                    + pnl_pool
                    - market.amm.quote_asset_amount / 1e6
                    + (
                        market.amm.base_asset_amount_with_amm
                        + market.amm.base_asset_amount_with_unsettled_lp
                    )
                    / 1e9
                    * market.amm.historical_oracle_data.last_oracle_price
                    / 1e6
                )

                dfff = pd.DataFrame(
                    {"pnl_pool": pnl_pool, "fee_pool": fee_pool, "rev_pool": rev_pool},
                    index=[0],
                ).T.reset_index()
                dfff["color"] = "balance"

                df_flow = pd.DataFrame(
                    {
                        "pnl_pool": market.amm.net_revenue_since_last_funding / 1e6,
                        "fee_pool": market.insurance_claim.revenue_withdraw_since_last_settle
                        / 1e6,
                        "rev_pool": -(
                            market.insurance_claim.revenue_withdraw_since_last_settle
                            / 1e6
                        ),
                    },
                    index=[0],
                ).T.reset_index()
                df_flow["color"] = "flows"

                df_flow_max = pd.DataFrame(
                    {
                        "pnl_pool": market.unrealized_pnl_max_imbalance / 1e6,
                        "fee_pool": market.insurance_claim.max_revenue_withdraw_per_period
                        / 1e6,
                        "rev_pool": -(
                            market.insurance_claim.max_revenue_withdraw_per_period / 1e6
                        ),
                    },
                    index=[0],
                ).T.reset_index()
                df_flow_max["color"] = "max_flow"
                dfff = pd.concat([dfff, df_flow, df_flow_max]).reset_index(drop=True)
                # print(dfff)
                fig = px.funnel(dfff, y="index", x=0, color="color")
                st.plotly_chart(fig, key=f"funnel-{market_index}")

                st.text(
                    f"Revenue Withdrawn Since Settle Ts: {market.insurance_claim.revenue_withdraw_since_last_settle / 1e6}"
                )
                st.text(f"""
                Last Settle Ts: {str(pd.to_datetime(market.insurance_claim.last_revenue_withdraw_ts * 1e9))} vs
                Last Spot Settle Ts: {str(pd.to_datetime(usdc_market.insurance_fund.last_revenue_settle_ts * 1e9))}

                """)

                st.text(
                    f"Ext. Insurance: {(market.insurance_claim.quote_max_insurance - market.insurance_claim.quote_settled_insurance) / 1e6} ({market.insurance_claim.quote_settled_insurance / 1e6}/{market.insurance_claim.quote_max_insurance / 1e6})"
                )
                st.text(f"Int. Insurance: {fee_pool}")
                st.text(f"PnL Pool: {pnl_pool}")
                st.text(
                    f"Excess PnL: {excess_pnl} ({fee_pool + pnl_pool} - {market.amm.quote_asset_amount / 1e6 + (market.amm.base_asset_amount_with_amm / 1e9 * market.amm.historical_oracle_data.last_oracle_price / 1e6)})"
                )

    with cat_tabs[2]:
        tabs = st.tabs([str(x) for x in range(num_spot_markets)])
        for market_index, tab in enumerate(tabs):
            market_index = int(market_index)
            with tab:
                market = ch.get_spot_market_account(market_index)
                market_name = "".join(map(chr, market.name)).strip(" ")
                (spot_col1,) = st.columns(1)
                with st.expander(
                    "Spot"
                    + " market market_index="
                    + str(market_index)
                    + " "
                    + market_name
                ):
                    mdf = serialize_spot_market(market).T
                    st.table(mdf)

                conn = ch.program.provider.connection
                ivault_pk = market.insurance_fund.vault
                svault_pk = market.vault
                iv_amount = await load_token_balance(conn, ivault_pk)
                sv_amount = await load_token_balance(conn, svault_pk)

                token_scale = 10**market.decimals
                spot_col1.metric(
                    f"{market_name} vault balance",
                    str(sv_amount / token_scale),
                    str(iv_amount / token_scale) + " (insurance)",
                )

                opt_util = market.optimal_utilization / PERCENTAGE_PRECISION * 100
                opt_borrow = market.optimal_borrow_rate / PERCENTAGE_PRECISION
                max_borrow = market.max_borrow_rate / PERCENTAGE_PRECISION

                ir_curve_index = [x / 100 for x in range(0, 100 * 100 + 1, 10)]
                bor_ir_curve = [
                    opt_borrow * (100 / opt_util) * x / 100
                    if x <= opt_util
                    else ((max_borrow - opt_borrow) * (100 / (100 - opt_util)))
                    * (x - opt_util)
                    / 100
                    + opt_borrow
                    for x in ir_curve_index
                ]

                dep_ir_curve = [
                    ir
                    * ir_curve_index[idx]
                    * (1 - market.insurance_fund.total_factor / 1e6)
                    / 100
                    for idx, ir in enumerate(bor_ir_curve)
                ]

                ir_fig = (
                    pd.DataFrame(
                        [dep_ir_curve, bor_ir_curve],
                        index=["deposit interest", "borrow interest"],
                        columns=ir_curve_index,
                    ).T
                    * 100
                ).plot()

                deposits = (
                    market.deposit_balance
                    * market.cumulative_deposit_interest
                    / 1e10
                    / (1e9)
                )
                borrows = (
                    market.borrow_balance
                    * market.cumulative_borrow_interest
                    / 1e10
                    / (1e9)
                )
                utilization = borrows / (deposits + 1e-12)

                ir_fig.add_vline(
                    x=market.utilization_twap / 1e6 * 100,
                    line_color="blue",
                    annotation_text="util_twap",
                )
                ir_fig.add_vline(
                    x=utilization * 100, line_color="green", annotation_text="util"
                )

                ir_fig.update_layout(
                    title=market_name + " Interest Rate",
                    xaxis_title="utilization (%)",
                    yaxis_title="interest rate (%)",
                    legend_title="Curves",
                )
                st.plotly_chart(ir_fig, key=f"ir_fig-{market_index}")

                st.markdown("## Simulate New Interest Rate Curve")

                new_input_cols = st.columns(3)
                with new_input_cols[0]:
                    new_opt_util = st.number_input("New Optimal Utilization",
                        value=float(opt_util),
                        min_value=0.0,
                        max_value=100.0,
                        key=f"opt_util_{market_index}")
                with new_input_cols[1]:
                    new_opt_borrow = st.number_input("New Optimal Borrow Rate",
                        value=float(opt_borrow),
                        min_value=0.0,
                        max_value=100.0,
                        key=f"opt_borrow_{market_index}")
                with new_input_cols[2]:
                    new_max_borrow = st.number_input("New Max Borrow Rate",
                        value=float(max_borrow),
                        min_value=0.0,
                        max_value=100.0,
                        key=f"max_borrow_{market_index}")

                # Display comparison
                st.markdown("### Current vs New Parameters")
                comp_cols = st.columns(3)
                with comp_cols[0]:
                    st.metric(f"Optimal Utilization", f"{opt_util:.2f}% -> {new_opt_util:.2f}%", f"{new_opt_util - opt_util:.2f}%")
                with comp_cols[1]:
                    st.metric(f"Optimal Borrow Rate", f"{opt_borrow:.2f}% -> {new_opt_borrow:.2f}%", f"{new_opt_borrow - opt_borrow:.2f}%")
                with comp_cols[2]:
                    st.metric(f"Max Borrow Rate", f"{max_borrow:.2f}% -> {new_max_borrow:.2f}%", f"{new_max_borrow - max_borrow:.2f}%")

                # Calculate new interest rate curves
                new_bor_ir_curve = [
                    new_opt_borrow * (100 / new_opt_util) * x / 100
                    if x <= new_opt_util
                    else ((new_max_borrow - new_opt_borrow) * (100 / (100 - new_opt_util)))
                    * (x - new_opt_util)
                    / 100
                    + new_opt_borrow
                    for x in ir_curve_index
                ]

                new_dep_ir_curve = [
                    ir
                    * ir_curve_index[idx]
                    * (1 - market.insurance_fund.total_factor / 1e6)
                    / 100
                    for idx, ir in enumerate(new_bor_ir_curve)
                ]

                # Create new figure with both current and new curves
                sim_ir_fig = (
                    pd.DataFrame(
                        [dep_ir_curve, bor_ir_curve, new_dep_ir_curve, new_bor_ir_curve],
                        index=["current deposit", "current borrow", "new deposit", "new borrow"],
                        columns=ir_curve_index,
                    ).T
                    * 100
                ).plot()

                sim_ir_fig.add_vline(
                    x=market.utilization_twap / 1e6 * 100,
                    line_color="blue",
                    annotation_text="util_twap",
                )
                sim_ir_fig.add_vline(
                    x=utilization * 100, line_color="green", annotation_text="util"
                )

                sim_ir_fig.update_layout(
                    title=market_name + " Interest Rate Simulation",
                    xaxis_title="utilization (%)",
                    yaxis_title="interest rate (%)",
                    legend_title="Curves",
                )
                st.plotly_chart(sim_ir_fig, key=f"sim_ir_fig-{market_index}")


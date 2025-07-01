import asyncio
import datetime
import json

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from anchorpy import EventParser
from anchorpy.provider import Signature
from driftpy.addresses import get_if_rebalance_config_public_key
from driftpy.constants.spot_markets import mainnet_spot_market_configs
from driftpy.drift_client import DriftClient
from solders.pubkey import Pubkey

from datafetch.transaction_fetch import transaction_history_for_account


def get_drift_market_index():
    for spot in mainnet_spot_market_configs:
        if "DRIFT" in spot.symbol.upper():
            return spot.market_index
    return None


async def drift_buyback_dashboard(ch: DriftClient):
    """Dashboard for tracking DRIFT smart acquisition/rebalancing program"""

    st.title("DRIFT Smart Acquisition Dashboard")
    st.markdown(
        "**Real-time tracking of DRIFT smart acquisition via insurance fund rebalancing**"
    )

    with st.expander("⚙️ Configuration", expanded=False):
        col1, col2, col3 = st.columns(3)

        with col1:
            program_id = str(ch.program_id)
            st.text(f"Program ID: {program_id}")

        with col2:
            max_transactions = st.number_input(
                "Max transactions to fetch",
                min_value=5,
                max_value=100,
                value=5,
                help="Higher values = more complete data but slower loading",
            )

        with col3:
            auto_refresh = st.toggle("Auto-refresh (30s)", value=False)

    summary_tab, transactions_tab, analytics_tab, debug_tab = st.tabs(
        ["📊 Summary", "💱 Transactions", "📈 Analytics", "🔧 Debug"]
    )

    with summary_tab:
        metrics_col1, metrics_col2, metrics_col3, metrics_col4 = st.columns(4)

        progress_bar = st.progress(0, text="Initializing...")

        try:
            progress_bar.progress(1, text="Finding DRIFT market configuration...")
            drift_market_index = get_drift_market_index()

            total_usdc_allocated = 0
            if drift_market_index is None:
                st.error("Could not find DRIFT spot market.")
                progress_bar.empty()
                return
            else:
                progress_bar.progress(2, text="Fetching rebalance configuration...")
                rebalance_config_pk = get_if_rebalance_config_public_key(
                    ch.program_id, 0, drift_market_index
                )
                try:
                    rebalance_config = await ch.program.account[
                        "IfRebalanceConfig"
                    ].fetch(rebalance_config_pk)
                    with st.expander("Rebalance Config"):
                        st.write(rebalance_config)
                    total_usdc_allocated = rebalance_config.total_in_amount / 1e6
                    st.write(f"Total USDC allocated: ${total_usdc_allocated:,.2f}")
                except Exception as e:
                    st.warning(
                        f"Could not fetch rebalancing config: {e}. Using default allocation."
                    )
                    total_usdc_allocated = 1000000

            progress_bar.progress(3, text="Fetching transaction history...")
            swap_events, debug_info = await fetch_insurance_fund_swap_events(
                ch,
                max_transactions,
                drift_market_index,
                Pubkey.from_string("BuynBZjr5yiZCFpXngFQ31BAwechmFE1Ab6vNP3f5PTt"),
                progress_bar,
            )

            progress_bar.progress(100, text="Transaction loading complete!")

        finally:
            # Clean up progress bar after a short delay
            import asyncio

            await asyncio.sleep(0.5)
            progress_bar.empty()

        if not swap_events:
            st.warning(
                "No DRIFT buyback transactions found yet. The program may not have started or transactions are still pending."
            )
            st.info(
                "💡 The TWAP strategy is expected to begin around June 16-23, 22024"
            )
            with st.expander("Debug Info"):
                st.json(debug_info)
            return

        current_usdc_sold = calculate_current_usdc_sold(swap_events)
        current_drift_bought = calculate_current_drift_bought(swap_events)
        avg_drift_price = calculate_average_drift_price(swap_events)

        # Display key metrics
        with metrics_col1:
            st.metric(
                label="Total USDC Allocated",
                value=f"${total_usdc_allocated:,.2f}",
                help="Total USDC allocated for DRIFT buyback program",
            )

        with metrics_col2:
            st.metric(
                label="USDC Sold",
                value=f"${current_usdc_sold:,.2f}",
                delta=f"{(current_usdc_sold / total_usdc_allocated * 100):.1f}% of total"
                if total_usdc_allocated > 0
                else None,
            )

        with metrics_col3:
            st.metric(
                label="DRIFT Bought",
                value=f"{current_drift_bought:,.2f}",
                help="Total DRIFT tokens purchased through the program",
            )

        with metrics_col4:
            st.metric(
                label="Avg DRIFT Price",
                value=f"${avg_drift_price:.4f}",
                help="Volume-weighted average price of DRIFT purchases",
            )

        # Progress bar
        progress = (
            min(current_usdc_sold / total_usdc_allocated, 1.0)
            if total_usdc_allocated > 0
            else 0
        )
        st.progress(progress, text=f"Program Progress: {progress * 100:.1f}%")

        # Recent activity summary
        st.subheader("📈 Recent Activity")
        recent_events = sorted(swap_events, key=lambda x: x["timestamp"], reverse=True)[
            :5
        ]

        if recent_events:
            for event in recent_events:
                with st.container():
                    col1, col2, col3, col4 = st.columns([2, 1, 1, 1])
                    with col1:
                        st.text(
                            f"🕐 {datetime.datetime.fromtimestamp(event['timestamp']).strftime('%Y-%m-%d %H:%M:%S')}"
                        )
                    with col2:
                        st.text(f"💰 ${event['usdc_amount']:,.2f} USDC")
                    with col3:
                        st.text(f"🎯 {event['drift_amount']:,.2f} DRIFT")
                    with col4:
                        st.text(f"💲 ${event['price']:.4f}")
                    st.divider()

    with transactions_tab:
        st.subheader("💱 Swap Transaction History")

        try:
            if "swap_events" in locals() and swap_events:
                df = create_transactions_dataframe(swap_events)

                col1, col2 = st.columns(2)
                with col1:
                    date_filter = st.date_input(
                        "Filter by date (from)",
                        value=datetime.date.today() - datetime.timedelta(days=30),
                    )
                with col2:
                    min_amount = st.number_input(
                        "Min USDC amount", min_value=0.0, value=0.0
                    )

                filtered_df = df[
                    (df["date"] >= date_filter) & (df["usdc_amount"] >= min_amount)
                ]

                st.dataframe(
                    filtered_df,
                    use_container_width=True,
                    column_config={
                        "timestamp": st.column_config.DatetimeColumn("Timestamp"),
                        "usdc_amount": st.column_config.NumberColumn(
                            "USDC Amount", format="$%.2f"
                        ),
                        "drift_amount": st.column_config.NumberColumn(
                            "DRIFT Amount", format="%.2f"
                        ),
                        "price": st.column_config.NumberColumn(
                            "DRIFT Price", format="$%.4f"
                        ),
                        "tx_sig": st.column_config.TextColumn(
                            "Transaction", width="small"
                        ),
                        "solscan_link": st.column_config.LinkColumn(
                            "View on Solscan", display_text="View", width="small"
                        ),
                    },
                )

                csv = filtered_df.to_csv(index=False)
                st.download_button(
                    label="📥 Download as CSV",
                    data=csv,
                    file_name=f"drift_buyback_transactions_{datetime.date.today()}.csv",
                    mime="text/csv",
                )
            else:
                st.info("No transaction data available yet.")

        except Exception as e:
            st.error(f"Error displaying transactions: {str(e)}")

    with analytics_tab:
        st.subheader("📈 Analytics & Charts")

        try:
            if "swap_events" in locals() and swap_events:
                df = create_transactions_dataframe(swap_events)

                fig_price = px.line(
                    df,
                    x="timestamp",
                    y="price",
                    title="DRIFT Purchase Price Over Time",
                    labels={"price": "DRIFT Price (USD)", "timestamp": "Date"},
                )
                fig_price.update_layout(showlegend=False)
                st.plotly_chart(fig_price, use_container_width=True)

                fig_volume = px.bar(
                    df.groupby(df["timestamp"].dt.date)["usdc_amount"]
                    .sum()
                    .reset_index(),
                    x="timestamp",
                    y="usdc_amount",
                    title="Daily USDC Volume Used for DRIFT Purchases",
                    labels={"usdc_amount": "USDC Volume", "timestamp": "Date"},
                )
                st.plotly_chart(fig_volume, use_container_width=True)

                df_sorted = df.sort_values("timestamp")
                df_sorted["cumulative_drift"] = df_sorted["drift_amount"].cumsum()
                df_sorted["cumulative_usdc"] = df_sorted["usdc_amount"].cumsum()

                fig_cumulative = go.Figure()
                fig_cumulative.add_trace(
                    go.Scatter(
                        x=df_sorted["timestamp"],
                        y=df_sorted["cumulative_drift"],
                        mode="lines",
                        name="Cumulative DRIFT Bought",
                        yaxis="y",
                    )
                )
                fig_cumulative.add_trace(
                    go.Scatter(
                        x=df_sorted["timestamp"],
                        y=df_sorted["cumulative_usdc"],
                        mode="lines",
                        name="Cumulative USDC Spent",
                        yaxis="y2",
                    )
                )

                fig_cumulative.update_layout(
                    title="Cumulative DRIFT Buyback Progress",
                    xaxis_title="Date",
                    yaxis=dict(title="DRIFT Tokens", side="left"),
                    yaxis2=dict(title="USDC Spent", side="right", overlaying="y"),
                    showlegend=True,
                )
                st.plotly_chart(fig_cumulative, use_container_width=True)

            else:
                st.info("No data available for analytics yet.")

        except Exception as e:
            st.error(f"Error generating analytics: {str(e)}")

    with debug_tab:
        st.subheader("🔧 Debug Information")

        if "debug_info" in locals():
            st.json(debug_info)

        if "swap_events" in locals():
            st.subheader("Raw Swap Events")
            st.json(swap_events)

    if auto_refresh:
        await asyncio.sleep(30)
        st.rerun()


async def fetch_insurance_fund_swap_events(
    ch: DriftClient,
    max_transactions: int = 1000,
    drift_market_index: int = -1,
    rebalance_config_pk: Pubkey = None,
    progress_bar=None,
):
    """Fetch and parse InsuranceFundSwapRecord events from recent transactions"""

    debug_info = {
        "total_transactions": 0,
        "successful_parses": 0,
        "failed_parses": 0,
        "swap_events_found": 0,
        "all_event_types": [],
        "errors": [],
    }

    try:
        if progress_bar:
            progress_bar.progress(4, text="Fetching transaction signatures...")

        transactions = await transaction_history_for_account(
            ch.program.provider.connection,
            rebalance_config_pk,
            None,  # before_sig
            1000,  # limit per batch
            max_transactions,  # max total
        )

        debug_info["total_transactions"] = len(transactions)

        if not transactions:
            debug_info["errors"].append("No transactions found for program")
            return [], debug_info

        if progress_bar:
            progress_bar.progress(
                5, text=f"Processing {len(transactions)} transactions..."
            )

        parser = EventParser(ch.program.program_id, ch.program.coder)
        swap_events = []

        # Process transactions in parallel batches
        semaphore = asyncio.Semaphore(50)  # Limit concurrent requests

        async def process_transaction(tx, tx_index):
            async with semaphore:
                try:
                    tx_sig = tx["signature"]
                    full_tx = await ch.program.provider.connection.get_transaction(
                        Signature.from_string(tx_sig),
                        max_supported_transaction_version=0,
                    )

                    if not full_tx or not full_tx.value:
                        return None, "failed_parse"

                    logs = []
                    local_event_types = []

                    def event_callback(event):
                        logs.append(event)
                        if event.name not in local_event_types:
                            local_event_types.append(event.name)

                    tx_json = json.loads(full_tx.to_json())
                    if (
                        "result" in tx_json
                        and tx_json["result"]
                        and "meta" in tx_json["result"]
                    ):
                        log_messages = tx_json["result"]["meta"].get("logMessages", [])
                        parser.parse_logs(log_messages, event_callback)

                        tx_swap_events = []
                        for event in logs:
                            if event.name == "InsuranceFundSwapRecord":
                                swap_data = parse_swap_event(
                                    event.data, tx, drift_market_index
                                )
                                if swap_data:
                                    tx_swap_events.append(swap_data)
                            elif event.name in [
                                "TransferProtocolIfSharesToRevenuePoolRecord",
                                "InsuranceFundRecord",
                            ]:
                                pass

                        return {
                            "swap_events": tx_swap_events,
                            "event_types": local_event_types,
                            "status": "success",
                        }, None
                    else:
                        return None, "failed_parse"

                except Exception as e:
                    return None, f"TX {tx_index}: {str(e)}"

        batch_size = 50
        for batch_start in range(0, len(transactions), batch_size):
            batch = transactions[batch_start : batch_start + batch_size]

            if progress_bar:
                progress = 5 + int((batch_start / len(transactions)) * 40)
                progress_bar.progress(
                    progress,
                    text=f"Processing batch {batch_start // batch_size + 1}/{(len(transactions) + batch_size - 1) // batch_size}...",
                )

            tasks = [
                process_transaction(tx, batch_start + i) for i, tx in enumerate(batch)
            ]
            results = await asyncio.gather(*tasks, return_exceptions=True)

            for result, error in results:
                if error:
                    debug_info["failed_parses"] += 1
                    debug_info["errors"].append(error)
                elif result and result["status"] == "success":
                    debug_info["successful_parses"] += 1
                    swap_events.extend(result["swap_events"])
                    debug_info["swap_events_found"] += len(result["swap_events"])
                    for event_type in result["event_types"]:
                        if event_type not in debug_info["all_event_types"]:
                            debug_info["all_event_types"].append(event_type)
                else:
                    debug_info["failed_parses"] += 1

        if progress_bar:
            progress_bar.progress(
                90, text=f"Found {len(swap_events)} buyback transactions!"
            )

        return swap_events, debug_info

    except Exception as e:
        debug_info["errors"].append(f"Failed to fetch swap events: {str(e)}")
        return [], debug_info


def parse_swap_event(event_data, tx_info, drift_market_index: int):
    """Parse InsuranceFundSwapRecord event data into our tracking format"""
    try:
        timestamp = tx_info.get("blockTime", 0)
        event_dict = event_data.__dict__ if hasattr(event_data, "__dict__") else {}
        possible_fields = [
            "inMarketIndex",
            "in_market_index",
            "marketIndexIn",
            "outMarketIndex",
            "out_market_index",
            "marketIndexOut",
            "inAmount",
            "in_amount",
            "amountIn",
            "outAmount",
            "out_amount",
            "amountOut",
            "ts",
            "timestamp",
        ]

        in_market = None
        out_market = None
        in_amount = None
        out_amount = None

        for field in possible_fields:
            value = getattr(event_data, field, None) or event_dict.get(field)
            if value is not None:
                if "in" in field.lower() and "market" in field.lower():
                    in_market = value
                elif "out" in field.lower() and "market" in field.lower():
                    out_market = value
                elif "in" in field.lower() and "amount" in field.lower():
                    in_amount = value
                elif "out" in field.lower() and "amount" in field.lower():
                    out_amount = value

        if in_market is None:
            in_market = getattr(event_data, "marketIndex", 0)
        if out_market is None:
            out_market = getattr(event_data, "marketIndex", 0)
        if in_amount is None:
            in_amount = getattr(event_data, "amount", 0)
        if out_amount is None:
            out_amount = getattr(event_data, "amount", 0)

        if (
            drift_market_index != -1
            and in_market == drift_market_index
            and out_market == 0
        ):
            usdc_amount = in_amount / 1e6
            drift_amount = out_amount / 1e6

        elif (
            drift_market_index != -1
            and in_market == 0
            and out_market == drift_market_index
        ):
            return None
        else:
            return None

        if usdc_amount <= 0 or drift_amount <= 0:
            return None

        return {
            "timestamp": timestamp,
            "tx_sig": tx_info.get("signature", ""),
            "in_market": in_market,
            "out_market": out_market,
            "usdc_amount": usdc_amount,
            "drift_amount": drift_amount,
            "price": usdc_amount / drift_amount if drift_amount > 0 else 0,
            "raw_event": event_dict,
        }

    except Exception as e:
        st.error(f"Error parsing swap event: {str(e)}")
        return None


def calculate_current_usdc_sold(swap_events):
    """Calculate total USDC sold for DRIFT"""
    print("Calculating current USDC sold")
    return sum(event["usdc_amount"] for event in swap_events)


def calculate_current_drift_bought(swap_events):
    """Calculate total DRIFT bought"""
    print("Calculating current DRIFT bought")
    return sum(event["drift_amount"] for event in swap_events)


def calculate_average_drift_price(swap_events):
    """Calculate volume-weighted average DRIFT price"""
    print("Calculating average DRIFT price")
    if not swap_events:
        return 0.0

    total_usdc = sum(event["usdc_amount"] for event in swap_events)
    total_drift = sum(event["drift_amount"] for event in swap_events)

    return total_usdc / total_drift if total_drift > 0 else 0.0


def create_transactions_dataframe(swap_events):
    """Create a pandas DataFrame from swap events for display"""
    df = pd.DataFrame(swap_events)

    if not df.empty:
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s")
        df["date"] = df["timestamp"].dt.date
        df = df.sort_values("timestamp", ascending=False)
        df["solscan_link"] = df["tx_sig"].apply(lambda x: f"https://solscan.io/tx/{x}")

        if "raw_event" in df.columns:
            df = df.drop("raw_event", axis=1)

    return df

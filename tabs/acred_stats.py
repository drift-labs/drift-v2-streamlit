import pandas as pd  
import streamlit as st  
import base58
from driftpy.drift_client import DriftClient  
from solders.pubkey import Pubkey # type: ignore
from solana.rpc.types import MemcmpOpts

# Define the search term in plain text for better readability
ACCOUNT_NAME_SEARCH_TERM = "acred"
  
async def show_acred_stats(drift_client: DriftClient):  
    st.title("ACRED Accounts Statistics")  
    
    try:  
        with st.spinner("Fetching ACRED accounts..."):
            # First verify the RPC endpoint is working and then fetch user accounts
            st.toast("Fetching ACRED accounts...")
            
            # Convert the search term to base58 for account filtering
            encoded_name = ACCOUNT_NAME_SEARCH_TERM.encode("utf-8")
            encoded_name_b58 = base58.b58encode(encoded_name).decode("utf-8")
            
            # Add Pool ID dropdown selector
            pool_options = {
                "All": None,
                "1 (JLP Pool)": 1,
                "2 (LST Pool)": 2,
                "3 (Exponent Pool)": 3,
                "4 (ACRED Pool)": 4
            }
            selected_label = st.selectbox("Select Pool ID to filter", list(pool_options.keys()))
            selected_pool_id = pool_options[selected_label]

            # Build filters for the RPC call
            filters = [MemcmpOpts(offset=72, bytes=encoded_name_b58)]
            if selected_pool_id is not None:
                # pool_id is a uint8 at offset 344, encode as a single byte and then base58
                pool_id_bytes = bytes([selected_pool_id])
                pool_id_b58 = base58.b58encode(pool_id_bytes).decode("utf-8")
                filters.append(MemcmpOpts(offset=344, bytes=pool_id_b58))

            # Fetch users with the constructed filters
            users = await drift_client.program.account["User"].all(filters=filters)
          
        if not users:  
            st.warning("No ACRED accounts found")  
            return  
          
        # Process the accounts data  
        accounts_data = []  
        total_deposits = 0  
          
        for user_account in users:  
            account = user_account.account  
            authority = str(account.authority)  
            sub_id = account.sub_account_id  
            name = bytes(account.name).decode('utf-8', errors='ignore').strip('\x00')  
            deposits = account.total_deposits / 1e6  # Convert to USDC  
            withdrawals = account.total_withdraws / 1e6  
            net_deposits = deposits - withdrawals  
            pool_id = account.pool_id
              
            accounts_data.append({  
                "Authority": authority,  
                "SubAccount ID": sub_id,  
                "Name": name,  
                "Total Deposits": deposits,  
                "Total Withdrawals": withdrawals,  
                "Net Deposits": net_deposits,
                "Pool ID": pool_id
            })  
              
            total_deposits += deposits  
          
        # Create DataFrame and display  
        df = pd.DataFrame(accounts_data)  
        
        # Display summary metrics
        st.metric("Total ACRED Accounts", len(accounts_data))
        st.metric("Total Deposits to ACRED", f"${total_deposits:,.2f}")
        
        # Display detailed table
        st.subheader("ACRED Accounts Details")
        st.dataframe(df)
        
        # Optional: Add visualizations
        if not df.empty:
            st.subheader("Deposits Distribution")
            st.bar_chart(df.set_index("Authority")["Total Deposits"])
    
    except Exception as e:
        st.error(f"Error fetching ACRED accounts: {str(e)}")
        st.info("Please check the server logs for more details.")
        
        # Add RPC endpoint info and debug information
        try:
            st.info(f"Using RPC endpoint: {drift_client.program.provider.connection._provider.endpoint_uri}")
            import traceback
            st.code(traceback.format_exc(), language="python")
        except Exception as debug_error:
            st.error(f"Error generating debug info: {str(debug_error)}")
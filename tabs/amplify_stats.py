import pandas as pd  
import streamlit as st  
import base58
from driftpy.drift_client import DriftClient  
from solders.pubkey import Pubkey # type: ignore
from solana.rpc.types import MemcmpOpts

# Define the search term in plain text for better readability
ACCOUNT_NAME_SEARCH_TERM = "Amplify"
  
async def show_amplify_stats(drift_client: DriftClient):  
    st.title("Amplify Accounts Statistics")  
    
    try:  
        with st.spinner("Fetching Amplify accounts..."):
            # First verify the RPC endpoint is working and then fetch user accounts
            st.toast("Fetching Amplify accounts...")
            
            # Convert the search term to base58 for account filtering
            encoded_name = ACCOUNT_NAME_SEARCH_TERM.encode("utf-8")
            encoded_name_b58 = base58.b58encode(encoded_name).decode("utf-8")
            
            # This is the correct User discriminator used in other parts of the codebase
            users = await drift_client.program.account["User"].all(
                filters=[
                    MemcmpOpts(offset=72, bytes=encoded_name_b58) # Dynamically encoded account name
                ]
            )
          
        if not users:  
            st.warning("No Amplify accounts found")  
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
              
            accounts_data.append({  
                "Authority": authority,  
                "SubAccount ID": sub_id,  
                "Name": name,  
                "Total Deposits": deposits,  
                "Total Withdrawals": withdrawals,  
                "Net Deposits": net_deposits  
            })  
              
            total_deposits += deposits  
          
        # Create DataFrame and display  
        df = pd.DataFrame(accounts_data)  
          
        # Display summary metrics  
        st.metric("Total Amplify Accounts", len(accounts_data))  
        st.metric("Total Deposits to Amplify", f"${total_deposits:,.2f}")  
          
        # Display detailed table  
        st.subheader("Amplify Accounts Details")  
        st.dataframe(df)  
          
        # Optional: Add visualizations  
        if not df.empty:  
            st.subheader("Deposits Distribution")  
            st.bar_chart(df.set_index("Authority")["Total Deposits"])
    
    except Exception as e:
        st.error(f"Error fetching Amplify accounts: {str(e)}")
        st.info("Please check the server logs for more details.")
        
        # Add RPC endpoint info and debug information
        try:
            st.info(f"Using RPC endpoint: {drift_client.program.provider.connection._provider.endpoint_uri}")
            import traceback
            st.code(traceback.format_exc(), language="python")
        except Exception as debug_error:
            st.error(f"Error generating debug info: {str(debug_error)}")
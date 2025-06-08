import os
import timeit
from typing import Literal

import boto3
from boto3.dynamodb.conditions import Key
import pandas as pd
import streamlit as st

# Configuration
DYNAMODB_TABLE_NAME = os.getenv("DYNAMODB_TABLE_NAME") or "mainnet-beta-analytics"
DYNAMODB_REGION = "eu-west-1"
AWS_PROFILE = os.getenv("AWS_PROFILE")

# Initialize DynamoDB client
@st.cache_resource
def get_dynamodb_client():
    try:
        # Use AWS SSO profile
        session = boto3.Session(region_name=DYNAMODB_REGION, profile_name=AWS_PROFILE)
        return session.resource('dynamodb', region_name=DYNAMODB_REGION)
    except Exception as e:
        st.error(f"Failed to initialize DynamoDB client with profile '{AWS_PROFILE}': {e}")
        st.error("Make sure you've run: `aws sso login --profile drift-non-prod`")
        return None

def get_trigger_order_fill_pk(
        market: str,
        order_type: Literal['triggerMarket', 'triggerLimit', 'all'],
        cohort: Literal['0', '1000', '10000', '100000'] = '0') -> str:
    """Generate the partition key for trigger order fill stats"""
    return f"ANALYTICS#TRIGGER_ORDER_FILL#{market}#{order_type}#{cohort}"

def get_auction_latency_pk(market: str, cohort: str, bit_flags: str, taker_order_type: str, taker_order_direction: str) -> str:
    """Generate the partition key for auction latency stats"""
    return f"ANALYTICS#AUCTION_LATENCY#{market}#D#{cohort}#{bit_flags}#{taker_order_type}#{taker_order_direction}"

def get_fill_speed_pk(market: str, cohort: str, bit_flags: str) -> str:
    """Generate the partition key for fill speed stats (same as auction latency)"""
    return f"ANALYTICS#AUCTION_LATENCY#{market}#D#{cohort}#{bit_flags}"

def get_liquidity_source_pk(market: str, cohort: str, taker_order_type: str, bit_flag: str) -> str:
    """Generate the partition key for liquidity source stats"""
    return f"ANALYTICS#LIQUIDITY_SOURCE#{market}#{cohort}#{taker_order_type}#{bit_flag}"

@st.cache_data(ttl=300)
def fetch_trigger_speed_data_dynamodb(start_ts, end_ts, market_symbol, order_type, cohort='0'):
    """
    Fetch trigger speed data from DynamoDB
    
    Args:
        start_ts: Start timestamp (Unix seconds)
        end_ts: End timestamp (Unix seconds) 
        market_symbol: Market symbol (e.g., 'SOL-PERP')
        order_type: 'triggerMarket', 'triggerLimit', or 'all'
        cohort: Cohort identifier (default '0')
    """
    try:
        dynamodb = get_dynamodb_client()
        
        # Check if client initialization failed
        if dynamodb is None:
            st.error("Cannot proceed without valid DynamoDB connection.")
            return pd.DataFrame()
            
        table = dynamodb.Table(DYNAMODB_TABLE_NAME)
        
        pk = get_trigger_order_fill_pk(market_symbol, order_type, cohort)
        
        start_time = timeit.default_timer()
        
        # Collect all items with pagination
        all_items = []
        last_evaluated_key = None
        page_count = 0
        
        while True:
            page_count += 1
            query_params = {
                'KeyConditionExpression': Key('pk').eq(pk) & Key('sk').between(str(start_ts), str(end_ts)),
                'ScanIndexForward': True
            }
            
            if last_evaluated_key:
                query_params['ExclusiveStartKey'] = last_evaluated_key
            
            response = table.query(**query_params)
            
            items = response.get('Items', [])
            all_items.extend(items)
            
            # Check if there are more items to fetch
            last_evaluated_key = response.get('LastEvaluatedKey')
            if not last_evaluated_key:
                break
        
        if not all_items:
            st.warning(f"No records found in DynamoDB for PK: {pk} in the specified time range.")
            return pd.DataFrame()
        
        # Convert to DataFrame
        records_df = pd.DataFrame(all_items)
        
        # Convert sk back to ts for compatibility with existing code
        if 'sk' in records_df.columns:
            records_df['ts'] = pd.to_numeric(records_df['sk'], errors='coerce')
            records_df.sort_values('ts', inplace=True)
        
        return records_df
        
    except Exception as e:
        st.error(f"❌ Error fetching data from DynamoDB: {str(e)}")
        st.error(f"Table: {DYNAMODB_TABLE_NAME}, Region: {DYNAMODB_REGION}")
        return pd.DataFrame()

@st.cache_data(ttl=300)
def fetch_auction_latency_data_dynamodb(start_ts, end_ts, market_symbol, bit_flags, cohort='1000', taker_order_type='all', taker_order_direction='all'):
    """
    Fetch auction latency data from DynamoDB
    
    Args:
        start_ts: Start timestamp (Unix seconds)
        end_ts: End timestamp (Unix seconds) 
        market_symbol: Market symbol (e.g., 'SOL-PERP')
        bit_flags: Bit flags for the auction
        cohort: Cohort identifier (default '1000')
        taker_order_type: Taker order type (default 'all')
        taker_order_direction: Taker order direction (default 'all')
    """
    try:
        dynamodb = get_dynamodb_client()
        
        # Check if client initialization failed
        if dynamodb is None:
            st.error("Cannot proceed without valid DynamoDB connection.")
            return pd.DataFrame()
            
        table = dynamodb.Table(DYNAMODB_TABLE_NAME)
        
        pk = get_auction_latency_pk(market_symbol, cohort, bit_flags, taker_order_type, taker_order_direction)
        
        start_time = timeit.default_timer()
        
        # Collect all items with pagination
        all_items = []
        last_evaluated_key = None
        page_count = 0
        
        while True:
            page_count += 1
            query_params = {
                'KeyConditionExpression': Key('pk').eq(pk) & Key('sk').between(str(start_ts), str(end_ts)),
                'ScanIndexForward': True
            }
            
            if last_evaluated_key:
                query_params['ExclusiveStartKey'] = last_evaluated_key
            
            response = table.query(**query_params)
            
            items = response.get('Items', [])
            all_items.extend(items)
            
            # Check if there are more items to fetch
            last_evaluated_key = response.get('LastEvaluatedKey')
            if not last_evaluated_key:
                break
        
        if not all_items:
            st.warning(f"No records found in DynamoDB for PK: {pk} in the specified time range.")
            return pd.DataFrame()
        
        # Convert to DataFrame
        records_df = pd.DataFrame(all_items)
        
        # Convert sk back to ts for compatibility with existing code
        if 'sk' in records_df.columns:
            records_df['ts'] = pd.to_numeric(records_df['sk'], errors='coerce')
            records_df.sort_values('ts', inplace=True)
        
        return records_df
        
    except Exception as e:
        st.error(f"❌ Error fetching data from DynamoDB: {str(e)}")
        st.error(f"Table: {DYNAMODB_TABLE_NAME}, Region: {DYNAMODB_REGION}")
        return pd.DataFrame()

@st.cache_data(ttl=300)
def fetch_fill_quality_data_dynamodb(start_ts, end_ts, market_symbol, order_type, cohort='0'):
    """
    Fetch fill speed data from DynamoDB (uses auction latency data)
    
    Args:
        start_ts: Start timestamp (Unix seconds)
        end_ts: End timestamp (Unix seconds) 
        market_symbol: Market symbol (e.g., 'SOL-PERP')
        order_type: Order type for bit flags mapping ('swift', 'normal', 'all')
        cohort: Cohort identifier (default '0')
    """
    try:
        dynamodb = get_dynamodb_client()
        
        # Check if client initialization failed
        if dynamodb is None:
            st.error("Cannot proceed without valid DynamoDB connection.")
            return pd.DataFrame()
            
        table = dynamodb.Table(DYNAMODB_TABLE_NAME)
        
        # Map order type to bit flags
        if order_type == "swift":
            bit_flags = "1"
        elif order_type == "normal":
            bit_flags = "0"
        else:  # "all"
            bit_flags = "all"
        
        pk = get_fill_speed_pk(market_symbol, cohort, bit_flags)
        
        start_time = timeit.default_timer()
        
        # Collect all items with pagination
        all_items = []
        last_evaluated_key = None
        page_count = 0
        
        while True:
            page_count += 1
            query_params = {
                'KeyConditionExpression': Key('pk').eq(pk) & Key('sk').between(str(start_ts), str(end_ts)),
                'ScanIndexForward': True
            }
            
            if last_evaluated_key:
                query_params['ExclusiveStartKey'] = last_evaluated_key
            
            response = table.query(**query_params)
            
            items = response.get('Items', [])
            all_items.extend(items)
            
            # Check if there are more items to fetch
            last_evaluated_key = response.get('LastEvaluatedKey')
            if not last_evaluated_key:
                break
        
        if not all_items:
            st.warning(f"No records found in DynamoDB for PK: {pk} in the specified time range.")
            return pd.DataFrame()
        
        # Convert to DataFrame
        records_df = pd.DataFrame(all_items)
        
        # Convert sk back to ts for compatibility with existing code
        if 'sk' in records_df.columns:
            records_df['ts'] = pd.to_numeric(records_df['sk'], errors='coerce')
            records_df.sort_values('ts', inplace=True)
        
        # Convert percentile values from decimal to percentage (multiply by 100)
        # The data in DynamoDB is stored as decimals (0.0-1.0), but fill_speed.py expects percentages
        percentile_cols = [f"p{p}" for p in [10, 20, 25, 30, 40, 50, 60, 70, 75, 80, 90, 99]]
        for col in percentile_cols + ['min', 'max']:
            if col in records_df.columns:
                records_df[col] = pd.to_numeric(records_df[col], errors='coerce') * 100
        
        # Convert count column to numeric to prevent string concatenation when summing
        if 'count' in records_df.columns:
            records_df['count'] = pd.to_numeric(records_df['count'], errors='coerce')
        
        # Map cohort values to display names expected by fill_speed.py
        cohort_mapping = {
            '0': '0-1k',
            '1000': '1k-10k', 
            '10000': '10k-100k',
            '100000': '100k+'
        }
        
        records_df['cohort'] = records_df.get('cohort', cohort).map(cohort_mapping).fillna(cohort_mapping.get(cohort, '0-1k'))
        
        # Add order_type column based on bit_flags
        order_type_mapping = {
            '0': 'normal',
            '1': 'swift',
            'all': 'all'
        }
        
        records_df['order_type'] = order_type_mapping.get(bit_flags, 'all')
        
        return records_df
        
    except Exception as e:
        st.error(f"❌ Error fetching fill speed data from DynamoDB: {str(e)}")
        st.error(f"Table: {DYNAMODB_TABLE_NAME}, Region: {DYNAMODB_REGION}")
        return pd.DataFrame()

@st.cache_data(ttl=300)
def fetch_liquidity_source_data_dynamodb(start_ts, end_ts, market_symbol, cohort='all', taker_order_type='all', bit_flag='all'):
    """
    Fetch liquidity source data from DynamoDB
    
    Args:
        start_ts: Start timestamp (Unix seconds)
        end_ts: End timestamp (Unix seconds) 
        market_symbol: Market symbol (e.g., 'SOL-PERP')
        cohort: Cohort identifier (default 'all')
        taker_order_type: Taker order type (default 'all')
        bit_flag: Bit flag (default 'all')
    """
    try:
        dynamodb = get_dynamodb_client()
        
        # Check if client initialization failed
        if dynamodb is None:
            st.error("Cannot proceed without valid DynamoDB connection.")
            return pd.DataFrame()
            
        table = dynamodb.Table(DYNAMODB_TABLE_NAME)
        
        pk = get_liquidity_source_pk(market_symbol, cohort, taker_order_type, bit_flag)
        
        start_time = timeit.default_timer()
        
        # Collect all items with pagination
        all_items = []
        last_evaluated_key = None
        page_count = 0
        
        while True:
            page_count += 1
            query_params = {
                'KeyConditionExpression': Key('pk').eq(pk) & Key('sk').between(str(start_ts), str(end_ts)),
                'ScanIndexForward': True
            }
            
            if last_evaluated_key:
                query_params['ExclusiveStartKey'] = last_evaluated_key
            
            response = table.query(**query_params)
            
            items = response.get('Items', [])
            all_items.extend(items)
            
            # Check if there are more items to fetch
            last_evaluated_key = response.get('LastEvaluatedKey')
            if not last_evaluated_key:
                break
        
        if not all_items:
            st.warning(f"No records found in DynamoDB for PK: {pk} in the specified time range.")
            return pd.DataFrame()
        
        # Convert to DataFrame
        records_df = pd.DataFrame(all_items)
        
        # Convert sk back to ts for compatibility with existing code
        if 'sk' in records_df.columns:
            records_df['ts'] = pd.to_numeric(records_df['sk'], errors='coerce')
            records_df.sort_values('ts', inplace=True)
        
        # Convert numeric columns to proper types
        numeric_columns = [
            'totalMatch', 'totalMatchJit', 'totalAmm', 'totalAmmJit', 
            'totalAmmJitLpSplit', 'totalLpJit', 'totalSerum', 'totalPhoenix',
            'countMatch', 'countMatchJit', 'countAmm', 'countAmmJit',
            'countAmmJitLpSplit', 'countLpJit', 'countSerum', 'countPhoenix'
        ]
        
        for col in numeric_columns:
            if col in records_df.columns:
                records_df[col] = pd.to_numeric(records_df[col], errors='coerce')
        
        return records_df
        
    except Exception as e:
        st.error(f"❌ Error fetching liquidity source data from DynamoDB: {str(e)}")
        st.error(f"Table: {DYNAMODB_TABLE_NAME}, Region: {DYNAMODB_REGION}")
        return pd.DataFrame()

@st.cache_data(ttl=300)
def fetch_fill_metrics_data_dynamodb(start_ts, end_ts, market_symbol, cohort='all', taker_order_type='all', taker_order_direction='all', bit_flag='all'):
    """
    Fetch fill metrics data from DynamoDB (AuctionLatencyStats)
    
    Args:
        start_ts: Start timestamp (Unix seconds)
        end_ts: End timestamp (Unix seconds) 
        market_symbol: Market symbol (e.g., 'SOL-PERP')
        cohort: Cohort identifier (default 'all')
        taker_order_type: Taker order type (default 'all')
        taker_order_direction: Taker order direction (default 'all')
        bit_flag: Bit flag (default 'all')
    """
    try:
        dynamodb = get_dynamodb_client()
        
        # Check if client initialization failed
        if dynamodb is None:
            st.error("Cannot proceed without valid DynamoDB connection.")
            return pd.DataFrame()
            
        table = dynamodb.Table(DYNAMODB_TABLE_NAME)
        
        pk = get_auction_latency_pk(market_symbol, cohort, bit_flag, taker_order_type, taker_order_direction)
        
        start_time = timeit.default_timer()
        
        # Collect all items with pagination
        all_items = []
        last_evaluated_key = None
        page_count = 0
        
        while True:
            page_count += 1
            query_params = {
                'KeyConditionExpression': Key('pk').eq(pk) & Key('sk').between(str(start_ts), str(end_ts)),
                'ScanIndexForward': True
            }
            
            if last_evaluated_key:
                query_params['ExclusiveStartKey'] = last_evaluated_key
            
            response = table.query(**query_params)
            
            items = response.get('Items', [])
            all_items.extend(items)
            
            # Check if there are more items to fetch
            last_evaluated_key = response.get('LastEvaluatedKey')
            if not last_evaluated_key:
                break
        
        if not all_items:
            st.warning(f"No records found in DynamoDB for PK: {pk} in the specified time range.")
            return pd.DataFrame()
        
        # Convert to DataFrame
        records_df = pd.DataFrame(all_items)
        
        # Convert sk back to ts for compatibility with existing code
        if 'sk' in records_df.columns:
            records_df['ts'] = pd.to_numeric(records_df['sk'], errors='coerce')
            records_df.sort_values('ts', inplace=True)
        
        # Convert all numeric columns based on AuctionLatencyStats interface
        numeric_columns = [
            'auctionProgressCount', 'auctionProgressMin', 'auctionProgressMax', 'auctionProgressAvg',
            'auctionProgressP10', 'auctionProgressP25', 'auctionProgressP50', 'auctionProgressP75', 'auctionProgressP99',
            'fillVsOracleAbsMin', 'fillVsOracleAbsMax', 'fillVsOracleAbsAvg',
            'fillVsOracleAbsP10', 'fillVsOracleAbsP25', 'fillVsOracleAbsP50', 'fillVsOracleAbsP75', 'fillVsOracleAbsP99',
            'fillVsOracleAbsBpsMin', 'fillVsOracleAbsBpsMax', 'fillVsOracleAbsBpsAvg',
            'fillVsOracleAbsBpsP10', 'fillVsOracleAbsBpsP25', 'fillVsOracleAbsBpsP50', 'fillVsOracleAbsBpsP75', 'fillVsOracleAbsBpsP99'
        ]
        
        for col in numeric_columns:
            if col in records_df.columns:
                records_df[col] = pd.to_numeric(records_df[col], errors='coerce')
        
        # Add segment identifiers for easier filtering
        records_df['market'] = market_symbol
        records_df['cohort'] = cohort
        records_df['takerOrderType'] = taker_order_type
        records_df['takerOrderDirection'] = taker_order_direction
        records_df['bitFlag'] = bit_flag
        
        return records_df
        
    except Exception as e:
        st.error(f"❌ Error fetching fill metrics data from DynamoDB: {str(e)}")
        st.error(f"Table: {DYNAMODB_TABLE_NAME}, Region: {DYNAMODB_REGION}")
        return pd.DataFrame() 

import pandas as pd


def load_user_snapshot(user_account_pubkey: str, commit_hash="main"):
    root = "https://raw.githubusercontent.com/0xbigz/drift-v2-flat-data"
    commit_hash = "main"
    ff = f"{root}/{commit_hash}/data/users/{user_account_pubkey}.csv"
    df = pd.read_csv(ff)
    return df, ff

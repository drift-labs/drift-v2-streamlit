import pandas as pd
import requests
import streamlit as st
from driftpy.drift_client import DriftClient

pd.options.plotting.backend = "plotly"


async def show_api(clearing_house: DriftClient):
    tags = []
    tag_url = "https://api.github.com/repos/drift-labs/protocol-v2/tags"
    try:
        tags = requests.get(tag_url).json()
        tags = [x["name"] for x in tags]
    except Exception as e:
        print(e)
        st.warning("trouble loading " + tag_url)
    tags = ["master"] + tags
    tt1, tt2 = st.columns(2)
    tag = tt1.selectbox("tag:", tags)

    url = (
        "https://raw.githubusercontent.com/drift-labs/protocol-v2/"
        + tag
        + "/sdk/src/idl/drift.json"
    )
    tt2.write(url)

    response = requests.get(url)
    data = response.json()
    tabs = st.tabs(["function", "types", "error"])

    def get_signer(x):
        for y in x:
            if y["isSigner"]:
                if y["name"] == "admin":
                    return "admin"
                return "user"

    with tabs[0]:
        instrs = data["instructions"]
        df = pd.DataFrame(instrs)
        df["signer"] = df["accounts"].apply(lambda x: get_signer(x))
        st.dataframe(df[["name", "signer"]])
    with tabs[1]:
        instrs = data["types"]
        df = pd.DataFrame(instrs)
        st.dataframe(df)
    with tabs[2]:
        s1, s2 = st.columns(2)
        errors = data["errors"]
        df = pd.DataFrame(errors).set_index("code")
        print(str(df.to_markdown()))
        s1.dataframe(df)
        hex_string = s2.text_input(
            "error code lookup:",
        )
        if hex_string:
            if hex_string[:2] == "0x":
                decimal_number = int(hex_string, 16)
                s2.write(decimal_number)
            else:
                decimal_number = int(hex_string)
                s2.write(decimal_number)

            if decimal_number >= 6000:
                if decimal_number - 6000 < len(errors):
                    error = errors[decimal_number - 6000]
                    s2.write(error)
            else:
                s2.write("not a drift error")
                s2.warning("if number is in `hex` please add `0x`")

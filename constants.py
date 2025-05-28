from driftpy.constants.perp_markets import mainnet_perp_market_configs
from driftpy.constants.spot_markets import mainnet_spot_market_configs


def get_all_markets():
    perp_markets = [m.symbol for m in mainnet_perp_market_configs]
    spot_markets = [m.symbol for m in mainnet_spot_market_configs]

    markets = []
    for perp in perp_markets:
        markets.append(perp)
        base_asset = perp.replace("-PERP", "")
        if base_asset in spot_markets:
            markets.append(base_asset)

    for spot in spot_markets:
        if spot not in markets:
            markets.append(spot)

    return markets


ALL_MARKET_NAMES = [
    "1KWEN-PERP",
    "1MBONK-PERP",
    "1MPEPE-PERP",
    "APT-PERP",
    "ARB-PERP",
    "AVAX-PERP",
    "BNB-PERP",
    "BNSOL",
    "BREAKPOINT-IGGYERIC-BET",
    "BTC-PERP",
    "CLOUD-PERP",
    "CLOUD",
    "DBR-PERP",
    "DEMOCRATS-WIN-MICHIGAN-BET",
    "DOGE-PERP",
    "DRIFT-PERP",
    "DRIFT",
    "DYM-PERP",
    "ETH-PERP",
    "FED-CUT-50-SEPT-2024-BET",
    "HNT-PERP",
    "HYPE-PERP",
    "INF",
    "INJ-PERP",
    "IO-PERP",
    "JLP",
    "JTO-PERP",
    "JTO",
    "JUP-PERP",
    "JUP",
    "KAMALA-POPULAR-VOTE-2024-BET",
    "KMNO-PERP",
    "LAUNCHCOIN-PERP",
    "LANDO-F1-SGP-WIN-BET",
    "LINK-PERP",
    "MATIC-PERP",
    "MOODENG-PERP",
    "MOTHER-PERP",
    "MOTHER",
    "OP-PERP",
    "PAXG-PERP",
    "POPCAT-PERP",
    "POPCAT",
    "PYTH-PERP",
    "PYTH",
    "PYUSD",
    "RAY-PERP",
    "RENDER-PERP",
    "RENDER",
    "REPUBLICAN-POPULAR-AND-WIN-BET",
    "RLB-PERP",
    "RNDR-PERP",
    "RNDR",
    "SEI-PERP",
    "SOL-PERP",
    "SOL",
    "SUI-PERP",
    "TAO-PERP",
    "TIA-PERP",
    "TNSR-PERP",
    "TNSR",
    "TON-PERP",
    "TRUMP-WIN-2024-BET",
    "USDC",
    "USDT",
    "USDY",
    "USDe",
    "W-PERP",
    "W",
    "WARWICK-FIGHT-WIN-BET",
    "WIF-PERP",
    "WIF",
    "WLF-5B-1W-BET",
    "XRP-PERP",
    "ZEX-PERP",
    "bSOL",
    "dSOL",
    "jitoSOL",
    "mSOL",
    "sUSDe",
    "wBTC",
    "wETH",
]

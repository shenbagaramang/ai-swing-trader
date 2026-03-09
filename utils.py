"""
utils.py - Utility functions for AI Swing Trader
Supports both NSE (.NS) and BSE (.BO) listed stocks
"""

import pandas as pd
import numpy as np
import requests
import logging
import os
import json
from datetime import datetime, timedelta
import time

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ─── Exchange Constants ───────────────────────────────────────────────────────

EXCHANGE_NSE = "NSE"
EXCHANGE_BSE = "BSE"
EXCHANGE_BOTH = "Both"

# yfinance suffixes
NSE_SUFFIX = ".NS"
BSE_SUFFIX = ".BO"

# ─── BSE Sensex 30 ───────────────────────────────────────────────────────────

SENSEX30_SYMBOLS = [
    "ADANIENT", "ADANIPORTS", "ASIANPAINT", "AXISBANK", "BAJAJ-AUTO",
    "BAJFINANCE", "BHARTIARTL", "HCLTECH", "HDFCBANK", "HINDUNILVR",
    "ICICIBANK", "INDUSINDBK", "INFY", "ITC", "JSWSTEEL",
    "KOTAKBANK", "LT", "M&M", "MARUTI", "NESTLEIND",
    "NTPC", "POWERGRID", "RELIANCE", "SBIN", "SUNPHARMA",
    "TATAMOTORS", "TATASTEEL", "TCS", "TITAN", "WIPRO"
]

# ─── BSE 100 Extra (stocks on BSE, many also on NSE) ─────────────────────────

BSE100_EXTRA = [
    "ABBOTINDIA", "AMBUJACEM", "APOLLOHOSP", "AUROPHARMA", "BANDHANBNK",
    "BANKBARODA", "BEL", "BERGEPAINT", "BIOCON", "BOSCHLTD",
    "BPCL", "BRITANNIA", "CANBK", "CHOLAFIN", "CIPLA",
    "COALINDIA", "COLPAL", "CONCOR", "DABUR", "DIVISLAB",
    "DLF", "DRREDDY", "EICHERMOT", "GAIL", "GODREJCP",
    "GRASIM", "HAVELLS", "HDFCLIFE", "HEROMOTOCO", "HINDALCO",
    "ICICIPRULI", "IDFCFIRSTB", "IOC", "IRCTC", "JUBLFOOD",
    "LUPIN", "MARICO", "MUTHOOTFIN", "NAUKRI", "ONGC",
    "PAGEIND", "PIDILITIND", "PNB", "SAIL", "SBILIFE",
    "SIEMENS", "SRF", "TATACONSUM", "TATAPOWER", "ULTRACEMCO",
]

# ─── BSE SmallCap / MidCap extra stocks ──────────────────────────────────────

BSE_MIDCAP_EXTRA = [
    "AARTIIND", "ABB", "ACC", "AFFLE", "AJANTPHARM",
    "ALKEM", "ANGELONE", "APOLLOTYRE", "ASTRAL", "ATUL",
    "AUBANK", "BAJAJFINSV", "BALRAMCHIN", "BATAINDIA", "BHARATFORG",
    "BHEL", "BLUESTAR", "BRIGADE", "CAMS", "CANFINHOME",
    "CASTROLIND", "CDSL", "CESC", "CHAMBLFERT", "CLEAN",
    "COFORGE", "CROMPTON", "CUMMINSIND", "CYIENT", "DATAPATTNS",
    "DEEPAKNTR", "DELHIVERY", "DELTACORP", "ELGIEQUIP", "EMAMILTD",
    "ESCORTS", "EXIDEIND", "FEDERALBNK", "FINCABLES", "FIVESTAR",
    "FORTIS", "GALAXYSURF", "GLAXO", "GMRINFRA", "GNFC",
    "GODFRYPHLP", "GODREJPROP", "GRANULES", "GRAPHITE", "GUJGASLTD",
    "HAPPSTMNDS", "HAVELLS", "HFCL", "HINDPETRO", "HONAUT",
    "IBULHSGFIN", "ICICIGI", "IDEA", "IGL", "INDHOTEL",
    "INDIGO", "IPCALAB", "JBCHEPHARM", "JKCEMENT", "JKLAKSHMI",
    "JMFINANCIL", "JSWENERGY", "JUBLFOOD", "KAJARIACER", "KALPATPOWR",
    "KANSAINER", "KEC", "KPITTECH", "KPRMILL", "LATENTVIEW",
    "LAURUSLABS", "LEMONTREE", "LTIM", "LUXIND", "MANAPPURAM",
    "MASFIN", "MAXHEALTH", "METROPOLIS", "MFSL", "MINDAIND",
    "MOTHERSON", "MPHASIS", "MRPL", "NATCOPHARM", "NAVINFLUOR",
    "NETWORK18", "NILKAMAL", "NLC", "NMDC", "NUCLEUS",
    "OBEROIRLTY", "OFSS", "OIL", "OLECTRA", "PCBL",
    "PERSISTENT", "PFIZER", "PHOENIXLTD", "PIIND", "POLYCAB",
    "POLYMED", "POONAWALLA", "RADICO", "RAILTEL", "RATNAMANI",
    "RAYMOND", "RECLTD", "REDINGTON", "RITES", "RVNL",
    "SAFARI", "SAREGAMA", "SCHAEFFLER", "SHREECEM", "SHRIRAMFIN",
    "SOBHA", "SONACOMS", "STAR", "STLTECH", "SUDARSCHEM",
    "SUMICHEM", "SUPREMEIND", "SUZLON", "SYNGENE", "TANLA",
    "TATACHEM", "TATACOMM", "TATAELXSI", "TATATECH", "THERMAX",
    "TIMKEN", "TRENT", "UCOBANK", "UJJIVAN", "UNIONBANK",
    "UTIAMC", "VBL", "VEDL", "VINATIORGA", "VOLTAS",
    "WELCORP", "WELSPUNIND", "WHIRLPOOL", "WOCKPHARMA", "YESBANK",
    "ZOMATO", "ZYDUSLIFE",
]

# ─── BSE-Only unique stocks (trade on BSE but not active on NSE yfinance) ─────
# These use .BO suffix exclusively

BSE_ONLY_SYMBOLS = [
    "BAJAJHLDNG",   # Bajaj Holdings
    "BAJAJELEC",    # Bajaj Electricals
    "CENTURYTEX",   # Century Textiles
    "CHENNPETRO",   # Chennai Petroleum
    "CRISIL",       # CRISIL
    "ELECTCAST",    # Electrosteel Castings
    "GESHIP",       # Great Eastern Shipping
    "GILLETTE",     # Gillette India
    "GOODYEAR",     # Goodyear India
    "GREAVESCOT",   # Greaves Cotton
    "GSPL",         # Gujarat State Petronet
    "HDFC",         # HDFC Ltd
    "HINDOILEXP",   # Hindustan Oil Exploration
    "IBREALEST",    # Indiabulls Real Estate
    "IIFL",         # IIFL Finance
    "ISEC",         # ICICI Securities
    "JKPAPER",      # JK Paper
    "JSWHL",        # JSW Holdings
    "KANSAINER",    # Kansai Nerolac
    "KESORAMIND",   # Kesoram Industries
    "KSCL",         # Kaveri Seed
    "LAKSHVILAS",   # Lakshmi Vilas Bank
    "LINDEINDIA",   # Linde India
    "MAXVIL",       # Max Ventures
    "MIDHANI",      # Mishra Dhatu Nigam
    "MOLDTKPAC",    # Mold-Tek Packaging
    "MOREPENLAB",   # Morepen Laboratories
    "NBCC",         # NBCC India
    "NHPC",         # NHPC
    "NSLNISP",      # NMDC Steel
    "OPTIEMUS",     # Optiemus Infracom
    "ORIENTELEC",   # Orient Electric
    "PEARLPOLY",    # Pearl Polymers
    "PRECWIRE",     # Precision Wires
    "PRIMESECU",    # Prime Securities
    "PURVA",        # Puravankara
    "RAJESHEXPO",   # Rajesh Exports
    "RESPONIND",    # Responsive Industries
    "RKFORGE",      # Ramkrishna Forgings
    "ROUTE",        # Route Mobile
    "RPOWER",       # Reliance Power
    "SASKEN",       # Sasken Technologies
    "SEQUENT",      # Sequent Scientific
    "SOLARA",       # Solara Active Pharma
    "SPARC",        # Sun Pharma Advanced
    "SUPRAJIT",     # Suprajit Engineering
    "SUVEN",        # Suven Pharmaceuticals
    "SVBL",         # Shivalik Bimetal Controls
    "TASTYBITE",    # Tasty Bite Eatables
    "TRITURBINE",   # Triveni Turbine
    "UNIPARTS",     # Uniparts India
    "V2RETAIL",     # V2 Retail
    "VAIBHAVGBL",   # Vaibhav Global
    "VAKRANGEE",    # Vakrangee
    "VARROC",       # Varroc Engineering
    "VENKEYS",      # Venky's India
    "VESUVIUS",     # Vesuvius India
    "VIPIND",       # VIP Industries
    "VMART",        # V-Mart Retail
    "VSTIND",       # VST Industries
    "XCHANGING",    # Xchanging Solutions
    "ZENTEC",       # Zen Technologies
]

# ─── NSE Index Compositions ──────────────────────────────────────────────────

NIFTY50_SYMBOLS = [
    "ADANIENT", "ADANIPORTS", "APOLLOHOSP", "ASIANPAINT", "AXISBANK",
    "BAJAJ-AUTO", "BAJFINANCE", "BAJAJFINSV", "BPCL", "BHARTIARTL",
    "BRITANNIA", "CIPLA", "COALINDIA", "DIVISLAB", "DRREDDY",
    "EICHERMOT", "GRASIM", "HCLTECH", "HDFCBANK", "HDFCLIFE",
    "HEROMOTOCO", "HINDALCO", "HINDUNILVR", "ICICIBANK", "ITC",
    "INDUSINDBK", "INFY", "JSWSTEEL", "KOTAKBANK", "LTIM",
    "LT", "M&M", "MARUTI", "NESTLEIND", "NTPC",
    "ONGC", "POWERGRID", "RELIANCE", "SBILIFE", "SBIN",
    "SUNPHARMA", "TCS", "TATACONSUM", "TATAMOTORS", "TATASTEEL",
    "TECHM", "TITAN", "ULTRACEMCO", "UPL", "WIPRO"
]

NIFTY100_EXTRA = [
    "ABB", "ADANIGREEN", "ADANITRANS", "AMBUJACEM", "AUROPHARMA",
    "BANDHANBNK", "BANKBARODA", "BEL", "BERGEPAINT", "BIOCON",
    "BOSCHLTD", "CANBK", "CHOLAFIN", "COLPAL", "CONCOR",
    "DABUR", "DLF", "GAIL", "GODREJCP", "GODREJPROP",
    "HAVELLS", "ICICIGI", "ICICIPRULI", "IDEA", "IDFCFIRSTB",
    "IGL", "INDHOTEL", "INDIGO", "IOC", "IRCTC",
    "JINDALSTEL", "JUBLFOOD", "LUPIN", "MARICO", "MCDOWELL-N",
    "MUTHOOTFIN", "NAUKRI", "NHPC", "NMDC", "OBEROIRLTY",
    "OFSS", "PAGEIND", "PETRONET", "PIDILITIND", "PNB",
    "RECLTD", "SAIL", "SIEMENS", "SRF", "TATAPOWER"
]

NIFTY500_EXTRA = [
    "AARTIIND", "ABBOTINDIA", "ACC", "AFFLE", "AJANTPHARM",
    "ALKEM", "ALKYLAMINE", "AMARAJABAT", "AMBUJACEM", "ANGELONE",
    "APOLLOTYRE", "ASTRAL", "ATGL", "ATUL", "AUBANK",
    "AWHCL", "BALRAMCHIN", "BASF", "BATAINDIA", "BAYERCROP",
    "BHARATFORG", "BHEL", "BIKAJI", "BLUESTAR", "BRIGADE",
    "CAMS", "CANFINHOME", "CARBORUNIV", "CASTROLIND", "CDSL",
    "CESC", "CHAMBLFERT", "CLEAN", "COFORGE", "CROMPTON",
    "CUMMINSIND", "CYIENT", "DATAPATTNS", "DEEPAKNTR", "DELHIVERY",
    "DELTACORP", "ELANTAS", "ELGIEQUIP", "EMAMILTD", "ENGINERSIN",
    "EPL", "ESCORTS", "EXIDEIND", "FACT", "FEDERALBNK",
    "FINCABLES", "FINPIPE", "FIVESTAR", "FORTIS", "FSL",
    "GALAXYSURF", "GILLETTE", "GLAXO", "GMRINFRA", "GNFC",
    "GODFRYPHLP", "GPPL", "GRANULES", "GRAPHITE", "GUJGASLTD",
    "HAPPSTMNDS", "HATSUN", "HBLPOWER", "HFCL", "HIL",
    "HINDPETRO", "HONAUT", "IBREALEST", "IBULHSGFIN", "IDBI",
    "IFBIND", "IFINANCE", "IGPL", "IIFL", "IPCALAB",
    "JBCHEPHARM", "JKCEMENT", "JKLAKSHMI", "JMFINANCIL", "JSWENERGY",
    "JTEKTINDIA", "KAJARIACER", "KALPATPOWR", "KANSAINER", "KEC",
    "KPITTECH", "KPRMILL", "KRBL", "KRISHNADEF", "LATENTVIEW",
    "LAURUSLABS", "LAXMIMACH", "LEMONTREE", "LICI", "LINDEINDIA",
    "LUXIND", "MANAPPURAM", "MASFIN", "MAXHEALTH", "METROPOLIS",
    "MFSL", "MGFL", "MIDHANI", "MINDAIND", "MOREPENLAB",
    "MOTHERSON", "MPHASIS", "MRPL", "NATCOPHARM", "NBCC",
    "NAVINFLUOR", "NETWORK18", "NILKAMAL", "NLC", "NSLNISP",
    "NUCLEUS", "OIL", "OLECTRA", "OPTIEMUS", "ORIENTELEC",
    "PCBL", "PERSISTENT", "PFIZER", "PHOENIXLTD", "PIIND",
    "POLYCAB", "POLYMED", "POONAWALLA", "PRIVISCL", "PURVA",
    "RADICO", "RAILTEL", "RAINBOW", "RAJESHEXPO", "RATNAMANI",
    "RAYMOND", "RCF", "REDINGTON", "RESPONIND", "RITES",
    "RKFORGE", "ROUTE", "RPOWER", "RVNL", "SAFARI",
    "SAREGAMA", "SCHAEFFLER", "SEQUENT", "SHREECEM", "SHRIRAMFIN",
    "SOBHA", "SOLARA", "SONACOMS", "SPANDANA", "SPARC",
    "STAR", "STLTECH", "SUDARSCHEM", "SUMICHEM", "SUPRAJIT",
    "SUPREMEIND", "SUVEN", "SUVENPHAR", "SUZLON", "SYNGENE",
    "TANLA", "TASTYBITE", "TATACHEM", "TATACOMM", "TATAELXSI",
    "TATATECH", "TECHM", "THERMAX", "TIMKEN", "TRENT",
    "TRITURBINE", "UCOBANK", "UJJIVAN", "UNIONBANK", "UNIPARTS",
    "UTIAMC", "V2RETAIL", "VAIBHAVGBL", "VAKRANGEE", "VARROC",
    "VBL", "VEDL", "VENKEYS", "VESUVIUS", "VINATIORGA",
    "VIPIND", "VMART", "VOLTAS", "VSTIND", "WELCORP",
    "WELSPUNIND", "WHIRLPOOL", "WOCKPHARMA", "XCHANGING", "YESBANK",
    "ZENTEC", "ZOMATO", "ZYDUSLIFE"
]


def get_nse_symbols(universe: str = "NIFTY50") -> list:
    """Return list of NSE (.NS) symbols for the given universe."""
    if universe == "NIFTY50":
        return [f"{s}.NS" for s in NIFTY50_SYMBOLS]
    elif universe == "NIFTY100":
        symbols = NIFTY50_SYMBOLS + NIFTY100_EXTRA
        return list(set([f"{s}.NS" for s in symbols]))
    elif universe == "NIFTY500":
        symbols = NIFTY50_SYMBOLS + NIFTY100_EXTRA + NIFTY500_EXTRA
        return list(set([f"{s}.NS" for s in symbols]))
    else:  # ALL NSE
        symbols = NIFTY50_SYMBOLS + NIFTY100_EXTRA + NIFTY500_EXTRA
        return list(set([f"{s}.NS" for s in symbols]))


def get_bse_symbols(universe: str = "SENSEX30") -> list:
    """Return list of BSE (.BO) symbols for the given universe."""
    if universe == "SENSEX30":
        return [f"{s}.BO" for s in SENSEX30_SYMBOLS]
    elif universe == "BSE100":
        symbols = SENSEX30_SYMBOLS + BSE100_EXTRA
        return list(set([f"{s}.BO" for s in symbols]))
    elif universe == "BSE_MIDCAP":
        symbols = SENSEX30_SYMBOLS + BSE100_EXTRA + BSE_MIDCAP_EXTRA
        return list(set([f"{s}.BO" for s in symbols]))
    elif universe == "BSE_ONLY":
        return [f"{s}.BO" for s in BSE_ONLY_SYMBOLS]
    else:  # ALL BSE
        symbols = SENSEX30_SYMBOLS + BSE100_EXTRA + BSE_MIDCAP_EXTRA + BSE_ONLY_SYMBOLS
        return list(set([f"{s}.BO" for s in symbols]))


def get_symbols(universe: str = "NIFTY50", exchange: str = "NSE") -> list:
    """
    Unified symbol getter for NSE, BSE, or Both exchanges.

    universe options:
      NSE  → NIFTY50, NIFTY100, NIFTY500, ALL NSE
      BSE  → SENSEX30, BSE100, BSE_MIDCAP, BSE_ONLY, ALL BSE
      Both → NIFTY50+SENSEX30, NIFTY100+BSE100, NIFTY500+BSE_MIDCAP, ALL
    """
    if exchange == EXCHANGE_NSE:
        return get_nse_symbols(universe)
    elif exchange == EXCHANGE_BSE:
        bse_universe_map = {
            "NIFTY50": "SENSEX30",
            "NIFTY100": "BSE100",
            "NIFTY500": "BSE_MIDCAP",
            "ALL NSE": "ALL BSE",
            "SENSEX30": "SENSEX30",
            "BSE100": "BSE100",
            "BSE_MIDCAP": "BSE_MIDCAP",
            "BSE_ONLY": "BSE_ONLY",
            "ALL BSE": "ALL BSE",
        }
        return get_bse_symbols(bse_universe_map.get(universe, universe))
    else:  # Both
        nse_universe_map = {
            "NIFTY50+SENSEX30": "NIFTY50",
            "NIFTY100+BSE100": "NIFTY100",
            "NIFTY500+BSE_MIDCAP": "NIFTY500",
            "ALL": "ALL NSE",
        }
        bse_universe_map = {
            "NIFTY50+SENSEX30": "SENSEX30",
            "NIFTY100+BSE100": "BSE100",
            "NIFTY500+BSE_MIDCAP": "BSE_MIDCAP",
            "ALL": "ALL BSE",
        }
        nse_uni = nse_universe_map.get(universe, "NIFTY50")
        bse_uni = bse_universe_map.get(universe, "SENSEX30")
        nse = get_nse_symbols(nse_uni)
        bse = get_bse_symbols(bse_uni)
        # Deduplicate: remove BSE ticker if same base symbol already in NSE list
        nse_bases = {s.replace(".NS", "") for s in nse}
        bse_unique = [s for s in bse if s.replace(".BO", "") not in nse_bases]
        return nse + bse_unique


def get_universe_options(exchange: str) -> list:
    """Return list of universe choices for given exchange."""
    if exchange == EXCHANGE_NSE:
        return ["NIFTY50", "NIFTY100", "NIFTY500", "ALL NSE"]
    elif exchange == EXCHANGE_BSE:
        return ["SENSEX30", "BSE100", "BSE_MIDCAP", "BSE_ONLY", "ALL BSE"]
    else:
        return ["NIFTY50+SENSEX30", "NIFTY100+BSE100", "NIFTY500+BSE_MIDCAP", "ALL"]


def get_exchange_from_symbol(symbol: str) -> str:
    """Detect exchange from symbol suffix."""
    if symbol.endswith(".NS"):
        return EXCHANGE_NSE
    elif symbol.endswith(".BO"):
        return EXCHANGE_BSE
    return EXCHANGE_NSE


def symbol_to_display(symbol: str) -> str:
    """Strip exchange suffix for display."""
    if not symbol:
        return ""
    return symbol.replace(".NS", "").replace(".BO", "")


def fetch_nse_equity_list() -> pd.DataFrame:
    """Try to fetch NSE equity list from NSE website."""
    try:
        url = "https://nsearchives.nseindia.com/content/equities/EQUITY_L.csv"
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.5",
            "Accept-Encoding": "gzip, deflate",
            "Connection": "keep-alive",
        }
        resp = requests.get(url, headers=headers, timeout=10)
        if resp.status_code == 200:
            from io import StringIO
            df = pd.read_csv(StringIO(resp.text))
            return df
    except Exception as e:
        logger.warning(f"NSE equity list fetch failed: {e}")

    # Fallback: return built-in list
    all_symbols = NIFTY50_SYMBOLS + NIFTY100_EXTRA + NIFTY500_EXTRA
    return pd.DataFrame({
        "SYMBOL": list(set(all_symbols)),
        "NAME OF COMPANY": [""] * len(set(all_symbols)),
        "SERIES": ["EQ"] * len(set(all_symbols)),
        "EXCHANGE": ["NSE"] * len(set(all_symbols)),
    })


def fetch_bse_equity_list() -> pd.DataFrame:
    """Return BSE equity list (built-in, with BSE-only extras marked)."""
    all_bse = SENSEX30_SYMBOLS + BSE100_EXTRA + BSE_MIDCAP_EXTRA
    bse_only = BSE_ONLY_SYMBOLS

    rows = []
    for s in set(all_bse):
        rows.append({"SYMBOL": s, "NAME OF COMPANY": "", "EXCHANGE": "BSE", "BSE_ONLY": False})
    for s in bse_only:
        rows.append({"SYMBOL": s, "NAME OF COMPANY": "", "EXCHANGE": "BSE", "BSE_ONLY": True})

    return pd.DataFrame(rows)


def safe_float(val, default=0.0) -> float:
    try:
        v = float(val)
        return v if not np.isnan(v) else default
    except Exception:
        return default


def format_currency(value: float) -> str:
    if value >= 1e7:
        return f"₹{value/1e7:.2f}Cr"
    elif value >= 1e5:
        return f"₹{value/1e5:.2f}L"
    else:
        return f"₹{value:.2f}"


def format_number(value: float) -> str:
    if value >= 1e9:
        return f"{value/1e9:.2f}B"
    elif value >= 1e6:
        return f"{value/1e6:.2f}M"
    elif value >= 1e3:
        return f"{value/1e3:.2f}K"
    else:
        return f"{value:.2f}"


def retry_request(func, retries=3, delay=1.0):
    """Retry a function call with exponential backoff."""
    for attempt in range(retries):
        try:
            return func()
        except Exception as e:
            if attempt < retries - 1:
                time.sleep(delay * (2 ** attempt))
            else:
                raise e


def chunk_list(lst: list, chunk_size: int) -> list:
    """Split list into chunks."""
    return [lst[i:i + chunk_size] for i in range(0, len(lst), chunk_size)]


def calculate_returns(prices: pd.Series) -> pd.Series:
    return prices.pct_change().dropna()


def calculate_max_drawdown(equity_curve: pd.Series) -> float:
    if equity_curve.empty:
        return 0.0
    peak = equity_curve.cummax()
    drawdown = (equity_curve - peak) / peak
    return float(drawdown.min())


SECTOR_MAP = {
    "RELIANCE": "Energy", "ONGC": "Energy", "BPCL": "Energy", "IOC": "Energy",
    "HDFCBANK": "Banking", "ICICIBANK": "Banking", "SBIN": "Banking", "AXISBANK": "Banking",
    "KOTAKBANK": "Banking", "INDUSINDBK": "Banking", "BANDHANBNK": "Banking",
    "TCS": "IT", "INFY": "IT", "WIPRO": "IT", "HCLTECH": "IT", "TECHM": "IT",
    "LTIM": "IT", "MPHASIS": "IT", "COFORGE": "IT", "PERSISTENT": "IT",
    "SUNPHARMA": "Pharma", "DRREDDY": "Pharma", "CIPLA": "Pharma", "DIVISLAB": "Pharma",
    "APOLLOHOSP": "Healthcare", "FORTIS": "Healthcare", "MAXHEALTH": "Healthcare",
    "TATASTEEL": "Metals", "JSWSTEEL": "Metals", "HINDALCO": "Metals", "SAIL": "Metals",
    "TATAMOTORS": "Auto", "MARUTI": "Auto", "M&M": "Auto", "BAJAJ-AUTO": "Auto",
    "BHARTIARTL": "Telecom", "IDEA": "Telecom",
    "NESTLEIND": "FMCG", "HINDUNILVR": "FMCG", "ITC": "FMCG", "BRITANNIA": "FMCG",
    "TITAN": "Consumer", "ASIANPAINT": "Consumer",
    "LT": "Capital Goods", "SIEMENS": "Capital Goods",
    "ADANIPORTS": "Infrastructure", "CONCOR": "Infrastructure",
    "POWERGRID": "Power", "NTPC": "Power", "ADANIGREEN": "Power",
}


def get_sector(symbol: str) -> str:
    base = symbol.replace(".NS", "").replace(".BO", "")
    return SECTOR_MAP.get(base, "Others")

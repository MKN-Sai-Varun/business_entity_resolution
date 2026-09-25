import re, unicodedata
import pandas as pd

LEGAL_SUFFIXES = [
    "incorporated", "inc", "corporation", "corp", "limited", "ltd",
    "llc", "l l c", "private", "pvt", "plc", "company", "co",
    "llp", "pty", "gmbh", "sarl", "sas",
]
_LEGAL_SUFFIX_RE = re.compile(
    r"\b(" + "|".join(sorted(LEGAL_SUFFIXES, key=len, reverse=True)) + r")\b\.?", re.IGNORECASE
)
STREET_ABBR = {
    "rd": "road", "st": "street", "ave": "avenue", "blvd": "boulevard",
    "dr": "drive", "ln": "lane", "ct": "court", "pl": "place",
    "hwy": "highway", "apt": "apartment", "flr": "floor", "bldg": "building",
}
_NUMERIC_RE = re.compile(r"\d+")
_POSTAL_RE = re.compile(r"\b\d{4,6}\b")

def _strip_accents(text):
    nfkd = unicodedata.normalize("NFKD", text)
    return "".join(c for c in nfkd if not unicodedata.combining(c))

def basic_clean(text: str) -> str:
    text = (text or "").lower().strip()
    text = _strip_accents(text)
    text = text.replace("&", " and ")
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def normalize_name_row(raw):
    cleaned = basic_clean(raw)
    no_suffix = re.sub(r"\s+", " ", _LEGAL_SUFFIX_RE.sub("", cleaned)).strip()
    tokens = no_suffix.split()
    return pd.Series({
        "name_clean": cleaned,
        "name_no_suffix": no_suffix,
        "name_compact": cleaned.replace(" ", ""),
        "name_sorted_tokens": " ".join(sorted(tokens)),
        "name_first_token": tokens[0] if tokens else "",
    })

def normalize_addr_row(raw):
    cleaned = basic_clean(raw)
    tokens = cleaned.split()
    expanded = " ".join(STREET_ABBR.get(t, t) for t in tokens)
    numeric_tokens = _NUMERIC_RE.findall(cleaned)
    postal_candidates = _POSTAL_RE.findall(cleaned)
    addr_postal = postal_candidates[-1] if postal_candidates else ""
    house_number = numeric_tokens[0] if numeric_tokens else ""
    return pd.Series({
        "addr_clean": expanded,
        "addr_compact": cleaned.replace(" ", ""),
        "addr_house_no": house_number,
        "addr_postal": addr_postal,
    })

def normalize_df(df: pd.DataFrame) -> pd.DataFrame:
    name_feats = df["business_name"].apply(normalize_name_row)
    addr_feats = df["business_address"].apply(normalize_addr_row)
    out = pd.concat([df[["entity_id", "country"]], name_feats, addr_feats], axis=1)
    return out

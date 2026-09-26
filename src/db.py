import duckdb
from src.config import DB_PATH

def get_connection():
    """Persistent, disk-backed DuckDB connection — never :memory: for this pipeline."""
    con = duckdb.connect(str(DB_PATH))
    con.execute("PRAGMA memory_limit='20GB'")   # leave headroom below instance RAM
    con.execute("PRAGMA threads=8")
    return con
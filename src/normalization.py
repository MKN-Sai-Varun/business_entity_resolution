from src.db import get_connection
from src.config import TRAIN_DIR, TEST_DIR, SAMPLE_ROWS

NORMALIZE_SQL = """
CREATE OR REPLACE TABLE {table} AS
SELECT
    entity_id,
    business_name AS name_raw,
    business_address AS address_raw,
    country,
    trim(regexp_replace(
        regexp_replace(lower(business_name), '&', ' and ', 'g'),
        '[^a-z0-9 ]', ' ', 'g'
    )) AS name_norm,
    trim(regexp_replace(
        trim(regexp_replace(
            regexp_replace(lower(business_name), '&', ' and ', 'g'),
            '[^a-z0-9 ]', ' ', 'g'
        )),
        '\\b(inc|incorporated|ltd|limited|llc|corp|corporation|pvt|private|co)\\b', '', 'g'
    )) AS name_core,
    trim(regexp_replace(lower(business_address), '[^a-z0-9 ]', ' ', 'g')) AS address_norm,
    regexp_extract(business_address, '^\\s*([0-9]+)', 1) AS house_number,
    regexp_extract(business_address, '([0-9]{{5,6}})', 1) AS postal_code
FROM read_csv('{path}', delim='\t', header=true, quote='')
{limit_clause}
"""

def _limit_clause():
    return f"LIMIT {SAMPLE_ROWS}" if SAMPLE_ROWS > 0 else ""

def build_normalized_tables():
    con = get_connection()
    limit = _limit_clause()
    con.execute(NORMALIZE_SQL.format(table="s1_norm", path=str(TRAIN_DIR / "train_source1.tsv"), limit_clause=limit))
    con.execute(NORMALIZE_SQL.format(table="s2_norm", path=str(TRAIN_DIR / "train_source2.tsv"), limit_clause=limit))
    con.execute(NORMALIZE_SQL.format(table="s3_norm", path=str(TRAIN_DIR / "train_source3.tsv"), limit_clause=limit))
    print("Row counts:", con.execute(
        "SELECT (SELECT count(*) FROM s1_norm), (SELECT count(*) FROM s2_norm), (SELECT count(*) FROM s3_norm)"
    ).fetchone())
    con.close()

def build_normalized_test_tables():
    con = get_connection()
    limit = _limit_clause()
    con.execute(NORMALIZE_SQL.format(table="s1_norm_test", path=str(TEST_DIR / "test_source1.tsv"), limit_clause=limit))
    con.execute(NORMALIZE_SQL.format(table="s2_norm_test", path=str(TEST_DIR / "test_source2.tsv"), limit_clause=limit))
    con.execute(NORMALIZE_SQL.format(table="s3_norm_test", path=str(TEST_DIR / "test_source3.tsv"), limit_clause=limit))
    con.close()
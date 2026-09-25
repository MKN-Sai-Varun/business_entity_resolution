import sys, csv
from pathlib import Path

def read_tsv(path, id_col, list_col):
    rows = {}
    with open(path, newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for r in reader:
            rows[r[id_col]] = [x for x in r[list_col].split(",") if x]
    return rows

def main(matching_path, candidate_path, test_dir):
    test_dir = Path(test_dir)
    s1_ids = set()
    with open(test_dir / "test_source1.tsv") as f:
        next(f)
        for line in f: s1_ids.add(line.split("\t")[0])
    s2_ids, s3_ids = set(), set()
    with open(test_dir / "test_source2.tsv") as f:
        next(f)
        for line in f: s2_ids.add(line.split("\t")[0])
    with open(test_dir / "test_source3.tsv") as f:
        next(f)
        for line in f: s3_ids.add(line.split("\t")[0])
    valid_match_ids = s2_ids | s3_ids

    matching = read_tsv(matching_path, "source1_entity_id", "matched_entity_ids")
    candidates = read_tsv(candidate_path, "source1_entity_id", "candidate_entity_ids")

    issues = []
    if set(matching.keys()) != s1_ids:
        issues.append(f"S1 coverage mismatch: missing={len(s1_ids-set(matching))}, extra={len(set(matching)-s1_ids)}")
    for eid, ids in matching.items():
        if len(ids) != len(set(ids)):
            issues.append(f"{eid}: duplicate IDs")
        bad = [i for i in ids if i not in valid_match_ids]
        if bad: issues.append(f"{eid}: invalid IDs {bad}")
        cand_set = set(candidates.get(eid, []))
        missing = [i for i in ids if i not in cand_set]
        if missing: issues.append(f"{eid}: matched not in candidates {missing}")

    if issues:
        print(f"{len(issues)} ISSUES:")
        for i in issues[:50]: print(" -", i)
        sys.exit(1)
    print("PASS (local checks)")

if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3])

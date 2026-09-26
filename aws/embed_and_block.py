import subprocess, sys
subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                 "sentence-transformers", "faiss-cpu", "duckdb", "pyarrow"])

import argparse, os
import duckdb
import numpy as np
import pandas as pd
import faiss
from sentence_transformers import SentenceTransformer

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"  # Apache-2.0, 22M params

def encode_table(model, con, table, batch_size=512):
    df = con.execute(
        f"SELECT entity_id, country, name_norm || ' ' || address_norm AS text FROM {table}"
    ).fetchdf()
    emb = model.encode(
        df["text"].fillna("").tolist(),
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,   # inner product == cosine similarity
    )
    return df[["entity_id", "country"]].reset_index(drop=True), emb.astype("float32")

def build_ann_candidates(s1_meta, s1_emb, other_meta, other_emb, k=15):
    """Partitioned by whatever countries are actually present -- never hardcoded
    to {US, India}, so France is handled automatically."""
    s1_ids_arr = s1_meta["entity_id"].to_numpy()
    other_ids_arr = other_meta["entity_id"].to_numpy()

    chunks = []
    countries = set(s1_meta["country"]) & set(other_meta["country"])
    for country in countries:
        s1_idx = s1_meta.index[s1_meta["country"] == country].to_numpy()
        other_idx = other_meta.index[other_meta["country"] == country].to_numpy()
        if len(s1_idx) == 0 or len(other_idx) == 0:
            continue

        index = faiss.IndexFlatIP(other_emb.shape[1])
        index.add(other_emb[other_idx])
        sims, nn = index.search(s1_emb[s1_idx], min(k, len(other_idx)))

        # nn/sims are shape (len(s1_idx), k_actual); flatten row-major (default 'C'
        # order for both np.repeat and .flatten()) so the three arrays stay aligned
        # element-for-element without any Python-level loop.
        s1_ids_flat = np.repeat(s1_ids_arr[s1_idx], nn.shape[1])
        other_ids_flat = other_ids_arr[other_idx][nn.flatten()]
        sims_flat = sims.flatten()

        chunks.append(pd.DataFrame({
            "s1_id": s1_ids_flat,
            "other_id": other_ids_flat,
            "cosine_sim": sims_flat,
        }))

    if not chunks:
        return pd.DataFrame(columns=["s1_id", "other_id", "cosine_sim"])
    return pd.concat(chunks, ignore_index=True)

def main(db_path, out_dir, split):
    con = duckdb.connect(db_path)
    model = SentenceTransformer(MODEL_NAME, device="cuda")

    suffix = "" if split == "train" else "_test"
    s1_meta, s1_emb = encode_table(model, con, f"s1_norm{suffix}")
    s2_meta, s2_emb = encode_table(model, con, f"s2_norm{suffix}")
    s3_meta, s3_emb = encode_table(model, con, f"s3_norm{suffix}")

    ann_s2 = build_ann_candidates(s1_meta, s1_emb, s2_meta, s2_emb)
    ann_s3 = build_ann_candidates(s1_meta, s1_emb, s3_meta, s3_emb)

    os.makedirs(out_dir, exist_ok=True)
    ann_s2.to_parquet(f"{out_dir}/ann_candidates_s2_{split}.parquet", index=False)
    ann_s3.to_parquet(f"{out_dir}/ann_candidates_s3_{split}.parquet", index=False)
    print(f"ANN candidates: S2={len(ann_s2):,}  S3={len(ann_s3):,}")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--db-path", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--split", choices=["train", "test"], required=True)
    a = p.parse_args()
    main(a.db_path, a.out_dir, a.split)
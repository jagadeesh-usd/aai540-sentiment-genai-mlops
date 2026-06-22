
import pandas as pd
import numpy as np
import os
import sys
import subprocess
import time

print("=" * 60)
print("CX CORTALYST — Pipeline Preprocessing Step")
print("=" * 60)

# ── Step 0: Install dependencies (Python 3.10 container) ──────
print("\nStep 0: Installing dependencies...")
start = time.time()

# subprocess.check_call([
#     sys.executable, "-m", "pip", "install",
#     "sentence-transformers==5.1.2",  # ← works on Python 3.10
#     "safetensors==0.7.0",
#     "--quiet",
#     "--no-cache-dir"
# ])

# Force install older, stable versions compatible with the container's built-in PyTorch 2.0.1
subprocess.check_call([sys.executable, "-m", "pip", "install", "sentence-transformers==2.2.2", "transformers==4.30.2"])

print(f"✅ Dependencies installed ({time.time()-start:.0f}s)")

from sentence_transformers import SentenceTransformer
print("✅ SentenceTransformer imported successfully")

# ── Step 1: Load data ─────────────────────────────────────────
print("\nStep 1: Loading parquet splits from S3...")

df_train = pd.read_parquet("/opt/ml/processing/input/train/")
df_val   = pd.read_parquet("/opt/ml/processing/input/validation/")
df_test  = pd.read_parquet("/opt/ml/processing/input/test/")

print(f"✅ Loaded — Train: {len(df_train):,} | Val: {len(df_val):,} | Test: {len(df_test):,}")
print(f"   Columns: {df_train.columns.tolist()}")

# ── Step 2: Validate columns ──────────────────────────────────
print("\nStep 2: Validating columns...")
for col in ["text", "sentiment_label"]:
    if col not in df_train.columns:
        raise ValueError(f"Required column {col} not found!")
    print(f"   OK: {col}")

neg = (df_train["sentiment_label"]==0).sum()
pos = (df_train["sentiment_label"]==1).sum()
print(f"   Class balance — Neg: {neg:,} | Pos: {pos:,}")

# ── Step 3: Stratified sample ─────────────────────────────────
print("\nStep 3: Stratified sampling (2k per class = 4k total)...")

TARGET_PER_CLASS = 2000

def stratified_sample(df, n_per_class, seed=42):
    neg_avail = (df["sentiment_label"]==0).sum()
    pos_avail = (df["sentiment_label"]==1).sum()
    actual_n  = min(n_per_class, neg_avail, pos_avail)
    sampled = df.groupby("sentiment_label", group_keys=False)        .apply(lambda x: x.sample(min(actual_n, len(x)), random_state=seed))        .reset_index(drop=True)
    print(f"   {len(sampled):,} rows | Neg: {(sampled['sentiment_label']==0).sum():,} | Pos: {(sampled['sentiment_label']==1).sum():,}")
    return sampled

print("   Train:"); df_train = stratified_sample(df_train, TARGET_PER_CLASS)
print("   Val:");   df_val   = stratified_sample(df_val,   TARGET_PER_CLASS // 4)
print("   Test:");  df_test  = stratified_sample(df_test,  TARGET_PER_CLASS // 4)

print(f"✅ Sampling complete")
print(f"   Train: {len(df_train):,} | Val: {len(df_val):,} | Test: {len(df_test):,}")

# ── Step 4: Generate embeddings ───────────────────────────────
print("\nStep 4: Loading embedding model (all-MiniLM-L6-v2)...")
embedder = SentenceTransformer("all-MiniLM-L6-v2")
print("✅ Model loaded")

def embed(df, name):
    print(f"   Embedding {name} ({len(df):,} records)...")
    start = time.time()
    emb = embedder.encode(
        df["text"].fillna("").astype(str).tolist(),
        batch_size=16,
        show_progress_bar=False,
        convert_to_numpy=True
    )
    print(f"   ✅ {name}: {emb.shape} ({time.time()-start:.0f}s)")
    return pd.DataFrame(emb, columns=[f"emb_{i}" for i in range(emb.shape[1])])

train_emb = embed(df_train, "Train")
val_emb   = embed(df_val,   "Val")
test_emb  = embed(df_test,  "Test")

# ── Step 5: Build feature matrix ─────────────────────────────
print("\nStep 5: Building feature matrix...")

STRUCTURED = ["text_char_length","is_elite","review_useful_votes",
              "review_funny_votes","review_cool_votes"]

def build(df, emb, name):
    available = [c for c in STRUCTURED if c in df.columns]
    y   = df["sentiment_label"].astype(int).reset_index(drop=True)
    X_s = df[available].fillna(0).reset_index(drop=True)
    X_e = emb.reset_index(drop=True)
    out = pd.concat([y, X_s, X_e], axis=1)
    print(f"   ✅ {name}: {out.shape} (1 label + {len(available)} structured + {X_e.shape[1]} embeddings)")
    return out

train_xgb = build(df_train, train_emb, "Train")
val_xgb   = build(df_val,   val_emb,   "Val")
test_xgb  = build(df_test,  test_emb,  "Test")

# ── Step 6: Write outputs ─────────────────────────────────────
print("\nStep 6: Writing CSVs...")

os.makedirs("/opt/ml/processing/output/train",      exist_ok=True)
os.makedirs("/opt/ml/processing/output/validation", exist_ok=True)
os.makedirs("/opt/ml/processing/output/test",       exist_ok=True)

train_xgb.to_csv("/opt/ml/processing/output/train/train.csv",
                 header=False, index=False)
val_xgb.to_csv(  "/opt/ml/processing/output/validation/validation.csv",
                 header=False, index=False)
test_xgb.to_csv( "/opt/ml/processing/output/test/test.csv",
                 header=False, index=False)

for path, name in [
    ("/opt/ml/processing/output/train/train.csv",           "Train"),
    ("/opt/ml/processing/output/validation/validation.csv", "Val"),
    ("/opt/ml/processing/output/test/test.csv",             "Test")
]:
    size = os.path.getsize(path) / 1024 / 1024
    print(f"   ✅ {name}: {size:.1f} MB")

print("\n" + "=" * 60)
print("PREPROCESSING COMPLETE")
print(f"  Train: {len(train_xgb):,} x {train_xgb.shape[1]} cols")
print(f"  Val:   {len(val_xgb):,} x {val_xgb.shape[1]} cols")
print(f"  Test:  {len(test_xgb):,} x {test_xgb.shape[1]} cols")
print(f"  Features: 1 label + 5 structured + 384 embeddings = 390 total")
print("=" * 60)

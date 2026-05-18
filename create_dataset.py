"""
create_dataset.py
==================
Cleans the raw dirty CSV and generates a properly labelled fraud dataset.

Usage:
    python create_dataset.py
    python create_dataset.py --input path/to/raw.csv --output clean_dataset.csv

Input columns expected:
    Transaction_ID, Transaction_Date, Customer_ID, Product_Name,
    Quantity, Price, Payment_Method, Transaction_Status

Output adds:
    NegativePrice, NegativeQty, Amount, BadDate  (extra features)
    Class  (0 = Normal, 1 = Fraud)
"""

import re
import argparse
import pandas as pd
import numpy as np

# ── normalisation maps ────────────────────────────────────────────────────

PM_MAP = {
    "paypal":       "PayPal",
    "pay pal":      "PayPal",
    "paypal ":      "PayPal",
    "pay  pal":     "PayPal",
    "creditcard":   "CreditCard",
    "credit card":  "CreditCard",
    "credit  card": "CreditCard",
    "debitcard":    "CreditCard",
    "debit card":   "CreditCard",
    "cash":         "Cash",
    "crypto":       "Crypto",
    "bitcoin":      "Crypto",
    "unknown":      "Unknown",
    "n/a":          "Unknown",
    "":             "Unknown",
}

STATUS_MAP = {
    "completed": "Completed",
    "complete":  "Completed",
    "failed":    "Failed",
    "fail":      "Failed",
    "pending":   "Pending",
    "0":         "Unknown",
    "":          "Unknown",
}

PRODUCTS = [
    "Coffee Machine",
    "Headphones",
    "Smartphone",
    "Laptop",
    "Tablet",
]



def normalise_payment(v) -> str:
    key = str(v).strip().lower()
    return PM_MAP.get(key, str(v).strip())


def normalise_status(v) -> str:
    key = str(v).strip().lower()
    return STATUS_MAP.get(key, str(v).strip().capitalize())


def repair_product(v) -> str:
    s = str(v).strip()
    sl = s.lower()
    for p in PRODUCTS:
     
        if p.lower().startswith(sl):
            return p
    return s


def clean_price(v) -> float:
    cleaned = re.sub(r"[^\d.\-]", "", str(v))
    try:
        return float(cleaned)
    except ValueError:
        return 0.0


def is_bad_date(s) -> int:
    try:
        parts = str(s).strip().split("-")
        if len(parts) != 3:
            return 1
        y, m, d = int(parts[0]), int(parts[1]), int(parts[2])
        if not (1 <= m <= 12):
            return 1
        if not (1 <= d <= 31):
            return 1
        if m == 2 and d > 28:
            return 1
        if m in (4, 6, 9, 11) and d > 30:
            return 1
        if not (2000 <= y <= 2030):
            return 1
        return 0
    except Exception:
        return 1



def build_dataset(input_path: str, output_path: str) -> pd.DataFrame:
    print(f"\n{'='*55}")
    print(f"  Building fraud dataset")
    print(f"{'='*55}")
    print(f"  Input  : {input_path}")

    df = pd.read_csv(input_path)
    print(f"  Rows   : {len(df):,}")
    print(f"  Cols   : {df.columns.tolist()}")

    raw_price = df["Price"].apply(clean_price)
    raw_qty   = pd.to_numeric(df["Quantity"], errors="coerce").fillna(0)

    df["NegativePrice"] = (raw_price < 0).astype(int)
    df["NegativeQty"]   = (raw_qty   < 0).astype(int)
    df["Price"]         = raw_price.abs()
    df["Quantity"]      = raw_qty.abs()
    df["Amount"]        = df["Price"] * df["Quantity"]

    df["Payment_Method"]     = df["Payment_Method"].apply(normalise_payment)
    df["Transaction_Status"] = df["Transaction_Status"].apply(normalise_status)
    df["Product_Name"]       = df["Product_Name"].apply(repair_product)

    df["BadDate"] = df["Transaction_Date"].apply(is_bad_date)

    missing = df["Transaction_ID"].isna() | (df["Transaction_ID"].astype(str).str.strip() == "")
    n_miss  = missing.sum()
    if n_miss > 0:
        df.loc[missing, "Transaction_ID"] = [f"GEN{i:06d}" for i in range(n_miss)]
        print(f"  Filled {n_miss} missing Transaction_IDs")

    p90_price = df["Price"].quantile(0.90)
    p90_qty   = df["Quantity"].quantile(0.90)
    print(f"\n  P90 price    : {p90_price:.2f}")
    print(f"  P90 quantity : {p90_qty:.2f}")

    df["Class"] = 0
    df.loc[df["Price"]    > p90_price,                           "Class"] = 1
    df.loc[df["Quantity"] > p90_qty,                             "Class"] = 1
    df.loc[df["Payment_Method"].isin(["Crypto", "Unknown"]),     "Class"] = 1
    df.loc[df["Transaction_Status"] == "Failed",                 "Class"] = 1
    df.loc[df["BadDate"]       == 1,                             "Class"] = 1
    df.loc[df["NegativePrice"] == 1,                             "Class"] = 1
    df.loc[df["NegativeQty"]   == 1,                             "Class"] = 1

    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    print(f"\n  Class distribution:")
    vc = df["Class"].value_counts()
    print(f"    Normal (0): {vc.get(0, 0):,}")
    print(f"    Fraud  (1): {vc.get(1, 0):,}")
    print(f"    Fraud rate: {df['Class'].mean()*100:.1f}%")

    df.to_csv(output_path, index=False)
    print(f"\n  ✅  Saved → {output_path}")
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",  default="dataset/dirty_financial_transactions.csv")
    parser.add_argument("--output", default="realistic_fraud_dataset.csv")
    args = parser.parse_args()
    build_dataset(args.input, args.output)

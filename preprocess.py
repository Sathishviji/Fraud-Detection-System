"""
preprocess.py
Shared feature engineering pipeline used by train.py AND api.py.
Both must use the SAME scaler/PCA — saved in preprocessor.pkl after training.
"""

import re
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

FEATURE_COLS = [
    "Price", "Quantity", "Amount",
    "LogPrice", "LogQty", "LogAmount",
    "PricePerUnit",
    "WasNegativePrice", "WasNegativeQty",
    "Year", "Month", "MonthSin", "MonthCos",
    "PaymentRisk", "StatusRisk",
    "BadDate", "TruncName",
]

N_PCA = 12   

_PM_RISK = {
    "cash": 0,
    "paypal": 1, "pay pal": 1,
    "credit card": 2, "creditcard": 2, "credit  card": 2,
    "debit card": 2, "debitcard": 2,
    "bank transfer": 1,
    "unknown": 4,
    "crypto": 5,
}

_STATUS_RISK = {
    "completed": 0, "complete": 0,
    "pending": 1,
    "failed": 3,
    "0": 4,
    "": 4,
}

_PRODUCTS = [
    "Coffee Machine", "Headphones", "Smartphone", "Laptop", "Tablet",
]


def _to_float(v) -> float:
    """Strip currency symbols and parse to float."""
    cleaned = re.sub(r"[^\d.\-]", "", str(v))
    try:
        return float(cleaned)
    except ValueError:
        return 0.0


def _payment_risk(v: str) -> int:
    key = str(v).strip().lower()
    return _PM_RISK.get(key, 2)


def _status_risk(v: str) -> int:
    key = str(v).strip().lower()
    return _STATUS_RISK.get(key, 2)


def _bad_date(s: str) -> int:
    """Returns 1 for impossible dates like 2025-02-30 or 2023-13-01."""
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


def _trunc_name(v: str) -> int:
    """Returns 1 if product name looks truncated (length ≤ 5 chars)."""
    return 1 if 0 < len(str(v).strip()) <= 5 else 0


def _parse_year_month(s: str):
    try:
        parts = str(s).strip().split("-")
        y = int(parts[0]) if len(parts) >= 1 else 2020
        m = int(parts[1]) if len(parts) >= 2 else 1
        m = max(1, min(12, m))   
        return y, m
    except Exception:
        return 2020, 1




def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply feature engineering to raw dataframe.
    Returns a new dataframe with FEATURE_COLS columns.
    Does NOT require any fitted objects — pure transformations.
    """
    df = df.copy()

    
    raw_price = df["Price"].apply(_to_float) if "Price" in df.columns else pd.Series(np.zeros(len(df)))
    raw_qty   = pd.to_numeric(df["Quantity"], errors="coerce").fillna(0) if "Quantity" in df.columns else pd.Series(np.zeros(len(df)))

    was_neg_price = (raw_price < 0).astype(int)
    was_neg_qty   = (raw_qty   < 0).astype(int)

    price = raw_price.abs()
    qty   = raw_qty.abs()
    amt   = price * qty

    
    safe_qty      = qty.replace(0, np.nan)
    price_per_unit = (price / safe_qty).fillna(0)

   
    log_price  = np.log1p(price)
    log_qty    = np.log1p(qty)
    log_amount = np.log1p(amt)

  
    date_col = df["Transaction_Date"] if "Transaction_Date" in df.columns else pd.Series(["2020-01-01"] * len(df))
    bad_date = date_col.apply(_bad_date)
    ym       = date_col.apply(_parse_year_month)
    years    = ym.apply(lambda t: t[0])
    months   = ym.apply(lambda t: t[1])
    month_sin = np.sin(2 * np.pi * months / 12)
    month_cos = np.cos(2 * np.pi * months / 12)

  
    pm_col     = df["Payment_Method"] if "Payment_Method" in df.columns else pd.Series([""] * len(df))
    status_col = df["Transaction_Status"] if "Transaction_Status" in df.columns else pd.Series([""] * len(df))
    prod_col   = df["Product_Name"] if "Product_Name" in df.columns else pd.Series(["x"] * len(df))

    pay_risk    = pm_col.apply(_payment_risk)
    status_risk = status_col.apply(_status_risk)
    trunc_name  = prod_col.apply(_trunc_name)

    out = pd.DataFrame({
        "Price":            price.values,
        "Quantity":         qty.values,
        "Amount":           amt.values,
        "LogPrice":         log_price.values,
        "LogQty":           log_qty.values,
        "LogAmount":        log_amount.values,
        "PricePerUnit":     price_per_unit.values,
        "WasNegativePrice": was_neg_price.values,
        "WasNegativeQty":   was_neg_qty.values,
        "Year":             years.values,
        "Month":            months.values,
        "MonthSin":         month_sin.values,
        "MonthCos":         month_cos.values,
        "PaymentRisk":      pay_risk.values,
        "StatusRisk":       status_risk.values,
        "BadDate":          bad_date.values,
        "TruncName":        trunc_name.values,
    })

    return out[FEATURE_COLS]


def build_features(df: pd.DataFrame, fit: bool = True, preprocessor: dict = None):
    """
    Engineer features, then scale + PCA.

    fit=True  → fits scaler & PCA (call during training, save result)
    fit=False → transforms using preprocessor dict (call during inference)

    Returns: (X_pca: np.ndarray, preprocessor: dict)
    """
    X = engineer_features(df).fillna(0).astype('float32')

    if fit:
        scaler = StandardScaler()
        X_sc   = scaler.fit_transform(X)

        n      = min(N_PCA, X.shape[1])
        pca    = PCA(n_components=n, random_state=42)
        X_pca  = pca.fit_transform(X_sc)

        var_exp = pca.explained_variance_ratio_.sum() * 100
        print(f"  PCA: {n} components, {var_exp:.1f}% variance explained")

        preprocessor = {
            "scaler":       scaler,
            "pca":          pca,
            "feature_cols": FEATURE_COLS,
            "n_components": n,
        }
    else:
        
        X_sc  = preprocessor["scaler"].transform(X)
        X_pca = preprocessor["pca"].transform(X_sc)

    return X_pca, preprocessor

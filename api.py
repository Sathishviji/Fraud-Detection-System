"""
api.py
=======
FraudSentinel FastAPI Backend

Endpoints:
  GET  /health              → model status + info
  POST /upload              → predict on uploaded CSV
  GET  /evaluate            → full evaluation on training dataset
  GET  /export/csv          → download last predictions as CSV
  GET  /export/json         → download last predictions as JSON

Run:
    cd backend
    uvicorn api:app --host 0.0.0.0 --port 8000 --reload

Then open:  frontend/index.html
"""

import io
import json
import os
import joblib
import numpy as np
import pandas as pd
import torch

from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from sklearn.metrics import (
    f1_score, roc_auc_score, confusion_matrix,
    classification_report, precision_score, recall_score,
)

from model import RNN_SGRU
from preprocess import build_features

MODEL_FILE = "fraud_rnn_sgru_model.pth"
PREP_FILE  = "preprocessor.pkl"
EVAL_CSV   = "realistic_fraud_dataset.csv"


def _load_model():
    for p in (MODEL_FILE, PREP_FILE):
        if not os.path.exists(p):
            raise FileNotFoundError(
                f"\n\nFile not found: '{p}'\n"
                f"You must train the model first!\n\n"
                f"Run:  python train.py\n"
            )

    prep   = joblib.load(PREP_FILE)
    n_feat = prep["n_features"]
    cfg    = prep.get("model_cfg", {})

    m = RNN_SGRU(
        input_size  = n_feat,
        hidden_size = cfg.get("hidden_size", 128),
        num_layers  = cfg.get("num_layers",  2),
        dropout     = cfg.get("dropout",     0.3),
    )
    m.load_state_dict(torch.load(MODEL_FILE, map_location="cpu"))
    m.eval()
    print(f"✅  Model loaded  ({n_feat} features)")
    return m, prep


model, preprocessor = _load_model()

app = FastAPI(title="FraudSentinel", version="2.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_last_results: list = []


def _preprocess(df: pd.DataFrame) -> np.ndarray:
    """Transform raw dataframe using the SAVED preprocessor (never re-fit)."""
    X_pca, _ = build_features(df, fit=False, preprocessor=preprocessor)
    return X_pca


def _predict(X: np.ndarray, threshold: float = 0.5):
    """Run model inference. Returns (probs list, labels list)."""
    tensor = torch.tensor(X, dtype=torch.float32)
    with torch.no_grad():
        logits = model(tensor)
        probs  = torch.sigmoid(logits).squeeze().numpy()

    if probs.ndim == 0:
        probs = np.array([float(probs)])

    labels = ["Fraud" if float(p) >= threshold else "Normal" for p in probs]
    return probs.tolist(), labels



@app.get("/health")
def health():
    return {
        "status":     "ok",
        "model_file": MODEL_FILE,
        "prep_file":  PREP_FILE,
        "n_features": preprocessor["n_features"],
    }


@app.post("/upload")
async def upload(
    file:      UploadFile = File(...),
    threshold: float      = Query(0.5,   ge=0.1, le=0.9),
    max_rows:  int        = Query(1000,  ge=1,   le=20000),
):
    """
    Upload a CSV file and receive fraud predictions.

    Query params:
      threshold  float  0.1–0.9   fraud decision boundary  (default 0.5)
      max_rows   int    1–20000   max rows to process       (default 5000)
    """
    if not file.filename.lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only .csv files are supported.")

    content = await file.read()
    try:
        df = pd.read_csv(io.BytesIO(content))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not parse CSV: {e}")

    if df.empty:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    has_labels = "Class" in df.columns
    y_true     = df["Class"].head(max_rows).tolist() if has_labels else None
    df_feat    = df.drop(columns=["Class"], errors="ignore").head(max_rows)

    try:
        X = _preprocess(df_feat)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Preprocessing error: {e}")

    probs, labels = _predict(X, threshold)

    # build results list
    RAW_COLS = [
        "Transaction_ID", "Payment_Method", "Transaction_Status",
        "Product_Name", "Price", "Quantity", "Transaction_Date",
    ]
    results = []
    for i, (p, lbl) in enumerate(zip(probs, labels)):
        row = {
            "id":          i + 1,
            "prediction":  lbl,
            "probability": round(float(p), 4),
            "true_label":  int(y_true[i]) if y_true else None,
        }
        for col in RAW_COLS:
            if col in df_feat.columns:
                row[col.lower()] = str(df_feat.iloc[i].get(col, ""))
        results.append(row)

    global _last_results
    _last_results = results

    fraud  = sum(1 for r in results if r["prediction"] == "Fraud")
    normal = len(results) - fraud
    summary = {
        "total":     len(results),
        "fraud":     fraud,
        "normal":    normal,
        "fraud_pct": round(fraud / max(len(results), 1) * 100, 1),
        "threshold": threshold,
    }

    if y_true:
        yt    = [int(v) for v in y_true[:len(labels)]]
        yp    = [1 if l == "Fraud" else 0 for l in labels]
        prb_t = probs[:len(yt)]
        summary["accuracy"]  = round(sum(a == b for a, b in zip(yt, yp)) / max(len(yt), 1), 4)
        summary["f1_score"]  = round(float(f1_score(yt, yp, zero_division=0)), 4)
        summary["precision"] = round(float(precision_score(yt, yp, zero_division=0)), 4)
        summary["recall"]    = round(float(recall_score(yt, yp, zero_division=0)), 4)
        try:
            summary["auc_roc"] = round(float(roc_auc_score(yt, prb_t)), 4)
        except Exception:
            pass

    return {"results": results, "summary": summary}


@app.get("/evaluate")
def evaluate(
    sample_size: int   = Query(10000, ge=100, le=50000),
    threshold:   float = Query(0.5,   ge=0.1, le=0.9),
):
    """Evaluate model performance on the training dataset."""
    if not os.path.exists(EVAL_CSV):
        raise HTTPException(status_code=404, detail=f"Dataset not found: {EVAL_CSV}")

    df = pd.read_csv(EVAL_CSV)
    df = df.sample(min(sample_size, len(df)), random_state=42)

    y   = df["Class"].values
    X   = _preprocess(df.drop(columns=["Class"], errors="ignore"))
    probs, labels = _predict(X, threshold)

    yp  = [1 if l == "Fraud" else 0 for l in labels]
    cm  = confusion_matrix(y, yp).tolist()
    rep = classification_report(
        y, yp, target_names=["Normal", "Fraud"], output_dict=True)

    return {
        "samples":               len(y),
        "threshold":             threshold,
        "accuracy":              round(float(rep["accuracy"]), 4),
        "f1_score":              round(float(f1_score(y, yp, zero_division=0)), 4),
        "precision":             round(float(precision_score(y, yp, zero_division=0)), 4),
        "recall":                round(float(recall_score(y, yp, zero_division=0)), 4),
        "auc_roc":               round(float(roc_auc_score(y, probs)), 4),
        "confusion_matrix":      cm,
        "classification_report": rep,
    }


@app.get("/export/{fmt}")
def export(fmt: str):
    """Download last /upload predictions as CSV or JSON."""
    if not _last_results:
        raise HTTPException(status_code=404,
                            detail="No results to export. Run /upload first.")

    if fmt == "csv":
        buf = io.StringIO()
        pd.DataFrame(_last_results).to_csv(buf, index=False)
        buf.seek(0)
        return StreamingResponse(
            buf, media_type="text/csv",
            headers={"Content-Disposition":
                     "attachment; filename=fraud_predictions.csv"})

    if fmt == "json":
        data = json.dumps(_last_results, indent=2).encode()
        return StreamingResponse(
            io.BytesIO(data), media_type="application/json",
            headers={"Content-Disposition":
                     "attachment; filename=fraud_predictions.json"})

    raise HTTPException(status_code=400, detail="fmt must be 'csv' or 'json'.")

"""
train.py
==========
Trains the RNN-SGRU fraud detection model.

Key fixes over the original broken version:
  ✓ NO random Class override — uses actual labels from CSV
  ✓ Saves scaler+PCA → preprocessor.pkl  (loaded by api.py)
  ✓ BCEWithLogitsLoss + pos_weight for class imbalance
  ✓ Mini-batch training with DataLoader
  ✓ Stratified 70/15/15 train/val/test split
  ✓ Early stopping on AUC-ROC
  ✓ ReduceLROnPlateau scheduler
  ✓ Per-epoch metrics: loss, F1, AUC, precision, recall
  ✓ Final classification report on held-out test set

Usage:
    python train.py
    python train.py --dataset realistic_fraud_dataset.csv --epochs 50
"""

import os
import argparse
import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    f1_score, roc_auc_score, classification_report,
    precision_score, recall_score,
)

from model import RNN_SGRU
from preprocess import build_features


DEFAULTS = dict(
    dataset      = "realistic_fraud_dataset.csv",
    model_out    = "fraud_rnn_sgru_model.pth",
    prep_out     = "preprocessor.pkl",
    epochs       = 60,
    batch_size   = 512,
    lr           = 0.001,
    patience     = 8,
    hidden_size  = 128,
    num_layers   = 2,
    dropout      = 0.3,
    sample       = 0,          
    seed         = 42,
)


def main(cfg):
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    print(f"\n{'='*60}")
    print(f"  FraudSentinel — Training")
    print(f"{'='*60}")


    if not os.path.exists(cfg.dataset):
        raise FileNotFoundError(
            f"\nDataset not found: '{cfg.dataset}'\n"
            f"Run first:  python create_dataset.py\n"
        )

    df = pd.read_csv(cfg.dataset).sample(30000)

    if cfg.sample > 0 and cfg.sample < len(df):
        df = df.sample(cfg.sample, random_state=cfg.seed)
        print(f"  Using sample of {cfg.sample:,} rows")

    print(f"  Total rows  : {len(df):,}")
    print(f"  Fraud rate  : {df['Class'].mean()*100:.1f}%")

    y = df["Class"].values
    df_features = df.drop(columns=["Class"], errors="ignore")

   
    print("\n[ Feature engineering ]")
    X_pca, preprocessor = build_features(df_features, fit=True)
    n_features = X_pca.shape[1]
    print(f"  Feature shape: {X_pca.shape}")

    
    X_tr, X_tmp, y_tr, y_tmp = train_test_split(
        X_pca, y, test_size=0.30, stratify=y, random_state=cfg.seed)
    X_val, X_te, y_val, y_te = train_test_split(
        X_tmp, y_tmp, test_size=0.50, stratify=y_tmp, random_state=cfg.seed)

    print(f"\n  Train : {len(y_tr):,}   Val : {len(y_val):,}   Test : {len(y_te):,}")

    def tensors(X, y):
        return (
            torch.tensor(X, dtype=torch.float32),
            torch.tensor(y, dtype=torch.float32).unsqueeze(1),
        )

    Xt, yt   = tensors(X_tr,  y_tr)
    Xv, yv   = tensors(X_val, y_val)
    Xte, yte = tensors(X_te,  y_te)

    train_loader = DataLoader(
        TensorDataset(Xt, yt),
        batch_size=cfg.batch_size,
        shuffle=True,
        drop_last=False,
    )

    
    model = RNN_SGRU(
        input_size=n_features,
        hidden_size=cfg.hidden_size,
        num_layers=cfg.num_layers,
        dropout=cfg.dropout,
    )


    pos_weight = torch.tensor(
        [(y == 0).sum() / max((y == 1).sum(), 1)],
        dtype=torch.float32,
    )
    print(f"\n  pos_weight   : {pos_weight.item():.2f}")

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", patience=3, factor=0.5, verbose=False)


    best_auc      = 0.0
    best_weights  = None
    patience_ctr  = 0

    header = (f"{'Ep':>4}  {'Train Loss':>11}  {'Val Loss':>9}  "
              f"{'F1':>7}  {'AUC':>7}  {'Prec':>7}  {'Rec':>7}")
    print(f"\n[ Training — max {cfg.epochs} epochs, early-stop patience={cfg.patience} ]\n")
    print(header)
    print("-" * len(header))

    for ep in range(1, cfg.epochs + 1):

       
        model.train()
        train_loss = 0.0
        for xb, yb in train_loader:
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item() * len(xb)
        train_loss /= len(Xt)

      
        model.eval()
        with torch.no_grad():
            val_logits = model(Xv)
            val_loss   = criterion(val_logits, yv).item()
            val_probs  = torch.sigmoid(val_logits).squeeze().numpy()
            val_preds  = (val_probs >= 0.5).astype(int)

        f1   = f1_score(y_val, val_preds, zero_division=0)
        auc  = roc_auc_score(y_val, val_probs)
        prec = precision_score(y_val, val_preds, zero_division=0)
        rec  = recall_score(y_val, val_preds, zero_division=0)

        print(f"{ep:4d}  {train_loss:11.5f}  {val_loss:9.5f}  "
              f"{f1:7.4f}  {auc:7.4f}  {prec:7.4f}  {rec:7.4f}")

        scheduler.step(auc)

        if auc > best_auc:
            best_auc     = auc
            best_weights = {k: v.clone() for k, v in model.state_dict().items()}
            patience_ctr = 0
        else:
            patience_ctr += 1
            if patience_ctr >= cfg.patience:
                print(f"\n  Early stop at epoch {ep}  (best AUC = {best_auc:.4f})")
                break

    model.load_state_dict(best_weights)
    model.eval()
    with torch.no_grad():
        te_logits = model(Xte)
        te_probs  = torch.sigmoid(te_logits).squeeze().numpy()
        te_preds  = (te_probs >= 0.5).astype(int)

    print(f"\n{'='*60}")
    print(f"  Test set evaluation  (best model, AUC={best_auc:.4f})")
    print(f"{'='*60}")
    print(classification_report(
        y_te, te_preds, target_names=["Normal", "Fraud"]))
    print(f"  AUC-ROC : {roc_auc_score(y_te, te_probs):.4f}")

    
    preprocessor["n_features"] = n_features
    preprocessor["model_cfg"]  = {
        "input_size":  n_features,
        "hidden_size": cfg.hidden_size,
        "num_layers":  cfg.num_layers,
        "dropout":     cfg.dropout,
    }

    torch.save(model.state_dict(), cfg.model_out)
    joblib.dump(preprocessor, cfg.prep_out)

    print(f"\n  ✅  Model saved        → {cfg.model_out}")
    print(f"  ✅  Preprocessor saved → {cfg.prep_out}")
    print(f"\n  Next step: uvicorn api:app --port 8000 --reload\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train FraudSentinel")
    for k, v in DEFAULTS.items():
        parser.add_argument(f"--{k}", type=type(v), default=v)
    args = parser.parse_args()
    cfg  = argparse.Namespace(**{**DEFAULTS, **vars(args)})
    main(cfg)

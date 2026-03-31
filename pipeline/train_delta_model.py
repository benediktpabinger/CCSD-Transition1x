"""
Delta model training (ridge regression on PaiNN features).

Loads the 64-dim fingerprints cached by run_curator_selection.py and
the CCSD(T) delta values from ccsdt_singlepoints.json, fits ridge
regression, saves weights for inference.

Usage:
    python train_delta_model.py \
        --round-dir ~/curator_results/round1 \
        --alpha 1.0
"""
import argparse
import json
import os

import numpy as np
import torch
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error
import joblib


def main(args):
    round_dir = args.round_dir

    # Load cached features (N_all x 64) and db_ids from CURATOR step
    features_all = torch.load(os.path.join(round_dir, 'features.pt')).numpy()
    db_ids_all   = torch.load(os.path.join(round_dir, 'db_ids.pt'))
    db_id_to_idx = {db_id: i for i, db_id in enumerate(db_ids_all)}

    # Load selected config metadata (for db_ids)
    with open(os.path.join(round_dir, 'selected_configs.json')) as f:
        selected = json.load(f)

    # Load delta values
    with open(os.path.join(round_dir, 'ccsdt_singlepoints.json')) as f:
        singlepoints = json.load(f)

    # Build X (features) and y (deltas) for training
    X, y = [], []
    for r in singlepoints:
        if 'delta_eV' not in r:
            print(f"  Skipping config {r['image_idx']}: {r.get('error', 'no delta')}")
            continue
        i      = r['image_idx']
        db_id  = selected[i]['db_id']
        feat_idx = db_id_to_idx.get(db_id)
        if feat_idx is None:
            print(f"  WARNING: db_id {db_id} not found in features")
            continue
        X.append(features_all[feat_idx])
        y.append(r['delta_eV'])

    X = np.array(X)
    y = np.array(y)

    print(f"Training delta model on {len(y)} configs, feature dim={X.shape[1]}")
    print(f"Delta range: {y.min():.4f} to {y.max():.4f} eV  (mean={y.mean():.4f} eV)")

    # Normalize features
    X_mean = X.mean(axis=0)
    X_std  = X.std(axis=0) + 1e-8
    X_norm = (X - X_mean) / X_std

    # Ridge regression
    model = Ridge(alpha=args.alpha)
    model.fit(X_norm, y)

    y_pred   = model.predict(X_norm)
    train_mae = mean_absolute_error(y, y_pred)
    print(f"Train MAE: {train_mae*1000:.2f} meV  (alpha={args.alpha})")

    # Save weights + normalization stats as JSON (for inference)
    out = {
        'coef':          model.coef_.tolist(),
        'intercept':     float(model.intercept_),
        'X_mean':        X_mean.tolist(),
        'X_std':         X_std.tolist(),
        'alpha':         args.alpha,
        'n_train':       len(y),
        'train_mae_eV':  train_mae,
        'delta_mean_eV': float(y.mean()),
        'delta_std_eV':  float(y.std()),
    }
    out_json = os.path.join(round_dir, 'delta_model.json')
    with open(out_json, 'w') as f:
        json.dump(out, f, indent=2)

    out_pkl = os.path.join(round_dir, 'delta_model.pkl')
    joblib.dump({'model': model, 'X_mean': X_mean, 'X_std': X_std}, out_pkl)

    print(f"\nSaved to {out_json}")
    print(f"         {out_pkl}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--round-dir', required=True)
    parser.add_argument('--alpha',     type=float, default=1.0,
                        help='Ridge regularization strength (try 0.1, 1.0, 10.0)')
    args = parser.parse_args()
    main(args)

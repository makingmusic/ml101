#!/usr/bin/env python3
"""
Sales Multi-Variable Linear Regression Model (robust interactive)
Dataset: dataset/sales_multi.csv with columns: f1, f2, f3, sales

Upgrades:
- Standardized neighbor search (fair tolerance across features)
- Adaptive tolerance growth (×1.5) up to a max
- k-nearest fallback so you always get context
- Clean plotting and resilient interactive loop
"""

import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import train_test_split
from dataclasses import dataclass


# ---------- Data & Model Utilities ----------

def load_and_explore_data(path="dataset/sales_multi.csv"):
    print("Loading dataset...")
    df = pd.read_csv(path, header=None, names=["f1", "f2", "f3", "sales"])

    print("Dataset shape:", df.shape)
    print("\nFirst few rows:")
    print(df.head())

    # Quick feature vs target scatter row
    plt.figure(figsize=(15, 4))
    features = ["f1", "f2", "f3"]
    titles = ["Feature f1 vs Sales", "Feature f2 vs Sales", "Feature f3 vs Sales"]
    for i, (feat, title) in enumerate(zip(features, titles), start=1):
        plt.subplot(1, 3, i)
        plt.scatter(df[feat], df["sales"], alpha=0.7)
        plt.xlabel(feat)
        plt.ylabel("Sales")
        plt.title(title)
        plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    return df


def train_model(df):
    """Train on all rows (to match your original behavior)."""
    print("\nTraining Linear Regression Model (3 features)...")
    X = df[["f1", "f2", "f3"]].values
    y = df["sales"].values

    model = LinearRegression()
    model.fit(X, y)
    y_pred = model.predict(X)

    print("\nLinear Regression Model Results (fit on full data):")
    for feat, coef in zip(["f1", "f2", "f3"], model.coef_):
        print(f"  Coefficient for {feat}: {coef:.4f}")
    print(f"  Intercept: {model.intercept_:.4f}")
    print(f"  R-squared (train): {r2_score(y, y_pred):.4f}")
    print(f"  MSE (train): {mean_squared_error(y, y_pred):.4f}")
    print(f"  MAE (train): {mean_absolute_error(y, y_pred):.4f}")

    # Diagnostics: Predictions vs Actual and Residuals
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.scatter(y, y_pred, alpha=0.7, color="green")
    line_min, line_max = min(y.min(), y_pred.min()), max(y.max(), y_pred.max())
    plt.plot([line_min, line_max], [line_min, line_max], "r--", linewidth=2)
    plt.xlabel("Actual Sales")
    plt.ylabel("Predicted Sales")
    plt.title("Predictions vs Actual")
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 2, 2)
    residuals = y - y_pred
    plt.scatter(y_pred, residuals, alpha=0.7, color="blue")
    plt.axhline(0, color="red", linestyle="--", linewidth=2)
    plt.xlabel("Predicted Sales")
    plt.ylabel("Residuals")
    plt.title("Residuals vs Predicted")
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    return model


# ---------- Neighbor Search (tolerant & fair) ----------

@dataclass
class Standardizer:
    mean_: np.ndarray
    std_: np.ndarray

    @classmethod
    def from_df(cls, df, cols):
        m = df[cols].mean().values.astype(float)
        s = df[cols].std(ddof=0).values.astype(float)
        s[s == 0.0] = 1.0  # guard
        return cls(mean_=m, std_=s)

    def transform(self, X):
        return (X - self.mean_) / self.std_

    def transform_row(self, row):
        return (row - self.mean_) / self.std_


def standardized_distance(a, b):
    """Euclidean distance in standardized space; a,b are 1D arrays."""
    diff = a - b
    return float(np.sqrt(np.dot(diff, diff)))


def get_nearby_points(df, query_row, stdzr, base_tolerance=2.0, max_tolerance=12.0, k_fallback=10):
    """
    Returns a tuple: (nearby_df, used_tolerance, used_fallback)
    - nearby_df is sorted by standardized distance ascending.
    - If no points within adaptive tolerance, falls back to k nearest.
    """
    feats = ["f1", "f2", "f3"]

    # Standardize dataset features and query once
    X_std = stdzr.transform(df[feats].values)
    q_std = stdzr.transform_row(np.array([query_row[f] for f in feats], dtype=float))

    # Axis-aligned ±tol in *standardized* units for fairness, then distance sort
    tol = base_tolerance
    used_fallback = False
    idx = np.array([], dtype=int)

    while tol <= max_tolerance:
        mask = np.all(np.abs(X_std - q_std) <= tol, axis=1)
        idx = np.where(mask)[0]
        if idx.size > 0:
            break
        tol *= 1.5  # expand

    if idx.size == 0:
        # Fallback: k nearest by standardized distance
        used_fallback = True
        dists = np.sqrt(((X_std - q_std) ** 2).sum(axis=1))
        k = min(k_fallback, len(df))
        idx = np.argsort(dists)[:k]

    sel = df.iloc[idx].copy()
    # compute standardized distance column for sorting / display
    dists = np.sqrt(((X_std[idx] - q_std) ** 2).sum(axis=1))
    sel["std_distance"] = dists
    sel = sel.sort_values("std_distance").reset_index(drop=True)
    return sel, (None if used_fallback else tol), used_fallback


# ---------- Prediction & Analysis ----------

def predict_sales(model, f1, f2, f3):
    pred = model.predict([[f1, f2, f3]])
    return float(pred[0])


def analyze_prediction(model, df, f1, f2, f3, base_tolerance=2.0, max_tolerance=12.0, k_fallback=10):
    """More tolerant analysis around a query point."""
    feats = ["f1", "f2", "f3"]
    stdzr = Standardizer.from_df(df, feats)

    predicted_sales = predict_sales(model, f1, f2, f3)
    query = {"f1": f1, "f2": f2, "f3": f3}

    nearby, used_tol, used_fallback = get_nearby_points(
        df, query, stdzr, base_tolerance=base_tolerance,
        max_tolerance=max_tolerance, k_fallback=k_fallback
    )

    print(f"Analysis for features f1={f1}, f2={f2}, f3={f3}")
    print("=" * 60)
    print()

    # Header text describing how neighbors were chosen
    if used_fallback:
        print(f"No axis-aligned neighbors within standardized tolerance up to ±{max_tolerance}.")
        print(f"Showing the {len(nearby)} nearest points by standardized distance instead.")
    else:
        print(f"Found {len(nearby)} nearby points within standardized ±{used_tol:.2f} on each feature.")
    print("-" * 60)

    # Compose comparison table: query first, then neighbors
    rows = [{
        "f1": f1, "f2": f2, "f3": f3,
        "Actual Sales": "—",
        "Predicted Value": f"{predicted_sales:.4f}",
        "StdDist": 0.0
    }]

    for _, r in nearby.iterrows():
        pv = predict_sales(model, float(r["f1"]), float(r["f2"]), float(r["f3"]))
        rows.append({
            "f1": float(r["f1"]),
            "f2": float(r["f2"]),
            "f3": float(r["f3"]),
            "Actual Sales": float(r["sales"]),
            "Predicted Value": f"{pv:.4f}",
            "StdDist": float(r["std_distance"])
        })

    comp_df = pd.DataFrame(rows)
    print(comp_df.to_string(index=False))

    # Neighborhood summary
    if len(nearby) > 0:
        avg_nearby = nearby["sales"].mean()
        print(f"\nAverage nearby actual sales: {avg_nearby:.4f}")
        print(f"|Difference| vs model prediction: {abs(avg_nearby - predicted_sales):.4f}")
        print(f"Std. distance of closest point: {nearby['std_distance'].iloc[0]:.3f}")

    # Small hint for user
    print("\nHint: You can tweak base_tolerance / max_tolerance / k_fallback in code if needed.")


# ---------- Interactive Loop ----------

def interactive_prediction_loop(model, df):
    print("\n" + "=" * 60)
    print("INTERACTIVE PREDICTION TOOL (3 features; tolerant neighbor search)")
    print("=" * 60)
    print("Enter f1, f2, f3 to see prediction and neighborhood.")
    print("Formats: 'f1,f2,f3' or space-separated; type 'quit' to exit")
    print("Press Ctrl-C to exit at any time")
    print()

    # Defaults for the session (can be edited at top if you like)
    base_tol = 2.0
    max_tol = 12.0
    k_fb = 10

    try:
        while True:
            try:
                user_input = input("Enter f1,f2,f3 (or 'quit'): ").strip()
                if user_input.lower() == "quit":
                    print("Goodbye!")
                    break

                tokens = [t for t in user_input.replace(",", " ").split() if t]
                if len(tokens) != 3:
                    print("Please enter exactly 3 numbers for f1, f2, f3.\n")
                    continue

                f1, f2, f3 = map(float, tokens)
                print()
                analyze_prediction(
                    model, df, f1, f2, f3,
                    base_tolerance=base_tol, max_tolerance=max_tol, k_fallback=k_fb
                )
                print()

            except ValueError:
                print("Please enter valid numbers for f1, f2, f3.\n")
            except KeyboardInterrupt:
                print("\nGoodbye!")
                break

    except KeyboardInterrupt:
        print("\nGoodbye!")
        sys.exit(0)


# ---------- Main ----------

def main():
    print("Sales Multi-Variable Linear Regression Analysis")
    print("=" * 40)

    try:
        df = load_and_explore_data()
        model = train_model(df)

        print("\nExample predictions:")
        examples = [(10.0, 10.0, 10.0), (20.0, 30.0, 40.0), (80.0, 20.0, 60.0)]
        for a, b, c in examples:
            pred = predict_sales(model, a, b, c)
            print(f"  For f1={a:.1f}, f2={b:.1f}, f3={c:.1f} -> predicted Sales = {pred:.4f}")

        interactive_prediction_loop(model, df)

    except FileNotFoundError:
        print("Error: Could not find 'dataset/sales_multi.csv' file.")
        print("Please make sure the dataset file exists in the correct location.")
        sys.exit(1)
    except Exception as e:
        print("An error occurred:", e)
        sys.exit(1)


if __name__ == "__main__":
    main()

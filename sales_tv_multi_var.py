#!/usr/bin/env python3
"""
Sales Multi-Variable Linear Regression Model (closest dataset analogs; no dataclasses)
Dataset: dataset/sales_multi.csv with columns: f1, f2, f3, sales

Features:
- Neighbor search in raw feature space (no standardization)
- Adaptive tolerance growth (×1.5) with k-nearest fallback
- 'Closest dataset analogs':
    * Closest by features (min Euclidean distance)
    * Closest by output (actual sales closest to model's predicted sales)
"""

import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error


# ---------- Data & Model ----------

def load_and_explore_data(path="dataset/sales_multi.csv"):
    print("Loading dataset...")
    df = pd.read_csv(path, header=None, names=["f1", "f2", "f3", "sales"])

    plt.figure(figsize=(15, 4))
    for i, feat in enumerate(["f1", "f2", "f3"], start=1):
        plt.subplot(1, 3, i)
        plt.scatter(df[feat], df["sales"], alpha=0.7)
        plt.xlabel(feat)
        plt.ylabel("Sales")
        plt.title(f"{feat} vs Sales")
        plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    return df


def train_model(df):
    print("\nTraining Linear Regression Model (3 features)...")
    X = df[["f1", "f2", "f3"]].values
    y = df["sales"].values

    model = LinearRegression()
    model.fit(X, y)
    y_pred = model.predict(X)

    print("the whole equation can be represented as:")
    print(f"  sales = ({model.coef_[0]:.4f} * f1) + ({model.coef_[1]:.4f} * f2) + ({model.coef_[2]:.4f} * f3) + {model.intercept_:.4f} ")
    print(f"  R² (train): {r2_score(y, y_pred):.4f}")
    print(f"  MSE (train): {mean_squared_error(y, y_pred):.4f}")
    print(f"  MAE (train): {mean_absolute_error(y, y_pred):.4f}")

    # Diagnostics: Predictions vs Actual and Residuals
    plt.figure(figsize=(12, 5))
    # Predictions vs Actual
    plt.subplot(1, 2, 1)
    a, b = min(y.min(), y_pred.min()), max(y.max(), y_pred.max())
    plt.scatter(y, y_pred, alpha=0.7, color="green")
    plt.plot([a, b], [a, b], "r--", linewidth=2)
    plt.xlabel("Actual Sales")
    plt.ylabel("Predicted Sales")
    plt.title("Predictions vs Actual")
    plt.grid(True, alpha=0.3)
    # Residuals vs Predicted
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


# ---------- Neighbors (raw feature space) ----------

def get_nearby_points(df, query_row, base_tolerance=2.0, max_tolerance=12.0, k_fallback=10):
    """
    Adaptive axis-aligned box in raw feature space; k-NN fallback.
    Returns (nearby_df_sorted, used_tolerance, used_fallback)
    """
    feats = ["f1", "f2", "f3"]
    X = df[feats].values
    q = np.array([query_row[f] for f in feats], dtype=float)

    tol = base_tolerance
    used_fallback = False
    idx = np.array([], dtype=int)

    while tol <= max_tolerance:
        mask = np.all(np.abs(X - q) <= tol, axis=1)
        idx = np.where(mask)[0]
        if idx.size > 0:
            break
        tol *= 1.5

    if idx.size == 0:
        used_fallback = True
        dists = np.sqrt(((X - q) ** 2).sum(axis=1))
        k = min(k_fallback, len(df))
        idx = np.argsort(dists)[:k]

    dists = np.sqrt(((X[idx] - q) ** 2).sum(axis=1))
    sel = df.iloc[idx].copy()
    sel["distance"] = dists
    sel = sel.sort_values("distance").reset_index(drop=True)
    return sel, (None if used_fallback else tol), used_fallback


# ---------- Prediction & Analysis ----------

def predict_sales(model, f1, f2, f3):
    return float(model.predict([[f1, f2, f3]])[0])


def analyze_prediction(model, df, f1, f2, f3, base_tolerance=2.0, max_tolerance=12.0, k_fallback=10):
    feats = ["f1", "f2", "f3"]
    query = {"f1": f1, "f2": f2, "f3": f3}
    predicted_sales = predict_sales(model, f1, f2, f3)

    # Nearby points (adaptive / fallback)
    nearby, used_tol, used_fallback = get_nearby_points(
        df, query, base_tolerance=base_tolerance,
        max_tolerance=max_tolerance, k_fallback=k_fallback
    )

    print(f"Analysis for features f1={f1}, f2={f2}, f3={f3}")
    print("=" * 70)
    print(f"Model prediction: {predicted_sales:.4f}\n")

    # Neighbor summary
    if used_fallback:
        print(f"No axis-aligned neighbors within tolerance up to ±{max_tolerance}.")
        print(f"Showing the {len(nearby)} nearest points by Euclidean distance.\n")
    else:
        print(f"Found {len(nearby)} nearby points within ±{used_tol:.2f} on each feature.\n")

    # Comparison table
    rows = [{
        "f1": f1, "f2": f2, "f3": f3,
        "Actual Sales": "—",
        "Predicted Value": f"{predicted_sales:.4f}",
        "Distance": 0.0
    }]
    for _, r in nearby.iterrows():
        pv = predict_sales(model, float(r["f1"]), float(r["f2"]), float(r["f3"]))
        rows.append({
            "f1": float(r["f1"]),
            "f2": float(r["f2"]),
            "f3": float(r["f3"]),
            "Actual Sales": float(r["sales"]),
            "Predicted Value": f"{pv:.4f}",
            "Distance": float(r["distance"])
        })
    comp_df = pd.DataFrame(rows)
    print("Nearby points (feature space):")
    print(comp_df.to_string(index=False))

    if len(nearby) > 0:
        avg_nearby = nearby["sales"].mean()
        print(f"\nAverage nearby actual sales: {avg_nearby:.4f}")
        print(f"|Difference| vs model prediction: {abs(avg_nearby - predicted_sales):.4f}")


# ---------- Interactive Loop ----------

def interactive_prediction_loop(model, df):
    print("\n" + "=" * 60)
    print("INTERACTIVE PREDICTION TOOL (3 features; tolerant neighbors + closest analogs)")
    print("=" * 60)
    print("Enter f1, f2, f3 to see prediction, neighborhood, and closest dataset analogs.")
    print("Formats: 'f1,f2,f3' or space-separated; type 'quit' to exit")
    print("Press Ctrl-C to exit at any time\n")

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

#!/usr/bin/env python3
"""
Sales TV Linear Regression Model
Converts the Jupyter notebook analysis into a standalone Python script
with interactive prediction capabilities.
"""

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import sys


def load_and_explore_data():
    """Load the dataset and display basic information"""
    print("Loading dataset...")
    df = pd.read_csv("dataset/sales_tv.csv", header=None, names=["tvAdSpend", "sales"])

    print("Dataset shape:", df.shape)
    print("\nFirst few rows:")
    print(df.head())

    # Plot the data
    plt.figure(figsize=(10, 6))
    plt.scatter(df["tvAdSpend"], df["sales"], alpha=0.7, color="blue")
    plt.xlabel("TV Ad Spend")
    plt.ylabel("Sales")
    plt.title("TV Ad Spend vs Sales Scatter Plot")
    plt.grid(True, alpha=0.3)
    plt.show()

    return df


def train_model(df):
    """Train the linear regression model and display results"""
    print("\nTraining Linear Regression Model...")

    # Prepare the data for training
    X = df[["tvAdSpend"]].values  # Features (tvAdSpend values)
    y = df["sales"].values  # Target (sales values)

    # Create and train the linear regression model
    model = LinearRegression()
    model.fit(X, y)

    # Make predictions
    y_pred = model.predict(X)

    # Display model parameters
    print("Linear Regression Model Results:")
    print(f"Slope (coefficient): {model.coef_[0]:.4f}")
    print(f"Intercept: {model.intercept_:.4f}")
    print(f"R-squared: {r2_score(y, y_pred):.4f}")
    print(f"Mean Squared Error: {mean_squared_error(y, y_pred):.4f}")

    # Plot the original data and the regression line
    # plt.figure(figsize=(12, 6))

    # Subplot 1: Original data with regression line
    plt.subplot(1, 2, 1)
    plt.scatter(X, y, alpha=0.7, color="blue", label="Original Data")
    plt.plot(X, y_pred, color="red", linewidth=2, label="Linear Regression Line")
    plt.xlabel("TV Ad Spend")
    plt.ylabel("Sales")
    plt.title("Linear Regression Fit")
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Subplot 2: Predictions vs Actual values
    plt.subplot(1, 2, 2)
    plt.scatter(y, y_pred, alpha=0.7, color="green")
    plt.plot([y.min(), y.max()], [y.min(), y.max()], "r--", linewidth=2)
    plt.xlabel("Actual Sales")
    plt.ylabel("Predicted Sales")
    plt.title("Predictions vs Actual")
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    return model


def predict_sales(model, tv_ad_spend):
    """Predict sales value for a given TV ad spend"""
    prediction = model.predict([[tv_ad_spend]])
    return prediction[0]


def analyze_prediction(model, df, tv_ad_spend, tolerance=1.0):
    """
    Analyze a prediction for a given TV ad spend value and show nearby data points
    """
    # Get the model's prediction
    predicted_sales = predict_sales(model, tv_ad_spend)

    # Find nearby data points in the dataset
    nearby_data = df[abs(df["tvAdSpend"] - tv_ad_spend) <= tolerance].copy()
    nearby_data = nearby_data.sort_values("tvAdSpend")

    # Create comparison table
    print(f"Analysis for TV Ad Spend = {tv_ad_spend}")
    print("=" * 50)
    print()

    if len(nearby_data) > 0:
        print(f"Nearby data points (within ±{tolerance} of TV Ad Spend={tv_ad_spend}):")
        print("-" * 50)

        # Create a formatted table
        comparison_data = []

        # Add the user's input value first
        comparison_data.append(
            {
                "TV Ad Spend": tv_ad_spend,
                "Actual Sales": "N/A",
                "Predicted Value": f"{predicted_sales:.4f}",
            }
        )

        # Add nearby data points
        for _, row in nearby_data.iterrows():
            actual_x, actual_y = row["tvAdSpend"], row["sales"]
            predicted_y = predict_sales(model, actual_x)
            comparison_data.append(
                {
                    "TV Ad Spend": actual_x,
                    "Actual Sales": actual_y,
                    "Predicted Value": f"{predicted_y:.4f}",
                }
            )

        # Display as a nice table
        comparison_df = pd.DataFrame(comparison_data)
        # Sort by TV Ad Spend, but handle the "N/A" value for actual sales
        comparison_df_sorted = comparison_df.sort_values("TV Ad Spend")
        print(comparison_df_sorted.to_string(index=False))

        # Calculate average nearby y value for comparison
        avg_nearby_y = nearby_data["sales"].mean()
        print(f"\nAverage sales value of nearby points: {avg_nearby_y:.4f}")
        print(
            f"Difference from model prediction: {abs(avg_nearby_y - predicted_sales):.4f}"
        )

    else:
        print(f"No data points found within ±{tolerance} of TV Ad Spend={tv_ad_spend}")
        print(
            "Try increasing the tolerance or check if the TV Ad Spend value is within the dataset range."
        )


def interactive_prediction_loop(model, df):
    """Interactive loop for user predictions"""
    print("\n" + "=" * 60)
    print("INTERACTIVE PREDICTION TOOL")
    print("=" * 60)
    print(
        "Enter a TV Ad Spend value to see the model's prediction and nearby data points"
    )
    print("Type 'quit' to exit")
    print("Press Ctrl-C to exit at any time")
    print()

    try:
        while True:
            try:
                # Get user input
                user_input = input(
                    "Enter TV Ad Spend value (or 'quit' to exit): "
                ).strip()

                if user_input.lower() == "quit":
                    print("Goodbye!")
                    break

                tv_ad_spend = float(user_input)

                # Use default tolerance of 1.0
                tolerance = 1.0

                print()
                analyze_prediction(model, df, tv_ad_spend, tolerance)
                print()

            except ValueError:
                print("Please enter a valid number for TV Ad Spend value.")
                print()
            except KeyboardInterrupt:
                print("\nGoodbye!")
                break

    except KeyboardInterrupt:
        print("\nGoodbye!")
        sys.exit(0)


def main():
    """Main function to run the complete analysis"""
    print("Sales TV Linear Regression Analysis")
    print("=" * 40)

    try:
        # Load and explore data
        df = load_and_explore_data()

        # Train the model
        model = train_model(df)

        # Show example predictions
        print("\nExample predictions:")
        print(
            f"For TV Ad Spend = 5.0, predicted Sales = {predict_sales(model, 5.0):.4f}"
        )
        print(
            f"For TV Ad Spend = 10.0, predicted Sales = {predict_sales(model, 10.0):.4f}"
        )
        print(
            f"For TV Ad Spend = 15.0, predicted Sales = {predict_sales(model, 15.0):.4f}"
        )

        # Start interactive prediction loop
        interactive_prediction_loop(model, df)

    except FileNotFoundError:
        print("Error: Could not find 'dataset/sales_tv.csv' file.")
        print("Please make sure the dataset file exists in the correct location.")
        sys.exit(1)
    except Exception as e:
        print("An error occurred:", e)
        sys.exit(1)


if __name__ == "__main__":
    main()

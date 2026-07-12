import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score


def main():
    data_file = Path("data") / "student_scores.csv"
    output_folder = Path("outputs")
    output_file = output_folder / "linear_regression_results.csv"
    chart_file = output_folder / "linear_regression_chart.png"

    output_folder.mkdir(exist_ok=True)

    df = pd.read_csv(data_file)

    print("Student Score Dataset")
    print("---------------------")
    print(df)

    print()
    print("Dataset Information")
    print("-------------------")
    print(df.info())

    X = df[["StudyHours"]]
    y = df["Score"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    model = LinearRegression()

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    results = pd.DataFrame({
        "StudyHours": X_test["StudyHours"],
        "ActualScore": y_test,
        "PredictedScore": predictions.round(2)
    })

    mae = mean_absolute_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)

    print()
    print("Prediction Results")
    print("------------------")
    print(results)

    print()
    print("Model Details")
    print("-------------")
    print(f"Slope: {model.coef_[0]:.2f}")
    print(f"Intercept: {model.intercept_:.2f}")

    print()
    print("Model Evaluation")
    print("----------------")
    print(f"Mean Absolute Error: {mae:.2f}")
    print(f"R-squared Score: {r2:.2f}")

    new_student = pd.DataFrame({
        "StudyHours": [6.5]
    })

    predicted_score = model.predict(new_student)

    print()
    print(f"Predicted score for 6.5 study hours: {predicted_score[0]:.2f}")

    results.to_csv(output_file, index=False)

    predicted_line = model.predict(X)

    plt.figure(figsize=(8, 5))
    plt.scatter(df["StudyHours"], df["Score"], label="Actual Data")
    plt.plot(df["StudyHours"], predicted_line, label="Regression Line")

    plt.title("Linear Regression: Study Hours vs Score")
    plt.xlabel("Study Hours")
    plt.ylabel("Score")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    plt.savefig(chart_file)
    plt.show()

    print()
    print(f"Prediction results saved to: {output_file}")
    print(f"Chart saved to: {chart_file}")


main()


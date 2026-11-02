
import pandas as pd
import joblib

from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load data
df = pd.read_csv("data/raw/student_data.csv")

# Features and target
X = df.drop(columns=["final_marks"])
y = df["final_marks"]

# Train test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

# Create ML pipeline
model_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
    ("model", RandomForestRegressor(
        n_estimators=100,
        random_state=42
    ))
])

# Train model
model_pipeline.fit(X_train, y_train)

# Predictions
predictions = model_pipeline.predict(X_test)

# Evaluation
mae = mean_absolute_error(y_test, predictions)
rmse = mean_squared_error(y_test, predictions) ** 0.5
r2 = r2_score(y_test, predictions)

print("\nModel Evaluation Results")
print("------------------------")
print("MAE:", round(mae, 2))
print("RMSE:", round(rmse, 2))
print("R2 Score:", round(r2, 3))

# Save model
Path("models").mkdir(exist_ok=True)

joblib.dump(
    model_pipeline,
    "models/student_model.pkl"
)

# Save evaluation report
Path("outputs").mkdir(exist_ok=True)

with open("outputs/evaluation_report.txt", "w") as file:
    file.write(f"MAE: {mae:.2f}\n")
    file.write(f"RMSE: {rmse:.2f}\n")
    file.write(f"R2 Score: {r2:.3f}\n")

print("\nModel saved successfully!")

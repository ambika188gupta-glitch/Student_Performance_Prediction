
import numpy as np
import pandas as pd
from pathlib import Path

np.random.seed(42)

n = 1000

study_hours = np.random.uniform(1, 10, n)
attendance = np.random.uniform(50, 100, n)
previous_marks = np.random.uniform(30, 100, n)
sleep_hours = np.random.uniform(4, 10, n)

# Generate target variable
noise = np.random.normal(0, 5, n)

final_marks = (
    3.5 * study_hours
    + 0.25 * attendance
    + 0.35 * previous_marks
    + 1.2 * sleep_hours
    + noise
)

final_marks = np.clip(final_marks, 0, 100)

df = pd.DataFrame({
    "study_hours": study_hours,
    "attendance": attendance,
    "previous_marks": previous_marks,
    "sleep_hours": sleep_hours,
    "final_marks": final_marks
})

# Add some missing values for practice
for column in ["study_hours", "attendance"]:
    missing_indices = np.random.choice(
        df.index, size=20, replace=False
    )
    df.loc[missing_indices, column] = np.nan

output_path = Path("data/raw/student_data.csv")
output_path.parent.mkdir(parents=True, exist_ok=True)

df.to_csv(output_path, index=False)

print("Dataset created successfully!")
print("Dataset shape:", df.shape)
print(df.head())

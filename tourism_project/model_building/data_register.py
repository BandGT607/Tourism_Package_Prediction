import pandas as pd
import sys

RAW_PATH = "tourism_project/data/tourism.csv"

# Load the raw dataset
try:
    df = pd.read_csv(RAW_PATH)
except FileNotFoundError:
    print(f"CRITICAL ERROR: {RAW_PATH} not found. Did you move the file?")
    sys.exit(1)

# Validate that the expected columns are present
expected_columns = [
    "CustomerID", "ProdTaken", "Age", "TypeofContact", "CityTier", 
    "Occupation", "Gender", "NumberOfPersonVisiting", "PreferredPropertyStar",
    "MaritalStatus", "NumberOfTrips", "Passport", "OwnCar", 
    "NumberOfChildrenVisiting", "Designation", "MonthlyIncome",
    "PitchSatisfactionScore", "ProductPitched", "NumberOfFollowups", "DurationOfPitch"
]

missing_cols = [col for col in expected_columns if col not in df.columns]
if missing_cols:
    raise ValueError(f"Dataset Registration Failed. Missing columns: {missing_cols}")

print("Dataset registered and validated successfully.")
print(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")

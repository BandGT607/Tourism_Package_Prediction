import pandas as pd
from sklearn.model_selection import train_test_split

DATA_PATH = "tourism_project/data/tourism.csv"

# 1. Load the registered dataset
df = pd.read_csv(DATA_PATH)

# 2. Data Cleaning: Drop identifier columns
df.drop(columns=["CustomerID"], inplace=True)

# 3. Separate features (X) and target (y)
target = "ProdTaken"
X = df.drop(columns=[target])
y = df[target]

# 4. Train-Test Split (stratified to maintain class balance)
Xtrain, Xtest, ytrain, ytest = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 5. Save the splits locally for the pipeline to use
Xtrain.to_csv("Xtrain.csv", index=False)
Xtest.to_csv("Xtest.csv", index=False)
ytrain.to_csv("ytrain.csv", index=False)
ytest.to_csv("ytest.csv", index=False)

print("Data Preparation Completed Successfully.")
print(f"Training set size: {Xtrain.shape[0]} samples")
print(f"Test set size: {Xtest.shape[0]} samples")

import pandas as pd
from sklearn.model_selection import train_test_split

DATA_PATH = "tourism_project/data/tourism.csv"

# Load data and securely remove any accidental index columns
df = pd.read_csv(DATA_PATH)
df = df.loc[:, ~df.columns.str.contains('^Unnamed')]

if "CustomerID" in df.columns:
    df.drop(columns=["CustomerID"], inplace=True)

target = "ProdTaken"
X = df.drop(columns=[target])
y = df[target]

# Split with class stratification
Xtrain, Xtest, ytrain, ytest = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

Xtrain.to_csv("Xtrain.csv", index=False)
Xtest.to_csv("Xtest.csv", index=False)
ytrain.to_csv("ytrain.csv", index=False)
ytest.to_csv("ytest.csv", index=False)
print("Data Preparation Completed Successfully.")
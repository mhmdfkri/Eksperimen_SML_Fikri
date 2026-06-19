import pandas as pd

# Load data
df = pd.read_csv("../data/telco.csv")

# Drop leakage columns
drop_columns = [
    "Customer ID",
    "Customer Status",
    "Churn Category",
    "Churn Reason",
    "Churn Score",
    "CLTV",
    "Satisfaction Score",
    "City",
    "State",
    "Zip Code",
    "Latitude",
    "Longitude"
]

df = df.drop(columns=drop_columns, errors="ignore")

# Missing values
df["Offer"] = df["Offer"].fillna("Unknown")
df["Internet Type"] = df["Internet Type"].fillna("Unknown")

# Encode target
df["Churn Label"] = (
    df["Churn Label"]
    .astype(str)
    .str.strip()
    .map({
        "No": 0,
        "Yes": 1
    })
)

# Save
df.to_csv(
    "../data/telco_processed.csv",
    index=False
)

print("Preprocessing selesai")
print(df.shape)


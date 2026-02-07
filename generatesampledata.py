import pandas as pd


url = "https://archive.ics.uci.edu/ml/machine-learning-databases/spambase/spambase.data"
df = pd.read_csv(url, header=None)


# Sample 50 rows (stratified for balance)
sample_df = df.groupby(df.iloc[:, -1], group_keys=False).apply(
    lambda x: x.sample(n=25, random_state=42)
)

sample_df = sample_df.sample(frac=1, random_state=42).reset_index(drop=True)

sample_df.to_csv("sampledata/spambase_test_sample_50.csv", index=False, header=False)

print("Sample test data generated: sampledata/spambase_test_sample_50.csv")

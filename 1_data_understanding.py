import pandas as pd

df = pd.read_csv("winequality-red.csv", sep=";")

print(df.head())
print()
print(df.shape)
print()
print(df.columns)

print()
print(df.isnull().sum())

print()
print(df["quality"].value_counts().sort_index())

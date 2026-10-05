from sklearn.model_selection import train_test_split
import pandas as pd

df = pd.read_csv("winequality-red.csv", sep=";")

X=df.drop("quality", axis=1)
y=df["quality"] 

print("X shape:", X.shape)
print("y shape:", y.shape)  

print()
print("X columns:")
print(X.columns)

print()
print("First 5 values of y:")
print(y.head())

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print()
print("X_train:", X_train.shape)
print("X_test:", X_test.shape)
print("y_train:", y_train.shape)
print("y_test:", y_test.shape)
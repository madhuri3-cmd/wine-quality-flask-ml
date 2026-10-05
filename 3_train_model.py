import pandas as pd
from sklearn.metrics import accuracy_score
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier


# 1. Load dataset
df = pd.read_csv("winequality-red.csv", sep=";")


# 2. Separate features and target
X = df.drop("quality", axis=1)
y = df["quality"]


# 3. Split data into training and testing
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


# 4. Create the Random Forest model
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)


# 5. Train the model
model.fit(X_train, y_train)

# 6. Make predictions
y_pred = model.predict(X_test)

# 7. Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)

print()
print("Model Accuracy:", accuracy)
print("Model Accuracy (%):", accuracy * 100)

joblib.dump(model, "wine_quality_model.joblib")

print()
print("Model saved successfully!")


print("Model training completed!")
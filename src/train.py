import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from model import get_model
from sklearn.metrics import mean_squared_error

# Load dataset
data = pd.read_csv("../data/train.csv")

# Split into features and target
X = data.iloc[:, :-1]  # All columns except last
y = data.iloc[:, -1]   # Last column as target

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = get_model()
model.fit(X_train, y_train)

# Save trained model
joblib.dump(model, "../models/model.pkl")

# Evaluate model
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse:.2f}")


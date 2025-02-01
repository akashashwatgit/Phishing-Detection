import joblib
import pandas as pd
from sklearn.metrics import mean_squared_error

# Load dataset
data = pd.read_csv("../data/test.csv")

# Load model
model = joblib.load("../models/model.pkl")

# Split into features and target
X_test = data.iloc[:, :-1]
y_test = data.iloc[:, -1]

# Predict
y_pred = model.predict(X_test)

# Evaluate model
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse:.2f}")

import numpy as np
import pandas as pd
import joblib  # Replacing pickle with joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.impute import SimpleImputer

# Load dataset
df = pd.read_csv('car_prices.csv')

# Prepare data using .iloc
X = df.iloc[:, 0:1].values  # Independent variable (Car_Age)
y = df.iloc[:, 1].values  # Dependent variable (Car_Price)

# Handle missing values
imputer = SimpleImputer(strategy='mean')  # Replace missing values with the mean
X = imputer.fit_transform(X)
y = imputer.fit_transform(y.reshape(-1, 1)).ravel()  # Reshape for compatibility

# Split data (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Transform input for Polynomial Regression (degree 2)
poly = PolynomialFeatures(degree=2)
X_train_poly = poly.fit_transform(X_train)
X_test_poly = poly.transform(X_test)

# Train the model
model = LinearRegression()
model.fit(X_train_poly, y_train)

# Make predictions
y_pred = model.predict(X_test_poly)

# Calculate accuracy as percentage of predictions within a threshold (e.g., 10%)
threshold = 0.10  # 10% tolerance
accuracy = np.mean(np.abs((y_test - y_pred) / y_test) <= threshold) * 100

print(f"Model Accuracy: {accuracy:.2f}%")

# Save model, transformer, and imputer using joblib
joblib.dump(model, 'poly_model.pkl')
joblib.dump(poly, 'poly_transformer.pkl')
joblib.dump(imputer, 'imputer.pkl')

print("Model trained, evaluated, and saved successfully!")

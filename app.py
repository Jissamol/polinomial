from flask import Flask, render_template, request
import numpy as np
import joblib  # Using joblib instead of pickle

# Load trained model and transformer
model = joblib.load('poly_model.pkl')
poly = joblib.load('poly_transformer.pkl')

# Initialize Flask app
app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get user input
        car_age = float(request.form['car_age'])

        # Transform input for polynomial regression
        car_age_poly = poly.transform([[car_age]])

        # Predict price
        predicted_price = model.predict(car_age_poly)[0]

        return render_template('index.html', prediction=f"Predicted Car Price: ${predicted_price:.2f}")

    except Exception as e:
        return render_template('index.html', error="Invalid input. Please enter a valid number.")

if __name__ == '__main__':
    app.run(debug=True)

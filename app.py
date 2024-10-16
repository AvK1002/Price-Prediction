from flask import Flask, render_template, request, jsonify # type: ignore
import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler # type: ignore

app = Flask(__name__)

# Load the trained model (update this path to point to your model file)
model = pickle.load(open('trained_model.pkl', 'rb'))

# Load the scaler used during training (update this path to point to your scaler file)
scaler = pickle.load(open('scaler.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    # Define the reverse mappings for Locality and Residential
    locality_mapping = {
        'Bridgeport': 0, 'Waterbury': 1, 'Fairfield': 2, 'West Hartford': 3,
        'Greenwich': 4, 'Norwalk': 5, 'Stamford': 6, 'Unknown': 7
    }

    residential_mapping = {
        'Detached House': 0, 'Duplex': 1, 'Triplex': 2, 'Fourplex': 3, 'Unknown': 4
    }

    # Get input values from the form
    estimated_value = float(request.form['estimated_value'])
    locality = request.form['locality']  # Get locality as a string
    num_bathrooms = int(request.form['num_bathrooms'])
    residential = request.form['residential']  # Get residential type as a string
    property_tax_rate = float(request.form['property_tax_rate'])

    # Convert categorical values to their encoded numerical values
    locality_encoded = locality_mapping.get(locality, 7)  # Default to 'Unknown' if not found
    residential_encoded = residential_mapping.get(residential, 4)  # Default to 'Unknown' if not found

    # Create a DataFrame for the input data
    input_data = pd.DataFrame({
        'Estimated Value': [estimated_value],
        'Locality': [locality_encoded],
        'num_bathrooms': [num_bathrooms],
        'Residential': [residential_encoded],
        'property_tax_rate': [property_tax_rate]
    })

    # Standardize the input features
    input_data_scaled = scaler.transform(input_data)

    # Predict the sale price
    predicted_price = model.predict(input_data_scaled)[0]

    return render_template('result.html', predict = predicted_price)


if __name__ == '__main__':
    app.run(debug=True)

from flask import Flask, request, jsonify
import joblib

app = Flask(__name__)

# Load your trained model
model, scaler = joblib.load('fall_detection_lrmodel.joblib')
rfModel, rfScaler = joblib.load('fall_detection_rfmodel.joblib')

# Define a route for prediction using Logistic Regression
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get the JSON data from the request body
        data = request.json

        # Assuming your JSON data contains the features you want to predict
        features = data['features']

        # Perform data scaling if necessary (assuming 'scaler' is defined)
        data_point_scaled = scaler.transform(features)
        
        # Make predictions using your loaded model
        predictions = model.predict(data_point_scaled)
        
        # Iterate through predictions and build a new list
        prediction_values = []
        for prediction in predictions:
            message = "Not a fall" if prediction == 0 else "Fall"
            # Append each prediction value to the list
            prediction_values.append({'message': message, 'isFalled': prediction.tolist()})

        # Return the predictions as JSON
        return jsonify(prediction_values)

    except Exception as e:
        return jsonify({'error': str(e)})
    
# Define a route for prediction using Random Forest
@app.route('/predict/rf', methods=['POST'])
def predictRf():
    try:
        # Get the JSON data from the request body
        data = request.json

        # Assuming your JSON data contains the features you want to predict
        features = data['features']

        # Perform data scaling if necessary (assuming 'scaler' is defined)
        data_point_scaled = rfScaler.transform(features)
        
        # Make predictions using your loaded model
        predictions = rfModel.predict(data_point_scaled)

        # Iterate through predictions and build a new list
        prediction_values = []
        for prediction in predictions:
            message = "Not a fall" if prediction == 0 else "Fall"
            # Append each prediction value to the list
            prediction_values.append({'message': message, 'isFalled': prediction.tolist()})

        # Return the predictions as JSON
        return jsonify(prediction_values)

    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
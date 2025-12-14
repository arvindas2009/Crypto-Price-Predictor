"""
Flask Web Server for Crypto Price Prediction

Team Zephyrus - OlympAI Hackathon 2025

This file contains the Flask web application that gives the interface.
It handles HTTP requests, processes user input, and returns prediction results in JSON format.

Routes:
    / (GET) - Serves the main HTML interface
    /predict (POST) - Handles prediction requests and returns analysis results
"""

# Main Flask app

from flask import Flask, request, jsonify, render_template
import traceback

# importing the model function
from model import analyze_crypto

app = Flask(__name__)
app.config['JSON_SORT_KEYS'] = False  # keep json order as is


@app.route('/')
def index():
    # render the html page
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    # This function handles the prediction request
    # Gets the symbol from user and returns prediction
    try:
        # get the json data from request
        data = request.get_json()
        
        if not data:
            return jsonify({'error': 'No data provided'}), 400

        # extract symbol and clean it up
        symbol = data.get('symbol', '').strip().upper()
        
        if not symbol:
            return jsonify({'error': 'Symbol is required'}), 400

        # call our model to analyze
        result = analyze_crypto(symbol)
        
        # check if we got a result
        if result is None:
            return jsonify({'error': 'Unable to analyze symbol. Check if it exists.'}), 400

        return jsonify(result)

    except Exception as e:
        # if something goes wrong, print error and return 500
        print(f"Error in prediction: {e}")
        traceback.print_exc()
        return jsonify({'error': 'Server error occurred'}), 500


# run the app when this file is executed
if __name__ == '__main__':
    print("Starting Crypto Prediction Server...")
    print("Access the application at: http://127.0.0.1:5000")
    app.run(debug=True, host='127.0.0.1', port=5000)

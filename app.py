from flask import Flask, request, render_template, jsonify
from pycaret.classification import *
from pycaret.regression import *
import pandas as pd
import pickle
import numpy as np

app = Flask(__name__)

# Load the classification model
classification_model = load_model('mlruns/2/8deec9957f874038b8798141e73620dc/artifacts/model/model')
classification_cols = ['age', 'gender', 'chest_pain', 'resting_BP', 'cholesterol', 'fasting_BS', 'resting_ECG', 'max_HR',
                       'exercise_angina', 'old_peak', 'ST_slope']

# Load the regression model
regression_model = load_model('mlruns/1/5665d3f8f6e74146ae59908425cf241a/artifacts/model/model')
regression_cols = ['town', 'flat_type', 'storey_range', 'floor_area_sqm', 'flat_model', 'lease_commence_date']

@app.route('/')
def home():
    return render_template("home.html")

@app.route('/predict_classification', methods=['POST'])
def predict_classification():
    int_features = [x for x in request.form.values()]
    final = np.array(int_features)
    data_unseen = pd.DataFrame([final], columns=classification_cols)
    prediction = predict_model(classification_model, data=data_unseen, round=0)
    prediction = int(prediction.prediction_label)
    return render_template('cardiovascular.html', pred='Your Cardiovascular Issue Presence is: {}'.format(prediction))

@app.route('/predict_regression', methods=['POST'])
def predict_regression():
    int_features = [x for x in request.form.values()]
    final = np.array(int_features)
    data_unseen = pd.DataFrame([final], columns=regression_cols)
    prediction = predict_model(regression_model, data=data_unseen, round=0)
    prediction = int(prediction.prediction_label)
    return render_template('hdb.html', pred='The Predicted Resale Price $SGD  {}'.format(prediction))

@app.route('/predict_api', methods=['POST'])
def predict_api():
    data = request.get_json(force=True)
    classification_data = pd.DataFrame([data], columns=classification_cols)
    regression_data = pd.DataFrame([data], columns=regression_cols)
    
    classification_prediction = predict_model(classification_model, data=classification_data)
    regression_prediction = predict_model(regression_model, data=regression_data)
    
    output = {
        'classification_prediction': classification_prediction.prediction_label[0],
        'regression_prediction': regression_prediction.prediction_label[0]
    }
    return jsonify(output)

@app.route('/hdb')
def hdb():
   return render_template('hdb.html')

@app.route('/cardiovascular')
def cardiovascular():
   return render_template('cardiovascular.html')



if __name__ == '__main__':
    app.run(debug=True)

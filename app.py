from flask import Flask, request, jsonify
import pandas as pd
import pickle
from sklearn.preprocessing import LabelEncoder

app = Flask(__name__)

label_encoder=LabelEncoder()
with open('model.pkl',"rb") as f:
    model=pickle.load(f)
@app.route("/")
def hello_world():
    return "<p> Hello world </p>"

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get the JSON data from the request
        data = request.get_json()
        
        # Transform input data using the fitted label encoder
        data['Time of Day'] = label_encoder.fit_transform([data['Time of Day']])[0]
        data['Location'] = label_encoder.fit_transform([data['Location']])[0]
        data['Service Type'] = label_encoder.fit_transform([data['Service Type']])[0]
        #data['Provider'] = label_encoder.fit_transform([data['Provider']])[0]
        #data['Historical Transaction Amount'] = label_encoder.fit_transform([data['Historical Transaction Amount']])[0]

        # Make predictions using the ensemble model
        risk_score = model.predict_proba([[
            data['Transaction Amount'],
            data['Historical Transaction Amount'],
            data['Frequency'],
            data['Time of Day'],
            data['Location'],
            data['Service Type']
        ]])[:, 1].item()

        return jsonify({'risk_score': risk_score})

    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(port=5000, debug=True)

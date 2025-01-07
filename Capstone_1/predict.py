import pickle

from flask import Flask
from flask import request
from flask import jsonify


model_file = 'C:/Users/Sandra/Documents/Cursos/MLZoomcamp2024/Capstone_1_Project/models/model_RF.bin'

with open(model_file, 'rb') as f_in:
    dv, model = pickle.load(f_in)

app = Flask('target')

@app.route('/predict', methods=['POST'])
def predict():
    patient = request.get_json()

    X = dv.transform([patient])
    y_pred = model.predict_proba(X)[0, 1]
    target_pred = y_pred >= 0.5

    result = {
        'target_probability': float(y_pred),
        'target': bool(target_pred)
    }

    return jsonify(result)


if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=9696)

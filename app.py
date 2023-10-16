from flask import Flask, request

from mortality_prediction import XGBoostEnsemble

import logging

app = Flask(__name__)

logging.basicConfig(filename='logs/flask.log', level=logging.DEBUG, format=f'%(asctime)s %(levelname)s %(name)s %(threadName)s : %(message)s')


@app.route('/', methods=['GET'])
def hello():
    return('Welcome to Clinical Outcome Prediction Services!!')


@app.route('/predict_mortality', methods=['POST'])
def predict_mortality():
    id = str(request.json['id'])
    targetStart = request.json['targetStart']
    targetEnd = request.json['targetEnd']
    windowEnd = request.json['windowEnd']
    saveFileName = 'ts_' + str(targetStart) + '_te_' + str(targetEnd) + '_model.pkl'
    return XGBoostEnsemble.predict(id=id, targetStart=targetStart, targetEnd=targetEnd, windowEnd=windowEnd, savePath='models/mortality_prediction', saveFileName=saveFileName)


@app.route('/train_mortality', methods=['POST'])
def train_mortality():
    targetStart = request.json['targetStart']
    targetEnd = request.json['targetEnd']
    windowEnd = request.json['windowEnd']
    saveFileName = 'ts_' + str(targetStart) + '_te_' + str(targetEnd) + '_model.pkl'
    saveFilePath = XGBoostEnsemble.train(targetStart=targetStart, targetEnd=targetEnd, windowEnd=windowEnd, savePath='models/mortality_prediction', saveFileName=saveFileName)
    return('Successfully created and stored the models at: ' + saveFilePath)


if __name__ == "__main__":
    app.run(debug=True)

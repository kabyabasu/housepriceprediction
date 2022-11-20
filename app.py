import pickle
import json

from flask import Flask,request,app,jsonify,url_for,render_template
import numpy as np
import pandas as pd

app = Flask(__name__)
regModel = pickle.load(open('regression.pkl','rb'))
scaler = pickle.load(open('scaler.pkl','rb'))


@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict_api',methods=['POST'])
def predict_api():
    data = request.json['data']
    data_trasform= scaler.tranform(np.array(list(data.values())).reshape(1,-1))
    output = regModel.predict(data_trasform)
    return jsonify(output[0])







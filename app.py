import pickle
import json

from flask import Flask,request,app,jsonify,url_for,render_template
import numpy as np
import pandas as pd

app = Flask(__name__)
regModel = pickle.load(open('regression.pkl','rb'))
scalar = pickle.load(open('scaler.pkl','rb'))


@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict_api',methods=['POST'])
def predict_api():
    # data = request.json['data']
    # data_trasform= scaler.transform(np.array(list(data.values())).reshape(1,-1))
    # output = regModel.predict(data_trasform)
    # return jsonify(output[0])

    data=request.json['data']
    print(data)
    print(np.array(list(data.values())).reshape(1,-1))
    new_data=scalar.transform(np.array(list(data.values())).reshape(1,-1))
    output=regModel.predict(new_data)
    print(output[0])
    return jsonify(output[0])

@app.route('/predict',methods=['POST'])
def predict():
    data = map(lambda n: n.astype(float), list(request.form.values()))
    print(data)
    new_data = scalar.transform(data)
    output = regModel.predict(new_data)[0]
    return render_template('home.html',prediction_text = "This house price prediction is {}".format(output))


if __name__ == "__main__":
    app.run(debug=True)
from flask import Flask, request, jsonify, render_template

import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

application = Flask(__name__)
app = application


### import ridge regresser and standard scaler pickle

ridge = pickle.load(open('models/ridge.pkl', 'rb'))
scaler = pickle.load(open('models/scaler.pkl', 'rb'))

@app.route("/")
def index():
    return render_template('index.html')

@app.route('/predictdata', methods = ['GET', 'POST'])
def predict_datapoint():
    if request.method=="POST":
        Temperature =  float(request.form.get('Temperature'))
        RH =  float(request.form.get('RH'))
        Ws =  float(request.form.get('Ws'))
        Rain =  float(request.form.get('Rain'))
        FFMC =  float(request.form.get('FFMC'))
        DMC =  float(request.form.get('DMC'))
        ISI =  float(request.form.get('ISI'))
        Region =  float(request.form.get('Region'))
        
        new_data = scaler.transform([[Temperature, RH, Ws, Rain, FFMC, DMC, ISI, Region]])
        result = ridge.predict(new_data)

        print("I am here")
        print(result)
        print(result[0])

        return render_template("home.html", results =round(result[0], 2))
        
    else:
        return render_template("home.html")

if __name__ =="__main__":
    app.run(host="0.0.0.0")
 
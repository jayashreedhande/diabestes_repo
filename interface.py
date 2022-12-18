from flask import Flask,jsonify,request,render_template
import config_file
import sklearn
import pandas as pd
import numpy as np
from utils import DiabetesData

app = Flask(__name__)
@app.route('/')
def hello_flask():

    return render_template("index.html")

@app.route('/predict_diabetes',methods = ['GET',"POST"])

def prediction():
    if request.method == 'POST':
        data = request.form

        print('data :',data)

        diab_data = DiabetesData(data)
        outcome = diab_data.get_predicted_class()
        print("********",outcome)
        return render_template('after_index.html',data = outcome)
        # return  jsonify({'outcome: ':str(outcome)})

if __name__ == "__main__":
        app.run(host='0.0.0.0', port=8080)
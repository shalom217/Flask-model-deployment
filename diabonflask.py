import numpy as np
from flask import Flask, request, render_template
import pickle

app = Flask(__name__)
model = pickle.load(open('diabyRF.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('check2.html')#check2 and index1

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    a=float(request.form['pg'])
    b=float(request.form['glu'])
    c=float(request.form['bp'])
    d=float(request.form['insu'])
    e=float(request.form['bmi'])
    f=float(request.form['age'])                    
    prediction = model.predict([[a,b,c,d,e,f]])
    if prediction==1:
    	return render_template('check2.html', prediction_text='Patient will be Diabetic :-(')
    else:
    	return render_template('check2.html', prediction_text='Patient will not be Diabetic :-)')



if __name__ == "__main__":
    app.run(debug=True)
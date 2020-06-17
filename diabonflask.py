from numpy import *
from flask import Flask, request, render_template
import pickle

app = Flask(__name__)#just a module in python
model = pickle.load(open('diaBalanced_tuned_notscaled.pkl', 'rb'))#loading the saved model

@app.route('/')#will route URL with the function
def home():
    return render_template('check2.html')#will redirect to the template,#check2 and index1

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    a=float(request.form['pg'])
    b=float(request.form['glu'])
    c=float(request.form['bp'])
    d=float(request.form['st'])    
    e=float(request.form['insu'])
    f=float(request.form['bmi'])
    g=float(request.form['dpf'])    
    h=float(request.form['age'])
    data=[[a,b,c,d,e,f,g,h]]                    
    prediction = model.predict(data)
    if prediction==1:
    	return render_template('check2.html', prediction_text='Patient will be Diabetic :-(')
    elif prediction==0:
    	return render_template('check2.html', prediction_text='Patient will not be Diabetic :-)')



if __name__ == "__main__":#if this code is running other than python then this command will come into existence
    app.run(debug=True)#means it will show the realtime changes done by the user without stopping the command prompt


    
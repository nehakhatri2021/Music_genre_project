
import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle


app = Flask(__name__)
modelKNN=pickle.load(open('KNN_placed.pkl', 'rb')) 
modelRF=pickle.load(open('RandomForest_placed.pkl', 'rb')) 
modelrbf=pickle.load(open('rbf_placed.pkl', 'rb')) 
modelliner=pickle.load(open('liner_placed.pkl', 'rb')) 
modelpoly=pickle.load(open('poly_placed.pkl', 'rb')) 
modelsigmoid=pickle.load(open('sigmoid_placed.pkl', 'rb'))
modeldt=pickle.load(open('decisiontree_placed.pkl', 'rb')) 
modellogic=pickle.load(open('logistic_placed.pkl', 'rb'))

@app.route('/')
def home():
  
    return render_template("index.html")
  
@app.route('/predict',methods=['GET'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    ten = float(request.args.get('ten') or 0)
    twelve = float(request.args.get('twell') or 0)
    btech = float(request.args.get('btech') or 0)
    seven = float(request.args.get('seven') or 0)
    six = float(request.args.get('six') or 0)
    five = float(request.args.get('five') or 0)
    final = float(request.args.get('final') or 0)
    medium = float(request.args.get('med') or 0)

    prediction1 = modelKNN.predict([[ten, twelve, btech, seven, six, five, final, medium]])
    if prediction1==[0]:
      print("placed")
    else:
      print("not placed")


    prediction2 = modelRF.predict([[ten, twelve, btech, seven, six, five, final, medium]])
    if prediction2==[0]:
      print("placed")
    else:
      print("not placed")

    prediction3 = modelrbf.predict([[ten, twelve, btech, seven, six, five, final, medium]])
    if prediction3==[0]:
      print("placed")
    else:
      print("not placed")

    prediction4 = modelliner.predict([[ten, twelve, btech, seven, six, five, final, medium]])
    if prediction4==[0]:
      print("placed")
    else:
      print("not placed")

    prediction5 = modelpoly.predict([[ten, twelve, btech, seven, six, five, final, medium]])
    if prediction5==[0]:
      print("placed")
    else:
      print("not placed")

    prediction6 = modelsigmoid.predict([[ten, twelve, btech, seven, six, five, final, medium]])
    if prediction6==[0]:
      print("placed")
    else:
      print("not placed")

    prediction7 = modeldt.predict([[ten, twelve, btech, seven, six, five, final, medium]])
    if prediction7==[0]:
      print("placed")
    else:
      print("not placed")

    prediction8 = modellogic.predict([[ten, twelve, btech, seven, six, five, final, medium]])
    if prediction8==[0]:
      print("placed")
    else:
      print("not placed")
    
    return render_template('index.html', prediction_text1='{}'.format(prediction1), prediction_text2='{}'.format(prediction2),prediction_text3='{}'.format(prediction3),prediction_text4='{}'.format(prediction4),prediction_text5='{}'.format(prediction5),prediction_text6='{}'.format(prediction6),prediction_text7='{}'.format(prediction7),prediction_text8='{}'.format(prediction8))


if __name__ == "__main__":
    app.run(debug=True)
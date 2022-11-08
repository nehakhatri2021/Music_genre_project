import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle


app = Flask(__name__)
classifier_dt = pickle.load(open('dtmodel.pkl','rb'))
classifier_knn = pickle.load(open('rfmodel.pkl','rb'))
classifier_svm = pickle.load(open('nbmodel.pkl','rb'))
classifier_rf = pickle.load(open('rfmodel.pkl','rb'))
classifier_NB = pickle.load(open('nbmodel.pkl','rb'))

@app.route('/predict1')
def home():
  
    return render_template("index.html")
  
@app.route('/predict',methods=['GET'])
def predict():

    
    popularity = float(request.args.get('popularity'))
    danceability = float(request.args.get('danceablilty'))
    key = float(request.args.get('key'))
    loudness_trans = float(request.args.get('loudness_trans'))
    mode = int(request.args.get('mode'))
    speechiness_trans = float(request.args.get('speechiness_trans'))
    acoustiness_trans = float(request.args.get('acoustiness_trans'))
    intrumentalness_trans = float(request.args.get('instrumentalness_trans'))
    liveness_trans = float(request.args.get('liveness_trans'))
    valence = float(request.args.get('valence'))
    tempo_trans = float(request.args.get('tempo_trans'))
    duration_in_ms = float(request.args.get('duration_in_ms'))
    time_signature = int(request.args.get('time_ignature'))
    
    


    
# CreditScore	Geography	Gender	Age	Tenure	Balance	NumOfProducts	HasCrCard	IsActiveMember	EstimatedSalary	
    Model = (request.args.get('Model'))

    if Model=="Random Forest":
      prediction = classifier_dt.predict([[popularity, danceability, key, loudness_trans, mode, speechiness_trans, acoustiness_trans, instrumentalness_trans, liveness_trans, valence, tempo_trans, duration_in_ms, time_signature]])

    elif Model=="Decision Tree":
      prediction = classifier_dt.predict([[popularity, danceability, key, loudness_trans, mode, speechiness_trans, acoustiness_trans, instrumentalness_trans, liveness_trans, valence, tempo_trans, duration_in_ms, time_signature]])

    elif Model=="KNN":
      prediction = classifier_svm.predict([[popularity, danceability, key, loudness_trans, mode, speechiness_trans, acoustiness_trans, instrumentalness_trans, liveness_trans, valence, tempo_trans, duration_in_ms, time_signature]])

    elif Model=="SVM":
      prediction = classifier_rf.predict([[popularity, danceability, key, loudness_trans, mode, speechiness_trans, acoustiness_trans, instrumentalness_trans, liveness_trans, valence, tempo_trans, duration_in_ms, time_signature]])

    else:
      prediction = classifier_NB.predict([[popularity, danceability, key, loudness_trans, mode, speechiness_trans, acoustiness_trans, instrumentalness_trans, liveness_trans, valence, tempo_trans, duration_in_ms, time_signature]])

    
    if prediction == [0]:
      return render_template('index.html', prediction_text='Music belongs to 0 group', extra_text =" as per Prediction by " + Model)
    
    elif prediction ==[1]:
      return render_template('index.html', prediction_text='Music belongs to 1 group', extra_text ="as per Prediction by " + Model)

    elif prediction ==[2]:
      return render_template('index.html', prediction_text='Music belongs to 2nd group', extra_text ="as per Prediction by " + Model)
    elif prediction ==[3]:
      return render_template('index.html', prediction_text='Music belongs to 3rd group', extra_text ="as per Prediction by " + Model)
    elif prediction ==[4]:
      return render_template('index.html', prediction_text='Music belongs to 4 group', extra_text ="as per Prediction by " + Model)
     elif prediction ==[5]:
      return render_template('index.html', prediction_text='Music belongs to 5 group', extra_text ="as per Prediction by " + Model)
    elif prediction ==[6]:
      return render_template('index.html', prediction_text='Music belongs to 6 group', extra_text ="as per Prediction by " + Model)
    elif prediction ==[7]:
      return render_template('index.html', prediction_text='Music belongs to 7 group', extra_text ="as per Prediction by " + Model)
    elif prediction ==[8]:
      return render_template('index.html', prediction_text='Music belongs to 8 group', extra_text ="as per Prediction by " + Model)
    elif prediction ==[9]:
      return render_template('index.html', prediction_text='Music belongs to 9 group', extra_text ="as per Prediction by " + Model)
    else:
      return render_template('index.html', prediction_text='Music belongs to 10 group', extra_text ="as per Prediction by " + Model)

#---------------------------------------------------------

@app.route('/aboutusnew')
def aboutusnew():
    return render_template('aboutus.html')

#----------------------------------------------------------

@app.route('/')
def first():
    return render_template('first.html')


if __name__=="__main__":
    app.run(debug=True)


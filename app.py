#app.py

from flask import Flask, request, jsonify, render_template
import numpy as np
import pickle

app = Flask(__name__)

with open('HeartDisease_App.pkl', 'rb') as file:
    model = pickle.load(file)

@app.route('/')
def home():
    return render_template('home.html', prediction_text=None)

@app.rpute('/predict', methods = ['POST'])
def predict():
    form_data = request.form
    Age = int(form_data['Age'])
    Sex	= int(form_data['Sex'])
    ChestPainType = int(form_data['ChestPainType'])
    RestingBP = int(form_data['RestingBP'])
    Cholesterol = int(form_data['Cholesterol']) 
    FastingBS = int(form_data['FastingBS'])
    RestingECG = int(form_data['RestingECG'])
    MaxHR = int(form_data['MaxHR'])
    ExerciseAngina  = int(form_data['ExerciseAngina'])
    Oldpeak = float(form_data['Oldpeak'])
    ST_Slope = int(form_data['ST_Slope'])

    input_data = np.array([[Age,	Sex,	ChestPainType,	RestingBP,	Cholesterol,	FastingBS,	RestingECG,	MaxHR,	ExerciseAngina,	Oldpeak, ST_Slope]])

    prediction = model.predict(input_data)
    output = "Patient has Heart Disease" if prediction[0] == 1 else "Patient doesn't have Heart Diease"

    return render_template('home.html', f"<h2>Prediction: {output}</h2>")

if __name__ == "__main__":
    app.run(debug=True)    
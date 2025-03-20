#app.py

from flask import Flask, request, jsonify, render_template
import numpy as np
import pickle
import pandas as pd

app = Flask(__name__)

with open('HeartFailure_App.pkl', 'rb') as file:
    model = pickle.load(file)

@app.route('/')
def home():
    return render_template('home.html', prediction_text=None)

@app.route('/predict', methods = ['POST'])
def predict():
    try:
        form_data = request.form
        Age = int(form_data['Age'])
        
        # Map numeric form values to categorical labels
        sex_map = {"0": "F", "1": "M"}
        chest_pain_map = {"0": "TA", "1": "ATA", "2": "NAP", "3": "ASY"}
        resting_ecg_map = {"0": "Normal", "1": "ST", "2": "LVH"}
        exercise_angina_map = {"0": "N", "1": "Y"}
        st_slope_map = {"0": "Down", "1": "Flat", "2": "Up"}
        
        Sex = sex_map[form_data['Sex']]
        ChestPainType = chest_pain_map[form_data['ChestPainType']]
        RestingBP = int(form_data['RestingBP'])
        Cholesterol = float(form_data['Cholesterol'])
        FastingBS = int(form_data['FastingBS'])
        RestingECG = resting_ecg_map[form_data['RestingECG']]
        MaxHR = int(form_data['MaxHR'])
        ExerciseAngina = exercise_angina_map[form_data['ExerciseAngina']]
        Oldpeak = float(form_data['Oldpeak'])
        ST_Slope = st_slope_map[form_data['ST_Slope']]

    except ValueError:
        return jsonify({"error": "Invalid input: Please enter valid numbers"})

    data_dict = {'Age': Age, 'Sex': Sex, 'ChestPainType': ChestPainType, 'RestingBP': RestingBP, 'Cholesterol': Cholesterol, 'FastingBS': FastingBS, 'RestingECG': RestingECG, 'MaxHR': MaxHR, 'ExerciseAngina': ExerciseAngina, 'Oldpeak': Oldpeak, 'ST_Slope': ST_Slope}
    
    data_df = pd.DataFrame.from_dict([data_dict])

    print(data_df.head())
    print(data_df.dtypes)

    cols_to_convert = ['Oldpeak', 'Cholesterol', 'RestingBP']

    # Ensure numeric conversion
    data_df[cols_to_convert] = data_df[cols_to_convert].apply(pd.to_numeric, errors='coerce')

    # Debugging: Print unique values
    print("Unique values before replacement:")
    for col in cols_to_convert:
        print(f"{col}: {data_df[col].unique()}")

    # Replace 0 with NaN
    try:
        data_df[cols_to_convert] = data_df[cols_to_convert].replace(0, np.nan)
        print("Replacement successful!")
    except Exception as e:
        print("Error replacing 0 with NaN:", str(e))


    with open('SimpleImputer.pkl', 'rb') as file:
        simpleimputer = pickle.load(file)
    
    #imputing the values
    data_df[['Oldpeak', 'Cholesterol']] = simpleimputer.transform(data_df[['Oldpeak', 'Cholesterol']])
    
    print('before ohe encoding')

    print(data_df.head())
    print(data_df.dtypes)

    with open('OneHotEncoder.pkl', 'rb') as file:
        oheencoder = pickle.load(file)

    #onehotencoding
    oheencoded_array = oheencoder.transform(data_df[['ChestPainType']])
    print('one encoded array =', oheencoded_array)
    oheencoded_data_df = pd.DataFrame(oheencoded_array, columns=oheencoder.get_feature_names_out(['ChestPainType']))
    
    print('before label encoding')

    print(data_df.head())
    print(data_df.dtypes)

    with open('LabelEncoders.pkl', 'rb') as file:
        labelencoders = pickle.load(file)



    #labelencoding
    for col in ['FastingBS', 'ExerciseAngina', 'Sex']:
        data_df[col] = labelencoders[col].transform(data_df[col])

    with open('OrdinalEncoder.pkl', 'rb') as file:
        ordinalencoder = pickle.load(file)
    
    #ordinalencoding
    data_df[['ST_Slope',  'RestingECG']] = ordinalencoder.transform(data_df[['ST_Slope', 'RestingECG']])

    data_df.drop(['RestingBP', 'ChestPainType'], axis=1, inplace=True)


    #resetting the indexes
    data_df.reset_index(drop=True, inplace=True)
    oheencoded_data_df.reset_index(drop=True, inplace=True)

    final_data_df = pd.concat([data_df, oheencoded_data_df], axis=1)

    input_data = final_data_df.to_numpy()

    print('input data =', input_data)

    prediction = model.predict(input_data)
    output = "Model Prediction: Patient has Heart Disease" if prediction[0] == 1 else "Model Prediction: Patient doesn't have Heart Diease"

    return render_template('home.html', prediction_text=output)


if __name__ == "__main__":
    app.run(debug=True)    
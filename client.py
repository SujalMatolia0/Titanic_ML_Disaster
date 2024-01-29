from flask import Flask, render_template, request
import numpy as np
import joblib

app = Flask(__name__)

model = joblib.load("predicting_model.joblib")
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        passId = 312
        pclass = int(request.form['pclass'])
        sex = int(request.form['sex'])
        age = float(request.form['age'])
        sibsp = int(request.form['sibsp'])
        parch = int(request.form['parch'])
        embarked = str(request.form['embarked'])
        fare = 262.3750
        print(" Recieved from Data - \n Pclass : {} \n Sex : {} \n Age : {} \n Siblings\Spouse : {} \n Parch : {} \n Embarked : {}".format(pclass, sex, age, sibsp, parch, embarked))
        if sex:
            Male = 0.0
            Female = 1.0
        else:
            Male = 1.0
            Female = 0.0
        if (embarked == 's'):
            C = 0.0
            Q = 0.0
            S = 1.0
        elif (embarked == 'c'):
            C = 1.0
            Q = 0.0
            S = 0.0
        else:
            C = 0.0
            Q = 1.0
            S = 0.0
        input_data = np.array([[passId, pclass, age, sibsp, parch, fare,  C, S, Q, Female, Male]], dtype = object)
        print("Input Data:", input_data)

        prediction = model.predict(input_data)[0]
        print("The prediction is : ", prediction)
        return render_template('result.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)

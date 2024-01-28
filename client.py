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
        pclass = float(request.form['pclass'])
        sex = float(request.form['sex'])
        age = float(request.form['age'])
        sibsp = int(request.form['sibsp'])
        parch = int(request.form['parch'])
        embarked = int(request.form['embarked'])

        input_data = np.array([[pclass, sex, age, sibsp, embarked]])
        print("Input Data:", input_data)

        prediction = model.predict(input_data)[0]

        return render_template('result.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)

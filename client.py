from flask import Flask, render_template, request
import numpy as np
from sklearn.externals import joblib

app = Flask(__name__)

model = joblib.load("main.py")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        pclass = float(request.form['pclass'])
        sex = float(request.form['sex'])
        age = float(request.form['age'])
        fare = float(request.form['fare'])

        input_data = np.array([[pclass, sex, age, fare]])
        prediction = model.predict(input_data)[0]

        return render_template('result.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)

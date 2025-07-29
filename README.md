Titanic Survival Predictor
=========================

A machine learning project that predicts passenger survival on the Titanic using Random Forest classification.

Features:
- Machine Learning Pipeline
- Interactive Web App
- Probability Display
- Explainability of predictions

Requirements:
- Python 3.8+
- Required packages: streamlit, scikit-learn, pandas, numpy, joblib, pillow

Installation:
1. Clone the repository:
   git clone https://github.com/SujalMatolia0/Titanic_ML_Disaster.git
   cd Titanic_ML_Disaster

2. Install dependencies:
   pip install -r requirements.txt

3. Download the Kaggle Titanic dataset (train.csv and test.csv)

Usage:
- Training: python main.py --train
- Web App: streamlit run main.py

Project Structure:
main.py - Main application and ML pipeline
train.csv - Training data
test.csv - Test data
requirements.txt - Python dependencies

Model Performance:
- Accuracy: ~80%
- Key Features: Passenger class, Gender, Age, Ticket fare

License: MIT

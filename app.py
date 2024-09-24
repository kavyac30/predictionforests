from flask import Flask, request, render_template
import joblib
import numpy as np
import pickle
from sklearn.preprocessing import LabelEncoder
app = Flask(__name__)

# Load the trained model
model = joblib.load('model.pkl')
with open('label_encoder.pkl','rb')as f:
    le_state= pickle.load(f)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/home')
def home():
    return render_template('home.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        # Extract input values from the form
        States = request.form['state']
        temperature = float(request.form['temperature'])
        rain = float(request.form['rain'])
        humidity = float(request.form['humidity'])
        wind_speed = float(request.form['wind speed'])
        # oxygen = float(request.form['oxygen'])
        
        
        state_encoded = le_state.transform([States])[0]

        # Create feature array
        features = np.array([[ state_encoded, temperature, rain, humidity,wind_speed]])

        # Make prediction
        prediction = model.predict(features)
        result = 'fire' if prediction[0] == 1 else 'No fire'

        return render_template('result.html', result=result)
    
    return render_template('predict.html')

@app.route('/instructions')
def instructions():
    return render_template('predict.html', show_instructions=True)

@app.route('/back_to_predict')
def back_to_predict():
    return render_template('predict.html', show_instructions=False)

if __name__ == '__main__':
    app.run(debug=True)
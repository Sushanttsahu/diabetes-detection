from flask import Flask, render_template, request
import numpy as np
import joblib

app = Flask(__name__)

# Load the trained model
model = joblib.load('diabetes_model.pkl')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get form data
        features = [float(request.form[key]) for key in request.form]
        
        # Convert to NumPy array and reshape for prediction
        features_array = np.array(features).reshape(1, -1)
        
        # Predict using the model
        prediction = model.predict(features_array)
        result = "Diabetic" if prediction[0] == 1 else "Not Diabetic"

        return render_template('result.html', prediction=result)
    except Exception as e:
        return f"Error occurred: {str(e)}"

if __name__ == '__main__':
    app.run(debug=True)

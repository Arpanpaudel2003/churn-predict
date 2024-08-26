from flask import Flask, render_template, request, jsonify
import numpy as np
import pickle

app = Flask(__name__)

# Load the model, scaler, and label encoders
model = pickle.load(open("model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))
label_encoders = pickle.load(open("label_encoders.pkl", "rb"))

@app.route('/')
def home():
    return render_template('index.html')

@app.route("/Churn", methods=['POST'])
def churn():
    try:
        # Extract features from form input
        input_features = [request.form.get(field, 0) for field in [
            "Account length", "Area code", "International plan", "Voice mail plan",
            "Number vmail messages", "Total day minutes", "Total day calls",
            "Total day charge", "Total eve minutes", "Total eve calls",
            "Total eve charge", "Total night minutes", "Total night calls",
            "Total night charge", "Total intl minutes", "Total intl calls",
            "Total intl charge", "Customer service calls"
        ]]
        
        # Encode categorical features
        for i, column in enumerate(["International plan", "Voice mail plan"]):
            input_features[i + 2] = label_encoders[column].transform([input_features[i + 2]])[0]
        
        # Convert all inputs to float and form into an array
        float_features = [float(x) for x in input_features]
        features = np.array([float_features])
        
        # Scale the features
        scaled_features = scaler.transform(features)
        
        # Make prediction
        prediction = model.predict(scaled_features)[0]
        
        # Map the prediction to the original label
        prediction_label = label_encoders["Churn"].inverse_transform([prediction])[0]
        
        return render_template("index.html", prediction_text=f"The Churn prediction is: {prediction_label}")
    
    except ValueError as e:
        return render_template("index.html", prediction_text=f"Invalid input data: {str(e)}")
    except Exception as e:
        return render_template("index.html", prediction_text=f"Error: {str(e)}")

if __name__ == "__main__":
    app.run(port=3000, debug=True)

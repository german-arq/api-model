import boto3
import joblib
from flask import Flask, request, jsonify

# AWS Resource
s3 = boto3.resource('s3')
s3.meta.client.download_file('german-credit-22', 'model/final_model.joblib', 'final_model.joblib')

# Load the model
model = joblib.load('final_model.joblib')

#Create a flask app
app = Flask(__name__)

@app.route("/")
def index():
    return "Hi Flask"

# Create an endpoint for predict
@app.route("/predict", methods=["POST"])
def predict():
    request_data = request.get_json()
    age = request_data['age']
    credit_amount = request_data['credit_amount']
    duration = request_data['duration']
    sex = request_data['sex']
    purpose = request_data['purpose']
    housing = request_data['housing']

    request_data = f"Age: {age}, Credit Amount: {credit_amount}, Duration: {duration} Sex: {sex}, Purpose: {purpose}, Housing: {housing}"

    # Make prediction
    prediction = model.predict([[age, credit_amount, duration, sex, purpose, housing]])
    #prediction = prediction[0]
    #return jsonify({"prediction": prediction, "request_data": request_data})
    return jsonify(str(prediction))


if __name__ == "__main__":
    app.run(host='0.0.0.0')

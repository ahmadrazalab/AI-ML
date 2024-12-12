import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import joblib
import os
import requests
from bs4 import BeautifulSoup

# Step 1: Data Ingestion from Website
def ingest_data_from_website():
    # URL of the website (replace with your website URL)
    url = "https://docs.ahmadraza.in"

    # Fetch the content of the website
    response = requests.get(url)
    if response.status_code != 200:
        print(f"Failed to fetch website content. Status code: {response.status_code}")
        return

    # Parse the website content
    soup = BeautifulSoup(response.text, 'html.parser')

    # Extract text content (modify as per the structure of your website)
    content = []
    for paragraph in soup.find_all('p'):
        content.append(paragraph.get_text())

    # Generate synthetic data for training
    data = pd.DataFrame({
        'Text': content,
        'Label': [len(text.split()) for text in content]  # Label: Number of words in the text
    })

    data.to_csv('website_data.csv', index=False)
    print("Data saved to website_data.csv")
    return data

# Step 2: Training and Testing
def train_model():
    # Load data
    data = pd.read_csv('website_data.csv')
    X = data[['Text']]
    y = data['Label']

    # Preprocess text data (simple example)
    X_processed = X['Text'].apply(lambda x: len(x.split())).values.reshape(-1, 1)

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X_processed, y, test_size=0.2, random_state=42)

    # Train a simple Linear Regression model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Evaluate the model
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    print(f"Model trained. Mean Squared Error: {mse}")

    # Save the model
    os.makedirs('models', exist_ok=True)
    model_path = 'models/linear_regression_model.pkl'
    joblib.dump(model, model_path)
    print(f"Model saved to {model_path}")

    return model_path

# Step 3: Build and Deploy on EC2
# A simple FastAPI app for serving the model
fastapi_code = """
from fastapi import FastAPI
import joblib
import numpy as np

# Load the model
model = joblib.load('linear_regression_model.pkl')

# Create FastAPI app
app = FastAPI()

@app.get('/')
def read_root():
    return {"message": "Welcome to the Website Text Model API"}

@app.post('/predict/')
def predict(text: str):
    feature = len(text.split())
    prediction = model.predict(np.array([[feature]]))[0]
    return {"prediction": prediction}
"""

# Save the FastAPI code to a file
os.makedirs('app', exist_ok=True)
with open('app/main.py', 'w') as f:
    f.write(fastapi_code)
print("FastAPI app code written to app/main.py")

# Step 4: CI/CD Pipeline
cicd_pipeline_code = """
name: ML Model CI/CD Pipeline

on:
  push:
    branches:
      - main

jobs:
  build-and-deploy:
    runs-on: ubuntu-latest

    steps:
    - name: Checkout code
      uses: actions/checkout@v3

    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.9'

    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install scikit-learn pandas numpy fastapi uvicorn joblib beautifulsoup4 requests

    - name: Train model
      run: |
        python train.py

    - name: Deploy to EC2
      env:
        AWS_ACCESS_KEY_ID: ${{ secrets.AWS_ACCESS_KEY_ID }}
        AWS_SECRET_ACCESS_KEY: ${{ secrets.AWS_SECRET_ACCESS_KEY }}
      run: |
        scp -i ${{ secrets.EC2_KEY }} -r app/ ec2-user@<EC2_PUBLIC_IP>:/home/ec2-user/app/
        ssh -i ${{ secrets.EC2_KEY }} ec2-user@<EC2_PUBLIC_IP> 'nohup uvicorn app.main:app --host 0.0.0.0 --port 8000 &'
"""

# Save CI/CD pipeline to a file
with open('.github/workflows/main.yml', 'w') as f:
    f.write(cicd_pipeline_code)
print("CI/CD pipeline YAML written to .github/workflows/main.yml")

if __name__ == "__main__":
    # Run the entire pipeline
    ingest_data_from_website()
    model_path = train_model()
    print(f"Pipeline completed. Model available at: {model_path}")

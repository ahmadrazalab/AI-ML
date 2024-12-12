# AI-ML


```python
python3 -m venv venv
source venv/bin/activate  # Use venv\\Scripts\\activate for Windows
python3 -m pip install --upgrade pip
pip3 install scikit-learn pandas numpy fastapi uvicorn transformers joblib beautifulsoup4 requests
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip3 install tensorflow
pip install transformers datasets markdown


```

```
python3 train.py
```

This will create:
A website_data.csv file with training data.
A linear_regression_model.pkl in the models/ directory.


### Setting up amazon linux for training

ssh -i <your-key.pem> ec2-user@<EC2_PUBLIC_IP>


```bash
sudo yum update -y
sudo amazon-linux-extras enable python3.8
sudo yum install python3.8 git -y
python3.8 -m ensurepip --upgrade
python3.8 -m pip install --upgrade pip
pip3 install fastapi uvicorn scikit-learn pandas numpy joblib beautifulsoup4 requests
```

scp -i <your-key.pem> -r app/ models/ ec2-user@<EC2_PUBLIC_IP>:/home/ec2-user/


# run 
```
cd /home/ec2-user
nohup uvicorn app.main:app --host 0.0.0.0 --port 8000 &
```


## Step 4: Public Access
####Test the API: Open a browser and navigate to http://<EC2_PUBLIC_IP>:8000/docs. This will show FastAPI's Swagger UI where you can test the /predict/ endpoint.

Predict Example: Use the /predict/ endpoint in Swagger UI or via curl:
```bash
curl -X 'POST' \\
  'http://<EC2_PUBLIC_IP>:8000/predict/' \\
  -H 'accept: application/json' \\
  -H 'Content-Type: application/json' \\
  -d '{ "text": "Your sample input text here." }'
```

```bash
curl -X POST 'http://<EC2_PUBLIC_IP>:8000/predict/' \
     -H 'Content-Type: application/json' \
     -d '{"text": "Sample text to analyze"}'
```


sudo yum install nginx -y
sudo systemctl start nginx
sudo systemctl enable nginx

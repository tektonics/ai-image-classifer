from flask import Flask
from app.models.classifier import ImageClassifier
import torch
import os
from app.utils.aws_utils import AWSManager

app = Flask(__name__)

def initialize_model():
    print("Initializing model...")
    try:
        # First try to load from S3
        print("Attempting to load model from S3...")
        model = ImageClassifier(load_from_s3=True)
        model.eval()
        print("Model loaded successfully from S3")
    except Exception as e:
        print(f"Error loading from S3: {e}")
        print("Falling back to default pretrained model...")
        model = ImageClassifier(load_from_s3=False)
        model.eval()
        print("Default model loaded successfully")
        
        # Attempt to save the default model to S3
        print("Saving default model to S3...")
        model.save_model_to_s3()
    
    return model

# Initialize the model
model = initialize_model()

from app import routes

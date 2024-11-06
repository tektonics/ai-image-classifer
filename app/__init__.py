from flask import Flask
from app.models.classifier import ImageClassifier
import torch
import os

app = Flask(__name__)

# Initialize the model
print("Loading ViT model...")
model = ImageClassifier()
model.eval()
print("Model loaded successfully")

from app import routes

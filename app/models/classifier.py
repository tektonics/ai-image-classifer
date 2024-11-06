from transformers import ViTForImageClassification, ViTImageProcessor
import torch
import torch.nn as nn
from app.utils.aws_utils import AWSManager

class ImageClassifier(nn.Module):
    def __init__(self, load_from_s3=False):
        super(ImageClassifier, self).__init__()
        self.model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224')
        self.processor = ViTImageProcessor.from_pretrained(
            'google/vit-base-patch16-224',
            use_fast=True
        )
        self.labels = self.model.config.id2label
        
        if load_from_s3:
            self.load_model_from_s3()
    
    def load_model_from_s3(self):
        aws_manager = AWSManager()
        model_buffer = aws_manager.download_model()
        if model_buffer:
            state_dict = torch.load(model_buffer, weights_only=True)
            self.model.load_state_dict(state_dict)
            print("Model loaded successfully from S3")
        else:
            print("Failed to load model from S3")
    
    def save_model_to_s3(self):
        aws_manager = AWSManager()
        success = aws_manager.upload_model(self.model)
        if success:
            print("Model saved successfully to S3")
        else:
            print("Failed to save model to S3")

    def forward(self, x):
        outputs = self.model(x)
        return outputs.logits

    def preprocess_image(self, image):
        inputs = self.processor(
            images=image, 
            return_tensors="pt",
            do_rescale=True,
            do_normalize=True
        )
        return inputs.pixel_values

    def get_class_name(self, idx):
        label = self.labels[idx]
        return label.replace("_", " ").title()

    def get_top_predictions(self, outputs, top_k=3):
        probs = torch.nn.functional.softmax(outputs, dim=1)
        top_probs, top_indices = torch.topk(probs, top_k)
        
        predictions = []
        for i in range(top_k):
            class_name = self.get_class_name(top_indices[0][i].item())
            confidence = top_probs[0][i].item() * 100
            predictions.append({
                'class': class_name,
                'confidence': f'{confidence:.2f}%'
            })
        return predictions

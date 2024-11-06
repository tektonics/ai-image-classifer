from transformers import ViTForImageClassification, ViTImageProcessor
import torch
import torch.nn as nn

class ImageClassifier(nn.Module):
    def __init__(self):
        super(ImageClassifier, self).__init__()
        self.model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224')
        self.processor = ViTImageProcessor.from_pretrained(
            'google/vit-base-patch16-224',
            use_fast=True
        )
        self.labels = self.model.config.id2label
        
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

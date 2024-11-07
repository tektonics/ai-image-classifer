# AI Image Classifier

A Flask-based web application that uses Vision Transformer (ViT) to classify images. The application features AWS S3 integration for model storage and management.

## Features

- Image classification using Vision Transformer (ViT)
- AWS S3 integration for model storage
- Real-time classification results
- Support for various image formats
- Progress tracking for model uploads/downloads
- Fallback to pretrained models when S3 is unavailable

## Tech Stack

- Python 3.9
- Flask
- PyTorch
- Transformers (Hugging Face)
- AWS S3
- Docker

## Setup

1. Clone the repository
2. Create a `.env` file with your AWS credentials:
```env
AWS_ACCESS_KEY_ID=your_access_key
AWS_SECRET_ACCESS_KEY=your_secret_key
AWS_REGION=your_region
S3_BUCKET=your_bucket_name
MODEL_KEY=models/vit-model.pth
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Run the application:
```bash
python run.py
```

## Docker Deployment

Build and run using Docker:
```bash
docker build -t ai-classifier .
docker run -p 5000:5000 ai-classifier
```

## Testing

Run AWS integration tests:
```bash
python -m tests.test_aws
```

## API Endpoints

- `GET /`: Main interface
- `POST /classify`: Image classification endpoint

## AWS Deployment

This application is deployed on AWS Elastic Beanstalk. To deploy:

1. Install EB CLI:
```bash
pip install awsebcli
```

2. Initialize EB application:
```bash
eb init -p python-3.9 ai-classifier
```

3. Create and deploy to production environment:
```bash
eb create production
```

4. Set environment variables:
```bash
eb setenv \
  AWS_ACCESS_KEY_ID=your_key \
  AWS_SECRET_ACCESS_KEY=your_secret \
  AWS_REGION=your_region \
  S3_BUCKET=your_bucket
```

The application will be available at the provided Elastic Beanstalk URL.



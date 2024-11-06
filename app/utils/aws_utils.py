import boto3
from botocore.exceptions import ClientError
import torch
from app.config import AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_REGION, S3_BUCKET, MODEL_KEY
import io
import sys
import threading

class ProgressPercentage:
    def __init__(self, file_obj):
        if isinstance(file_obj, io.BytesIO):
            self._size = file_obj.getbuffer().nbytes
        else:
            self._size = 0  # Will be set on first call for download
        self._seen_so_far = 0
        self._lock = threading.Lock()

    def __call__(self, bytes_amount):
        with self._lock:
            self._seen_so_far += bytes_amount
            if self._size == 0:
                print(f"\rDownloaded {self._seen_so_far} bytes", end='')
            else:
                percentage = (self._seen_so_far / self._size) * 100
                print(f"\rProgress: {self._seen_so_far}/{self._size} bytes ({percentage:.2f}%)", end='')
            sys.stdout.flush()

class AWSManager:
    def __init__(self):
        self.s3_client = boto3.client(
            's3',
            aws_access_key_id=AWS_ACCESS_KEY_ID,
            aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
            region_name=AWS_REGION
        )
        self.bucket = S3_BUCKET

    def upload_model(self, model, model_key=None):
        if model_key is None:
            model_key = MODEL_KEY
            
        print(f"Preparing model for upload to {self.bucket}/{model_key}...")
        buffer = io.BytesIO()
        torch.save(model.state_dict(), buffer)
        buffer.seek(0)
        
        try:
            print("Starting upload...")
            self.s3_client.upload_fileobj(
                buffer, 
                self.bucket, 
                model_key,
                Callback=ProgressPercentage(buffer)
            )
            print("\nUpload completed successfully!")
            return True
        except ClientError as e:
            print(f"\nError uploading model to S3: {e}")
            return False
        except Exception as e:
            print(f"\nUnexpected error: {e}")
            return False

    def download_model(self, model_key=None):
        if model_key is None:
            model_key = MODEL_KEY
            
        buffer = io.BytesIO()
        try:
            print(f"Downloading model from {self.bucket}/{model_key}...")
            self.s3_client.download_fileobj(
                self.bucket, 
                model_key, 
                buffer,
                Callback=ProgressPercentage(buffer)
            )
            buffer.seek(0)
            print("\nDownload completed successfully!")
            return buffer
        except ClientError as e:
            print(f"\nError downloading model from S3: {e}")
            return None
        except Exception as e:
            print(f"\nUnexpected error: {str(e)}")
            return None

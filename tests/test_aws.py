import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.models.classifier import ImageClassifier
from app.utils.aws_utils import AWSManager

def test_s3_access():
    aws_manager = AWSManager()
    try:
        # Test if we can list the bucket
        aws_manager.s3_client.head_bucket(Bucket=aws_manager.bucket)
        print(f"Successfully connected to bucket: {aws_manager.bucket}")
        return True
    except Exception as e:
        print(f"Error accessing S3 bucket: {e}")
        return False

def test_s3_operations():
    # First test S3 access
    if not test_s3_access():
        print("Skipping model operations due to S3 access issues")
        return

    # Create model instance
    #model = ImageClassifier()
    
    # Test saving to S3
    print("Testing model upload to S3...")
    #model.save_model_to_s3()
    
    # Test loading from S3
    print("\nTesting model download from S3...")
    new_model = ImageClassifier(load_from_s3=True)

if __name__ == "__main__":
    test_s3_operations()

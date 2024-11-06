from flask import render_template, request, jsonify
from app import app, model
from PIL import Image
import torch
import io
import torch.nn.functional as F

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/classify', methods=['POST'])
def classify_image():
    if not model:
        return jsonify({'error': 'Model not initialized'}), 500
        
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400
    
    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No image selected'}), 400

    try:
        image_bytes = file.read()
        image = Image.open(io.BytesIO(image_bytes))
        
        if image.mode in ('RGBA', 'LA') or (image.mode == 'P' and 'transparency' in image.info):
            background = Image.new('RGB', image.size, (255, 255, 255))
            if image.mode == 'RGBA':
                background.paste(image, mask=image.split()[3])
            else:
                background.paste(image)
            image = background
        elif image.mode != 'RGB':
            image = image.convert('RGB')

        try:
            inputs = model.preprocess_image(image)
        except Exception as e:
            print(f"Preprocessing error: {str(e)}")
            return jsonify({'error': 'Error preprocessing image'}), 500

        try:
            with torch.no_grad():
                outputs = model(inputs)
                predictions = model.get_top_predictions(outputs)
                return jsonify({'predictions': predictions})
        except Exception as e:
            print(f"Prediction error: {str(e)}")
            return jsonify({'error': 'Error making prediction'}), 500
            
    except Exception as e:
        print(f"General error: {str(e)}")
        return jsonify({'error': str(e)}), 500

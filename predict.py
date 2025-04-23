from flask import Flask, request, jsonify, send_from_directory
import torch
from PIL import Image
import io
import torchvision.transforms as transforms
import torch.nn.functional as F

from training.train_cnn import PlantDiseaseCNN

import logging

# Initialize Flask app with better configuration
app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # Limit uploads to 16MB

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# GPU configuration for GTX 1050 Ti (4GB VRAM)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.backends.cudnn.benchmark = True  # Enable cuDNN auto-tuner
torch.cuda.empty_cache()  # Clear unused memory

# Load model with error handling
def load_model():
    try:
        model = PlantDiseaseCNN(num_classes=38)
        
        # Load checkpoint with map_location to handle device properly
        checkpoint = torch.load('./best_model.pth', map_location=device)
        
        # Handle different checkpoint formats
        state_dict = checkpoint.get('model_state_dict', checkpoint)
        
        # Load state dict with strict=False to handle potential mismatches
        model.load_state_dict(state_dict, strict=False)
        model = model.to(device)
        model.eval()
        
        logger.info(f"Model loaded successfully on {device}")
        logger.info(f"CUDA memory allocated: {torch.cuda.memory_allocated()/1024**2:.2f} MB")
        return model
    except Exception as e:
        logger.error(f"Model loading failed: {str(e)}")
        raise

model = load_model()

# Define your class names (update with your actual class names)
CLASS_NAMES = ['Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy', 'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew', 'Cherry_(including_sour)___healthy', 'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 'Corn_(maize)___Common_rust_', 'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy', 'Grape___Black_rot', 'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 'Grape___healthy', 'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot', 'Peach___healthy', 'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy', 'Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy', 'Raspberry___healthy', 'Soybean___healthy', 'Squash___Powdery_mildew', 'Strawberry___Leaf_scorch', 'Strawberry___healthy', 'Tomato___Bacterial_spot', 'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold', 'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite', 'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus', 'Tomato___healthy']

# Image transformations (must match training preprocessing)
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

@app.route('/')
def home():
    return send_from_directory('static', 'index.html')

@app.route('/api/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'Empty filename'}), 400

    try:
        # Verify image
        if not file.content_type.startswith('image/'):
            return jsonify({'error': 'File is not an image'}), 400

        # Load and preprocess image with error handling
        image_bytes = file.read()
        image = Image.open(io.BytesIO(image_bytes))
        
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Apply transformations and add batch dimension
        tensor = transform(image).unsqueeze(0).to(device)
        
        # Free up memory from image processing
        del image, image_bytes
        
        # Prediction with memory management
        with torch.no_grad():
            try:
                output = model(tensor)
                probabilities = F.softmax(output, dim=1)
                top_probs, top_idxs = torch.topk(probabilities, 3)  # Get top 3 predictions
                
                # Convert to CPU for JSON serialization
                top_probs = top_probs.cpu().numpy()[0]
                top_idxs = top_idxs.cpu().numpy()[0]
                
                # Prepare response
                results = {
                    'predictions': [
                        {
                            'class': CLASS_NAMES[idx],
                            'confidence': round(float(prob * 100), 2)
                        }
                        for prob, idx in zip(top_probs, top_idxs)
                    ],
                    'device': str(device)
                }
                
                return jsonify(results)
            
            except RuntimeError as e:
                if 'CUDA out of memory' in str(e):
                    torch.cuda.empty_cache()
                    return jsonify({
                        'error': 'GPU memory exhausted',
                        'solution': 'Try with a smaller image or restart the server'
                    }), 500
                raise

    except Exception as e:
        logger.error(f"Prediction failed: {str(e)}", exc_info=True)
        return jsonify({'error': 'Prediction failed', 'details': str(e)}), 500

@app.route('/api/health', methods=['GET'])
def health_check():
    gpu_info = {
        'available': torch.cuda.is_available(),
        'device_name': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        'memory_allocated': f"{torch.cuda.memory_allocated()/1024**2:.2f}MB",
        'memory_cached': f"{torch.cuda.memory_reserved()/1024**2:.2f}MB"
    }
    
    return jsonify({
        'status': 'healthy',
        'model_loaded': True,
        'device': str(device),
        'gpu_info': gpu_info
    })

if __name__ == '__main__':
    # Production-ready configuration
    app.run(
        host='0.0.0.0',
        port=5000,
        debug=False,  # Disable debug mode in production
        threaded=True  # Better for concurrent requests
    )
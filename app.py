from flask import Flask, render_template, request, jsonify
import torch
import torchvision.transforms as transforms
from PIL import Image
import io
import base64
import cv2
import numpy as np
import os
from torchvision import models
import torch.nn as nn
import random  # For demo purposes, replace with actual sensor readings
from datetime import datetime

app = Flask(__name__)

# Configure upload folder
UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Thresholds for alerts (adjusted for Kathmandu climate)
THRESHOLDS = {
    'temperature': {'min': 15, 'max': 30},  # Kathmandu's comfortable range
    'humidity': {'min': 40, 'max': 80},     # in percentage
    'soil_moisture': {'min': 30, 'max': 70}  # in percentage
}

# Plant care recommendations based on conditions
PLANT_CARE_TIPS = {
    'high_temperature': 'Temperature is high for Kathmandu! Consider providing shade or moving plants to a cooler spot.',
    'low_temperature': 'Temperature is low for Kathmandu! Move plants to a warmer location or provide protection.',
    'high_humidity': 'Humidity is high! Improve air circulation to prevent fungal diseases.',
    'low_humidity': 'Humidity is low! Consider using a humidifier or misting the plants.',
    'high_moisture': 'Soil is too wet! Reduce watering frequency and ensure proper drainage.',
    'low_moisture': 'Soil is too dry! Water your plants and consider adding mulch to retain moisture.'
}

# Load the disease detection models
def create_model(num_classes):
    model = models.resnet50(weights=None)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    return model

vegetables = ['Corn', 'Potato', 'Rice', 'Tomato']
models_dict = {}

# Disease classes for each vegetable
disease_classes = {
    'Corn': ['Blight', 'Common_Rust', 'Gray_Leaf_Spot', 'Healthy'],
    'Potato': ['Early_blight', 'Late_blight', 'Healthy'],  # 3 classes
    'Rice': ['Bacterial_leaf_blight', 'Brown_spot', 'Healthy', 'Leaf_blast', 'Leaf_scald', 'Narrow_brown_spot'],  # 6 classes
    'Tomato': [  # 10 classes
        'Bacterial_spot', 'Early_blight', 'Late_blight', 'Leaf_Mold', 
        'Septoria_leaf_spot', 'Spider_mites', 'Target_Spot', 
        'Yellow_Leaf_Curl_Virus', 'Mosaic_virus', 'Healthy'
    ]
}

# Load models with correct number of classes
for vegetable in vegetables:
    model_path = f'{vegetable}_disease_model.pth'
    if os.path.exists(model_path):
        num_classes = len(disease_classes[vegetable])
        model = create_model(num_classes)
        try:
            state_dict = torch.load(model_path, map_location=torch.device('cpu'))
            model.load_state_dict(state_dict)
            model.eval()
            models_dict[vegetable] = model
            print(f"Loaded model for {vegetable} with {num_classes} classes")
        except Exception as e:
            print(f"Error loading model for {vegetable}: {str(e)}")

# Transform for the model
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def predict_disease(image):
    """Predict disease from image"""
    try:
        # Convert to PIL Image if it's not already
        if not isinstance(image, Image.Image):
            image = Image.open(io.BytesIO(image))
        
        # Convert to RGB if necessary
        if image.mode != 'RGB':
            image = image.convert('RGB')
            
        # First check for human presence
        # Convert PIL image to OpenCV format
        opencv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        
        # Load the face cascade classifier
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
        # Convert to grayscale for face detection
        gray = cv2.cvtColor(opencv_image, cv2.COLOR_BGR2GRAY)
        
        # Detect faces
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        
        # If human detected, return immediately
        if len(faces) > 0:
            return {
                'success': True,
                'human_detected': True,
                'message': 'Human detected in image'
            }
        
        # If no human detected, proceed with plant disease detection
        # Transform image for disease detection
        image_tensor = transform(image).unsqueeze(0)
        
        # Get predictions from all models
        best_confidence = -1
        detected_disease = None
        detected_vegetable = None
        
        for vegetable, model in models_dict.items():
            with torch.no_grad():
                outputs = model(image_tensor)
                probabilities = torch.nn.functional.softmax(outputs, dim=1)
                confidence, predicted = torch.max(probabilities, 1)
                
                if confidence.item() > best_confidence:
                    best_confidence = confidence.item()
                    detected_disease = disease_classes[vegetable][predicted.item()]
                    detected_vegetable = vegetable
        
        if detected_disease and best_confidence > 0.5:  # Confidence threshold
            return {
                'success': True,
                'human_detected': False,
                'disease': f'{detected_vegetable} - {detected_disease}',
                'confidence': min(best_confidence * 100, 100)  # Ensure confidence is between 0-100%
            }
        else:
            return {
                'success': False,
                'error': 'No disease detected with high confidence'
            }
    except Exception as e:
        return {
            'success': False,
            'error': str(e)
        }

def get_sensor_data():
    """Get current sensor readings with Kathmandu's typical ranges"""
    # Kathmandu's temperature typically ranges from 12°C to 30°C
    hour = datetime.now().hour
    
    # Simulate daily temperature variation
    if 6 <= hour < 12:  # Morning
        temp_range = (15, 25)
    elif 12 <= hour < 17:  # Afternoon
        temp_range = (20, 30)
    elif 17 <= hour < 22:  # Evening
        temp_range = (18, 25)
    else:  # Night
        temp_range = (12, 18)
    
    # Generate temperature within the appropriate range for the time of day
    temperature = round(random.uniform(temp_range[0], temp_range[1]), 1)
    
    # Humidity is typically higher in the morning and evening
    if 6 <= hour < 10 or 17 <= hour < 20:
        humidity_range = (60, 80)
    else:
        humidity_range = (40, 60)
    
    humidity = round(random.uniform(humidity_range[0], humidity_range[1]), 1)
    
    return {
        'temperature': temperature,
        'humidity': humidity,
        'soil_moisture': round(random.uniform(30, 70), 1)  # Soil moisture remains consistent
    }

def get_alerts(sensor_data):
    """Generate alerts based on sensor readings"""
    alerts = []
    
    # Temperature alerts
    if sensor_data['temperature'] > THRESHOLDS['temperature']['max']:
        alerts.append({
            'type': 'danger',
            'message': PLANT_CARE_TIPS['high_temperature'],
            'sensor': 'temperature',
            'value': sensor_data['temperature'],
            'unit': '°C'
        })
    elif sensor_data['temperature'] < THRESHOLDS['temperature']['min']:
        alerts.append({
            'type': 'warning',
            'message': PLANT_CARE_TIPS['low_temperature'],
            'sensor': 'temperature',
            'value': sensor_data['temperature'],
            'unit': '°C'
        })
    
    # Humidity alerts
    if sensor_data['humidity'] > THRESHOLDS['humidity']['max']:
        alerts.append({
            'type': 'danger',
            'message': PLANT_CARE_TIPS['high_humidity'],
            'sensor': 'humidity',
            'value': sensor_data['humidity'],
            'unit': '%'
        })
    elif sensor_data['humidity'] < THRESHOLDS['humidity']['min']:
        alerts.append({
            'type': 'warning',
            'message': PLANT_CARE_TIPS['low_humidity'],
            'sensor': 'humidity',
            'value': sensor_data['humidity'],
            'unit': '%'
        })
    
    # Soil moisture alerts
    if sensor_data['soil_moisture'] > THRESHOLDS['soil_moisture']['max']:
        alerts.append({
            'type': 'danger',
            'message': PLANT_CARE_TIPS['high_moisture'],
            'sensor': 'soil_moisture',
            'value': sensor_data['soil_moisture'],
            'unit': '%'
        })
    elif sensor_data['soil_moisture'] < THRESHOLDS['soil_moisture']['min']:
        alerts.append({
            'type': 'warning',
            'message': PLANT_CARE_TIPS['low_moisture'],
            'sensor': 'soil_moisture',
            'value': sensor_data['soil_moisture'],
            'unit': '%'
        })
    
    return alerts

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'success': False, 'error': 'No file uploaded'})
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'success': False, 'error': 'No file selected'})
    
    try:
        # Read the image file
        image_bytes = file.read()
        # Save the file
        filename = os.path.join(UPLOAD_FOLDER, file.filename)
        with open(filename, 'wb') as f:
            f.write(image_bytes)
        
        # Predict disease
        result = predict_disease(image_bytes)
        return jsonify(result)
    
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/detect', methods=['POST'])
def detect():
    try:
        data = request.get_json()
        if not data or 'image' not in data:
            return jsonify({'success': False, 'error': 'No image data received'})

        # Get the base64 string and convert to image
        image_data = data['image'].split('base64,')[1]
        image_bytes = base64.b64decode(image_data)
        
        # Predict disease
        result = predict_disease(image_bytes)
        return jsonify(result)

    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/sensor_data')
def sensor_data():
    """Endpoint to get current sensor readings and alerts"""
    data = get_sensor_data()
    alerts = get_alerts(data)
    return jsonify({
        'success': True,
        'data': data,
        'alerts': alerts,
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    })

if __name__ == '__main__':
    app.run(debug=True)

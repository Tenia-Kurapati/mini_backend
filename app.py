import torch
import torch.nn as nn
from torchvision import models, transforms
import pandas as pd
from PIL import Image
from flask import Flask, request, jsonify
import io
import os

print("Loading all components for the Flask application...")

# --- 1. SETUP: LOAD MODEL, CLASS NAMES FILE, AND CSV ---

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- PATHS SECTION (For Local Deployment) ---
# Path to the trained PyTorch model
model_path = os.path.join('model', 'medicinal_leaf_classifier_pytorch.pth')

# Path to the medicinal properties CSV file
csv_path = os.path.join('data', 'medicinal_data.csv')

# Path to the class names file
class_names_path = os.path.join('data', 'class_names.txt')
# --- END OF PATHS SECTION ---

# Image transformations (must be the same as during training)
IMG_HEIGHT = 224
IMG_WIDTH = 224
val_transforms = transforms.Compose([
    transforms.Resize((IMG_HEIGHT, IMG_WIDTH)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Load the class names from our text file
try:
    with open(class_names_path, 'r') as f:
        class_names = [line.strip() for line in f.readlines()]
    num_classes = len(class_names)
except FileNotFoundError:
    print(f"FATAL ERROR: '{class_names_path}' not found. Please check your project structure.")
    class_names = []
    num_classes = 0

# Re-create the model's architecture
if num_classes > 0:
    model = models.resnet50()
    num_ftrs = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(num_ftrs, 1024),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(1024, num_classes)
    )
    # Load the saved weights into the model structure
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval() # Set the model to evaluation mode

# Load the medicinal properties from the CSV file
try:
    properties_df = pd.read_csv(csv_path)
    print("Model, class names, and medicinal database loaded successfully.")
except FileNotFoundError:
    print(f"ERROR: The CSV file was not found at '{csv_path}'.")
    properties_df = pd.DataFrame()


# --- 2. CORE CLASSIFICATION FUNCTION ---

def classify_and_get_info(image_bytes):
    """
    Function to take image bytes, make a prediction, and return structured data.
    """
    if properties_df.empty:
        return {"error": "Could not load medicinal database."}
    if not class_names:
        return {"error": "Could not load model class names."}

    # Open the image from bytes
    image_to_test = Image.open(io.BytesIO(image_bytes))
    
    # Ensure image is in RGB format, as some PNGs have an alpha channel
    if image_to_test.mode != 'RGB':
        image_to_test = image_to_test.convert('RGB')
        
    img_t = val_transforms(image_to_test)
    batch_t = torch.unsqueeze(img_t, 0).to(device)

    with torch.no_grad():
        out = model(batch_t)
        
    probabilities = torch.nn.functional.softmax(out, dim=1)[0]
    top_prob, top_idx = torch.max(probabilities, 0)
    
    predicted_class_index = top_idx.item()
    prediction_confidence = top_prob.item()
    predicted_class_name = class_names[predicted_class_index]

    info = properties_df[properties_df['leaf_name'] == predicted_class_name]

    # Prepare the output data in a JSON-friendly format
    result = {
        "prediction": predicted_class_name.replace('_', ' '),
        "confidence": f"{prediction_confidence:.2%}",
    }

    if not info.empty:
        uses_string = info.iloc[0]['medicinal_uses']
        # Clean up the list of uses
        points = [p.strip().capitalize() for p in uses_string.split(',') if p.strip()]
        
        result["scientific_name"] = info.iloc[0]['scientific_name']
        result["medicinal_uses"] = points
    else:
        result["info"] = "Medicinal properties for this specific leaf are not yet available in our database."
        
    return result


# --- 3. SET UP THE FLASK APPLICATION ---

app = Flask(__name__)

@app.route('/')
def index():
    return "<h1>Medicinal Leaf Classifier Server is running!</h1><p>Send a POST request to /predict with an image file.</p>"

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({"error": "No file part in the request"}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({"error": "No file selected for uploading"}), 400

    if file:
        # Read the image bytes from the file
        img_bytes = file.read()
        # Get the classification result
        classification_result = classify_and_get_info(img_bytes)
        return jsonify(classification_result)

    return jsonify({"error": "Failed to process the file"}), 500

# Run the Flask server
if __name__ == '__main__':
    # Use host='0.0.0.0' to make it accessible on your local network
    app.run(host='0.0.0.0', port=5000, debug=True)
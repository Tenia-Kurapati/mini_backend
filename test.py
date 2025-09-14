import torch
import torch.nn as nn
from torchvision import models, transforms
import pandas as pd
from PIL import Image
import os

def predict_single_image(image_path, model, class_names, properties_df):
    """
    Takes a path to an image and returns a formatted string with the prediction and medicinal info.
    """
    # --- 1. PRE-PROCESS THE IMAGE ---
    try:
        image = Image.open(image_path)
        # Ensure image is in RGB format
        if image.mode != 'RGB':
            image = image.convert('RGB')
    except FileNotFoundError:
        return f"Error: The file was not found at '{image_path}'"

    # Image transformations (must be the same as during training)
    val_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    # Apply transformations and add a batch dimension
    img_t = val_transforms(image)
    batch_t = torch.unsqueeze(img_t, 0).to(device)

    # --- 2. MAKE PREDICTION ---
    model.eval() # Set the model to evaluation mode
    with torch.no_grad():
        out = model(batch_t)
        
    probabilities = torch.nn.functional.softmax(out, dim=1)[0]
    top_prob, top_idx = torch.max(probabilities, 0)
    
    predicted_class_index = top_idx.item()
    prediction_confidence = top_prob.item()
    predicted_class_name = class_names[predicted_class_index]

    # --- 3. FORMAT THE OUTPUT ---
    info = properties_df[properties_df['leaf_name'] == predicted_class_name]

    # Start building the output string
    prediction = predicted_class_name.replace('_', ' ')
    confidence = f"{prediction_confidence:.2%}"
    
    output_string = (
        f"========================================\n"
        f"            PREDICTION RESULT\n"
        f"========================================\n"
        f"Prediction:     {prediction}\n"
        f"Confidence:     {confidence}\n"
        f"----------------------------------------\n"
    )

    if not info.empty:
        scientific_name = info.iloc[0]['scientific_name']
        uses_string = info.iloc[0]['medicinal_uses']
        points = [p.strip().capitalize() for p in uses_string.split(',') if p.strip()]
        
        output_string += f"Scientific Name: {scientific_name}\n\n"
        output_string += "Medicinal Uses:\n"
        for point in points:
            output_string += f"- {point}\n"
    else:
        output_string += "Info: Medicinal properties for this specific leaf are not yet available in our database.\n"
    
    output_string += "========================================"
    
    return output_string

# ======================================================================
#                   MAIN EXECUTION BLOCK
# ======================================================================
if __name__ == '__main__':
    
    # --- IMPORTANT: HARDCODE YOUR IMAGE PATH HERE ---
    # Replace this with the actual path to the image you want to test.
    # Example for Windows: 'C:\\Users\\YourUser\\Pictures\\leaf.jpg'
    # Example for macOS/Linux: '/home/youruser/pictures/leaf.jpg'
    path_to_your_image = r'D:\Projects\Mini\backend\data\aloevera.jpg' 
    # or
    # path_to_your_image = 'D:\\Projects\\Mini\\backend\\data\\aloevera.jpg'

    # --- SETUP: LOAD MODEL AND DATA ---
    print("Loading model and data for testing...")

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Define paths based on project structure
    model_path = os.path.join('model', 'medicinal_leaf_classifier_pytorch.pth')
    csv_path = os.path.join('data', 'medicinal_data.csv')
    class_names_path = os.path.join('data', 'class_names.txt')

    # Load class names
    try:
        with open(class_names_path, 'r') as f:
            class_names = [line.strip() for line in f.readlines()]
        num_classes = len(class_names)
    except FileNotFoundError:
        print(f"FATAL ERROR: Class names file not found at '{class_names_path}'. Exiting.")
        exit()

    # Re-create model architecture
    model = models.resnet50()
    num_ftrs = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(num_ftrs, 1024),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(1024, num_classes)
    )

    # Load the trained weights
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
        model = model.to(device)
    except FileNotFoundError:
        print(f"FATAL ERROR: Model file not found at '{model_path}'. Exiting.")
        exit()

    # Load medicinal properties CSV
    try:
        properties_df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"WARNING: CSV file not found at '{csv_path}'. Medicinal properties will not be available.")
        properties_df = pd.DataFrame() # Create empty dataframe to avoid errors

    print("Model and data loaded successfully.\n")

    # --- RUN PREDICTION ---
    # Call the function with the hardcoded path and the loaded components
    result = predict_single_image(path_to_your_image, model, class_names, properties_df)
    
    # Print the final, formatted result
    print(result)
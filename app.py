# ===================================================================
# FINAL HUGGING FACE APP SCRIPT: app.py
# ===================================================================
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import gradio as gr
import numpy as np
import google.generativeai as genai
import os

print("--- Initializing App ---")

# --- 1. SETUP AND CONFIGURATION ---
# On Hugging Face Spaces, the device will be CPU for the free tier.
device = torch.device("cpu")

# --- 2. DEFINE THE LIST OF 30 CLASS NAMES ---
# This list MUST be in the exact same order as the one you used for training.
chosen_classes_names = [
    'beef_carpaccio', 'beef_tartare', 'hamburger', 'pork_chop', 'steak', 'prime_rib',
    'chicken_curry', 'chicken_quesadilla', 'chicken_wings', 'fish_and_chips',
    'crab_cakes', 'clam_chowder', 'lobster_bisque', 'mussels', 'oysters', 'paella',
    'scallops', 'shrimp_and_grits', 'sushi', 'gnocchi', 'lasagna',
    'macaroni_and_cheese', 'pad_thai', 'pho', 'ramen', 'ravioli', 'risotto',
    'spaghetti_bolognese', 'spaghetti_carbonara', 'pizza'
]
num_classes = len(chosen_classes_names)

# --- 3. LOAD THE TRAINED EFFICIENTNET-B2 MODEL ---
# Define the model architecture first.
model = models.efficientnet_b2(weights=None) # Load architecture without any weights.
num_ftrs = model.classifier[1].in_features
model.classifier[1] = nn.Linear(num_ftrs, num_classes)

# Load your trained weights from the uploaded .pth file.
# map_location=torch.device('cpu') is CRUCIAL for loading a GPU-trained model onto a CPU.
model_path = 'food101_advanced_BEST.pth'
model.load_state_dict(torch.load(model_path, map_location=device))
model.to(device)
model.eval() # Set the model to evaluation mode permanently for this app.
print(f"Model '{model_path}' loaded successfully and set to evaluation mode.")


# --- 4. DEFINE PREPROCESSING AND PIPELINE FUNCTIONS ---
# The same validation/test transform from training.
val_test_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Configure Google Gemini API using Hugging Face Secrets.
try:
    # In Hugging Face, secrets are accessed as environment variables.
    GOOGLE_API_KEY = os.environ.get('GOOGLE_API_KEY')
    genai.configure(api_key=GOOGLE_API_KEY)
    print("Google Generative AI configured successfully.")
except (TypeError, ValueError):
    print("WARNING: GOOGLE_API_KEY secret not found. Recipe generation will fail.")
    GOOGLE_API_KEY = None

# --- Prediction and Recipe Functions ---
def classify_food_image(input_image):
    if input_image is None: return {}, gr.Button(visible=False)
    img = Image.fromarray(input_image.astype('uint8'), 'RGB')
    img_tensor = val_test_transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(img_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)[0]
    confidences = {name: float(prob) for name, prob in zip(chosen_classes_names, probabilities)}
    top_confidence = max(confidences.values())
    if top_confidence < 0.50:
        return confidences, gr.Button(visible=False, value="Get Recipe (Confidence too low)")
    else:
        predicted_class_name = max(confidences, key=confidences.get)
        button_text = f"2. Get Recipe for {predicted_class_name.replace('_', ' ').title()}"
        return confidences, gr.Button(value=button_text, visible=True)

def generate_recipe_with_gemini(predictions):
    if not GOOGLE_API_KEY: return "Error: Gemini API Key is not configured. Please set it in the Space's settings."
    if not predictions: return "First, get a prediction."
    
    predicted_class_name = max(predictions, key=predictions.get)
    dish_name = predicted_class_name.replace('_', ' ').title()
    
    try:
        gemini_model = genai.GenerativeModel('gemini-pro')
        prompt = f"""
        You are a helpful assistant chef. Provide a clear, concise, and easy-to-follow recipe for the dish: {dish_name}.
        Format your response in Markdown with '### Description', '### Ingredients', and '### Instructions' sections.
        """
        yield "🤖 Generating your recipe with Gemini AI... please wait a moment."
        response = gemini_model.generate_content(prompt)
        yield response.text
    except Exception as e:
        return f"An error occurred while calling the Gemini API: {e}"

def clear_outputs():
    return None, None, gr.Button(value="2. Get Recipe", visible=False)

# --- 5. DEFINE AND LAUNCH THE GRADIO APP ---
with gr.Blocks(theme=gr.themes.Soft()) as app:
    gr.Markdown("# 🧑‍🍳 AI Food Critic: From Photo to Recipe (Advanced Model)")
    gr.Markdown("Upload a photo of a meal. Our advanced AI will identify the dish from 30 possibilities and then ask Gemini to generate a custom recipe for you.")
    
    with gr.Row():
        with gr.Column(scale=1):
            image_input = gr.Image(type="numpy", label="Upload a Food Image")
            submit_button = gr.Button(value="1. Classify Image")
        
        with gr.Column(scale=2):
            label_output = gr.Label(num_top_classes=3, label="Top Predictions")
            recipe_button = gr.Button(value="2. Get Recipe", visible=False)
            recipe_output = gr.Markdown(label="Generated Recipe")

    submit_button.click(fn=classify_food_image, inputs=image_input, outputs=[label_output, recipe_button])
    recipe_button.click(fn=generate_recipe_with_gemini, inputs=label_output, outputs=recipe_output)
    image_input.clear(fn=clear_outputs, inputs=[], outputs=[label_output, recipe_output, recipe_button])

# Launch the app from Hugging Face Spaces.
app.launch()
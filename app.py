# ===================================================================
# Updated: app.py
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
device = torch.device("cpu")

# --- 2. DEFINE THE LIST OF 30 CLASS NAMES ---
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
print("Loading trained model architecture...")
model = models.efficientnet_b2(weights=None)
num_ftrs = model.classifier[1].in_features
model.classifier[1] = nn.Linear(num_ftrs, num_classes)

print("Loading trained weights...")
model_path = 'food101_advanced_BEST.pth'
try:
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    print(f"Model '{model_path}' loaded successfully.")
except Exception as e:
    print(f"Error loading model weights: {e}")
    model = None

# --- 4. DEFINE PREPROCESSING AND PIPELINE FUNCTIONS ---
val_test_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

try:
    GOOGLE_API_KEY = os.environ.get('GOOGLE_API_KEY')
    if GOOGLE_API_KEY:
        genai.configure(api_key=GOOGLE_API_KEY)
        print("Google Generative AI configured successfully.")
    else:
        print("WARNING: GOOGLE_API_KEY secret not found.")
except Exception as e:
    print(f"Error configuring Google API: {e}")
    GOOGLE_API_KEY = None

# --- Prediction and Recipe Functions ---
def classify_food_image(input_image):
    confidence_threshold = 0.60
    
    if model is None:
        return {"Error": 1.0}, gr.Button(visible=False), "Model not loaded. Please check logs."
    if input_image is None:
        return None, gr.Button(visible=False), ""

    img = Image.fromarray(input_image.astype('uint8'), 'RGB')
    img_tensor = val_test_transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(img_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)[0]
    confidences = {name: float(prob) for name, prob in zip(chosen_classes_names, probabilities)}
    
    top_confidence = max(confidences.values())
    predicted_class_name = max(confidences, key=confidences.get)
    
    if top_confidence < confidence_threshold:
        # If confidence is too low, hide the button AND show a warning message.
        warning_message = f"Hmm, I'm only {top_confidence:.0%} sure this is {predicted_class_name.replace('_', ' ')}. Try a clearer photo or a different dish!"
        return confidences, gr.Button(visible=False), warning_message
    else:
        # If confidence is high, show the button AND clear any previous warning message.
        button_text = f"2. Generate Recipe for {predicted_class_name.replace('_', ' ').title()}"
        return confidences, gr.Button(value=button_text, visible=True), "" # Return an empty string to clear the message box

def generate_recipe_with_gemini(predictions):
    if not GOOGLE_API_KEY: return "Error: Gemini API Key is not configured."
    if not predictions: return "First, please classify an image."
    
    predicted_class_name = max(predictions, key=predictions.get)
    dish_name = predicted_class_name.replace('_', ' ').title()
    
    try:
        gemini_model = genai.GenerativeModel('ggemini-2.0-flash-001')
        prompt = f"You are a helpful assistant chef. Provide a clear, concise recipe for the dish: {dish_name}. Format your response in Markdown with '### Description', '### Ingredients', and '### Instructions' sections."
        yield "🤖 Generating your recipe with Gemini AI... please wait."
        response = gemini_model.generate_content(prompt)
        yield response.text
    except Exception as e:
        return f"An error occurred while calling the Gemini API: {e}"

def clear_outputs():
    # This now needs to return a value for each of our output components
    return None, None, gr.Button(visible=False), None

# --- 5. DEFINE AND LAUNCH THE GRADIO APP ---
with gr.Blocks(theme=gr.themes.Soft(), css="footer {display: none !important}") as app:
    gr.Markdown("# 🧑‍🍳 AI Food Critic: From Photo to Recipe")
    gr.Markdown("Upload a photo of your meal. Our AI will identify the dish and generate a recipe for you.")
    
    with gr.Row():
        with gr.Column(scale=1):
            image_input = gr.Image(type="numpy", label="Upload a Food Image")
            submit_button = gr.Button(value="1. Identify Dish")
        
        with gr.Column(scale=2):
            label_output = gr.Label(num_top_classes=3, label="Top Predictions")
            
            # This is the new message box for feedback
            status_message_box = gr.Textbox(label="Status", interactive=False)
            
            recipe_button = gr.Button(value="2. Generate Recipe", visible=False)
            recipe_output = gr.Markdown(label="Generated Recipe")

    # The click events now need to handle the new message box
    submit_button.click(
        fn=classify_food_image,
        inputs=image_input,
        outputs=[label_output, recipe_button, status_message_box] # Added the new output
    )

    recipe_button.click(
        fn=generate_recipe_with_gemini,
        inputs=label_output,
        outputs=recipe_output
    )
    
    image_input.clear(
        fn=clear_outputs,
        inputs=[],
        outputs=[label_output, recipe_output, recipe_button, status_message_box] # Added the new output
    )

print("\nLaunching Gradio App...")
app.launch()
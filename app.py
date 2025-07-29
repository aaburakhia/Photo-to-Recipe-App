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
import time

print("--- Initializing The AI Chef's Kitchen ---")
start_time = time.time()

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
model = models.efficientnet_b2(weights=None)
num_ftrs = model.classifier[1].in_features
model.classifier[1] = nn.Linear(num_ftrs, num_classes)
model_path = 'food101_advanced_BEST.pth'
try:
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    print(f"Vision model '{model_path}' loaded in {time.time() - start_time:.2f} seconds.")
except Exception as e:
    print(f"FATAL: Error loading model weights: {e}")
    model = None

# --- 4. DEFINE PREPROCESSING AND PIPELINE FUNCTIONS ---
val_test_transform = transforms.Compose([
    transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

try:
    GOOGLE_API_KEY = os.environ.get('GOOGLE_API_KEY')
    if GOOGLE_API_KEY:
        genai.configure(api_key=GOOGLE_API_KEY)
        print("Generative AI module configured successfully.")
    else:
        print("WARNING: GOOGLE_API_KEY secret not found.")
except Exception as e:
    print(f"Error configuring Google API: {e}")
    GOOGLE_API_KEY = None

# --- Prediction and Recipe Functions ---
def classify_food_image(input_image):
    if model is None: return {"Error": 1.0}, "The vision model failed to load. Please check the logs.", gr.Button(visible=False)
    if input_image is None: return None, "", gr.Button(visible=False)
    
    img = Image.fromarray(input_image.astype('uint8'), 'RGB')
    img_tensor = val_test_transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(img_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)[0]
    confidences = {name: float(prob) for name, prob in zip(chosen_classes_names, probabilities)}
    
    confidence_threshold = 0.60
    top_confidence = max(confidences.values())
    predicted_class_name = max(confidences, key=confidences.get)
    
    if top_confidence < confidence_threshold:
        warning_message = f"🤔 **Hmm, I'm not quite sure...**\nMy top guess is **{predicted_class_name.replace('_', ' ')}** ({top_confidence:.0%} confidence), but that's a bit of a long shot. Try a clearer photo for a recipe!"
        return confidences, warning_message, gr.Button(visible=False)
    else:
        button_text = f"Get Recipe for {predicted_class_name.replace('_', ' ').title()}"
        return confidences, "", gr.Button(value=button_text, visible=True)

def generate_recipe_with_gemini(predictions):
    if not GOOGLE_API_KEY: return "Error: Gemini API Key is not configured. Please set the secret in the Space's settings."
    if not predictions: return "First, please classify an image."
    
    predicted_class_name = max(predictions, key=predictions.get)
    dish_name = predicted_class_name.replace('_', ' ').title()
    
    try:
        # Using the specified model
        gemini_model = genai.GenerativeModel('gemini-2.0-flash-001')
        
        prompt = f"You are a helpful and creative assistant chef. Provide a clear, concise, and easy-to-follow recipe for the dish: {dish_name}. Format the response in Markdown with '### Description', '### Ingredients', and '### Instructions' sections."
        
        # Witty and clever status message
        yield "🧑‍🍳 The AI chef is in the kitchen! The ones and zeros are working hard to craft your recipe..."
        
        response = gemini_model.generate_content(prompt)
        yield response.text
    except Exception as e:
        return f"Sorry, there was an issue generating the recipe. The AI might be busy, or an error occurred: {str(e)}"

def clear_all():
    return None, None, gr.Button(visible=False), None

# --- 5. DEFINE AND LAUNCH THE POLISHED GRADIO APP ---
with gr.Blocks(theme=gr.themes.Monochrome(), css="footer {display: none !important}") as app:
    gr.Markdown("# 🧑‍🍳 AI Food Critic: From Photo to Recipe")
    gr.Markdown("Upload a photo of your meal. Our AI will identify the dish and generate a recipe for you.")
    
    with gr.Row():
        with gr.Column(scale=1):
            image_input = gr.Image(type="numpy", label="Upload Your Food Photo")
            with gr.Accordion("Show Detailed Predictions", open=False):
                label_output = gr.Label(num_top_classes=5, label="Confidence Scores")
            submit_button = gr.Button(value="1. Identify Dish", variant="primary")
            recipe_button = gr.Button(value="2. Generate Recipe", visible=False)

        with gr.Column(scale=2):
            recipe_output = gr.Markdown(label="Status & Recipe")

    # Define the interactive logic
    submit_button.click(
        fn=classify_food_image,
        inputs=image_input,
        outputs=[label_output, recipe_output, recipe_button]
    )
    recipe_button.click(
        fn=generate_recipe_with_gemini,
        inputs=label_output,
        outputs=recipe_output
    )
    image_input.clear(
        fn=clear_all,
        inputs=[],
        outputs=[image_input, label_output, recipe_button, recipe_output]
    )

print("\nLaunching Professional Gradio App...")
app.launch()
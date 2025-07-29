# ===================================================================
# Updated: app.py (Seamless UX and Improved Logic)
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
    print(f"Vision model loaded in {time.time() - start_time:.2f} seconds.")
except Exception as e:
    print(f"FATAL: Error loading model weights: {e}")
    model = None

# --- 4. DEFINE PREPROCESSING AND GEMINI SETUP ---
val_test_transform = transforms.Compose([
    transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

try:
    GOOGLE_API_KEY = os.environ.get('GOOGLE_API_KEY')
    if GOOGLE_API_KEY:
        genai.configure(api_key=GOOGLE_API_KEY)
        gemini_model = genai.GenerativeModel('gemini-2.0-flash-001')
        print("Generative AI module configured successfully.")
    else:
        print("WARNING: GOOGLE_API_KEY secret not found.")
        gemini_model = None
except Exception as e:
    print(f"Error configuring Google API: {e}")
    GOOGLE_API_KEY = None

# --- 5. THE NEW, UNIFIED PIPELINE FUNCTION ---
def identify_and_generate_recipe(input_image):
    if model is None:
        return {"Error": 1.0}, "The vision model failed to load. Please check the logs."
    if input_image is None:
        return None, "Please upload an image first."

    # --- Stage 1: Local Model Classification ---
    yield {"Status": 0.2, "Analyzing...": 0.8}, "🔍 Analyzing image with our food classifier..."
    img_pil = Image.fromarray(input_image.astype('uint8'), 'RGB')
    img_tensor = val_test_transform(img_pil).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(img_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)[0]
    confidences = {name: float(prob) for name, prob in zip(chosen_classes_names, probabilities)}
    
    top_confidence = max(confidences.values())
    predicted_class_name = max(confidences, key=confidences.get)
    final_dish_name = predicted_class_name
    
    # --- Stage 2: Gemini "Second Opinion" if Needed ---
    confidence_threshold = 0.80
    if top_confidence < confidence_threshold:
        yield confidences, "🤔 Hmm, that's a tricky one. Consulting our culinary expert for a second look..."
        if not gemini_model:
            yield confidences, "Error: The AI expert is unavailable (API Key might be missing)."
            return

        try:
            prompt = [
                "Identify the primary food dish in this image. Respond with ONLY the name of the dish from the following list, choosing the best fit: " + ", ".join(chosen_classes_names),
                img_pil
            ]
            response = gemini_model.generate_content(prompt)
            gemini_prediction = response.text.strip().lower().replace(" ", "_")
            if gemini_prediction in chosen_classes_names:
                final_dish_name = gemini_prediction
        except Exception as e:
            # If Gemini fails, we just stick with the original prediction.
            print(f"Gemini vision call failed: {e}")
            pass # Fail gracefully

    # --- Stage 3: Recipe Generation ---
    yield confidences, f"🧑‍🍳 Found it! It looks like **{final_dish_name.replace('_', ' ').title()}**. Crafting your recipe now..."
    if not gemini_model:
        yield confidences, "Error: The AI chef is unavailable (API Key might be missing)."
        return

    try:
        dish_name_for_prompt = final_dish_name.replace('_', ' ').title()
        prompt = f"You are a helpful assistant chef. Provide a clear, concise recipe for the dish: {dish_name_for_prompt}. Format your response in Markdown with '### Description', '### Ingredients', and '### Instructions' sections."
        response = gemini_model.generate_content(prompt)
        yield confidences, response.text
    except Exception as e:
        yield confidences, f"Sorry, there was an issue generating the recipe: {str(e)}"

def clear_all():
    return None, None, None

# --- 6. DEFINE AND LAUNCH THE PROFESSIONAL GRADIO APP ---
with gr.Blocks(theme=gr.themes.Monochrome(), css="footer {display: none !important}") as app:
    gr.Markdown("# 🧑‍🍳 AI Food Critic: From Photo to Recipe")
    gr.Markdown("Upload a photo of your meal. Our AI will identify the dish and generate a recipe for you.")
    
    with gr.Row():
        with gr.Column(scale=1):
            image_input = gr.Image(type="numpy", label="Upload Your Food Photo")
            with gr.Accordion("Show Detailed Predictions", open=False):
                label_output = gr.Label(num_top_classes=5, label="Confidence Scores")
            submit_button = gr.Button(value="Identify & Get Recipe", variant="primary")

        with gr.Column(scale=2):
            recipe_output = gr.Markdown(label="Status & Recipe")

    submit_button.click(
        fn=identify_and_generate_recipe,
        inputs=image_input,
        outputs=[label_output, recipe_output]
    )
    
    image_input.clear(
        fn=clear_all,
        inputs=[],
        outputs=[image_input, label_output, recipe_output]
    )

print("\nLaunching Professional Gradio App...")
app.launch()
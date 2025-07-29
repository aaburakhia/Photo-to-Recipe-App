# ===================================================================
# Updated: app.py (Adding CSS)
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
model = models.efficientnet_b2(weights=None)
num_ftrs = model.classifier[1].in_features
model.classifier[1] = nn.Linear(num_ftrs, num_classes)
model_path = 'food101_advanced_BEST.pth'
try:
    # map_location=torch.device('cpu') is CRUCIAL for loading a GPU-trained model onto a CPU.
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval() # Set the model to evaluation mode permanently for this app.
    print(f"Vision model loaded in {time.time() - start_time:.2f} seconds.")
except Exception as e:
    print(f"FATAL: Error loading model weights: {e}")
    model = None

# --- 4. DEFINE PREPROCESSING AND GEMINI SETUP ---
# The same validation/test transform from training.
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
    GOOGLE_API_KEY = None
    gemini_model = None
    print(f"Error configuring Google API: {e}")

# --- 5. THE UNIFIED PIPELINE FUNCTION ---
def identify_and_generate_recipe(input_image):
    if model is None:
        return None, "The vision model failed to load. Please check the logs."
    if input_image is None:
        return None, "Please upload an image first."

    # Stage 1: Local Model Classification
    yield None, "🔍 **Analyzing Photo...**"
    img_pil = Image.fromarray(input_image.astype('uint8'), 'RGB')
    img_tensor = val_test_transform(img_pil).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(img_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)[0]
    confidences = {name: float(prob) for name, prob in zip(chosen_classes_names, probabilities)}
    
    top_confidence = max(confidences.values())
    local_prediction = max(confidences, key=confidences.get)
    final_dish_name = local_prediction
    
    # Stage 2: Gemini "Second Opinion" if Needed
    confidence_threshold = 0.80
    status_message = f"✅ **High confidence!**\n- **Identified as:** `{local_prediction.replace('_',' ')}` ({top_confidence:.1%})"

    if top_confidence < confidence_threshold:
        status_message = f"🤔 **Local model is uncertain** (Top guess: `{local_prediction.replace('_',' ')}` at {top_confidence:.0%}).\n\n**Consulting our culinary expert...**"
        yield confidences, status_message
        
        if not gemini_model:
            yield confidences, status_message + "\n\n**Error:** The AI expert is unavailable (API Key might be missing)."
            return

        try:
            prompt = ["What is the most specific name of the food dish in this image?", img_pil]
            response = gemini_model.generate_content(prompt)
            gemini_prediction_raw = response.text.strip().lower()
            gemini_prediction_clean = gemini_prediction_raw.split(',')[0].split('(')[0].strip().replace(" ", "_")
            
            # Harmonization Logic
            if gemini_prediction_clean in chosen_classes_names:
                final_dish_name = gemini_prediction_clean
                status_message += f"\n\n✅ **Expert confirmed!** Proceeding with recipe for `{final_dish_name.replace('_', ' ')}`."
            else:
                 status_message += f"\n\n⚠️ **Expert is unsure.** Sticking with our best guess: `{final_dish_name.replace('_', ' ')}`."

        except Exception as e:
            status_message += f"\n\n**Error during expert consultation.** Sticking with best guess. Details: {e}"
            pass

    # Stage 3: Recipe Generation
    yield confidences, status_message + "\n\n" + "🧑‍🍳 **The ones and zeros are firing up the kitchen to craft your recipe...**"

    if not gemini_model:
        yield confidences, status_message + "\n\nError: The AI chef is unavailable (API Key might be missing)."
        return

    try:
        dish_name_for_prompt = final_dish_name.replace('_', ' ').title()
        recipe_prompt = f"You are an expert chef. Provide a clear, high-quality recipe for the dish: {dish_name_for_prompt}. Format the response in Markdown with '### Description', '### Ingredients', and '### Instructions' sections."
        recipe_response = gemini_model.generate_content(recipe_prompt)
        # Combine the final status update with the recipe
        final_output = status_message + "\n\n---\n\n" + recipe_response.text
        yield confidences, final_output
    except Exception as e:
        yield confidences, f"Sorry, there was an issue generating the recipe: {str(e)}"

def clear_all():
    return None, None, None

# --- 6. DEFINE AND LAUNCH THE POLISHED GRADIO APP ---
# Custom CSS for a more professional look
custom_css = """
#recipe-output .prose { font-size: 16px !important; line-height: 1.6 !important; }
#recipe-output .prose h3 { font-size: 24px !important; font-weight: 600 !important; margin-top: 20px !important; margin-bottom: 10px !important; }
#recipe-output .prose ul li { margin-bottom: 8px !important; }
#recipe-output .prose ol li { margin-bottom: 12px !important; line-height: 1.7 !important; }
"""

with gr.Blocks(theme=gr.themes.Monochrome(), css=custom_css) as app:
    gr.Markdown("# 🧑‍🍳 AI Food Critic: From Photo to Recipe")
    gr.Markdown("Upload a photo of your meal. Our AI will analyze the image and generate a recipe for you.")
    
    with gr.Row():
        with gr.Column(scale=1):
            image_input = gr.Image(type="numpy", label="Upload Your Food Photo")
            submit_button = gr.Button(value="Identify & Get Recipe", variant="primary")
            with gr.Accordion("Show Confidence Scores", open=False):
                label_output = gr.Label(num_top_classes=5, label="Model Confidence")

        with gr.Column(scale=2):
            recipe_output = gr.Markdown(label="Analysis Log & Recipe", elem_id="recipe-output")

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
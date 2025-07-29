# ===================================================================
# FINAL HYBRID "AI SECOND OPINION" APP SCRIPT: app.py
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

# --- All setup, model loading, etc., remains the same ---
print("--- Initializing AI Food Recipe App ---")
device = torch.device("cpu")
chosen_classes_names = [
    'beef_carpaccio', 'beef_tartare', 'hamburger', 'pork_chop', 'steak', 'prime_rib',
    'chicken_curry', 'chicken_quesadilla', 'chicken_wings', 'fish_and_chips',
    'crab_cakes', 'clam_chowder', 'lobster_bisque', 'mussels', 'oysters', 'paella',
    'scallops', 'shrimp_and_grits', 'sushi', 'gnocchi', 'lasagna',
    'macaroni_and_cheese', 'pad_thai', 'pho', 'ramen', 'ravioli', 'risotto',
    'spaghetti_bolognese', 'spaghetti_carbonara', 'pizza'
]
model = models.efficientnet_b2(weights=None)
model.classifier[1] = nn.Linear(model.classifier[1].in_features, len(chosen_classes_names))
model_path = 'food101_advanced_BEST.pth'
try:
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    print("Local vision model loaded successfully.")
except Exception as e:
    model = None
    print(f"FATAL: Error loading model weights: {e}")
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
        gemini_model = None
except Exception as e:
    GOOGLE_API_KEY = None
    gemini_model = None
    print(f"Error configuring Google API: {e}")

# --- THE NEW, UNIFIED HYBRID PIPELINE FUNCTION ---
def analyze_image_and_generate_recipe(input_image):
    if model is None:
        return None, "The vision model failed to load. Please check the logs."
    if input_image is None:
        return None, "Please upload an image first."

    # --- Stage 1: Local Model Classification ---
    yield None, "🔍 **Step 1: Analyzing with our specialized food model...**"
    img_pil = Image.fromarray(input_image.astype('uint8'), 'RGB')
    img_tensor = val_test_transform(img_pil).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(img_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)[0]
    confidences = {name: float(prob) for name, prob in zip(chosen_classes_names, probabilities)}
    
    top_confidence = max(confidences.values())
    local_prediction = max(confidences, key=confidences.get)
    final_dish_name = local_prediction

    # --- Stage 2: Gemini "Second Opinion" if local model is not highly confident ---
    confidence_threshold = 0.80
    if top_confidence < confidence_threshold:
        yield confidences, f"🤔 **Local model is uncertain** (Top guess: `{local_prediction.replace('_',' ')}` at {top_confidence:.0%}).\n\n**Consulting our AI expert (Gemini) for a second opinion...**"
        if not gemini_model:
            yield confidences, "Error: The AI expert is unavailable (API Key might be missing)."
            return

        try:
            # Gemini Vision Prompt - an open-ended question
            vision_prompt = ["What is the name of the primary food dish in this image? Be specific.", img_pil]
            gemini_response = gemini_model.generate_content(vision_prompt)
            gemini_text_prediction = gemini_response.text.strip().lower()
            
            # Harmonization Logic
            status_update = f"🧠 **AI Expert Analysis:**\n- **Local Model Guess:** `{local_prediction.replace('_',' ')}` ({top_confidence:.1%})\n- **Gemini's Description:** *'{gemini_text_prediction}'*"
            
            # Check if Gemini's description confirms our local model's guess
            if local_prediction.replace('_', ' ') in gemini_text_prediction:
                status_update += "\n\n✅ **Agreement!** Both models agree. Proceeding with recipe."
                final_dish_name = local_prediction
            else:
                # If they disagree, we can't be sure.
                status_update += "\n\n⚠️ **Disagreement!** Our models have different opinions. I can't confidently generate a recipe for this image. Please try a different photo."
                yield confidences, status_update
                return # Stop the process here

        except Exception as e:
            yield confidences, f"Error during AI expert consultation: {e}"
            return
    else:
        status_update = f"✅ **High confidence!**\n- **Local Model identified:** `{local_prediction.replace('_',' ')}` ({top_confidence:.1%})"
        final_dish_name = local_prediction

    # --- Stage 3: Recipe Generation ---
    yield confidences, status_update + "\n\n" + "🧑‍🍳 **Crafting your recipe...**"

    try:
        dish_name_for_prompt = final_dish_name.replace('_', ' ').title()
        recipe_prompt = f"You are a helpful assistant chef. Provide a clear recipe for the dish: {dish_name_for_prompt}. Format in Markdown with '### Description', '### Ingredients', and '### Instructions' sections."
        recipe_response = gemini_model.generate_content(recipe_prompt)
        yield confidences, status_update + "\n\n" + recipe_response.text
    except Exception as e:
        yield confidences, f"Sorry, there was an issue generating the recipe: {str(e)}"

def clear_all():
    return None, None, None

# --- THE FINAL, SEAMLESS GRADIO APP ---
with gr.Blocks(theme=gr.themes.Monochrome(), css="footer {display: none !important}") as app:
    gr.Markdown("# 🧑‍🍳 AI Food Critic: From Photo to Recipe")
    gr.Markdown("Upload a photo of your meal. Our AI will analyze the image and generate a recipe for you.")
    
    with gr.Row():
        with gr.Column(scale=1):
            image_input = gr.Image(type="numpy", label="Upload Your Food Photo")
            with gr.Accordion("Show Detailed Scores (from local model)", open=False):
                label_output = gr.Label(num_top_classes=5, label="Confidence Scores")
            submit_button = gr.Button(value="Identify & Get Recipe", variant="primary")

        with gr.Column(scale=2):
            # This single component shows the entire process
            status_and_recipe_output = gr.Markdown(label="Analysis Log & Recipe")

    submit_button.click(
        fn=analyze_image_and_generate_recipe,
        inputs=image_input,
        outputs=[label_output, status_and_recipe_output]
    )
    
    image_input.clear(
        fn=clear_all,
        inputs=[],
        outputs=[image_input, label_output, status_and_recipe_output]
    )

print("\nLaunching AI Food Recipe Gradio App...")
app.launch()
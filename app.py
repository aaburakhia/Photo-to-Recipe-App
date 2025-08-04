# ===================================================================
# Fat Julia: An AI Food Classifier and Recipe Generator
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

# --- App Initialization ---
print("--- Initializing Fat Julia's Kitchen ---")
start_time = time.time()
device = torch.device("cpu")

# --- Define the List of 30 Trained Class Names ---
chosen_classes_names = [
    'beef_carpaccio', 'beef_tartare', 'hamburger', 'pork_chop', 'steak', 'prime_rib',
    'chicken_curry', 'chicken_quesadilla', 'chicken_wings', 'fish_and_chips',
    'crab_cakes', 'clam_chowder', 'lobster_bisque', 'mussels', 'oysters', 'paella',
    'scallops', 'shrimp_and_grits', 'sushi', 'gnocchi', 'lasagna',
    'macaroni_and_cheese', 'pad_thai', 'pho', 'ramen', 'ravioli', 'risotto',
    'spaghetti_bolognese', 'spaghetti_carbonara', 'pizza'
]
num_classes = len(chosen_classes_names)

# --- Load the Trained Vision Model ---
model = models.efficientnet_b2(weights=None)
model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
model_path = 'food101_advanced_BEST.pth'
try:
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    print(f"Vision model loaded in {time.time() - start_time:.2f} seconds.")
except Exception as e:
    model = None
    print(f"FATAL: Error loading vision model weights: {e}")

# --- Define Preprocessing and Configure Generative AI ---
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

# --- Core Pipeline Function ---
def identify_and_generate_recipe(input_image):
    if model is None:
        return None, "The vision model failed to load. Please check the logs."
    if input_image is None:
        return None, "Please upload an image first."

    yield None, "🔍 **Analyzing your photo...**"
    img_pil = Image.fromarray(input_image.astype('uint8'), 'RGB')
    img_tensor = val_test_transform(img_pil).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(img_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)[0]
    confidences = {name: float(prob) for name, prob in zip(chosen_classes_names, probabilities)}
    
    top_confidence = max(confidences.values())
    local_prediction = max(confidences, key=confidences.get)
    final_dish_name = local_prediction
    
    if top_confidence < 0.80:
        yield confidences, "🤔 **Hmm, a tricky one. Consulting a culinary expert...**"
        if not gemini_model:
            yield confidences, "Error: The culinary expert is unavailable (API Key might be missing)."
            return

        try:
            prompt = ["What is the most specific name of the food dish in this image?", img_pil]
            response = gemini_model.generate_content(prompt)
            gemini_prediction_raw = response.text.strip().lower()
            gemini_prediction_clean = gemini_prediction_raw.split(',')[0].split('(')[0].strip().replace(" ", "_")
            if gemini_prediction_clean in chosen_classes_names:
                final_dish_name = gemini_prediction_clean
        except Exception as e:
            print(f"Gemini vision call failed: {e}")
            pass

    dish_name_for_prompt = final_dish_name.replace('_', ' ').title()
    yield confidences, f"✅ **Dish Identified:** `{dish_name_for_prompt}`\n\n**🧑‍🍳 The ones and zeros are firing up the kitchen to craft your recipe...**"
    if not gemini_model:
        yield confidences, "Error: The AI chef is unavailable (API Key might be missing)."
        return

    try:
        recipe_prompt = f"""
                        You are an expert chef with a big personality named 'Fat Julia'. You are direct, a little impatient, and funny. Your goal is to provide a clear, high-quality recipe for the dish: {dish_name_for_prompt}.

                        Format the response in Markdown with '### Description', '### Ingredients', and '### Instructions' sections.

                        Maintain a friendly but firm, "tough love" tone. Start with a witty or sassy comment about the dish. Throughout the instructions, throw in one or two short, funny, and direct comments or tips. For example, "Don't even think about using pre-shredded cheese," or "If you burn the garlic, just throw it out and start over. I'm not kidding."
                        """
        recipe_response = gemini_model.generate_content(recipe_prompt)
        yield confidences, recipe_response.text
    except Exception as e:
        yield confidences, f"Sorry, there was an issue generating the recipe: {str(e)}"

def clear_all():
    return None, None, None

# --- Gradio Interface ---
with gr.Blocks(theme=gr.themes.Monochrome(), css="footer {display: none !important}") as app:
    gr.Markdown("# 🧑‍🍳 Fat Julia: From Photo to Recipe")
    gr.Markdown("Upload a photo of your meal. I'll identify the dish and generate a recipe so you can make it yourself!")
    
    with gr.Row():
        with gr.Column(scale=1):
            image_input = gr.Image(type="numpy", label="Upload Your Food Photo")
            submit_button = gr.Button(value="Identify & Get Recipe", variant="primary")
            with gr.Accordion("Show Confidence Scores", open=False):
                label_output = gr.Label(num_top_classes=5, label="Model Confidence")

        with gr.Column(scale=2):
            recipe_output = gr.Markdown(label="Your Recipe")

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

print("\nLaunching Fat Julia...")
app.launch()
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

# --- 1. DEFINE THE LIST OF 30 CLASS NAMES ---
chosen_classes_names = [
    'beef_carpaccio', 'beef_tartare', 'hamburger', 'pork_chop', 'steak', 'prime_rib',
    'chicken_curry', 'chicken_quesadilla', 'chicken_wings', 'fish_and_chips',
    'crab_cakes', 'clam_chowder', 'lobster_bisque', 'mussels', 'oysters', 'paella',
    'scallops', 'shrimp_and_grits', 'sushi', 'gnocchi', 'lasagna',
    'macaroni_and_cheese', 'pad_thai', 'pho', 'ramen', 'ravioli', 'risotto',
    'spaghetti_bolognese', 'spaghetti_carbonara', 'pizza'
]
num_classes = len(chosen_classes_names)

# --- 2. LOAD THE TRAINED VISION MODEL ---
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

# --- 3. DEFINE PREPROCESSING AND CONFIGURE GENERATIVE AI ---
val_test_transform = transforms.Compose([
    transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
try:
    GOOGLE_API_KEY = os.environ.get('GOOGLE_API_KEY')
    if GOOGLE_API_KEY:
        genai.configure(api_key=GOOGLE_API_KEY)
        gemini_model = genai.GenerativeModel('gemini-2.0-flash-001')
        print("Generative AI module configured.")
    else:
        gemini_model = None
except Exception as e:
    GOOGLE_API_KEY = None
    gemini_model = None
    print(f"Error configuring Google API: {e}")

# --- 4. DEFINE PIPELINE FUNCTIONS ---
def classify_food_image(input_image):
    if model is None: return None, "Model failed to load.", gr.Button(visible=False)
    if input_image is None: return None, "", gr.Button(visible=False)
    
    img = Image.fromarray(input_image.astype('uint8'), 'RGB')
    img_tensor = val_test_transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(img_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)[0]
    confidences = {name: float(prob) for name, prob in zip(chosen_classes_names, probabilities)}
    
    top_confidence = max(confidences.values())
    predicted_class_name = max(confidences, key=confidences.get)
    
    confidence_threshold = 0.80
    if top_confidence < confidence_threshold:
        status_message = f"🤔 **Hmm, I'm not quite sure...**\nMy top guess is **{predicted_class_name.replace('_', ' ')}** ({top_confidence:.0%} confidence). I can't generate a recipe unless I'm more confident. Try a clearer photo!"
        return confidences, status_message, gr.Button(visible=False)
    else:
        button_text = f"Get Recipe for {predicted_class_name.replace('_', ' ').title()}"
        return confidences, "", gr.Button(value=button_text, visible=True)

def generate_recipe_with_gemini(predictions):
    if not gemini_model: return "Error: AI Chef is unavailable (API Key might be missing)."
    if not predictions: return "First, please classify an image."
    
    predicted_class_name = max(predictions, key=predictions.get)
    dish_name = predicted_class_name.replace('_', ' ').title()
    
    try:
        prompt = f"""
        You are an expert chef with a big personality named 'Fat Julia'. You are direct, a little impatient, and funny. Your goal is to provide a clear, high-quality recipe for the dish: {dish_name}.
        Format the response in Markdown with '### Description', '### Ingredients', and '### Instructions' sections.
        Maintain a friendly but firm, "tough love" tone. Start with a witty or sassy comment about the dish. Throughout the instructions, throw in one or two short, funny, and direct comments or tips.
        """
        yield "🧑‍🍳 **Alright, I know what this is. Don't mess this up. I'm writing your instructions now...**"
        response = gemini_model.generate_content(prompt)
        yield response.text
    except Exception as e:
        return f"Sorry, there was an issue generating the recipe: {str(e)}"

def clear_all():
    return None, None, gr.Button(visible=False), None

# --- 5. DEFINE THE POLISHED GRADIO APP ---
custom_css = """
#recipe-output .prose { font-size: 16px !important; line-height: 1.6 !important; }
#recipe-output .prose h3 { font-size: 24px !important; font-weight: 600 !important; margin-top: 20px !important; margin-bottom: 10px !important; }
#recipe-output .prose ul li { margin-bottom: 8px !important; }
#recipe-output .prose ol li { margin-bottom: 12px !important; line-height: 1.7 !important; }
"""

with gr.Blocks(theme=gr.themes.Monochrome(), css=custom_css) as app:
    gr.Markdown("# 🧑‍🍳 Fat Julia: From Photo to Recipe")
    gr.Markdown("Upload a photo of your meal. I'll identify the dish and generate a recipe so you can make it yourself. Let's get cooking.")
    
    # Hidden state to reliably pass prediction data
    prediction_state = gr.State()
    
    with gr.Row():
        with gr.Column(scale=1):
            image_input = gr.Image(type="numpy", label="Upload Your Food Photo")
            
            gr.Examples(
                examples=["examples/pizza.webp", "examples/lasagna.webp", "examples/steak.webp"], 
                inputs=image_input,
                label="Click an example to try!"
            )
            
            submit_button = gr.Button(value="1. Identify Dish", variant="primary")
            recipe_button = gr.Button(value="2. Generate Recipe", visible=False)
            
            with gr.Accordion("Show Confidence Scores", open=False):
                label_output = gr.Label(num_top_classes=5, label="Model Confidence")

        with gr.Column(scale=2):
            status_and_recipe_output = gr.Markdown(label="Status & Recipe", elem_id="recipe-output")

    # Define the interactive logic
    submit_button.click(
        fn=classify_food_image,
        inputs=image_input,
        outputs=[label_output, status_and_recipe_output, recipe_button]
    ).then(
        # This .then() part is a trick to pass the label_output to the hidden state
        # after it has been updated by the first function.
        fn=lambda x: x, # A simple function that just passes the value through
        inputs=label_output,
        outputs=prediction_state
    )
    
    recipe_button.click(
        fn=generate_recipe_with_gemini,
        inputs=prediction_state, # Input is now the reliable hidden state
        outputs=status_and_recipe_output
    )
    
    image_input.clear(
        fn=clear_all,
        inputs=[],
        outputs=[image_input, label_output, recipe_button, status_and_recipe_output]
    )

print("\nLaunching Polished Gradio App...")
app.launch()
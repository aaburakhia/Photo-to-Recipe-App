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

# ===================================================================
# --- 4. DEFINE PIPELINE FUNCTIONS (REVISED FOR NEW UI) ---
# ===================================================================
def classify_food_image(input_image):
    # This function now returns FIVE things to update the UI:
    # 1. Confidence scores (for the label component)
    # 2. A status message (for the markdown component)
    # 3. The state of the "Get Recipe" button
    # 4. The mood image to display
    # 5. The state of the "Secret Ingredient" button
    if model is None:
        return None, "The vision model failed to load. Please check the logs.", gr.Button(visible=False), None, gr.Button(visible=False)
    if input_image is None:
        return None, "", gr.Button(visible=False), None, gr.Button(visible=False)
    
    img = Image.fromarray(input_image.astype('uint8'), 'RGB')
    img_tensor = val_test_transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(img_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)[0]
    confidences = {name: float(prob) for name, prob in zip(chosen_classes_names, probabilities)}
    
    top_confidence = max(confidences.values())
    predicted_class_name = max(confidences, key=confidences.get)
    
    # Logic for dynamic moods and sassy responses
    if top_confidence < 0.40: # Low confidence
        status_message = f"**Are you kidding me?** With this lighting, it could be `{predicted_class_name.replace('_', ' ')}` or a leather shoe. I can't work with this. Try a better photo."
        mood_image = "examples/julia_disappointed.png" # Make sure you upload this file
        return confidences, status_message, gr.Button(visible=False), mood_image, gr.Button(visible=False)
    elif top_confidence < 0.80: # Medium confidence
        status_message = f"**Hmm, I'm skeptical.** My gut says this is `{predicted_class_name.replace('_', ' ')}` ({top_confidence:.0%}). I'm watching you, but we can proceed."
        mood_image = "examples/julia_skeptical.png" # Make sure you upload this file
        button_text = f"Fine, Get Recipe for {predicted_class_name.replace('_', ' ').title()}"
        return confidences, status_message, gr.Button(value=button_text, visible=True), mood_image, gr.Button(visible=False)
    else: # High confidence
        status_message = f"✅ **Alright, that's clearly `{predicted_class_name.replace('_', ' ')}`.** Let's not waste time."
        mood_image = "examples/julia_confident.png" # Make sure you upload this file
        button_text = f"Get Recipe for {predicted_class_name.replace('_', ' ').title()}"
        return confidences, status_message, gr.Button(value=button_text, visible=True), mood_image, gr.Button(visible=False)

def generate_recipe_with_gemini(predictions):
    # This function now returns TWO things: the recipe text and the state of the secret button
    if not gemini_model: return "Error: AI Chef is unavailable.", gr.Button(visible=False)
    if not predictions: return "First, classify an image.", gr.Button(visible=False)
    
    predicted_class_name = max(predictions, key=predictions.get)
    dish_name = predicted_class_name.replace('_', ' ').title()
    
    try:
        prompt = f"""
        You are an expert chef with a big personality named 'Fat Julia'. You are direct, a little impatient, and funny. Your goal is to provide a clear, high-quality recipe for the dish: {dish_name}.
        Format the response in Markdown with '### Description', '### Ingredients', and '### Instructions' sections.
        Maintain a friendly but firm, "tough love" tone. Start with a witty or sassy comment about the dish. Throughout the instructions, throw in one or two short, funny, and direct comments or tips.
        """
        yield "🧑‍🍳 **Alright, listen up. I'm writing your instructions now. Don't mess this up...**", gr.Button(visible=False)
        response = gemini_model.generate_content(prompt)
        # Show the "Secret Ingredient" button after the recipe is generated
        yield response.text, gr.Button(visible=True)
    except Exception as e:
        return f"Sorry, there was an issue generating the recipe: {str(e)}", gr.Button(visible=False)

def get_secret_ingredient(predictions):
    if not predictions: return "No secret for you."
    dish = max(predictions, key=predictions.get)
    secrets = {
        'pizza': "The secret is calling for takeout if you burn it. No shame.",
        'steak': "Confidence. Cook it like you know what you're doing, even if you don't.",
        'sushi': "Admitting it's harder than it looks and respecting the professionals.",
        'ramen': "Patience. A good broth doesn't happen in five minutes."
    }
    # Return the secret, or a default one
    return secrets.get(dish, "The secret ingredient is to actually follow the recipe for once.")

def clear_all():
    # This now needs to return a value for each of our output components
    return None, None, "", gr.Button(visible=False), None, gr.Button(visible=False)

# ===================================================================
# --- 5. DEFINE THE POLISHED GRADIO APP (REVISED UI) ---
# ===================================================================
with gr.Blocks(theme=gr.themes.Monochrome(), css="footer {display: none !important}") as app:
    gr.Markdown("# 🧑‍🍳 Fat Julia: From Photo to Recipe")
    gr.Markdown("Upload a photo of your meal. I'll identify the dish and generate a recipe. Let's get cooking.")
    
    # Hidden state to reliably pass prediction data
    prediction_state = gr.State()
    
    with gr.Row():
        with gr.Column(scale=1):
            image_input = gr.Image(type="numpy", label="Upload Your Food Photo")
            
            gr.Examples(
                examples=["examples/pizza.webp", "examples/lasagna.jpg", "examples/steak.webp"], 
                inputs=image_input,
                label="Click an example to try!"
            )
            
            submit_button = gr.Button(value="Well? What is it?", variant="primary")
            recipe_button = gr.Button(value="Generate Recipe", visible=False)
            secret_button = gr.Button(value="What's the Secret Ingredient?", visible=False)
            
            with gr.Accordion("Show Confidence Scores", open=False):
                label_output = gr.Label(num_top_classes=5, label="Model Confidence")

        with gr.Column(scale=2):
            with gr.Row():
                # A component to show the character's mood
                mood_image_output = gr.Image(label="Fat Julia's Mood", interactive=False, show_label=False, height=150, width=150)
                # A component for all status messages
                status_output = gr.Markdown(label="Status")
            
            recipe_output = gr.Markdown(label="Your Recipe")

    # Define the interactive logic
    submit_button.click(
        fn=classify_food_image,
        inputs=image_input,
        outputs=[label_output, status_output, recipe_button, mood_image_output, secret_button]
    ).then(
        fn=lambda x: x,
        inputs=label_output,
        outputs=prediction_state
    )
    
    recipe_button.click(
        fn=generate_recipe_with_gemini,
        inputs=prediction_state,
        outputs=[recipe_output, secret_button]
    )

    secret_button.click(
        fn=get_secret_ingredient,
        inputs=prediction_state,
        outputs=status_output # We'll display the secret in the status box
    )
    
    image_input.clear(
        fn=clear_all,
        inputs=[],
        outputs=[image_input, label_output, status_output, recipe_button, mood_image_output, secret_button]
    )

print("\nLaunching Fat Julia (Full Personality Edition...")
app.launch()
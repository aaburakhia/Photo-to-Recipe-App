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
# This list MUST be in the exact same order as the one you used for training.
chosen_classes_names = [
    'beef_carpaccio', 'beef_tartare', 'hamburger', 'pork_chop', 'steak', 'prime_rib',
    'chicken_curry', 'chicken_quesadilla', 'chicken_wings', 'fish_and_chips',
    'crab_cakes', 'clam_chowder', 'lobster_bisque', 'mussels', 'oysters', 'paella',
    'scallops', 'shrimp_and_grits', 'sushi', 'gnocchi', 'lasagna',
    'macaroni_and_cheese', 'pad_thai', 'pho', 'ramen', 'ravioli', 'risotto',
    'spaghetti_bolognese', 'spaghetti_carbonara', 'pizza'
]

# --- 2. LOAD THE TRAINED VISION MODEL ---
model = models.efficientnet_b2(weights=None)
model.classifier[1] = nn.Linear(model.classifier[1].in_features, len(chosen_classes_names))
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
    if model is None: return None, "Model failed to load.", gr.Button(visible=False), None, gr.Button(visible=False), gr.Row(visible=False)
    if input_image is None: return None, "", gr.Button(visible=False), None, gr.Button(visible=False), gr.Row(visible=False)
    
    img = Image.fromarray(input_image.astype('uint8'), 'RGB')
    img_tensor = val_test_transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(img_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)[0]
    confidences = {name: float(prob) for name, prob in zip(chosen_classes_names, probabilities)}
    
    top_confidence = max(confidences.values())
    predicted_class_name = max(confidences, key=confidences.get)
    
    if top_confidence < 0.40:
        status_message = f"**Are you kidding me?** With this lighting, it could be `{predicted_class_name.replace('_', ' ')}` or a leather shoe. I can't work with this. Try a better photo."
        mood_image = "examples/julia_disappointed.png"
        return confidences, status_message, gr.Button(visible=False), mood_image, gr.Button(visible=False), gr.Row(visible=True)
    elif top_confidence < 0.80:
        status_message = f"**Hmm, I'm skeptical.** My gut says this is `{predicted_class_name.replace('_', ' ')}` ({top_confidence:.0%}). I'm watching you, but we can proceed."
        mood_image = "examples/julia_skeptical.png"
        button_text = f"Fine, Get Recipe for {predicted_class_name.replace('_', ' ').title()}"
        return confidences, status_message, gr.Button(value=button_text, visible=True), mood_image, gr.Button(visible=False), gr.Row(visible=True)
    else:
        status_message = f"✅ **Alright, that's clearly `{predicted_class_name.replace('_', ' ')}`.** Let's not waste time."
        mood_image = "examples/julia_confident.png"
        button_text = f"Get Recipe for {predicted_class_name.replace('_', ' ').title()}"
        return confidences, status_message, gr.Button(value=button_text, visible=True), mood_image, gr.Button(visible=False), gr.Row(visible=True)

def generate_recipe_with_gemini(predictions):
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
        yield response.text, gr.Button(visible=True)
    except Exception as e:
        return f"Sorry, there was an issue generating the recipe: {str(e)}", gr.Button(visible=False)

def get_secret_ingredient(predictions):
    # This function now returns an update object to make the tip box visible
    if not predictions:
        return gr.update(value="No secret for you.", visible=True)
        
    dish = max(predictions, key=predictions.get)
    secrets = {
        'beef_carpaccio': "The secret is slicing the beef paper-thin. If you can read through it, you're doing it right.",
        'beef_tartare': "Using a high-quality, incredibly fresh egg yolk. Don't you dare use the pasteurized stuff in a carton.",
        'hamburger': "Don't overwork the meat! Just form the patty and leave it alone. It's a burger, not a stress ball.",
        'pork_chop': "Brining it beforehand. A little saltwater bath is the difference between a hockey puck and a masterpiece.",
        'steak': "Confidence. Cook it like you know what you're doing, even if you don't. The steak can smell fear.",
        'prime_rib': "A low and slow roast, then a final blast of high heat for the crust. Patience, darling, patience.",
        'chicken_curry': "Blooming your spices. Toast them in the hot oil until they smell amazing. It wakes them up.",
        'chicken_quesadilla': "Grating your own cheese. The pre-shredded stuff has anti-caking agents that prevent a perfect melt. It's a sin.",
        'chicken_wings': "Making sure they are bone-dry before you cook them. Pat them down. Use a fan. I'm not kidding. Dry skin equals crispy skin.",
        'fish_and_chips': "A splash of cold beer in the batter. It makes it light and incredibly crispy. Don't drink it all.",
        'crab_cakes': "More crab, less cake. The secret is using just enough binder to hold it together and not a speck more.",
        'clam_chowder': "A little bit of bacon fat to start the soup. It's not a health food, let's be honest with ourselves.",
        'lobster_bisque': "Using the shells to make the stock. All the flavor is in there. Throwing them out is a crime.",
        'mussels': "A big splash of white wine and a lot of garlic. Don't be shy with either.",
        'oysters': "Serving them ice-cold. Anything less is an insult to the oyster.",
        'paella': "Not stirring the rice! Let it sit and form that beautiful, crispy crust at the bottom called the 'socarrat'.",
        'scallops': "A screaming hot, dry pan. They only need about 90 seconds per side. If you blink, you've overcooked them.",
        'shrimp_and_grits': "Finishing the grits with an obscene amount of butter and cheese. This is not the time for counting calories.",
        'sushi': "The rice. It's always about the perfectly seasoned, perfectly cooked rice. The fish is just the fancy hat.",
        'gnocchi': "A light touch. The more you handle the dough, the tougher it gets. Treat it like a delicate flower.",
        'lasagna': "Letting it rest for at least 20 minutes after it comes out of the oven. If you cut it right away, it'll be a sloppy mess.",
        'macaroni_and_cheese': "Making a proper béchamel sauce. None of that powdered cheese nonsense. We have standards.",
        'pad_thai': "Tamarind paste. It's the source of that authentic sweet and sour tang. There is no substitute.",
        'pho': "Charring your ginger and onions before making the broth. It adds a smoky depth you can't get any other way.",
        'ramen': "A good soft-boiled egg (Ajitama). It's what separates a decent bowl from a great one.",
        'ravioli': "Using wonton wrappers if you're too lazy to make fresh pasta. It's a great cheat. Don't tell any Italian grandmothers I told you that.",
        'risotto': "Stirring constantly and adding the warm broth one ladle at a time. It's needy, but it's worth it.",
        'spaghetti_bolognese': "A splash of milk or cream in the sauce. It makes the meat incredibly tender.",
        'spaghetti_carbonara': "Using the heat of the pasta to cook the eggs, not direct heat. If you make scrambled eggs, you've failed.",
        'pizza': "A screaming hot oven and a pizza stone. You want to shock the dough into a crispy crust."
    }
    secret_text = secrets.get(dish, "The secret ingredient is to actually follow the recipe for once.")
    
    # The key change is returning a Gradio update
    return gr.update(value=f"{secret_text}", visible=True)

def clear_all_ui_update():
    # We add one more output to hide the secret_tip_box
    return None, None, "", gr.Button(visible=False), None, gr.Button(visible=False), gr.Row(visible=False), gr.update(visible=False)

# --- 5. DEFINE THE POLISHED GRADIO APP ---
custom_css = """
.gradio-container {
    overflow-y: auto !important; /* Allow vertical scrolling */
}
/* 1. Apply Background Image to the main app container */
body {
    background-image: url('/background.png') !important;
    background-size: cover !important;
    background-position: center center !important;
    background-repeat: no-repeat !important;
    height: 100vh !important;
}
/* Add a semi-transparent overlay to ensure text is readable */
body::before {
    content: "";
    position: fixed; top: 0; left: 0; width: 100%; height: 100%;
    background-color: rgba(255, 255, 255, 0.85) !important;
    z-index: -1;
}
/* Style the recipe output for readability */
#recipe-output .prose { font-size: 16px !important; line-height: 1.6 !important; }
#recipe-output .prose h3 { font-size: 24px !important; font-weight: 600 !important; margin-top: 20px !important; margin-bottom: 10px !important; }
#recipe-output .prose ul li { margin-bottom: 8px !important; }
#recipe-output .prose ol li { margin-bottom: 12px !important; line-height: 1.7 !important; }
"""

with gr.Blocks(theme=gr.themes.Monochrome(), css=custom_css) as app:
    gr.Markdown("# 🧑‍🍳 Fat Julia: From Photo to Recipe")
    gr.Markdown("Upload a photo of your meal. I'll identify the dish and generate a recipe. Let's get cooking.")
    
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
            
            with gr.Row(visible=False) as controls_row:
                mood_image_output = gr.Image(interactive=False, 
                                             show_share_button=False, 
                                             show_label=False, 
                                             show_download_button=False, 
                                             show_fullscreen_button=False,
                                             width=100, height=100, scale=0)
                with gr.Column():
                    recipe_button = gr.Button(value="Generate Recipe")
                    secret_button = gr.Button(value="What's the Secret Ingredient?")
            
            with gr.Accordion("Show Confidence Scores", open=False):
                label_output = gr.Label(num_top_classes=5, label="Model Confidence")

        with gr.Column(scale=2):
            status_and_recipe_output = gr.Markdown(label="Status & Recipe", elem_id="recipe-output")
            # It's a Textbox, and it's hidden by default.
            secret_tip_box = gr.Textbox(label="🤫 Fat Julia's Secret Tip", visible=False, interactive=False)

    # Define the interactive logic
    submit_button.click(
        fn=classify_food_image,
        inputs=image_input,
        outputs=[label_output, status_and_recipe_output, recipe_button, mood_image_output, secret_button, controls_row]
    ).then(
        fn=lambda x: x,
        inputs=label_output,
        outputs=prediction_state
    )
    
    recipe_button.click(
        fn=generate_recipe_with_gemini,
        inputs=prediction_state,
        outputs=[status_and_recipe_output, secret_button]
    )

    # UPDATE: The secret button now updates the new tip box
    secret_button.click(
        fn=get_secret_ingredient,
        inputs=prediction_state,
        outputs=secret_tip_box # Target the new Textbox
    )
    
    # UPDATE: The clear function now needs to reset the new tip box
    image_input.clear(
        fn=clear_all_ui_update,
        inputs=[],
        outputs=[image_input, label_output, status_and_recipe_output, recipe_button, mood_image_output, secret_button, controls_row, secret_tip_box]
    )

print("\nLaunching Fat Julia ...")
app.launch()
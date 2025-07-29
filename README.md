---
title: AI Food Critic
emoji: 🧑‍🍳
colorFrom: red
colorTo: yellow
sdk: gradio
sdk_version: 4.19.1
app_file: app.py
pinned: true
---

# 🧑‍🍳 AI Food Critic: From Photo to Recipe

Welcome to the AI Food Critic, a deep learning application that identifies a food dish from an uploaded image and generates a custom recipe using generative AI.

This project is a capstone demonstrating a complete, end-to-end MLOps workflow, from data curation and advanced model training to deployment as a robust, interactive web application.



---

## 🚀 Key Features

*   **High-Accuracy Vision Model:** Utilizes a fine-tuned **EfficientNet-B2**, a state-of-the-art convolutional neural network, trained on a curated dataset of 30 food classes to achieve **over 85% accuracy**.
*   **Hybrid AI System:** Employs a sophisticated, two-tier analysis. A fast, specialized model makes an initial guess. If its confidence is low, it escalates to **Google's Gemini 2.0 Flash** for an expert "second opinion," ensuring higher accuracy and robustness.
*   **Dynamic Recipe Generation:** Leverages the Gemini API to generate unique, high-quality recipes in real-time based on the final identified dish.
*   **Interactive UI:** A polished and user-friendly interface built with **Gradio**, deployed permanently on Hugging Face Spaces.

## 🛠️ How It Was Built

### 1. Data Curation & Preprocessing
The model was trained on a challenging, fine-grained dataset of **30,000 images** across 30 food classes. This dataset was curated by combining the official training and testing splits of the **Food-101** dataset to create a balanced set of 1,000 images per class.

Advanced data augmentation techniques, including `TrivialAugmentWide`, were used during training to improve the model's generalization capabilities.

### 2. The Model: Advanced Transfer Learning
The core of the classifier is an **EfficientNet-B2** model pre-trained on ImageNet. A professional two-stage fine-tuning process was implemented:
*   **Stage 1 (Head-Training):** The base of the model was frozen, and a new 30-class classifier head was trained for 5 epochs to learn the basic features of the new dataset without corrupting the pre-trained weights.
*   **Stage 2 (Full Fine-Tuning):** All layers were unfrozen and trained with the **AdamW optimizer** and an advanced **OneCycleLR learning rate scheduler** for 20+ epochs, allowing the entire network to gently adapt to the nuances of food photography.

### 3. The Hybrid AI Pipeline
The final application uses a seamless, two-step AI pipeline for maximum accuracy:
1.  Our fine-tuned **EfficientNet model** performs the initial, high-speed classification.
2.  If the model's confidence is below a certain threshold (e.g., 80%), the image is passed to the **Gemini 2. Flash vision model** for an unconstrained, expert analysis to arrive at a more accurate final decision.

### 4. Deployment
The application is built with **Gradio** and is hosted on this **Hugging Face Space**, providing a stable, permanent, and publicly accessible demo. The backend uses the **Google Generative AI API** for recipe generation, with the API key securely stored using Hugging Face's secrets management.

## 📈 Final Performance
The model achieved a final **test accuracy of 85.6%** across the 30 classes. Its performance was particularly strong on dishes with unique visual signatures (like `Spaghetti Carbonara` and `Pho`) and demonstrated the classic challenges of fine-grained classification on visually similar dishes (like `Pork Chop` vs. `Steak`).

This project showcases a complete cycle of developing and deploying a sophisticated AI application, from data analysis and model training to building a robust, user-friendly interface.
---
title: AI Food Critic
emoji: 🧑‍🍳
colorFrom: red
colorTo: yellow
sdk: gradio
sdk_version: 5.38.2
app_file: app.py
pinned: false
---

# AI Food Critic: From Photo to Recipe

This is a deep learning application that identifies food dishes from an uploaded image and then uses the Gemini API to generate a recipe.

**Features:**
-   Image classification powered by a fine-tuned **EfficientNet-B2** model trained on 30 classes from the Food-101 dataset.
-   Recipe generation using **Google's Gemini AI**.
-   Interactive and user-friendly interface built with **Gradio**.
# 🧑‍🍳 Fat Julia: An AI Food Classifier & Recipe Generator

[![Python Version](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch Version](https://img.shields.io/badge/PyTorch-2.0+-orange.svg)](https://pytorch.org/)
[![Gradio Version](https://img.shields.io/badge/Gradio-4.0+-yellow.svg)](https://gradio.app/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

**A Capstone Project for the INFO-6147 Course at Fanshawe College.**

---

## 🚀 Live Demo on Hugging Face Spaces

Experience the app live! Upload your own food photos and get a recipe in real-time.

[**>> Launch Fat Julia <<**](https://huggingface.co/spaces/aaburakhia/Photo-to-Recipe-App) 



---

## 📖 Table of Contents
1. [Project Vision](#-the-vision-from-photo-to-plate)
2. [Key Features](#-key-features)
3. [Technical Workflow](#-the-technical-recipe)
4. [Performance & Analysis](#-performance--analysis)
5. [How to Run Locally](#-how-to-run-locally)

---

## 🎯 The Vision: From Photo to Plate

Have you ever seen a delicious-looking meal and wished you had the recipe? **Fat Julia** is an AI-powered culinary assistant that does just that. This application uses a state-of-the-art deep learning model to **identify a food dish from any photo** and then instantly **generates a complete, high-quality recipe** for it.

This project is a complete, end-to-end demonstration of a modern hybrid AI system, blending the power of a specialized vision model with the creativity of a large language model.

## ✨ Key Features

*   **Fine-Grained Image Classification:** Powered by a fine-tuned **EfficientNet-B2** model, achieving **86.8% accuracy** on a challenging 30-class food recognition task.
*   **Hybrid Intelligence System:** A smart, two-tiered logic. A fast, specialized model makes the initial guess. If its confidence is low, the image is escalated to **Google's Gemini 1.5 Flash** for an expert "second opinion," ensuring robust and reliable identification.
*   **Dynamic Recipe Generation:** Integrates with the Gemini API to generate unique, high-quality recipes in real-time, complete with a witty "tough love" personality inspired by Julia Child.
*   **Polished User Interface:** A clean, professional, and interactive web app built with Gradio, featuring clickable examples and dynamic, personality-driven feedback.

## 🧠 The Technical "Recipe"

This project was built using a professional machine learning workflow:

1.  **Data Curation:** A challenging dataset of **30,000 images** across 30 distinct main dishes was curated from the Food-101 dataset by combining its official training and test splits.
2.  **Advanced Training:** An `EfficientNet-B2` model was fine-tuned using a **two-stage training strategy** and modern tools like the **AdamW optimizer** and the **OneCycleLR learning rate scheduler** to maximize performance.
3.  **Deployment:** The final application is built with Gradio and permanently hosted on Hugging Face Spaces for public access and demonstration.

## 📊 Performance & Analysis

The final model is a powerful engine, achieving a **weighted-average F1-score of 0.8679** on the held-out test set.

The model excels at identifying dishes with unique visual signatures like **Spaghetti Carbonara (95.4% F1-score)** but finds fine-grained distinctions between visually similar dishes like **Steak vs. Pork Chop** more challenging. This is where the hybrid "second opinion" system provides a crucial safety net.


## 🚀 How to Run Locally

To run this application on your own machine, follow these steps:

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/aaburakhia/Photo-to-Recipe-App.git
    cd Photo-to-Recipe-App
    ```

2.  **Set up a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3.  **Install the required packages:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Set up your API Key:**
    *   You will need a Google Gemini API key.
    *   Set it as an environment variable named `GOOGLE_API_KEY`.

5.  **Run the application:**
    ```bash
    python app.py
    ```
    The application will be available at a local URL (e.g., `http://127.0.0.1:7860`).

---

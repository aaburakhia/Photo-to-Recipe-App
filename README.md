---
title: 'Fat Julia: AI Recipe Generator'
emoji: 🧑‍🍳
colorFrom: red
colorTo: yellow
sdk: gradio
pinned: false
sdk_version: 5.39.0
---

# 🧑‍🍳 Fat Julia: An AI Food Classifier & Recipe Generator

**A Capstone Project for the INFO-6147 Course at Fanshawe College.**

![App Screenshot](URL_TO_YOUR_BEST_APP_SCREENSHOT_HERE) 

---

## 🚀 The Vision: From Photo to Plate

Ever seen a delicious meal online and wished you had the recipe? **Fat Julia** is an AI-powered culinary assistant that brings that wish to life. This application uses a state-of-the-art deep learning model to **identify a food dish from any photo** and then instantly **generates a complete, high-quality recipe** for it.

This project demonstrates a full end-to-end AI pipeline, from advanced model training on a large-scale dataset to a polished, interactive application powered by a hybrid AI system.

## ✨ Key Features

*   **Fine-Grained Image Classification:** Powered by a fine-tuned **EfficientNet-B2** model, achieving **86.8% accuracy** on a challenging 30-class food recognition task.
*   **Hybrid Intelligence System:** A smart, two-tiered logic. A fast, specialized model makes the initial guess. If its confidence is low, the image is escalated to **Google's Gemini 1.5 Flash** for an expert "second opinion," ensuring robust and reliable identification.
*   **Dynamic Recipe Generation:** Integrates with the Gemini API to generate unique, high-quality recipes in real-time based on the final identified dish.
*   **Polished User Interface:** A clean, professional, and interactive web app built with Gradio.

## 🧠 The Technical "Recipe"

This project was built using a professional machine learning workflow:

1.  **Data Curation:** A challenging dataset of **30,000 images** across 30 distinct main dishes was curated from the Food-101 dataset.
2.  **Advanced Training:** An `EfficientNet-B2` model was fine-tuned using a **two-stage training strategy** and modern tools like the **AdamW optimizer** and the **OneCycleLR learning rate scheduler** to maximize performance.
3.  **Deployment:** The final application is built with Gradio and permanently hosted on Hugging Face Spaces for public access and demonstration.

## 📊 Performance & Analysis

The final model is a powerful engine, achieving a **weighted-average F1-score of 0.8679** on the held-out test set. The model excels at identifying dishes with unique visual signatures like **Spaghetti Carbonara (95.4% F1-score)** but finds fine-grained distinctions between visually similar dishes like **Steak vs. Pork Chop** more challenging. This is where the hybrid "second opinion" system provides a crucial safety net.

*For a complete breakdown of the training process and a deep dive into the results, please see the full [**Project Report (PDF)**](LINK_TO_YOUR_PDF_REPORT_HERE).*

## 🚀 Try It Live!

The Gradio application is embedded at the top of this page. Upload an image to get started!

---
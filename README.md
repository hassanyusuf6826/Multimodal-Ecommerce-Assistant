# Multimodal E-commerce Assistant

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)](https://tensorflow.org/)
[![Pinecone](https://img.shields.io/badge/Vector_DB-Pinecone-green.svg)](https://www.pinecone.io/)
[![Flask](https://img.shields.io/badge/Backend-Flask-lightgrey.svg)](https://flask.palletsprojects.com/)

## 📋 Executive Summary
This project implements a **Multimodal Information Retrieval System** designed for e-commerce applications. It bridges the gap between unstructured user inputs (handwritten notes, raw images, natural language) and structured product databases.

By leveraging **Computer Vision (CNNs)**, **Optical Character Recognition (OCR)**, and **Semantic Vector Search**, the application provides a unified interface for product discovery, simulating a real-world recommendation engine.

## 🏗 System Architecture

The solution operates on a tri-modal pipeline:

1.  **Semantic Text Search:**
    * **Input:** Natural language queries.
    * **Engine:** Sentence Transformers (`all-MiniLM-L6-v2`) generate 384-dimensional embeddings.
    * **Retrieval:** Pinecone Vector Database performs a k-Nearest Neighbor (k-NN) search using Cosine Similarity.
2.  **OCR-Based Retrieval:**
    * **Input:** Images of handwritten shopping lists or notes.
    * **Engine:** Tesseract OCR extracts text, which is then fed into the semantic embedding pipeline.
3.  **Visual Product Classification:**
    * **Input:** Raw product images.
    * **Engine:** A custom Convolutional Neural Network (CNN) trained on scraped product data classifies the image into product categories.
    * **Retrieval:** The predicted category triggers a vector search to recommend specific stock items.

## 🛠 Tech Stack

* **Deep Learning:** TensorFlow/Keras (CNN), Sentence-Transformers (Embeddings)
* **Computer Vision:** OpenCV, Pytesseract (OCR wrapper)
* **Vector Database:** Pinecone (Serverless)
* **Data Engineering:** Pandas, DuckDuckGo Search (Data gathering)
* **Web Framework:** Flask, PyNgrok (Tunneling)
* **Visualization:** Matplotlib, HTML/CSS (Frontend)

## 📂 Dataset & Preprocessing

The model utilizes a cleaned version of the **Online Retail Dataset**.

* **Data Cleaning:** Regex-based removal of special characters from `StockCode`, handling of null descriptions, and deduplication.
* **Image Acquisition:** Automated data gathering pipeline using `ddgs` (DuckDuckGo Search) to map textual product descriptions to training images.
* **Augmentation:** `ImageDataGenerator` applied rotation, zoom, and horizontal flips to prevent overfitting in the CNN.

## 🧠 Model Details

### 1. The Vectorizer
* **Model:** `all-MiniLM-L6-v2`
* **Performance:** Optimized for speed and low latency. Maps semantic meaning (e.g., "device for light") to product descriptions (e.g., "White Metal Lantern").

### 2. The CNN Classifier
A custom sequential architecture designed for 128x128 input images:
* **Layers:** 3x Conv2D (32/64/128 filters) + MaxPooling.
* **Regularization:** Dropout (0.5) to mitigate overfitting.
* **Output:** Softmax activation over unique product classes.

## 🚀 Installation & Setup

### Prerequisites
* Python 3.8+
* Tesseract OCR installed at the system level:
    * *Linux/Colab:* `sudo apt install tesseract-ocr`
    * *Windows:* [Download Installer](https://github.com/UB-Mannheim/tesseract/wiki)

### Step 1: Clone the Repository
```bash
git clone [https://github.com/hassanyusuf6826/Multimodal-Ecommerce-Assistant.git](https://github.com/hassanyusuf6826/Multimodal-Ecommerce-Assistant.git)
cd Multimodal-Ecommerce-Assistant

### Step 2: Install Dependencies
```bash
pip install -r requirements.txt

### Step 3: Configure API Keys
Open app.py and configure the environment variables or placeholders:
PINECONE_API_KEY = "your-pinecone-key"
NGROK_AUTH_TOKEN = "your-ngrok-token"

### Step 4: Run the Application
python app1.py



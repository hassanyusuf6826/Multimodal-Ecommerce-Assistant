import os
import pickle
import numpy as np
import pandas as pd
import tensorflow as tf
from flask import Flask, request, render_template_string
from pyngrok import ngrok
from tensorflow.keras.preprocessing import image as keras_image
from tensorflow.keras.models import load_model

# --- CONNECTING TO YOUR PROJECT FILES ---
import config
from search_engine import VectorDBManager
from ocr_service import OCRProcessor
from vision_model import ProductClassifier

# --- GLOBAL INITIALIZATION (Run once on startup) ---
print("Initializing System Components...")

# 1. Load the Cleaned Dataset
try:
    df_clean = pd.read_csv(config.CLEAN_DATA_PATH)
    # Ensure Description column is string type for matching
    df_clean['Description'] = df_clean['Description'].astype(str)
    print(f"Dataset loaded: {len(df_clean)} products.")
except Exception as e:
    print(f"Error loading {config.CLEAN_DATA_PATH}: {e}")
    df_clean = pd.DataFrame()

# 2. Initialize Vector Database
try:
    vdb = VectorDBManager(config.PINECONE_API_KEY, config.INDEX_NAME)
    vdb.setup_index()
    print("Vector DB connected.")
except Exception as e:
    print(f"Error connecting to Vector DB: {e}")

# 3. Initialize OCR
ocr = OCRProcessor()

# 4. Initialize Vision Model
trainer = ProductClassifier()
if os.path.exists(config.MODEL_PATH):
    print("Loading Vision Model...")
    trainer.model = load_model(config.MODEL_PATH)
    if os.path.exists(config.CLASS_NAMES_FILE):
        with open(config.CLASS_NAMES_FILE, 'rb') as f:
            trainer.class_names = pickle.load(f)
else:
    print("Warning: Vision model not found.")

# --- HELPER FUNCTIONS ---

def recommend_products(query_text):
    """Searches Pinecone and returns structured data."""
    if not query_text:
        return "No query provided.", []

    try:
        results = vdb.query_product(query_text, top_k=5)
    except Exception as e:
        return f"Error querying database: {str(e)}", []

    recommendations = []

    # Handle Pinecone response format
    matches = results.get('matches', []) if isinstance(results, dict) else getattr(results, 'matches', [])

    for match in matches:
        # Extract metadata
        metadata = match.get('metadata', {}) if isinstance(match, dict) else getattr(match, 'metadata', {})
        score = match.get('score', 0) if isinstance(match, dict) else getattr(match, 'score', 0)

        # --- THE FIX IS HERE ---
        # We try 'Description' (Capital D) first, which matches your database.
        # We also try 'description' (lowercase) just in case.
        description = metadata.get('Description', metadata.get('description', 'Unknown Product'))

        # Look up Price/StockCode in our local CSV
        price = "N/A"
        stock_code = "N/A"

        # Find matching row in dataframe
        if description != "Unknown Product":
            product_row = df_clean[df_clean['Description'] == description]
            if not product_row.empty:
                price = product_row.iloc[0]['UnitPrice']
                stock_code = product_row.iloc[0]['StockCode']

        recommendations.append({
            'description': description,
            'price': price,
            'stock_code': stock_code,
            'score': round(score, 2)
        })

    response_text = f"Found {len(recommendations)} matches for '{query_text}'."
    return response_text, recommendations

def process_ocr_query(image_path):
    try:
        extracted_text = ocr.extract_text(image_path)
    except Exception as e:
        return f"OCR Error: {e}", [], ""

    if not extracted_text:
        return "No text detected.", [], ""

    response_text, recommendations = recommend_products(extracted_text)
    return response_text, recommendations, extracted_text

def predict_product_from_image(image_path):
    try:
        if trainer.model is None:
            return "Error: Vision model not loaded.", "System Error", []

        img = keras_image.load_img(image_path, target_size=config.IMG_SIZE)
        img_array = keras_image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array /= 255.0

        predictions = trainer.model.predict(img_array)
        class_idx = np.argmax(predictions[0])

        predicted_class = trainer.class_names[class_idx] if trainer.class_names else "Unknown"
        response_text, recommendations = recommend_products(predicted_class)

        return predicted_class, response_text, recommendations
    except Exception as e:
        return f"Error: {str(e)}", "Could not process image", []

# --- FLASK APP SETUP ---
app = Flask(__name__)
ngrok.set_auth_token(config.NGROK_AUTH_TOKEN)
ngrok.kill()
public_url = ngrok.connect(5000).public_url
print(f" * Public URL: {public_url}")

# --- HTML TEMPLATE (YOUR EXACT DESIGN) ---
html_template = """
<!DOCTYPE html>
<html>
<head>
    <title>AI E-commerce Assistant</title>
    <style>
        body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; padding: 20px; background-color: #f4f4f9; }
        .container { max-width: 900px; margin: auto; background: white; padding: 30px; border-radius: 10px; box-shadow: 0 4px 8px rgba(0,0,0,0.1); }
        h1 { color: #333; text-align: center; }
        nav { text-align: center; margin-bottom: 20px; }
        nav a { margin: 0 15px; text-decoration: none; color: #007bff; font-weight: bold; font-size: 1.1em; }
        nav a:hover { text-decoration: underline; }
        .result { background: #e9ecef; padding: 20px; margin-top: 20px; border-radius: 8px; }

        /* Table Styles */
        table { width: 100%; border-collapse: collapse; margin-top: 15px; background: white; }
        th, td { padding: 12px; text-align: left; border-bottom: 1px solid #ddd; }
        th { background-color: #007bff; color: white; }
        tr:hover { background-color: #f1f1f1; }

        form { margin-top: 20px; text-align: center; }
        input[type="text"] { padding: 10px; width: 60%; border: 1px solid #ddd; border-radius: 4px; }
        button { padding: 10px 20px; background-color: #28a745; color: white; border: none; border-radius: 4px; cursor: pointer; }
        button:hover { background-color: #218838; }
    </style>
</head>
<body>
<div class="container">
    <h1>AI Product Assistant</h1>
    <nav>
        <a href="/">Text Search</a> |
        <a href="/ocr">Handwritten Search</a> |
        <a href="/vision">Image Detection</a>
    </nav>
    <hr>

    {% if page == 'text' %}
    <h2 style="text-align:center;">Find Products by Description</h2>
    <form method="post">
        <input type="text" name="query" placeholder="E.g., 'White metal lantern'..." required>
        <button type="submit">Search</button>
    </form>
    {% elif page == 'ocr' %}
    <h2 style="text-align:center;">Upload Handwritten Note</h2>
    <form method="post" enctype="multipart/form-data">
        <input type="file" name="file" required>
        <button type="submit">Upload & Read</button>
    </form>
    {% elif page == 'vision' %}
    <h2 style="text-align:center;">Upload Product Image</h2>
    <form method="post" enctype="multipart/form-data">
        <input type="file" name="file" required>
        <button type="submit">Identify Product</button>
    </form>
    {% endif %}

    {% if result %}
    <div class="result">
        <h3>Analysis Results:</h3>
        <p><strong>System Response:</strong> {{ response_text }}</p>
        {% if extracted %} <p><strong>Extracted Text:</strong> <em>"{{ extracted }}"</em></p> {% endif %}
        {% if predicted_class %} <p><strong>Detected Category:</strong> <span style="color:green; font-weight:bold;">{{ predicted_class }}</span></p> {% endif %}

        <h4>Recommended Products:</h4>
        {% if recommendations %}
        <table>
            <thead>
                <tr>
                    <th style="width: 50%;">Product Name</th>
                    <th>Stock Code</th>
                    <th>Price ($)</th>
                    <th>Relevance Score</th>
                </tr>
            </thead>
            <tbody>
                {% for item in recommendations %}
                <tr>
                    <td>{{ item.description }}</td>
                    <td>{{ item.stock_code }}</td>
                    <td>{{ item.price }}</td>
                    <td>{{ item.score }}</td>
                </tr>
                {% endfor %}
            </tbody>
        </table>
        {% else %}
        <p>No products found matching your query.</p>
        {% endif %}
        </div>
    {% endif %}
</div>
</body>
</html>
"""

# --- ROUTES ---
@app.route('/', methods=['GET', 'POST'])
def text_query():
    result = None
    response_text = ""
    recommendations = []

    if request.method == 'POST':
        query = request.form.get('query', '')
        response_text, recommendations = recommend_products(query)
        result = True

    return render_template_string(html_template, page='text', result=result, response_text=response_text, recommendations=recommendations)

@app.route('/ocr', methods=['GET', 'POST'])
def ocr_query():
    result = None
    response_text = ""
    recommendations = []
    extracted = ""

    if request.method == 'POST':
        if 'file' not in request.files:
            return "No file uploaded"
        file = request.files['file']
        if file.filename != '':
            filepath = os.path.join('static', file.filename)
            os.makedirs('static', exist_ok=True)
            file.save(filepath)

            response_text, recommendations, extracted = process_ocr_query(filepath)
            result = True

    return render_template_string(html_template, page='ocr', result=result, response_text=response_text, recommendations=recommendations, extracted=extracted)

@app.route('/vision', methods=['GET', 'POST'])
def vision_query():
    result = None
    response_text = ""
    recommendations = []
    predicted_class = ""

    if request.method == 'POST':
        if 'file' not in request.files:
            return "No file uploaded"
        file = request.files['file']
        if file.filename != '':
            filepath = os.path.join('static', file.filename)
            os.makedirs('static', exist_ok=True)
            file.save(filepath)

            predicted_class, response_text, recommendations = predict_product_from_image(filepath)
            result = True

    return render_template_string(html_template, page='vision', result=result, response_text=response_text, recommendations=recommendations, predicted_class=predicted_class)

if __name__ == '__main__':
    app.run(port=5000)

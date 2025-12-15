import os

# --- API KEYS ---
# Replace these with your actual keys
PINECONE_API_KEY = "INPUT YOUR PINECONE_API_KEY"
NGROK_AUTH_TOKEN = "INPUT YOUR NGROK_AUTH_TOKEN"

# --- SETTINGS ---
INDEX_NAME = "ecommerce-product-index"
IMG_SIZE = (128, 128)
BATCH_SIZE = 32
DATASET_PATH = "dataset.csv"
CLEAN_DATA_PATH = "dataset_cleaned.csv"
IMAGE_DIR = "dataset_images"
MODEL_PATH = "product_model.h5"
CLASS_NAMES_FILE = "class_names.pkl"
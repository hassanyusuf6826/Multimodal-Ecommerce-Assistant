import cv2
import pytesseract
import os
import sys

class OCRProcessor:
    def __init__(self):
        # Tesseract is usually found automatically on Linux/Colab.
        # If on Windows, you might need to uncomment and set the path:
        # pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
        pass

    def extract_text(self, image_path):
        """Loads an image, applies preprocessing, and extracts text."""

        # 1. Validation
        if not os.path.exists(image_path):
            return f"Error: Image not found at {image_path}"

        # 2. Load Image
        image = cv2.imread(image_path)
        if image is None:
            return "Error: Could not read image format."

        # 3. Preprocessing (Critical for Accuracy)
        # Convert to Grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Apply Thresholding (Binarization)
        # This converts the image to strict Black & White, removing shadows/noise
        # Otsu's method automatically finds the best separation point.
        processed_img = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

        # 4. Extract Text
        try:
            text = pytesseract.image_to_string(processed_img)
            return text.strip()
        except Exception as e:
            return f"OCR Error: {e}"

# --- TEST EXECUTION ---
if __name__ == "__main__":
    ocr = OCRProcessor()

    # Replace this with an actual image path you have uploaded to your environment
    test_path = '/content/WhatsApp Image 2025-02-20 at 01.47.25.jpeg.jpg'

    if os.path.exists(test_path):
        print(f"Processing {test_path}...")
        result = ocr.extract_text(test_path)
        print("\n--- Extracted Text ---")
        print(result)
    else:
        print(f"Test image not found at: {test_path}")
        print("Please upload an image and update the path to test.")

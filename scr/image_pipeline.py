import os
import time
import requests
import pandas as pd
import re
# from duckduckgo_search import DDGS
from ddgs import DDGS  # Changed from duckduckgo_search
from PIL import Image
import config

class ImageScraper:
    def __init__(self, save_dir=config.IMAGE_DIR):
        self.save_dir = save_dir
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

    def download_images(self, query, num_images=50):
        """Download images with enhanced error handling and debugging."""
        safe_query = re.sub(r'[^\w\s-]', '', query).strip()
        folder_path = os.path.join(self.save_dir, safe_query)

        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        print(f"\n{'='*60}")
        print(f"Searching for: '{query}'")
        print(f"Folder: {folder_path}")

        # Check if folder already has enough images
        existing_images = len([f for f in os.listdir(folder_path) if f.endswith(('.jpg', '.png', '.jpeg'))])
        if existing_images >= num_images:
            print(f"✓ Already have {existing_images} images, skipping...")
            return existing_images

        results = []
        try:
            # Try with context manager
            with DDGS() as ddgs:
                results = list(ddgs.images(query, max_results=num_images))
                print(f"Found {len(results)} image URLs")
        except Exception as e:
            print(f"❌ Error with DDGS context manager: {e}")
            # Try without context manager as fallback
            try:
                print("Trying alternative method...")
                ddgs = DDGS()
                results = list(ddgs.images(query, max_results=num_images))
                print(f"Found {len(results)} image URLs")
            except Exception as e2:
                print(f"❌ Alternative method also failed: {e2}")
                return 0

        if not results:
            print(f"⚠️ No results returned for '{query}'")
            return 0

        count = existing_images  # Continue numbering from existing images
        downloaded_this_run = 0

        for idx, res in enumerate(results):
            try:
                # Add user agent to avoid blocks
                headers = {
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
                }

                img_url = res.get('image')
                if not img_url:
                    print(f"  ⚠️ Result {idx} has no 'image' key")
                    continue

                print(f"  Downloading {idx+1}/{len(results)}...", end='', flush=True)

                img_data = requests.get(img_url, timeout=10, headers=headers).content

                # Verify it's actually an image before saving
                try:
                    img = Image.open(requests.get(img_url, timeout=10, headers=headers, stream=True).raw)

                    # Convert RGBA to RGB if needed
                    if img.mode == 'RGBA':
                        img = img.convert('RGB')

                    file_path = os.path.join(folder_path, f"{count}.jpg")
                    img.save(file_path, 'JPEG', quality=95)

                    count += 1
                    downloaded_this_run += 1
                    print(f" ✓")

                    if downloaded_this_run >= num_images - existing_images:
                        break

                except Exception as img_err:
                    print(f" ✗ (invalid image)")
                    continue

                time.sleep(1)  # Increased delay to avoid rate limiting

            except requests.exceptions.Timeout:
                print(f" ✗ (timeout)")
            except requests.exceptions.RequestException as req_err:
                print(f" ✗ (request failed: {req_err})")
            except Exception as e:
                print(f" ✗ (error: {str(e)[:50]})")

        print(f"{'='*60}")
        print(f"✓ Downloaded {downloaded_this_run} new images for '{query}'")
        print(f"✓ Total images in folder: {count}")

        return count

def cleanup_corrupt_images(directory):
    """Clean up corrupt or invalid images."""
    print(f"\n{'='*60}")
    print(f"Scanning '{directory}' for corrupt images...")
    deleted_count = 0
    total_checked = 0

    for root, dirs, files in os.walk(directory):
        for file in files:
            if not file.lower().endswith(('.jpg', '.jpeg', '.png', '.gif', '.bmp')):
                continue

            file_path = os.path.join(root, file)
            total_checked += 1

            try:
                with Image.open(file_path) as img:
                    img.verify()

                # Re-open and try to load (verify doesn't catch all issues)
                with Image.open(file_path) as img:
                    img.load()

            except Exception as e:
                print(f"  ✗ Removing corrupt: {file_path} ({e})")
                os.remove(file_path)
                deleted_count += 1

    print(f"{'='*60}")
    print(f"✓ Checked {total_checked} files")
    print(f"✓ Removed {deleted_count} corrupt files")

if __name__ == "__main__":
    print("\n" + "="*60)
    print("IMAGE SCRAPING PIPELINE")
    print("="*60)

    # 1. Initialize Scraper
    scraper = ImageScraper(save_dir=config.IMAGE_DIR)

    # 2. Load Cleaned Data
    if os.path.exists(config.CLEAN_DATA_PATH):
        df_clean = pd.read_csv(config.CLEAN_DATA_PATH)
        print(f"✓ Loaded {len(df_clean)} rows from {config.CLEAN_DATA_PATH}")
    else:
        print(f"❌ Error: {config.CLEAN_DATA_PATH} not found. Run data_pipeline.py first.")
        exit()

    # 3. Load Target List
    target_csv = 'CNN_Model_Train_Data.csv'

    if os.path.exists(target_csv):
        print(f"✓ Loading target list from {target_csv}...")
        cnn_data = pd.read_csv(target_csv)
    else:
        print(f"⚠️ '{target_csv}' not found. Using all unique StockCodes.")
        cnn_data = pd.DataFrame(df_clean['StockCode'].unique(), columns=['StockCode'])

    # Clean StockCode
    cnn_data['StockCode'] = cnn_data['StockCode'].astype(str).apply(lambda x: re.sub(r'[^A-Za-z0-9]', '', x))

    # 4. Map StockCodes to Descriptions
    code_to_desc = df_clean.set_index('StockCode')['Description'].to_dict()

    # 5. Get Unique Codes
    unique_codes = cnn_data['StockCode'].unique()

    # TEST MODE: Uncomment to test with just 3 items
    # unique_codes = unique_codes[:3]

    print(f"\n✓ Will download images for {len(unique_codes)} unique items")
    print(f"✓ Target: 40 images per item")
    print(f"✓ Total target images: {len(unique_codes) * 40}")

    # Ask for confirmation
    response = input("\nContinue? (y/n): ").lower()
    if response != 'y':
        print("Aborted.")
        exit()

    # 6. Run Scraping Loop
    successful = 0
    failed = 0

    for idx, code in enumerate(unique_codes):
        print(f"\n[{idx+1}/{len(unique_codes)}] Processing StockCode: {code}")

        description = code_to_desc.get(code)

        if description:
            search_query = f"{description} product"
            count = scraper.download_images(search_query, num_images=40)

            if count > 0:
                successful += 1
            else:
                failed += 1
                print(f"⚠️ Failed to download any images for {code}")
        else:
            print(f"❌ No description found for StockCode: {code}")
            failed += 1

        # Add delay between different products
        if idx < len(unique_codes) - 1:
            time.sleep(2)

    # 7. Summary
    print(f"\n{'='*60}")
    print(f"SCRAPING COMPLETE")
    print(f"{'='*60}")
    print(f"✓ Successful: {successful}/{len(unique_codes)}")
    print(f"✗ Failed: {failed}/{len(unique_codes)}")

    # 8. Run Cleanup
    cleanup_corrupt_images(config.IMAGE_DIR)

    # 9. Final Stats
    print(f"\n{'='*60}")
    print(f"FINAL STATISTICS")
    print(f"{'='*60}")
    for root, dirs, files in os.walk(config.IMAGE_DIR):
        if dirs:
            for d in dirs:
                folder_path = os.path.join(root, d)
                img_count = len([f for f in os.listdir(folder_path) if f.endswith(('.jpg', '.png', '.jpeg'))])
                print(f"  {d}: {img_count} images")
            break

import pandas as pd
import time
from pinecone import Pinecone, ServerlessSpec
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import config

class VectorDBManager:
    def __init__(self, api_key, index_name):
        self.pc = Pinecone(api_key=api_key)
        self.index_name = index_name
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.index = None

    def setup_index(self):
        """Creates the Pinecone index if it doesn't exist."""
        existing_indexes = [i.name for i in self.pc.list_indexes()]

        if self.index_name not in existing_indexes:
            print(f"Creating index: {self.index_name}...")
            self.pc.create_index(
                name=self.index_name,
                dimension=384,
                metric='cosine',
                spec=ServerlessSpec(cloud='aws', region='us-east-1')
            )
            # Wait for index to be ready
            time.sleep(10)

        self.index = self.pc.Index(self.index_name)
        return self.index

    def vectorize_and_upsert(self, df):
        """Batches data, converts to vectors, and uploads to Pinecone."""
        if self.index is None:
            self.setup_index()

        print(f"Starting ingestion of raw data: {len(df)} rows...")

        # 1. Deduplicate by Description (we only need one vector per product type)
        # Ensure we don't have empty descriptions
        df = df.dropna(subset=['Description'])
        unique_products = df.drop_duplicates(subset=['Description']).copy()

        print(f"Unique products to vectorize: {len(unique_products)}")

        batch_size = 100
        total_rows = len(unique_products)

        # 2. Loop through data in batches
        for i in tqdm(range(0, total_rows, batch_size), desc="Upserting"):
            batch = unique_products.iloc[i : i + batch_size]

            # A. Convert Text to Vectors
            descriptions = batch['Description'].astype(str).tolist()
            embeddings = self.model.encode(descriptions).tolist()

            # B. Prepare Metadata
            # Ensure StockCode is a string for the ID
            ids = batch['StockCode'].astype(str).tolist()
            
            # Create metadata dicts
            metadatas = batch[['Description', 'UnitPrice', 'Country']].to_dict('records')

            # C. Zip it all together: (ID, Vector, Metadata)
            vectors_to_upsert = list(zip(ids, embeddings, metadatas))

            # D. Upload to Pinecone
            self.index.upsert(vectors=vectors_to_upsert)

        print("Ingestion complete!")

    def query_product(self, query_text, top_k=5):
        """Search the database."""
        if self.index is None:
            self.setup_index()

        query_vector = self.model.encode(query_text).tolist()
        result = self.index.query(vector=query_vector, top_k=top_k, include_metadata=True)
        return result

# --- MAIN EXECUTION ---
if __name__ == "__main__":
    # 1. Initialize
    vdb = VectorDBManager(config.PINECONE_API_KEY, config.INDEX_NAME)
    vdb.setup_index()

    # 2. Load Data
    try:
        print(f"Loading data from {config.CLEAN_DATA_PATH}...")
        df_clean = pd.read_csv(config.CLEAN_DATA_PATH)
        df_clean['Description'] = df_clean['Description'].astype(str)
        df_clean['StockCode'] = df_clean['StockCode'].astype(str)

        # 3. Ask User to Upload
        choice = input("Do you want to upload vectors to Pinecone? (yes/no): ")
        if choice.lower() in ['yes', 'y']:
            vdb.vectorize_and_upsert(df_clean)

        # 4. Test Search
        print("\n--- Testing Search Engine ---")
        test_query = "white hanging heart"
        results = vdb.query_product(test_query)

        # FIX: Handling Pinecone v3+ Response Objects
        # We check if 'matches' exists as an attribute (dot notation) or key
        matches = getattr(results, 'matches', []) or results.get('matches', [])

        if matches:
            for match in matches:
                # FIX: Access properties with Dot Notation for v3+ Client
                # (We use getattr to be safe if it falls back to a dict)
                score = getattr(match, 'score', None) or match.get('score')
                metadata = getattr(match, 'metadata', {}) or match.get('metadata', {})
                
                # Handle cases where metadata is None
                if metadata:
                    description = metadata.get('Description', 'Unknown Description')
                else:
                    description = "No Metadata Found"

                print(f"Score: {score:.4f} | {description}")
        else:
            print("No matches found.")

    except FileNotFoundError:
        print(f"Error: Could not find {config.CLEAN_DATA_PATH}. Run data_pipeline.py first.")
    # Commented out the generic catch so you can see the REAL error line if it fails again
    # except Exception as e:
    #     print(f"An unexpected error occurred: {e}")

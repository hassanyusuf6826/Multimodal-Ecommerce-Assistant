import pandas as pd
import re
import os
import config # Ensure this is imported as lowercase 'config'

class DataManager:
    def __init__(self):
        self.df = None

    def load_and_clean_data(self):
        """Loads CSV and cleans special characters."""
        print("Loading and cleaning dataset...")
        try:
            self.df = pd.read_csv(config.DATASET_PATH, encoding='ISO-8859-1')
        except UnicodeDecodeError:
            self.df = pd.read_csv(config.DATASET_PATH, encoding='utf-8')

        # 1. Clean StockCode: Remove special chars, BUT KEEP LETTERS (e.g., '85123A')
        self.df['StockCode'] = self.df['StockCode'].astype(str).apply(lambda x: re.sub(r'[^A-Za-z0-9]', '', x))
        # REMOVED the line that stripped letters, assuming you need alphanumeric codes.

        # 2. Clean Description
        self.df = self.df.dropna(subset=['Description'])
        self.df['Description'] = self.df['Description'].astype(str).apply(lambda x: x.replace('$', '').strip())

        # 3. Clean Quantity
        self.df['Quantity'] = self.df['Quantity'].astype(str).apply(lambda x: re.sub(r'[^\d-]', '', x))
        self.df['Quantity'] = pd.to_numeric(self.df['Quantity'], errors='coerce').fillna(0).astype(int)

        # 4. Clean UnitPrice
        self.df['UnitPrice'] = self.df['UnitPrice'].astype(str).apply(lambda x: re.sub(r'[^\d\.]', '', x))
        self.df['UnitPrice'] = pd.to_numeric(self.df['UnitPrice'], errors='coerce').fillna(0.0)

        # 5. Clean CustomerID
        self.df['CustomerID'] = self.df['CustomerID'].astype(str).apply(lambda x: re.sub(r'[^\d\.]', '', x))

        # 6. Clean Country
        self.df['Country'] = self.df['Country'].astype(str).apply(lambda x: re.sub(r'[^a-zA-Z\s]', '', x).replace('XxY', '').strip())

        # 7. Clean InvoiceNo (careful not to lose 'C' prefix if it exists)
        # If you are SURE you only want numbers, keep your original line.
        # If you want to keep 'C' for cancellations, use this:
        self.df['InvoiceNo'] = self.df['InvoiceNo'].astype(str).apply(lambda x: re.sub(r'[^A-Za-z0-9]', '', x))

        # 8. Remove Duplicates
        self.df = self.df.drop_duplicates()

        # 9. Drop rows containing ANY NaN values
        self.df = self.df.dropna()

        # SAVE cleaned data
        # FIXED: Changed Config.CLEAN_DATA_PATH to config.CLEAN_DATA_PATH
        self.df.to_csv(config.CLEAN_DATA_PATH, index=False)
        print(f"Data cleaned and saved to {config.CLEAN_DATA_PATH}. Shape: {self.df.shape}")
        return self.df

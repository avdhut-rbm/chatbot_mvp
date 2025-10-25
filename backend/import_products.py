#!/usr/bin/env python3
"""
Bulk import products into OpenSearch rh_products index
"""

import json
import requests
from typing import List, Dict, Any

def bulk_import_products(products: List[Dict[str, Any]], opensearch_url: str = "http://localhost:9200"):
    """Import products in bulk to OpenSearch"""
    
    # Prepare bulk data
    bulk_data = []
    for product in products:
        # Add index action
        bulk_data.append({
            "index": {
                "_index": "rh_products",
                "_id": product["id"]
            }
        })
        # Add product data
        bulk_data.append(product)
    
    # Convert to NDJSON format
    ndjson_data = ""
    for item in bulk_data:
        ndjson_data += json.dumps(item) + "\n"
    
    # Send bulk request
    url = f"{opensearch_url}/_bulk"
    headers = {"Content-Type": "application/x-ndjson"}
    
    print(f"Importing {len(products)} products to OpenSearch...")
    
    try:
        response = requests.post(url, data=ndjson_data, headers=headers, auth=("admin", "admin"))
        response.raise_for_status()
        
        result = response.json()
        
        if result.get("errors"):
            print(f"Some errors occurred during import:")
            for item in result.get("items", []):
                if "index" in item and item["index"].get("error"):
                    print(f"Error for ID {item['index']['_id']}: {item['index']['error']}")
        else:
            print(f"Successfully imported {len(products)} products!")
            
        print(f"Took: {result.get('took')}ms")
        
    except requests.exceptions.RequestException as e:
        print(f"Error importing products: {e}")

def main():
    """Load products and import them"""
    try:
        # Load products from JSON file
        with open('products_1000.json', 'r') as f:
            products = json.load(f)
        
        print(f"Loaded {len(products)} products from products_1000.json")
        
        # Import to OpenSearch
        bulk_import_products(products)
        
    except FileNotFoundError:
        print("Error: products_1000.json not found. Please run generate_products.py first.")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Script to recreate the amazon_electronics index with proper dense vector mapping for kNN search
"""

import asyncio
import httpx
import json
import logging
from typing import List, Dict, Any

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class IndexRecreator:
    def __init__(self, opensearch_host: str = "localhost", opensearch_port: int = 9200):
        self.opensearch_url = f"http://{opensearch_host}:{opensearch_port}"
        self.index_name = "amazon_electronics"
        
    async def get_index_mapping(self) -> Dict[str, Any]:
        """Get current index mapping"""
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(f"{self.opensearch_url}/{self.index_name}/_mapping")
                response.raise_for_status()
                return response.json()
        except Exception as e:
            logger.error(f"Error getting mapping: {e}")
            return {}
    
    async def create_proper_mapping(self) -> Dict[str, Any]:
        """Create proper mapping with dense vector field"""
        return {
            "mappings": {
                "properties": {
                    "id": {"type": "keyword"},
                    "name": {
                        "type": "text",
                        "fields": {
                            "keyword": {"type": "keyword"}
                        },
                        "analyzer": "product_analyzer"
                    },
                    "description": {
                        "type": "text",
                        "analyzer": "product_analyzer"
                    },
                    "category": {"type": "keyword"},
                    "subcategory": {"type": "keyword"},
                    "brand": {"type": "keyword"},
                    "price": {"type": "float"},
                    "currency": {"type": "keyword"},
                    "rating": {"type": "float"},
                    "reviews_count": {"type": "integer"},
                    "availability": {"type": "keyword"},
                    "color": {"type": "keyword"},
                    "material": {"type": "keyword"},
                    "tags": {"type": "keyword"},
                    "image_url": {"type": "keyword"},
                    "image_urls": {"type": "keyword"},
                    "sku": {"type": "keyword"},
                    "dimensions": {
                        "properties": {
                            "width": {"type": "float"},
                            "height": {"type": "float"},
                            "depth": {"type": "float"},
                            "weight": {"type": "float"}
                        }
                    },
                    "created_at": {"type": "date"},
                    "updated_at": {"type": "date"},
                    "embedding_text": {
                        "type": "text",
                        "fields": {
                            "keyword": {"type": "keyword", "ignore_above": 256}
                        }
                    },
                    "product_embedding": {
                        "type": "dense_vector",
                        "dims": 4096,
                        "index": True,
                        "similarity": "cosine"
                    }
                }
            },
            "settings": {
                "number_of_shards": 1,
                "number_of_replicas": 0,
                "analysis": {
                    "analyzer": {
                        "product_analyzer": {
                            "type": "custom",
                            "tokenizer": "standard",
                            "filter": ["lowercase", "stop", "snowball"]
                        }
                    }
                }
            }
        }
    
    async def backup_documents(self) -> List[Dict[str, Any]]:
        """Backup all documents from the current index"""
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{self.opensearch_url}/{self.index_name}/_search",
                    json={
                        "size": 1000,
                        "query": {"match_all": {}},
                        "_source": True
                    },
                    headers={"Content-Type": "application/json"},
                    timeout=30.0
                )
                response.raise_for_status()
                
                result = response.json()
                hits = result.get("hits", {}).get("hits", [])
                
                documents = []
                for hit in hits:
                    doc = hit["_source"]
                    doc["_id"] = hit["_id"]
                    documents.append(doc)
                
                logger.info(f"Backed up {len(documents)} documents")
                return documents
                
        except Exception as e:
            logger.error(f"Error backing up documents: {e}")
            return []
    
    async def delete_index(self):
        """Delete the current index"""
        try:
            async with httpx.AsyncClient() as client:
                response = await client.delete(f"{self.opensearch_url}/{self.index_name}")
                response.raise_for_status()
                logger.info(f"Deleted index {self.index_name}")
        except Exception as e:
            logger.error(f"Error deleting index: {e}")
    
    async def create_index(self, mapping: Dict[str, Any]):
        """Create new index with proper mapping"""
        try:
            async with httpx.AsyncClient() as client:
                response = await client.put(
                    f"{self.opensearch_url}/{self.index_name}",
                    json=mapping,
                    headers={"Content-Type": "application/json"},
                    timeout=30.0
                )
                response.raise_for_status()
                logger.info(f"Created index {self.index_name} with proper mapping")
        except Exception as e:
            logger.error(f"Error creating index: {e}")
    
    async def restore_documents(self, documents: List[Dict[str, Any]]):
        """Restore documents to the new index"""
        try:
            async with httpx.AsyncClient() as client:
                for doc in documents:
                    doc_id = doc.pop("_id")
                    
                    response = await client.put(
                        f"{self.opensearch_url}/{self.index_name}/_doc/{doc_id}",
                        json=doc,
                        headers={"Content-Type": "application/json"},
                        timeout=10.0
                    )
                    response.raise_for_status()
                
                logger.info(f"Restored {len(documents)} documents")
        except Exception as e:
            logger.error(f"Error restoring documents: {e}")
    
    async def recreate_index(self):
        """Main method to recreate the index"""
        logger.info("Starting index recreation process...")
        
        # Step 1: Backup documents
        documents = await self.backup_documents()
        if not documents:
            logger.error("No documents to backup")
            return
        
        # Step 2: Create proper mapping
        mapping = await self.create_proper_mapping()
        
        # Step 3: Delete old index
        await self.delete_index()
        
        # Step 4: Create new index with proper mapping
        await self.create_index(mapping)
        
        # Step 5: Restore documents
        await self.restore_documents(documents)
        
        logger.info("Index recreation completed!")
    
    async def verify_index(self):
        """Verify the new index is working correctly"""
        try:
            async with httpx.AsyncClient() as client:
                # Check mapping
                response = await client.get(f"{self.opensearch_url}/{self.index_name}/_mapping")
                response.raise_for_status()
                mapping = response.json()
                
                embedding_field = mapping.get(self.index_name, {}).get("mappings", {}).get("properties", {}).get("product_embedding", {})
                
                if embedding_field.get("type") == "dense_vector":
                    logger.info("✅ Index has proper dense_vector mapping!")
                else:
                    logger.warning(f"⚠️ Embedding field type: {embedding_field.get('type')}")
                
                # Test search
                response = await client.post(
                    f"{self.opensearch_url}/{self.index_name}/_search",
                    json={
                        "size": 1,
                        "query": {"match_all": {}},
                        "_source": ["name", "product_embedding"]
                    },
                    headers={"Content-Type": "application/json"},
                    timeout=10.0
                )
                response.raise_for_status()
                
                result = response.json()
                hits = result.get("hits", {}).get("hits", [])
                
                if hits:
                    doc = hits[0]["_source"]
                    embedding = doc.get("product_embedding", [])
                    logger.info(f"Sample document: {doc.get('name', 'Unknown')}")
                    logger.info(f"Embedding dimensions: {len(embedding)}")
                    
                    if len(embedding) == 4096:
                        logger.info("✅ Embeddings are correctly stored!")
                    else:
                        logger.warning(f"⚠️ Embedding dimensions: {len(embedding)} (expected 4096)")
                
        except Exception as e:
            logger.error(f"Error verifying index: {e}")

async def main():
    recreator = IndexRecreator()
    
    # Recreate index
    await recreator.recreate_index()
    
    # Verify index
    await recreator.verify_index()

if __name__ == "__main__":
    asyncio.run(main())

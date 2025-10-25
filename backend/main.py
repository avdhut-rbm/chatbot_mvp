from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import json
import logging
import httpx
import numpy as np
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Chatbot API", version="1.0.0")

# Global conversation memory (in production, use Redis or database)
conversation_sessions = {}

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Allow your Next.js frontend origin
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = None
    conversation_history: Optional[List[Dict[str, str]]] = None

class ChatResponse(BaseModel):
    response: str
    session_id: str
    timestamp: str
    sources: Optional[List[Dict[str, Any]]] = None

class OpenSearchClient:
    def __init__(self, host: str = "localhost", port: int = 9200):
        self.base_url = f"http://{host}:{port}"
        
    async def search_products(self, query: str, size: int = 5) -> List[Dict[str, Any]]:
        """Search for products using hybrid search (semantic + keyword)"""
        try:
            async with httpx.AsyncClient() as client:
                # First, get embeddings for the query using Ollama
                embedding = await self._get_query_embedding(query)
                
                # Test with keyword search only first
                search_query = {
                        "size": size,
                        "query": {
                            "multi_match": {
                                "query": query,
                                "fields": ["name^3", "description^2", "category^2", "brand", "tags", "subcategory"],
                                "type": "best_fields",
                                "fuzziness": "AUTO"
                            }
                        },
                        "_source": {
                            "excludes": ["product_embedding"]  # Exclude embeddings from response
                        }
                    }
                
                response = await client.post(
                    f"{self.base_url}/amazon_electronics/_search",
                    json=search_query,
                    headers={"Content-Type": "application/json"},
                    timeout=10.0
                )
                response.raise_for_status()
                
                result = response.json()
                hits = result.get("hits", {}).get("hits", [])
                
                return [hit["_source"] for hit in hits]
                
        except Exception as e:
            logger.error(f"OpenSearch search error: {e}")
            return []
    
    async def _get_query_embedding(self, query: str) -> List[float]:
        """Get embedding for the query using Ollama"""
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    "http://localhost:11434/api/embeddings",
                    json={
                        "model": "llama3.1:8b",
                        "prompt": query
                    },
                    timeout=10.0
                )
                response.raise_for_status()
                result = response.json()
                return result.get("embedding", [])
        except Exception as e:
            logger.error(f"Embedding generation error: {e}")
            # Return a zero vector as fallback
            return [0.0] * 4096  # Llama 3.1 8B embedding dimension

# Initialize OpenSearch client
opensearch_client = OpenSearchClient()

def get_conversation_history(session_id: str, max_messages: int = 6) -> str:
    """Get formatted conversation history"""
    if session_id not in conversation_sessions:
        conversation_sessions[session_id] = []
    
    # Get last max_messages from history
    recent_history = conversation_sessions[session_id][-max_messages:]
    
    # Format as text
    history_text = "\n".join([
        f"{msg['role']}: {msg['content']}" 
        for msg in recent_history
    ])
    
    return history_text

def save_conversation(session_id: str, user_message: str, assistant_response: str):
    """Save conversation to memory"""
    if session_id not in conversation_sessions:
        conversation_sessions[session_id] = []
    
    conversation_sessions[session_id].extend([
        {"role": "user", "content": user_message},
        {"role": "assistant", "content": assistant_response}
    ])
    
    # Keep only last 20 messages to prevent memory bloat
    if len(conversation_sessions[session_id]) > 20:
        conversation_sessions[session_id] = conversation_sessions[session_id][-20:]

async def extract_filters_with_context(query: str, history_text: str) -> Optional[List[Dict]]:
    """Extract filters using LLM with conversation context"""
    prompt = f"""Extract filters from this query. Return ONLY a JSON object.

Previous conversation: {history_text}
Current query: {query}

Extract these filters if mentioned:
- price_min, price_max (numbers)
- brand (exact match)
- color (exact match)
- category (exact match)
- min_rating (number)

Return format:
{{"price_min": 100, "brand": "Samsung"}}

If no filters, return: {{}}

JSON:"""
    
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                "http://localhost:11434/api/generate",
                json={
                    "model": "llama3.1:8b",
                    "prompt": prompt,
                    "stream": False
                },
                timeout=15.0
            )
            response.raise_for_status()
            result = response.json()
            result_text = result.get("response", "{}")
            
            # Parse JSON response
            filters_dict = json.loads(result_text.strip())
            
            # Convert to OpenSearch filter format
            os_filters = []
            
            if "price_min" in filters_dict or "price_max" in filters_dict:
                price_range = {}
                if "price_min" in filters_dict:
                    price_range["gte"] = filters_dict["price_min"]
                if "price_max" in filters_dict:
                    price_range["lte"] = filters_dict["price_max"]
                os_filters.append({"range": {"price": price_range}})
            
            if "brand" in filters_dict:
                os_filters.append({"term": {"brand": filters_dict["brand"]}})
            
            if "color" in filters_dict:
                os_filters.append({"term": {"color": filters_dict["color"]}})
            
            if "category" in filters_dict:
                os_filters.append({"term": {"category": filters_dict["category"]}})
            
            if "min_rating" in filters_dict:
                os_filters.append({"range": {"rating": {"gte": filters_dict["min_rating"]}}})
            
            return os_filters if os_filters else None
            
    except Exception as e:
        logger.error(f"Filter extraction error: {e}")
        return None

def create_enhanced_rag_prompt(user_message: str, products: List[Dict[str, Any]], history_text: str) -> str:
    """Create RAG prompt with conversation history"""
    if not products:
        return f"""You are a helpful ecommerce assistant. Answer the user's question about products or shopping.

Previous conversation:
{history_text}

User question: {user_message}

Please provide a helpful response about ecommerce and shopping."""

    # Format product information - USE ACTUAL PRODUCT NAMES
    product_context = "\n\n".join([
        f"""{product.get('name', 'N/A')}:
- Brand: {product.get('brand', 'N/A')}
- Price: ${product.get('price', 'N/A')} {product.get('currency', 'USD')}
- Category: {product.get('category', 'N/A')}
- Description: {product.get('description', 'N/A')}
- Rating: {product.get('rating', 'N/A')}/5 ({product.get('reviews_count', 0)} reviews)
- Availability: {product.get('availability', 'N/A')}
- Color: {product.get('color', 'N/A')}
- Material: {product.get('material', 'N/A')}"""
        for i, product in enumerate(products)
    ])

    return f"""You are a helpful ecommerce assistant. Use the product information below to answer the user's question about products or shopping.

Previous conversation:
{history_text}

User question: {user_message}

Available products:
{product_context}

Instructions:
1. Use the product information to provide accurate, helpful responses
2. Consider the conversation history for context and follow-up questions
3. If the user asks about specific products, reference the relevant products from the list
4. If no relevant products are found, suggest similar alternatives or explain what's available
5. Be conversational and helpful
6. Include pricing, ratings, and availability information when relevant
7. If the user asks for recommendations, suggest the best products based on ratings and features
8. Remember previous preferences mentioned in the conversation

Please provide a helpful response based on the available product information and conversation context."""

# Enhanced RAG-powered chat endpoint with conversation memory
@app.post("/api/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Enhanced chat endpoint with conversation memory"""
    logger.info(f"Received message: {request.message}")
    
    session_id = request.session_id or "default"
    
    try:
        # Step 1: Get conversation history
        history_text = get_conversation_history(session_id)
        logger.info(f"Retrieved conversation history for session: {session_id}")
        
        # Step 2: Extract filters with context
        filters = await extract_filters_with_context(request.message, history_text)
        if filters:
            logger.info(f"Extracted filters: {filters}")
        
        # Step 3: Search for relevant products using OpenSearch
        logger.info("Searching for relevant products...")
        products = await opensearch_client.search_products(request.message, size=3)
        logger.info(f"Found {len(products)} relevant products")
        
        # Step 4: Create enhanced RAG prompt with conversation history
        rag_prompt = create_enhanced_rag_prompt(request.message, products, history_text)
        
        # Step 5: Generate response using Ollama with RAG context
        logger.info("Generating response with Ollama...")
        async with httpx.AsyncClient() as client:
            ollama_response = await client.post(
                "http://localhost:11434/api/generate",
                json={
                    "model": "llama3.1:8b",
                    "prompt": rag_prompt,
                    "stream": False
                },
                timeout=30.0
            )
            ollama_response.raise_for_status()
            result = ollama_response.json()
            response_text = result.get("response", "Sorry, I couldn't generate a response.")
            
        # Step 6: Save conversation to memory
        save_conversation(session_id, request.message, response_text)
        logger.info(f"Saved conversation to session: {session_id}")
            
    except Exception as e:
        logger.error(f"RAG API error: {e}")
        response_text = f"Sorry, I'm having trouble processing your request: {request.message}"
        products = []
    
    return ChatResponse(
        response=response_text,
        session_id=session_id,
        timestamp=datetime.now().isoformat() + "Z",
        sources=products if products else None
    )

# Enhanced WebSocket endpoint for real-time chat with conversation memory
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """Enhanced WebSocket endpoint with conversation memory"""
    await websocket.accept()
    logger.info("WebSocket connection established")
    
    try:
        while True:
            data = await websocket.receive_text()
            message_data = json.loads(data)
            
            logger.info(f"WebSocket received: {message_data}")
            
            user_message = message_data.get('message', '')
            session_id = message_data.get('session_id', 'default')
            
            try:
                # Step 1: Get conversation history
                history_text = get_conversation_history(session_id)
                logger.info(f"Retrieved conversation history for session: {session_id}")
                
                # Step 2: Extract filters with context
                filters = await extract_filters_with_context(user_message, history_text)
                if filters:
                    logger.info(f"Extracted filters: {filters}")
                
                # Step 3: Search for relevant products using OpenSearch
                logger.info("Searching for relevant products...")
                products = await opensearch_client.search_products(user_message, size=3)
                logger.info(f"Found {len(products)} relevant products")
                
                # Step 4: Create enhanced RAG prompt with conversation history
                rag_prompt = create_enhanced_rag_prompt(user_message, products, history_text)
                
                # Step 5: Generate response using Ollama with RAG context
                logger.info("Generating response with Ollama...")
                async with httpx.AsyncClient() as client:
                    ollama_response = await client.post(
                        "http://localhost:11434/api/generate",
                        json={
                            "model": "llama3.1:8b",
                            "prompt": rag_prompt,
                            "stream": False
                        },
                        timeout=30.0
                    )
                    ollama_response.raise_for_status()
                    result = ollama_response.json()
                    response_text = result.get("response", "Sorry, I couldn't generate a response.")
                    
                # Step 6: Save conversation to memory
                save_conversation(session_id, user_message, response_text)
                logger.info(f"Saved conversation to session: {session_id}")
                    
            except Exception as e:
                logger.error(f"RAG API error: {e}")
                response_text = f"Sorry, I'm having trouble processing your request: {user_message}"
                products = []
            
            response = {
                "response": response_text,
                "session_id": session_id,
                "timestamp": datetime.now().isoformat() + "Z",
                "sources": products if products else None
            }
            
            await websocket.send_text(json.dumps(response))
            
    except WebSocketDisconnect:
        logger.info("WebSocket connection closed")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")

@app.get("/health")
async def health_check():
    return {"status": "healthy", "message": "RAG-powered backend with conversation memory is running"}

@app.get("/api/conversation/{session_id}")
async def get_conversation_history_endpoint(session_id: str):
    """Get conversation history for a session (for debugging)"""
    if session_id not in conversation_sessions:
        return {"session_id": session_id, "history": [], "message": "No conversation history found"}
    
    return {
        "session_id": session_id,
        "history": conversation_sessions[session_id],
        "message": f"Found {len(conversation_sessions[session_id])} messages"
    }

@app.get("/api/search")
async def search_products(query: str, size: int = 5):
    """Direct search endpoint for testing OpenSearch functionality"""
    try:
        products = await opensearch_client.search_products(query, size)
        return {
            "query": query,
            "results": products,
            "count": len(products)
        }
    except Exception as e:
        logger.error(f"Search error: {e}")
        return {"error": str(e), "query": query, "results": []}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

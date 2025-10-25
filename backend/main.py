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


# ============================================================================
# DOMAIN CONFIGURATION - Static Fallback + Dynamic from OpenSearch
# ============================================================================

# Static fallback configuration
DOMAIN_CONFIG = {
    "business_type": "ecommerce",
    "business_name": "Your Store",
    "categories": ["electronics", "furniture", "home goods"],
    "allowed_topics": [
        # Base shopping topics
        'product', 'item', 'buy', 'purchase', 'price', 'show', 'find', 'search',
        'browse', 'catalog', 'inventory', 'recommend', 'suggest', 'compare',
        'review', 'rating', 'best', 'top', 'popular', 'brand', 'color', 'size',
        'available', 'stock', 'order', 'delivery', 'shipping', 'return',
        'warranty', 'deal', 'discount', 'sale', 'help', 'question'
    ],
    "off_topic_keywords": [
        'code', 'python', 'javascript', 'programming', 'function', 'optimize',
        'debug', 'algorithm', 'server', 'database', 'api', 'backend', 'frontend',
        'deploy', 'docker', 'kubernetes', 'git', 'repository', 'compile',
        'syntax', 'variable', 'loop', 'array', 'class', 'object'
    ]
}

# Dynamic configuration loaded from OpenSearch
DYNAMIC_CONFIG = {}


# Global conversation memory (in production, use Redis or database)
conversation_sessions = {}


# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
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


# ============================================================================
# DYNAMIC CONFIG LOADING FROM OPENSEARCH
# ============================================================================

async def load_dynamic_config_from_opensearch():
    """Load categories, brands, and attributes from OpenSearch index"""
    global DYNAMIC_CONFIG
    
    try:
        async with httpx.AsyncClient() as client:
            # Get aggregations for categories, brands, colors, materials
            agg_query = {
                "size": 0,
                "aggs": {
                    "categories": {
                        "terms": {"field": "category.keyword", "size": 50}
                    },
                    "subcategories": {
                        "terms": {"field": "subcategory.keyword", "size": 100}
                    },
                    "brands": {
                        "terms": {"field": "brand.keyword", "size": 100}
                    },
                    "colors": {
                        "terms": {"field": "color.keyword", "size": 50}
                    },
                    "materials": {
                        "terms": {"field": "material.keyword", "size": 50}
                    }
                }
            }
            
            response = await client.post(
                "http://localhost:9200/amazon_electronics/_search",
                json=agg_query,
                headers={"Content-Type": "application/json"},
                timeout=10.0
            )
            response.raise_for_status()
            result = response.json()
            
            aggs = result.get("aggregations", {})
            
            # Extract values from aggregations
            categories = [bucket["key"] for bucket in aggs.get("categories", {}).get("buckets", [])]
            subcategories = [bucket["key"] for bucket in aggs.get("subcategories", {}).get("buckets", [])]
            brands = [bucket["key"] for bucket in aggs.get("brands", {}).get("buckets", [])]
            colors = [bucket["key"] for bucket in aggs.get("colors", {}).get("buckets", [])]
            materials = [bucket["key"] for bucket in aggs.get("materials", {}).get("buckets", [])]
            
            # Build dynamic allowed topics from actual data
            dynamic_topics = set(DOMAIN_CONFIG['allowed_topics'])  # Start with base topics
            
            # Add all actual categories, brands, colors, materials as allowed topics
            dynamic_topics.update([cat.lower() for cat in categories])
            dynamic_topics.update([sub.lower() for sub in subcategories])
            dynamic_topics.update([brand.lower() for brand in brands])
            dynamic_topics.update([color.lower() for color in colors])
            dynamic_topics.update([mat.lower() for mat in materials])
            
            DYNAMIC_CONFIG = {
                "categories": categories,
                "subcategories": subcategories,
                "brands": brands,
                "colors": colors,
                "materials": materials,
                "allowed_topics": list(dynamic_topics),
                "last_updated": datetime.now().isoformat()
            }
            
            logger.info(f"✅ Loaded dynamic config from OpenSearch:")
            logger.info(f"   Categories: {len(categories)}")
            logger.info(f"   Brands: {len(brands)}")
            logger.info(f"   Total allowed topics: {len(dynamic_topics)}")
            
            return True
            
    except Exception as e:
        logger.error(f"Failed to load dynamic config from OpenSearch: {e}")
        logger.info("Using static DOMAIN_CONFIG as fallback")
        return False


def get_allowed_topics() -> List[str]:
    """Get allowed topics from dynamic config or fall back to static"""
    if DYNAMIC_CONFIG and "allowed_topics" in DYNAMIC_CONFIG:
        return DYNAMIC_CONFIG["allowed_topics"]
    return DOMAIN_CONFIG["allowed_topics"]


def get_categories() -> List[str]:
    """Get categories from dynamic config or fall back to static"""
    if DYNAMIC_CONFIG and "categories" in DYNAMIC_CONFIG:
        return DYNAMIC_CONFIG["categories"]
    return DOMAIN_CONFIG["categories"]


def get_business_name() -> str:
    """Get business name from static config"""
    return DOMAIN_CONFIG["business_name"]


# ============================================================================
# OPENSEARCH CLIENT
# ============================================================================

class OpenSearchClient:
    def __init__(self, host: str = "localhost", port: int = 9200):
        self.base_url = f"http://{host}:{port}"
        
    async def search_products(self, query: str, size: int = 5) -> List[Dict[str, Any]]:
        """Search for products using keyword search"""
        try:
            async with httpx.AsyncClient() as client:
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
                        "excludes": ["product_embedding"]
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


# Initialize OpenSearch client
opensearch_client = OpenSearchClient()


# ============================================================================
# CONTEXT VALIDATION WITH DYNAMIC CONFIG
# ============================================================================

def is_on_topic(query: str) -> bool:
    """Check if query is related to the business domain (using dynamic config)"""
    query_lower = query.lower()
    
    # Get allowed topics from dynamic or static config
    allowed_topics = get_allowed_topics()
    
    # Check if query contains allowed topics
    has_allowed_topic = any(
        keyword in query_lower 
        for keyword in allowed_topics
    )
    
    # Check if query contains off-topic keywords (always static)
    has_off_topic = any(
        keyword in query_lower 
        for keyword in DOMAIN_CONFIG['off_topic_keywords']
    )
    
    return has_allowed_topic and not has_off_topic


def get_redirect_response(query: str) -> str:
    """Return appropriate redirect message (using dynamic config)"""
    categories = get_categories()
    categories_text = ", ".join(categories[:3])
    business_name = get_business_name()
    
    return f"I'm your {business_name} shopping assistant! I specialize in helping you find and compare products in {categories_text}, and more. What product are you looking for today?"


# ============================================================================
# CONVERSATION MANAGEMENT
# ============================================================================

def get_conversation_history(session_id: str, max_messages: int = 4) -> str:
    """Get formatted conversation history"""
    if session_id not in conversation_sessions:
        conversation_sessions[session_id] = []
    
    recent_history = conversation_sessions[session_id][-max_messages:]
    
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
    
    # Keep only last 12 messages
    if len(conversation_sessions[session_id]) > 12:
        conversation_sessions[session_id] = conversation_sessions[session_id][-12:]


def truncate_response(text: str, max_sentences: int = 4) -> str:
    """Truncate response to max sentences"""
    sentences = text.split('. ')
    if len(sentences) > max_sentences:
        return '. '.join(sentences[:max_sentences]) + '.'
    return text


# ============================================================================
# FILTER EXTRACTION
# ============================================================================

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
                    "stream": False,
                    "options": {
                        "temperature": 0.3,
                        "num_predict": 50
                    }
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


# ============================================================================
# DYNAMIC RAG PROMPT
# ============================================================================

def create_enhanced_rag_prompt(user_message: str, products: List[Dict[str, Any]], history_text: str) -> str:
    """Create concise RAG prompt with dynamic business context"""
    
    # Use dynamic categories if available
    categories = get_categories()
    categories_list = ", ".join(categories)
    business_name = get_business_name()
    
    system_context = f"""You are a focused shopping assistant for {business_name}.

Your responsibilities:
- Help customers discover products in: {categories_list}
- Compare products, prices, and features
- Provide personalized recommendations
- Answer questions about product specifications, availability, and policies

Stay focused on shopping assistance. Politely redirect off-topic questions."""

    if not products:
        return f"""{system_context}

Previous conversation:
{history_text}

Customer question: {user_message}

Response (2-3 sentences, helpful and product-focused):"""

    # Format product information
    product_context = "\n\n".join([
        f"""Product {i+1}: {product.get('name', 'N/A')}
Brand: {product.get('brand', 'N/A')} | Price: ${product.get('price', 'N/A')} | Rating: {product.get('rating', 'N/A')}/5
{product.get('description', 'N/A')[:80]}"""
        for i, product in enumerate(products[:3])
    ])

    return f"""{system_context}

Previous conversation:
{history_text}

Customer question: {user_message}

Available products:
{product_context}

Instructions:
- Answer in 2-4 sentences
- Focus on products and shopping value
- Mention names, prices, key features
- Be conversational and helpful

Response:"""


# ============================================================================
# API ENDPOINTS
# ============================================================================

@app.on_event("startup")
async def startup_event():
    """Load dynamic configuration on startup"""
    logger.info("🚀 Starting up chatbot API...")
    logger.info("📊 Loading dynamic configuration from OpenSearch...")
    
    success = await load_dynamic_config_from_opensearch()
    
    if success:
        logger.info("✅ Dynamic configuration loaded successfully")
        categories = get_categories()
        logger.info(f"   Available categories: {', '.join(categories[:5])}{'...' if len(categories) > 5 else ''}")
    else:
        logger.warning("⚠️  Using static fallback configuration")


@app.post("/api/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Enhanced chat endpoint with conversation memory and context filtering"""
    logger.info(f"Received message: {request.message}")
    
    session_id = request.session_id or "default"
    
    # Guard against off-topic queries
    if not is_on_topic(request.message):
        redirect_response = get_redirect_response(request.message)
        logger.info(f"Off-topic query detected: {request.message}")
        save_conversation(session_id, request.message, redirect_response)
        return ChatResponse(
            response=redirect_response,
            session_id=session_id,
            timestamp=datetime.now().isoformat() + "Z",
            sources=None
        )
    
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
                    "stream": False,
                    "options": {
                        "temperature": 0.7,
                        "top_p": 0.9,
                        "num_predict": 150,
                        "stop": ["\n\n\n", "User:", "Question:", "Previous conversation:"]
                    }
                },
                timeout=30.0
            )
            ollama_response.raise_for_status()
            result = ollama_response.json()
            response_text = result.get("response", "Sorry, I couldn't generate a response.")
            
            # Enforce brevity
            response_text = truncate_response(response_text, max_sentences=4)
        
        # Step 6: Save conversation to memory
        save_conversation(session_id, request.message, response_text)
        logger.info(f"Saved conversation to session: {session_id}")
            
    except Exception as e:
        logger.error(f"RAG API error: {e}")
        response_text = f"Sorry, I'm having trouble processing your request right now."
        products = []
    
    return ChatResponse(
        response=response_text,
        session_id=session_id,
        timestamp=datetime.now().isoformat() + "Z",
        sources=products if products else None
    )


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """Enhanced WebSocket endpoint with conversation memory and context filtering"""
    await websocket.accept()
    logger.info("WebSocket connection established")
    
    try:
        while True:
            data = await websocket.receive_text()
            message_data = json.loads(data)
            
            logger.info(f"WebSocket received: {message_data}")
            
            user_message = message_data.get('message', '')
            session_id = message_data.get('session_id', 'default')
            
            # Guard against off-topic queries
            if not is_on_topic(user_message):
                redirect_response = get_redirect_response(user_message)
                logger.info(f"Off-topic query detected: {user_message}")
                save_conversation(session_id, user_message, redirect_response)
                
                response = {
                    "response": redirect_response,
                    "session_id": session_id,
                    "timestamp": datetime.now().isoformat() + "Z",
                    "sources": None
                }
                
                await websocket.send_text(json.dumps(response))
                continue
            
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
                            "stream": False,
                            "options": {
                                "temperature": 0.7,
                                "top_p": 0.9,
                                "num_predict": 150,
                                "stop": ["\n\n\n", "User:", "Question:", "Previous conversation:"]
                            }
                        },
                        timeout=30.0
                    )
                    ollama_response.raise_for_status()
                    result = ollama_response.json()
                    response_text = result.get("response", "Sorry, I couldn't generate a response.")
                    
                    # Enforce brevity
                    response_text = truncate_response(response_text, max_sentences=4)
                    
                # Step 6: Save conversation to memory
                save_conversation(session_id, user_message, response_text)
                logger.info(f"Saved conversation to session: {session_id}")
                    
            except Exception as e:
                logger.error(f"RAG API error: {e}")
                response_text = f"Sorry, I'm having trouble processing your request right now."
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
    return {"status": "healthy", "message": "RAG-powered backend with dynamic config is running"}


@app.post("/api/config/refresh")
async def refresh_config():
    """Manually refresh dynamic configuration from OpenSearch"""
    logger.info("🔄 Refreshing dynamic configuration...")
    
    success = await load_dynamic_config_from_opensearch()
    
    if success:
        return {
            "status": "success",
            "message": "Dynamic configuration refreshed",
            "categories": get_categories(),
            "total_topics": len(get_allowed_topics()),
            "last_updated": DYNAMIC_CONFIG.get("last_updated")
        }
    else:
        return {
            "status": "error",
            "message": "Failed to refresh configuration, using fallback",
            "categories": get_categories()
        }


@app.get("/api/config")
async def get_config():
    """Get current configuration"""
    return {
        "business_name": get_business_name(),
        "categories": get_categories(),
        "brands": DYNAMIC_CONFIG.get("brands", [])[:10] if DYNAMIC_CONFIG else [],
        "using_dynamic": bool(DYNAMIC_CONFIG),
        "total_allowed_topics": len(get_allowed_topics()),
        "last_updated": DYNAMIC_CONFIG.get("last_updated") if DYNAMIC_CONFIG else None
    }


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

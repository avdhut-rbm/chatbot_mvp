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
        # Product discovery
        'product', 'item', 'buy', 'purchase', 'price', 'show', 'find', 'search',
        'browse', 'catalog', 'inventory', 'recommend', 'suggest', 'compare',
        'review', 'rating', 'best', 'top', 'popular', 'brand', 'color', 'size',
        'available', 'stock', 'order', 'delivery', 'shipping', 'return',
        'warranty', 'deal', 'discount', 'sale', 'help', 'question',
        
        # Product variations and listing
        'model', 'models', 'list', 'all', 'more', 'other', 'variant', 'version',
        'type', 'kind', 'option', 'selection', 'range', 'series', 'tell', 'down',
        
        # Common brands
        'lenovo', 'samsung', 'apple', 'dell', 'hp', 'sony', 'lg', 'microsoft',
        'asus', 'acer', 'canon', 'nikon', 'intel', 'amd', 'nvidia',
        
        # Material and specifications
        'material', 'style', 'design', 'feature', 'specification', 'dimension', 'weight',
        
        # Furniture specific
        'sofa', 'chair', 'table', 'bed', 'desk', 'cabinet', 'shelf', 'decor',
        'fabric', 'leather', 'wood', 'metal', 'finish', 'upholstery',
        
        # Electronics specific
        'phone', 'laptop', 'camera', 'tablet', 'headphone', 'watch', 'speaker',
        'screen', 'battery', 'storage', 'processor', 'display',
        
        # Fashion specific
        'clothing', 'shoe', 'dress', 'shirt', 'pants', 'jacket', 'accessory',
        
        # Customer service
        'how', 'what', 'where', 'when', 'which', 'why', 'tell', 'explain', 'yes', 'no'
    ],
    "off_topic_keywords": [
        'code', 'python', 'javascript', 'programming', 'function', 'optimize',
        'debug', 'algorithm', 'server', 'database', 'api', 'backend', 'frontend',
        'deploy', 'docker', 'kubernetes', 'git', 'repository', 'compile',
        'syntax', 'variable', 'loop', 'array', 'class', 'object', 'method'
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


# ============================================================================
# RESPONSE MODELS WITH PRODUCT CARD SUPPORT
# ============================================================================

class ProductCard(BaseModel):
    """Structured product data for UI rendering"""
    id: str
    name: str
    brand: str
    price: float
    rating: Optional[float] = None
    image_url: Optional[str] = None
    description: Optional[str] = None
    category: Optional[str] = None
    subcategory: Optional[str] = None


class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = None
    conversation_history: Optional[List[Dict[str, str]]] = None


class ChatResponse(BaseModel):
    response: str
    session_id: str
    timestamp: str
    sources: Optional[List[Dict[str, Any]]] = None
    product_cards: Optional[List[ProductCard]] = None
    display_mode: str = "text"  # "text", "products", "mixed"


# ============================================================================
# DYNAMIC CONFIG LOADING FROM OPENSEARCH
# ============================================================================

async def load_dynamic_config_from_opensearch():
    """Load categories, brands, and attributes from OpenSearch index"""
    global DYNAMIC_CONFIG
    
    try:
        async with httpx.AsyncClient() as client:
            # Get aggregations - WITHOUT .keyword suffix
            agg_query = {
                "size": 0,
                "aggs": {
                    "categories": {
                        "terms": {"field": "category", "size": 50}
                    },
                    "subcategories": {
                        "terms": {"field": "subcategory", "size": 100}
                    },
                    "brands": {
                        "terms": {"field": "brand", "size": 100}
                    },
                    "colors": {
                        "terms": {"field": "color", "size": 50}
                    },
                    "materials": {
                        "terms": {"field": "material", "size": 50}
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
# OPENSEARCH CLIENT - UNLIMITED RESULTS
# ============================================================================

class OpenSearchClient:
    def __init__(self, host: str = "localhost", port: int = 9200):
        self.base_url = f"http://{host}:{port}"
        
    async def search_products(self, query: str, max_results: int = 10000) -> List[Dict[str, Any]]:
        """Search for ALL products matching query (no size limit)"""
        try:
            async with httpx.AsyncClient() as client:
                # First, get the total count
                count_query = {
                    "size": 0,
                    "query": {
                        "multi_match": {
                            "query": query,
                            "fields": ["name^3", "description^2", "category^2", "brand^1.5", "tags", "subcategory"],
                            "type": "best_fields",
                            "fuzziness": "AUTO"
                        }
                    },
                    "track_total_hits": True
                }
                
                count_response = await client.post(
                    f"{self.base_url}/amazon_electronics/_search",
                    json=count_query,
                    headers={"Content-Type": "application/json"},
                    timeout=10.0
                )
                count_response.raise_for_status()
                count_result = count_response.json()
                total_hits = count_result.get("hits", {}).get("total", {}).get("value", 0)
                
                logger.info(f"Total matching products: {total_hits}")
                
                # Cap at max_results for safety
                actual_size = min(total_hits, max_results)
                
                # Get ALL matching products
                search_query = {
                    "size": actual_size,
                    "query": {
                        "multi_match": {
                            "query": query,
                            "fields": ["name^3", "description^2", "category^2", "brand^1.5", "tags", "subcategory"],
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
                    timeout=15.0
                )
                response.raise_for_status()
                
                result = response.json()
                hits = result.get("hits", {}).get("hits", [])
                
                logger.info(f"Returning {len(hits)} products")
                return [hit["_source"] for hit in hits]
                
        except Exception as e:
            logger.error(f"OpenSearch search error: {e}")
            return []


# Initialize OpenSearch client
opensearch_client = OpenSearchClient()


# ============================================================================
# INTELLIGENT CONTEXT VALIDATION WITH LLM
# ============================================================================

async def is_on_topic_llm(query: str, history_text: str = "") -> bool:
    """Use LLM to intelligently determine if query is shopping-related"""
    
    # Quick keyword check for obvious off-topic cases (optimization)
    obvious_off_topic = ['code', 'python', 'javascript', 'programming', 'debug', 'server', 'database', 'deploy', 'git']
    if any(word in query.lower() for word in obvious_off_topic):
        return False
    
    # Quick keyword check for obvious on-topic (optimization)
    obvious_on_topic = ['product', 'buy', 'price', 'show', 'laptop', 'phone', 'compare', 'recommend']
    if any(word in query.lower() for word in obvious_on_topic):
        return True
    
    # For ambiguous cases, ask LLM
    prompt = f"""Determine if this customer message is related to shopping/products.

Previous conversation context:
{history_text}

Customer message: {query}

Shopping-related topics include:
- Asking about products, features, prices, availability
- Requesting product recommendations or comparisons
- Follow-up questions about previously discussed products (like "yes please", "tell me more", "what about", "show me", "yes", "sure", "okay")
- Questions about delivery, warranty, returns
- General product inquiries
- Affirmative responses to product-related questions

NOT shopping-related:
- Programming/coding questions
- Technical implementation questions
- Unrelated general knowledge

Consider the conversation context. If the customer previously asked about products and now says "yes please", "tell me more", "yes", or similar affirmative responses, that IS shopping-related.

Answer with ONLY one word: YES or NO

Answer:"""
    
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                "http://localhost:11434/api/generate",
                json={
                    "model": "llama3.1:8b",
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.1,
                        "num_predict": 5
                    }
                },
                timeout=10.0
            )
            response.raise_for_status()
            result = response.json()
            answer = result.get("response", "NO").strip().upper()
            
            is_shopping = "YES" in answer
            logger.info(f"LLM topic classification for '{query}': {is_shopping}")
            return is_shopping
            
    except Exception as e:
        logger.error(f"LLM topic classification error: {e}")
        # Fallback to permissive - allow the query
        return True


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


# ============================================================================
# PRODUCT CARD FORMATTING
# ============================================================================

def format_product_cards(products: List[Dict[str, Any]]) -> List[ProductCard]:
    """Convert OpenSearch products to structured UI cards"""
    cards = []
    for product in products:
        try:
            card = ProductCard(
                id=str(product.get('product_id', product.get('id', hash(product.get('name', ''))))),
                name=product.get('name', 'Unknown Product'),
                brand=product.get('brand', 'Unknown'),
                price=float(product.get('price', 0)),
                rating=float(product.get('rating', 0)) if product.get('rating') else None,
                image_url=product.get('image_url', product.get('image')),
                description=product.get('description', '')[:200] if product.get('description') else None,
                category=product.get('category'),
                subcategory=product.get('subcategory')
            )
            cards.append(card)
        except Exception as e:
            logger.error(f"Error formatting product card: {e}")
            continue
    
    return cards


def detect_display_mode(user_query: str, product_count: int) -> str:
    """Determine if response should show product cards"""
    query_lower = user_query.lower()
    
    # Show product grid for listing queries
    listing_keywords = ['show', 'list', 'all', 'display', 'see', 'browse', 'available', 'give me']
    if any(kw in query_lower for kw in listing_keywords) and product_count > 0:
        return "products"
    
    # Show mixed (text + cards) for comparison queries with multiple products
    comparison_keywords = ['compare', 'difference', 'versus', 'vs', 'better', 'between']
    if any(kw in query_lower for kw in comparison_keywords) and product_count > 1:
        return "mixed"
    
    # Default text-only for questions
    return "text"


# ============================================================================
# FILTER EXTRACTION WITH IMPROVED CONTEXT AWARENESS
# ============================================================================

async def extract_filters_with_context(query: str, history_text: str) -> Optional[List[Dict]]:
    """Extract filters using LLM with conversation context"""
    prompt = f"""Extract filters from this query using conversation history for context. Return ONLY a JSON object.

Previous conversation: {history_text}
Current query: {query}

IMPORTANT: If the current query refers to products mentioned in previous conversation (like "all models", "list them", "show more", "other options"), 
extract the brand/category from the conversation history.

Extract these filters if mentioned or implied from context:
- price_min, price_max (numbers)
- brand (exact match - check conversation history if not explicit in current query)
- color (exact match)
- category (exact match - check conversation history if not explicit in current query)
- min_rating (number)

Examples:
- Previous: "show me lenovo laptops" + Current: "list all models" → {{"brand": "Lenovo", "category": "laptops"}}
- Previous: "laptops under $500" + Current: "show more" → {{"price_max": 500, "category": "laptops"}}
- Previous: "Samsung phones" + Current: "under $200" → {{"brand": "Samsung", "category": "phones", "price_max": 200}}

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
                        "num_predict": 100
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
# IMPROVED RAG PROMPT - FORCES COMPLETE LISTING
# ============================================================================

def create_enhanced_rag_prompt(user_message: str, products: List[Dict[str, Any]], history_text: str, display_mode: str = "text") -> str:
    """Create prompt based on display mode"""
    
    categories = get_categories()
    business_name = get_business_name()
    
    if not products:
        return f"""You are a shopping assistant for {business_name}.

Previous conversation:
{history_text}

Customer question: {user_message}

Respond helpfully."""

    # For products display mode, just return brief intro
    if display_mode == "products":
        return f"""Customer asked: "{user_message}"

Found {len(products)} products. The UI will display product cards with images, prices, and details.

Write a brief 1-2 sentence introduction like: "I found {len(products)} products matching your search. You can browse them below."

Do NOT list the products - just provide a friendly introduction.

Response:"""

    # For text/mixed mode, provide full conversational response
    # Pre-format the COMPLETE product list
    product_lines = []
    for i, product in enumerate(products):
        line = f"{i+1}. **{product.get('name', 'Unknown')}** - {product.get('brand', 'Unknown')} - ${product.get('price', 'N/A')} - Rating: {product.get('rating', 'N/A')}/5"
        product_lines.append(line)
    
    product_list = "\n".join(product_lines)
    
    # Count by brand for verification
    brands = {}
    for p in products:
        brand = p.get('brand', 'Unknown')
        brands[brand] = brands.get(brand, 0) + 1
    
    brand_counts = ", ".join([f"{count} {brand}" for brand, count in sorted(brands.items())])

    return f"""You are a shopping assistant. A customer asked: "{user_message}"

You found {len(products)} products ({brand_counts}).

CRITICAL: You MUST output ALL {len(products)} products below. DO NOT skip, group, or summarize ANY products.

Copy this EXACT list:

{product_list}

After showing ALL {len(products)} products above, add ONE short sentence asking what they're looking for.

Rules:
- Output the numbered list EXACTLY as shown above
- Include every single product from 1 to {len(products)}
- Do NOT reorganize, group by brand, or skip products
- Do NOT add descriptions beyond what's in the list
- After the complete list, ask ONE question

Begin your response with "Here are all {len(products)} products:"
"""


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
    """Enhanced chat endpoint with product card support"""
    logger.info(f"Received message: {request.message}")
    
    session_id = request.session_id or "default"
    
    # Get conversation history first for context
    history_text = get_conversation_history(session_id)
    
    # Guard against off-topic queries with LLM-based classification
    is_shopping_related = await is_on_topic_llm(request.message, history_text)
    
    if not is_shopping_related:
        redirect_response = get_redirect_response(request.message)
        logger.info(f"Off-topic query detected: {request.message}")
        save_conversation(session_id, request.message, redirect_response)
        return ChatResponse(
            response=redirect_response,
            session_id=session_id,
            timestamp=datetime.now().isoformat() + "Z",
            sources=None,
            product_cards=None,
            display_mode="text"
        )
    
    try:
        # Step 1: Conversation history already retrieved above
        logger.info(f"Retrieved conversation history for session: {session_id}")
        
        # Step 2: Extract filters with context
        filters = await extract_filters_with_context(request.message, history_text)
        if filters:
            logger.info(f"Extracted filters: {filters}")
        
        # Step 3: Search for ALL relevant products (no limit)
        logger.info("Searching for ALL relevant products...")
        products = await opensearch_client.search_products(request.message)
        logger.info(f"Found {len(products)} relevant products")
        
        # Step 4: Detect display mode
        display_mode = detect_display_mode(request.message, len(products))
        logger.info(f"Display mode: {display_mode}")
        
        # Step 5: Format product cards for UI if needed
        product_cards = None
        if display_mode in ["products", "mixed"] and products:
            product_cards = format_product_cards(products)
            logger.info(f"Formatted {len(product_cards)} product cards")
        
        # Step 6: Create prompt based on display mode
        rag_prompt = create_enhanced_rag_prompt(request.message, products, history_text, display_mode)
        
        # Step 7: Generate response using Ollama
        logger.info("Generating response with Ollama...")
        async with httpx.AsyncClient() as client:
            ollama_response = await client.post(
                "http://localhost:11434/api/generate",
                json={
                    "model": "llama3.1:8b",
                    "prompt": rag_prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.2,
                        "top_p": 0.85,
                        "num_predict": 100 if display_mode == "products" else 2500,
                        "stop": [],
                        "repeat_penalty": 1.1
                    }
                },
                timeout=90.0
            )
            ollama_response.raise_for_status()
            result = ollama_response.json()
            response_text = result.get("response", "Sorry, I couldn't generate a response.")
        
        # Step 8: Save conversation to memory
        save_conversation(session_id, request.message, response_text)
        logger.info(f"Saved conversation to session: {session_id}")
            
    except Exception as e:
        logger.error(f"RAG API error: {e}")
        response_text = f"Sorry, I'm having trouble processing your request right now."
        products = []
        product_cards = None
        display_mode = "text"
    
    return ChatResponse(
        response=response_text,
        session_id=session_id,
        timestamp=datetime.now().isoformat() + "Z",
        sources=products if products else None,
        product_cards=product_cards,
        display_mode=display_mode
    )


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """Enhanced WebSocket endpoint with product card support"""
    await websocket.accept()
    logger.info("WebSocket connection established")
    
    try:
        while True:
            data = await websocket.receive_text()
            message_data = json.loads(data)
            
            logger.info(f"WebSocket received: {message_data}")
            
            user_message = message_data.get('message', '')
            session_id = message_data.get('session_id', 'default')
            
            # Get conversation history for context
            history_text = get_conversation_history(session_id)
            
            # Guard against off-topic queries with LLM-based classification
            is_shopping_related = await is_on_topic_llm(user_message, history_text)
            
            if not is_shopping_related:
                redirect_response = get_redirect_response(user_message)
                logger.info(f"Off-topic query detected: {user_message}")
                save_conversation(session_id, user_message, redirect_response)
                
                response = {
                    "response": redirect_response,
                    "session_id": session_id,
                    "timestamp": datetime.now().isoformat() + "Z",
                    "sources": None,
                    "product_cards": None,
                    "display_mode": "text"
                }
                
                await websocket.send_text(json.dumps(response))
                continue
            
            try:
                # Steps 1-8: Same as /api/chat endpoint
                logger.info(f"Retrieved conversation history for session: {session_id}")
                
                filters = await extract_filters_with_context(user_message, history_text)
                if filters:
                    logger.info(f"Extracted filters: {filters}")
                
                logger.info("Searching for ALL relevant products...")
                products = await opensearch_client.search_products(user_message)
                logger.info(f"Found {len(products)} relevant products")
                
                display_mode = detect_display_mode(user_message, len(products))
                logger.info(f"Display mode: {display_mode}")
                
                product_cards = None
                if display_mode in ["products", "mixed"] and products:
                    product_cards = format_product_cards(products)
                    logger.info(f"Formatted {len(product_cards)} product cards")
                
                rag_prompt = create_enhanced_rag_prompt(user_message, products, history_text, display_mode)
                
                logger.info("Generating response with Ollama...")
                async with httpx.AsyncClient() as client:
                    ollama_response = await client.post(
                        "http://localhost:11434/api/generate",
                        json={
                            "model": "llama3.1:8b",
                            "prompt": rag_prompt,
                            "stream": False,
                            "options": {
                                "temperature": 0.2,
                                "top_p": 0.85,
                                "num_predict": 100 if display_mode == "products" else 2500,
                                "stop": [],
                                "repeat_penalty": 1.1
                            }
                        },
                        timeout=90.0
                    )
                    ollama_response.raise_for_status()
                    result = ollama_response.json()
                    response_text = result.get("response", "Sorry, I couldn't generate a response.")
                    
                save_conversation(session_id, user_message, response_text)
                logger.info(f"Saved conversation to session: {session_id}")
                    
            except Exception as e:
                logger.error(f"RAG API error: {e}")
                response_text = f"Sorry, I'm having trouble processing your request right now."
                products = []
                product_cards = None
                display_mode = "text"
            
            response = {
                "response": response_text,
                "session_id": session_id,
                "timestamp": datetime.now().isoformat() + "Z",
                "sources": products if products else None,
                "product_cards": [card.dict() for card in product_cards] if product_cards else None,
                "display_mode": display_mode
            }
            
            await websocket.send_text(json.dumps(response))
            
    except WebSocketDisconnect:
        logger.info("WebSocket connection closed")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")


@app.get("/health")
async def health_check():
    return {"status": "healthy", "message": "RAG-powered backend with product cards"}


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
async def search_products(query: str):
    """Direct search endpoint for testing OpenSearch functionality - returns ALL results"""
    try:
        products = await opensearch_client.search_products(query)
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

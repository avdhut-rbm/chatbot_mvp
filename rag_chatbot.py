# rag_chatbot.py - FIXED VERSION
from opensearchpy import OpenSearch
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_core.prompts import PromptTemplate
import json

# OpenSearch connection
client = OpenSearch(
    hosts=[{'host': 'localhost', 'port': 9200}],
    http_auth=('admin', 'admin'),
    use_ssl=False,
    verify_certs=False
)

# Initialize models
embeddings = OllamaEmbeddings(model="nomic-embed-text")
llm = ChatOllama(
    model="llama3.1:8b",
    temperature=0.7
)

# Simple conversation memory
conversation_history = []

INDEX_NAME = "amazon_electronics"


def hybrid_search(query_text, filters=None, top_k=3):
    """Hybrid search: vector similarity + keyword + filters"""
    
    # Generate query embedding
    query_vector = embeddings.embed_documents([query_text])[0]
    
    # Build simpler query - keyword search only for now
    query = {
        "size": top_k,
        "query": {
            "bool": {
                "must": [
                    {
                        "multi_match": {
                            "query": query_text,
                            "fields": ["name^3", "description^2", "brand^2", "tags", "category"],
                            "type": "best_fields"
                        }
                    }
                ],
            }
        },
        "_source": {
            "excludes": ["product_embedding"]
        }
    }
    
    # Add filters if provided
    if filters:
        if "filter" not in query["query"]["bool"]:
            query["query"]["bool"]["filter"] = []
        query["query"]["bool"]["filter"].extend(filters)
    
    # Execute search
    try:
        response = client.search(index=INDEX_NAME, body=query)
        return response['hits']['hits']
    except Exception as e:
        print(f"Search error: {e}")
        # Fallback to match_all if search fails
        response = client.search(
            index=INDEX_NAME, 
            body={"size": top_k, "query": {"match_all": {}}}
        )
        return response['hits']['hits']


def hybrid_searchERROR(query_text, filters=None, top_k=3):
    """Hybrid search: vector similarity + keyword + filters"""
    
    # Generate query embedding
    query_vector = embeddings.embed_documents([query_text])[0]
    
    # Build query
    query = {
        "size": top_k,
        "query": {
            "bool": {
                "should": [
                    # Vector similarity search
                    {
                        "script_score": {
                            "query": {"match_all": {}},
                            "script": {
                                "source": "cosineSimilarity(params.query_vector, 'product_embedding') + 1.0",
                                "params": {"query_vector": query_vector}
                            }
                        }
                    },
                    # Keyword search
                    {
                        "multi_match": {
                            "query": query_text,
                            "fields": ["name^3", "description^2", "brand", "tags"],
                            "type": "best_fields"
                        }
                    }
                ],
                "minimum_should_match": 1
            }
        },
        "_source": {
            "excludes": ["product_embedding"]
        }
    }
    
    # Add filters if provided
    if filters:
        query["query"]["bool"]["filter"] = filters
    
    # Execute search
    response = client.search(index=INDEX_NAME, body=query)
    return response['hits']['hits']

def extract_filters(query, history_text):
    """Extract filters from user query using LLM"""
    
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
    
    result = llm.invoke(prompt)
    result_text = result.content if hasattr(result, 'content') else str(result)
    
    try:
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
            os_filters.append({"term": {"brand.keyword": filters_dict["brand"]}})
        
        if "color" in filters_dict:
            os_filters.append({"term": {"color.keyword": filters_dict["color"]}})
        
        if "category" in filters_dict:
            os_filters.append({"term": {"category.keyword": filters_dict["category"]}})
        
        if "min_rating" in filters_dict:
            os_filters.append({"range": {"rating": {"gte": filters_dict["min_rating"]}}})
        
        return os_filters if os_filters else None
    
    except:
        return None

def format_products(hits):
    """Format retrieved products for LLM context"""
    products = []
    for hit in hits:
        source = hit['_source']
        product_str = f"""
Product: {source.get('name', 'N/A')}
Brand: {source.get('brand', 'N/A')}
Price: ${source.get('price', 'N/A')} {source.get('currency', 'USD')}
Description: {source.get('description', 'N/A')}
Color: {source.get('color', 'N/A')}
Material: {source.get('material', 'N/A')}
Rating: {source.get('rating', 'N/A')}/5 ({source.get('reviews_count', 0)} reviews)
Availability: {source.get('availability', 'N/A')}
"""
        products.append(product_str.strip())
    
    return "\n\n---\n\n".join(products)

def chat(user_query):
    """Main chat function with RAG"""
    
    # Get last 4 messages from history
    history_text = "\n".join([f"{msg['role']}: {msg['content']}" for msg in conversation_history[-4:]])
    
    # Extract filters from query
    filters = extract_filters(user_query, history_text)
    
    # Search for relevant products
    results = hybrid_search(user_query, filters=filters, top_k=3)
    
    if not results:
        response = "I couldn't find any products matching your query. Could you try rephrasing or being more specific?"
        conversation_history.append({"role": "user", "content": user_query})
        conversation_history.append({"role": "assistant", "content": response})
        return response, []
    
    # Format products for context
    context = format_products(results)
    
    # Generate response
    prompt = f"""You are a helpful electronics shopping assistant. Answer based ONLY on the provided product information.

Previous conversation:
{history_text}

Available Products:
{context}

User Question: {user_query}

Instructions:
- Recommend products from the context above
- Compare options if multiple products match
- Mention key specs: brand, price, rating, availability
- If asking for clarification, be specific
- If no products match exactly, suggest closest alternatives from the context
- DO NOT make up products or specifications not in the context

Answer:"""
    
    result = llm.invoke(prompt)
    response = result.content if hasattr(result, 'content') else str(result)
    
    # Save to conversation history
    conversation_history.append({"role": "user", "content": user_query})
    conversation_history.append({"role": "assistant", "content": response})
    
    return response, results

def generate_related_questions(query, response, products):
    """Generate follow-up questions"""
    if not products:
        return []
    
    prompt = f"""Based on this conversation, suggest 3 natural follow-up questions.

User asked: {query}
Response: {response[:200]}...

Generate 3 specific questions the user might ask next about these products.

Questions:
1."""
    
    result = llm.invoke(prompt)
    content = result.content if hasattr(result, 'content') else str(result)
    
    # Parse questions
    lines = content.split('\n')
    questions = [line.strip('123. ').strip() for line in lines if line.strip() and len(line) > 0 and line[0].isdigit()]
    return questions[:3]

def start_chat():
    """Start interactive chatbot"""
    print("🤖 Electronics Shopping Assistant (type 'quit' to exit)\n")
    
    while True:
        try:
            user_input = input("\n👤 You: ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'bye']:
                print("👋 Thanks for shopping! Goodbye!")
                break
            
            if not user_input:
                continue
            
            # Get response
            print("\n🔍 Searching...")
            response, products = chat(user_input)
            print(f"\n🤖 Assistant: {response}")
            
            # Show related questions
            if products:
                print("\n⏳ Generating suggestions...")
                related = generate_related_questions(user_input, response, products)
                if related:
                    print("\n💡 You might also ask:")
                    for i, q in enumerate(related, 1):
                        print(f"   {i}. {q}")
                        
        except Exception as e:
            print(f"\n❌ Error: {e}")
            import traceback
            traceback.print_exc()


def start_chatERROR():
    """Start interactive chatbot"""
    print("🤖 Electronics Shopping Assistant (type 'quit' to exit)\n")
    
    while True:
        user_input = input("\n👤 You: ").strip()
        
        if user_input.lower() in ['quit', 'exit', 'bye']:
            print("👋 Thanks for shopping! Goodbye!")
            break

# Add at the very bottom of the file
start_chat()

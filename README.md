# Chatbot MVP - Intelligent E-commerce Assistant

A sophisticated RAG-powered chatbot built with FastAPI, Next.js, Ollama, and OpenSearch for intelligent product search and recommendations.

## 🚀 Features

### Core Functionality
- **RAG-Powered Search**: Retrieval-Augmented Generation using OpenSearch and Ollama
- **Conversation Memory**: Context-aware conversations with session-based memory
- **Intelligent Filtering**: LLM-powered filter extraction from natural language
- **Real-time Chat**: WebSocket support for instant responses
- **Product Recommendations**: Smart product suggestions based on user preferences

### Technical Stack
- **Backend**: FastAPI with async/await support
- **Frontend**: Next.js 14 with TypeScript and Tailwind CSS
- **LLM**: Ollama with Llama 3.1 8B model
- **Search Engine**: OpenSearch with hybrid search (semantic + keyword)
- **Embeddings**: Ollama-generated embeddings for semantic search
- **Memory**: Session-based conversation history

## 📁 Project Structure

```
chatbot_mvp/
├── backend/                 # FastAPI backend
│   ├── main.py             # Main application with RAG and conversation memory
│   ├── requirements.txt    # Python dependencies
│   └── import_products.py  # Data import utilities
├── frontend/               # Next.js frontend
│   ├── src/
│   │   └── app/
│   │       ├── page.tsx    # Main chat interface
│   │       └── layout.tsx  # App layout
│   ├── package.json        # Node.js dependencies
│   └── tailwind.config.js  # Tailwind CSS configuration
├── rag_chatbot.py          # Standalone RAG chatbot implementation
├── recreate_index.py       # OpenSearch index management
└── README.md              # This file
```

## 🛠️ Installation & Setup

### Prerequisites
- Python 3.8+
- Node.js 18+
- Docker (for OpenSearch)
- Ollama installed locally

### 1. Install Ollama
```bash
brew install ollama
ollama pull llama3.1:8b
```

### 2. Start OpenSearch
```bash
docker run -d --name opensearch \
  -p 9200:9200 -p 9600:9600 \
  -e "discovery.type=single-node" \
  -e "OPENSEARCH_INITIAL_ADMIN_PASSWORD=admin" \
  opensearchproject/opensearch:2.11.0
```

### 3. Backend Setup
```bash
cd backend
pip install -r requirements.txt
python main.py
```

### 4. Frontend Setup
```bash
cd frontend
npm install
npm run dev
```

### 5. Data Import
```bash
# Import sample products into OpenSearch
python backend/import_products.py
```

## 🎯 Usage

### API Endpoints

#### Chat with RAG
```bash
curl -X POST "http://localhost:8000/api/chat" \
  -H "Content-Type: application/json" \
  -d '{"message": "I want to buy a phone", "session_id": "user123"}'
```

#### Direct Search
```bash
curl -X GET "http://localhost:8000/api/search?query=Samsung&size=5"
```

#### Conversation History
```bash
curl -X GET "http://localhost:8000/api/conversation/user123"
```

### WebSocket Chat
```javascript
const ws = new WebSocket('ws://localhost:8000/ws');
ws.send(JSON.stringify({
  message: "Show me laptops",
  session_id: "user123"
}));
```

## 🧠 Key Features Explained

### 1. RAG (Retrieval-Augmented Generation)
- **Semantic Search**: Uses Ollama embeddings for understanding user intent
- **Keyword Search**: BM25-based text matching for precise results
- **Hybrid Approach**: Combines both methods for optimal results
- **Context Injection**: Product information is injected into LLM prompts

### 2. Conversation Memory
- **Session Management**: Each user gets a unique session ID
- **Context Window**: Maintains last 6 messages for context
- **Memory Persistence**: Conversations persist during the session
- **Smart Filtering**: Remembers user preferences across messages

### 3. Intelligent Filtering
- **LLM-Powered**: Uses Ollama to extract filters from natural language
- **Context-Aware**: Considers conversation history when extracting filters
- **Supported Filters**: Price ranges, brand, color, category, ratings
- **Dynamic Queries**: Builds OpenSearch queries based on extracted filters

## 🔧 Configuration

### Environment Variables
```bash
# Frontend
NEXT_PUBLIC_API_URL=http://localhost:8000

# Backend
OPENSEARCH_HOST=localhost
OPENSEARCH_PORT=9200
OLLAMA_HOST=localhost:11434
```

## 🚀 Advanced Features

### 1. Context-Aware Responses
The chatbot remembers previous conversations and can answer follow-up questions:
```
User: "I want to buy a phone"
Assistant: "Here are some phone options..."

User: "What about Samsung?"
Assistant: "Here are Samsung phones..." // Remembers phone context

User: "Show me the red one"
Assistant: "Here's the red Samsung phone..." // Remembers Samsung context
```

### 2. Smart Filter Extraction
```python
# Input: "I want a phone under $500"
# Extracted: {"price_max": 500, "category": "phones"}

# Input: "Show me red Samsung phones"
# Extracted: {"brand": "Samsung", "color": "red", "category": "phones"}
```

## 📊 Performance

- **Response Time**: < 2 seconds for most queries
- **Memory Usage**: ~2GB RAM for Ollama + OpenSearch
- **Concurrent Users**: Supports multiple sessions simultaneously
- **Search Accuracy**: High precision with hybrid search approach

## 🔍 Troubleshooting

### Common Issues

1. **Ollama Connection Error**
   ```bash
   # Check if Ollama is running
   curl http://localhost:11434/api/tags
   ```

2. **OpenSearch Connection Error**
   ```bash
   # Check OpenSearch status
   curl http://localhost:9200/_cluster/health
   ```

3. **Memory Issues**
   ```bash
   # Restart Ollama with more memory
   ollama serve
   ```

### Debug Endpoints
- `GET /health` - Backend health check
- `GET /api/conversation/{session_id}` - View conversation history
- `GET /api/search?query=test` - Test search functionality

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- **Ollama** for local LLM inference
- **OpenSearch** for powerful search capabilities
- **FastAPI** for high-performance API framework
- **Next.js** for modern React development
- **Tailwind CSS** for utility-first styling

---

Built with ❤️ for intelligent e-commerce experiences

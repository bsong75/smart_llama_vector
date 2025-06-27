## command to build/ pull models
docker-compose up -d --build --force-recreate
docker exec -it llama-container ollama run llama3.2:3b

-------------

# Smart Kusco Chat App

## Overview
Smart Kusco is an intelligent chatbot application that combines conversational AI with semantic memory. It uses a local Llama model for chat responses and implements continual learning through vector-based conversation storage and retrieval.

## Core Architecture

### Technology Stack
- **Frontend**: Gradio web interface
- **AI Model**: Llama 3.2:3b (via Ollama API)
- **Vector Database**: ChromaDB with persistent storage
- **Embeddings**: Generated using the same Llama model
- **Data Persistence**: JSON files for sessions, ChromaDB for semantic search

### Key Features
- **Session Management**: Persistent chat sessions with titles and timestamps
- **Semantic Memory**: Vector-based storage of all conversations for context retrieval
- **Continual Learning**: Each conversation improves future responses through context injection
- **Sidebar Navigation**: Easy access to previous conversations
- **Real-time Context**: Automatically finds and uses relevant past conversations

## Technical Implementation Details

### 1. Session Storage Mechanism

**How it works in plain English:**
The app treats each conversation like a notebook that gets saved to your computer. Every time you start chatting, it creates a new "notebook" with a unique ID based on the current date and time. All your messages and the AI's responses get written into this notebook, along with a title (based on your first message) and when you started the conversation.

**Technical Implementation:**
```python
# Sessions are stored as a dictionary in memory
chat_sessions = {
    "chat_20241226_143052": {
        "history": [["user message", "ai response"], ...],
        "title": "First 30 characters of first message...",
        "timestamp": datetime.now()
    }
}

# Saved to disk as JSON
def save_chat_sessions():
    with open(CHAT_SESSIONS_FILE, 'w') as f:
        json.dump(serializable_sessions, f, indent=2)
```

**Key Functions:**
- `generate_session_id()`: Creates unique IDs like "chat_20241226_143052"
- `get_session_title()`: Uses first message as session title
- `save_chat_sessions()`: Persists all sessions to `/app/data/chat_sessions.json`
- `load_chat_sessions()`: Restores sessions when app starts

### 2. Conversation Addition to Vector Database

**How it works in plain English:**
Think of this like creating a smart filing system for every conversation. After each exchange, the app takes your question and the AI's answer, combines them into one text block, and then creates a mathematical "fingerprint" (embedding) that represents the meaning of that conversation. This fingerprint gets stored in a special database that can find similar conversations later based on meaning, not just keywords.

**Technical Implementation:**
```python
def add_conversation_to_vector_db(user_msg, ai_response, session_id):
    # Combine user question and AI response
    full_conversation = f"User: {user_msg}\nAssistant: {ai_response}"
    
    # Generate mathematical representation using Ollama
    embedding = get_ollama_embedding(full_conversation)
    
    # Store with metadata
    vector_collection.add(
        embeddings=[embedding],           # The mathematical fingerprint
        documents=[full_conversation],    # The actual text
        metadatas=[{                     # Additional information
            "user_question": user_msg,
            "ai_response": ai_response,
            "session_id": session_id,
            "timestamp": datetime.now().isoformat()
        }],
        ids=[str(uuid.uuid4())]          # Unique identifier
    )
```

**Process Flow:**
1. User sends message → AI responds
2. Combine both into single text block
3. Send to Ollama API to generate embedding vector
4. Store embedding + original text + metadata in ChromaDB
5. Conversation is now searchable by semantic similarity

### 3. Relevant Conversation Discovery

**How it works in plain English:**
When you ask a new question, the app creates a mathematical fingerprint of your question and compares it to all the fingerprints of previous conversations. It finds the conversations that are most similar in meaning (not just words) and pulls up the top 3 most relevant ones. These get shown to the AI as context, so it can give you a better answer based on what you've discussed before.

**Technical Implementation:**
```python
def find_relevant_conversations(query, max_results=3, similarity_threshold=0.7):
    # Create embedding for the new question
    query_embedding = get_ollama_embedding(query)
    
    # Search vector database for similar conversations
    results = vector_collection.query(
        query_embeddings=[query_embedding],
        n_results=max_results * 2,      # Get extras to filter
        include=['documents', 'metadatas', 'distances']
    )
    
    # Filter by similarity threshold and return best matches
    relevant_conversations = []
    for doc, metadata, distance in zip(results['documents'][0], 
                                      results['metadatas'][0], 
                                      results['distances'][0]):
        similarity = 1 / (1 + distance)  # Convert distance to similarity
        if similarity >= similarity_threshold:
            relevant_conversations.append({
                'user_question': metadata['user_question'],
                'ai_response': metadata['ai_response'],
                'similarity': similarity
            })
    
    return sorted(relevant_conversations, key=lambda x: x['similarity'], reverse=True)
```

**Context Injection Process:**
1. New question comes in
2. Generate embedding for question
3. ChromaDB finds similar conversation embeddings
4. Filter results by similarity threshold (70%+)
5. Format top 3 results as context for the AI
6. AI receives: "Based on similar conversations: [context] ... Now answer: [new question]"

## Data Flow Summary

```
User Question → Generate Embedding → Search Vector DB → Find Similar Conversations
     ↓                                                            ↓
Store New Conversation ← AI Response ← Context-Enhanced Prompt ← Format Context
     ↓
Update Session & Save to Disk
```

## Benefits of This Architecture

1. **Semantic Understanding**: Finds conversations based on meaning, not keywords
2. **Continual Learning**: Each conversation improves future responses
3. **Session Persistence**: Conversations survive app restarts
4. **Context Awareness**: AI has memory of relevant past discussions
5. **Scalable**: Vector database efficiently handles thousands of conversations

## File Structure
```
/app/data/
├── chat_sessions.json          # Session metadata and history
└── vector_db/                  # ChromaDB persistent storage
    ├── chroma.sqlite3          # Vector embeddings database
    └── [other ChromaDB files]
```

This architecture creates an intelligent chatbot that learns from every conversation and provides increasingly relevant and contextual responses over time.
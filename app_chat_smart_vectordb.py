import gradio as gr
import requests
import json
import os
from datetime import datetime
import uuid
import chromadb
import numpy as np

LLAMA_URL = "http://llama-container_vector:11434/v1/chat/completions"
OLLAMA_EMBEDDINGS_URL = "http://llama-container_vector:11434/api/embeddings"
DATA_DIR = "/app/data"
CHAT_SESSIONS_FILE = os.path.join(DATA_DIR, "chat_sessions.json")
VECTOR_DB_PATH = os.path.join(DATA_DIR, "vector_db")

# Global storage
chat_sessions = {}
current_session_id = None
chroma_client = None
vector_collection = None
embedding_model_name = "llama3.2:3b"  # Use your existing Llama model for embeddings

def ensure_data_dir():
    """Ensure the data directory exists"""
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(VECTOR_DB_PATH, exist_ok=True)

def get_ollama_embedding(text):
    """Get embeddings from Ollama API"""
    try:
        payload = {
            "model": embedding_model_name,
            "prompt": text
        }
        
        response = requests.post(OLLAMA_EMBEDDINGS_URL, 
                               json=payload, 
                               timeout=30)
        
        if response.status_code == 200:
            result = response.json()
            embedding = result.get("embedding", [])
            if embedding:
                return embedding
            else:
                print(f"⚠️ No embedding returned for text: {text[:50]}...")
                return None
        else:
            print(f"❌ Ollama embeddings API error: {response.status_code} - {response.text}")
            return None
            
    except requests.exceptions.Timeout:
        print("⏱️ Timeout getting embedding from Ollama")
        return None
    except Exception as e:
        print(f"❌ Error getting embedding from Ollama: {e}")
        return None

def test_ollama_embeddings():
    """Test if Ollama embeddings API is working"""
    test_embedding = get_ollama_embedding("Hello, this is a test.")
    if test_embedding:
        print(f"✅ Ollama embeddings working! Dimension: {len(test_embedding)}")
        return True
    else:
        print("❌ Ollama embeddings not available")
        return False

def initialize_vector_db():
    """Initialize ChromaDB and test embedding connectivity"""
    global chroma_client, vector_collection
    
    try:
        print("🔄 Initializing vector database with Ollama embeddings...")
        
        # Test Ollama embeddings first
        if not test_ollama_embeddings():
            raise Exception("Ollama embeddings API not available")
        
        # Initialize ChromaDB
        chroma_client = chromadb.PersistentClient(path=VECTOR_DB_PATH)
        
        # Get or create collection
        try:
            vector_collection = chroma_client.get_collection("chat_conversations_ollama")
            existing_count = vector_collection.count()
            print(f"✅ Connected to existing vector collection with {existing_count} conversations")
        except:
            vector_collection = chroma_client.create_collection(
                name="chat_conversations_ollama",
                metadata={"description": "Semantic search using Ollama embeddings"}
            )
            print("✅ Created new vector collection with Ollama embeddings")
            
    except Exception as e:
        print(f"❌ Error initializing vector database: {e}")
        print("📝 Make sure Ollama is running and supports embeddings API")
        print("🔧 Try: ollama pull llama3.2:3b (if not already done)")

def add_conversation_to_vector_db(user_msg, ai_response, session_id):
    """Add a conversation to the vector database using Ollama embeddings"""
    if not vector_collection:
        return
    
    try:
        # Create a unique ID for this conversation
        conversation_id = str(uuid.uuid4())
        
        # Combine user message and AI response for better context
        full_conversation = f"User: {user_msg}\nAssistant: {ai_response}"
        
        # Generate embedding using Ollama
        embedding = get_ollama_embedding(full_conversation)
        if not embedding:
            print(f"⚠️ Failed to get embedding, skipping conversation storage")
            return
        
        # Prepare metadata
        metadata = {
            "user_question": user_msg,
            "ai_response": ai_response,
            "session_id": session_id,
            "timestamp": datetime.now().isoformat(),
            "conversation_length": len(full_conversation),
            "embedding_model": embedding_model_name
        }
        
        # Add to collection
        vector_collection.add(
            embeddings=[embedding],
            documents=[full_conversation],
            metadatas=[metadata],
            ids=[conversation_id]
        )
        
        print(f"📝 Added conversation to vector DB: {user_msg[:50]}...")
        
    except Exception as e:
        print(f"❌ Error adding to vector DB: {e}")

def find_relevant_conversations(query, max_results=3, similarity_threshold=0.7):
    """Find semantically similar conversations using Ollama embeddings"""
    if not vector_collection:
        return []
    
    try:
        # Generate embedding for the query using Ollama
        query_embedding = get_ollama_embedding(query)
        if not query_embedding:
            print(f"⚠️ Failed to get embedding for query, no context will be provided")
            return []
        
        # Search for similar conversations
        results = vector_collection.query(
            query_embeddings=[query_embedding],
            n_results=max_results * 2,  # Get more results to filter
            include=['documents', 'metadatas', 'distances']
        )
        
        relevant_conversations = []
        
        if results['documents'] and len(results['documents'][0]) > 0:
            for i, (doc, metadata, distance) in enumerate(zip(
                results['documents'][0], 
                results['metadatas'][0], 
                results['distances'][0]
            )):
                # Convert distance to similarity (ChromaDB uses squared euclidean distance)
                similarity = 1 / (1 + distance)
                
                if similarity >= similarity_threshold:
                    relevant_conversations.append({
                        'user_question': metadata['user_question'],
                        'ai_response': metadata['ai_response'],
                        'similarity': similarity,
                        'timestamp': metadata['timestamp']
                    })
        
        # Sort by similarity and return top results
        relevant_conversations.sort(key=lambda x: x['similarity'], reverse=True)
        return relevant_conversations[:max_results]
        
    except Exception as e:
        print(f"❌ Error searching vector DB: {e}")
        return []

def get_vector_db_stats():
    """Get statistics about the vector database"""
    if not vector_collection:
        return "📊 Vector DB: Not initialized"
    
    try:
        count = vector_collection.count()
        return f"🧠 Vector DB: {count} conversations stored"
    except:
        return "📊 Vector DB: Error getting stats"

def load_chat_sessions():
    """Load chat sessions from disk"""
    global chat_sessions
    ensure_data_dir()
    
    if os.path.exists(CHAT_SESSIONS_FILE):
        try:
            with open(CHAT_SESSIONS_FILE, 'r', encoding='utf-8') as f:
                data = json.load(f)
                for session_id, session_data in data.items():
                    if 'timestamp' in session_data:
                        session_data['timestamp'] = datetime.fromisoformat(session_data['timestamp'])
                chat_sessions = data
                print(f"📂 Loaded {len(chat_sessions)} chat sessions from disk")
        except Exception as e:
            print(f"❌ Error loading chat sessions: {e}")
            chat_sessions = {}
    else:
        chat_sessions = {}

def save_chat_sessions():
    """Save chat sessions to disk"""
    ensure_data_dir()
    
    try:
        serializable_sessions = {}
        for session_id, session_data in chat_sessions.items():
            serializable_data = session_data.copy()
            if 'timestamp' in serializable_data:
                serializable_data['timestamp'] = serializable_data['timestamp'].isoformat()
            serializable_sessions[session_id] = serializable_data
        
        with open(CHAT_SESSIONS_FILE, 'w', encoding='utf-8') as f:
            json.dump(serializable_sessions, f, indent=2, ensure_ascii=False)
        print(f"💾 Saved {len(chat_sessions)} chat sessions to disk")
    except Exception as e:
        print(f"❌ Error saving chat sessions: {e}")

def generate_session_id():
    return f"chat_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

def get_session_title(history):
    """Generate a title for the chat session based on first message"""
    if history and len(history) > 0:
        first_message = history[0][0]
        return first_message[:30] + "..." if len(first_message) > 30 else first_message
    return "New Chat"

def chat_with_llama(message, history, session_id):
    global chat_sessions, current_session_id
    
    # Find relevant context using vector search
    relevant_conversations = find_relevant_conversations(message)
    
    # Build messages array
    messages = []
    
    # Add context from vector database if available
    if relevant_conversations:
        context_prompt = "Based on semantically similar conversations from our chat history:\n"
        for i, conv in enumerate(relevant_conversations, 1):
            similarity_pct = int(conv['similarity'] * 100)
            context_prompt += f"\n{i}. [{similarity_pct}% similar] Q: {conv['user_question']}\n   A: {conv['ai_response'][:200]}...\n"
        context_prompt += "\nConsidering this relevant context, please answer the current question:\n"
        messages.append({"role": "system", "content": context_prompt})
    
    # Add current conversation history
    for human_msg, ai_msg in history:
        messages.append({"role": "user", "content": human_msg})
        messages.append({"role": "assistant", "content": ai_msg})
    
    # Add current message
    messages.append({"role": "user", "content": message})
    
    payload = {"messages": messages, 
               "max_tokens": 512,
               "temperature": 0.1, 
               "model": "llama3.2:3b"}
    
    try:
        response = requests.post(LLAMA_URL, json=payload)
        result = response.json()
        bot_response = result.get("choices", [{}])[0].get("message", {}).get("content", "No response")
    except Exception as e:
        bot_response = f"Error connecting to Llama: {str(e)}"
    
    # Add this conversation to vector database
    add_conversation_to_vector_db(message, bot_response, session_id)
    
    history.append([message, bot_response])
    
    # Update session storage
    if session_id:
        chat_sessions[session_id] = {
            "history": history.copy(),
            "title": get_session_title(history),
            "timestamp": datetime.now()
        }
        save_chat_sessions()
    
    # Update chat list
    chat_list = update_chat_list()
    current_selection = get_session_title(history) if session_id else None
    
    return history, "", gr.update(choices=chat_list, value=current_selection)

def new_chat():
    """Start a new chat session"""
    global current_session_id
    current_session_id = generate_session_id()
    chat_list = update_chat_list()
    return [], current_session_id, gr.update(choices=chat_list, value=None)

def update_chat_list():
    """Return list of chat sessions for the dropdown"""
    if not chat_sessions:
        return []
    
    sorted_sessions = sorted(
        chat_sessions.items(), 
        key=lambda x: x[1]["timestamp"], 
        reverse=True
    )
    
    return [data['title'] for session_id, data in sorted_sessions]

def load_chat_session(selected_chat, current_session):
    """Load a selected chat session"""
    global current_session_id
    
    if not selected_chat or selected_chat == "":
        return [], current_session
    
    try:
        for session_id, data in chat_sessions.items():
            if data['title'] == selected_chat:
                current_session_id = session_id
                history = data["history"]
                return history, session_id
    except Exception as e:
        print(f"❌ Error loading chat session: {e}")
    
    return [], current_session

def delete_chat_session(selected_chat):
    """Delete a selected chat session"""
    global chat_sessions
    
    if not selected_chat or selected_chat == "":
        return gr.update(choices=update_chat_list(), value=None), [], None
    
    try:
        session_to_delete = None
        for session_id, data in chat_sessions.items():
            if data['title'] == selected_chat:
                session_to_delete = session_id
                break
        
        if session_to_delete:
            del chat_sessions[session_to_delete]
            save_chat_sessions()
            print(f"🗑️ Deleted chat session: {selected_chat}")
    except Exception as e:
        print(f"❌ Error deleting chat session: {e}")
    
    chat_list = update_chat_list()
    return gr.update(choices=chat_list, value=None), [], None

def toggle_sidebar():
    """Toggle the visibility of the sidebar"""
    return gr.update(visible=True), gr.update(visible=False)

def hide_sidebar():
    """Hide the sidebar"""
    return gr.update(visible=False), gr.update(visible=True)

# Custom CSS
custom_css = """
footer {display: none !important}
.sidebar {
    border-right: 1px solid #e5e5e5;
    height: 100vh;
    padding: 1rem;
    min-width: 300px;
    max-width: 300px;
}
.chat-list {
    max-height: 400px;
    overflow-y: auto;
}
.toggle-btn {
    margin: 0 auto 15px auto;
    width: 30px;
    height: 30px;
    min-width: 30px !important;
    padding: 0 !important;
    display: block;
}
.show-sidebar-btn {
    position: fixed;
    top: 20px;
    left: 20px;
    z-index: 1000;
    width: 40px;
    height: 40px;
    min-width: 40px !important;
    padding: 0 !important;
    border-radius: 50%;
    box-shadow: 0 2px 8px rgba(0,0,0,0.15);
}
.main-content {
    flex: 1;
    padding: 0 1rem;
}
.chat-container {
    max-width: 1200px;
    margin: 0 auto;
}
.delete-btn {
    background-color: #ff4444 !important;
    border-color: #ff4444 !important;
}
.delete-btn:hover {
    background-color: #cc0000 !important;
    border-color: #cc0000 !important;
}
.vector-stats {
    font-size: 0.8em;
    color: #333 !important;
    margin-top: 10px;
    padding: 8px;
    background-color: #e8f4f8 !important;
    border: 1px solid #b3d9e6;
    border-radius: 5px;
    font-weight: 500;
}

/* Dark mode support */
@media (prefers-color-scheme: dark) {
    .vector-stats {
        color: #ffffff !important;
        background-color: #2d3748 !important;
        border-color: #4a5568;
    }
}
"""

with gr.Blocks(title="Smart Kusco", theme="soft", css=custom_css) as demo:
    # State variables
    session_state = gr.State(value=None)
    
    # Show sidebar button (when sidebar is hidden)
    show_btn = gr.Button("☰", size="sm", elem_classes="show-sidebar-btn", visible=False)
    
    with gr.Row():
        # Left sidebar
        with gr.Column(scale=1, elem_classes="sidebar") as sidebar:
            hide_btn = gr.Button("←", size="sm", elem_classes="toggle-btn")
            
            new_chat_btn = gr.Button("🗨️ New Chat", variant="primary", size="lg")
            
            chat_dropdown = gr.Dropdown(
                choices=[],
                interactive=True,
                elem_classes="chat-list",
                show_label=False
            )
            
            delete_chat_btn = gr.Button("🗑️ Delete Chat", 
                                      variant="secondary", 
                                      size="sm",
                                      elem_classes="delete-btn")
            
            # Vector database stats
            vector_stats = gr.Markdown("🧠 Vector DB: Initializing...", 
                                     elem_classes="vector-stats")
        
        # Main chat area
        with gr.Column(scale=3, elem_classes="main-content") as main_area:
            with gr.Column(elem_classes="chat-container"):
                gr.Markdown("# Smart Kusco with continual learning!")
                
                chatbot = gr.Chatbot(
                    label="Chat History",
                    height=500,
                    show_copy_button=True
                )
                
                msg = gr.Textbox(
                    label="How can I assist you today?", 
                    placeholder="Type your question here...",
                    lines=2
                )
                
                with gr.Row():
                    submit_btn = gr.Button("Send", variant="primary")
                    clear_btn = gr.Button("Clear", variant="secondary")
    
    # Event handlers
    def submit_message(message, history, session_id):
        if not message.strip():
            return history, "", gr.update(), get_vector_db_stats()
        result = chat_with_llama(message, history, session_id)
        return result + (get_vector_db_stats(),)
    
    # Message submission
    msg.submit(
        submit_message,
        inputs=[msg, chatbot, session_state],
        outputs=[chatbot, msg, chat_dropdown, vector_stats]
    )
    
    submit_btn.click(
        submit_message,
        inputs=[msg, chatbot, session_state],
        outputs=[chatbot, msg, chat_dropdown, vector_stats]
    )
    
    # Other event handlers
    new_chat_btn.click(
        new_chat,
        outputs=[chatbot, session_state, chat_dropdown]
    )
    
    chat_dropdown.change(
        load_chat_session,
        inputs=[chat_dropdown, session_state],
        outputs=[chatbot, session_state]
    )
    
    delete_chat_btn.click(
        delete_chat_session,
        inputs=[chat_dropdown],
        outputs=[chat_dropdown, chatbot, session_state]
    )
    
    clear_btn.click(
        lambda: ([], ""),
        outputs=[chatbot, msg]
    )
    
    hide_btn.click(
        hide_sidebar,
        outputs=[sidebar, show_btn]
    )
    
    show_btn.click(
        toggle_sidebar,
        outputs=[sidebar, show_btn]
    )
    
    # Initialize app with vector database
    def initialize_app():
        load_chat_sessions()
        initialize_vector_db()
        chat_list = update_chat_list()
        stats = get_vector_db_stats()
        return [], None, gr.update(choices=chat_list, value=None), stats
    
    demo.load(
        initialize_app,
        outputs=[chatbot, session_state, chat_dropdown, vector_stats]
    )

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0", 
        server_port=7860,
        show_api=False
    )
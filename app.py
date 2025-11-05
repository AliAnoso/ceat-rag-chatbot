import streamlit as st
import os
from langchain_community.document_loaders import DirectoryLoader, PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Qdrant
from langchain.prompts import ChatPromptTemplate
from langchain.schema import StrOutputParser
from operator import itemgetter
from langchain.retrievers.multi_query import MultiQueryRetriever
import time

# Page configuration
st.set_page_config(
    page_title="CEAT-OCS Assistant",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #1e3a8a 0%, #4338ca 100%);
        padding: 2rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        color: white;
    }
    .chat-message {
        padding: 1.5rem;
        border-radius: 10px;
        margin-bottom: 1rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .user-message {
        background-color: #3b82f6;
        color: white;
        margin-left: 20%;
    }
    .assistant-message {
        background-color: #f3f4f6;
        color: #1f2937;
        margin-right: 20%;
    }
    .source-badge {
        display: inline-block;
        background-color: #dbeafe;
        color: #1e40af;
        padding: 0.25rem 0.75rem;
        border-radius: 12px;
        font-size: 0.75rem;
        margin: 0.25rem;
    }
    .stButton>button {
        width: 100%;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'vectorstore' not in st.session_state:
    st.session_state.vectorstore = None
if 'retriever' not in st.session_state:
    st.session_state.retriever = None

# Initialize components
@st.cache_resource
def initialize_rag_system():
    """Initialize the RAG system with document loading and vectorstore creation"""
    try:
        # Load documents
        loader = DirectoryLoader("data/", glob="*.pdf", loader_cls=PyMuPDFLoader)
        docs = loader.load()
        
        # Split documents
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,
            chunk_overlap=100
        )
        split_docs = text_splitter.split_documents(docs)
        
        # Create embeddings
        embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
        
        # Create vectorstore
        vectorstore = Qdrant.from_documents(
            documents=split_docs,
            embedding=embeddings,
            location=":memory:",
            collection_name="ceat_docs"
        )
        
        # Create retrievers
        naive_retriever = vectorstore.as_retriever(search_kwargs={"k": 10})
        
        # Initialize LLM for multi-query
        chat_model = ChatOpenAI(model="gpt-4o-mini", temperature=0)
        
        # Create multi-query retriever
        multi_query_retriever = MultiQueryRetriever.from_llm(
            retriever=naive_retriever,
            llm=chat_model
        )
        
        return vectorstore, multi_query_retriever, chat_model
        
    except Exception as e:
        st.error(f"Error initializing RAG system: {str(e)}")
        return None, None, None

# Header
st.markdown("""
<div class="main-header">
    <h1>🎓 CEAT Office of College Secretary Assistant</h1>
    <p>College of Engineering and Agro-Industrial Technology - UPLB</p>
</div>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.image("https://via.placeholder.com/200x100.png?text=CEAT+UPLB", use_column_width=True)
    
    st.markdown("### 📞 Contact Information")
    st.info("""
    **Phone:** 0998-556-6874  
    **Email:** ceat_ocs.uplb@up.edu.ph  
    **Office Hours:** Mon-Fri, 8:00 AM - 5:00 PM  
    **Location:** Dr. Dante B. De Padua Hall, Pili Drive
    """)
    
    st.markdown("### ⚙️ Configuration")
    
    # API Key input
    api_key = st.text_input("OpenAI API Key", type="password", 
                           help="Enter your OpenAI API key to use the chatbot")
    
    if api_key:
        os.environ["OPENAI_API_KEY"] = api_key
        st.success("✅ API Key configured")
    
    # Temperature slider
    temperature = st.slider("Response Creativity", 0.0, 1.0, 0.0, 0.1,
                           help="0 = More focused, 1 = More creative")
    
    # Initialize system button
    if st.button("🔄 Initialize RAG System"):
        with st.spinner("Loading documents and creating vector database..."):
            vectorstore, retriever, llm = initialize_rag_system()
            if vectorstore and retriever:
                st.session_state.vectorstore = vectorstore
                st.session_state.retriever = retriever
                st.session_state.llm = llm
                st.success("✅ RAG system initialized successfully!")
    
    st.markdown("---")
    
    # Sample questions
    st.markdown("### 💡 Sample Questions")
    sample_questions = [
        "How do I apply for graduation?",
        "What are the shifting requirements?",
        "Who is the College Secretary?",
        "What scholarships are available?",
        "How do I apply for LOA?",
        "What is the process for dropping courses?",
        "How can I get a waiver of prerequisite?"
    ]
    
    for question in sample_questions:
        if st.button(question, key=f"sample_{question}"):
            st.session_state.messages.append({"role": "user", "content": question})
            st.rerun()
    
    st.markdown("---")
    
    if st.button("🗑️ Clear Chat History"):
        st.session_state.messages = []
        st.rerun()

# Main chat interface
if not api_key:
    st.warning("⚠️ Please enter your OpenAI API key in the sidebar to start chatting.")
    st.stop()

if not st.session_state.vectorstore:
    st.info("👆 Click 'Initialize RAG System' in the sidebar to load the knowledge base.")
    st.stop()

# Create RAG chain
@st.cache_resource
def create_rag_chain(_retriever, _llm):
    """Create the RAG chain for question answering"""
    
    TECHNICAL_RAG_PROMPT = """You are a helpful technical assistant for Students in the College of Engineering and Agro-Industrial Technology (CEAT) at UPLB.

INSTRUCTIONS:
1. Answer the question using ONLY the information provided in the Context below
2. If the Context does not contain enough information to answer the question, respond with: "I don't have enough information in the knowledge base to answer that question. Please visit the CEAT-OCS website at https://ceatocs.uplb.edu.ph or contact the office at ceat_ocs.uplb@up.edu.ph or 0998-556-6874 for assistance."
3. Be precise and helpful - provide clear, actionable information
4. If referencing specific procedures, requirements, or guidelines, be detailed and accurate
5. Keep responses well-organized and easy to understand
6. If the question involves forms or documents, provide the complete process

CONTEXT:
{context}

QUESTION:
{question}

ANSWER:"""
    
    technical_rag_prompt = ChatPromptTemplate.from_template(TECHNICAL_RAG_PROMPT)
    
    technical_chain = (
        {"context": itemgetter("question") | _retriever, 
         "question": itemgetter("question")}
        | technical_rag_prompt 
        | _llm
        | StrOutputParser()
    )
    
    return technical_chain

# Display chat messages
for message in st.session_state.messages:
    role = message["role"]
    content = message["content"]
    
    if role == "user":
        st.markdown(f'<div class="chat-message user-message">👤 {content}</div>', 
                   unsafe_allow_html=True)
    else:
        sources = message.get("sources", [])
        source_html = ""
        if sources:
            source_html = "<br><br><strong>📚 Sources:</strong><br>"
            for source in sources:
                source_html += f'<span class="source-badge">{source}</span>'
        
        st.markdown(f'<div class="chat-message assistant-message">🤖 {content}{source_html}</div>', 
                   unsafe_allow_html=True)

# Chat input
user_input = st.chat_input("Ask about enrollment, graduation, forms, scholarships...")

if user_input:
    # Add user message
    st.session_state.messages.append({"role": "user", "content": user_input})
    
    # Generate response
    with st.spinner("🔍 Searching knowledge base and generating response..."):
        try:
            # Create chain
            rag_chain = create_rag_chain(st.session_state.retriever, st.session_state.llm)
            
            # Get response
            response = rag_chain.invoke({"question": user_input})
            
            # Get source documents
            retrieved_docs = st.session_state.retriever.get_relevant_documents(user_input)
            sources = list(set([doc.metadata.get('source', 'Unknown').split('/')[-1] 
                               for doc in retrieved_docs[:3]]))
            
            # Add assistant message
            st.session_state.messages.append({
                "role": "assistant", 
                "content": response,
                "sources": sources
            })
            
        except Exception as e:
            st.error(f"Error generating response: {str(e)}")
            st.session_state.messages.append({
                "role": "assistant",
                "content": "I apologize, but I encountered an error processing your question. Please try again or contact the CEAT-OCS directly.",
                "sources": []
            })
    
    st.rerun()

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #6b7280; font-size: 0.875rem;">
    <p>CEAT Office of College Secretary RAG Assistant • Powered by OpenAI GPT-4 • Last updated: November 2025</p>
    <p>⚠️ This is an AI assistant. For official transactions, please contact the office directly.</p>
</div>
""", unsafe_allow_html=True)
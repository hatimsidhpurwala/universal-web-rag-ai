"""
Universal Web RAG AI System
Main Streamlit Application
"""

import streamlit as st
import time
from typing import Callable

# Import custom modules
from web_scraper import WebScraper
from data_cleaner import DataCleaner
from embeddings import EmbeddingGenerator
from vector_store import VectorStore
from llm import LLMGenerator

# Page configuration
st.set_page_config(
    page_title="Universal Web-Based AI System",
    page_icon="🌐",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .step-container {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
    }
    .step-complete {
        color: #4CAF50;
        font-weight: bold;
    }
    .step-pending {
        color: #999;
    }
    .step-active {
        color: #1E88E5;
        font-weight: bold;
    }
    .info-box {
        background-color: #e3f2fd;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #1E88E5;
    }
    .success-box {
        background-color: #e8f5e9;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #4CAF50;
    }
    .answer-box {
        background-color: #fff3e0;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 4px solid #FF9800;
        margin-top: 1rem;
    }
</style>
""", unsafe_allow_html=True)


# Initialize session state
if 'pipeline_state' not in st.session_state:
    st.session_state.pipeline_state = {
        'scraped': False,
        'cleaned': False,
        'embedded': False,
        'indexed': False,
        'ready': False,
        'url': '',
        'chunks': [],
        'vector_store': None,
        'llm': None
    }

if 'processing' not in st.session_state:
    st.session_state.processing = False


def update_progress(step_name: str, progress: float):
    """Update progress in session state"""
    st.session_state.current_step = step_name
    st.session_state.current_progress = progress


def reset_pipeline():
    """Reset the pipeline state"""
    st.session_state.pipeline_state = {
        'scraped': False,
        'cleaned': False,
        'embedded': False,
        'indexed': False,
        'ready': False,
        'url': '',
        'chunks': [],
        'vector_store': None,
        'llm': None
    }


def display_progress_steps():
    """Display the progress steps with visual indicators"""
    steps = [
        ("Scraping Started", 'scraped', 0.1),
        ("Data Extracted", 'scraped', 0.3),
        ("Data Cleaned", 'cleaned', 0.5),
        ("Embeddings Generated", 'embedded', 0.7),
        ("Vector Database Created", 'indexed', 0.9),
        ("Model Ready", 'ready', 1.0)
    ]
    
    cols = st.columns(6)
    
    for i, (step_name, state_key, _) in enumerate(steps):
        with cols[i]:
            if st.session_state.pipeline_state.get(state_key, False):
                if state_key == 'ready':
                    st.markdown(f"<div class='step-complete'>✅<br>{step_name}</div>", unsafe_allow_html=True)
                else:
                    st.markdown(f"<div class='step-complete'>✓<br>{step_name}</div>", unsafe_allow_html=True)
            elif st.session_state.get('current_step') == step_name:
                st.markdown(f"<div class='step-active'>⏳<br>{step_name}</div>", unsafe_allow_html=True)
            else:
                st.markdown(f"<div class='step-pending'>○<br>{step_name}</div>", unsafe_allow_html=True)


def process_website(url: str):
    """Process website through the entire pipeline"""
    try:
        st.session_state.processing = True
        
        # Step 1: Scraping
        update_progress("Scraping Started", 0.1)
        st.info("🌐 Step 1/6: Scraping website...")
        
        scraper = WebScraper()
        content = scraper.scrape_website(url)
        
        if 'error' in content:
            st.error(f"Error scraping website: {content['error']}")
            st.session_state.processing = False
            return False
        
        st.session_state.pipeline_state['scraped'] = True
        st.session_state.pipeline_state['url'] = url
        
        # Extract chunks
        chunks = scraper.get_text_chunks(content)
        st.info(f"📄 Extracted {len(chunks)} text chunks")
        
        time.sleep(0.5)  # Small delay for UI feedback
        
        # Step 2: Data Cleaning
        update_progress("Data Cleaned", 0.5)
        st.info("🧹 Step 2/6: Cleaning data...")
        
        cleaner = DataCleaner(min_length=20)
        cleaned_chunks = cleaner.clean_chunks(chunks)
        
        if len(cleaned_chunks) == 0:
            st.error("No valid content found after cleaning. Please try a different URL.")
            st.session_state.processing = False
            return False
        
        st.session_state.pipeline_state['cleaned'] = True
        st.session_state.pipeline_state['chunks'] = cleaned_chunks
        
        st.info(f"✨ Cleaned to {len(cleaned_chunks)} chunks")
        time.sleep(0.5)
        
        # Step 3: Generate Embeddings
        update_progress("Embeddings Generated", 0.7)
        st.info("🔢 Step 3/6: Generating embeddings...")
        
        embedding_gen = EmbeddingGenerator()
        embeddings = embedding_gen.generate_embeddings(cleaned_chunks, show_progress=True)
        
        st.session_state.pipeline_state['embedded'] = True
        time.sleep(0.5)
        
        # Step 4: Create Vector Store
        update_progress("Vector Database Created", 0.9)
        st.info("💾 Step 4/6: Creating vector database...")
        
        vector_store = VectorStore(dimension=embedding_gen.get_embedding_dimension())
        vector_store.create_index(embeddings, cleaned_chunks)
        
        st.session_state.pipeline_state['indexed'] = True
        st.session_state.pipeline_state['vector_store'] = vector_store
        time.sleep(0.5)
        
        # Step 5: Load LLM
        update_progress("Model Ready", 1.0)
        st.info("🤖 Step 5/6: Loading language model...")
        
        llm = LLMGenerator()
        llm.load_model()
        
        st.session_state.pipeline_state['llm'] = llm
        st.session_state.pipeline_state['ready'] = True
        
        st.success("✅ Pipeline complete! Model is ready for queries.")
        st.session_state.processing = False
        return True
        
    except Exception as e:
        st.error(f"Error processing website: {str(e)}")
        st.session_state.processing = False
        return False


def answer_query(query: str) -> str:
    """Answer a user query using the RAG pipeline"""
    try:
        vector_store = st.session_state.pipeline_state['vector_store']
        llm = st.session_state.pipeline_state['llm']
        
        if not vector_store or not llm:
            return "Error: Pipeline not initialized. Please process a website first."
        
        # Step 1: Generate query embedding
        st.info("🔍 Searching for relevant context...")
        embedding_gen = EmbeddingGenerator()
        query_embedding = embedding_gen.generate_query_embedding(query)
        
        # Step 2: Retrieve similar chunks
        similar_chunks, scores = vector_store.search(query_embedding, top_k=5)
        
        st.info(f"📚 Found {len(similar_chunks)} relevant passages")
        
        # Display retrieved context (collapsible)
        with st.expander("View Retrieved Context"):
            for i, (chunk, score) in enumerate(zip(similar_chunks, scores), 1):
                st.markdown(f"**Passage {i}** (relevance: {score:.3f})")
                st.text(chunk[:300] + "..." if len(chunk) > 300 else chunk)
                st.markdown("---")
        
        # Step 3: Generate answer
        st.info("🤖 Generating answer...")
        answer = llm.generate_answer(query, similar_chunks)
        
        return answer
        
    except Exception as e:
        return f"Error generating answer: {str(e)}"


def main():
    """Main application"""
    
    # Header
    st.markdown("<h1 class='main-header'>🌐 Universal Web-Based AI System</h1>", unsafe_allow_html=True)
    st.markdown("<p class='sub-header'>Process any website and query its content with AI-powered RAG</p>", unsafe_allow_html=True)
    
    # Pipeline Status Section
    st.header("📊 Pipeline Status")
    
    if st.session_state.pipeline_state['ready']:
        st.success("✅ Model Ready")
        st.markdown(f"**URL:** {st.session_state.pipeline_state['url'][:50]}...")
        st.markdown(f"**Chunks:** {len(st.session_state.pipeline_state['chunks'])}")
    else:
        st.warning("⏳ Not Initialized")
    
    # Reset Button
    if st.button("🔄 Reset Pipeline"):
        reset_pipeline()
        st.rerun()
    
    # Main content area
    tab1, tab2 = st.tabs(["🌐 Process Website", "❓ Ask Questions"])
    
    with tab1:
        st.header("Step 1: Process a Website")
        
        # URL Input
        url = st.text_input(
            "Enter Website URL:",
            placeholder="https://example.com",
            help="Enter the full URL of the website you want to process"
        )
        
        # Process button
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            process_button = st.button(
                "🚀 Process Website",
                use_container_width=True,
                disabled=st.session_state.processing,
                type="primary"
            )
        
        # Progress display
        if st.session_state.processing or st.session_state.pipeline_state['ready']:
            st.subheader("Processing Progress:")
            display_progress_steps()
        
        # Process website
        if process_button and url:
            if not url.startswith(('http://', 'https://')):
                url = 'https://' + url
            
            with st.spinner("Processing..."):
                success = process_website(url)
                
                if success:
                    st.balloons()
    
    with tab2:
        st.header("Step 2: Ask Questions")
        
        if not st.session_state.pipeline_state['ready']:
            st.warning("⚠️ Please process a website first (Step 1)")
        else:
            # Show website info
            st.markdown(f"<div class='info-box'>💡 Querying: <strong>{st.session_state.pipeline_state['url']}</strong></div>", unsafe_allow_html=True)
            
            # Query input
            query = st.text_area(
                "Enter your question:",
                placeholder="What is this website about?",
                help="Ask any question about the processed website content"
            )
            
            # Get Answer button
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                answer_button = st.button(
                    "💬 Get Answer",
                    use_container_width=True,
                    type="primary"
                )
            
            # Generate and display answer
            if answer_button and query:
                with st.spinner("Generating answer..."):
                    answer = answer_query(query)
                
                st.markdown("<div class='answer-box'>", unsafe_allow_html=True)
                st.subheader("📝 Answer:")
                st.write(answer)
                st.markdown("</div>", unsafe_allow_html=True)
                
                # Feedback buttons
                col1, col2 = st.columns(2)
                with col1:
                    st.button("👍 Helpful")
                with col2:
                    st.button("👎 Not Helpful")
    
    # Footer
    st.markdown("---")
    st.markdown("<p style='text-align: center; color: #666;'>Built with ❤️ using Streamlit, Transformers, and FAISS</p>", unsafe_allow_html=True)


if __name__ == "__main__":
    main()

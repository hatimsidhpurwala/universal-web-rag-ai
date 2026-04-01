# 🌐 Universal Web RAG AI System

A cloud-based Retrieval-Augmented Generation (RAG) AI system that allows users to process any website and query its content using natural language.

## 🚀 Features

- **Web Scraping**: Extract content from any website using requests + BeautifulSoup
- **Data Cleaning**: Remove duplicates, short text, and irrelevant content
- **Embeddings**: Convert text to embeddings using SentenceTransformers (all-MiniLM-L6-v2)
- **Vector Database**: Store and search embeddings using FAISS
- **AI Q&A**: Generate answers using HuggingFace's flan-t5-base model
- **User-Friendly UI**: Clean Streamlit interface with step-by-step progress

## 🛠️ Local Setup Instructions

### Prerequisites

- Python 3.9 - 3.12 (recommended: 3.11)
- pip or conda
- 4GB+ RAM
- 2GB+ free disk space (for models)

### Step 1: Clone/Download the Project

```bash
cd universal-web-rag-ai
```

### Step 2: Create Virtual Environment (Recommended)

**Using venv:**
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

**Using conda:**
```bash
conda create -n rag-ai python=3.11
conda activate rag-ai
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

**Note:** First installation downloads ~1GB of ML models.

### Step 4: Run the Application

```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`

## ☁️ Deploy to Streamlit Cloud

1. **Push to GitHub:**
```bash
git init
git add .
git commit -m "Initial commit"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/scrapperv4.git
git push -u origin main
```

2. **Deploy on Streamlit Cloud:**
   - Go to [share.streamlit.io](https://share.streamlit.io)
   - Sign in with GitHub
   - Click "New app"
   - Select your repository
   - Main file path: `app.py`
   - Click "Deploy"

**Note:** Streamlit Cloud uses Python 3.14. The requirements.txt has been updated for compatibility.

## 📁 Project Structure

```
universal-web-rag-ai/
├── app.py              # Main Streamlit application
├── web_scraper.py      # Web scraping module
├── data_cleaner.py     # Data cleaning module
├── embeddings.py       # Embeddings generation module
├── vector_store.py     # FAISS vector database module
├── llm.py              # LLM integration module
├── requirements.txt    # Python dependencies
└── README.md          # This file
```

## 🔧 Pipeline Flow

```
URL Input → Scraping → Cleaning → Embeddings → FAISS → Query → Answer
```

1. **Scraping**: Extract headings and paragraphs from the website
2. **Cleaning**: Remove duplicates, short text, and irrelevant words
3. **Embeddings**: Generate vector representations using all-MiniLM-L6-v2
4. **Vector DB**: Store embeddings in FAISS for similarity search
5. **Query**: Retrieve relevant context and generate answers with flan-t5-base

## 🧠 Models Used

| Component | Model | Purpose |
|-----------|-------|---------|
| Embeddings | `all-MiniLM-L6-v2` | Convert text to vectors |
| LLM | `google/flan-t5-base` | Generate natural language answers |
| Vector DB | `FAISS` | Efficient similarity search |

## 📝 Example Usage

1. Enter a website URL (e.g., `https://en.wikipedia.org/wiki/Artificial_intelligence`)
2. Click "Process Website" and wait for the pipeline to complete
3. Ask questions like:
   - "What is this website about?"
   - "What are the main topics covered?"
   - "Explain the key concepts mentioned"

## ⚠️ Troubleshooting

### Common Issues

**1. torch installation fails:**
```bash
# For CPU-only (recommended for local)
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

**2. Out of memory:**
- Close other applications
- Use a smaller website for testing
- Increase swap space

**3. Model download slow:**
- First run downloads ~1GB of models
- Subsequent runs are faster (models cached)

**4. Website blocked error:**
- Some websites block scrapers
- Try a different URL
- Check if website has `robots.txt` restrictions

## 📄 License

MIT License

## 🤝 Contributing

Contributions welcome! Please feel free to submit a Pull Request.

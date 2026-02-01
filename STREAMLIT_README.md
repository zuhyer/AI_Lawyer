# 🚀 Running the Streamlit App

## Prerequisites

1. **Build the embeddings** (if not already done):
   ```bash
   python main.py
   ```
   This creates the FAISS index from PDFs (takes ~15-20 minutes on first run).

2. **Update your Groq API key** in `config/secret.yaml`:
   ```yaml
   response_model_API_Key: "YOUR_ACTUAL_GROQ_API_KEY"
   ```
   Get your key from [console.groq.com](https://console.groq.com).

## Run the App

```bash
streamlit run app.py
```

The app will open at: `http://localhost:8501`

## Features

- **Interactive Q&A**: Ask questions about Indian legal documents
- **RAG System**: Retrieves relevant document chunks and generates answers using Groq LLM
- **Local Embeddings**: Uses Sentence Transformers (all-MiniLM-L6-v2) — no API key needed for embeddings
- **FAISS Vector Store**: Fast semantic search across 12,000+ document chunks
- **Clean UI**: Streamlit interface with formatted responses

## Troubleshooting

### FAISS Database Not Found
- Run `python main.py` to build the index from PDFs

### Groq 401 Invalid API Key
- Verify `config/secret.yaml` has a valid Groq API key
- Run `python check_groq_key.py` to validate the resolved key
- Ensure the key is not a Hugging Face token (`hf_...`)

### Slow Response
- First query may be slower due to LLM cold start
- Subsequent queries should be faster
- Check internet connection (Groq API requires network access)

## Example Queries

- "What are the key provisions of the Indian Penal Code?"
- "What is bail under the CrPC?"
- "Explain the right to equality under Article 14 of the Constitution"
- "What are the rules for evidence in criminal proceedings?"
- "What is the procedure for filing a civil suit?"

## Architecture

```
User Query (Streamlit UI)
    ↓
QueryComponent
    ├─ FAISS: Semantic search
    ├─ Groq LLM: Response generation
    └─ Prompt template: Legal-grade formatting
    ↓
Formatted Response (displayed in UI)
```

Enjoy! ⚖️

# App Structure - Two Entry Points, Same Hybrid Implementation

## Overview

Your project now has **two Streamlit entry points** that are identical:

| File | Purpose | Status |
|------|---------|--------|
| **`app.py`** | Primary entry point (Hybrid) | ✅ ACTIVE |
| **`streamlit_app.py`** | Backup/alternate entry point (Hybrid) | ✅ ACTIVE |

Both files contain the **same hybrid implementation** with the upload button prominently visible in the sidebar.

---

## How to Run

### Option 1: Using `app.py` (Recommended)
```bash
python -m streamlit run app.py
```

### Option 2: Using `streamlit_app.py`
```bash
python -m streamlit run streamlit_app.py
```

Both commands will launch the **identical hybrid legal assistant** with:
- ✅ Upload button in the sidebar
- ✅ Hybrid search mode (Legal DB + User Documents)
- ✅ Standard search mode (Legal DB only)
- ✅ Full file processing pipeline

---

## What Changed

### Before
- `app.py` = Basic version (no uploads)
- `streamlit_app.py` = Hybrid version (with uploads)
- ❌ Inconsistency and confusion

### After
- `app.py` = Hybrid version (with uploads) ✅
- `streamlit_app.py` = Hybrid version (with uploads) ✅
- ✅ Both are identical and synchronized

---

## Feature Comparison

### Standard Mode (Legal DB Only)
- Search only pre-indexed Indian legal documents
- No upload required
- Faster queries
- Smaller response scope

### Hybrid Mode (Legal DB + User Documents)
- Search legal database AND your uploaded documents
- Upload files in the sidebar
- Comprehensive coverage
- Requires at least one uploaded document

---

## Sidebar Layout

Both apps display the same sidebar:

```
⚙️ Settings & Upload
━━━━━━━━━━━━━━━━━━
📤 Upload Documents
[SELECT FILES BUTTON] ← Upload button is here!
[📤 Process Uploads]
📊 Total Chunks: X
━━━━━━━━━━━━━━━━━━
🔍 Search Settings
☑ Search Legal Database
☑ Search Uploaded Documents
🔹 Number of Results
━━━━━━━━━━━━━━━━━━
ℹ️ System Status
Legal DB: ✅ Ready
Uploads: X chunks
```

---

## File Support

Both apps support:
- **Documents**: PDF, DOCX, DOC, TXT
- **Images**: PNG, JPG, JPEG, BMP, TIFF (with OCR)

---

## Key Components

Both `app.py` and `streamlit_app.py` use:
- `ConfigurationManager` - Configuration management
- `FileExtractor` - Text/image extraction
- `UserUploadProcessor` - Document chunking
- `QueryComponent` - LLM query processing
- `LocalSentenceTransformerEmbeddings` - Embeddings
- `FAISS` - Vector similarity search

---

## Session State Management

Both apps initialize and maintain:
- `config_manager` - Configuration
- `embedding_model` - Sentence embeddings
- `faiss_db` - Legal database
- `query_component` - Query processor
- `user_documents` - Uploaded documents
- `upload_processor` - Upload handler
- `file_extractor` - File extraction

---

## Troubleshooting

### Upload button not visible?
- Make sure sidebar is expanded (⬅️ at top left)
- Check that you're running the right file (app.py or streamlit_app.py)
- Try refreshing the page

### Files not processing?
- Ensure file format is supported (PDF, DOCX, TXT, PNG, JPG, BMP, TIFF)
- Check file isn't corrupted
- Max file size is 50MB
- Click "📤 Process Uploads" button after selecting files

### Hybrid mode not working?
- Must have uploaded documents first
- Can switch modes using the "Query Mode" radio button
- System will show warning if trying Hybrid with no uploads

---

## Recommendation

**Use `app.py`** as your primary entry point since it's the standard name and both are now identical. Keep `streamlit_app.py` as a backup.

---

## Version Info

- **Implementation**: Hybrid Legal Assistant
- **Upload Support**: ✅ Yes
- **Hybrid Mode**: ✅ Yes
- **Status**: Production Ready
- **Last Updated**: January 15, 2026

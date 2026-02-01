# Source Name Display Fix

## Problem
The UI was showing file paths or generic names instead of the actual document filenames as sources when displaying query results.

## Root Cause
When PDFPlumberLoader loads documents, it stores the full file path in the `source` metadata field. When these were displayed in the UI, users saw the complete file paths instead of clean, readable filenames.

## Solution
Implemented `_extract_source_name()` helper method to properly extract and clean source names from metadata.

## Changes Made

### 1. **QueryComponent** (`src/AI_Lawyer/components/query_component.py`)
   - **Added** import: `from pathlib import Path`
   - **Added** helper method `_extract_source_name()` that:
     - Extracts just the filename from full file paths
     - Decodes URL-encoded characters (e.g., `%20` → space)
     - Handles edge cases gracefully
   
   - **Updated** `get_context()` method to use the helper
     - Now displays clean filenames in the context string
   
   - **Updated** `query_with_user_files()` (hybrid query) method to use the helper
     - Now provides clean source names in the response

### 2. **Streamlit App** (`streamlit_app.py`)
   - **Added** helper function `extract_clean_source_name()` 
     - Same functionality as in QueryComponent
     - Keeps Streamlit layer consistent with backend
   
   - **Updated** standard query source display
     - Now shows clean filenames in both table and expandable sections

## Example

### Before
- **Source:** `/workspaces/AI_Lawyer/artifacts/data/pdfs/A1872-15.pdf`
- **Source:** `/workspaces/AI_Lawyer/artifacts/data/pdfs/250882_english_01042024.pdf`

### After
- **Source:** `A1872-15.pdf`
- **Source:** `250882_english_01042024.pdf`

## Files Modified
1. `/workspaces/AI_Lawyer/src/AI_Lawyer/components/query_component.py`
2. `/workspaces/AI_Lawyer/streamlit_app.py`

## Testing
To verify the fix works:

1. **Standard Query Mode:**
   - Run a query on the legal database
   - Check that sources show actual filenames (e.g., `A1872-15.pdf`)

2. **Hybrid Query Mode:**
   - Upload a document
   - Run a hybrid query
   - Check that sources show:
     - Uploaded filenames for user documents (e.g., `my_document.pdf`)
     - Legal database filenames for permanent documents

3. **API Response:**
   - Query the `/query/hybrid` endpoint with user documents
   - Verify the response sources contain clean filenames

## Benefits
- ✅ Better user experience with readable source names
- ✅ Clear distinction between different document sources
- ✅ Works for both local database and uploaded documents
- ✅ Handles URL-encoded filenames properly
- ✅ Graceful fallback to "Unknown Source" if needed

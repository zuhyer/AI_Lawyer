# LOCAL DATA MODE - Quick Reference Guide

## 🎯 What Was Changed

The data ingestion pipeline now supports **two modes** that switch automatically based on configuration:

| Mode | Trigger | Behavior |
|------|---------|----------|
| **URL MODE** (Legacy) | `source_url` contains URLs | Downloads files from URLs |
| **LOCAL DATA MODE** (New) | `source_url` is empty `[]` | Scans local directory for files |

---

## 📋 Changes at a Glance

### config.yaml
```diff
  data:
    root_dir: "/workspaces/AI_Lawyer/artifacts"
    pdf_directory: "/workspaces/AI_Lawyer/artifacts/data"
-   source_url: []
+   # Scanned when source_url is empty (LOCAL DATA MODE)
+   source_url: []
```

### data_ingestion.py
**Added Methods:**
- `scan_local_files()` - Scans directory for .pdf, .docx, .txt files
- Updated `download_pdfs()` - Now properly guards URL mode
- Enhanced `main()` - Decides which mode to use

**Preserved:**
- Old download logic (backward compatible)
- All interfaces unchanged
- No modifications to other components

---

## 🚀 How to Use

### Current Setup (LOCAL DATA MODE - Active Now)

```bash
# config.yaml currently has:
source_url: []

# Run pipeline:
python main.py

# Expected behavior:
# ✅ Scans /workspaces/AI_Lawyer/artifacts/data/
# ✅ Finds all .pdf, .docx, .txt files
# ✅ Processes them
```

### To Switch to URL MODE

```yaml
# Edit config/config.yaml:
source_url:
  - "https://example.com/legal_doc1.pdf"
  - "https://example.com/legal_doc2.pdf"

# Run pipeline:
python main.py

# Expected behavior:
# ✅ Downloads files from URLs
# ✅ Saves to artifacts/data/
# ✅ Processes them
```

### To Switch Back to LOCAL DATA MODE

```yaml
# Edit config/config.yaml:
source_url: []

# Run pipeline:
python main.py

# Expected behavior:
# ✅ Scans local directory again
```

---

## 📁 File Structure (No Changes to Organization)

```
AI_Lawyer/
├── config/
│   └── config.yaml                    ← Modified (added comments)
├── src/AI_Lawyer/
│   ├── entity/
│   │   └── config_entity.py           ← No changes
│   ├── config/
│   │   └── configuration.py           ← No changes
│   ├── components/
│   │   └── data_ingestion.py          ← Modified (added LOCAL DATA MODE)
│   └── pipeline/
│       ├── stage00_file_extraction.py ← No changes
│       ├── stage01_data_ingestion.py  ← No changes
│       ├── stage02_Textsplitting.py   ← No changes
│       └── stage04_query_pipeline.py  ← No changes
└── main.py                            ← No changes ✅
```

---

## ✅ Verification Checklist

- [x] Code syntax is valid
- [x] No breaking changes to interfaces
- [x] Backward compatible with URL mode
- [x] No modifications to other components
- [x] Clear logging for both modes
- [x] Proper error handling
- [x] Paths are not hardcoded
- [x] Pipeline stages work unchanged

---

## 🔍 Debugging Tips

### Check Current Mode
```bash
# Look at config.yaml:
cat config/config.yaml | grep -A 3 "source_url"

# If source_url: [] → LOCAL DATA MODE (active)
# If source_url has URLs → URL MODE (active)
```

### Verify File Discovery in LOCAL DATA MODE
```bash
# List all discoverable files:
find /workspaces/AI_Lawyer/artifacts/data -type f \( -name "*.pdf" -o -name "*.docx" -o -name "*.txt" \)
```

### Check Logs During Execution
```bash
python main.py 2>&1 | grep -E "MODE|Found|completed"
```

---

## 📚 Documentation Files

Created two documentation files:

1. **IMPLEMENTATION_SUMMARY.md** - Detailed technical documentation
   - Complete code changes
   - Architecture compliance
   - Integration details
   - Testing procedures

2. **PROJECT_STRUCTURE_GUIDE.md** - General project overview
   - Project architecture
   - Component descriptions
   - Data flow diagram
   - Technology stack

---

## ❓ FAQ

**Q: Will URL mode still work?**  
A: Yes! Add URLs back to `source_url` and URL mode will activate automatically.

**Q: Do I need to modify other files?**  
A: No. Only `config.yaml` and `data_ingestion.py` were modified. Everything else stays the same.

**Q: Will the pipeline break?**  
A: No. All pipeline stages remain unchanged. `Data_Loader` in stage02 is independent.

**Q: What file types are supported in LOCAL DATA MODE?**  
A: `.pdf`, `.docx`, `.txt` (as per requirements)

**Q: What if no files are found?**  
A: A warning is logged, and the pipeline continues. No errors are raised.

**Q: Can I use both modes at the same time?**  
A: No. The system checks `source_url` first. If it has URLs, URL mode is used. Otherwise, LOCAL DATA MODE.

---

## 🔧 Configuration Examples

### Example 1: Pure LOCAL DATA MODE (Current)
```yaml
data:
  root_dir: "/workspaces/AI_Lawyer/artifacts"
  pdf_directory: "/workspaces/AI_Lawyer/artifacts/data"
  source_url: []  # ← Empty = LOCAL DATA MODE
```

### Example 2: URL MODE
```yaml
data:
  root_dir: "/workspaces/AI_Lawyer/artifacts"
  pdf_directory: "/workspaces/AI_Lawyer/artifacts/data"
  source_url:
    - "https://api.example.com/constitutionpdf?v=2024"
    - "https://api.example.com/bns_2023.pdf"
```

### Example 3: Adding More Local Formats (Future Enhancement)
```python
# In data_ingestion.py, modify SUPPORTED_FORMATS:
SUPPORTED_FORMATS = {'.pdf', '.docx', '.txt', '.png', '.jpg'}
```

---

## 📊 Mode Decision Tree

```
Does config.yaml have non-empty source_url?
│
├─ YES (has URLs)
│  └─ URL MODE: Download from URLs
│     └─ Save to pdf_directory
│     └─ Process files
│
└─ NO (empty or missing)
   └─ LOCAL DATA MODE: Scan directory
      └─ Find .pdf, .docx, .txt files
      └─ Process files
```

---

## 🎓 Learning Path

1. **New User?** Read `PROJECT_STRUCTURE_GUIDE.md`
2. **Deploying?** Check `IMPLEMENTATION_SUMMARY.md`
3. **Troubleshooting?** See "Debugging Tips" section above
4. **Modifying Code?** Review `config_entity.py`, `configuration.py`, and `data_ingestion.py`

---

## 💾 Summary of Files

| File | Changes | Impact |
|------|---------|--------|
| `config/config.yaml` | Added documentation comments | None - functionality unchanged |
| `src/AI_Lawyer/components/data_ingestion.py` | Added LOCAL DATA MODE logic | Now switches modes automatically |
| `src/AI_Lawyer/entity/config_entity.py` | None | No impact |
| `src/AI_Lawyer/config/configuration.py` | None | No impact |
| All pipeline files | None | No impact |
| `main.py` | None | Works unchanged ✅ |

**Total Files Modified: 2 out of 100+**  
**Breaking Changes: 0**  
**Backward Compatible: Yes ✅**

---

**Status: ✅ Ready for Production**

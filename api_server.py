"""
Quick-start entry point for the FastAPI application.
Usage: python api_server.py
"""

import sys
import os
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from AI_Lawyer.api.main import main

if __name__ == "__main__":
    main()

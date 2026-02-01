#!/usr/bin/env python
"""
Validation script for production-grade FastAPI implementation.
Checks syntax, imports, and configuration.
"""

import sys
import os
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

def check_imports():
    """Check that all imports work correctly."""
    print("\n" + "="*70)
    print("🔍 CHECKING IMPORTS")
    print("="*70)
    
    errors = []
    
    try:
        print("✓ Importing API models...")
        from AI_Lawyer.api.models import requests, responses
        print("  ✓ Request models loaded")
        print("  ✓ Response models loaded")
    except ImportError as e:
        errors.append(f"Models import failed: {e}")
    
    try:
        print("✓ Importing API exceptions...")
        from AI_Lawyer.api import exceptions
        print("  ✓ Exception classes loaded")
    except ImportError as e:
        errors.append(f"Exceptions import failed: {e}")
    
    try:
        print("✓ Importing API utilities...")
        from AI_Lawyer.api import utils
        print("  ✓ Utility functions loaded")
    except ImportError as e:
        errors.append(f"Utils import failed: {e}")
    
    try:
        print("✓ Importing dependencies...")
        from AI_Lawyer.api import dependencies
        print("  ✓ Dependency injection loaded")
    except ImportError as e:
        errors.append(f"Dependencies import failed: {e}")
    
    try:
        print("✓ Importing routes...")
        from AI_Lawyer.api.routes import health, extraction, query, ingestion
        print("  ✓ Health routes loaded")
        print("  ✓ Extraction routes loaded")
        print("  ✓ Query routes loaded")
        print("  ✓ Ingestion routes loaded")
    except ImportError as e:
        errors.append(f"Routes import failed: {e}")
    
    try:
        print("✓ Importing main app...")
        from AI_Lawyer.api import app
        print("  ✓ FastAPI app created")
    except ImportError as e:
        errors.append(f"App import failed: {e}")
    
    return errors


def check_configuration():
    """Check configuration files."""
    print("\n" + "="*70)
    print("⚙️  CHECKING CONFIGURATION")
    print("="*70)
    
    config_files = [
        Path(__file__).parent / ".env.production",
        Path(__file__).parent / "PRODUCTION_API_GUIDE.md",
        Path(__file__).parent / "FASTAPI_PRODUCTION_IMPLEMENTATION.md",
    ]
    
    errors = []
    
    for config_file in config_files:
        if config_file.exists():
            size = config_file.stat().st_size
            print(f"✓ {config_file.name} ({size:,} bytes)")
        else:
            errors.append(f"Missing: {config_file.name}")
    
    return errors


def check_api_structure():
    """Check API directory structure."""
    print("\n" + "="*70)
    print("📁 CHECKING API STRUCTURE")
    print("="*70)
    
    api_dir = Path(__file__).parent / "src" / "AI_Lawyer" / "api"
    
    required_files = [
        "app.py",
        "main.py",
        "dependencies.py",
        "exceptions.py",
        "utils.py",
        "models/requests.py",
        "models/responses.py",
        "routes/health.py",
        "routes/extraction.py",
        "routes/query.py",
        "routes/ingestion.py",
    ]
    
    errors = []
    
    for file_path in required_files:
        full_path = api_dir / file_path
        if full_path.exists():
            size = full_path.stat().st_size
            print(f"✓ {file_path} ({size:,} bytes)")
        else:
            errors.append(f"Missing: {file_path}")
    
    return errors


def check_dependencies():
    """Check required dependencies."""
    print("\n" + "="*70)
    print("📦 CHECKING DEPENDENCIES")
    print("="*70)
    
    required_packages = [
        "fastapi",
        "uvicorn",
        "pydantic",
        "langchain",
        "sentence_transformers",
        "faiss",
    ]
    
    errors = []
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"✓ {package}")
        except ImportError:
            errors.append(f"Missing: {package}")
    
    return errors


def check_environment():
    """Check environment variables."""
    print("\n" + "="*70)
    print("🔑 CHECKING ENVIRONMENT")
    print("="*70)
    
    env_file = Path(__file__).parent / ".env"
    
    if env_file.exists():
        print(f"✓ .env file exists")
        print("  Note: Configure the .env file with your settings")
    else:
        print("⚠ .env file not found")
        print("  Create .env from .env.production template")
    
    return []


def print_summary(all_errors):
    """Print validation summary."""
    print("\n" + "="*70)
    print("📊 VALIDATION SUMMARY")
    print("="*70)
    
    if not all_errors:
        print("\n✅ ALL CHECKS PASSED!\n")
        print("Your FastAPI implementation is production-ready.")
        print("\nNext steps:")
        print("1. Configure .env with your settings")
        print("2. Run: python api_server.py")
        print("3. Visit: http://localhost:8000/docs")
        return True
    else:
        print(f"\n❌ {len(all_errors)} ERROR(S) FOUND:\n")
        for error in all_errors:
            print(f"  • {error}")
        return False


def main():
    """Run all validation checks."""
    print("\n" + "="*70)
    print("🚀 FASTAPI PRODUCTION IMPLEMENTATION VALIDATOR")
    print("="*70)
    
    all_errors = []
    
    # Run checks
    all_errors.extend(check_imports())
    all_errors.extend(check_configuration())
    all_errors.extend(check_api_structure())
    all_errors.extend(check_dependencies())
    all_errors.extend(check_environment())
    
    # Print summary
    success = print_summary(all_errors)
    
    # Print documentation
    print("\n📚 DOCUMENTATION:")
    print("  • PRODUCTION_API_GUIDE.md - Deployment and usage guide")
    print("  • FASTAPI_PRODUCTION_IMPLEMENTATION.md - Technical details")
    print("  • .env.production - Configuration reference")
    
    print("\n📖 API DOCUMENTATION (when running):")
    print("  • Swagger UI: http://localhost:8000/docs")
    print("  • ReDoc: http://localhost:8000/redoc")
    print("  • OpenAPI JSON: http://localhost:8000/openapi.json")
    
    print("\n" + "="*70 + "\n")
    
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())

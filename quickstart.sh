#!/bin/bash
# AI Lawyer API - Quick Start Script
# This script helps you get the API running in minutes

set -e

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Print with color
print_header() {
    echo -e "\n${BLUE}═══════════════════════════════════════════════════════════════${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}═══════════════════════════════════════════════════════════════${NC}\n"
}

print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠ $1${NC}"
}

print_error() {
    echo -e "${RED}✗ $1${NC}"
}

print_step() {
    echo -e "\n${BLUE}→ $1${NC}"
}

# Main script
main() {
    print_header "🚀 AI LAWYER API - QUICK START SETUP"

    # Check Python
    print_step "Checking Python installation..."
    if ! command -v python3 &> /dev/null; then
        print_error "Python 3 not found. Please install Python 3.8+"
        exit 1
    fi
    PYTHON_VERSION=$(python3 --version | awk '{print $2}')
    print_success "Python $PYTHON_VERSION found"

    # Check current directory
    print_step "Checking directory structure..."
    if [ ! -f "requirements.txt" ]; then
        print_error "requirements.txt not found. Please run from project root."
        exit 1
    fi
    print_success "Project structure verified"

    # Install dependencies
    print_step "Installing dependencies..."
    if python3 -m pip install -r requirements.txt --quiet; then
        print_success "Dependencies installed"
    else
        print_error "Failed to install dependencies"
        exit 1
    fi

    # Check for .env file
    print_step "Checking environment configuration..."
    if [ ! -f ".env" ]; then
        if [ -f ".env.production" ]; then
            print_warning ".env not found, creating from .env.production"
            cp .env.production .env
            print_success ".env created from template"
            print_warning "Please edit .env and configure your API keys"
        else
            print_error ".env.production not found"
            exit 1
        fi
    else
        print_success ".env configuration file found"
    fi

    # Validate setup
    print_step "Validating setup..."
    if python3 validate_api.py 2>&1 | tail -20; then
        print_success "Setup validation passed"
    else
        print_warning "Validation script encountered issues"
    fi

    # Display next steps
    print_header "✅ SETUP COMPLETE - NEXT STEPS"

    echo -e "${YELLOW}1. Configure your API keys:${NC}"
    echo "   Edit .env and set these important variables:"
    echo "   - GROQ_API_KEY (required for LLM)"
    echo "   - EMBEDDING_MODEL (optional, defaults to all-MiniLM-L6-v2)"
    echo "   - LLM_MODEL (optional, defaults to mixtral-8x7b-32768)"
    echo ""

    echo -e "${YELLOW}2. Start the API server:${NC}"
    echo "   python api_server.py"
    echo ""

    echo -e "${YELLOW}3. Access the API:${NC}"
    echo "   - Swagger UI:  ${GREEN}http://localhost:8000/docs${NC}"
    echo "   - ReDoc:       ${GREEN}http://localhost:8000/redoc${NC}"
    echo "   - Health:      ${GREEN}http://localhost:8000/health${NC}"
    echo ""

    echo -e "${YELLOW}4. Test an endpoint:${NC}"
    echo "   curl http://localhost:8000/health"
    echo ""

    echo -e "${YELLOW}Documentation:${NC}"
    echo "   - Quick Reference:     FASTAPI_QUICK_REFERENCE.md"
    echo "   - Full Guide:          PRODUCTION_API_GUIDE.md"
    echo "   - Technical Details:   FASTAPI_PRODUCTION_IMPLEMENTATION.md"
    echo "   - Implementation:      IMPLEMENTATION_CHECKLIST.md"
    echo ""

    # Optional: Ask to start API
    read -p "Start the API server now? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        print_header "🚀 STARTING API SERVER..."
        echo "Press Ctrl+C to stop the server"
        echo ""
        python3 api_server.py
    else
        print_success "Setup complete! Run 'python api_server.py' when ready."
    fi
}

# Run main
main

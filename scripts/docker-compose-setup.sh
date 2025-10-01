#!/usr/bin/env bash
# ===========================================================================
# MyWebIntelligence - Docker Compose Quick Setup
# ===========================================================================
#
# This script provides a streamlined setup for MyWI using Docker Compose.
# It automates common installation tasks with sensible defaults.
#
# Usage:
#   ./scripts/docker-compose-setup.sh [LEVEL]
#
# LEVEL (optional):
#   basic - Core functionality only (default)
#   api   - Basic + external APIs (SerpAPI, SEO Rank, OpenRouter)
#   llm   - Complete with ML/embeddings/NLI
#
# Examples:
#   ./scripts/docker-compose-setup.sh           # Basic installation
#   ./scripts/docker-compose-setup.sh api       # With APIs
#   ./scripts/docker-compose-setup.sh llm       # Full installation
#
# ===========================================================================

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Helper functions
info() {
    echo -e "${BLUE}ℹ${NC} $1"
}

success() {
    echo -e "${GREEN}✓${NC} $1"
}

error() {
    echo -e "${RED}✗${NC} $1"
}

warning() {
    echo -e "${YELLOW}⚠${NC} $1"
}

section() {
    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo -e "${BLUE}$1${NC}"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
}

# Banner
echo ""
echo "┌─────────────────────────────────────────────────┐"
echo "│  MyWebIntelligence - Docker Compose Setup      │"
echo "└─────────────────────────────────────────────────┘"
echo ""

# Parse arguments
LEVEL="${1:-basic}"

if [[ ! "$LEVEL" =~ ^(basic|api|llm)$ ]]; then
    error "Invalid level: $LEVEL"
    echo ""
    echo "Usage: $0 [basic|api|llm]"
    exit 1
fi

info "Installation level: $LEVEL"
echo ""

# Check prerequisites
section "Checking Prerequisites"

# Check Docker
if ! command -v docker &> /dev/null; then
    error "Docker not found"
    info "Install Docker Desktop from: https://www.docker.com/products/docker-desktop/"
    exit 1
fi
success "Docker installed"

# Check Docker Compose
if ! docker compose version &> /dev/null; then
    error "Docker Compose not found"
    info "Docker Compose comes with Docker Desktop"
    exit 1
fi
success "Docker Compose installed"

# Check Python (for interactive script)
if ! command -v python3 &> /dev/null && ! command -v python &> /dev/null; then
    warning "Python not found - will use manual .env setup"
    USE_PYTHON=false
else
    success "Python installed"
    USE_PYTHON=true
fi

# Step 1: Create .env file
section "Configuration Setup"

if [ -f .env ]; then
    warning ".env file already exists"
    read -p "Overwrite existing .env? [y/N] " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        info "Keeping existing .env"
    else
        # Backup existing .env
        BACKUP_FILE=".env.backup.$(date +%Y%m%d_%H%M%S)"
        cp .env "$BACKUP_FILE"
        success "Backed up to $BACKUP_FILE"

        # Run interactive setup
        if [ "$USE_PYTHON" = true ]; then
            info "Running interactive configuration..."
            python3 scripts/install-docker-compose.py --level="$LEVEL" || python scripts/install-docker-compose.py --level="$LEVEL"
        else
            info "Copying .env.example to .env"
            cp .env.example .env
            warning "Please edit .env manually to add your API keys"
        fi
    fi
else
    # No existing .env, create new one
    if [ "$USE_PYTHON" = true ]; then
        info "Running interactive configuration..."
        python3 scripts/install-docker-compose.py --level="$LEVEL" || python scripts/install-docker-compose.py --level="$LEVEL"
    else
        info "Copying .env.example to .env"
        cp .env.example .env

        # Set ML flag for LLM level
        if [ "$LEVEL" = "llm" ]; then
            sed -i.bak 's/MYWI_WITH_ML=0/MYWI_WITH_ML=1/' .env && rm .env.bak
        fi

        warning "Please edit .env manually to add your API keys"
    fi
fi

# Step 2: Create data directory
section "Data Directory Setup"

# Read HOST_DATA_DIR from .env
if [ -f .env ]; then
    source .env
    DATA_DIR="${HOST_DATA_DIR:-./data}"
else
    DATA_DIR="./data"
fi

if [ ! -d "$DATA_DIR" ]; then
    mkdir -p "$DATA_DIR"
    success "Created data directory: $DATA_DIR"
else
    info "Data directory exists: $DATA_DIR"
fi

# Step 3: Create settings.py if needed
section "Settings File Check"

if [ ! -f settings.py ]; then
    info "Copying settings-example.py to settings.py"
    cp settings-example.py settings.py
    success "Created settings.py"
else
    info "settings.py already exists"
fi

# Step 4: Build and start Docker Compose
section "Building Docker Image"

info "This may take several minutes on first run..."
echo ""

# Build with appropriate flags
if docker compose up -d --build; then
    success "Docker Compose services built and started"
else
    error "Docker Compose build failed"
    exit 1
fi

# Step 5: Initialize database
section "Database Initialization"

# Wait for container to be ready
sleep 2

info "Initializing database..."
if docker compose exec mwi python mywi.py db setup; then
    success "Database initialized"
else
    error "Database initialization failed"
    exit 1
fi

# Step 6: Verify installation
section "Verification"

info "Checking installation..."
if docker compose exec mwi python mywi.py land list &> /dev/null; then
    success "MyWI is running correctly"
else
    warning "MyWI may not be configured correctly"
fi

# Test APIs if configured
if [ "$LEVEL" = "api" ] || [ "$LEVEL" = "llm" ]; then
    info "Testing API connections..."
    docker compose exec mwi python scripts/test-apis.py --all || warning "Some API tests failed (may need configuration)"
fi

# Check ML environment for LLM level
if [ "$LEVEL" = "llm" ]; then
    info "Checking ML environment..."
    docker compose exec mwi python mywi.py embedding check || warning "ML environment may need configuration"
fi

# Final summary
section "Installation Complete!"

echo ""
success "MyWI is ready to use!"
echo ""

info "Data location: $DATA_DIR"
info "Container: mwi (running)"
echo ""

info "Next steps:"
echo "  1. Create a research land:"
echo "     docker compose exec mwi python mywi.py land create --name=\"MyResearch\" --desc=\"Description\""
echo ""
echo "  2. Add terms:"
echo "     docker compose exec mwi python mywi.py land addterm --land=\"MyResearch\" --terms=\"keyword1, keyword2\""
echo ""
echo "  3. Add URLs:"
echo "     docker compose exec mwi python mywi.py land addurl --land=\"MyResearch\" --urls=\"https://example.com\""
echo ""
echo "  4. Crawl URLs:"
echo "     docker compose exec mwi python mywi.py land crawl --name=\"MyResearch\""
echo ""

info "Management commands:"
echo "  docker compose up -d          # Start services"
echo "  docker compose down           # Stop services"
echo "  docker compose logs mwi       # View logs"
echo "  docker compose exec mwi bash  # Enter container"
echo ""

if [ "$LEVEL" = "llm" ]; then
    info "LLM features:"
    echo "  Generate embeddings:"
    echo "     docker compose exec mwi python mywi.py embedding generate --name=\"MyResearch\""
    echo ""
    echo "  Compute similarities:"
    echo "     docker compose exec mwi python mywi.py embedding similarity --name=\"MyResearch\" --method=cosine"
    echo ""
    echo "  Export pseudolinks:"
    echo "     docker compose exec mwi python mywi.py land export --name=\"MyResearch\" --type=pseudolinks"
    echo ""
fi

warning "Important:"
echo "  - Do NOT commit .env to version control (contains API keys)"
echo "  - Your data is persistent in: $DATA_DIR"
echo ""

success "Setup complete!"
echo ""

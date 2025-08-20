#!/bin/bash

echo "🚀 Starting MCP Web Scraper & QA System"
echo "========================================"

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is not installed. Please install Python 3.8+ first."
    exit 1
fi

# Check if Node.js is installed
if ! command -v node &> /dev/null; then
    echo "❌ Node.js is not installed. Please install Node.js 16+ first."
    exit 1
fi

# Check if npm is installed
if ! command -v npm &> /dev/null; then
    echo "❌ npm is not installed. Please install npm first."
    exit 1
fi

echo "✅ Prerequisites check passed"

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "📦 Creating Python virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "🔧 Activating Python virtual environment..."
source venv/bin/activate

# Install Python dependencies
echo "📥 Installing Python dependencies..."
pip install -r requirements.txt

# Install Node.js dependencies
echo "📥 Installing Node.js dependencies..."
npm install

# Create .env file for backend if it doesn't exist
if [ ! -f "server/.env" ]; then
    echo "📝 Creating backend environment file..."
    cat > server/.env << EOF
OPENAI_API_KEY=your_api_key_here
SERVER_URL=http://127.0.0.1:8000
DATA_DIR=./scraped_pages
EOF
    echo "⚠️  Please update server/.env with your actual OpenAI API key"
fi

# Create .env file for frontend if it doesn't exist
if [ ! -f ".env" ]; then
    echo "📝 Creating frontend environment file..."
    cat > .env << EOF
VITE_SERVER_URL=http://127.0.0.1:8000
EOF
fi

echo ""
echo "🎯 Setup complete! To run the application:"
echo ""
echo "1. Update server/.env with your OpenAI API key"
echo "2. Start the hybrid MCP + HTTP server:"
echo "   cd server && python3 mcp_server.py"
echo ""
echo "3. In a new terminal, start the frontend:"
echo "   npm run dev"
echo ""
echo "4. Open http://localhost:3000 in your browser"
echo ""
echo "⚠️  IMPORTANT: If you experience routing issues, restart the frontend:"
echo "   - Stop the frontend (Ctrl+C)"
echo "   - Run 'npm run dev' again"
echo ""
echo "🔧 Server Options:"
echo "   - Default: python3 mcp_server.py (starts both MCP + HTTP)"
echo "   - MCP only: python3 mcp_server.py --mcp-only (for AI integration)"
echo ""
echo "Happy scraping! 🕷️" 
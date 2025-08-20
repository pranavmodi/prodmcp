# MCP Web Scraper & QA System

A complete end-to-end application that scrapes web pages and provides AI-powered question answering capabilities.

## 🚀 Features

- **Web Scraping**: Scrape any website and save HTML content locally
- **Web Crawling**: Advanced website crawling to discover and scrape entire websites
- **AI-Powered QA**: Ask questions about scraped content using OpenAI GPT-4o-mini
- **Modern UI**: Clean, responsive React interface with Tailwind CSS
- **FastAPI Backend**: Robust Python backend with proper error handling
- **File Management**: Automatic organization of scraped content with timestamps
- **Intelligent Filtering**: Smart URL filtering to focus on relevant content

## 🏗️ Architecture

### Backend (FastAPI)
- **`main.py`**: FastAPI application with endpoints and CORS middleware
- **`scraper.py`**: Web scraping logic using requests + BeautifulSoup
- **`crawler.py`**: Advanced web crawling with session management and URL filtering
- **`qa.py`**: AI question answering using OpenAI API
- **`storage.py`**: HTML file management and storage

### Frontend (React + Tailwind)
- **Home Page**: URL input and scraping interface
- **Ask Page**: Question input with chat-like conversation
- **Components**: Reusable UI components with loading states
- **API Service**: Axios-based communication with backend

## 📋 Prerequisites

- Python 3.8+
- Node.js 16+
- OpenAI API key
- Modern web browser

## 🛠️ Installation

### 1. Clone the Repository
```bash
git clone <repository-url>
cd MCP_Server_Sample_QA
```

### 2. Backend Setup
```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Create environment file
cat > server/.env << EOF
OPENAI_API_KEY=your_actual_api_key_here
SERVER_URL=http://127.0.0.1:8000
DATA_DIR=./scraped_pages
EOF
```

### 3. Frontend Setup
```bash
# Install dependencies
npm install

# Create environment file
cat > .env << EOF
VITE_SERVER_URL=http://127.0.0.1:8000
EOF
```

## 🚀 Running the Application

### 1. Start the Backend Server
```bash
cd server
python main.py
```
The server will start on `http://127.0.0.1:8000`

### 2. Start the Frontend Client
```bash
# In a new terminal
npm run dev
```
The client will start on `http://localhost:3000`

## 📖 Usage

### Single Page Scraping
1. Navigate to the **Scrape** page
2. Enter a website URL (e.g., `https://example.com`)
3. Click "Scrape Website"
4. The HTML content is saved locally with timestamp

### Website Crawling & Scraping
1. Use the new `/crawl` API endpoint
2. Provide a base URL to start crawling from
3. The system will discover all pages and scrape them automatically
4. Content is organized by domain and page structure

### Asking Questions
1. Navigate to the **Ask** page
2. Type your question about the scraped content
3. Click "Ask" to get AI-generated answers
4. View conversation history and context information

## 🔧 Configuration

### Environment Variables

#### Backend (`.env` in server directory)
- `OPENAI_API_KEY`: Your OpenAI API key
- `SERVER_URL`: Server URL (default: http://127.0.0.1:8000)
- `DATA_DIR`: Directory for storing HTML files (default: ./scraped_pages)

#### Frontend (`.env` in root directory)
- `VITE_SERVER_URL`: Backend server URL (default: http://127.0.0.1:8000)

### Customization
- Modify `tailwind.config.js` for custom styling
- Update CORS origins in `server/main.py` for production
- Adjust OpenAI model in `server/qa.py`

## 📁 File Structure

```
MCP_Server_Sample_QA/
├── server/
│   ├── __init__.py
│   ├── main.py          # FastAPI application
│   ├── scraper.py       # Web scraping logic
│   ├── qa.py           # AI question answering
│   ├── storage.py      # File management
│   └── .env            # Backend environment
├── src/
│   ├── components/
│   │   ├── Navbar.jsx
│   │   └── Loader.jsx
│   ├── pages/
│   │   ├── Home.jsx    # Scraping interface
│   │   └── Ask.jsx     # QA interface
│   ├── services/
│   │   └── api.js      # API communication
│   ├── App.jsx
│   ├── main.jsx
│   └── index.css
├── scraped_pages/      # Generated HTML files
├── requirements.txt     # Python dependencies
├── package.json        # Node.js dependencies
├── tailwind.config.js  # Tailwind configuration
└── README.md
```

## 🧪 Testing

### Backend Testing
```bash
cd server
python -m pytest  # If you add tests
```

### Frontend Testing
```bash
npm test  # If you add tests
```

## 🚀 Production Deployment

### Backend
- Use production WSGI server (Gunicorn)
- Set proper CORS origins
- Use environment variables for configuration
- Consider using Redis for caching

### Frontend
- Build optimized version: `npm run build`
- Serve static files from Nginx/Apache
- Update API endpoints for production

## 🔒 Security Considerations

- Never commit API keys to version control
- Use HTTPS in production
- Implement rate limiting for scraping endpoints
- Validate and sanitize user inputs
- Consider implementing user authentication

## 🐛 Troubleshooting

### Common Issues

1. **Server Connection Failed**
   - Check if backend is running on correct port
   - Verify CORS configuration
   - Check firewall settings

2. **OpenAI API Errors**
   - Verify API key is correct
   - Check API quota and billing
   - Ensure proper model access

3. **Scraping Failures**
   - Check website accessibility
   - Verify URL format
   - Check network connectivity

### Debug Mode
Enable debug logging in `server/main.py`:
```python
logging.basicConfig(level=logging.DEBUG)
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- FastAPI for the excellent Python web framework
- React and Tailwind CSS for the modern UI
- OpenAI for the AI capabilities
- BeautifulSoup for HTML parsing

## 📞 Support

For issues and questions:
1. Check the troubleshooting section
2. Review existing GitHub issues
3. Create a new issue with detailed information

---

**Happy Scraping and Question Answering! 🎉** 
#!/usr/bin/env python3
"""
Hybrid MCP Server for Web Scraping and QA
Implements both Model Context Protocol and HTTP REST API
"""

import asyncio
import json
import logging
from typing import Any, Dict, List, Optional
from pathlib import Path
import sys

# Import our existing functionality
from storage import HTMLStorage
from scraper import WebScraper
from qa import QASystem
import os
from dotenv import load_dotenv

# Import FastAPI for HTTP endpoints
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import uvicorn

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Pydantic models for HTTP API
class ScrapeRequest(BaseModel):
    url: str

class ScrapeResponse(BaseModel):
    status: str
    file: str
    message: str
    page_info: dict = None

class CrawlRequest(BaseModel):
    url: str
    max_pages: int = 50
    delay_range: tuple = (3, 7)

class CrawlResponse(BaseModel):
    status: str
    message: str
    scraped_files: List[str] = []
    failed_urls: List[str] = []
    crawl_stats: dict = None

class QuestionRequest(BaseModel):
    question: str

class QuestionResponse(BaseModel):
    answer: str
    context_info: dict = None

class ErrorResponse(BaseModel):
    error: str
    detail: str = None

# Initialize FastAPI app
app = FastAPI(
    title="MCP Web Scraper & QA Server",
    description="A hybrid server that provides both MCP protocol and HTTP REST API",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class MCPServer:
    """Traditional MCP Server implementing the MCP protocol"""
    
    def __init__(self):
        # Initialize our existing components
        self.storage = HTMLStorage(os.getenv("DATA_DIR", "./scraped_pages"))
        self.scraper = WebScraper()  # This now includes the crawler!
        
        # Initialize OpenAI QA system
        openai_api_key = os.getenv("OPENAI_API_KEY")
        if not openai_api_key:
            logger.warning("OPENAI_API_KEY not found")
            self.qa_system = None
        else:
            self.qa_system = QASystem(openai_api_key)
        
        # MCP tool definitions - Updated with new crawling tools
        self.tools = [
            {
                "name": "scrape_website",
                "description": "Scrape a single website page and save the HTML content for later analysis",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "url": {
                            "type": "string",
                            "description": "The URL of the website page to scrape"
                        }
                    },
                    "required": ["url"]
                }
            },
            {
                "name": "crawl_website",
                "description": "Crawl an entire website to discover and scrape ALL pages automatically",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "url": {
                            "type": "string",
                            "description": "The starting URL of the website to crawl"
                        },
                        "max_pages": {
                            "type": "integer",
                            "description": "Maximum number of pages to crawl (default: 50, max: 1000)",
                            "default": 50
                        },
                        "delay_range": {
                            "type": "array",
                            "description": "Delay range in seconds between scrapes [min, max] (default: [3, 7])",
                            "items": {"type": "number"},
                            "default": [3, 7]
                        }
                    },
                    "required": ["url"]
                }
            },
            {
                "name": "get_crawl_stats",
                "description": "Get statistics about the website crawling operation",
                "inputSchema": {
                    "type": "object",
                    "properties": {}
                }
            },
            {
                "name": "get_discovered_urls",
                "description": "Get all discovered URLs from crawling operations",
                "inputSchema": {
                    "type": "object",
                    "properties": {}
                }
            },
            {
                "name": "clear_crawl_data",
                "description": "Clear all crawl data and start fresh",
                "inputSchema": {
                    "type": "object",
                    "properties": {}
                }
            },
            {
                "name": "ask_question",
                "description": "Ask a question about the scraped website content using AI",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "question": {
                            "type": "string",
                            "description": "The question to ask about the scraped content"
                        }
                    },
                    "required": ["question"]
                }
            },
            {
                "name": "list_scraped_files",
                "description": "List all currently scraped website files",
                "inputSchema": {
                    "type": "object",
                    "properties": {}
                }
            },
            {
                "name": "get_server_status",
                "description": "Get the current status of the MCP server and storage",
                "inputSchema": {
                    "type": "object",
                    "properties": {}
                }
            }
        ]
    
    async def handle_initialize(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle MCP initialization"""
        return {
            "protocolVersion": "2024-11-05",
            "capabilities": {
                "tools": {}
            },
            "serverInfo": {
                "name": "MCP Web Scraper & QA Server with Crawler",
                "version": "2.0.0"
            }
        }
    
    async def handle_tools_list(self) -> Dict[str, Any]:
        """Handle tools/list request"""
        return {
            "tools": self.tools
        }
    
    async def handle_tools_call(self, name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Handle tools/call request"""
        try:
            if name == "scrape_website":
                return await self._scrape_website(arguments)
            elif name == "crawl_website":
                return await self._crawl_website(arguments)
            elif name == "get_crawl_stats":
                return await self._get_crawl_stats()
            elif name == "get_discovered_urls":
                return await self._get_discovered_urls()
            elif name == "clear_crawl_data":
                return await self._clear_crawl_data()
            elif name == "ask_question":
                return await self._ask_question(arguments)
            elif name == "list_scraped_files":
                return await self._list_scraped_files()
            elif name == "get_server_status":
                return await self._get_server_status()
            else:
                raise ValueError(f"Unknown tool: {name}")
        except Exception as e:
            logger.error(f"Error in tool call {name}: {e}")
            return {
                "isError": True,
                "error": {
                    "code": -32603,
                    "message": str(e)
                }
            }
    
    async def _scrape_website(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Scrape a single website page using our existing scraper"""
        url = arguments.get("url")
        if not url:
            raise ValueError("URL is required")
        
        # Use our existing scraper
        html_content, domain = self.scraper.scrape_url(url)
        filename = self.storage.save_html(domain, html_content)
        page_info = self.scraper.get_page_info(html_content)
        
        return {
            "content": [
                {
                    "type": "text",
                    "text": f"Successfully scraped single page {url} and saved as {filename}"
                }
            ],
            "isError": False,
            "toolCallId": "scrape_result",
            "metadata": {
                "filename": filename,
                "domain": domain,
                "page_info": page_info,
                "type": "single_page"
            }
        }
    
    async def _crawl_website(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Crawl an entire website to discover and scrape all pages"""
        url = arguments.get("url")
        if not url:
            raise ValueError("URL is required")
        
        max_pages = arguments.get("max_pages", 50)
        delay_range = arguments.get("delay_range", [3, 7])
        
        # Validate parameters
        if max_pages < 1 or max_pages > 1000:
            raise ValueError("max_pages must be between 1 and 1000")
        
        if len(delay_range) != 2 or delay_range[0] < 0 or delay_range[1] < delay_range[0]:
            raise ValueError("delay_range must be [min_delay, max_delay] with min_delay <= max_delay")
        
        # Use the new integrated crawling and scraping functionality
        result = self.scraper.crawl_and_scrape_website(
            base_url=url,
            max_pages=max_pages,
            delay_range=tuple(delay_range)
        )
        
        # Get crawl statistics
        crawl_stats = self.scraper.get_crawl_stats()
        
        return {
            "content": [
                {
                    "type": "text",
                    "text": f"Successfully crawled and scraped entire website {url}\n" +
                           f"📁 Scraped files: {len(result['scraped_files'])}\n" +
                           f"❌ Failed URLs: {len(result['failed_urls'])}\n" +
                           f"📊 Total discovered URLs: {crawl_stats['total_urls']}"
                }
            ],
            "isError": False,
            "toolCallId": "crawl_result",
            "metadata": {
                "base_url": url,
                "scraped_files": result["scraped_files"],
                "failed_urls": result["failed_urls"],
                "crawl_stats": crawl_stats,
                "type": "full_website_crawl"
            }
        }
    
    async def _get_crawl_stats(self) -> Dict[str, Any]:
        """Get crawling statistics"""
        try:
            stats = self.scraper.get_crawl_stats()
            
            return {
                "content": [
                    {
                        "type": "text",
                        "text": f"📊 Crawl Statistics:\n" +
                               f"  Total accepted URLs: {stats['total_accepted']}\n" +
                               f"  Total rejected URLs: {stats['total_rejected']}\n" +
                               f"  Total URLs: {stats['total_urls']}\n" +
                               f"  Data files: {stats['accepted_file']}, {stats['rejected_file']}"
                    }
                ],
                "isError": False,
                "toolCallId": "crawl_stats",
                "metadata": stats
            }
        except Exception as e:
            raise ValueError(f"Failed to get crawl stats: {e}")
    
    async def _get_discovered_urls(self) -> Dict[str, Any]:
        """Get all discovered URLs"""
        try:
            accepted_urls, rejected_urls = self.scraper.get_discovered_urls()
            
            return {
                "content": [
                    {
                        "type": "text",
                        "text": f"🔍 Discovered URLs:\n" +
                               f"  ✅ Accepted ({len(accepted_urls)}):\n" +
                               "\n".join([f"    • {url}" for url in sorted(accepted_urls)[:10]]) +
                               (f"\n    ... and {len(accepted_urls) - 10} more" if len(accepted_urls) > 10 else "") +
                               f"\n\n  ❌ Rejected ({len(rejected_urls)}):\n" +
                               "\n".join([f"    • {url}" for url in sorted(rejected_urls)[:10]]) +
                               (f"\n    ... and {len(rejected_urls) - 10} more" if len(rejected_urls) > 10 else "")
                    }
                ],
                "isError": False,
                "toolCallId": "discovered_urls",
                "metadata": {
                    "accepted_urls": list(accepted_urls),
                    "rejected_urls": list(rejected_urls),
                    "total_accepted": len(accepted_urls),
                    "total_rejected": len(rejected_urls)
                }
            }
        except Exception as e:
            raise ValueError(f"Failed to get discovered URLs: {e}")
    
    async def _clear_crawl_data(self) -> Dict[str, Any]:
        """Clear all crawl data"""
        try:
            self.scraper.clear_crawl_data()
            
            return {
                "content": [
                    {
                        "type": "text",
                        "text": "🧹 All crawl data cleared successfully. You can start fresh crawling operations."
                    }
                ],
                "isError": False,
                "toolCallId": "clear_crawl_data",
                "metadata": {
                    "message": "Crawl data cleared",
                    "timestamp": asyncio.get_event_loop().time()
                }
            }
        except Exception as e:
            raise ValueError(f"Failed to clear crawl data: {e}")
    
    async def _ask_question(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Ask a question using our existing QA system"""
        if not self.qa_system:
            raise ValueError("OpenAI API not configured")
        
        question = arguments.get("question")
        if not question:
            raise ValueError("Question is required")
        
        # Get all HTML files
        html_files = self.storage.get_all_html_files()
        if not html_files:
            raise ValueError("No scraped pages found. Please scrape some websites first.")
        
        # Build context and ask question
        file_paths = [str(f) for f in html_files]
        context = self.qa_system.build_context_from_files(file_paths)
        answer = self.qa_system.ask_question(question, context)
        
        return {
            "content": [
                {
                    "type": "text",
                    "text": answer
                }
            ],
            "isError": False,
            "toolCallId": "qa_result",
            "metadata": {
                "files_used": len(html_files),
                "question": question
            }
        }
    
    async def _list_scraped_files(self) -> Dict[str, Any]:
        """List all scraped files"""
        storage_info = self.storage.get_storage_info()
        
        return {
            "content": [
                {
                    "type": "text",
                    "text": f"📁 Found {storage_info['total_files']} scraped files:\n" + 
                           "\n".join([f"• {f}" for f in storage_info['files']])
                }
            ],
            "isError": False,
            "toolCallId": "files_list",
            "metadata": storage_info
        }
    
    async def _get_server_status(self) -> Dict[str, Any]:
        """Get server status"""
        storage_info = self.storage.get_storage_info()
        
        # Get crawl stats if available
        try:
            crawl_stats = self.scraper.get_crawl_stats()
            crawl_info = f"\n🕷️  Crawler: {crawl_stats['total_urls']} URLs discovered"
        except:
            crawl_info = "\n🕷️  Crawler: Not initialized"
        
        return {
            "content": [
                {
                    "type": "text",
                    "text": f"🚀 Server Status: Healthy\n" +
                           f"📁 Storage: {storage_info['total_files']} files" +
                           crawl_info +
                           f"\n🤖 OpenAI Configured: {self.qa_system is not None}"
                }
            ],
            "isError": False,
            "toolCallId": "status_result",
            "metadata": {
                "status": "healthy",
                "storage": storage_info,
                "openai_configured": self.qa_system is not None,
                "crawl_stats": crawl_stats if 'crawl_stats' in locals() else None
            }
        }

# Global exception handler for HTTP API
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Global exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={"error": "Internal server error", "detail": str(exc)}
    )

# HTTP Endpoints - Same as main.py but integrated with MCP server
@app.get("/")
async def root():
    """Root endpoint with server information."""
    return {
        "message": "MCP Web Scraper & QA Server (Hybrid MCP + HTTP)",
        "version": "1.0.0",
        "endpoints": ["/scrape", "/crawl", "/ask", "/status", "/files"],
        "mcp_enabled": True,
        "status": "running"
    }

@app.get("/status")
async def get_status():
    """Get server status and storage information."""
    try:
        # Create a temporary server instance to get status
        temp_server = MCPServer()
        storage_info = temp_server.storage.get_storage_info()
        crawl_stats = temp_server.scraper.get_crawl_stats()
        
        return {
            "status": "healthy",
            "storage": storage_info,
            "openai_configured": temp_server.qa_system is not None,
            "mcp_enabled": True,
            "crawl_stats": crawl_stats
        }
    except Exception as e:
        logger.error(f"Error getting status: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/scrape", response_model=ScrapeResponse)
async def scrape_website(request: ScrapeRequest):
    """Scrape a website and save the HTML content."""
    try:
        logger.info(f"Scraping request received for URL: {request.url}")
        
        # Validate URL
        if not request.url or not request.url.strip():
            raise HTTPException(status_code=400, detail="URL is required")
        
        # Create a temporary server instance
        temp_server = MCPServer()
        
        # Scrape the website
        html_content, domain = temp_server.scraper.scrape_url(request.url.strip())
        
        # Save the HTML content
        filename = temp_server.storage.save_html(domain, html_content)
        
        # Get page information
        page_info = temp_server.scraper.get_page_info(html_content)
        
        logger.info(f"Successfully scraped and saved: {filename}")
        
        return ScrapeResponse(
            status="success",
            file=filename,
            message=f"Successfully scraped {request.url}",
            page_info=page_info
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error scraping {request.url}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to scrape website: {str(e)}")

@app.post("/crawl", response_model=CrawlResponse)
async def crawl_website(request: CrawlRequest):
    """Crawl a website to discover URLs and then scrape all discovered pages."""
    try:
        logger.info(f"Crawl request received for URL: {request.url}")
        
        # Validate URL
        if not request.url or not request.url.strip():
            raise HTTPException(status_code=400, detail="URL is required")
        
        # Validate parameters
        if request.max_pages < 1 or request.max_pages > 1000:
            raise HTTPException(status_code=400, detail="max_pages must be between 1 and 1000")
        
        if len(request.delay_range) != 2 or request.delay_range[0] < 0 or request.delay_range[1] < request.delay_range[0]:
            raise HTTPException(status_code=400, detail="delay_range must be a tuple of (min_delay, max_delay) with min_delay <= max_delay")
        
        logger.info(f"Starting crawl with max_pages={request.max_pages}, delay_range={request.delay_range}")
        
        # Create a temporary server instance
        temp_server = MCPServer()
        
        # Perform crawl and scrape operation
        result = temp_server.scraper.crawl_and_scrape_website(
            base_url=request.url.strip(),
            max_pages=request.max_pages,
            delay_range=request.delay_range
        )
        
        # Get crawl statistics
        crawl_stats = temp_server.scraper.get_crawl_stats()
        
        # Get discovered URLs for debugging
        accepted_urls, rejected_urls = temp_server.scraper.get_discovered_urls()
        
        logger.info(f"Crawl completed. Result: {result}")
        logger.info(f"Accepted URLs: {len(accepted_urls)}")
        logger.info(f"Rejected URLs: {len(rejected_urls)}")
        logger.info(f"Scraped files: {len(result['scraped_files'])}")
        logger.info(f"Failed URLs: {len(result['failed_urls'])}")
        
        return CrawlResponse(
            status="success",
            message=f"Successfully crawled and scraped {request.url}",
            scraped_files=result["scraped_files"],
            failed_urls=result["failed_urls"],
            crawl_stats=crawl_stats
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error crawling {request.url}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to crawl website: {str(e)}")

@app.post("/ask", response_model=QuestionResponse)
async def ask_question(request: QuestionRequest):
    """Ask a question about the scraped content using AI."""
    try:
        # Create a temporary server instance
        temp_server = MCPServer()
        
        if not temp_server.qa_system:
            raise HTTPException(
                status_code=500, 
                detail="OpenAI API not configured. Please set OPENAI_API_KEY environment variable."
            )
        
        if not request.question or not request.question.strip():
            raise HTTPException(status_code=400, detail="Question is required")
        
        logger.info(f"Question received: {request.question[:50]}...")
        
        # Get all HTML files
        html_files = temp_server.storage.get_all_html_files()
        
        if not html_files:
            raise HTTPException(
                status_code=400, 
                detail="No scraped pages found. Please scrape some websites first."
            )
        
        # Build context from HTML files
        file_paths = [str(f) for f in html_files]
        context = temp_server.qa_system.build_context_from_files(file_paths)
        
        if not context.strip():
            raise HTTPException(
                status_code=500, 
                detail="Failed to extract meaningful content from scraped pages."
            )
        
        # Ask the question
        answer = temp_server.qa_system.ask_question(request.question.strip(), context)
        
        # Get context information
        context_info = temp_server.qa_system.get_available_files_summary(file_paths)
        
        logger.info(f"Question answered successfully")
        
        return QuestionResponse(
            answer=answer,
            context_info=context_info
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing question: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to process question: {str(e)}")

@app.get("/files")
async def list_files():
    """List all scraped HTML files."""
    try:
        # Create a temporary server instance
        temp_server = MCPServer()
        storage_info = temp_server.storage.get_storage_info()
        return storage_info
    except Exception as e:
        logger.error(f"Error listing files: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/crawl/stats")
async def get_crawl_stats():
    """Get statistics about the crawling operation."""
    try:
        temp_server = MCPServer()
        stats = temp_server.scraper.get_crawl_stats()
        return {
            "status": "success",
            "crawl_stats": stats
        }
    except Exception as e:
        logger.error(f"Error getting crawl stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/crawl/urls")
async def get_discovered_urls():
    """Get the currently discovered accepted and rejected URLs."""
    try:
        temp_server = MCPServer()
        accepted_urls, rejected_urls = temp_server.scraper.get_discovered_urls()
        return {
            "status": "success",
            "accepted_urls": list(accepted_urls),
            "rejected_urls": list(rejected_urls),
            "total_accepted": len(accepted_urls),
            "total_rejected": len(rejected_urls)
        }
    except Exception as e:
        logger.error(f"Error getting discovered URLs: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/crawl/clear")
async def clear_crawl_data():
    """Clear all crawl data and start fresh."""
    try:
        temp_server = MCPServer()
        temp_server.scraper.clear_crawl_data()
        return {
            "status": "success",
            "message": "All crawl data cleared successfully"
        }
    except Exception as e:
        logger.error(f"Error clearing crawl data: {e}")
        raise HTTPException(status_code=500, detail=str(e))

class MCPTransport:
    """Handles MCP protocol communication over stdio"""
    
    def __init__(self, server: MCPServer):
        self.server = server
        self.request_id = 0
    
    async def run(self):
        """Run the MCP server over stdio"""
        logger.info("Starting MCP server over stdio")
        
        while True:
            try:
                # Read request from stdin
                line = await asyncio.get_event_loop().run_in_executor(None, sys.stdin.readline)
                if not line:
                    break
                
                request = json.loads(line.strip())
                response = await self._handle_request(request)
                
                # Send response to stdout
                print(json.dumps(response), flush=True)
                
            except json.JSONDecodeError as e:
                logger.error(f"Invalid JSON: {e}")
                error_response = self._create_error_response(None, -32700, "Parse error")
                print(json.dumps(error_response), flush=True)
            except Exception as e:
                logger.error(f"Unexpected error: {e}")
                error_response = self._create_error_response(None, -32603, "Internal error")
                print(json.dumps(error_response), flush=True)
    
    async def _handle_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Handle incoming MCP request"""
        method = request.get("method")
        request_id = request.get("id")
        params = request.get("params", {})
        
        try:
            if method == "initialize":
                result = await self.server.handle_initialize(params)
            elif method == "tools/list":
                result = await self.server.handle_tools_list()
            elif method == "tools/call":
                name = params.get("name")
                arguments = params.get("arguments", {})
                result = await self.server.handle_tools_call(name, arguments)
            else:
                return self._create_error_response(request_id, -32601, "Method not found")
            
            return {
                "jsonrpc": "2.0",
                "id": request_id,
                "result": result
            }
            
        except Exception as e:
            logger.error(f"Error handling request {method}: {e}")
            return self._create_error_response(request_id, -32603, str(e))
    
    def _create_error_response(self, request_id: Any, code: int, message: str) -> Dict[str, Any]:
        """Create error response"""
        return {
            "jsonrpc": "2.0",
            "id": request_id,
            "error": {
                "code": code,
                "message": message
            }
        }

async def main():
    """Main entry point - starts both MCP and HTTP servers"""
    logger.info("🚀 Starting Hybrid MCP + HTTP Server")
    logger.info("📡 MCP Protocol: Available via stdio")
    logger.info("🌐 HTTP API: Available on port 8000")
    
    # Start the HTTP server
    config = uvicorn.Config(
        app=app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )
    server = uvicorn.Server(config)
    
    # Run the HTTP server
    await server.serve()

async def start_mcp_only():
    """Start only the MCP server over stdio (for AI integration)"""
    logger.info("📡 Starting MCP Server (stdio only)")
    server = MCPServer()
    transport = MCPTransport(server)
    await transport.run()

if __name__ == "__main__":
    # Check command line arguments
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "--mcp-only":
        # Start only MCP server
        asyncio.run(start_mcp_only())
    else:
        # Start hybrid server (default)
        asyncio.run(main()) 
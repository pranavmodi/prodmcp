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
from scraper import ModernScraper
from qa import QASystem
import os
from dotenv import load_dotenv

# Import FastAPI for HTTP endpoints
from fastapi import FastAPI, HTTPException, Request, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import uvicorn
from mcp_client import MCPStdIoClient, spawn_mcp_server
import mimetypes
import shutil
from datetime import datetime
from db import create_crawl_job, set_crawl_status
import uuid

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Per-tenant crawl job state
_crawl_jobs: Dict[str, dict] = {}
_crawl_tasks: Dict[str, asyncio.Task] = {}

def _empty_job_state() -> dict:
    return {
        "status": "idle",  # idle | crawl | scrape | done | error
        "message": None,
        "crawl": {"accepted": 0, "rejected": 0, "discovered": 0, "percent": 0},
        "scrape": {"total": 0, "completed": 0, "percent": 0},
        "started_at": None,
        "updated_at": None,
    }

# Pydantic models for HTTP API
class CrawlRequest(BaseModel):
    tenant_id: Optional[str] = None
    url: str
    # Optional list of URL substrings or full URLs to exclude from crawling/scraping
    exclusions: Optional[List[str]] = None

class CrawlResponse(BaseModel):
    status: str
    message: str
    scraped_files: List[str] = []
    failed_urls: List[str] = []
    crawl_stats: dict = None
    job_id: Optional[int] = None
    tenant_id: Optional[str] = None

class QuestionRequest(BaseModel):
    tenant_id: Optional[str] = None
    question: str

class QuestionResponse(BaseModel):
    answer: str
    context_info: Optional[dict] = None
    mcp_debug: Optional[dict] = None

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
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000", "http://localhost:9000", "http://127.0.0.1:9000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class MCPServer:
    """Traditional MCP Server implementing the MCP protocol"""
    
    def __init__(self):
        # Initialize our existing components
        self.storage = HTMLStorage(os.getenv("DATA_DIR", "./scraped_pages"))
        self.scraper = ModernScraper(os.getenv("DATA_DIR", "./scraped_pages"))
        
        # Initialize OpenAI QA system
        openai_api_key = os.getenv("OPENAI_API_KEY")
        if not openai_api_key:
            logger.warning("OPENAI_API_KEY not found")
            self.qa_system = None
        else:
            self.qa_system = QASystem(openai_api_key)
        
        # MCP tool definitions - expose only lookup_kb as an MCP tool
        self.tools = [
            {
                "name": "lookup_kb",
                "description": "Answer a question using the knowledge base built from scraped JSON (RAG-like)",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "The user's question"
                        },
                        "k": {
                            "type": "integer",
                            "description": "Top documents to retrieve",
                            "default": 5
                        }
                    },
                    "required": ["query"]
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
            if name == "lookup_kb":
                return await self._lookup_kb(arguments)
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
        
        # Use async modern scraper
        result = await self.scraper.scrape_url(url)
        if not result:
            raise ValueError("Failed to scrape content")
        await self.scraper.save_content(result, url)
        
        return {
            "content": [
                {"type": "text", "text": f"Successfully scraped single page {url}"}
            ],
            "isError": False,
            "toolCallId": "scrape_result",
            "metadata": {
                "result": result,
                "type": "single_page_json"
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
        
        # Use file-based crawler and then batch scrape discovered URLs
        from crawler import crawl_website, load_existing_urls

        data_dir = os.getenv("DATA_DIR", "./scraped_pages")
        accepted_file = os.path.join(data_dir, "accepted_urls.txt")
        rejected_file = os.path.join(data_dir, "rejected_urls.txt")

        existing_accepted = load_existing_urls(accepted_file)
        existing_rejected = load_existing_urls(rejected_file)

        crawl_website(
            base_url=url.strip(),
            accepted_file=accepted_file,
            rejected_file=rejected_file,
            existing_accepted=existing_accepted,
            existing_rejected=existing_rejected
        )

        all_accepted = load_existing_urls(accepted_file)
        scraped_files = []
        failed_urls = []
        if all_accepted:
            try:
                # Prioritize root/base URL first for better seeding
                sorted_urls = sorted(all_accepted, key=lambda u: len(u))
                results = await self.scraper.scrape_urls_batch(sorted_urls)
                scraped_files = [r.get("url") for r in results if r]
                failed_urls = [u for u in sorted_urls if u not in scraped_files]
            except Exception as e:
                logger.error(f"Batch scrape error: {e}")
                failed_urls = list(all_accepted)

        crawl_stats = {
            "total_accepted": len(load_existing_urls(accepted_file)),
            "total_rejected": len(load_existing_urls(rejected_file)),
            "total_urls": len(load_existing_urls(accepted_file)) + len(load_existing_urls(rejected_file)),
            "accepted_file": accepted_file,
            "rejected_file": rejected_file
        }
        
        return {
            "content": [
                {
                    "type": "text",
                    "text": f"Successfully crawled and scraped entire website {url}\n" +
                           f"📁 Scraped files: {len(scraped_files)}\n" +
                           f"❌ Failed URLs: {len(failed_urls)}\n" +
                           f"📊 Total discovered URLs: {crawl_stats['total_urls']}"
                }
            ],
            "isError": False,
            "toolCallId": "crawl_result",
            "metadata": {
                "base_url": url,
                "scraped_files": scraped_files,
                "failed_urls": failed_urls,
                "crawl_stats": crawl_stats,
                "type": "full_website_crawl"
            }
        }
    
    async def _get_crawl_stats(self) -> Dict[str, Any]:
        """Get crawling statistics"""
        try:
            data_dir = os.getenv("DATA_DIR", "./scraped_pages")
            accepted_file = os.path.join(data_dir, "accepted_urls.txt")
            rejected_file = os.path.join(data_dir, "rejected_urls.txt")
            from crawler import load_existing_urls
            stats = {
                "total_accepted": len(load_existing_urls(accepted_file)),
                "total_rejected": len(load_existing_urls(rejected_file)),
                "total_urls": len(load_existing_urls(accepted_file)) + len(load_existing_urls(rejected_file)),
                "accepted_file": accepted_file,
                "rejected_file": rejected_file
            }
            
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
            data_dir = os.getenv("DATA_DIR", "./scraped_pages")
            accepted_file = os.path.join(data_dir, "accepted_urls.txt")
            rejected_file = os.path.join(data_dir, "rejected_urls.txt")
            from crawler import load_existing_urls
            accepted_urls = load_existing_urls(accepted_file)
            rejected_urls = load_existing_urls(rejected_file)
            
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
            data_dir = os.getenv("DATA_DIR", "./scraped_pages")
            for name in ["accepted_urls.txt", "rejected_urls.txt", "crawler_cookies.txt"]:
                path = os.path.join(data_dir, name)
                try:
                    if os.path.exists(path):
                        os.remove(path)
                except Exception:
                    pass
            
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

    async def _lookup_kb(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Lookup knowledge base from local scraped JSON using a minimal RAG approach."""
        query = arguments.get("query")
        if not query:
            raise ValueError("query is required")
        k = int(arguments.get("k", 5))
        from kb import lookup_kb_minimal
        result = lookup_kb_minimal(query, k=k)
        text = result.get("answer", "")
        return {
            "content": [
                {"type": "text", "text": text[:4000]}
            ],
            "isError": False,
            "toolCallId": "lookup_kb_result",
            "metadata": result
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
        "endpoints": ["/crawl", "/ask", "/status", "/files", "/upload"],
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

# Removed single-page /scrape endpoint per new requirements

@app.post("/crawl", response_model=CrawlResponse)
async def crawl_website(request: CrawlRequest):
    """Start a crawl job and return immediately; progress can be polled via /crawl/progress."""
    try:
        logger.info(f"Crawl request received for URL: {request.url}")

        if not request.url or not request.url.strip():
            raise HTTPException(status_code=400, detail="URL is required")
        # Generate random tenant_id if not provided or set to 'default'
        tenant_raw = (request.tenant_id or "").strip()
        tenant_id = uuid.uuid4().hex if (not tenant_raw or tenant_raw.lower() == "default") else tenant_raw

        # Per-tenant job state
        if tenant_id not in _crawl_jobs:
            _crawl_jobs[tenant_id] = _empty_job_state()
        if tenant_id in _crawl_tasks and _crawl_tasks[tenant_id] and not _crawl_tasks[tenant_id].done():
            raise HTTPException(status_code=409, detail="A crawl is already in progress for this tenant")

        # Reset tenant job state
        _crawl_jobs[tenant_id].update({
            "status": "crawl",
            "message": None,
            "crawl": {"accepted": 0, "rejected": 0, "discovered": 0, "percent": 0},
            "scrape": {"total": 0, "completed": 0, "percent": 0},
            "started_at": asyncio.get_event_loop().time(),
            "updated_at": asyncio.get_event_loop().time(),
        })

        async def _run_job(base_url: str, exclusions: Optional[List[str]], tenant_id_local: str, job_id_local: Optional[int]):
            try:
                from crawler import crawl_website as crawl_urls, load_existing_urls
                base_dir = os.getenv("DATA_DIR", "./scraped_pages")
                data_dir = os.path.join(base_dir, tenant_id_local)
                os.makedirs(data_dir, exist_ok=True)
                accepted_file = os.path.join(data_dir, "accepted_urls.txt")
                rejected_file = os.path.join(data_dir, "rejected_urls.txt")

                existing_accepted = load_existing_urls(accepted_file)
                existing_rejected = load_existing_urls(rejected_file)

                # Crawl in a worker thread and stream progress back to event loop
                loop = asyncio.get_running_loop()

                def _crawl_progress(visited, queue, accepted, rejected, percent):
                    def _update():
                        _crawl_jobs[tenant_id_local]["crawl"].update({
                            "accepted": int(accepted),
                            "rejected": int(rejected),
                            "discovered": int(accepted) + int(rejected),
                            "percent": int(percent),
                        })
                        _crawl_jobs[tenant_id_local]["updated_at"] = loop.time()
                    try:
                        loop.call_soon_threadsafe(_update)
                    except Exception:
                        pass

                await asyncio.to_thread(
                    crawl_urls,
                    base_url,
                    accepted_file,
                    rejected_file,
                    existing_accepted,
                    existing_rejected,
                    (exclusions or []),
                    _crawl_progress,
                    data_dir,
                )

                # Update crawl progress
                all_accepted = load_existing_urls(accepted_file)
                _crawl_jobs[tenant_id_local]["crawl"]["accepted"] = len(all_accepted)
                _crawl_jobs[tenant_id_local]["crawl"]["rejected"] = len(load_existing_urls(rejected_file))
                _crawl_jobs[tenant_id_local]["crawl"]["discovered"] = _crawl_jobs[tenant_id_local]["crawl"]["accepted"] + _crawl_jobs[tenant_id_local]["crawl"]["rejected"]
                _crawl_jobs[tenant_id_local]["crawl"]["percent"] = 100
                _crawl_jobs[tenant_id_local]["status"] = "scrape"
                _crawl_jobs[tenant_id_local]["updated_at"] = asyncio.get_running_loop().time()

                # Scrape with progress callback (run in background thread to avoid blocking event loop)
                sorted_urls = sorted(all_accepted)
                total = len(sorted_urls)
                _crawl_jobs[tenant_id_local]["scrape"]["total"] = total
                _crawl_jobs[tenant_id_local]["scrape"]["completed"] = 0
                _crawl_jobs[tenant_id_local]["scrape"]["percent"] = 0 if total == 0 else 0

                loop = asyncio.get_running_loop()

                async def _progress_callback_async(done: int, total_cb: int):
                    def _update():
                        _crawl_jobs[tenant_id_local]["scrape"]["completed"] = done
                        _crawl_jobs[tenant_id_local]["scrape"]["percent"] = int(done * 100 / max(total_cb, 1))
                        _crawl_jobs[tenant_id_local]["updated_at"] = loop.time()
                    try:
                        loop.call_soon_threadsafe(_update)
                    except Exception:
                        pass

                def _run_scrape_blocking():
                    import asyncio as _asyncio
                    from scraper import ModernScraper
                    scraper = ModernScraper(output_dir=data_dir)
                    _asyncio.run(scraper.scrape_urls_batch(sorted_urls, progress_callback=_progress_callback_async))

                if total > 0:
                    await asyncio.to_thread(_run_scrape_blocking)

                _crawl_jobs[tenant_id_local]["status"] = "done"
                _crawl_jobs[tenant_id_local]["message"] = f"Crawled and scraped {base_url}"
                _crawl_jobs[tenant_id_local]["updated_at"] = asyncio.get_running_loop().time()
                try:
                    set_crawl_status(int(job_id_local or 0), "done", f"Crawled and scraped {base_url}", finished=True)
                except Exception:
                    pass
            except Exception as e:
                _crawl_jobs[tenant_id_local]["status"] = "error"
                _crawl_jobs[tenant_id_local]["message"] = str(e)
                _crawl_jobs[tenant_id_local]["updated_at"] = asyncio.get_running_loop().time()
                try:
                    set_crawl_status(int(job_id_local or 0), "error", str(e), finished=True)
                except Exception:
                    pass

        job_id: Optional[int] = None
        try:
            job_id = create_crawl_job(tenant_id=tenant_id, url=request.url.strip(), status="started")
        except Exception as e:
            logger.warning(f"DB: failed to create crawl job: {e}")

        _crawl_tasks[tenant_id] = asyncio.create_task(_run_job(request.url.strip(), (request.exclusions or []), tenant_id, job_id))

        return CrawlResponse(
            status="started",
            message="Crawl job started",
            scraped_files=[],
            failed_urls=[],
            crawl_stats=_crawl_jobs[tenant_id],
            job_id=job_id,
            tenant_id=tenant_id,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error starting crawl {request.url}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to start crawl: {str(e)}")

@app.post("/ask", response_model=QuestionResponse)
async def ask_question(request: QuestionRequest):
    """Ask a question about the scraped content using AI."""
    try:
        if not request.question or not request.question.strip():
            raise HTTPException(status_code=400, detail="Question is required")
        logger.info(f"Question received: {request.question[:50]}...")

        # For asking, do NOT generate a new tenant. Use provided tenant_id or 'default'
        tenant_id = (request.tenant_id or "default").strip() or "default"

        # Run KB lookup in a worker thread to avoid blocking the event loop or crawl job
        def _run_lookup_sync(q: str, data_dir: str):
            from kb import lookup_kb_minimal
            return lookup_kb_minimal(q, k=10, data_dir=data_dir)

        base_dir = os.getenv("DATA_DIR", "./scraped_pages")
        tenant_dir = os.path.join(base_dir, tenant_id)
        os.makedirs(tenant_dir, exist_ok=True)
        result = await asyncio.to_thread(_run_lookup_sync, request.question.strip(), tenant_dir)

        citations = result.get("citations") or []
        answer = (result.get("answer") or "").strip()
        # Strict guardrail: if no citations or no answer text, return a neutral refusal
        if not citations or not answer:
            fallback = "The context provided does not mention this topic. Therefore, I cannot provide an answer regarding it."
            return QuestionResponse(
                answer=fallback,
                context_info={"total_files": 0, "files": []},
                mcp_debug=None,
            )
        context_info = {
            "total_files": len(citations),
            "files": [
                {"filename": (c.get("url") or c.get("title") or "unknown")} for c in citations if isinstance(c, dict)
            ]
        }

        return QuestionResponse(
            answer=answer,
            context_info=context_info,
            mcp_debug=None
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing question: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to process question: {str(e)}")

@app.get("/files")
async def list_files(tenant_id: Optional[str] = "default"):
    """List all scraped HTML files."""
    try:
        tenant_id = (tenant_id or "default").strip() or "default"
        from storage import HTMLStorage
        base_dir = os.getenv("DATA_DIR", "./scraped_pages")
        data_dir = os.path.join(base_dir, tenant_id)
        storage = HTMLStorage(data_dir)
        storage_info = storage.get_storage_info()

        # Detect uploaded documents by reading json metadata/url
        uploaded_files: list[str] = []
        uploaded_files_meta: list[dict] = []
        try:
            base = storage_info.get("storage_path")
            for name in storage_info.get("files", []):
                if not name.endswith('.json'):
                    continue
                fpath = os.path.join(base, name)
                try:
                    with open(fpath, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    url = str(data.get('url', ''))
                    meta = data.get('metadata', {}) or {}
                    if url.startswith('uploaded://') or ('uploaded_at' in meta):
                        uploaded_files.append(name)
                        uploaded_files_meta.append({
                            "filename": name,
                            "source_file": meta.get('source_file') or data.get('title') or name,
                            "url": url,
                            "type": meta.get('type') or 'unknown'
                        })
                except Exception:
                    continue
        except Exception:
            pass

        storage_info["uploaded_files"] = uploaded_files
        storage_info["uploaded_count"] = len(uploaded_files)
        storage_info["uploaded_files_meta"] = uploaded_files_meta
        return storage_info
    except Exception as e:
        logger.error(f"Error listing files: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/crawl/stats")
async def get_crawl_stats(tenant_id: Optional[str] = "default"):
    """Get statistics about the crawling operation."""
    try:
        tenant_id = (tenant_id or "default").strip() or "default"
        if tenant_id not in _crawl_jobs:
            _crawl_jobs[tenant_id] = _empty_job_state()
        return {"status": "success", "crawl_stats": _crawl_jobs[tenant_id]}
    except Exception as e:
        logger.error(f"Error getting crawl stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/crawl/urls")
async def get_discovered_urls(tenant_id: Optional[str] = "default"):
    """Get the currently discovered accepted and rejected URLs."""
    try:
        tenant_id = (tenant_id or "default").strip() or "default"
        base_dir = os.getenv("DATA_DIR", "./scraped_pages")
        data_dir = os.path.join(base_dir, tenant_id)
        accepted_file = os.path.join(data_dir, "accepted_urls.txt")
        rejected_file = os.path.join(data_dir, "rejected_urls.txt")
        from crawler import load_existing_urls
        accepted_urls = load_existing_urls(accepted_file)
        rejected_urls = load_existing_urls(rejected_file)
        # Prefer rejected: don't show items in both; subtract from accepted for UI consistency
        accepted_urls = accepted_urls.difference(rejected_urls)
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
async def clear_crawl_data(tenant_id: Optional[str] = "default"):
    """Clear all crawl data and start fresh."""
    try:
        tenant_id = (tenant_id or "default").strip() or "default"
        base_dir = os.getenv("DATA_DIR", "./scraped_pages")
        data_dir = os.path.join(base_dir, tenant_id)
        for name in ["accepted_urls.txt", "rejected_urls.txt", "crawler_cookies.txt"]:
            path = os.path.join(data_dir, name)
            try:
                if os.path.exists(path):
                    os.remove(path)
            except Exception:
                pass
        return {
            "status": "success",
            "message": "All crawl data cleared successfully"
        }
    except Exception as e:
        logger.error(f"Error clearing crawl data: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/upload")
async def upload_file(file: UploadFile = File(...), tenant_id: Optional[str] = None):
    """Upload a .txt, .docx, or .pdf file and store as JSON in DATA_DIR so KB can use it."""
    try:
        if not file:
            raise HTTPException(status_code=400, detail="File is required")

        filename = file.filename or "uploaded"
        ext = (filename.split('.')[-1] or '').lower()
        allowed = {"txt", "docx", "pdf"}
        if ext not in allowed:
            raise HTTPException(status_code=400, detail="Only .txt, .docx, .pdf are allowed")

        # Resolve tenant-scoped directory
        base_dir = os.getenv("DATA_DIR", "./scraped_pages")
        tenant_raw = (tenant_id or "").strip()
        final_tenant = uuid.uuid4().hex if (not tenant_raw or tenant_raw.lower() == "default") else tenant_raw
        data_dir = Path(base_dir) / final_tenant
        data_dir.mkdir(parents=True, exist_ok=True)
        # Save to temp
        tmp_path = data_dir / f"_upload_tmp_{datetime.utcnow().timestamp()}_{filename}"
        with open(tmp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Extract text
        text_content = ""
        title = filename
        if ext == "txt":
            with open(tmp_path, "r", encoding="utf-8", errors="ignore") as f:
                text_content = f.read()
        elif ext == "docx":
            try:
                import docx
            except Exception:
                raise HTTPException(status_code=500, detail="python-docx is not installed on server")
            doc = docx.Document(str(tmp_path))
            text_content = "\n".join([p.text for p in doc.paragraphs])
        elif ext == "pdf":
            try:
                import PyPDF2
            except Exception:
                raise HTTPException(status_code=500, detail="PyPDF2 is not installed on server")
            reader = PyPDF2.PdfReader(str(tmp_path))
            parts = []
            for page in reader.pages:
                try:
                    parts.append(page.extract_text() or "")
                except Exception:
                    continue
            text_content = "\n".join(parts)

        # Remove temp
        try:
            os.remove(tmp_path)
        except Exception:
            pass

        if not text_content.strip():
            raise HTTPException(status_code=400, detail="No extractable text from file")

        # Save as a JSON doc in DATA_DIR, compatible with KB lookup
        # Use pseudo-url schema to keep format uniform
        pseudo_url = f"uploaded://{filename}"
        content = {
            "url": pseudo_url,
            "title": title,
            "content": text_content,
            "markdown": text_content,
            "links": [],
            "metadata": {
                "uploaded_at": datetime.utcnow().isoformat(),
                "source_file": filename,
                "type": ext,
            },
        }

        # Reuse scraper save path convention (tenant-scoped output_dir)
        from scrapers.enhanced_scraper import SimpleHTTPScraper
        saver = SimpleHTTPScraper(output_dir=data_dir)
        saved_path = await saver.save_content(content, pseudo_url)

        return {
            "status": "success",
            "message": "File uploaded and indexed",
            "file": filename,
            "stored": saved_path,
            "tenant_id": final_tenant,
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error uploading file: {e}")
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
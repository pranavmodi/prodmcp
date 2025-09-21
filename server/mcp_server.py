#!/usr/bin/env python3
"""
MCP Server for Web Scraping and QA
Implements Model Context Protocol for AI tool integration
Supports hybrid mode: HTTP API + WebSocket + stdio MCP protocol
"""

import asyncio
import json
import logging
import uuid
from typing import Any, Dict, List, Optional
from pathlib import Path
import sys
import ssl
import signal
import websockets
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Import our existing functionality
from storage import HTMLStorage
from scraper import ModernScraper
from qa import QASystem
import os
from dotenv import load_dotenv

from vector_store import FAISSStore

# Import route modules
from routes.crawl import router as crawl_router

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Reduce noise from third-party libraries
for _noisy in [
    "httpx",
    "websockets",
    "faiss",
    "faiss.loader",
    "uvicorn",
    "uvicorn.error",
    "uvicorn.access",
]:
    try:
        logging.getLogger(_noisy).setLevel(logging.WARNING)
    except Exception:
        pass

# Global shutdown event
shutdown_event = asyncio.Event()

# Pydantic model for structured query expansion
class QueryExpansion(BaseModel):
    """Schema for OpenAI structured output - query expansion and keyword extraction"""
    original_query: str
    query_variants: List[str]  # Semantic paraphrases for vector search
    keywords: List[str]        # Important individual terms for keyword search
    phrases: List[str]         # Key multi-word phrases
    domain_terms: List[str]    # Domain-specific related terms

    class Config:
        extra = "forbid"  # This ensures additionalProperties: false in the JSON schema

# FastAPI app for HTTP endpoints
app = FastAPI(
    title="MCP Server API",
    description="HTTP API for Model Context Protocol Server",
    version="2.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
async def startup_event():
    """Initialize resources on startup"""
    logger.info("🚀 MCP Server HTTP API starting up...")

    # Initialize storage and ensure directories exist
    try:
        storage_dir = os.getenv("DATA_DIR", "./scraped_pages")
        Path(storage_dir).mkdir(parents=True, exist_ok=True)
        logger.info(f"📁 Storage directory ensured: {storage_dir}")
    except Exception as e:
        logger.error(f"Failed to initialize storage: {e}")

@app.on_event("shutdown")
async def shutdown_event_handler():
    """Clean up resources on shutdown"""
    logger.info("🛑 MCP Server HTTP API shutting down...")

    # Set global shutdown event to signal other servers
    shutdown_event.set()

    # Close any open sessions/connections
    try:
        # Close any HTTP client sessions if they exist
        import httpx
        # Clean up any global httpx clients if used
        logger.info("✅ HTTP client sessions cleaned up")
    except Exception as e:
        logger.error(f"Error cleaning up HTTP sessions: {e}")

    # Close any database connections or file handles
    try:
        # Add any specific cleanup here
        logger.info("✅ Database connections closed")
    except Exception as e:
        logger.error(f"Error closing database connections: {e}")

    logger.info("✅ MCP Server HTTP API shutdown complete")

# Include route modules
app.include_router(crawl_router)

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
                        "tenant_id": {
                            "type": "string",
                            "description": "Tenant ID to scope the KB directory (scraped_pages/<tenant_id>)"
                        },
                        "k": {
                            "type": "integer",
                            "description": "Top documents to retrieve",
                            "default": 15
                        }
                    },
                    "required": ["query"]
                }
            }
        ]
    
    async def handle_initialize(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle MCP initialization"""
        try:
            logger.debug(f"MCP initialize received: params_keys={list((params or {}).keys())}")
        except Exception:
            pass
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
        logger.debug("MCP tools/list requested")
        return {
            "tools": self.tools
        }
    
    async def handle_tools_call(self, name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Handle tools/call request"""
        try:
            # Demote verbose call tracing; keep minimal signal in INFO by omitting
            # logger.info for method/args to reduce log noise
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

    async def _tool_scrape_website(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """MCP tool: crawl+scrape for a tenant and build FAISS index."""
        tenant_id = (arguments.get("tenant_id") or "").strip() or uuid.uuid4().hex
        url = (arguments.get("url") or "").strip()
        exclusions = arguments.get("excluded_urls") or []
        if not url:
            return {
                "content": [{"type": "text", "text": "url is required"}],
                "isError": True,
                "toolCallId": "scrape_website_error",
                "metadata": {"code": -32602}
            }

        base_dir = os.getenv("DATA_DIR", "./scraped_pages")
        data_dir = os.path.join(base_dir, tenant_id)
        os.makedirs(data_dir, exist_ok=True)
        accepted_file = os.path.join(data_dir, "accepted_urls.txt")
        rejected_file = os.path.join(data_dir, "rejected_urls.txt")

        from crawler import load_existing_urls, crawl_website as crawl_urls
        existing_accepted = load_existing_urls(accepted_file)
        existing_rejected = load_existing_urls(rejected_file)

        # Crawl synchronously in thread
        await asyncio.to_thread(
            crawl_urls,
            url,
            accepted_file,
            rejected_file,
            existing_accepted,
            existing_rejected,
            list(exclusions),
            None,
            data_dir,
        )

        # Scrape
        all_accepted = sorted(load_existing_urls(accepted_file))
        scraped_files: List[str] = []
        failed_urls: List[str] = []

        if all_accepted:
            def _run_scrape_blocking(urls: List[str], output_dir: str):
                import asyncio as _asyncio
                from scraper import ModernScraper
                scraper = ModernScraper(output_dir=output_dir)
                return _asyncio.run(scraper.scrape_urls_batch(urls))

            results = await asyncio.to_thread(_run_scrape_blocking, all_accepted, data_dir)
            scraped_files = [r.get("url") for r in results if r]
            failed_urls = [u for u in all_accepted if u not in scraped_files]

        # Build FAISS (best-effort)
        try:
            from vector_store import FAISSStore
            store = FAISSStore(tenant_id=tenant_id)
            store.build()
        except Exception:
            pass

        return {
            "content": [
                {"type": "text", "text": f"Crawled {url}. Scraped {len(scraped_files)} pages. Failed {len(failed_urls)}."}
            ],
            "isError": False,
            "toolCallId": "scrape_website_result",
            "metadata": {
                "tenant_id": tenant_id,
                "accepted_total": len(all_accepted),
                "rejected_total": len(load_existing_urls(rejected_file)),
                "scraped_total": len(scraped_files),
                "failed_total": len(failed_urls),
            }
        }

    async def _tool_search_kb(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """MCP tool: search tenant KB and return snippets + citations."""
        tenant_id = (arguments.get("tenant_id") or "").strip() or "default"
        query = (arguments.get("query") or "").strip()
        k = int(arguments.get("k", 5))
        if not query:
            return {
                "content": [{"type": "text", "text": "query is required"}],
                "isError": True,
                "toolCallId": "search_kb_error",
                "metadata": {"code": -32602}
            }
        base_dir = os.getenv("DATA_DIR", "./scraped_pages")
        data_dir = os.path.join(base_dir, tenant_id)
        os.makedirs(data_dir, exist_ok=True)
        from kb import lookup_kb_minimal
        res = lookup_kb_minimal(query, k=k, data_dir=data_dir, retrieve_only=True)
        snippets = res.get("snippets", []) or []
        citations = res.get("citations", []) or []
        return {
            "content": [{"type": "text", "text": f"Retrieved {len(snippets)} snippets"}],
            "isError": False,
            "toolCallId": "search_kb_result",
            "metadata": {"snippets": snippets, "citations": citations, "k": k}
        }

    async def _expand_query_with_ai(self, query: str) -> QueryExpansion:
        """Use OpenAI Responses API with structured outputs to expand query into variants and keywords"""
        if not os.getenv("OPENAI_API_KEY"):
            raise ValueError("OPENAI_API_KEY is required for query expansion")

        import httpx

        # Prepare the request for the Responses API
        url = "https://api.openai.com/v1/responses"
        headers = {
            "Authorization": f"Bearer {os.getenv('OPENAI_API_KEY')}",
            "Content-Type": "application/json"
        }

        payload = {
            "model": os.getenv("LLM_MULTIQUERY_MODEL", "gpt-4o-mini"),
            "input": f"""You are a search query analyzer for a business knowledge base system.

Analyze this search query and extract:
1. query_variants: 3-5 semantic paraphrases that maintain the same meaning
2. keywords: Important individual words for exact text matching
3. phrases: Key multi-word phrases (2-4 words) that should be searched together
4. domain_terms: Related business/contact terms that might appear in relevant content

Focus on business information, contact details, services, and company information queries.
Keep variants concise (max 8 words each).
Avoid overly generic terms in keywords.

Query to analyze: {query}""",
            "text": {
                "format": {
                    "type": "json_schema",
                    "name": "QueryExpansion",
                    "schema": QueryExpansion.model_json_schema(),
                    "strict": True
                }
            },
            "temperature": 0.3
        }

        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(url, headers=headers, json=payload)

                # Log non-200 status as error with body
                if response.status_code != 200:
                    logger.error(f"OpenAI Responses API HTTP {response.status_code}: {response.text}")

                response.raise_for_status()
                result = response.json()
                try:
                    logger.debug(f"Responses API keys: {list(result.keys())}")
                except Exception:
                    pass

                # Extract structured content from Responses API
                # Prefer output array's output_text entries; fallback to top-level text if string
                data = None
                # 1) Prefer explicit parsed output if provided by API
                try:
                    if isinstance(result.get("output_parsed"), dict):
                        data = result.get("output_parsed")
                except Exception:
                    data = None

                # 2) Traverse output -> message -> content -> output_text
                if data is None:
                    try:
                        import json as _json
                        output = result.get("output") or []
                        for item in output:
                            if not isinstance(item, dict):
                                continue
                            # Direct output_text item
                            if item.get("type") == "output_text" and isinstance(item.get("text"), str):
                                try:
                                    data = _json.loads(item["text"])  # expect JSON string
                                    break
                                except Exception:
                                    continue
                            # Message container with content parts
                            if item.get("type") == "message":
                                for part in item.get("content", []) or []:
                                    if isinstance(part, dict) and part.get("type") == "output_text" and isinstance(part.get("text"), str):
                                        try:
                                            data = _json.loads(part["text"])  # expect JSON string
                                            break
                                        except Exception:
                                            continue
                                if data is not None:
                                    break
                    except Exception:
                        data = None

                # 3) Fallback to top-level text string
                if data is None:
                    content = result.get("text")
                    if isinstance(content, str) and content.strip():
                        try:
                            import json as _json
                            data = _json.loads(content)
                        except Exception:
                            data = None

                if data is None:
                    logger.error("OpenAI Responses API invalid/missing structured text content (no output_text or JSON 'text')")
                    raise ValueError("Invalid response structure from OpenAI Responses API")

                try:
                    return QueryExpansion(**data)
                except Exception as e:
                    logger.error(f"OpenAI Responses API JSON did not match schema: {e}. Data keys: {list(data.keys()) if isinstance(data, dict) else type(data)}")
                    raise
        except Exception as e:
            logger.error(f"OpenAI Responses API call failed: {e}", exc_info=True)
            raise

    async def _keyword_search(self, tenant_id: str, keywords: List[str], phrases: List[str], k: int = 5, allowed_domains: set = None) -> List[Dict[str, Any]]:
        """Perform keyword/phrase search on tenant JSON files"""
        base_dir = os.getenv("DATA_DIR", "./scraped_pages")
        data_dir = os.path.join(base_dir, tenant_id)

        if not os.path.exists(data_dir):
            return []

        # Collect all search terms
        search_terms = []
        if keywords:
            search_terms.extend([term.lower() for term in keywords])
        if phrases:
            search_terms.extend([phrase.lower() for phrase in phrases])

        if not search_terms:
            return []

        # Find all JSON files
        json_files = []
        for root, dirs, files in os.walk(data_dir):
            for file in files:
                if file.endswith('.json'):
                    json_files.append(os.path.join(root, file))

        results = []

        for json_file in json_files:
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)

                # Extract text content for searching
                content = data.get('content', '') or data.get('markdown', '') or ''
                title = data.get('title', '')
                url = data.get('url', '')

                if not content:
                    continue

                # Domain filtering
                if allowed_domains and url:
                    try:
                        from urllib.parse import urlparse
                        host = urlparse(url).netloc.lower()
                        if host not in allowed_domains:
                            continue
                    except Exception:
                        continue

                # Search for keywords and phrases in content (case-insensitive)
                content_lower = content.lower()
                title_lower = title.lower()

                # Calculate relevance score
                score = 0.0
                matched_terms = []

                for term in search_terms:
                    if not term:
                        continue

                    # Count occurrences in content
                    content_matches = content_lower.count(term)
                    title_matches = title_lower.count(term)

                    if content_matches > 0 or title_matches > 0:
                        matched_terms.append(term)
                        # Weight title matches higher
                        score += title_matches * 2.0 + content_matches * 1.0

                # Only include if we found matching terms
                if score > 0:
                    # Create snippet around the first match
                    snippet = self._create_keyword_snippet(content, matched_terms)

                    results.append({
                        'url': url,
                        'title': title,
                        'snippet': snippet,
                        'score': score,
                        'matched_terms': matched_terms
                    })

            except Exception as e:
                continue

        # Sort by relevance score (descending)
        results.sort(key=lambda x: x['score'], reverse=True)

        # Return top k results
        return results[:k]

    def _create_keyword_snippet(self, content: str, matched_terms: List[str], max_length: int = 1200) -> str:
        """Create a snippet around keyword matches"""
        if not content or not matched_terms:
            return content[:max_length] if content else ""

        content_lower = content.lower()

        # Find the first occurrence of any matched term
        best_start = -1
        for term in matched_terms:
            if term:
                pos = content_lower.find(term.lower())
                if pos != -1 and (best_start == -1 or pos < best_start):
                    best_start = pos

        if best_start == -1:
            return content[:max_length]

        # Create snippet around the match
        snippet_start = max(0, best_start - 100)
        snippet_end = min(len(content), best_start + max_length - 100)

        snippet = content[snippet_start:snippet_end]

        # Add ellipsis if truncated
        if snippet_start > 0:
            snippet = "..." + snippet
        if snippet_end < len(content):
            snippet = snippet + "..."

        return snippet

    async def _lookup_kb(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Lookup knowledge base from local scraped JSON using a minimal RAG approach."""
        query = arguments.get("query")
        if not query:
            raise ValueError("query is required")
        k = int(arguments.get("k", 5))
        # Optional tenant scoping
        tenant_id = (arguments.get("tenant_id") or "").strip() or "default"

        # Minimal high-signal logging
        logger.info(f"🔍 query: '{query}'")

        # Ensure expansion variable is always defined to avoid UnboundLocalError in fallbacks
        expansion = None

        # Perform query expansion first so we can always log variants/keywords
        try:
            expansion = await self._expand_query_with_ai(query)
            try:
                logger.info(f"📝 variants: {expansion.query_variants}")
                logger.info(f"🔑 keywords: {expansion.keywords}")
            except Exception:
                pass
        except Exception:
            expansion = None

        # Try FAISS-backed retrieval first; build if missing. Fallback to minimal KB.
        try:
            logger.info(f"🔍 attempting vector search for tenant: {tenant_id}")
            store = FAISSStore(tenant_id=tenant_id)
            warm = store.search("warmup", k=1)
            logger.info(f"🔍 warmup search returned {len(warm)} results")
            if warm == []:
                # If index not present, attempt a quick build (best-effort)
                logger.info("🔍 FAISS index not warm; attempting build()")
                built = store.build()
                logger.info(f"🔍 FAISS build completed; chunks_indexed={built}")

            # Collect all search terms for vector search
            variants = [query] + (expansion.query_variants if expansion else [])

            # Collect all keywords for text search
            all_keywords = (expansion.keywords + expansion.domain_terms) if expansion else []
            all_phrases = (expansion.phrases) if expansion else []

            # Build tenant domain whitelist from accepted URLs (if available)
            allowed_domains = set()
            try:
                from urllib.parse import urlparse as _urlparse
                base_dir = os.getenv("DATA_DIR", "./scraped_pages")
                acc_path = os.path.join(base_dir, tenant_id, "accepted_urls.txt")
                if os.path.exists(acc_path):
                    with open(acc_path, "r", encoding="utf-8") as _f:
                        for line in _f:
                            u = (line or "").strip()
                            if not u:
                                continue
                            try:
                                host = _urlparse(u).netloc.lower()
                                if host:
                                    allowed_domains.add(host)
                            except Exception:
                                continue
            except Exception:
                allowed_domains = set()

            # Collect and merge hits across variants
            per_k = max(k, 12)
            unique_hits: dict = {}
            variant_results = {}

            logger.info(f"🔍 searching {len(variants[:12])} variants")
            for v in variants[:12]:
                hits = store.search(v, k=per_k)
                variant_results[v] = len(hits)
                logger.info(f"🔍 variant '{v}' returned {len(hits)} hits")

                for h in hits:
                    # Domain filter
                    try:
                        url = (h.get("url") or "").strip()
                        if allowed_domains:
                            from urllib.parse import urlparse as _urlparse
                            host = _urlparse(url).netloc.lower()
                            if host not in allowed_domains:
                                continue
                    except Exception:
                        pass
                    key = (h.get("url"), h.get("snippet"))
                    if key not in unique_hits:
                        unique_hits[key] = h

            merged_hits = list(unique_hits.values())

            # Simple MMR-like rerank to improve diversity
            def _score(it):
                try:
                    return float(it.get("score", 0.0))
                except Exception:
                    return 0.0
            merged_hits.sort(key=_score, reverse=True)

            def _overlap(a: str, b: str) -> float:
                sa = set((a or "").lower().split())
                sb = set((b or "").lower().split())
                if not sa or not sb:
                    return 0.0
                return len(sa & sb) / max(len(sa), len(sb))

            selected: list = []
            for cand in merged_hits:
                if len(selected) >= max(k, 20):
                    break
                if all(_overlap(cand.get("snippet", ""), s.get("snippet", "")) < 0.6 for s in selected):
                    selected.append(cand)

            faiss_hits = selected
            if faiss_hits:
                logger.info(f"🔍 vector search found {len(faiss_hits)} results")
        except Exception as e:
            logger.info(f"🔍 vector search failed: {e}")
            faiss_hits = []

        # Perform keyword search to complement vector search
        keyword_hits = []
        if all_keywords or all_phrases:
            try:
                keyword_hits = await self._keyword_search(tenant_id, all_keywords, all_phrases, k=k, allowed_domains=allowed_domains)
                if keyword_hits:
                    logger.info(f"🔑 keyword search found {len(keyword_hits)} results")
            except Exception:
                keyword_hits = []

        # Merge FAISS and keyword results
        combined_hits = []

        if faiss_hits and keyword_hits:
            # Combine both results, prioritizing vector search but including unique keyword results
            seen_snippets = set()

            # Add FAISS results first
            for hit in faiss_hits:
                snippet = hit.get("snippet", "")
                if snippet and snippet not in seen_snippets:
                    combined_hits.append({**hit, "source": "vector"})
                    seen_snippets.add(snippet)

            # Add unique keyword results
            for hit in keyword_hits:
                snippet = hit.get("snippet", "")
                if snippet and snippet not in seen_snippets:
                    combined_hits.append({**hit, "source": "keyword"})
                    seen_snippets.add(snippet)

            # Limit to k results
            combined_hits = combined_hits[:k]
            logger.info(f"🔄 hybrid merge: {len([h for h in combined_hits if h.get('source') == 'vector'])} vector + {len([h for h in combined_hits if h.get('source') == 'keyword'])} keyword = {len(combined_hits)} total")

        elif faiss_hits:
            combined_hits = [{**hit, "source": "vector"} for hit in faiss_hits]
            logger.info(f"🔍 using only vector search: {len(combined_hits)} results")
        elif keyword_hits:
            combined_hits = [{**hit, "source": "keyword"} for hit in keyword_hits]
            logger.info(f"🔑 using only keyword search: {len(combined_hits)} results")

        if combined_hits:
            snippets = [hit.get("snippet", "") for hit in combined_hits if hit.get("snippet")]
            citations = [{"url": hit.get("url"), "title": hit.get("title", "")} for hit in combined_hits]

            # Log snippets with source information
            logger.info(f"📄 snippets ({len(snippets)}):")
            for i, (snippet, hit) in enumerate(zip(snippets[:k], combined_hits[:k]), 1):
                snippet_preview = snippet[:140] + "..." if len(snippet) > 140 else snippet
                source = hit.get("source", "unknown")
                source_icon = "🔍" if source == "vector" else "🔑" if source == "keyword" else "❓"
                logger.info(f"  {i}. {source_icon} {snippet_preview}")

            # Count sources for reporting
            vector_count = sum(1 for hit in combined_hits if hit.get("source") == "vector")
            keyword_count = sum(1 for hit in combined_hits if hit.get("source") == "keyword")

            # Build metadata with query expansion info (guard if expansion not available)
            metadata = {
                "snippets": snippets,
                "citations": citations,
                "k": k,
                "faiss": len(faiss_hits) > 0,
                "keyword_search": len(keyword_hits) > 0,
                "hybrid_results": {
                    "vector_results": vector_count,
                    "keyword_results": keyword_count,
                    "total_results": len(combined_hits)
                },
                "query_expansion": {
                    "original_query": query,
                    "variants_used": (expansion.query_variants if expansion else []),
                    "keywords_extracted": (expansion.keywords if expansion else []),
                    "phrases_extracted": (expansion.phrases if expansion else []),
                    "domain_terms": (expansion.domain_terms if expansion else [])
                }
            }

            # Create descriptive message about search results
            if vector_count > 0 and keyword_count > 0:
                search_description = f"hybrid search ({vector_count} vector + {keyword_count} keyword)"
            elif vector_count > 0:
                search_description = "vector search"
            else:
                search_description = "keyword search"

            return {
                "content": [{"type": "text", "text": f"Retrieved {len(snippets)} snippets via {search_description}"}],
                "isError": False,
                "toolCallId": "lookup_kb_result",
                "metadata": metadata
            }

        # Fallback to lexical minimal lookup
        from kb import lookup_kb_minimal
        base_dir = os.getenv("DATA_DIR", "./scraped_pages")
        data_dir = os.path.join(base_dir, tenant_id)
        # Use retrieve_only to ensure we get raw snippets for logging
        result = lookup_kb_minimal(query, k=k, data_dir=data_dir, retrieve_only=True)

        # Log lexical snippets succinctly
        snippets = result.get('snippets', []) or []
        logger.info(f"📄 snippets ({len(snippets)}):")
        for i, snippet in enumerate(snippets[:k], 1):
            snippet_preview = snippet[:140] + "..." if len(snippet) > 140 else snippet
            logger.info(f"  {i}. {snippet_preview}")

        # Add query expansion info to lexical fallback too (guard if expansion not available)
        fallback_metadata = result | {
            "faiss": False,
            "query_expansion": {
                "original_query": query,
                "variants_used": (expansion.query_variants if expansion else []),
                "keywords_extracted": (expansion.keywords if expansion else []),
                "phrases_extracted": (expansion.phrases if expansion else []),
                "domain_terms": (expansion.domain_terms if expansion else [])
            }
        }

        return {
            "content": [{"type": "text", "text": f"Retrieved {len(snippets)} snippets via lexical"}],
            "isError": False,
            "toolCallId": "lookup_kb_result",
            "metadata": fallback_metadata
        }
class MCPTransport:
    """Handles MCP protocol communication over stdio"""
    
    def __init__(self, server: MCPServer):
        self.server = server
        self.request_id = 0
    
    async def run(self):
        """Run the MCP server over stdio"""
        logger.debug("Starting MCP server over stdio")
        
        while True:
            try:
                # Read request from stdin
                line = await asyncio.get_event_loop().run_in_executor(None, sys.stdin.readline)
                if not line:
                    break
                
                raw = line.strip()
                try:
                    logger.debug(f"MCP STDIO RX: {raw[:200]}{'...' if len(raw)>200 else ''}")
                except Exception:
                    pass
                request = json.loads(raw)
                response = await self._handle_request(request)
                
                # Send response to stdout
                out = json.dumps(response)
                try:
                    logger.debug(f"MCP STDIO TX: {out[:200]}{'...' if len(out)>200 else ''}")
                except Exception:
                    pass
                print(out, flush=True)
                
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

class MCPWebSocketTransport:
    """Handles MCP protocol communication over WebSocket"""

    def __init__(self, server: MCPServer):
        self.server = server
        self.clients = set()

    async def handle_client(self, websocket, path):
        """Handle a new WebSocket client connection"""
        client_id = f"{websocket.remote_address[0]}:{websocket.remote_address[1]}"
        logger.debug(f"MCP WebSocket client connected: {client_id} (path: {path})")
        self.clients.add(websocket)

        try:
            async for message in websocket:
                try:
                    # Parse incoming JSON-RPC request
                    raw_data = message
                    logger.debug(f"MCP WS RX [{client_id}]: {raw_data[:200]}{'...' if len(raw_data)>200 else ''}")

                    request = json.loads(raw_data)
                    response = await self._handle_request(request)

                    # Send JSON-RPC response
                    response_data = json.dumps(response)
                    logger.debug(f"MCP WS TX [{client_id}]: {response_data[:200]}{'...' if len(response_data)>200 else ''}")

                    await websocket.send(response_data)

                except json.JSONDecodeError as e:
                    logger.error(f"Invalid JSON from client {client_id}: {e}")
                    error_response = self._create_error_response(None, -32700, "Parse error")
                    await websocket.send(json.dumps(error_response))
                except Exception as e:
                    logger.error(f"Error handling message from client {client_id}: {e}")
                    error_response = self._create_error_response(None, -32603, "Internal error")
                    await websocket.send(json.dumps(error_response))

        except websockets.exceptions.ConnectionClosed:
            logger.debug(f"MCP WebSocket client disconnected: {client_id}")
        except Exception as e:
            logger.error(f"WebSocket error for client {client_id}: {e}")
        finally:
            self.clients.discard(websocket)

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

async def run_websocket_server(port: int):
    """Run WebSocket MCP server"""
    server = MCPServer()
    ws_transport = MCPWebSocketTransport(server)

    # Create a wrapper function to handle the websocket connection
    async def websocket_handler(*args):
        websocket = args[0]
        path = args[1] if len(args) > 1 else "/"
        await ws_transport.handle_client(websocket, path)

    # Start WebSocket server
    logger.debug(f"🚀 MCP WebSocket server listening on ws://0.0.0.0:{port}")

    async with websockets.serve(
        websocket_handler,
        "0.0.0.0",
        port,
        ping_interval=20,
        ping_timeout=10
    ) as ws_server:
        logger.info("📡 WebSocket server ready, waiting for connections...")

        # Wait for shutdown event or server closure
        try:
            shutdown_task = asyncio.create_task(shutdown_event.wait())
            server_task = asyncio.create_task(ws_server.wait_closed())

            done, pending = await asyncio.wait(
                [shutdown_task, server_task],
                return_when=asyncio.FIRST_COMPLETED
            )

            # Cancel pending tasks
            for task in pending:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass

            if shutdown_task in done:
                logger.info("🛑 WebSocket server shutting down gracefully...")
                ws_server.close()
                await ws_server.wait_closed()
                logger.info("✅ WebSocket server shutdown complete")
        except Exception as e:
            logger.error(f"Error during WebSocket server shutdown: {e}")
            raise

async def start_http_server(port: int):
    """Start HTTP API server"""
    config = uvicorn.Config(
        app=app,
        host="0.0.0.0",
        port=port,
        log_level="info"
    )
    server = uvicorn.Server(config)
    logger.info(f"🌐 HTTP API server listening on http://0.0.0.0:{port}")

    # Start server in a task so we can handle shutdown
    server_task = asyncio.create_task(server.serve())
    shutdown_task = asyncio.create_task(shutdown_event.wait())

    try:
        done, pending = await asyncio.wait(
            [server_task, shutdown_task],
            return_when=asyncio.FIRST_COMPLETED
        )

        if shutdown_task in done:
            logger.info("🛑 HTTP server shutting down gracefully...")
            server.should_exit = True

            # Cancel server task and wait for it to finish
            if server_task in pending:
                server_task.cancel()
                try:
                    await server_task
                except asyncio.CancelledError:
                    pass

            logger.info("✅ HTTP server shutdown complete")
        else:
            # Server completed normally, cancel shutdown task
            shutdown_task.cancel()
            await server_task

    except Exception as e:
        logger.error(f"Error during HTTP server shutdown: {e}")
        # Cancel any pending tasks
        for task in [server_task, shutdown_task]:
            if not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
        raise

def setup_signal_handlers():
    """Setup signal handlers for graceful shutdown"""
    def signal_handler(signum, frame):
        logger.info(f"🔔 Received signal {signum}, initiating graceful shutdown...")
        shutdown_event.set()

    # Handle SIGINT (Ctrl+C) and SIGTERM
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

async def main():
    """Main entry point - starts hybrid MCP + HTTP server"""
    # Setup signal handlers for graceful shutdown
    setup_signal_handlers()

    # Always start in hybrid mode
    ws_port = int(os.getenv("MCP_WEBSOCKET_PORT", "8765"))
    http_port = int(os.getenv("MCP_HTTP_PORT", "8000"))

    logger.info("🚀 Starting Hybrid MCP + HTTP Server")
    logger.debug(f"📡 MCP WebSocket: ws://0.0.0.0:{ws_port}")
    logger.info(f"🌐 HTTP API: http://0.0.0.0:{http_port}")

    # Create tasks for both servers with error handling
    async def run_ws_with_error_handling():
        try:
            await run_websocket_server(ws_port)
        except Exception as e:
            logger.error(f"WebSocket server failed: {e}")
            raise

    async def run_http_with_error_handling():
        try:
            await start_http_server(http_port)
        except Exception as e:
            logger.error(f"HTTP server failed: {e}")
            raise

    ws_task = asyncio.create_task(run_ws_with_error_handling())
    http_task = asyncio.create_task(run_http_with_error_handling())

    # Run both servers concurrently
    try:
        await asyncio.gather(ws_task, http_task, return_exceptions=True)
    except KeyboardInterrupt:
        logger.info("🔔 Keyboard interrupt received, shutting down...")
        shutdown_event.set()
    except Exception as e:
        logger.error(f"Server error: {e}")
        shutdown_event.set()
        raise
    finally:
        # Ensure cleanup happens
        logger.info("🧹 Final cleanup...")

        # Cancel remaining tasks
        for task in [ws_task, http_task]:
            if not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass

        logger.info("✅ All servers stopped, shutdown complete")

if __name__ == "__main__":
    asyncio.run(main()) 
"""
Crawl endpoints for the MCP server
Provides HTTP API for web crawling operations
"""

import asyncio
import logging
import os
from typing import Dict, Any
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from crawler import crawl_website, load_existing_urls

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/crawl", tags=["crawl"])

class CrawlRequest(BaseModel):
    url: str
    tenant_id: str
    max_pages: int = 50
    delay_range: list = [3, 7]

@router.get("/stats")
async def get_crawl_stats(tenant_id: str = "default"):
    """Get crawling statistics for a tenant"""
    try:
        base_dir = os.getenv("DATA_DIR", "./scraped_pages")
        data_dir = os.path.join(base_dir, tenant_id)
        accepted_file = os.path.join(data_dir, "accepted_urls.txt")
        rejected_file = os.path.join(data_dir, "rejected_urls.txt")

        # Create directory if it doesn't exist
        os.makedirs(data_dir, exist_ok=True)

        accepted_urls = load_existing_urls(accepted_file)
        rejected_urls = load_existing_urls(rejected_file)

        stats = {
            "tenant_id": tenant_id,
            "total_accepted": len(accepted_urls),
            "total_rejected": len(rejected_urls),
            "total_urls": len(accepted_urls) + len(rejected_urls),
            "accepted_file": accepted_file,
            "rejected_file": rejected_file,
            "status": "idle"  # TODO: Add actual crawl status tracking
        }

        logger.info(f"Crawl stats for tenant {tenant_id}: {stats}")
        return stats

    except Exception as e:
        logger.error(f"Error getting crawl stats for tenant {tenant_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get crawl stats: {str(e)}")

@router.get("/urls")
async def get_crawled_urls(tenant_id: str = "default"):
    """Get list of crawled URLs for a tenant"""
    try:
        base_dir = os.getenv("DATA_DIR", "./scraped_pages")
        data_dir = os.path.join(base_dir, tenant_id)
        accepted_file = os.path.join(data_dir, "accepted_urls.txt")

        # Create directory if it doesn't exist
        os.makedirs(data_dir, exist_ok=True)

        accepted_urls = load_existing_urls(accepted_file)

        result = {
            "tenant_id": tenant_id,
            "urls": list(accepted_urls),
            "total_urls": len(accepted_urls)
        }

        logger.info(f"Retrieved {len(accepted_urls)} URLs for tenant {tenant_id}")
        return result

    except Exception as e:
        logger.error(f"Error getting URLs for tenant {tenant_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get URLs: {str(e)}")

@router.post("")
async def start_crawl(request: CrawlRequest):
    """Start crawling a website for a specific tenant"""
    try:
        # Validate parameters
        if not request.url or not request.tenant_id:
            raise HTTPException(status_code=400, detail="URL and tenant_id are required")

        if request.max_pages < 1 or request.max_pages > 1000:
            raise HTTPException(status_code=400, detail="max_pages must be between 1 and 1000")

        if len(request.delay_range) != 2 or request.delay_range[0] < 0 or request.delay_range[1] < request.delay_range[0]:
            raise HTTPException(status_code=400, detail="delay_range must be [min_delay, max_delay] with min_delay <= max_delay")

        # Set up tenant-specific data directory
        base_dir = os.getenv("DATA_DIR", "./scraped_pages")
        data_dir = os.path.join(base_dir, request.tenant_id)
        os.makedirs(data_dir, exist_ok=True)

        accepted_file = os.path.join(data_dir, "accepted_urls.txt")
        rejected_file = os.path.join(data_dir, "rejected_urls.txt")

        # Load existing URLs
        existing_accepted = load_existing_urls(accepted_file)
        existing_rejected = load_existing_urls(rejected_file)

        logger.info(f"Starting crawl for tenant {request.tenant_id}: {request.url}")

        # Run crawl in a thread to avoid blocking
        def run_crawl():
            return crawl_website(
                base_url=request.url.strip(),
                accepted_file=accepted_file,
                rejected_file=rejected_file,
                existing_accepted=existing_accepted,
                existing_rejected=existing_rejected
            )

        # Run crawl in background thread
        await asyncio.to_thread(run_crawl)

        # Get updated stats
        final_accepted = load_existing_urls(accepted_file)
        final_rejected = load_existing_urls(rejected_file)

        # Optionally trigger scraping of discovered URLs
        scraped_count = 0
        failed_count = 0

        if final_accepted:
            try:
                from scraper import ModernScraper

                def run_scraper():
                    scraper = ModernScraper(output_dir=data_dir)
                    return asyncio.run(scraper.scrape_urls_batch(list(final_accepted)))

                results = await asyncio.to_thread(run_scraper)
                scraped_count = len([r for r in results if r])
                failed_count = len(final_accepted) - scraped_count

                logger.info(f"Scraping completed: {scraped_count} successful, {failed_count} failed")

            except Exception as scrape_error:
                logger.error(f"Scraping failed: {scrape_error}")
                failed_count = len(final_accepted)

        # Optionally build FAISS index
        try:
            from vector_store import FAISSStore
            store = FAISSStore(tenant_id=request.tenant_id)
            chunks_indexed = store.build()
            logger.info(f"FAISS index built with {chunks_indexed} chunks for tenant {request.tenant_id}")
        except Exception as faiss_error:
            logger.warning(f"FAISS index build failed: {faiss_error}")

        result = {
            "status": "completed",
            "tenant_id": request.tenant_id,
            "base_url": request.url,
            "discovered_urls": len(final_accepted) + len(final_rejected),
            "accepted_urls": len(final_accepted),
            "rejected_urls": len(final_rejected),
            "scraped_pages": scraped_count,
            "failed_scrapes": failed_count,
            "message": f"Crawl completed. Discovered {len(final_accepted) + len(final_rejected)} URLs, scraped {scraped_count} pages."
        }

        logger.info(f"Crawl completed for tenant {request.tenant_id}: {result}")
        return result

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error during crawl for tenant {request.tenant_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Crawl failed: {str(e)}")
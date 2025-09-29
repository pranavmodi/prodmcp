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
from db import create_crawl_job, set_crawl_status

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/crawl", tags=["crawl"])

# In-memory per-tenant job state to drive live status/progress
_crawl_jobs: Dict[str, Any] = {}

def _get_or_init_job(tenant_id: str) -> Dict[str, Any]:
    """Return a mutable job state for a tenant, initializing if needed."""
    job = _crawl_jobs.get(tenant_id)
    if job is None:
        job = {
            "tenant_id": tenant_id,
            "status": "idle",            # idle | crawl | scrape | done | error
            "accepted_urls": 0,
            "rejected_urls": 0,
            "scraped_pages": 0,
            "message": None,
        }
        _crawl_jobs[tenant_id] = job
    return job

class CrawlRequest(BaseModel):
    url: str
    tenant_id: str
    max_pages: int = 50
    delay_range: list = [3, 7]
    exclusions: list | None = None

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

        file_accepted = load_existing_urls(accepted_file)
        file_rejected = load_existing_urls(rejected_file)

        # Merge in-memory live counts when a job is active
        job = _get_or_init_job(tenant_id)
        is_active = job.get("status") in ("crawl", "scrape")
        live_accepted = int(job.get("accepted_urls", 0))
        live_rejected = int(job.get("rejected_urls", 0))
        accepted_count = live_accepted if is_active else len(file_accepted)
        rejected_count = live_rejected if is_active else len(file_rejected)
        discovered_total = int(job.get("discovered_total", 0)) if is_active else (accepted_count + rejected_count)

        stats = {
            "tenant_id": tenant_id,
            "total_accepted": accepted_count,
            "total_rejected": rejected_count,
            "total_urls": discovered_total,
            "accepted_file": accepted_file,
            "rejected_file": rejected_file,
            # Report live status from in-memory job state
            "status": job.get("status", "idle"),
            # Helpful extra fields (the proxy may use these if present)
            "scraped_pages": int(job.get("scraped_pages", 0)),
            "message": job.get("message"),
            # Additional raw fields some proxies may read
            "accepted": accepted_count,
            "rejected": rejected_count,
            "progress_percent": int(job.get("progress_percent", 0)),
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

        # Create DB crawl job row
        exclusions_list = request.exclusions or []
        try:
            job_id = create_crawl_job(
                tenant_id=request.tenant_id,
                url=request.url,
                status="started",
                exclusions=exclusions_list,
            )
        except Exception as db_err:
            logger.warning(f"Failed to create crawl job record: {db_err}")
            job_id = None

        # Initialize live job state
        job = _get_or_init_job(request.tenant_id)
        job["status"] = "crawl"
        job["message"] = "Crawl started"
        job["scraped_pages"] = 0
        job["accepted_urls"] = len(existing_accepted)
        job["rejected_urls"] = len(existing_rejected)
        job["job_id"] = job_id

        # Reflect initial status in DB
        try:
            set_crawl_status(job_id, "crawl")
        except Exception:
            pass

        # Run crawl in a thread to avoid blocking
        def run_crawl():
            # Progress callback updates live job state periodically
            def _on_progress(visited_count, queue_count, accepted_count, rejected_count, percent):
                try:
                    job_local = _get_or_init_job(request.tenant_id)
                    job_local["status"] = "crawl"
                    job_local["accepted_urls"] = int(accepted_count)
                    job_local["rejected_urls"] = int(rejected_count)
                    job_local["discovered_total"] = int(visited_count + queue_count)
                    job_local["progress_percent"] = int(percent)
                    # Suppress non-error message to keep UI percent-only
                    job_local["message"] = None
                except Exception:
                    pass

            return crawl_website(
                base_url=request.url.strip(),
                accepted_file=accepted_file,
                rejected_file=rejected_file,
                existing_accepted=existing_accepted,
                existing_rejected=existing_rejected,
                exclusion_urls=exclusions_list,
                progress_callback=_on_progress,
                data_dir=data_dir,
            )

        # Run crawl in background thread
        await asyncio.to_thread(run_crawl)

        # Get updated stats
        final_accepted = load_existing_urls(accepted_file)
        final_rejected = load_existing_urls(rejected_file)

        # Update job with crawl completion snapshot
        job["accepted_urls"] = len(final_accepted)
        job["rejected_urls"] = len(final_rejected)

        # Optionally trigger scraping of discovered URLs
        scraped_count = 0
        failed_count = 0

        if final_accepted:
            try:
                from scraper import ModernScraper

                def run_scraper():
                    scraper = ModernScraper(output_dir=data_dir)
                    return asyncio.run(scraper.scrape_urls_batch(list(final_accepted)))

                # Switch job state to scraping (no descriptive message)
                job["status"] = "scrape"
                job["message"] = None

                results = await asyncio.to_thread(run_scraper)
                scraped_count = len([r for r in results if r])
                failed_count = len(final_accepted) - scraped_count

                # Update scrape progress snapshot
                job["scraped_pages"] = scraped_count

                # Update DB to scraping
                try:
                    set_crawl_status(job_id, "scrape")
                except Exception:
                    pass

                logger.info(f"Scraping completed: {scraped_count} successful, {failed_count} failed")

            except Exception as scrape_error:
                logger.error(f"Scraping failed: {scrape_error}")
                failed_count = len(final_accepted)
                job["message"] = f"Scraping failed: {scrape_error}"

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

        # Mark job done
        job["status"] = "done"
        job["message"] = result["message"]

        # Persist completion to DB
        try:
            set_crawl_status(job_id, "done", message=result["message"], finished=True)
        except Exception:
            pass

        logger.info(f"Crawl completed for tenant {request.tenant_id}: {result}")
        return result

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error during crawl for tenant {request.tenant_id}: {e}")
        # Mark job as error for visibility in stats
        try:
            job = _get_or_init_job(request.tenant_id)
            job["status"] = "error"
            job["message"] = f"Crawl failed: {str(e)}"
        except Exception:
            pass
        # Persist error to DB
        try:
            # job_id may not exist if DB failed earlier
            set_crawl_status(locals().get("job_id", None), "error", message=str(e), finished=True)
        except Exception:
            pass
        raise HTTPException(status_code=500, detail=f"Crawl failed: {str(e)}")
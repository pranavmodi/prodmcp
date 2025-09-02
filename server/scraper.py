import asyncio
import json
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional

from scrapers.enhanced_scraper import SimpleHTTPScraper


class ModernScraper:
    """
    Wrapper around the copied SimpleHTTPScraper to keep existing API stable.
    """

    def __init__(self, output_dir: Optional[str] = None):
        data_dir = output_dir or os.getenv("DATA_DIR", "./scraped_pages")
        self.output_dir = Path(data_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.http_scraper = SimpleHTTPScraper(output_dir=self.output_dir)

    async def scrape_urls_batch(self, urls: List[str], progress_callback=None) -> List[Dict]:
        try:
            results = await self.http_scraper.scrape_urls(urls, progress_callback)
            # Save results
            for r in results:
                if r and isinstance(r, dict) and r.get("url"):
                    await self.http_scraper.save_content(r, r["url"])
            return results
        except Exception as e:
            logging.error(f"Batch scrape failed: {e}")
            return []

    async def scrape_url(self, url: str) -> Optional[Dict]:
        results = await self.http_scraper.scrape_urls([url])
        return results[0] if results else None

    async def save_content(self, content: Dict, url: str) -> Optional[str]:
        return await self.http_scraper.save_content(content, url)

    async def get_stored_content(self, url: str) -> Optional[Dict]:
        try:
            safe_filename = url.replace("https://", "").replace("http://", "").replace("/", "_").replace(":", "_")
            safe_filename = safe_filename[:100]
            filename = f"{safe_filename}.json"
            path = self.output_dir / filename
            if path.exists():
                with open(path, "r", encoding="utf-8") as f:
                    return json.load(f)
            return None
        except Exception as e:
            logging.error(f"Error reading stored content for {url}: {e}")
            return None

    def get_crawl_stats(self) -> Dict[str, int]:
        data_dir = os.getenv("DATA_DIR", "./scraped_pages")
        accepted_file = Path(data_dir) / "accepted_urls.txt"
        rejected_file = Path(data_dir) / "rejected_urls.txt"
        def _load(p: Path) -> int:
            try:
                with open(p, "r") as f:
                    return len([line.strip() for line in f if line.strip()])
            except FileNotFoundError:
                return 0
        return {
            "total_accepted": _load(accepted_file),
            "total_rejected": _load(rejected_file),
            "total_urls": _load(accepted_file) + _load(rejected_file),
        }
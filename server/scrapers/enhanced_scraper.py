"""
Simple HTTP-based web scraper using requests and BeautifulSoup.
Adapted from work project: removed external config import and wired to DATA_DIR.
"""

import asyncio
import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional
from urllib.parse import urljoin, urlparse

import html2text
import requests
import httpx
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)


class SimpleHTTPScraper:
    """Simple HTTP-based web scraper with requests and BeautifulSoup."""

    def __init__(self, output_dir: Optional[Path] = None, tenant_id: str = "default"):
        self.tenant_id = tenant_id
        try:
            self.timeout = float(os.getenv("SCRAPER_TIMEOUT", "30"))
        except Exception:
            self.timeout = 30.0

        logger.info(f"HTTP scraper initialized with timeout: {self.timeout}s")

        # Setup HTTP session with retries
        self.session = requests.Session()
        try:
            from requests.adapters import HTTPAdapter
            from urllib3.util.retry import Retry

            retry = Retry(
                total=3,
                connect=3,
                read=3,
                backoff_factor=1.5,
                status_forcelist=[429, 500, 502, 503, 504],
                allowed_methods=["GET", "HEAD"],
            )
            adapter = HTTPAdapter(max_retries=retry)
            self.session.mount("http://", adapter)
            self.session.mount("https://", adapter)
        except Exception as _e:
            logger.debug(f"Retry adapter setup skipped: {_e}")

        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.9',
            # Avoid Brotli to prevent decompression issues without brotli installed
            'Accept-Encoding': 'gzip, deflate',
            'Connection': 'keep-alive',
        })

        # Output directory
        if output_dir is None:
            data_dir = os.getenv("DATA_DIR", "./scraped_pages")
            self.output_dir = Path(data_dir)
        else:
            self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # HTML to text converter
        self.h = html2text.HTML2Text()
        self.h.ignore_links = False
        self.h.ignore_images = True
        self.h.body_width = 0

    async def scrape_urls(self, urls: List[str], progress_callback=None) -> List[Dict]:
        if not urls:
            return []

        logger.info(f"Starting HTTP scraping of {len(urls)} URLs")
        results: List[Dict] = []

        for i, url in enumerate(urls):
            try:
                logger.info(f"[{i+1}/{len(urls)}] Scraping: {url}")
                result = await self._scrape_single_url(url, i + 1, len(urls))
                if result:
                    results.append(result)
                if progress_callback:
                    try:
                        await progress_callback(i + 1, len(urls))
                    except Exception:
                        pass
                if i < len(urls) - 1:
                    await asyncio.sleep(1)
            except Exception as e:
                logger.error(f"SCRAPER ERROR: Failed to scrape {url}: {str(e)}")
                continue

        logger.info(f"HTTP scraping completed: {len(results)}/{len(urls)} successful")
        return results

    async def _scrape_single_url(self, url: str, index: int, total: int) -> Optional[Dict]:
        try:
            parsed = urlparse(url)
            referer = f"{parsed.scheme}://{parsed.netloc}"
            headers = {'Referer': referer}
            response = self.session.get(url, timeout=self.timeout, headers=headers)
            response.raise_for_status()

            content_type = response.headers.get('content-type', '').lower()
            if 'text/html' not in content_type:
                logger.warning(f"Non-HTML content type for {url}: {content_type}")
                return None

            original_soup = BeautifulSoup(response.text, 'html.parser')
            all_links_before = original_soup.find_all('a', href=True)

            links = []
            for link in all_links_before:
                href = link.get('href', '').strip()
                link_text = link.get_text().strip()
                if href and not href.startswith(('#', 'javascript:', 'mailto:', 'tel:')):
                    absolute_url = urljoin(url, href)
                    links.append({'href': absolute_url, 'text': link_text, 'title': link.get('title', '')})

            title_tag = original_soup.find('title')
            title = title_tag.get_text().strip() if title_tag else ''

            soup = BeautifulSoup(str(original_soup), 'html.parser')
            for script in soup(["script", "style", "nav", "header", "footer", "aside"]):
                script.decompose()

            text_content = soup.get_text()
            lines = (line.strip() for line in text_content.splitlines())
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            text = ' '.join(chunk for chunk in chunks if chunk)

            markdown_content = self.h.handle(str(soup))

            return {
                'url': url,
                'title': title,
                'content': text,
                'markdown': markdown_content,
                'links': links,
                'metadata': {
                    'scraped_at': datetime.utcnow().isoformat(),
                    'status_code': response.status_code,
                    'content_type': content_type,
                    'content_length': len(text),
                    'links_found': len(links),
                },
            }

        except requests.exceptions.Timeout:
            logger.error(f"SCRAPER ERROR: Timeout after {self.timeout}s for {url}")
            return None
        except requests.exceptions.RequestException:
            # Final fallback: httpx with HTTP/2 and redirects
            try:
                with httpx.Client(http2=True, follow_redirects=True, timeout=self.timeout + 10, verify=False, headers={
                    'User-Agent': self.session.headers.get('User-Agent', ''),
                    'Accept': self.session.headers.get('Accept', ''),
                    'Accept-Language': self.session.headers.get('Accept-Language', ''),
                    'Accept-Encoding': 'gzip, deflate',
                }) as client:
                    r = client.get(url)
                    r.raise_for_status()
                    ctype = r.headers.get('content-type', '').lower()
                    if 'text/html' not in ctype:
                        return None
                    original_soup = BeautifulSoup(r.text, 'html.parser')
                    title_tag = original_soup.find('title')
                    title = title_tag.get_text().strip() if title_tag else ''
                    links = []
                    for link in original_soup.find_all('a', href=True):
                        href = link.get('href', '').strip()
                        link_text = link.get_text().strip()
                        if href and not href.startswith(('#', 'javascript:', 'mailto:', 'tel:')):
                            absolute_url = urljoin(url, href)
                            links.append({'href': absolute_url, 'text': link_text, 'title': link.get('title', '')})
                    soup = BeautifulSoup(str(original_soup), 'html.parser')
                    for script in soup(["script", "style", "nav", "header", "footer", "aside"]):
                        script.decompose()
                    text_content = soup.get_text()
                    lines = (line.strip() for line in text_content.splitlines())
                    chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
                    text = ' '.join(chunk for chunk in chunks if chunk)
                    markdown_content = self.h.handle(str(soup))
                    return {
                        'url': url,
                        'title': title,
                        'content': text,
                        'markdown': markdown_content,
                        'links': links,
                        'metadata': {
                            'scraped_at': datetime.utcnow().isoformat(),
                            'status_code': r.status_code,
                            'content_type': ctype,
                            'content_length': len(text),
                            'links_found': len(links),
                        },
                    }
            except Exception as e3:
                logger.error(f"SCRAPER ERROR: Final httpx fallback failed for {url}: {e3}")
                return None
        except Exception as e:
            logger.error(f"SCRAPER ERROR: Unexpected error scraping {url}: {str(e)}")
            return None

    async def save_content(self, content: Dict, url: str) -> Optional[str]:
        try:
            parsed = urlparse(url)
            filename = f"{parsed.netloc}{parsed.path}".replace('/', '_').replace(':', '_')
            if filename.endswith('_'):
                filename = filename[:-1]
            filename = f"{filename}.json"
            file_path = self.output_dir / filename
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(content, f, indent=2, ensure_ascii=False)
            return str(file_path)
        except Exception as e:
            logger.error(f"Failed to save content for {url}: {str(e)}")
            return None


# Alias for compatibility with crawler import
EnhancedScraper = SimpleHTTPScraper



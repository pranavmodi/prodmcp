import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
import time
import logging
from pathlib import Path
import re
import random
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import json
from http.cookiejar import LWPCookieJar
import urllib3
from typing import Set, List, Tuple, Optional
import os

# Disable SSL warnings
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Use app's logger instead of basic config
logger = logging.getLogger('app.utils.scrapers.crawler')

# Browser headers to mimic real browser traffic
HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36',
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
    'Accept-Language': 'en-US,en;q=0.9',
    'Accept-Encoding': 'gzip, deflate',
    'Connection': 'keep-alive',
    'Upgrade-Insecure-Requests': '1',
    'Cache-Control': 'no-cache',
    'Pragma': 'no-cache'
}

class WebsiteCrawler:
    """Advanced web crawler with intelligent filtering and session management."""
    
    def __init__(self, data_dir: str = "./scraped_pages"):
        self.data_dir = Path(data_dir)
        self.cookie_file = self.data_dir / "crawler_cookies.txt"
        self.accepted_file = self.data_dir / "accepted_urls.txt"
        self.rejected_file = self.data_dir / "rejected_urls.txt"
        
        # Ensure data directory exists
        self.data_dir.mkdir(exist_ok=True)
        
        # Load existing URLs
        self.existing_accepted = self._load_existing_urls(self.accepted_file)
        self.existing_rejected = self._load_existing_urls(self.rejected_file)
        
        logger.info(f"Initialized crawler with {len(self.existing_accepted)} existing accepted URLs")
        logger.info(f"Initialized crawler with {len(self.existing_rejected)} existing rejected URLs")
    
    def _load_existing_urls(self, file_path: Path) -> Set[str]:
        """Load existing URLs from file into a set."""
        try:
            if file_path.exists():
                with open(file_path, 'r') as f:
                    return set(line.strip() for line in f if line.strip())
        except Exception as e:
            logger.error(f"Failed to load URLs from {file_path}: {e}")
        return set()
    
    def _save_urls(self, file_path: Path, urls: Set[str]):
        """Save URLs to file."""
        try:
            with open(file_path, 'w') as f:
                for url in sorted(urls):
                    f.write(f"{url}\n")
            logger.info(f"Saved {len(urls)} URLs to {file_path}")
        except Exception as e:
            logger.error(f"Failed to save URLs to {file_path}: {e}")
    
    def create_session(self):
        """Create a session with retry strategy and cookie persistence"""
        session = requests.Session()
        
        # Set up cookie handling
        cookie_jar = LWPCookieJar()
        if self.cookie_file.exists():
            try:
                cookie_jar.load(str(self.cookie_file), ignore_discard=True)
                logger.debug(f"Loaded cookies from {self.cookie_file}")
            except Exception as e:
                logger.warning(f"Failed to load cookies: {e}")
        
        retry_strategy = Retry(
            total=3,
            backoff_factor=2,
            status_forcelist=[403, 429, 500, 502, 503, 504],
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        session.mount("http://", adapter)
        session.mount("https://", adapter)
        session.headers.update(HEADERS)
        
        # Disable SSL verification
        session.verify = False
        
        return session, cookie_jar
    
    def save_cookies(self, cookie_jar):
        """Save cookies for future use"""
        try:
            cookie_jar.save(str(self.cookie_file), ignore_discard=True)
        except Exception as e:
            logger.error(f"Failed to save cookies: {e}")
    
    def should_reject_url(self, url: str) -> bool:
        """Filter function to determine if a URL should be rejected."""
        parsed_url = urlparse(url)
        path = parsed_url.path.lower()
        
        # Pattern to match years between 2000-2024
        year_pattern = re.compile(r'/20[0-2][0-9]/')
        
        # Always accept the root page
        if path == "" or path == "/":
            logger.debug(f"Accepting root URL {url}: root page")
            return False
        
        # Reject conditions with logging - but be less aggressive
        if '/blog/' in path and path != '/blog/':
            logger.debug(f"Rejecting URL {url}: contains /blog/ (but not root blog)")
            return True
        if path.startswith('/request'):
            logger.debug(f"Rejecting URL {url}: starts with /request")
            return True
        if bool(year_pattern.search(url)) and len(path.split('/')) > 2:
            logger.debug(f"Rejecting URL {url}: contains year pattern in deep path")
            return True
        if 'email-protection' in url:
            logger.debug(f"Rejecting URL {url}: contains email-protection")
            return True
        if '/locations/' in path and len(path.split('/')) > 2:
            logger.debug(f"Rejecting URL {url}: contains /locations/ in deep path")
            return True
        if '/news/' in path and len(path.split('/')) > 2:
            logger.debug(f"Rejecting URL {url}: contains /news in deep path")
            return True
            
        # Accept by default
        logger.debug(f"Accepting URL {url}: passed all filters")
        return False
    
    def crawl_website(self, base_url: str, max_pages: int = 100, delay_range: Tuple[float, float] = (5, 10)) -> Tuple[Set[str], Set[str]]:
        """
        Crawl a website and return new accepted and rejected URLs.
        
        Args:
            base_url: The starting URL to crawl
            max_pages: Maximum number of pages to crawl
            delay_range: Tuple of (min_delay, max_delay) in seconds
            
        Returns:
            Tuple of (new_accepted_urls, new_rejected_urls)
        """
        visited = self.existing_accepted.union(self.existing_rejected)
        to_visit = [base_url]
        new_accepted = set()
        new_rejected = set()
        
        # Create a session for all requests
        session, cookie_jar = self.create_session()
        
        # First, try to establish a session with the main page
        try:
            logger.info(f"Initializing session with {base_url}")
            response = session.get(base_url, timeout=30.0)
            response.raise_for_status()
            
            # Log response details for debugging
            logger.debug(f"Response status: {response.status_code}")
            logger.debug(f"Response headers: {dict(response.headers)}")
            logger.debug(f"Response length: {len(response.text)}")
            
            self.save_cookies(cookie_jar)
            time.sleep(random.uniform(*delay_range))
        except Exception as e:
            logger.error(f"Failed to initialize session with {base_url}: {e}")
            return new_accepted, new_rejected
        
        page_count = 0
        while to_visit and page_count < max_pages:
            url = to_visit.pop(0)
            if url in visited:
                continue
                
            visited.add(url)
            page_count += 1
            
            # Filter URL
            if self.should_reject_url(url):
                if url not in self.existing_rejected:
                    new_rejected.add(url)
                continue
                
            if url not in self.existing_accepted:
                new_accepted.add(url)
                
            try:
                logger.info(f"[{page_count}/{max_pages}] Crawling URL: {url}")
                
                # Add referrer header from base_url
                local_headers = HEADERS.copy()
                local_headers['Referer'] = base_url
                session.headers.update(local_headers)
                
                response = session.get(url, timeout=30.0)
                response.raise_for_status()
                
                # Verify content type
                content_type = response.headers.get('content-type', '').lower()
                if 'text/html' not in content_type and 'application/xhtml' not in content_type:
                    logger.warning(f"Unexpected content type for {url}: {content_type}")
                    if url in new_accepted:
                        new_accepted.remove(url)
                    continue
                    
                # Check response length
                if len(response.text) == 0:
                    logger.warning(f"Empty response from {url}")
                    if url in new_accepted:
                        new_accepted.remove(url)
                    continue
                
                # Save cookies after successful request
                self.save_cookies(cookie_jar)
                
                soup = BeautifulSoup(response.text, 'html.parser')
                
                # Check if we got a real page (not a Cloudflare challenge)
                if any(text in soup.text for text in [
                    "Just a moment",
                    "Checking your browser",
                    "Please wait while we verify",
                    "Please turn JavaScript on"
                ]):
                    logger.warning(f"Received challenge page for {url}")
                    if url in new_accepted:
                        new_accepted.remove(url)
                    continue
                
                # Try to find main content
                main_content = None
                content_selectors = [
                    'article',
                    'main',
                    '.content',
                    '.main-content',
                    '#content',
                    '#main-content',
                    '.post-content',
                    '.entry-content',
                    '.article-content',
                    '.page-content'
                ]
                
                for selector in content_selectors:
                    if selector.startswith(('.', '#')):
                        main_content = soup.select_one(selector)
                    else:
                        main_content = soup.find(selector)
                    if main_content:
                        logger.debug(f"Found content using selector: {selector}")
                        break
                
                if not main_content:
                    main_content = soup.find('body')
                    if main_content:
                        # Clean up body content
                        for element in main_content.find_all(['script', 'style', 'nav', 'header', 'footer', 'aside']):
                            element.decompose()
                
                if not main_content:
                    logger.warning(f"No content found in {url}")
                    if url in new_accepted:
                        new_accepted.remove(url)
                    continue
                
                # Extract links for further crawling
                for link in soup.find_all("a", href=True):
                    href = link['href']
                    absolute_url = urljoin(url, href)
                    
                    # Only consider links within the same domain
                    if urlparse(absolute_url).netloc != urlparse(base_url).netloc:
                        continue
                        
                    if absolute_url not in visited:
                        to_visit.append(absolute_url)
                
                # Randomized delay between requests
                delay = random.uniform(*delay_range)
                logger.debug(f"Waiting {delay:.2f} seconds before next request")
                time.sleep(delay)
                
            except requests.exceptions.RequestException as e:
                logger.error(f"Failed to fetch {url}: {str(e)}")
                if url in new_accepted:
                    new_accepted.remove(url)
                
                # Add a longer delay after errors
                time.sleep(random.uniform(10, 15))
                continue
        
        # Update and save URL lists
        if new_accepted or new_rejected:
            self.existing_accepted.update(new_accepted)
            self.existing_rejected.update(new_rejected)
            
            self._save_urls(self.accepted_file, self.existing_accepted)
            self._save_urls(self.rejected_file, self.existing_rejected)
            
            logger.info(f"Added {len(new_accepted)} new URLs to accepted list")
            logger.info(f"Added {len(new_rejected)} new URLs to rejected list")
        else:
            logger.info("No new URLs found")
        
        return new_accepted, new_rejected
    
    def get_crawl_stats(self) -> dict:
        """Get statistics about the crawling operation."""
        return {
            "total_accepted": len(self.existing_accepted),
            "total_rejected": len(self.existing_rejected),
            "total_urls": len(self.existing_accepted) + len(self.existing_rejected),
            "accepted_file": str(self.accepted_file),
            "rejected_file": str(self.rejected_file),
            "cookie_file": str(self.cookie_file)
        }
    
    def clear_crawl_data(self):
        """Clear all crawl data and start fresh."""
        try:
            if self.accepted_file.exists():
                self.accepted_file.unlink()
            if self.rejected_file.exists():
                self.rejected_file.unlink()
            if self.cookie_file.exists():
                self.cookie_file.unlink()
            
            self.existing_accepted.clear()
            self.existing_rejected.clear()
            
            logger.info("Cleared all crawl data")
        except Exception as e:
            logger.error(f"Failed to clear crawl data: {e}")

# Convenience function for backward compatibility
def crawl_website(base_url: str, accepted_file: str, rejected_file: str, existing_accepted: set = None, existing_rejected: set = None):
    """
    Legacy function for backward compatibility.
    Creates a crawler instance and crawls the website.
    """
    crawler = WebsiteCrawler()
    return crawler.crawl_website(base_url)

if __name__ == "__main__":
    # Example usage
    base_urls = [
        "https://www.precisemri.com",
        "https://radflow360.com"
    ]
    
    crawler = WebsiteCrawler()
    
    # Crawl each website
    for base_url in base_urls:
        logger.info(f"\nStarting crawl of {base_url}")
        new_accepted, new_rejected = crawler.crawl_website(base_url, max_pages=50)
        
        # Print stats
        stats = crawler.get_crawl_stats()
        logger.info(f"Crawl completed. Stats: {stats}") 
import requests
from bs4 import BeautifulSoup
from urllib.parse import urlparse
import logging
from typing import Tuple, List, Dict, Optional
from pathlib import Path
import time
import random

from crawler import WebsiteCrawler

class WebScraper:
    def __init__(self, data_dir: str = "./scraped_pages"):
        self.data_dir = Path(data_dir)
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        })
        
        # Initialize crawler with the same data directory
        self.crawler = WebsiteCrawler(data_dir)
        
        # Ensure data directory exists
        self.data_dir.mkdir(exist_ok=True)
        
        logging.info(f"WebScraper initialized with data directory: {self.data_dir.absolute()}")
    
    def extract_domain(self, url: str) -> str:
        """Extract domain from URL for filename."""
        parsed = urlparse(url)
        domain = parsed.netloc
        if domain.startswith('www.'):
            domain = domain[4:]
        return domain
    
    def scrape_url(self, url: str) -> Tuple[str, str]:
        """
        Scrape a URL and return the HTML content and domain.
        
        Args:
            url: The URL to scrape
            
        Returns:
            Tuple of (html_content, domain)
            
        Raises:
            requests.RequestException: If the request fails
            ValueError: If the URL is invalid
        """
        try:
            logging.info(f"Scraping URL: {url}")
            
            # Validate URL
            if not url.startswith(('http://', 'https://')):
                url = 'https://' + url
            
            # Make the request
            response = self.session.get(url, timeout=30)
            response.raise_for_status()
            
            # Ensure proper encoding
            response.encoding = response.apparent_encoding
            
            # Parse with BeautifulSoup
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Clean up the HTML (remove scripts, styles, etc.)
            for script in soup(["script", "style", "nav", "footer"]):
                script.decompose()
            
            # Get the cleaned HTML
            html_content = str(soup)
            
            # Extract domain for filename
            domain = self.extract_domain(url)
            
            logging.info(f"Successfully scraped {url}, domain: {domain}")
            return html_content, domain
            
        except requests.RequestException as e:
            logging.error(f"Request failed for {url}: {e}")
            raise
        except Exception as e:
            logging.error(f"Unexpected error scraping {url}: {e}")
            raise
    
    def get_page_info(self, html_content: str) -> dict:
        """Extract basic information from HTML content."""
        soup = BeautifulSoup(html_content, 'html.parser')
        
        title = soup.find('title')
        title_text = title.get_text().strip() if title else "No title"
        
        h1_tags = soup.find_all('h1')
        h1_texts = [h1.get_text().strip() for h1 in h1_tags]
        
        h2_tags = soup.find_all('h2')
        h2_texts = [h2.get_text().strip() for h2 in h2_tags]
        
        p_tags = soup.find_all('p')
        p_texts = [p.get_text().strip() for p in p_tags]
        
        return {
            "title": title_text,
            "h1_count": len(h1_texts),
            "h2_count": len(h2_texts),
            "paragraph_count": len(p_texts),
            "total_text_length": len(html_content)
        }
    
    def _check_file_exists(self, domain: str, url: str) -> bool:
        """Check if a file already exists for the given URL."""
        try:
            # Create domain-specific directory path
            domain_dir = self.data_dir / domain
            
            # Create filename from URL path
            parsed_url = urlparse(url)
            path_parts = [p for p in parsed_url.path.strip('/').split('/') if p]
            
            if not path_parts:
                filename = "index.html"
            else:
                # Create a safe filename from path parts
                safe_parts = []
                for part in path_parts:
                    # Remove or replace unsafe characters
                    safe_part = "".join(c for c in part if c.isalnum() or c in ('-', '_'))
                    if safe_part:
                        safe_parts.append(safe_part)
                
                if safe_parts:
                    filename = "_".join(safe_parts) + ".html"
                else:
                    filename = "page.html"
            
            # Check if file exists
            file_path = domain_dir / filename
            return file_path.exists()
            
        except Exception as e:
            logging.error(f"Error checking if file exists for {url}: {e}")
            return False

    def crawl_and_scrape_website(self, base_url: str, max_pages: int = 50, delay_range: Tuple[float, float] = (3, 7)) -> Dict[str, List[str]]:
        """
        Crawl a website and then scrape all discovered URLs.
        
        Args:
            base_url: The starting URL to crawl
            max_pages: Maximum number of pages to crawl
            delay_range: Tuple of (min_delay, max_delay) in seconds between scrapes
            
        Returns:
            Dictionary with 'scraped_files' and 'failed_urls' lists
        """
        logging.info(f"Starting crawl and scrape operation for {base_url}")
        
        # First, crawl the website to discover URLs
        new_accepted, new_rejected = self.crawler.crawl_website(
            base_url, 
            max_pages=max_pages, 
            delay_range=(5, 10)  # Longer delays for crawling
        )
        
        # Get all accepted URLs (both new and existing)
        all_accepted = self.crawler.existing_accepted
        
        if not all_accepted:
            logging.warning(f"No URLs discovered for {base_url}")
            return {"scraped_files": [], "failed_urls": []}
        
        logging.info(f"Total accepted URLs: {len(all_accepted)}")
        logging.info(f"New URLs discovered: {len(new_accepted)}")
        
        # Now scrape each accepted URL (both new and existing)
        scraped_files = []
        failed_urls = []
        
        for i, url in enumerate(all_accepted, 1):
            try:
                logging.info(f"Scraping {i}/{len(all_accepted)}: {url}")
                
                # Check if file already exists
                domain = self.extract_domain(url)
                if self._check_file_exists(domain, url):
                    logging.info(f"Skipping {url} as file already exists.")
                    continue

                # Scrape the URL
                html_content, domain = self.scrape_url(url)
                
                # Save the HTML content
                filename = self._save_html_content(domain, url, html_content)
                scraped_files.append(filename)
                
                # Add delay between scrapes to be respectful
                if i < len(all_accepted):  # Don't delay after the last one
                    delay = random.uniform(*delay_range)
                    logging.debug(f"Waiting {delay:.2f} seconds before next scrape")
                    time.sleep(delay)
                    
            except Exception as e:
                logging.error(f"Failed to scrape {url}: {e}")
                failed_urls.append(url)
        
        logging.info(f"Crawl and scrape completed. {len(scraped_files)} files scraped, {len(failed_urls)} failed")
        
        return {
            "scraped_files": scraped_files,
            "failed_urls": failed_urls
        }
    
    def _save_html_content(self, domain: str, url: str, html_content: str) -> str:
        """Save HTML content to a file and return the filename."""
        try:
            # Create domain-specific directory
            domain_dir = self.data_dir / domain
            domain_dir.mkdir(exist_ok=True)
            
            # Create filename from URL path
            parsed_url = urlparse(url)
            path_parts = [p for p in parsed_url.path.strip('/').split('/') if p]
            
            if not path_parts:
                filename = "index.html"
            else:
                # Create a safe filename from path parts
                safe_parts = []
                for part in path_parts:
                    # Remove or replace unsafe characters
                    safe_part = "".join(c for c in part if c.isalnum() or c in ('-', '_'))
                    if safe_part:
                        safe_parts.append(safe_part)
                
                if safe_parts:
                    filename = "_".join(safe_parts) + ".html"
                else:
                    filename = "page.html"
            
            # Ensure filename is unique
            counter = 1
            original_filename = filename
            while (domain_dir / filename).exists():
                name, ext = original_filename.rsplit('.', 1)
                filename = f"{name}_{counter}.{ext}"
                counter += 1
            
            file_path = domain_dir / filename
            
            # Save the HTML content
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(html_content)
            
            logging.info(f"Saved HTML content to {file_path}")
            return str(file_path)
            
        except Exception as e:
            logging.error(f"Failed to save HTML content for {url}: {e}")
            raise
    
    def get_crawl_stats(self) -> dict:
        """Get statistics about the crawling operation."""
        return self.crawler.get_crawl_stats()
    
    def clear_crawl_data(self):
        """Clear all crawl data and start fresh."""
        self.crawler.clear_crawl_data()
    
    def get_discovered_urls(self) -> Tuple[set, set]:
        """Get the currently discovered accepted and rejected URLs."""
        return self.crawler.existing_accepted, self.crawler.existing_rejected 
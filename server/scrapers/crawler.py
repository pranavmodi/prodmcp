import os
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
import time
import logging
from pathlib import Path
import random
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from http.cookiejar import LWPCookieJar
import urllib3

# Disable SSL warnings
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

logger = logging.getLogger(__name__)

HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36',
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
    'Accept-Language': 'en-US,en;q=0.9',
    'Accept-Encoding': 'gzip, deflate',
    'Connection': 'keep-alive',
    'Upgrade-Insecure-Requests': '1',
    'Cache-Control': 'no-cache',
    'Pragma': 'no-cache',
    'Sec-Fetch-Dest': 'document',
    'Sec-Fetch-Mode': 'navigate',
    'Sec-Fetch-Site': 'none'
}


def _cookie_file_path() -> str:
    data_dir = os.getenv("DATA_DIR", "./scraped_pages")
    Path(data_dir).mkdir(parents=True, exist_ok=True)
    return str(Path(data_dir) / "crawler_cookies.txt")


def create_session():
    """Create a session with retry strategy and cookie persistence"""
    session = requests.Session()

    cookie_path = _cookie_file_path()
    cookie_jar = LWPCookieJar(cookie_path)
    try:
        cookie_jar.load(ignore_discard=True)
    except FileNotFoundError:
        pass
    session.cookies = cookie_jar

    retry_strategy = Retry(
        total=3,
        backoff_factor=2,
        status_forcelist=[403, 429, 500, 502, 503, 504],
    )
    adapter = HTTPAdapter(max_retries=retry_strategy)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    session.headers.update(HEADERS)

    session.verify = False
    return session, cookie_jar


def save_cookies(cookie_jar):
    try:
        cookie_jar.save(ignore_discard=True)
    except Exception as e:
        logger.error(f"Failed to save cookies: {e}")


def _normalize_for_compare(input_url: str) -> tuple:
    """Return (netloc, path) lowercased, without query/fragment, with single leading slash path, no trailing slash."""
    try:
        parsed = urlparse(input_url)
        netloc = parsed.netloc.lower()
        path = parsed.path or ""
        if not netloc and not path:
            # Treat bare strings as paths/prefixes
            raw = input_url
            # strip scheme-like patterns and leading '@'
            raw = raw.lstrip('@')
            if raw.startswith('http://') or raw.startswith('https://'):
                parsed2 = urlparse(raw)
                netloc = parsed2.netloc.lower()
                path = parsed2.path or ""
            else:
                netloc = ""
                path = raw
        path = path.strip()
        if not path.startswith('/'):
            path = '/' + path if path else '/'
        # drop trailing slash except root
        if len(path) > 1 and path.endswith('/'):
            path = path.rstrip('/')
        return netloc, path
    except Exception:
        # Fallback: treat whole string as path
        raw = (input_url or '').lower().lstrip('@')
        if not raw.startswith('/'):
            raw = '/' + raw if raw else '/'
        if len(raw) > 1 and raw.endswith('/'):
            raw = raw.rstrip('/')
        return "", raw


def should_reject_url(url: str, exclusion_urls: list = None) -> bool:
    if not exclusion_urls:
        return False

    # Normalize target URL
    try:
        # remove query/fragment for compare
        base_no_qf = url.split('#', 1)[0].split('?', 1)[0]
    except Exception:
        base_no_qf = url
    url_netloc, url_path = _normalize_for_compare(base_no_qf)

    for exclusion in exclusion_urls:
        if not exclusion:
            continue
        # Clean and normalize exclusion
        e = str(exclusion).strip().lstrip('@')
        if not e:
            continue
        e_no_qf = e.split('#', 1)[0].split('?', 1)[0]
        _ex_netloc, ex_path = _normalize_for_compare(e_no_qf)

        # Path-only prefix match (domain ignored). We already stay within base domain elsewhere.
        if url_path.startswith(ex_path):
            logger.debug(f"Rejecting URL {url}: path prefix matches exclusion '{exclusion}'")
            return True
    return False


def load_existing_urls(file_path: str) -> set:
    try:
        with open(file_path, 'r') as f:
            return set(line.strip() for line in f if line.strip())
    except FileNotFoundError:
        return set()


def append_new_urls(accepted_file: str, rejected_file: str, new_accepted: set, new_rejected: set):
    if new_accepted:
        with open(accepted_file, 'a') as f:
            for url in sorted(new_accepted):
                f.write(f"{url}\n")
    if new_rejected:
        with open(rejected_file, 'a') as f:
            for url in sorted(new_rejected):
                f.write(f"{url}\n")


async def crawl_website_enhanced(base_url: str, accepted_file: str, rejected_file: str, existing_accepted: set = None, existing_rejected: set = None, exclusion_urls: list = None, progress_callback=None):
    if existing_accepted is None:
        existing_accepted = set()
    if existing_rejected is None:
        existing_rejected = set()

    visited = existing_accepted.union(existing_rejected)
    to_visit = [base_url]
    new_accepted = set()
    new_rejected = set()

    try:
        from .enhanced_scraper import EnhancedScraper
        scraper = EnhancedScraper(tenant_id="crawler_temp")
        logger.info(f"Using HTTP scraper for crawling with timeout: {getattr(scraper, 'timeout', 30)}s")
    except ImportError:
        logger.warning("Enhanced scraper not available, falling back to basic HTTP")
        return crawl_website(base_url, accepted_file, rejected_file, existing_accepted, existing_rejected, exclusion_urls)

    # initial progress callback
    if progress_callback:
        try:
            progress_callback(0, len(to_visit), len(existing_accepted), len(existing_rejected), 0)
        except Exception:
            pass

    while to_visit:
        url = to_visit.pop(0)
        if url in visited:
            continue
        visited.add(url)

        if should_reject_url(url, exclusion_urls):
            if url not in existing_rejected:
                new_rejected.add(url)
            continue

        if url not in existing_accepted:
            new_accepted.add(url)

        try:
            logger.info(f"[{len(visited)}/{len(visited) + len(to_visit)}] Enhanced crawling URL: {url}")
            results = await scraper.scrape_urls([url])
            if results and results[0] is not None:
                result = results[0]
                content_length = len(result.get('content', ''))
                if content_length < 100:
                    logger.warning(f"Insufficient content found in {url} ({content_length} chars)")
                    if url in new_accepted:
                        new_accepted.remove(url)
                    continue

                scraped_links = result.get('links', [])
                markdown_content = result.get('markdown', '')
                if markdown_content:
                    import re as _re
                    markdown_links = _re.findall(r'\[([^\]]+)\]\(([^)]+)\)', markdown_content)
                    for text, href in markdown_links:
                        if href not in [link.get('href', '') for link in scraped_links]:
                            scraped_links.append({'href': href, 'text': text})

                for link_info in scraped_links:
                    if isinstance(link_info, dict) and 'href' in link_info:
                        href = link_info['href']
                    else:
                        continue
                    absolute_url = urljoin(url, href)
                    if urlparse(absolute_url).netloc != urlparse(base_url).netloc:
                        continue
                    if absolute_url in visited or absolute_url in to_visit:
                        continue
                    if href.startswith('#'):
                        continue
                    to_visit.append(absolute_url)

                # progress update
                if progress_callback:
                    try:
                        visited_count = len(visited)
                        queue_count = len(to_visit)
                        accepted_count = len(existing_accepted) + len(new_accepted)
                        rejected_count = len(existing_rejected) + len(new_rejected)
                        denom = max(visited_count + queue_count, 1)
                        percent = int(visited_count * 100 / denom)
                        if queue_count > 0 and percent >= 96:
                            percent = 95
                        progress_callback(visited_count, queue_count, accepted_count, rejected_count, percent)
                    except Exception:
                        pass

                time.sleep(random.uniform(2, 5))
            else:
                if url in new_accepted:
                    new_accepted.remove(url)
                continue
        except Exception as e:
            logger.error(f"CRAWLER ERROR: Failed to discover links from {url}: {str(e)}")
            if url in new_accepted:
                new_accepted.remove(url)
            continue

    if new_accepted or new_rejected:
        # Ensure same URL doesn't end up in both files: prefer rejected if conflict
        conflicts = new_accepted.intersection(new_rejected)
        if conflicts:
            new_accepted.difference_update(conflicts)
        append_new_urls(accepted_file, rejected_file, new_accepted, new_rejected)
        logger.info(f"Enhanced crawler: Added {len(new_accepted)} new URLs to {accepted_file}")
        logger.info(f"Enhanced crawler: Added {len(new_rejected)} new URLs to {rejected_file}")
    else:
        logger.info("Enhanced crawler: No new URLs found")

    # final progress
    if progress_callback:
        try:
            visited_count = len(visited)
            queue_count = len(to_visit)
            accepted_count = len(existing_accepted) + len(new_accepted)
            rejected_count = len(existing_rejected) + len(new_rejected)
            progress_callback(visited_count, queue_count, accepted_count, rejected_count, 100)
        except Exception:
            pass

    return new_accepted, new_rejected


def crawl_website(base_url: str, accepted_file: str, rejected_file: str, existing_accepted: set = None, existing_rejected: set = None, exclusion_urls: list = None, progress_callback=None):
    if existing_accepted is None:
        existing_accepted = set()
    if existing_rejected is None:
        existing_rejected = set()

    visited = existing_accepted.union(existing_rejected)
    to_visit = [base_url]
    new_accepted = set()
    new_rejected = set()

    session, cookie_jar = create_session()

    try:
        logger.info(f"Initializing session with {base_url}")
        response = session.get(base_url, timeout=30.0)
        response.raise_for_status()
        save_cookies(cookie_jar)
        time.sleep(random.uniform(5, 8))
    except Exception as e:
        logger.error(f"Failed to initialize session with {base_url}: {e}")
        return new_accepted, new_rejected

    # initial progress callback
    if progress_callback:
        try:
            progress_callback(0, len(to_visit), len(existing_accepted), len(existing_rejected), 0)
        except Exception:
            pass

    while to_visit:
        url = to_visit.pop(0)
        if url in visited:
            continue
        visited.add(url)

        if should_reject_url(url, exclusion_urls):
            if url not in existing_rejected:
                new_rejected.add(url)
            continue

        if url not in existing_accepted:
            new_accepted.add(url)

        try:
            logger.info(f"[{len(visited)}/{len(visited) + len(to_visit)}] Crawling URL: {url}")
            local_headers = HEADERS.copy()
            local_headers['Referer'] = base_url
            session.headers.update(local_headers)
            response = session.get(url, timeout=30.0)
            response.raise_for_status()

            content_type = response.headers.get('content-type', '').lower()
            if 'text/html' not in content_type and 'application/xhtml' not in content_type:
                if url in new_accepted:
                    new_accepted.remove(url)
                continue

            if len(response.text) == 0:
                if url in new_accepted:
                    new_accepted.remove(url)
                continue

            save_cookies(cookie_jar)

            soup = BeautifulSoup(response.text, 'html.parser')
            if any(text in soup.text for text in [
                "Just a moment",
                "Checking your browser",
                "Please wait while we verify",
                "Please turn JavaScript on",
            ]):
                if url in new_accepted:
                    new_accepted.remove(url)
                continue

            for link in soup.find_all("a", href=True):
                href = link['href']
                absolute_url = urljoin(url, href)
                if urlparse(absolute_url).netloc != urlparse(base_url).netloc:
                    continue
                if absolute_url not in visited:
                    to_visit.append(absolute_url)

            # progress update
            if progress_callback:
                try:
                    visited_count = len(visited)
                    queue_count = len(to_visit)
                    accepted_count = len(existing_accepted) + len(new_accepted)
                    rejected_count = len(existing_rejected) + len(new_rejected)
                    denom = max(visited_count + queue_count, 1)
                    percent = int(visited_count * 100 / denom)
                    if queue_count > 0 and percent >= 96:
                        percent = 95
                    progress_callback(visited_count, queue_count, accepted_count, rejected_count, percent)
                except Exception:
                    pass

            delay = random.uniform(5, 10)
            time.sleep(delay)
        except requests.exceptions.RequestException:
            if url in new_accepted:
                new_accepted.remove(url)
            time.sleep(random.uniform(10, 15))
            continue

    if new_accepted or new_rejected:
        # Ensure same URL doesn't end up in both files: prefer rejected if conflict
        conflicts = new_accepted.intersection(new_rejected)
        if conflicts:
            new_accepted.difference_update(conflicts)
        append_new_urls(accepted_file, rejected_file, new_accepted, new_rejected)
        logger.info(f"Added {len(new_accepted)} new URLs to {accepted_file}")
        logger.info(f"Added {len(new_rejected)} new URLs to {rejected_file}")
    else:
        logger.info("No new URLs found")

    # final progress
    if progress_callback:
        try:
            visited_count = len(visited)
            queue_count = len(to_visit)
            accepted_count = len(existing_accepted) + len(new_accepted)
            rejected_count = len(existing_rejected) + len(new_rejected)
            progress_callback(visited_count, queue_count, accepted_count, rejected_count, 100)
        except Exception:
            pass

    return new_accepted, new_rejected



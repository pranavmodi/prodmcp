import os
import glob
from pathlib import Path
from typing import List
import logging

class HTMLStorage:
    def __init__(self, data_dir: str = "./scraped_pages"):
        self.data_dir = Path(data_dir)
        self._ensure_directory_exists()
    
    def _ensure_directory_exists(self):
        """Create the data directory if it doesn't exist."""
        self.data_dir.mkdir(parents=True, exist_ok=True)
        logging.info(f"Storage directory ensured: {self.data_dir}")
    
    def save_html(self, domain: str, html_content: str) -> str:
        """Save HTML content to a file with domain and timestamp."""
        from datetime import datetime
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{domain}_{timestamp}.html"
        filepath = self.data_dir / filename
        
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(html_content)
            logging.info(f"HTML saved: {filepath}")
            return filename
        except Exception as e:
            logging.error(f"Error saving HTML: {e}")
            raise
    
    def get_all_html_files(self) -> List[Path]:
        """Get all HTML files in the storage directory and subdirectories."""
        html_files = []
        
        # Look for HTML files in the main directory
        main_pattern = self.data_dir / "*.html"
        main_files = list(Path(main_pattern).parent.glob("*.html"))
        html_files.extend(main_files)
        
        # Look for HTML files in subdirectories (domain directories)
        for subdir in self.data_dir.iterdir():
            if subdir.is_dir():
                subdir_html_files = list(subdir.glob("*.html"))
                html_files.extend(subdir_html_files)
        
        # Remove duplicates and sort
        html_files = list(set(html_files))
        html_files.sort()
        
        logging.info(f"Found {len(html_files)} HTML files in {self.data_dir}")
        for file in html_files:
            logging.debug(f"  - {file}")
        
        return html_files
    
    def read_html_file(self, filepath: Path) -> str:
        """Read HTML content from a file."""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                return f.read()
        except Exception as e:
            logging.error(f"Error reading HTML file {filepath}: {e}")
            raise
    
    def get_storage_info(self) -> dict:
        """Get information about stored files."""
        html_files = self.get_all_html_files()
        return {
            "total_files": len(html_files),
            "files": [f.name for f in html_files],
            "storage_path": str(self.data_dir)
        } 
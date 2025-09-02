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
        """Get all HTML/JSON files in the storage directory and subdirectories."""
        files: List[Path] = []

        # Look for HTML and JSON files in the main directory
        main_dir = self.data_dir
        if main_dir.exists():
            files.extend(list(main_dir.glob("*.html")))
            files.extend(list(main_dir.glob("*.json")))

        # Look for files in subdirectories (domain directories)
        if main_dir.exists():
            for subdir in main_dir.iterdir():
                if subdir.is_dir():
                    files.extend(list(subdir.glob("*.html")))
                    files.extend(list(subdir.glob("*.json")))

        # Remove duplicates and sort
        files = sorted(set(files))

        logging.info(f"Found {len(files)} data files in {self.data_dir}")
        for file in files:
            logging.debug(f"  - {file}")

        return files
    
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
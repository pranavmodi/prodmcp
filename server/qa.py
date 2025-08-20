import openai
from bs4 import BeautifulSoup
import logging
from typing import List, Dict
import os

class QASystem:
    def __init__(self, api_key: str):
        self.client = openai.OpenAI(api_key=api_key)
        self.model = "gpt-4o-mini"
    
    def extract_text_from_html(self, html_content: str) -> str:
        """
        Extract meaningful text content from HTML.
        
        Args:
            html_content: Raw HTML string
            
        Returns:
            Extracted text content
        """
        try:
            soup = BeautifulSoup(html_content, 'html.parser')
            
            # Remove script and style elements
            for script in soup(["script", "style", "nav", "footer", "header"]):
                script.decompose()
            
            # Extract text from important elements
            text_parts = []
            
            # Title
            title = soup.find('title')
            if title:
                text_parts.append(f"Title: {title.get_text().strip()}")
            
            # Headings
            for heading in soup.find_all(['h1', 'h2', 'h3']):
                text_parts.append(f"{heading.name.upper()}: {heading.get_text().strip()}")
            
            # Paragraphs
            for p in soup.find_all('p'):
                text = p.get_text().strip()
                if len(text) > 20:  # Only include substantial paragraphs
                    text_parts.append(text)
            
            # Lists
            for ul in soup.find_all(['ul', 'ol']):
                for li in ul.find_all('li'):
                    text = li.get_text().strip()
                    if text:
                        text_parts.append(f"• {text}")
            
            # Combine all text
            extracted_text = "\n\n".join(text_parts)
            
            # Clean up extra whitespace
            extracted_text = " ".join(extracted_text.split())
            
            logging.info(f"Extracted {len(extracted_text)} characters of text")
            return extracted_text
            
        except Exception as e:
            logging.error(f"Error extracting text from HTML: {e}")
            return ""
    
    def build_context_from_files(self, html_files: List[str]) -> str:
        """
        Build context from multiple HTML files.
        
        Args:
            html_files: List of HTML file paths
            
        Returns:
            Combined context string
        """
        contexts = []
        
        for file_path in html_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    html_content = f.read()
                
                extracted_text = self.extract_text_from_html(html_content)
                if extracted_text:
                    filename = os.path.basename(file_path)
                    contexts.append(f"--- File: {filename} ---\n{extracted_text}")
                    
            except Exception as e:
                logging.error(f"Error reading file {file_path}: {e}")
                continue
        
        combined_context = "\n\n".join(contexts)
        logging.info(f"Built context from {len(html_files)} files, total length: {len(combined_context)}")
        return combined_context
    
    def ask_question(self, question: str, context: str) -> str:
        """
        Ask a question to OpenAI using the provided context.
        
        Args:
            question: User's question
            context: Context from scraped HTML files
            
        Returns:
            AI-generated answer
        """
        try:
            # Prepare the prompt
            system_prompt = """You are a helpful assistant that answers questions based on the content of scraped web pages. 
            Use the provided context to answer questions accurately and concisely. 
            If the context doesn't contain enough information to answer the question, say so clearly.
            Always cite which file(s) the information comes from when possible."""
            
            user_prompt = f"""Context from scraped web pages:

{context}

Question: {question}

Please answer the question based on the context above."""

            # Make the API call
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                max_tokens=1000,
                temperature=0.3
            )
            
            answer = response.choices[0].message.content.strip()
            logging.info(f"Generated answer for question: {question[:50]}...")
            
            return answer
            
        except Exception as e:
            logging.error(f"Error calling OpenAI API: {e}")
            return f"Sorry, I encountered an error while processing your question: {str(e)}"
    
    def get_available_files_summary(self, html_files: List[str]) -> Dict:
        """
        Get a summary of available HTML files for context.
        
        Args:
            html_files: List of HTML file paths
            
        Returns:
            Summary dictionary
        """
        summary = {
            "total_files": len(html_files),
            "files": []
        }
        
        for file_path in html_files:
            try:
                filename = os.path.basename(file_path)
                with open(file_path, 'r', encoding='utf-8') as f:
                    html_content = f.read()
                
                # Extract basic info
                soup = BeautifulSoup(html_content, 'html.parser')
                title = soup.find('title')
                title_text = title.get_text().strip() if title else "No title"
                
                summary["files"].append({
                    "filename": filename,
                    "title": title_text,
                    "size": len(html_content)
                })
                
            except Exception as e:
                logging.error(f"Error processing file {file_path}: {e}")
                continue
        
        return summary 
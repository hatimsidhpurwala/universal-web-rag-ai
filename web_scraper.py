"""
Web Scraping Module for Universal Web RAG AI System
Uses requests + BeautifulSoup to extract content from websites
"""

import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
import re
import time
from typing import List, Dict, Optional


class WebScraper:
    """Scraper for extracting text content from websites"""
    
    def __init__(self, max_pages: int = 1, delay: float = 1.0):
        self.max_pages = max_pages
        self.delay = delay
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        })
        self.visited_urls = set()
        
    def fetch_page(self, url: str) -> Optional[str]:
        """Fetch HTML content from a URL"""
        try:
            response = self.session.get(url, timeout=30)
            response.raise_for_status()
            return response.text
        except requests.RequestException as e:
            print(f"Error fetching {url}: {e}")
            return None
    
    def clean_html(self, html: str) -> BeautifulSoup:
        """Clean HTML by removing scripts, styles, and other non-content elements"""
        soup = BeautifulSoup(html, 'html.parser')
        
        # Remove script and style elements
        for element in soup(['script', 'style', 'nav', 'footer', 'header', 'aside', 'form']):
            element.decompose()
        
        # Remove elements with common non-content classes/ids
        non_content_patterns = [
            'cookie', 'privacy', 'terms', 'login', 'signup', 'register',
            'advertisement', 'ad-', 'sidebar', 'menu', 'navigation',
            'comment', 'social', 'share', 'related'
        ]
        
        for element in soup.find_all(True):
            element_id = element.get('id', '').lower()
            element_class = ' '.join(element.get('class', [])).lower()
            
            for pattern in non_content_patterns:
                if pattern in element_id or pattern in element_class:
                    element.decompose()
                    break
        
        return soup
    
    def extract_text_content(self, soup: BeautifulSoup, base_url: str) -> Dict:
        """Extract headings and paragraphs from cleaned HTML"""
        content = {
            'url': base_url,
            'title': '',
            'headings': [],
            'paragraphs': [],
            'full_text': ''
        }
        
        # Extract title
        title_tag = soup.find('title')
        if title_tag:
            content['title'] = title_tag.get_text(strip=True)
        
        # Extract headings (h1, h2, h3)
        for level in ['h1', 'h2', 'h3']:
            headings = soup.find_all(level)
            for heading in headings:
                text = heading.get_text(strip=True)
                if text:
                    content['headings'].append({
                        'level': level,
                        'text': text
                    })
        
        # Extract paragraphs
        paragraphs = soup.find_all('p')
        for p in paragraphs:
            text = p.get_text(strip=True)
            if text:
                content['paragraphs'].append(text)
        
        # Also extract text from divs that look like content
        content_divs = soup.find_all(['div', 'article', 'section'], class_=re.compile(r'content|main|body|text', re.I))
        for div in content_divs:
            text = div.get_text(separator=' ', strip=True)
            if text and len(text) > 50:
                # Split long content into chunks
                sentences = re.split(r'(?<=[.!?])\s+', text)
                for sentence in sentences:
                    if len(sentence.strip()) > 30:
                        content['paragraphs'].append(sentence.strip())
        
        # Create full text
        all_text = []
        if content['title']:
            all_text.append(content['title'])
        for heading in content['headings']:
            all_text.append(heading['text'])
        all_text.extend(content['paragraphs'])
        
        content['full_text'] = '\n\n'.join(all_text)
        
        return content
    
    def scrape_website(self, url: str, progress_callback=None) -> Dict:
        """Main method to scrape a website"""
        if progress_callback:
            progress_callback("Scraping Started", 0.1)
        
        # Fetch the main page
        html = self.fetch_page(url)
        if not html:
            return {'error': f'Failed to fetch {url}'}
        
        if progress_callback:
            progress_callback("Data Extracted", 0.3)
        
        # Clean HTML
        soup = self.clean_html(html)
        
        if progress_callback:
            progress_callback("Data Cleaned", 0.5)
        
        # Extract content
        content = self.extract_text_content(soup, url)
        
        return content
    
    def get_text_chunks(self, content: Dict, chunk_size: int = 500, overlap: int = 50) -> List[str]:
        """Split content into overlapping chunks for embedding"""
        chunks = []
        
        # Add title as a chunk
        if content.get('title'):
            chunks.append(f"Title: {content['title']}")
        
        # Add headings as chunks
        for heading in content.get('headings', []):
            chunks.append(f"{heading['level'].upper()}: {heading['text']}")
        
        # Split paragraphs into chunks
        all_text = ' '.join(content.get('paragraphs', []))
        
        # Simple sentence-based chunking
        sentences = re.split(r'(?<=[.!?])\s+', all_text)
        current_chunk = []
        current_length = 0
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            
            sentence_length = len(sentence)
            
            if current_length + sentence_length > chunk_size and current_chunk:
                chunks.append(' '.join(current_chunk))
                # Keep overlap
                overlap_sentences = current_chunk[-2:] if len(current_chunk) > 2 else current_chunk
                current_chunk = overlap_sentences + [sentence]
                current_length = sum(len(s) for s in current_chunk)
            else:
                current_chunk.append(sentence)
                current_length += sentence_length
        
        # Add remaining chunk
        if current_chunk:
            chunks.append(' '.join(current_chunk))
        
        return chunks


def scrape_url(url: str, progress_callback=None) -> Dict:
    """Convenience function to scrape a URL"""
    scraper = WebScraper()
    return scraper.scrape_website(url, progress_callback)

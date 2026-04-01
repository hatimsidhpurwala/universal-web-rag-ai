"""
Data Cleaning Module for Universal Web RAG AI System
Cleans and preprocesses scraped text data
"""

import re
from typing import List, Set
import unicodedata


class DataCleaner:
    """Cleans and preprocesses text data for RAG pipeline"""
    
    # Irrelevant words and patterns to remove
    IRRELEVANT_WORDS = {
        'privacy', 'policy', 'terms', 'conditions', 'cookie', 'cookies',
        'login', 'signin', 'signup', 'register', 'password', 'username',
        'copyright', 'rights', 'reserved', 'trademark', 'legal',
        'contact', 'support', 'help', 'faq', 'sitemap',
        'subscribe', 'newsletter', 'unsubscribe',
        'advertisement', 'sponsored', 'promoted',
        'share', 'facebook', 'twitter', 'linkedin', 'instagram',
        'click', 'here', 'read', 'more', 'learn', 'continue'
    }
    
    def __init__(self, min_length: int = 20):
        self.min_length = min_length
        
    def normalize_text(self, text: str) -> str:
        """Normalize text: lowercase, remove extra whitespace, fix encoding"""
        # Normalize unicode characters
        text = unicodedata.normalize('NFKD', text)
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove special characters but keep basic punctuation
        text = re.sub(r'[^\w\s.,!?;:()-]', '', text)
        
        return text.strip()
    
    def remove_html_entities(self, text: str) -> str:
        """Remove HTML entities and codes"""
        # Remove HTML entities like &nbsp;, &amp;, etc.
        text = re.sub(r'&[a-zA-Z]+;', ' ', text)
        text = re.sub(r'&#\d+;', ' ', text)
        return text
    
    def remove_urls(self, text: str) -> str:
        """Remove URLs from text"""
        url_pattern = r'https?://\S+|www\.\S+'
        return re.sub(url_pattern, '', text)
    
    def remove_emails(self, text: str) -> str:
        """Remove email addresses from text"""
        email_pattern = r'\S+@\S+'
        return re.sub(email_pattern, '', text)
    
    def remove_phone_numbers(self, text: str) -> str:
        """Remove phone numbers from text"""
        phone_pattern = r'[\+]?[(]?[0-9]{3}[)]?[-\s\.]?[0-9]{3}[-\s\.]?[0-9]{4,6}'
        return re.sub(phone_pattern, '', text)
    
    def contains_irrelevant_words(self, text: str) -> bool:
        """Check if text contains irrelevant words"""
        words = set(text.lower().split())
        return len(words & self.IRRELEVANT_WORDS) > len(words) * 0.3
    
    def is_meaningful_text(self, text: str) -> bool:
        """Check if text is meaningful (not just navigation/footer text)"""
        # Check minimum length
        if len(text) < self.min_length:
            return False
        
        # Check word count
        words = text.split()
        if len(words) < 5:
            return False
        
        # Check for high ratio of irrelevant words
        if self.contains_irrelevant_words(text):
            return False
        
        # Check for repetitive content
        unique_words = set(words)
        if len(unique_words) / len(words) < 0.3:  # Too repetitive
            return False
        
        return True
    
    def clean_chunk(self, text: str) -> str:
        """Apply all cleaning steps to a single text chunk"""
        # Remove HTML entities
        text = self.remove_html_entities(text)
        
        # Remove URLs
        text = self.remove_urls(text)
        
        # Remove emails
        text = self.remove_emails(text)
        
        # Remove phone numbers
        text = self.remove_phone_numbers(text)
        
        # Normalize text
        text = self.normalize_text(text)
        
        return text
    
    def clean_chunks(self, chunks: List[str]) -> List[str]:
        """Clean a list of text chunks"""
        cleaned_chunks = []
        seen_chunks = set()  # For deduplication
        
        for chunk in chunks:
            # Clean the chunk
            cleaned = self.clean_chunk(chunk)
            
            # Skip if too short
            if len(cleaned) < self.min_length:
                continue
            
            # Skip if not meaningful
            if not self.is_meaningful_text(cleaned):
                continue
            
            # Skip duplicates (using first 100 chars as fingerprint)
            fingerprint = cleaned[:100].lower()
            if fingerprint in seen_chunks:
                continue
            
            seen_chunks.add(fingerprint)
            cleaned_chunks.append(cleaned)
        
        return cleaned_chunks
    
    def clean_content(self, content: dict) -> dict:
        """Clean all content from scraped data"""
        cleaned = {
            'url': content.get('url', ''),
            'title': '',
            'headings': [],
            'paragraphs': [],
            'chunks': []
        }
        
        # Clean title
        if content.get('title'):
            cleaned['title'] = self.clean_chunk(content['title'])
        
        # Clean headings
        for heading in content.get('headings', []):
            cleaned_heading = self.clean_chunk(heading['text'])
            if self.is_meaningful_text(cleaned_heading):
                cleaned['headings'].append({
                    'level': heading['level'],
                    'text': cleaned_heading
                })
        
        # Clean paragraphs
        for paragraph in content.get('paragraphs', []):
            cleaned_para = self.clean_chunk(paragraph)
            if self.is_meaningful_text(cleaned_para):
                cleaned['paragraphs'].append(cleaned_para)
        
        # Create chunks from cleaned content
        all_text = []
        if cleaned['title']:
            all_text.append(f"title: {cleaned['title']}")
        
        for heading in cleaned['headings']:
            all_text.append(f"{heading['level']}: {heading['text']}")
        
        all_text.extend(cleaned['paragraphs'])
        
        # Split into chunks
        cleaned['chunks'] = self.create_chunks(all_text)
        
        return cleaned
    
    def create_chunks(self, texts: List[str], chunk_size: int = 500, overlap: int = 50) -> List[str]:
        """Create overlapping chunks from texts"""
        chunks = []
        
        for text in texts:
            if len(text) <= chunk_size:
                chunks.append(text)
            else:
                # Split into sentences
                sentences = re.split(r'(?<=[.!?])\s+', text)
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
        
        # Final cleaning of chunks
        return self.clean_chunks(chunks)


def clean_data(chunks: List[str], min_length: int = 20) -> List[str]:
    """Convenience function to clean text chunks"""
    cleaner = DataCleaner(min_length=min_length)
    return cleaner.clean_chunks(chunks)

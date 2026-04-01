"""
LLM Module for Universal Web RAG AI System
Uses HuggingFace Transformers with flan-t5-base for text generation
"""

import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from typing import List
import streamlit as st


class LLMGenerator:
    """Text generation using HuggingFace models"""
    
    def __init__(self, model_name: str = 'google/flan-t5-base'):
        self.model_name = model_name
        self.tokenizer = None
        self.model = None
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
    def load_model(self):
        """Lazy load the LLM model"""
        if self.model is None:
            st.info(f"Loading LLM model: {self.model_name}...")
            st.info(f"Using device: {self.device}")
            
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name)
            self.model.to(self.device)
            
            st.success("LLM model loaded successfully!")
        return self.model, self.tokenizer
    
    def create_prompt(self, query: str, context: List[str]) -> str:
        """Create a prompt for the LLM with query and context"""
        # Join context pieces
        context_text = "\n\n".join(context)
        
        # Create prompt
        prompt = f"""Based on the following context, please answer the question. If the answer is not in the context, say "I don't have enough information to answer this question."

Context:
{context_text}

Question: {query}

Answer:"""
        
        return prompt
    
    def generate_answer(
        self, 
        query: str, 
        context: List[str],
        max_length: int = 512,
        min_length: int = 20,
        temperature: float = 0.7,
        num_beams: int = 4
    ) -> str:
        """Generate answer based on query and context"""
        model, tokenizer = self.load_model()
        
        # Create prompt
        prompt = self.create_prompt(query, context)
        
        st.info("Generating answer...")
        
        # Tokenize input
        inputs = tokenizer(
            prompt,
            return_tensors='pt',
            max_length=1024,
            truncation=True,
            padding=True
        ).to(self.device)
        
        # Generate
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_length=max_length,
                min_length=min_length,
                temperature=temperature,
                num_beams=num_beams,
                early_stopping=True,
                do_sample=temperature > 0,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id
            )
        
        # Decode output
        answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        return answer.strip()
    
    def generate_summary(self, texts: List[str], max_length: int = 200) -> str:
        """Generate a summary of the texts"""
        model, tokenizer = self.load_model()
        
        # Join texts
        context = " ".join(texts[:5])  # Use first 5 chunks
        
        prompt = f"Summarize the following text in a few sentences:\n\n{context}\n\nSummary:"
        
        inputs = tokenizer(
            prompt,
            return_tensors='pt',
            max_length=1024,
            truncation=True
        ).to(self.device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_length=max_length,
                num_beams=4,
                early_stopping=True
            )
        
        summary = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return summary.strip()


# Singleton instance for reuse
_llm_generator = None


def get_llm_generator(model_name: str = 'google/flan-t5-base') -> LLMGenerator:
    """Get or create singleton LLM generator"""
    global _llm_generator
    if _llm_generator is None:
        _llm_generator = LLMGenerator(model_name)
    return _llm_generator


def generate_answer(query: str, context: List[str]) -> str:
    """Convenience function to generate answer"""
    generator = get_llm_generator()
    return generator.generate_answer(query, context)


def generate_summary(texts: List[str]) -> str:
    """Convenience function to generate summary"""
    generator = get_llm_generator()
    return generator.generate_summary(texts)

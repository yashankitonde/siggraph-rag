#!/usr/bin/env python3
"""
RAG Generation Pipeline for SIGGRAPH 2025 Papers.

Uses the retrieval pipeline to find relevant chunks,
then generates an answer using an LLM via OpenRouter API.

Usage:
    from rag_generate import RAGGenerator, GenerationConfig, SYSTEM_PROMPT

    generator = RAGGenerator()
    result = generator.generate("What is 3D Gaussian Splatting?")
    print(result["answer"])
"""

import os
import requests
from typing import Optional
from dataclasses import dataclass

from dotenv import load_dotenv
load_dotenv()

from retrieval_pipeline import RetrievalPipeline, RetrievalResult


# =============================================================================
# SYSTEM PROMPT - This tells the LLM how to behave
# =============================================================================
SYSTEM_PROMPT = """You are an expert research assistant specializing in computer graphics, specifically SIGGRAPH 2025 papers.

Your task is to answer questions using ONLY the provided research paper excerpts.

Rules:
1. Cite sources using [Paper Title] format
2. Be comprehensive and technically accurate
3. If the excerpts don't contain the answer, say so
4. Use LaTeX for math: $inline$ or $$block$$
5. Do NOT make up information not in the excerpts
6. Do NOT include a References section at the end
"""


# =============================================================================
# QUERY REFINEMENT PROMPT
# =============================================================================
QUERY_REFINEMENT_PROMPT = """You are an expert at refining search queries for academic paper retrieval.

Given a user's question, rewrite it as a clear, focused search query that will retrieve the most relevant research papers.

Keep it concise (under 20 words). Focus on key technical terms.

User question: {query}

Refined search query:"""


# =============================================================================
# CONFIGURATION
# =============================================================================
@dataclass
class GenerationConfig:
    """Configuration for the RAG generator."""
    # Read the model from .env (LLM_MODEL) so the free model set there is
    # honored; fall back to the original default only if the env is missing.
    llm_model: str = os.getenv("LLM_MODEL") or "openai/gpt-4o"  # Model to use for answer generation
    temperature: float = 0.1  # Low temperature for factual answers
    max_tokens: int = 2000  # Max length of generated answer
    openrouter_api_key: Optional[str] = None  # Will load from env if not set
    refine_query: bool = True  # Whether to refine queries before retrieval
    # Refinement is a tiny task - reuse the same (free) model from .env
    # instead of the paid gpt-3.5-turbo the starter suggested.
    refinement_model: str = os.getenv("LLM_MODEL") or "openai/gpt-3.5-turbo"  # Cheaper model for refinement
    retrieval_top_k: int = 8  # Number of chunks to retrieve
    # api_server.py passes these two when it builds the config - without
    # them the Step 4 import swap would crash with "unexpected keyword".
    llm_provider: str = "openrouter"  # Accepted for api_server compatibility
    use_reranker: bool = True  # Passed through to the retrieval pipeline


# =============================================================================
# RAG GENERATOR CLASS
# =============================================================================
class RAGGenerator:
    """
    Main RAG class - this is what api_server.py uses!

    Flow:
    1. Refine the user's query (optional)
    2. Retrieve relevant chunks using the retrieval pipeline
    3. Format chunks into context
    4. Generate answer using LLM
    5. Return answer with source metadata
    """

    def __init__(self, config: Optional[GenerationConfig] = None, retrieval_pipeline=None):
        """
        Initialize the RAG generator.

        TODO:
        1. Set config (use default if not provided):
           self.config = config or GenerationConfig()
        2. Initialize the retrieval pipeline:
           self.retrieval = retrieval_pipeline or RetrievalPipeline()
        3. Get OpenRouter API key (from config or environment):
           self.openrouter_api_key = self.config.openrouter_api_key or os.getenv("OPENROUTER_API_KEY")
        4. Validate API key exists:
           if not self.openrouter_api_key:
               raise ValueError("OPENROUTER_API_KEY not set")
        5. Store the base URL:
           self.openrouter_base_url = "https://openrouter.ai/api/v1"

        Args:
            config: Optional configuration object
            retrieval_pipeline: Optional pre-initialized retrieval pipeline
        """
        # Set up config and the retrieval pipeline (reuse a pre-built one if
        # given, which avoids rebuilding the BM25 index), and pass the
        # use_reranker flag down to the pipeline where reranking happens.
        self.config = config or GenerationConfig()
        self.retrieval = retrieval_pipeline or RetrievalPipeline()
        self.retrieval.config.use_reranker = self.config.use_reranker

        # Resolve the API key (config wins, else .env) and fail fast if it's
        # missing rather than hitting a 401 mid-request later.
        self.openrouter_api_key = self.config.openrouter_api_key or os.getenv("OPENROUTER_API_KEY")
        if not self.openrouter_api_key:
            raise ValueError("OPENROUTER_API_KEY not set")
        self.openrouter_base_url = "https://openrouter.ai/api/v1"

    def refine_query(self, query: str) -> str:
        """
        Use LLM to improve the search query (optional but helps retrieval).

        TODO:
        1. If self.config.refine_query is False, return query unchanged
        2. Build the prompt using QUERY_REFINEMENT_PROMPT.format(query=query)
        3. Build headers:
           {"Authorization": f"Bearer {self.openrouter_api_key}", "Content-Type": "application/json"}
        4. Build payload:
           {
               "model": self.config.refinement_model,
               "messages": [{"role": "user", "content": prompt}],
               "temperature": 0.3,
               "max_tokens": 100
           }
        5. Make POST request to f"{self.openrouter_base_url}/chat/completions"
        6. If request fails, return original query (don't crash)
        7. Parse response and extract the refined query from the response
           refined = response_json["choices"][0]["message"]["content"].strip()
        8. Return refined query (strip any quotes)

        Args:
            query: Original user query

        Returns:
            Refined query (or original if refinement disabled/fails)
        """
        # Skip the API call entirely if refinement is turned off.
        if not self.config.refine_query:
            return query

        # Build the refinement request: the prompt template filled with the
        # user's question, sent at a slightly higher temperature with a small
        # token budget (a search query is short).
        prompt = QUERY_REFINEMENT_PROMPT.format(query=query)
        headers = {
            "Authorization": f"Bearer {self.openrouter_api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.config.refinement_model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.3,
            "max_tokens": 100,
        }

        # Refinement is a nice-to-have, so on ANY failure (bad status, network
        # error, empty answer) fall back to the original query. Strip quotes
        # the model sometimes wraps around its answer.
        try:
            response = requests.post(
                f"{self.openrouter_base_url}/chat/completions",
                headers=headers,
                json=payload,
                timeout=30,
            )
            if response.status_code != 200:
                return query
            refined = response.json()["choices"][0]["message"]["content"].strip()
            if not refined:
                return query
            return refined.strip('"').strip("'")
        except Exception:
            return query

    def _format_context(self, results: list[RetrievalResult]) -> str:
        """
        Format retrieved chunks into a context string for the LLM.

        TODO:
        1. Build a list of formatted source strings
        2. For each result (enumerate with index starting at 1):
           formatted = f'''
           --- Source {i} ---
           Title: {result.title}
           Authors: {result.authors}
           Section: {result.chunk_section}

           Content:
           {result.text}
           '''
        3. Join all formatted strings with newlines
        4. Return the combined context string

        Args:
            results: List of RetrievalResult objects

        Returns:
            Formatted context string
        """
        # Turn each retrieved chunk into a numbered block carrying the metadata
        # the LLM needs for [Paper Title] citations, then join them into one
        # context string.
        formatted_sources = []
        for i, result in enumerate(results, 1):
            formatted = f"""
--- Source {i} ---
Title: {result.title}
Authors: {result.authors}
Section: {result.chunk_section}

Content:
{result.text}
"""
            formatted_sources.append(formatted)
        return "\n".join(formatted_sources)

    def _build_sources_metadata(self, results: list[RetrievalResult]) -> dict:
        """
        Build list of unique source papers for citations.
        The frontend displays these as clickable source links.

        TODO:
        1. Create a dict to track seen titles (for deduplication):
           seen = {}
        2. For each result:
           - If title not in seen:
             - Add to seen with value:
               {
                   "title": result.title,
                   "authors": result.authors,
                   "pdf_url": result.pdf_url,
                   "github_link": result.github_link,
                   "video_link": result.video_link,
                   "acm_url": result.acm_url,
                   "abstract_url": result.abstract_url,
               }
        3. Return list(seen.values())

        Args:
            results: List of RetrievalResult objects

        Returns:
            List of unique source metadata dicts
        """
        # Deduplicate by title (multiple chunks of one paper would otherwise
        # show as duplicate citations) - first chunk of each paper wins, and
        # dict insertion order keeps the best paper first.
        seen = {}
        for result in results:
            if result.title not in seen:
                seen[result.title] = {
                    "title": result.title,
                    "authors": result.authors,
                    "pdf_url": result.pdf_url,
                    "github_link": result.github_link,
                    "video_link": result.video_link,
                    "acm_url": result.acm_url,
                    "abstract_url": result.abstract_url,
                }
        # NOTE: the TODO says return list(seen.values()), but api_server.py
        # calls .values() on this return value (and the mock returns a dict),
        # so we return the dict; callers turn it into a list themselves.
        return seen

    def _call_llm(self, query: str, context: str) -> str:
        """
        Call OpenRouter API to generate an answer.

        TODO:
        1. Build the user message:
           user_message = f'''Based on the following research paper excerpts, answer this question.

           Question: {query}

           Research Paper Excerpts:
           {context}

           Remember to cite papers using [Paper Title] format.'''
        2. Build headers:
           {"Authorization": f"Bearer {self.openrouter_api_key}", "Content-Type": "application/json"}
        3. Build payload:
           {
               "model": self.config.llm_model,
               "messages": [
                   {"role": "system", "content": SYSTEM_PROMPT},
                   {"role": "user", "content": user_message}
               ],
               "temperature": self.config.temperature,
               "max_tokens": self.config.max_tokens
           }
        4. Make POST request to f"{self.openrouter_base_url}/chat/completions"
        5. Check response status, raise error if not 200
        6. Parse response and extract answer from the response
           answer = response_json["choices"][0]["message"]["content"]
        7. Return the answer

        Args:
            query: User's question
            context: Formatted context from retrieved chunks

        Returns:
            Generated answer string
        """
        # Build the request: the system prompt sets the rules (cite, don't
        # invent), the user message carries the question + retrieved excerpts
        # (the "augmented" part of RAG - the LLM answers from THIS, not memory).
        user_message = f"""Based on the following research paper excerpts, answer this question.

Question: {query}

Research Paper Excerpts:
{context}

Remember to cite papers using [Paper Title] format."""
        headers = {
            "Authorization": f"Bearer {self.openrouter_api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.config.llm_model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_message},
            ],
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
        }

        # Send it (generous timeout since the answer can be long). Unlike
        # refinement, generation failing IS fatal - there is no answer without
        # it - so raise with details, then return the generated text.
        response = requests.post(
            f"{self.openrouter_base_url}/chat/completions",
            headers=headers,
            json=payload,
            timeout=120,
        )
        if response.status_code != 200:
            raise RuntimeError(
                f"OpenRouter chat request failed ({response.status_code}): {response.text}"
            )
        return response.json()["choices"][0]["message"]["content"]

    def generate(self, query: str, top_k: Optional[int] = None, return_sources: bool = True) -> dict:
        """
        Full RAG pipeline - retrieve relevant chunks and generate an answer.
        THIS IS THE MAIN METHOD THAT api_server.py CALLS!

        TODO:
        1. Refine the query:
        2. Retrieve relevant chunks:
        3. Handle empty results:
           if not results:
               return {
                   "query": query,
                   "refined_query": refined,
                   "answer": "I couldn't find any relevant papers to answer this question.",
                   "sources": []
               }
        4. Format context from results:
        5. Generate answer using LLM:
        6. Build and return response dict:
           {
               "query": query,
               "refined_query": refined,
               "answer": answer,
               "sources": self._build_sources_metadata(results) if return_sources else []
           }

        Args:
            query: User's question
            top_k: Number of chunks to retrieve (uses config default if None)
            return_sources: Whether to include source metadata

        Returns:
            Dict with query, refined_query, answer, and sources
        """
        # Refine the question into a better search query, then retrieve the
        # relevant chunks for it.
        refined = self.refine_query(query)
        results = self.retrieval.retrieve(refined, top_k=top_k or self.config.retrieval_top_k)

        # If nothing relevant was found, say so honestly instead of letting the
        # LLM improvise an answer from thin air.
        if not results:
            return {
                "query": query,
                "refined_query": refined,
                "answer": "I couldn't find any relevant papers to answer this question.",
                "sources": [],
            }

        # Format the chunks into context, generate the cited answer (passing
        # the ORIGINAL query - the answer should address what the user asked),
        # and package it with the deduplicated source metadata for the UI.
        context = self._format_context(results)
        answer = self._call_llm(query, context)
        return {
            "query": query,
            "refined_query": refined,
            "answer": answer,
            "sources": list(self._build_sources_metadata(results).values()) if return_sources else [],
        }


# =============================================================================
# CLI FOR TESTING
# =============================================================================
if __name__ == "__main__":
    import sys

    query = sys.argv[1] if len(sys.argv) > 1 else "What is 3D Gaussian Splatting?"

    print("Initializing RAG Generator...")
    generator = RAGGenerator()

    print(f"\nQuery: {query}")
    print("=" * 60)

    result = generator.generate(query)

    print(f"Refined Query: {result.get('refined_query', 'N/A')}")
    print("=" * 60)
    print("\nAnswer:")
    print(result['answer'])
    print("=" * 60)
    print(f"\nSources: {len(result.get('sources', []))} papers")
    for source in result.get('sources', []):
        print(f"  - {source['title']}")

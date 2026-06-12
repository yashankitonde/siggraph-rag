#!/usr/bin/env python3
"""
Retrieval Pipeline for SIGGRAPH 2025 Papers.

Implements hybrid search:
1. Semantic search (embeddings via OpenRouter + Qdrant Cloud)
2. Keyword search (BM25 - runs locally)
3. Reranking (Cohere API - optional)

Usage:
    from retrieval_pipeline import RetrievalPipeline

    pipeline = RetrievalPipeline()
    results = pipeline.retrieve("3D Gaussian Splatting", top_k=5)
"""

import json
import os
import re
import requests
import numpy as np
from typing import Optional, List
from dataclasses import dataclass
from qdrant_client import QdrantClient
from rank_bm25 import BM25Okapi

from dotenv import load_dotenv
load_dotenv()

# Must match the collection name used in upload_to_qdrant.py
COLLECTION_NAME = "siggraph2025_papers"


@dataclass
class RetrievalResult:
    """
    Represents a single search result.
    The api_server.py expects these exact fields - do not change!
    """
    chunk_id: str
    paper_id: str
    title: str
    authors: str
    text: str
    score: float
    chunk_type: str = ""
    chunk_section: str = ""
    pdf_url: Optional[str] = None
    github_link: Optional[str] = None
    video_link: Optional[str] = None
    acm_url: Optional[str] = None
    abstract_url: Optional[str] = None


@dataclass
class RetrievalPipelineConfig:
    """Configuration for the retrieval pipeline."""
    qdrant_url: str
    qdrant_api_key: str
    openrouter_api_key: str
    embedding_model: str = "baai/bge-large-en-v1.5"
    chunks_path: str = "./chunks.json"
    semantic_weight: float = 0.7
    bm25_weight: float = 0.3
    use_reranker: bool = True
    cohere_api_key: Optional[str] = None


class OpenRouterEmbedder:
    """
    Generate embeddings using OpenRouter API.
    Used to embed user queries for semantic search.
    """

    def __init__(self, api_key: str, model: str = "baai/bge-large-en-v1.5"):
        """
        Initialize the embedder.

        TODO:
        1. Store the api_key: self.api_key = api_key
        2. Store the model: self.model = model
        3. Store the base URL: self.base_url = "https://openrouter.ai/api/v1"

        Args:
            api_key: OpenRouter API key
            model: Embedding model to use
        """
        # Store the key, model, and base URL for later requests. The model
        # MUST match the one used to embed the chunks at upload time, or
        # query vectors and chunk vectors won't be comparable.
        self.api_key = api_key
        self.model = model
        self.base_url = "https://openrouter.ai/api/v1"

    def embed_query(self, text: str) -> np.ndarray:
        """
        Generate embedding for a single query.

        TODO:
        1. Build headers dict:
           - "Authorization": f"Bearer {self.api_key}"
           - "Content-Type": "application/json"
        2. Build payload dict:
           - "model": self.model
           - "input": text
        3. Make POST request to f"{self.base_url}/embeddings"
        4. Check response status code, raise error if not 200
        5. Parse response JSON
        6. Extract embedding: embedding = response_data["data"][0]["embedding"]
        7. Convert to numpy array and return:
           return np.array(embedding, dtype=np.float32)

        Args:
            text: Query text to embed

        Returns:
            Embedding vector as numpy array
        """
        # Build the authenticated request: which model, and the text to embed.
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model,
            "input": text,
        }
        # Send it; fail loudly on any non-200 so problems are debuggable.
        response = requests.post(
            f"{self.base_url}/embeddings",
            headers=headers,
            json=payload,
            timeout=30,
        )
        if response.status_code != 200:
            raise RuntimeError(
                f"OpenRouter embeddings request failed "
                f"({response.status_code}): {response.text}"
            )
        # Pull the embedding out of the response and return it as a numpy
        # array (one entry per input; we sent one text, so take index 0).
        response_data = response.json()
        embedding = response_data["data"][0]["embedding"]
        return np.array(embedding, dtype=np.float32)


class BM25Index:
    """
    BM25 index for keyword search.
    This runs entirely locally - no API calls needed!
    BM25 is good at finding exact keyword matches that semantic search might miss.
    """

    def __init__(self, chunks: list[dict]):
        """
        Build BM25 index from chunks.

        TODO:
        1. Store the chunks: self.chunks = chunks
        2. Create a lookup dict for quick access by chunk_id:
           self.chunk_id_to_idx = {c["chunk_id"]: i for i, c in enumerate(chunks)}
        3. Tokenize all documents (convert each chunk's text to list of words):
           self.tokenized_docs = [self._tokenize(c["text"]) for c in chunks]
        4. Create the BM25 index:
           self.bm25 = BM25Okapi(self.tokenized_docs)

        Args:
            chunks: List of chunk dictionaries from chunks.json
        """
        # Keep the chunks, tokenize every chunk's text, and build the BM25
        # index over those tokens (precomputes word statistics so searches
        # are fast). chunk_id_to_idx lets us map a chunk_id back to its row.
        self.chunks = chunks
        self.chunk_id_to_idx = {c["chunk_id"]: i for i, c in enumerate(chunks)}
        self.tokenized_docs = [self._tokenize(c["text"]) for c in chunks]
        self.bm25 = BM25Okapi(self.tokenized_docs)

    def _tokenize(self, text: str) -> list[str]:
        """
        Simple tokenization: lowercase and extract alphanumeric words.

        TODO:
        1. Convert text to lowercase
        2. Use regex to find all words (alphanumeric sequences)
        3. Return the list of tokens

        Args:
            text: Text to tokenize

        Returns:
            List of lowercase word tokens
        """
        # Lowercase and split into alphanumeric word tokens (this strips
        # punctuation too): "3D Gaussian-Splatting!" -> ["3d","gaussian","splatting"].
        text = text.lower()
        return re.findall(r"\w+", text)

    def search(self, query: str, top_k: int = 50) -> list[tuple[int, float]]:
        """
        Search for query and return top-k results.

        TODO:
        1. Tokenize the query
        2. Get BM25 scores for all documents
        3. Get indices of top-k highest scores (hint: use np.argsort)
        4. Build result list with only non-zero scores (hint: use list comprehension)
        5. Return results

        Args:
            query: Search query string
            top_k: Maximum number of results to return

        Returns:
            List of (chunk_index, score) tuples, sorted by score descending
        """
        # Score every chunk against the query (tokenized the same way as the
        # documents) - one float per chunk, 0 meaning no query words present.
        query_tokens = self._tokenize(query)
        scores = self.bm25.get_scores(query_tokens)
        # Take the top-k by score (argsort ascending, reversed), dropping any
        # zero scores since those chunks share no words with the query.
        top_indices = np.argsort(scores)[::-1][:top_k]
        return [
            (int(idx), float(scores[idx]))
            for idx in top_indices
            if scores[idx] > 0
        ]


class RetrievalPipeline:
    """
    Main retrieval pipeline combining semantic search + BM25 + reranking.
    This is what api_server.py uses to find relevant chunks.
    """

    def __init__(self, config: Optional[RetrievalPipelineConfig] = None):
        """
        Initialize all components of the retrieval pipeline.

        TODO:
        1. If config is None, create one from environment variables:
           config = RetrievalPipelineConfig(
               qdrant_url=os.getenv("QDRANT_URL"),
               qdrant_api_key=os.getenv("QDRANT_API_KEY"),
               openrouter_api_key=os.getenv("OPENROUTER_API_KEY"),
               cohere_api_key=os.getenv("COHERE_API_KEY"),
               chunks_path=os.getenv("CHUNKS_PATH", "./chunks.json"),
           )
        2. Validate required fields - raise ValueError if missing:
           - config.qdrant_url
           - config.qdrant_api_key
           - config.openrouter_api_key
        3. Initialize Qdrant client:
        4. Initialize the embedder:
        5. Load chunks from JSON file:
        6. Build BM25 index:
           self.bm25_index = BM25Index(self.chunks)
           print("BM25 index built")
        7. Store the config:
           self.config = config

        Args:
            config: Optional configuration. If None, loads from environment variables.
        """
        # If no config was passed, build one from the .env file. ("or" rather
        # than getenv's default because .env has CHUNKS_PATH= which returns "".)
        if config is None:
            config = RetrievalPipelineConfig(
                qdrant_url=os.getenv("QDRANT_URL"),
                qdrant_api_key=os.getenv("QDRANT_API_KEY"),
                openrouter_api_key=os.getenv("OPENROUTER_API_KEY"),
                cohere_api_key=os.getenv("COHERE_API_KEY"),
                chunks_path=os.getenv("CHUNKS_PATH") or "./chunks.json",
            )

        # Fail fast with a clear message if any required secret is missing -
        # nicer than a cryptic crash deep inside a library later.
        if not config.qdrant_url:
            raise ValueError("QDRANT_URL is not set (check your .env)")
        if not config.qdrant_api_key:
            raise ValueError("QDRANT_API_KEY is not set (check your .env)")
        if not config.openrouter_api_key:
            raise ValueError("OPENROUTER_API_KEY is not set (check your .env)")

        # Set up the Qdrant client and the query embedder, load all chunks
        # from disk (chunks.json wraps the list under "chunks", so unwrap it),
        # and build the local BM25 index over their texts.
        self.qdrant = QdrantClient(url=config.qdrant_url, api_key=config.qdrant_api_key)
        self.embedder = OpenRouterEmbedder(
            api_key=config.openrouter_api_key,
            model=config.embedding_model,
        )
        with open(config.chunks_path, "r") as f:
            data = json.load(f)
        self.chunks = data["chunks"] if isinstance(data, dict) else data
        print(f"Loaded {len(self.chunks)} chunks")
        self.bm25_index = BM25Index(self.chunks)
        print("BM25 index built")
        self.config = config

    def semantic_search(self, query: str, top_k: int = 30) -> list[dict]:
        """
        Perform semantic search using Qdrant.

        TODO:
        1. Embed the query:
           query_embedding = self.embedder.embed_query(query)
        2. Search Qdrant:
           results = self.qdrant.query_points(
               collection_name=COLLECTION_NAME,
               query=query_embedding.tolist(),
               limit=top_k,
               with_payload=True
           ).points
        3. Convert to list of dicts:
           return [
               {
                   "chunk_id": r.payload["chunk_id"],
                   "score": r.score,
                   "payload": r.payload
               }
               for r in results
           ]

        Args:
            query: Search query
            top_k: Number of results to return

        Returns:
            List of result dicts with chunk_id, score, and payload
        """
        # Embed the query, find the nearest vectors in Qdrant (with_payload
        # also returns each chunk's metadata), and reshape the hits into plain
        # dicts the rest of the pipeline understands.
        query_embedding = self.embedder.embed_query(query)
        results = self.qdrant.query_points(
            collection_name=COLLECTION_NAME,
            query=query_embedding.tolist(),
            limit=top_k,
            with_payload=True,
        ).points
        return [
            {"chunk_id": r.payload["chunk_id"], "score": r.score, "payload": r.payload}
            for r in results
        ]

    def bm25_search(self, query: str, top_k: int = 30) -> list[dict]:
        """
        Perform BM25 keyword search.

        TODO:
        1. Call BM25 search:
           results = self.bm25_index.search(query, top_k)
        2. Convert to list of dicts (same format as semantic_search):
           return [
               {
                   "chunk_id": self.chunks[idx]["chunk_id"],
                   "score": score,
                   "payload": self.chunks[idx]
               }
               for idx, score in results
           ]

        Args:
            query: Search query
            top_k: Number of results to return

        Returns:
            List of result dicts with chunk_id, score, and payload
        """
        # Run the local keyword search and reshape its (index, score) pairs
        # into the same dict format as semantic_search, so the two can merge.
        results = self.bm25_index.search(query, top_k)
        return [
            {"chunk_id": self.chunks[idx]["chunk_id"], "score": score, "payload": self.chunks[idx]}
            for idx, score in results
        ]

    def hybrid_search(self, query: str, semantic_top_k: int = 30, bm25_top_k: int = 30) -> list[dict]:
        """
        Combine semantic and BM25 results using weighted scoring.

        TODO:
        1. Get results from both search methods:
           semantic_results = self.semantic_search(query, semantic_top_k)
           bm25_results = self.bm25_search(query, bm25_top_k)
        2. Normalize semantic scores (divide by max score):
           if semantic_results:
               max_semantic = max(r["score"] for r in semantic_results)
               for r in semantic_results:
                   r["normalized_score"] = r["score"] / max_semantic if max_semantic > 0 else 0
        3. Normalize BM25 scores the same way
        4. Combine results into a single dict keyed by chunk_id:
           combined = {}
           For semantic results:
           - Add to combined with semantic_score and initial combined_score
           For BM25 results:
           - If chunk_id already in combined, add bm25_score and update combined_score
           - If new, add with just bm25_score
           Combined score formula:
           combined_score = (semantic_weight * semantic_score) + (bm25_weight * bm25_score)
        5. Sort by combined_score descending:
           results = sorted(combined.values(), key=lambda x: x["combined_score"], reverse=True)
        6. Return the sorted list

        Args:
            query: Search query
            semantic_top_k: Max results from semantic search
            bm25_top_k: Max results from BM25 search

        Returns:
            Combined and sorted list of results
        """
        # Run both searches on the same query.
        semantic_results = self.semantic_search(query, semantic_top_k)
        bm25_results = self.bm25_search(query, bm25_top_k)

        # Normalize each score list to 0-1 (divide by its own max). Cosine
        # scores (~0-1) and BM25 scores (0-30+) live on different scales;
        # without this, BM25 would drown out the semantic signal when added.
        if semantic_results:
            max_semantic = max(r["score"] for r in semantic_results)
            for r in semantic_results:
                r["normalized_score"] = r["score"] / max_semantic if max_semantic > 0 else 0
        if bm25_results:
            max_bm25 = max(r["score"] for r in bm25_results)
            for r in bm25_results:
                r["normalized_score"] = r["score"] / max_bm25 if max_bm25 > 0 else 0

        # Merge by chunk_id: a chunk found by both searches gets both weighted
        # contributions; one found by only a single search gets just that part.
        combined = {}
        for r in semantic_results:
            combined[r["chunk_id"]] = {
                "chunk_id": r["chunk_id"],
                "payload": r["payload"],
                "semantic_score": r["normalized_score"],
                "bm25_score": 0.0,
                "combined_score": self.config.semantic_weight * r["normalized_score"],
            }
        for r in bm25_results:
            if r["chunk_id"] in combined:
                entry = combined[r["chunk_id"]]
                entry["bm25_score"] = r["normalized_score"]
                entry["combined_score"] = (
                    self.config.semantic_weight * entry["semantic_score"]
                    + self.config.bm25_weight * entry["bm25_score"]
                )
            else:
                combined[r["chunk_id"]] = {
                    "chunk_id": r["chunk_id"],
                    "payload": r["payload"],
                    "semantic_score": 0.0,
                    "bm25_score": r["normalized_score"],
                    "combined_score": self.config.bm25_weight * r["normalized_score"],
                }

        # Sort all merged candidates best-first and return.
        return sorted(combined.values(), key=lambda x: x["combined_score"], reverse=True)

    def rerank(self, query: str, results: list[dict], top_k: int = 10) -> list[dict]:
        """
        Rerank results using Cohere API (optional but improves quality).

        TODO:
        1. If no Cohere API key or no results, return results[:top_k]
        2. Extract texts for reranking:
           texts = [r["payload"]["text"] for r in results]
        3. Call Cohere Rerank API:
           - URL: "https://api.cohere.ai/v1/rerank"
           - Headers: {"Authorization": f"Bearer {self.config.cohere_api_key}"}
           - Body: {
               "model": "rerank-english-v3.0",
               "query": query,
               "documents": texts,
               "top_n": top_k
           }
        4. Parse response and reorder results based on Cohere's ranking:
           - response.results contains items with index and relevance_score
           - Reorder your results list to match Cohere's order
           - Update each result's score with the rerank_score
        5. Return reranked results

        Note: If Cohere API fails, catch the error and fall back to returning results[:top_k]

        Args:
            query: Original query
            results: Results from hybrid_search
            top_k: Number of results to return after reranking

        Returns:
            Reranked list of results
        """
        # Without a key or any results there's nothing to rerank - just truncate.
        if not self.config.cohere_api_key or not results:
            return results[:top_k]

        # Ask Cohere to re-sort the candidate texts by true relevance, then
        # rebuild our list in Cohere's order with the new relevance scores
        # attached (Cohere's "index" points back into our results list).
        texts = [r["payload"]["text"] for r in results]
        try:
            response = requests.post(
                "https://api.cohere.ai/v1/rerank",
                headers={"Authorization": f"Bearer {self.config.cohere_api_key}"},
                json={
                    "model": "rerank-english-v3.0",
                    "query": query,
                    "documents": texts,
                    "top_n": top_k,
                },
                timeout=30,
            )
            if response.status_code != 200:
                raise RuntimeError(f"Cohere rerank failed ({response.status_code}): {response.text}")
            data = response.json()
            reranked = []
            for item in data["results"]:
                original = results[item["index"]]
                original["rerank_score"] = item["relevance_score"]
                reranked.append(original)
            return reranked
        except Exception as e:
            # Reranking is a bonus, not a requirement - if Cohere is down or
            # rate-limited, log it and fall back to the hybrid order.
            print(f"Reranking failed, falling back to hybrid order: {e}")
            return results[:top_k]

    def retrieve(self, query: str, top_k: int = 8) -> list[RetrievalResult]:
        """
        Full retrieval pipeline - THIS IS WHAT api_server.py CALLS!

        TODO:
        1. Run hybrid search to get candidates:
           candidates = self.hybrid_search(query)
        2. Rerank if enabled:
           if self.config.use_reranker:
               reranked = self.rerank(query, candidates, top_k=min(top_k * 2, len(candidates)))
           else:
               reranked = candidates
        3. Take top_k results:
           final = reranked[:top_k]
        4. Convert to RetrievalResult objects:
           return [
               RetrievalResult(
                   chunk_id=r["payload"]["chunk_id"],
                   paper_id=r["payload"]["paper_id"],
                   title=r["payload"]["title"],
                   authors=r["payload"]["authors"],
                   text=r["payload"]["text"],
                   score=r.get("rerank_score", r.get("combined_score", r.get("score", 0))),
                   chunk_type=r["payload"].get("chunk_type", ""),
                   chunk_section=r["payload"].get("chunk_section", ""),
                   pdf_url=r["payload"].get("pdf_url"),
                   github_link=r["payload"].get("github_link"),
                   video_link=r["payload"].get("video_link"),
                   acm_url=r["payload"].get("acm_url"),
                   abstract_url=r["payload"].get("abstract_url"),
               )
               for r in final
           ]

        Args:
            query: User's search query
            top_k: Number of results to return

        Returns:
            List of RetrievalResult objects ready for RAG generation
        """
        # Wide net first (hybrid search ~30-60 candidates), then optionally let
        # Cohere re-rank them - over top_k*2 so it has room to promote something
        # hybrid search under-ranked - and finally keep the top_k.
        candidates = self.hybrid_search(query)
        if self.config.use_reranker:
            reranked = self.rerank(query, candidates, top_k=min(top_k * 2, len(candidates)))
        else:
            reranked = candidates
        final = reranked[:top_k]

        # Convert the loose dicts into the typed RetrievalResult objects
        # api_server.py expects, using the best score available for each
        # (rerank_score > combined_score > raw score).
        return [
            RetrievalResult(
                chunk_id=r["payload"]["chunk_id"],
                paper_id=r["payload"]["paper_id"],
                title=r["payload"]["title"],
                authors=r["payload"]["authors"],
                text=r["payload"]["text"],
                score=r.get("rerank_score", r.get("combined_score", r.get("score", 0))),
                chunk_type=r["payload"].get("chunk_type", ""),
                chunk_section=r["payload"].get("chunk_section", ""),
                pdf_url=r["payload"].get("pdf_url"),
                github_link=r["payload"].get("github_link"),
                video_link=r["payload"].get("video_link"),
                acm_url=r["payload"].get("acm_url"),
                abstract_url=r["payload"].get("abstract_url"),
            )
            for r in final
        ]


# For testing this file directly
if __name__ == "__main__":
    import sys

    query = sys.argv[1] if len(sys.argv) > 1 else "3D Gaussian Splatting"

    print(f"Testing retrieval pipeline with query: '{query}'")
    print("=" * 60)

    pipeline = RetrievalPipeline()
    results = pipeline.retrieve(query, top_k=5)

    print(f"\nFound {len(results)} results:\n")

    for i, r in enumerate(results, 1):
        print(f"{i}. [{r.score:.4f}] {r.title[:60]}...")
        print(f"   Paper ID: {r.paper_id}")
        print(f"   Text preview: {r.text[:100]}...")
        print()

# ADR-001: Embedding provider boundary with a deterministic local fallback

## Context/Problem

Document chunks need numeric vectors before vector retrieval can be added. Tying ingestion directly to one hosted embedding API would make local development depend on credentials and network access, while hiding the boundary that a production model will eventually cross.

## Decision

Define a small `Embedder` protocol and provide a deterministic `HashingEmbedder` as the local development fallback. The fallback hashes normalized tokens into a fixed-size vector and L2-normalizes it. It is explicitly a development baseline rather than a semantic embedding model.

## Why this approach

The interface keeps ingestion and retrieval independent of a specific model vendor. The hashing implementation is deterministic, requires no secrets or downloads, and makes the embedding stage runnable and testable immediately. A hosted or local semantic model can later implement the same contract.

## Alternatives considered

- Call a hosted embedding API directly: better semantic quality, but requires credentials, network access, and vendor-specific code in the pipeline.
- Bundle a transformer model now: useful offline semantics, but adds a large dependency and model-download/runtime cost before retrieval behavior is established.
- Skip embeddings until deployment: simpler short term, but prevents an end-to-end local retrieval path.

## Tradeoffs/risks

Feature hashing has collisions and captures token overlap rather than semantic similarity, so recommendation-quality retrieval must not rely on it in production. The fixed dimensionality also becomes part of the vector-store configuration and must match the selected production embedder.

## How it fits the architecture

Parsing produces normalized text, chunking creates traceable source spans, and the embedder converts those chunks into vectors. Retrieval will consume vectors through this boundary without needing to know which embedding provider generated them.

## Interview explanation

"I separated the embedding contract from the provider and shipped a deterministic hashing fallback first. That let the whole pipeline stay runnable without API keys while keeping the production seam explicit. I would swap in a semantic embedding provider behind the same interface; I would not claim the hashing baseline provides semantic search quality."

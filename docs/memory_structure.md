# Memory Structure Documentation

This document provides detailed information about the memory system's structure and implementation.

## Memory Types

The memory system is built around three cognitive memory types inspired by human memory:

### 1. Semantic Memory

**Description**: Stores factual knowledge, similar to general world knowledge.

**Implementation Details**:
- Stored as key-value pairs in a dictionary structure
- Keys represent concepts (e.g., "albert_einstein", "napoleonic_wars")
- Values contain detailed factual information about the concept
- Access pattern is direct retrieval by concept key
- Search capability allows finding concepts by keywords

**Example Structure**:
```json
{
  "semantic": {
    "napoleon_bonaparte": {
      "birth": "1769, Corsica",
      "death": "1821, Saint Helena",
      "achievements": ["Napoleonic Code", "Military conquests"]
    },
    "theory_of_relativity": {
      "developed_by": "Albert Einstein",
      "year": "1915",
      "key_concepts": ["Space-time curvature", "Mass-energy equivalence"]
    }
  }
}
```

### 2. Episodic Memory

**Description**: Records experiences and conversations, similar to autobiographical memory.

**Implementation Details**:
- Stored as timestamped entries in a list
- Each entry contains the interaction content and metadata
- Chronologically ordered for sequential retrieval
- Can be filtered by time periods or content keywords
- Provides context about past interactions

**Example Structure**:
```json
{
  "episodic": [
    {
      "timestamp": "2023-07-15T14:32:45",
      "query": "Tell me about Napoleon's early life",
      "response": "Napoleon Bonaparte was born on August 15, 1769, in Corsica...",
      "metadata": {
        "session_id": "abc123",
        "concepts_referenced": ["napoleon_bonaparte", "corsica"]
      }
    },
    {
      "timestamp": "2023-07-15T14:35:12",
      "query": "What were his major achievements?",
      "response": "Napoleon's major achievements include the Napoleonic Code...",
      "metadata": {
        "session_id": "abc123",
        "concepts_referenced": ["napoleon_bonaparte", "napoleonic_code"]
      }
    }
  ]
}
```

### 3. Procedural Memory

**Description**: Stores knowledge about how to perform tasks or follow procedures.

**Implementation Details**:
- Stored as named procedures with structured steps
- Each procedure has a unique identifier
- Steps are ordered in sequence
- Can include contextual information about when/how to apply the procedure
- May reference semantic concepts

**Example Structure**:
```json
{
  "procedural": {
    "analyze_historical_figure": {
      "steps": [
        "1. Research early life and background",
        "2. Examine key achievements and contributions",
        "3. Analyze leadership style and decision-making",
        "4. Evaluate historical impact and legacy",
        "5. Compare with contemporaries"
      ],
      "context": "Use this procedure when conducting a comprehensive analysis of any significant historical figure."
    },
    "evaluate_scientific_theory": {
      "steps": [
        "1. Identify the core principles",
        "2. Review the empirical evidence",
        "3. Examine predictions and confirmations",
        "4. Consider criticisms and limitations",
        "5. Assess historical and current relevance"
      ]
    }
  }
}
```

## Memory Storage

The underlying storage system uses a simple key-value store implementation:

1. **In-Memory Storage**: Primary storage is an in-memory dictionary for fast access
2. **Persistence Layer**: Optional JSON file storage for maintaining memory across sessions
3. **Namespacing**: Each memory type has its own namespace to prevent collisions
4. **Simple Search**: Keyword-based search without requiring embeddings

## Memory Integration

The `IntegratedMemory` class combines all memory types into a unified interface:

1. **Access Methods**: Provides methods to add/retrieve from each memory type
2. **Context Generation**: Creates context summaries from all memory types for LLM queries
3. **Memory Management**: Handles storage, retrieval, and updating of memories
4. **Persistence**: Manages saving and loading memory from disk storage

## LLM Integration

The `OllamaMemoryAssistant` class connects the memory system to Ollama:

1. **Query Processing**: Formats user queries with appropriate memory context
2. **Response Generation**: Sends context-enriched queries to the LLM via Ollama
3. **Memory Updates**: Automatically stores interactions in episodic memory
4. **Error Handling**: Manages API errors and connection issues

## Persistence Implementation

Memory persistence is implemented using JSON file storage:

1. **File Path**: Each user gets a dedicated file based on their user ID
2. **Loading**: Memory is loaded from disk when initializing with an existing user ID
3. **Saving**: Memory is automatically saved after modifications
4. **Format**: Stored as a structured JSON document maintaining the memory hierarchy 
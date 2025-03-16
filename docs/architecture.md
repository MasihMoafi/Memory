# Memory System Architecture

## Overview

The memory system architecture is designed with a layered approach, separating concerns between:
1. Storage layer
2. Memory integration layer
3. Assistant layer (LLM integration)

This modular design allows for flexibility, extensibility, and ease of maintenance.

## System Layers

```
┌────────────────────────────────────────────────────┐
│                                                    │
│  ┌────────────────────────────────────────────┐    │
│  │           OllamaMemoryAssistant            │    │
│  │                                            │    │
│  │  - User interface for memory interaction   │    │
│  │  - LLM integration via Ollama              │    │
│  │  - Query processing with context           │    │
│  └────────────────────┬───────────────────────┘    │
│                       │                            │
│  ┌────────────────────▼───────────────────────┐    │
│  │           IntegratedMemory                 │    │
│  │                                            │    │
│  │  - Combines memory types                   │    │
│  │  - Manages context generation              │    │
│  │  - Coordinates memory operations           │    │
│  └────────────────────┬───────────────────────┘    │
│                       │                            │
│  ┌────────────────────▼───────────────────────┐    │
│  │           SimpleMemoryStore                │    │
│  │                                            │    │
│  │  - Basic storage implementation            │    │
│  │  - Persistence with JSON files             │    │
│  │  - Namespaced key-value storage            │    │
│  └────────────────────────────────────────────┘    │
│                                                    │
└────────────────────────────────────────────────────┘
```

## Key Components

### 1. SimpleMemoryStore

The foundation of the system, providing basic storage capabilities:

- **Responsibilities**:
  - Store and retrieve data by namespace and key
  - Provide basic search functionality
  - Handle persistence to disk (optional)
  - Manage file I/O for memory files

- **Key Methods**:
  - `put(namespace, key, value)`: Store data
  - `get(namespace, key)`: Retrieve data
  - `search(namespace, query)`: Find relevant information
  - `save()` and `load()`: Handle persistence

### 2. IntegratedMemory

The integration layer that combines different memory types:

- **Responsibilities**:
  - Provide typed access to memory (semantic, episodic, procedural)
  - Generate context for LLM queries
  - Manage memory-specific operations
  - Coordinate between memory types

- **Key Methods**:
  - `add_fact(concept, details)`: Add semantic memory
  - `add_procedure(name, steps)`: Add procedural memory
  - `add_interaction(query, response)`: Add episodic memory
  - `generate_context(query)`: Create LLM context from memories

### 3. OllamaMemoryAssistant

The application layer that interfaces with Ollama LLM:

- **Responsibilities**:
  - Process user queries with memory context
  - Send requests to Ollama API
  - Update memory with new interactions
  - Present responses to the user

- **Key Methods**:
  - `process_query(query)`: Process a query with memory context
  - `learn_fact(concept, details)`: Add to semantic memory
  - `learn_procedure(name, steps)`: Add to procedural memory

## Data Flow

1. **Query Processing**:
   ```
   User Query → OllamaMemoryAssistant → IntegratedMemory (context) → Ollama API → Response
   ```

2. **Memory Addition**:
   ```
   New Knowledge → OllamaMemoryAssistant → IntegratedMemory → SimpleMemoryStore
   ```

3. **Persistence**:
   ```
   Memory Update → SimpleMemoryStore → JSON File
   ```

## Extensibility

The system architecture allows for several extension points:

1. **Storage Backends**: Replace SimpleMemoryStore with other implementations (e.g., database, vector store)
2. **LLM Integration**: Swap Ollama with other LLM providers
3. **Memory Types**: Add new memory types beyond the current three
4. **Context Generation**: Customize how context is generated for different query types

## Configuration Options

- **Persistence**: Enable/disable memory persistence
- **Memory Directory**: Configure where memory files are stored
- **Model Selection**: Choose which LLM model to use via Ollama
- **Base URL**: Configure the Ollama API endpoint 
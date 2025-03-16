# Memory Model for AI Assistants

A Python implementation of a cognitive memory system for AI assistants, using Ollama for LLM capabilities. This system provides persistent memory across three distinct types: Semantic, Episodic, and Procedural.

## Memory Types

This system implements three types of memory inspired by human cognition:

1. **Semantic Memory**: Stores factual knowledge (e.g., information about historical figures, concepts, facts)
   - Example: "Napoleon was born in 1769", "Einstein developed the theory of relativity"
   - Implementation: Key-value store with concept identifiers and associated facts

2. **Episodic Memory**: Stores experiences and events (e.g., past conversations, interactions)
   - Example: "Last week we discussed quantum mechanics", "You previously asked about leadership styles"
   - Implementation: Timestamped interaction records with content and metadata

3. **Procedural Memory**: Stores information about how to perform tasks (e.g., analysis methods, procedures)
   - Example: "Steps to analyze a historical figure", "Scientific method procedure"
   - Implementation: Named procedures with structured step-by-step instructions

## Architecture

The system consists of several core components:

- `SimpleMemoryStore`: Basic storage backend with persistence capabilities
- `IntegratedMemory`: Combines all memory types with a unified interface
- `OllamaMemoryAssistant`: Connects memory to LLM capabilities via Ollama

### Data Flow

1. User submits a query
2. System searches across all memory types for relevant information
3. Relevant memories are formatted as context
4. Context and query are sent to LLM via Ollama
5. Response is generated and stored in episodic memory
6. Response is returned to user

## Persistence

This implementation supports persistent memory across sessions via JSON file storage:

- Each user gets a dedicated memory file
- Memory contents are automatically saved after modifications
- Memory is loaded when creating a new assistant with the same user ID

## Example Usage

```python
# Create an assistant with persistent memory
assistant = OllamaMemoryAssistant(
    user_id="history_researcher",
    model_name="gemma3:12b",
    memory_dir="./memories"
)

# Add factual knowledge (semantic memory)
assistant.learn_fact("napoleon_bonaparte", {
    "birth": "1769, Corsica",
    "death": "1821, Saint Helena",
    "achievements": ["Napoleonic Code", "Military conquests"]
})

# Add procedural knowledge
assistant.learn_procedure("analyze_historical_figure", {
    "steps": [
        "1. Research early life and background",
        "2. Examine key achievements and contributions",
        "3. Analyze leadership style and decision-making"
    ]
})

# Query the system
response = assistant.process_query("What do we know about Napoleon?")
```

## Advanced Features

- **Historical Figures Demo**: Specialized demo showing memory usage with historical figures
- **Persistent Memory**: Save and load memories across different sessions
- **Napoleon Persona**: Example adaptation using prompt engineering to create a character with memory

## Requirements

- Python 3.8+
- Ollama (running locally)
- A language model (e.g., gemma3:12b) available via Ollama

## Getting Started

1. Install Ollama from [ollama.ai](https://ollama.ai)
2. Pull a language model: `ollama pull gemma3:12b`
3. Initialize an assistant with memory
4. Add knowledge and start asking questions!

## Project Structure

- `src/memory_model.py` - Core memory implementation
- `src/test_ollama.py` - Tool to test Ollama connection
- `src/run_memory_system.py` - Interactive demo with historical figures
- `src/memory_demo.py` - Demonstration of queries using all memory types

## How It Works

The system consists of several components:

- `SimpleMemoryStore`: Basic storage for all memory types
- `EpisodicMemory`: Stores and retrieves past interactions
- `SemanticMemory`: Manages factual knowledge
- `ProceduralMemory`: Handles procedural knowledge
- `IntegratedMemory`: Combines all memory types
- `OllamaMemoryAssistant`: Uses LLM to answer questions with memory context

When a query is made, the system:
1. Searches across all memory types for relevant information
2. Formats the relevant memories as context
3. Sends the context and query to the LLM
4. Returns the response and stores the interaction

## Example Usage

The default setup includes knowledge about:
- Napoleon Bonaparte
- Elon Musk
- Joseph Stalin

And procedures for:
- Analyzing historical figures
- Comparing leadership styles

Example questions:
- "What do we know about Napoleon's achievements?"
- "How should I analyze Elon Musk as a historical figure?"
- "What similarities exist between Stalin and modern leaders?" 
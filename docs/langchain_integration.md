# LangChain Integration Guide

This guide explains how to integrate the Memory System with LangChain, allowing you to use LangChain's models and capabilities with our memory system.

## Prerequisites

- Memory System installed
- LangChain installed: `pip install langchain`
- LangChain integration model (e.g., `pip install langchain-openai`)

## Integration Approach

There are two main ways to integrate LangChain with our memory system:

1. **Use LangChain models with our memory system**
2. **Use our memory system as a memory component for LangChain**

## Method 1: Using LangChain Models

You can adapt the `OllamaMemoryAssistant` to work with LangChain models:

```python
from langchain.llms import OpenAI
from langchain_core.language_models import BaseLLM
from src.memory_model import IntegratedMemory, SimpleMemoryStore

class LangChainMemoryAssistant:
    """Memory assistant that uses LangChain models"""
    
    def __init__(self, user_id, llm: BaseLLM, memory_dir="./memories"):
        """Initialize with a LangChain model"""
        self.user_id = user_id
        self.llm = llm
        
        # Setup memory storage
        memory_file = f"{memory_dir}/{user_id}_memory.json"
        store = SimpleMemoryStore(storage_path=memory_file)
        self.memory = IntegratedMemory(store)
    
    def process_query(self, query):
        """Process a query using LangChain model and memory context"""
        # Generate context from memory
        context = self.memory.generate_context(query)
        
        # Combine context and query
        prompt = f"""You are an AI assistant with memory capabilities.
        
Memory Context:
{context}

User Question: {query}

Please respond to the user's question using the provided memory context where relevant.
"""
        
        # Use LangChain model to generate response
        response = self.llm.invoke(prompt)
        
        # Add to episodic memory
        self.memory.add_interaction(query, response)
        
        return response
    
    def learn_fact(self, concept, details):
        """Add to semantic memory"""
        self.memory.add_fact(concept, details)
    
    def learn_procedure(self, name, steps):
        """Add to procedural memory"""
        self.memory.add_procedure(name, steps)
```

### Example Usage:

```python
from langchain.llms import OpenAI
from langchain_community.llms import HuggingFaceHub
from langchain_openai import ChatOpenAI

# Using OpenAI
openai_assistant = LangChainMemoryAssistant(
    user_id="openai_user",
    llm=ChatOpenAI(model="gpt-3.5-turbo")
)

# Using Hugging Face
hf_assistant = LangChainMemoryAssistant(
    user_id="huggingface_user",
    llm=HuggingFaceHub(
        repo_id="google/flan-t5-xxl", 
        model_kwargs={"temperature": 0.7}
    )
)

# Add knowledge
openai_assistant.learn_fact("albert_einstein", {
    "birth": "1879",
    "theories": ["General Relativity", "Special Relativity"]
})

# Process queries
response = openai_assistant.process_query("What do we know about Einstein?")
print(response)
```

## Method 2: Using Memory System in LangChain Chains

You can also use our memory system as a custom memory component in LangChain chains:

```python
from langchain.memory import BaseMemory
from langchain.chains import ConversationChain
from src.memory_model import IntegratedMemory, SimpleMemoryStore

class IntegratedLangChainMemory(BaseMemory):
    """Adapter to use IntegratedMemory with LangChain chains"""
    
    def __init__(self, user_id, memory_dir="./memories"):
        """Initialize with user ID"""
        # Setup memory storage
        memory_file = f"{memory_dir}/{user_id}_langchain_memory.json"
        store = SimpleMemoryStore(storage_path=memory_file)
        self.memory = IntegratedMemory(store)
        self.return_messages = True
    
    @property
    def memory_variables(self):
        """The variables this memory provides"""
        return ["memory_context"]
    
    def load_memory_variables(self, inputs):
        """Load memory context based on input"""
        query = inputs.get("input", "")
        context = self.memory.generate_context(query)
        return {"memory_context": context}
    
    def save_context(self, inputs, outputs):
        """Save interaction to memory"""
        query = inputs.get("input", "")
        response = outputs.get("response", "")
        self.memory.add_interaction(query, response)
    
    def clear(self):
        """Clear memory (not implemented - would need direct access to store)"""
        pass
```

### Example Usage with LangChain Chain:

```python
from langchain.llms import OpenAI
from langchain.chains import ConversationChain
from langchain.prompts import PromptTemplate

# Create LangChain memory adapter
memory = IntegratedLangChainMemory(user_id="langchain_user")

# Create prompt template with memory context
template = """You are an AI assistant with memory.

Memory Context:
{memory_context}

Current conversation:
Human: {input}
AI: """

prompt = PromptTemplate(
    input_variables=["memory_context", "input"],
    template=template
)

# Create conversation chain
chain = ConversationChain(
    llm=OpenAI(temperature=0),
    memory=memory,
    prompt=prompt,
    verbose=True
)

# Add knowledge directly to memory
memory.memory.add_fact("neural_networks", {
    "definition": "Computational systems inspired by biological neural networks",
    "types": ["CNN", "RNN", "Transformer"]
})

# Use the chain
response = chain.predict(input="What are neural networks?")
print(response)
```

## Considerations

1. **API Differences**: LangChain models have different APIs than Ollama, requiring adaptation
2. **Prompt Engineering**: Different models may require different prompt templates
3. **Token Limits**: Be aware of token limits when providing memory context to models
4. **Cost**: LangChain with hosted models like OpenAI will incur API costs
5. **Persistence**: Ensure paths are set correctly for persistent memory storage

## Extended Use Cases

- **Agent Integration**: Use memory system with LangChain agents
- **Tool Use**: Combine memory system with LangChain tools
- **Retrieval**: Use memory system alongside LangChain retrievers for enhanced context
- **Output Parsers**: Use LangChain output parsers to structure memory additions 
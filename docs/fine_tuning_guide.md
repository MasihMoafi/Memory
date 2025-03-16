# Fine-Tuning Guide

This guide explains how to fine-tune a language model to adopt a specific persona (like a domain expert) and integrate it with the memory system.

## Overview

Fine-tuning a language model allows you to customize its behavior for specific use cases:

1. Adopt a particular persona or character
2. Specialize in domain-specific knowledge
3. Follow consistent patterns of interaction
4. Improve reliability for specific tasks

## Prerequisites

- Memory System installed
- A chosen base model (e.g., llama3, mistral, gemma3)
- Training data for fine-tuning
- Compute resources (GPU recommended)
- Ollama installed locally

## Process Overview

1. **Collect Training Data**
2. **Format Data for Fine-Tuning**
3. **Perform Fine-Tuning**
4. **Integrate with Memory System**
5. **Test and Iterate**

## Step 1: Collect Training Data

For a domain expert persona, you'll need:

- **Primary Sources**: Books, papers, and articles written by experts in the field
- **Secondary Sources**: Interviews, lectures, and presentations by experts
- **Synthetic Data**: Generated Q&A pairs demonstrating expert responses to domain questions

**Example Data Sources for a Physics Expert:**
- Academic papers on physics topics
- Transcripts of lectures by renowned physicists
- Books explaining complex physics concepts
- Interviews with physics experts

**Data Collection Tips:**
- Focus on material that captures the expert's voice and perspective
- Include a variety of topics within the domain
- Aim for at least 50-100 high-quality examples
- Balance technical knowledge with communication style

## Step 2: Format Data for Fine-Tuning

Most fine-tuning requires data in a specific format:

```json
[
  {
    "role": "system",
    "content": "You are a physics expert with deep knowledge of quantum mechanics, relativity, and particle physics. You explain complex concepts clearly and precisely, using analogies when helpful. You cite relevant research and provide context for your explanations."
  },
  {
    "role": "user",
    "content": "Can you explain quantum entanglement in simple terms?"
  },
  {
    "role": "assistant",
    "content": "Quantum entanglement is like having two coins that are magically connected. When you flip one coin and it lands on heads, the other coin instantly lands on tails, no matter how far apart they are. This happens faster than any signal could travel between them, even at the speed of light. In quantum physics, particles become 'entangled' when they interact in ways that their quantum states can't be described independently. When you measure one particle, you instantly know information about its entangled partner. Einstein called this 'spooky action at a distance' because it seemed to violate locality principles. Experiments by Alain Aspect in the 1980s and many others since have confirmed this bizarre property of quantum mechanics, which has applications in quantum computing and quantum cryptography."
  },
  {
    "role": "user",
    "content": "What's the difference between general and special relativity?"
  },
  {
    "role": "assistant",
    "content": "Special relativity and general relativity are Einstein's two revolutionary theories that fundamentally changed our understanding of space, time, and gravity. Special relativity (1905) deals with physics in the absence of gravity and acceleration. It introduces two key principles: the laws of physics are the same for all non-accelerating observers, and the speed of light is constant regardless of the observer's motion. This leads to effects like time dilation and length contraction. General relativity (1915) extends these concepts to include gravity and acceleration. Instead of viewing gravity as a force, Einstein reconceived it as a curvature of spacetime caused by mass and energy. Massive objects like stars and planets create a kind of 'dent' in the fabric of spacetime, causing other objects to follow curved paths. Special relativity is a special case of general relativity that applies when gravity is negligible. While special relativity unified space and time, general relativity unified space, time, and gravity."
  }
]
```

## Step 3: Perform Fine-Tuning

### Option 1: Using Ollama (Local Fine-Tuning)

Ollama provides a simple way to create a custom model:

1. Create a `Modelfile`:

```
FROM gemma3:12b
SYSTEM "You are a physics expert with deep knowledge of quantum mechanics, relativity, and particle physics. You explain complex concepts clearly and precisely, using analogies when helpful. You cite relevant research and provide context for your explanations."

# Training data incorporation
TEMPLATE """{{ if .System }}{{ .System }}{{ end }}
{{ if .Prompt }}{{ .Prompt }}{{ end }}

{{ .Response }}
"""

# Include your training examples as additional parameters
PARAMETER temperature 0.7
PARAMETER top_p 0.9
PARAMETER stop "Human:"
PARAMETER stop "Expert:"
```

2. Build the model:
```bash
ollama create physics-expert -f /path/to/Modelfile
```

3. Test your model:
```bash
ollama run physics-expert "What is the uncertainty principle?"
```

### Option 2: Using 3rd-Party Services (More Advanced)

For more advanced fine-tuning:

1. Choose a service:
   - [Hugging Face](https://huggingface.co/)
   - [Google Vertex AI](https://cloud.google.com/vertex-ai)

2. Upload your training data and configure the fine-tuning job
3. Monitor training progress and evaluate results
4. Export or deploy the model

## Step 4: Integrate with Memory System

After fine-tuning, integrate the model with the memory system:

1. Create a specialized version of `OllamaMemoryAssistant`:

```python
from src.memory_model import OllamaMemoryAssistant, IntegratedMemory, SimpleMemoryStore

class ExpertMemoryAssistant(OllamaMemoryAssistant):
    """Memory assistant that embodies a domain expert"""
    
    def process_query(self, query):
        """Process a query using the expert persona"""
        # Generate context from memory
        context = self.memory.generate_context(query)
        
        # Combine context and query with expert-specific system prompt
        prompt = f"""You are a physics expert with deep knowledge of quantum mechanics, relativity, and particle physics.
        You have access to your memories:
        
Memory Context:
{context}

User Question: {query}

Respond as a physics expert, drawing on your memories where relevant.
"""
        
        # Use Ollama API to generate response
        response = self._generate_response(prompt)
        
        # Add to episodic memory
        self.memory.add_interaction(query, response)
        
        return response
```

2. Initialize with your fine-tuned model:

```python
# Setup memory directory
memory_dir = "expert_memories"
os.makedirs(memory_dir, exist_ok=True)

# Create expert assistant
assistant = ExpertMemoryAssistant(
    user_id="physics_expert",
    model_name="physics-expert:latest",  # Your fine-tuned model
    memory_dir=memory_dir
)

# Pre-populate with relevant knowledge
assistant.learn_fact("quantum_mechanics", {
    "founder": "Max Planck, Niels Bohr, Werner Heisenberg",
    "key_principles": ["Wave-particle duality", "Uncertainty principle", "Quantum superposition"],
    "applications": ["Quantum computing", "Quantum cryptography"]
})

# Run interactive session
while True:
    query = input("You: ")
    if query.lower() in ["exit", "quit"]:
        break
    
    response = assistant.process_query(query)
    print(f"Expert: {response}")
```

## Step 5: Test and Iterate

After integration, evaluate and refine your model:

1. **Evaluation Criteria**:
   - Does the model consistently maintain the expert persona?
   - Does it accurately reflect domain knowledge?
   - Does it effectively use the memory system?
   - Is the interaction natural and engaging?

2. **Improvement Process**:
   - Add more training examples for areas where the model is weak
   - Adjust system prompts to better guide the model's behavior
   - Fine-tune hyperparameters to balance creativity and accuracy
   - Add more memories for common topics of discussion

## Alternative: Prompt Engineering Approach

If full fine-tuning is not feasible, a simpler approach is to use prompt engineering:

```python
class ExpertPromptAssistant(OllamaMemoryAssistant):
    """Uses prompt engineering to emulate an expert without fine-tuning"""
    
    def process_query(self, query):
        """Process a query using prompt engineering for expert persona"""
        # Generate context from memory
        context = self.memory.generate_context(query)
        
        # Detailed persona description in system prompt
        prompt = f"""You are roleplaying as a physics expert with specialization in quantum mechanics.
        
Key traits of this expert:
1. Deep knowledge of theoretical physics
2. Explains complex concepts clearly
3. Uses analogies to make difficult concepts accessible
4. Cites relevant research and theories
5. Acknowledges areas of ongoing research or debate
6. Maintains scientific accuracy
7. Balances technical detail with understandable explanations
8. Never breaks character

Memory Context:
{context}

User Question: {query}

Respond as a physics expert would, drawing on your memories where relevant. Maintain a helpful and educational tone throughout.
"""
        
        # Use Ollama API to generate response
        response = self._generate_response(prompt)
        
        # Add to episodic memory
        self.memory.add_interaction(query, response)
        
        return response
```

## Comparing Approaches

| Approach | Advantages | Disadvantages |
|----------|------------|--------------|
| **Fine-tuning** | More consistent character portrayal, Better domain knowledge, Deeper integration of expertise | Requires significant data, Computationally expensive, Takes time to train |
| **Prompt Engineering** | Quick to implement, No special hardware required, Easy to adjust and iterate | Less consistent characterization, Uses more tokens per request, May occasionally break character |

## Conclusion

For the most authentic and robust expert emulation, a combination of both approaches is ideal:

1. Start with prompt engineering to quickly test and refine the persona
2. Collect interaction data from your prompt-engineered sessions
3. Use this data to create a fine-tuning dataset
4. Fine-tune a model with the collected data
5. Integrate the fine-tuned model with the memory system
6. Continue gathering data from user interactions for future fine-tuning iterations

This iterative approach allows you to continuously improve the expert emulation while providing value to users at each stage of development.
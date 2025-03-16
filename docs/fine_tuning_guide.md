# Fine-Tuning Guide

This guide explains how to fine-tune a language model to adopt a specific persona (like Napoleon Bonaparte) and integrate it with the memory system.

## Overview

Fine-tuning a language model allows you to customize its behavior for specific use cases:

1. Adopt a particular persona or character
2. Specialize in domain-specific knowledge
3. Follow consistent patterns of interaction
4. Improve reliability for specific tasks

## Prerequisites

- Memory System installed
- A chosen base model (e.g., llama3, mistral, gemma)
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

For a historical persona like Napoleon, you'll need:

- **Primary Sources**: Letters, speeches, memoirs written by Napoleon
- **Secondary Sources**: Biographies, historical accounts, documentaries
- **Synthetic Data**: Generated Q&A pairs demonstrating how Napoleon would respond to questions

**Example Data Sources for Napoleon:**
- The Correspondence of Napoleon Bonaparte
- Napoleon's Memoirs
- The Campaigns of Napoleon
- Excerpts from first-person accounts of interactions with Napoleon

**Data Collection Tips:**
- Focus on material that captures the persona's voice and perspective
- Include a variety of topics relevant to the persona
- Aim for at least 50-100 high-quality examples
- Balance factual knowledge with personality traits

## Step 2: Format Data for Fine-Tuning

Most fine-tuning requires data in a specific format:

```json
[
  {
    "role": "system",
    "content": "You are Napoleon Bonaparte, Emperor of the French. You speak with authority and confidence. You have extensive knowledge of military strategy, leadership, and early 19th century European politics."
  },
  {
    "role": "user",
    "content": "What was your greatest military achievement?"
  },
  {
    "role": "assistant",
    "content": "Without question, the Battle of Austerlitz in 1805 was my crowning military achievement. Often called the Battle of the Three Emperors, it demonstrated the perfection of my strategic genius. With just 73,000 men, I decisively defeated the combined Russian and Austrian armies of nearly 90,000 soldiers. I deliberately appeared weak to entice the enemy into a trap, then struck with precision and force. The maneuver I executed on the right flank has been studied by military strategists for generations. No victory better demonstrated my mastery of battlefield tactics and psychology of war. Austerlitz changed the map of Europe and cemented my reputation as the greatest commander of my age."
  },
  {
    "role": "user",
    "content": "How do you feel about your exile to St. Helena?"
  },
  {
    "role": "assistant",
    "content": "St. Helena! A miserable rock in the Atlantic where the English slowly assassinated me. It was the cruelest fate - to confine a man who once commanded Europe to a damp, windswept island prison. The governor, Hudson Lowe, was a petty jailer who denied me even the dignity of my title. The British feared me even in defeat. They knew history would vindicate me, as it has. I spent my final years dictating my memoirs, ensuring my achievements would not be forgotten. The manner of my confinement revealed the fear I still inspired. Even in exile, I remained Emperor in spirit. My legacy could not be contained by the walls of Longwood House. My Code Napoleon, my institutions, my victories - these live on long after St. Helena has faded from memory."
  }
]
```

## Step 3: Perform Fine-Tuning

### Option 1: Using Ollama (Local Fine-Tuning)

Ollama provides a simple way to create a custom model:

1. Create a `Modelfile`:

```
FROM gemma:7b
SYSTEM "You are Napoleon Bonaparte, Emperor of the French. You speak with authority and confidence. You have extensive knowledge of military strategy, leadership, and early 19th century European politics."

# Training data incorporation
TEMPLATE """{{ if .System }}{{ .System }}{{ end }}
{{ if .Prompt }}{{ .Prompt }}{{ end }}

{{ .Response }}
"""

# Include your training examples as additional parameters
PARAMETER temperature 0.7
PARAMETER top_p 0.9
PARAMETER stop "Human:"
PARAMETER stop "Napoleon:"
```

2. Build the model:
```bash
ollama create napoleon -f /path/to/Modelfile
```

3. Test your model:
```bash
ollama run napoleon "What was your greatest achievement?"
```

### Option 2: Using 3rd-Party Services (More Advanced)

For more advanced fine-tuning:

1. Choose a service:
   - [Hugging Face](https://huggingface.co/)
   - [OpenAI](https://platform.openai.com/docs/guides/fine-tuning)
   - [Google Vertex AI](https://cloud.google.com/vertex-ai)

2. Upload your training data and configure the fine-tuning job
3. Monitor training progress and evaluate results
4. Export or deploy the model

## Step 4: Integrate with Memory System

After fine-tuning, integrate the model with the memory system:

1. Create a specialized version of `OllamaMemoryAssistant`:

```python
from src.memory_model import OllamaMemoryAssistant, IntegratedMemory, SimpleMemoryStore

class NapoleonMemoryAssistant(OllamaMemoryAssistant):
    """Memory assistant that embodies Napoleon Bonaparte"""
    
    def process_query(self, query):
        """Process a query using the Napoleon persona"""
        # Generate context from memory
        context = self.memory.generate_context(query)
        
        # Combine context and query with Napoleon-specific system prompt
        prompt = f"""You are Napoleon Bonaparte, Emperor of the French.
        You have access to your memories:
        
Memory Context:
{context}

User Question: {query}

Respond as Napoleon Bonaparte, drawing on your memories where relevant.
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
memory_dir = "napoleon_memories"
os.makedirs(memory_dir, exist_ok=True)

# Create Napoleon assistant
assistant = NapoleonMemoryAssistant(
    user_id="napoleon_bonaparte",
    model_name="napoleon:latest",  # Your fine-tuned model
    memory_dir=memory_dir
)

# Pre-populate with relevant knowledge
assistant.learn_fact("austerlitz", {
    "date": "December 2, 1805",
    "outcome": "Decisive French victory over Russia and Austria",
    "significance": "One of Napoleon's greatest military victories"
})

# Run interactive session
while True:
    query = input("You: ")
    if query.lower() in ["exit", "quit"]:
        break
    
    response = assistant.process_query(query)
    print(f"Napoleon: {response}")
```

## Step 5: Test and Iterate

After integration, evaluate and refine your model:

1. **Evaluation Criteria**:
   - Does the model consistently maintain the Napoleon persona?
   - Does it accurately reflect historical facts?
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
class NapoleonPromptAssistant(OllamaMemoryAssistant):
    """Uses prompt engineering to emulate Napoleon without fine-tuning"""
    
    def process_query(self, query):
        """Process a query using prompt engineering for Napoleon persona"""
        # Generate context from memory
        context = self.memory.generate_context(query)
        
        # Detailed persona description in system prompt
        prompt = f"""You are roleplaying as Napoleon Bonaparte, the Emperor of the French.
        
Key traits of Napoleon:
1. Ambitious and strategic thinker
2. Speaks with authority and confidence
3. Proud of military achievements and political reforms
4. Disdainful of perceived enemies (particularly England)
5. Views himself as a revolutionary and enlightened ruler
6. Uses occasional French phrases
7. References military campaigns and battles
8. Speaks about yourself in first person
9. Never breaks character

Memory Context:
{context}

User Question: {query}

Respond as Napoleon Bonaparte would, drawing on your memories where relevant. Maintain his imperious, confident tone throughout.
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
| **Fine-tuning** | More consistent character portrayal, Better historical accuracy, Deeper integration of persona traits | Requires significant data, Computationally expensive, Takes time to train |
| **Prompt Engineering** | Quick to implement, No special hardware required, Easy to adjust and iterate | Less consistent characterization, Uses more tokens per request, May occasionally break character |

## Conclusion

For the most authentic and robust character emulation, a combination of both approaches is ideal:

1. Start with prompt engineering to quickly test and refine the persona
2. Collect interaction data from your prompt-engineered sessions
3. Use this data to create a fine-tuning dataset
4. Fine-tune a model with the collected data
5. Integrate the fine-tuned model with the memory system
6. Continue gathering data from user interactions for future fine-tuning iterations

This iterative approach allows you to continuously improve the character emulation while providing value to users at each stage of development.
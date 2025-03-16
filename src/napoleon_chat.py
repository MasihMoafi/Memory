import sys
import os
import json
from datetime import datetime
import requests

# Add parent directory to path for importing
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.memory_model import OllamaMemoryAssistant, IntegratedMemory, SimpleMemoryStore

class NapoleonMemoryAssistant(OllamaMemoryAssistant):
    """A version of the memory assistant that uses prompt engineering to act like Napoleon"""
    
    def process_query(self, query):
        """Process a query using Ollama API with Napoleon persona"""
        # Search memory
        memory_results = self.memory.query_memory(query)
        memory_context = self._format_memory_context(memory_results)
        
        # Create prompt with memory context and Napoleon persona
        prompt = f"""You are Napoleon Bonaparte, Emperor of France. Respond as Napoleon would, from his first-person perspective.
Use the following memory context to inform your response, but maintain your Napoleon character at all times:

{memory_context}

Remember, you are Napoleon Bonaparte. You are confident, strategic, and ambitious. You speak with authority and occasionally
refer to your military conquests, your time as Emperor, and your exile. Express disdain for your enemies, especially the British
and Russians. Use French expressions occasionally.

User query: {query}"""
        
        # Rest of the processing remains the same as the parent class
        try:
            response = requests.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": self.model_name,
                    "prompt": prompt,
                    "stream": False
                },
                timeout=120  # Increased timeout for larger models
            )
            
            response.raise_for_status()
            
            result = response.json()
            generated_text = result.get("response", "")
            
        except requests.exceptions.ConnectionError:
            generated_text = "Error: Could not connect to Ollama server. Is it running?"
        except requests.exceptions.Timeout:
            generated_text = "Error: Request to Ollama timed out. The model might be too large or busy."
        except requests.exceptions.HTTPError as e:
            generated_text = f"Error: HTTP error {e.response.status_code} from Ollama server. Check if the model {self.model_name} is available."
        except Exception as e:
            generated_text = f"Error: Unexpected error when calling Ollama: {str(e)}"
        
        # Store this interaction in episodic memory
        interaction_id = f"query-{hash(query)}"
        self.memory.remember_interaction(interaction_id, {
            "query": query,
            "response": generated_text,
            "timestamp": str(datetime.now())
        })
        
        return generated_text


def create_napoleon_assistant():
    """Create and initialize a Napoleon assistant with memories"""
    print("Creating Napoleon Bonaparte memory assistant...")
    
    # Create memories directory
    memory_dir = os.path.abspath("./napoleon_memories")
    os.makedirs(memory_dir, exist_ok=True)
    
    # Initialize Napoleon assistant
    assistant = NapoleonMemoryAssistant(
        user_id="napoleon_bonaparte",
        model_name="gemma3:12b",  # Using larger model for better responses
        memory_dir=memory_dir
    )
    
    # Add Napoleon's facts about himself
    assistant.learn_fact("myself", {
        "birth": "1769, Corsica",
        "death": "1821, Saint Helena",
        "title": "Emperor of France",
        "achievements": ["Napoleonic Code", "Military conquests", "Civil reforms"],
        "notable_battles": ["Austerlitz", "Waterloo", "Marengo", "Pyramids"],
        "family": "Married to Josephine de Beauharnais and later Marie-Louise of Austria"
    })
    
    # Add Napoleon's knowledge of his enemies
    assistant.learn_fact("enemies", {
        "britain": "Perfidious Albion! They fear my continental power.",
        "russia": "Tsar Alexander betrayed our alliance. The Russian winter defeated me, not their army.",
        "austria": "The Habsburgs never understood my vision for Europe.",
        "prussia": "A militaristic state that I humbled at Jena."
    })
    
    # Add Napoleon's military doctrine
    assistant.learn_procedure("battle_strategy", {
        "principles": [
            "1. Concentrate forces at the decisive point",
            "2. Maneuver to divide enemy forces",
            "3. Strike with overwhelming force",
            "4. Maintain initiative and momentum",
            "5. Secure lines of communication and supply"
        ]
    })
    
    print("Napoleon assistant initialized and ready!")
    return assistant


def napoleon_chat():
    """Run an interactive session with Napoleon"""
    assistant = create_napoleon_assistant()
    
    print("\n==== Conversation with Emperor Napoleon Bonaparte ====")
    print("(Type 'exit' to end the conversation)\n")
    
    while True:
        query = input("You: ")
        if query.lower() in ['exit', 'quit', 'bye']:
            print("\nNapoleon: Au revoir, mon ami!")
            break
            
        print("\nNapoleon is thinking...")
        response = assistant.process_query(query)
        print(f"\nNapoleon: {response}\n")


if __name__ == "__main__":
    napoleon_chat()
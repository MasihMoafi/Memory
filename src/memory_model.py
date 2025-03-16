import os
from typing import Dict, List, Any, Optional
import json
from datetime import datetime
import requests 

class Result:
    """Wrapper class to mimic the result object from LangGraph's InMemoryStore"""
    def __init__(self, id, value):
        self.id = id
        self.value = value

class SimpleMemoryStore:
    """A simple in-memory store that doesn't require embeddings or external APIs"""
    
    def __init__(self, storage_path=None):
        # Main storage dictionary
        self.storage = {}
        self.storage_path = storage_path
        
        # Load existing data if storage path is provided
        if storage_path and os.path.exists(storage_path):
            self.load()
    
    def put(self, namespace, key, value):
        """Store data in the specified namespace under the given key"""
        ns_str = self._format_namespace(namespace)
        if ns_str not in self.storage:
            self.storage[ns_str] = {}
        self.storage[ns_str][key] = value
        
        # Save to disk if storage path is set
        if self.storage_path:
            self.save()
        return True
    
    def get(self, namespace, key):
        """Retrieve data from the specified namespace under the given key"""
        ns_str = self._format_namespace(namespace)
        if ns_str not in self.storage or key not in self.storage[ns_str]:
            return None
        
        return Result(key, self.storage[ns_str][key])
    
    def search(self, namespace, query, filter_func=None):
        """Simple keyword-based search (no embeddings)"""
        ns_str = self._format_namespace(namespace)
        if ns_str not in self.storage:
            return []
        
        results = []
        for key, value in self.storage[ns_str].items():
            # Very basic text search - check if query exists in the key or stringified value
            value_str = json.dumps(value).lower()
            match_found = (query.lower() in key.lower() or 
                          query.lower() in value_str)
            
            if match_found:
                # Apply filter if provided
                result = Result(key, value)
                if filter_func is None or filter_func(result):
                    results.append(result)
        
        return results
    
    def _format_namespace(self, namespace):
        """Convert tuple namespace to string representation"""
        if isinstance(namespace, tuple):
            return "/".join(namespace)
        return str(namespace)
    
    def save(self):
        """Save the memory store to disk"""
        try:
            os.makedirs(os.path.dirname(self.storage_path), exist_ok=True)
            with open(self.storage_path, 'w') as f:
                json.dump(self.storage, f, indent=2)
            return True
        except Exception as e:
            print(f"Error saving memory store: {e}")
            return False
    
    def load(self):
        """Load the memory store from disk"""
        try:
            with open(self.storage_path, 'r') as f:
                self.storage = json.load(f)
            return True
        except Exception as e:
            print(f"Error loading memory store: {e}")
            return False


class EpisodicMemory:
    """Stores specific experiences and events"""
    def __init__(self, store, namespace):
        self.store = store
        self.namespace = namespace
    
    def add_interaction(self, interaction_id, content):
        """Store a specific interaction or event"""
        self.store.put(
            self.namespace,
            f"interaction:{interaction_id}",
            {"content": content, "type": "episodic", "timestamp": str(datetime.now())}
        )
        return f"Stored interaction {interaction_id}"
    
    def recall_interaction(self, interaction_id):
        """Retrieve a specific interaction by ID"""
        result = self.store.get(self.namespace, f"interaction:{interaction_id}")
        if result is None:
            return f"No memory found for interaction {interaction_id}"
        return result.value['content']
    
    def search_interactions(self, query):
        """Search for relevant interactions based on content"""
        results = self.store.search(
            self.namespace, 
            query,
            filter_func=lambda x: x.value.get('type') == 'episodic'
        )
        return [r.value['content'] for r in results]


class SemanticMemory:
    """Stores factual knowledge and concepts"""
    def __init__(self, store, namespace):
        self.store = store
        self.namespace = namespace
    
    def add_knowledge(self, key, facts):
        """Store factual knowledge about a concept"""
        self.store.put(
            self.namespace,
            f"concept:{key}",
            {"facts": facts, "type": "semantic"}
        )
        return f"Stored knowledge about {key}"
    
    def get_knowledge(self, key):
        """Retrieve knowledge about a specific concept"""
        result = self.store.get(self.namespace, f"concept:{key}")
        if result is None:
            return f"No knowledge found about {key}"
        return result.value['facts']
    
    def search_knowledge(self, query):
        """Search for relevant knowledge"""
        results = self.store.search(
            self.namespace, 
            query,
            filter_func=lambda x: x.value.get('type') == 'semantic'
        )
        return [(r.id.split(':')[-1], r.value['facts']) for r in results]


class ProceduralMemory:
    """Stores information about how to perform tasks or follow procedures"""
    def __init__(self, store, namespace):
        self.store = store
        self.namespace = namespace
    
    def add_procedure(self, name, instructions):
        """Store a procedure or instructions for a task"""
        self.store.put(
            self.namespace,
            f"procedure:{name}",
            {"instructions": instructions, "type": "procedural"}
        )
        return f"Stored procedure {name}"
    
    def get_procedure(self, name):
        """Retrieve instructions for a specific procedure"""
        result = self.store.get(self.namespace, f"procedure:{name}")
        if result is None:
            return f"No procedure found for {name}"
        return result.value['instructions']
    
    def update_procedure(self, name, new_instructions):
        """Update existing procedure instructions"""
        existing = self.store.get(self.namespace, f"procedure:{name}")
        if existing is None:
            return f"No procedure found for {name}"
        
        self.store.put(
            self.namespace,
            f"procedure:{name}",
            {"instructions": new_instructions, "type": "procedural"}
        )
        return f"Updated procedure {name}"
    
    def search_procedures(self, query):
        """Search for relevant procedures"""
        results = self.store.search(
            self.namespace, 
            query,
            filter_func=lambda x: x.value.get('type') == 'procedural'
        )
        return [(r.id.split(':')[-1], r.value['instructions']) for r in results]


class IntegratedMemory:
    """Combines episodic, semantic, and procedural memory into a single system"""
    def __init__(self, user_id, storage_path=None):
        self.user_id = user_id
        
        # Create storage path if provided
        if storage_path:
            os.makedirs(os.path.dirname(storage_path), exist_ok=True)
            memory_file = f"{user_id}_memory.json"
            self.storage_path = os.path.join(storage_path, memory_file)
        else:
            self.storage_path = None
        
        # Create a store with persistence if path is provided
        self.store = SimpleMemoryStore(self.storage_path)
        
        # Initialize all memory types
        self.episodic = EpisodicMemory(self.store, (user_id, "episodes"))
        self.semantic = SemanticMemory(self.store, (user_id, "knowledge"))
        self.procedural = ProceduralMemory(self.store, (user_id, "procedures"))
    
    def query_memory(self, query):
        """Query across all memory types"""
        results = {
            "episodic": self.episodic.search_interactions(query),
            "semantic": self.semantic.search_knowledge(query),
            "procedural": self.procedural.search_procedures(query)
        }
        return results
    
    def remember_interaction(self, interaction_id, content):
        return self.episodic.add_interaction(interaction_id, content)
    
    def learn_fact(self, concept, facts):
        return self.semantic.add_knowledge(concept, facts)
    
    def learn_procedure(self, name, instructions):
        return self.procedural.add_procedure(name, instructions)


class OllamaMemoryAssistant:
    """Assistant using Ollama instead of HuggingFace models"""
    def __init__(self, user_id, model_name="gemma3:12b", base_url="http://localhost:11434", memory_dir="./memories"):
        # Create memory directory if it doesn't exist
        memory_path = os.path.abspath(memory_dir)
        os.makedirs(memory_path, exist_ok=True)
        
        self.memory = IntegratedMemory(user_id, storage_path=memory_path)
        self.model_name = model_name
        self.base_url = base_url
    
    def _format_memory_context(self, memory_results):
        context = ["MEMORY CONTEXT:"]
        
        if memory_results["episodic"]:
            context.append("\nPast experiences:")
            for memory in memory_results["episodic"][:3]:  # Limit to top 3
                context.append(f"- {memory}")
        
        if memory_results["semantic"]:
            context.append("\nKnown facts:")
            for concept, facts in memory_results["semantic"][:3]:  # Limit to top 3
                context.append(f"- {concept}: {facts}")
        
        if memory_results["procedural"]:
            context.append("\nRelevant procedures:")
            for name, instructions in memory_results["procedural"][:3]:  # Limit to top 3
                context.append(f"- {name}: {instructions}")
        
        return "\n".join(context)

    def process_query(self, query):
        """Process a query using Ollama API with error handling"""
        # Search memory
        memory_results = self.memory.query_memory(query)
        memory_context = self._format_memory_context(memory_results)
        
        # Create prompt with memory context
        prompt = f"""You are an assistant with access to the user's memories.
Use the following memory context to answer the query:

{memory_context}

User query: {query}"""
        
        # Call Ollama API with error handling
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
            
            response.raise_for_status()  # Raise exception for error status codes
            
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
    
    def learn_fact(self, concept, facts):
        return self.memory.learn_fact(concept, facts)
    
    def learn_procedure(self, name, instructions):
        return self.memory.learn_procedure(name, instructions)
    
    def remember_interaction(self, interaction_id, content):
        return self.memory.remember_interaction(interaction_id, content)


# Example usage with historical figures
def main():
    # Create a memory-enhanced assistant using Ollama
    assistant = OllamaMemoryAssistant("history_researcher", model_name="gemma3:12b")
    
    # Pre-load historical figure facts
    assistant.learn_fact("napoleon_bonaparte", {
        "birth": "1769, Corsica",
        "death": "1821, Saint Helena",
        "title": "Emperor of France",
        "achievements": ["Napoleonic Code", "Military conquests in Europe", "Civil administration reforms"],
        "notable_battles": ["Austerlitz", "Waterloo", "Battle of the Pyramids"],
        "legacy": "Reformed French legal system, reshaped European politics and warfare"
    })
    
    assistant.learn_fact("elon_musk", {
        "birth": "1971, South Africa",
        "companies": ["Tesla", "SpaceX", "Neuralink", "X (Twitter)"],
        "innovations": "Electric vehicles, reusable rockets, brain-machine interfaces",
        "notable_achievements": ["First private company to send humans to space", "Mainstream adoption of electric cars"],
        "background": "Born in South Africa, later moved to Canada and US"
    })
    
    assistant.learn_fact("joseph_stalin", {
        "birth": "1878, Georgia",
        "death": "1953, Moscow",
        "title": "General Secretary of the Communist Party of the Soviet Union",
        "rule_period": "1924-1953",
        "policies": ["Rapid industrialization", "Collectivization", "Great Purge"],
        "impact": "Transformed USSR into industrial and military superpower at great human cost"
    })
    
    # Procedural knowledge
    assistant.learn_procedure("analyze_historical_figure", {
        "steps": [
            "1. Research their early life and background",
            "2. Examine key achievements and contributions",
            "3. Understand historical context they operated in",
            "4. Analyze their leadership style and decision-making",
            "5. Evaluate their legacy and impact on future generations"
        ]
    })
    
    assistant.learn_procedure("compare_leadership_styles", {
        "steps": [
            "1. Identify the key figures to compare",
            "2. Research their background and rise to power",
            "3. Analyze decision-making patterns and key policies",
            "4. Evaluate communication styles and relationships with followers",
            "5. Compare how they handled crises or opposition",
            "6. Assess their historical impact and legacy"
        ]
    })
    
    # Sample episodic memory
    assistant.remember_interaction("research-session-2023-10-15", {
        "topic": "Comparing leadership styles",
        "figures_examined": ["Napoleon Bonaparte", "Joseph Stalin", "Elon Musk"],
        "insights": "Autocratic vs. innovative leadership approaches across different eras",
        "questions_raised": [
            "How might Napoleon have used modern technology?", 
            "What similarities exist between Stalin and modern tech leaders?",
            "How do crisis management approaches differ across these leaders?"
        ]
    })
    
    # Example queries
    queries = [
        "What do we know about Napoleon's achievements?",
        "How should I analyze Elon Musk as a historical figure?",
        "What was discussed in our last research session about leadership styles?"
    ]
    
    for query in queries:
        print(f"\nQuery: {query}")
        response = assistant.process_query(query)
        print(f"Response: {response}")

if __name__ == "__main__":
    main() 

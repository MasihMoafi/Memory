import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.memory_model import OllamaMemoryAssistant

def run_memory_demo():
    """Run a demonstration of the three memory types"""
    print("Creating memory assistant with gemma3:12b...")
    assistant = OllamaMemoryAssistant("demo_user", model_name="gemma3:12b")
    
    # Setup memories
    print("\nInitializing memories...")
    
    # Semantic memory (facts)
    assistant.learn_fact("albert_einstein", {
        "birth": "1879, Ulm, Germany",
        "death": "1955, Princeton, USA",
        "achievements": ["Theory of Relativity", "Photoelectric effect", "Nobel Prize in Physics (1921)"],
        "quote": "Imagination is more important than knowledge"
    })
    
    # Procedural memory (how to do things)
    assistant.learn_procedure("scientific_method", {
        "steps": [
            "1. Make an observation",
            "2. Ask a question",
            "3. Form a hypothesis",
            "4. Conduct an experiment",
            "5. Analyze data",
            "6. Draw a conclusion",
            "7. Report results"
        ]
    })
    
    # Episodic memory (past experiences)
    assistant.remember_interaction("physics_discussion_20231015", {
        "topic": "Einstein's contributions to quantum mechanics",
        "participants": ["User", "Professor Smith"],
        "key_points": [
            "Einstein's skepticism about quantum mechanics",
            "The EPR paradox and 'spooky action at a distance'",
            "Einstein's famous quote 'God does not play dice with the universe'"
        ]
    })
    
    # Demonstrate with queries that engage different memory types
    demo_queries = [
        # Semantic memory query (factual)
        "When was Albert Einstein born and what was he known for?",
        
        # Procedural memory query (how-to)
        "What are the steps of the scientific method?",
        
        # Episodic memory query (past experiences)
        "What did we discuss about Einstein's views on quantum mechanics?",
        
        # Mixed memory query (uses multiple memory types)
        "What methodology would Einstein use to prove his theories and what were his major achievements?"
    ]
    
    # Run the demo queries
    print("\n--- DEMO QUERIES ENGAGING DIFFERENT MEMORY TYPES ---\n")
    
    for i, query in enumerate(demo_queries, 1):
        print(f"\nDEMO QUERY {i}: {query}")
        print("Thinking...")
        response = assistant.process_query(query)
        print(f"\nRESPONSE:\n{response}\n")
        print("-" * 70)
    
    print("\nMemory types demonstration complete!")

if __name__ == "__main__":
    run_memory_demo() 
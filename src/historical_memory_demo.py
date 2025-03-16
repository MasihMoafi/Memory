import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.memory_model import OllamaMemoryAssistant

def run_historical_memory_demo():
    """Run a demonstration of the three memory types with historical figures"""
    print("Creating historical memory assistant with gemma3:12b...")
    assistant = OllamaMemoryAssistant("history_demo", model_name="gemma3:12b")
    
    # Setup memories
    print("\nInitializing memories...")
    
    # SEMANTIC MEMORY (facts about historical figures)
    print("Adding semantic memories (facts)...")
    
    assistant.learn_fact("albert_einstein", {
        "birth": "1879, Ulm, Germany",
        "death": "1955, Princeton, USA",
        "title": "Theoretical Physicist",
        "achievements": ["Theory of Relativity", "Photoelectric effect", "Mass-energy equivalence (E=mc²)"],
        "notable_works": ["Special Relativity", "General Relativity", "Quantum Theory contributions"],
        "quotes": ["Imagination is more important than knowledge.", 
                  "The important thing is not to stop questioning."],
        "scientific_approach": "Thought experiments, visual thinking, and mathematical formalism"
    })
    
    assistant.learn_fact("elon_musk", {
        "birth": "1971, South Africa",
        "companies": ["Tesla", "SpaceX", "Neuralink", "X (Twitter)"],
        "innovations": "Electric vehicles, reusable rockets, brain-machine interfaces",
        "notable_achievements": ["First private company to send humans to space", 
                                "Mainstream adoption of electric cars"],
        "quotes": ["When something is important enough, you do it even if the odds are not in your favor."],
        "leadership_style": "Visionary, hands-on, demanding, focused on first-principles thinking"
    })
    
    # PROCEDURAL MEMORY (how to analyze historical figures)
    print("Adding procedural memories (methods)...")
    
    assistant.learn_procedure("analyze_historical_figure", {
        "steps": [
            "1. Research their early life and background",
            "2. Examine key achievements and contributions",
            "3. Understand historical context they operated in",
            "4. Analyze their approach to problem-solving",
            "5. Evaluate their legacy and impact on future generations"
        ]
    })
    
    assistant.learn_procedure("compare_scientific_contributions", {
        "steps": [
            "1. Identify the key figures to compare",
            "2. Research their educational background and influences",
            "3. Analyze their major discoveries and methodologies",
            "4. Evaluate their impact on scientific understanding",
            "5. Compare how they communicated their ideas",
            "6. Assess their historical impact and legacy"
        ]
    })
    
    # EPISODIC MEMORY (past conversations/experiences)
    print("Adding episodic memories (past experiences)...")
    
    assistant.remember_interaction("research-session-2023-10-15", {
        "topic": "Comparing innovative thinkers",
        "figures_examined": ["Albert Einstein", "Elon Musk"],
        "insights": "Both figures challenged conventional thinking and used first principles reasoning",
        "questions_raised": [
            "How might Einstein have approached today's technological challenges?", 
            "What similarities exist between Einstein's theoretical approach and Musk's engineering approach?"
        ]
    })
    
    # Demonstrate with queries that engage different memory types
    demo_queries = [
        # Semantic memory query (factual)
        "When was Einstein born and what were his major achievements?",
        
        # Procedural memory query (how-to)
        "How should I analyze Elon Musk as a historical figure?",
        
        # Episodic memory query (past experiences)
        "What did we discuss in our last research session about innovative thinkers?",
        
        # Mixed memory query (uses multiple memory types)
        "Compare Einstein and Elon Musk's approaches to innovation based on our previous discussions and what we know about them."
    ]
    
    # Run the demo queries
    print("\n--- HISTORICAL FIGURES MEMORY DEMO ---\n")
    
    for i, query in enumerate(demo_queries, 1):
        print(f"\nQUERY {i}: {query}")
        print("Thinking...")
        response = assistant.process_query(query)
        print(f"\nRESPONSE:\n{response}\n")
        print("-" * 70)
    
    print("\nHistorical memory demonstration complete!")

if __name__ == "__main__":
    run_historical_memory_demo() 
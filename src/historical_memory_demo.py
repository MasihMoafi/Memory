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
    
    assistant.learn_fact("napoleon_bonaparte", {
        "birth": "1769, Corsica",
        "death": "1821, Saint Helena",
        "title": "Emperor of France",
        "achievements": ["Napoleonic Code", "Military conquests in Europe", "Civil administration reforms"],
        "notable_battles": ["Austerlitz", "Waterloo", "Battle of the Pyramids"],
        "quotes": ["Never interrupt your enemy when he is making a mistake.", 
                  "History is a set of lies agreed upon."],
        "leadership_style": "Authoritarian yet meritocratic, focused on efficiency and legal reform"
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
    
    # EPISODIC MEMORY (past conversations/experiences)
    print("Adding episodic memories (past experiences)...")
    
    assistant.remember_interaction("research-session-2023-10-15", {
        "topic": "Comparing leadership styles",
        "figures_examined": ["Napoleon Bonaparte", "Elon Musk"],
        "insights": "Both leaders leveraged cutting-edge technology of their era to expand influence",
        "questions_raised": [
            "How might Napoleon have used modern technology?", 
            "What similarities exist between Napoleon's military strategy and Musk's business approach?"
        ]
    })
    
    # Demonstrate with queries that engage different memory types
    demo_queries = [
        # Semantic memory query (factual)
        "When was Napoleon born and what were his major achievements?",
        
        # Procedural memory query (how-to)
        "How should I analyze Elon Musk as a historical figure?",
        
        # Episodic memory query (past experiences)
        "What did we discuss in our last research session about leadership styles?",
        
        # Mixed memory query (uses multiple memory types)
        "Compare Napoleon and Elon Musk's leadership styles based on our previous discussions and what we know about them."
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
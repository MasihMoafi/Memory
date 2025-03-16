import sys
from memory_model import OllamaMemoryAssistant

def create_historical_assistant():
    """Create and initialize a memory assistant with historical knowledge"""
    print("Creating historical figures memory assistant...")
    
    # Initialize assistant with the gemma3:12b model
    assistant = OllamaMemoryAssistant("history_researcher", model_name="gemma3:12b")
    
    # Add semantic memory (facts about historical figures)
    print("Adding knowledge about historical figures...")
    
    assistant.learn_fact("napoleon_bonaparte", {
        "birth": "1769, Corsica",
        "death": "1821, Saint Helena",
        "title": "Emperor of France",
        "achievements": ["Napoleonic Code", "Military conquests in Europe", "Civil administration reforms"],
        "notable_battles": ["Austerlitz", "Waterloo", "Battle of the Pyramids"],
        "quotes": ["Never interrupt your enemy when he is making a mistake.", 
                  "History is a set of lies agreed upon."],
        "leadership_style": "Authoritarian yet meritocratic, focused on efficiency and legal reform",
        "legacy": "Reformed French legal system, reshaped European politics and warfare"
    })
    
    assistant.learn_fact("elon_musk", {
        "birth": "1971, South Africa",
        "companies": ["Tesla", "SpaceX", "Neuralink", "X (Twitter)"],
        "innovations": "Electric vehicles, reusable rockets, brain-machine interfaces",
        "notable_achievements": ["First private company to send humans to space", 
                                "Mainstream adoption of electric cars"],
        "quotes": ["When something is important enough, you do it even if the odds are not in your favor.",
                  "I think it's very important to have a feedback loop, where you're constantly thinking about what you've done and how you could be doing it better."],
        "leadership_style": "Visionary, hands-on, demanding, focused on first-principles thinking",
        "background": "Born in South Africa, later moved to Canada and US"
    })
    
    assistant.learn_fact("joseph_stalin", {
        "birth": "1878, Georgia",
        "death": "1953, Moscow",
        "title": "General Secretary of the Communist Party of the Soviet Union",
        "rule_period": "1924-1953",
        "policies": ["Rapid industrialization", "Collectivization", "Great Purge"],
        "quotes": ["Death is the solution to all problems. No man - no problem.",
                  "Ideas are more powerful than guns. We would not let our enemies have guns, why should we let them have ideas?"],
        "leadership_style": "Totalitarian, paranoid, based on terror and personality cult",
        "impact": "Transformed USSR into industrial and military superpower at great human cost"
    })
    
    # Add procedural memory (how to analyze historical figures)
    print("Adding procedural knowledge...")
    
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
    
    # Add episodic memory (past conversations/experiences)
    print("Adding episodic memories...")
    
    assistant.remember_interaction("research-session-2023-10-15", {
        "topic": "Comparing leadership styles",
        "figures_examined": ["Napoleon Bonaparte", "Joseph Stalin", "Elon Musk"],
        "insights": "Autocratic vs. innovative leadership approaches across different eras",
        "questions_raised": [
            "How might Napoleon have used modern technology?", 
            "What similarities exist between Stalin's propaganda and modern social media?",
            "How do crisis management approaches differ across these leaders?"
        ],
        "conclusion": "All three leveraged cutting-edge technology of their era to expand influence"
    })
    
    assistant.remember_interaction("historical-analysis-2024-03-01", {
        "topic": "Technological innovation and social change",
        "focus": "How Elon Musk's approach compares to historical industrial revolutionaries",
        "comparisons": [
            "Musk vs. Ford: Mass production philosophy",
            "Musk vs. Edison: Marketing and public persona",
            "Musk vs. Napoleon: Institutional reform approach"
        ],
        "insights": "Modern tech leaders combine industrial revolution approaches with digital-age speed"
    })
    
    print("Historical memory assistant initialized and ready!\n")
    return assistant

def interactive_session(assistant):
    """Run an interactive session with the memory assistant"""
    print("Starting interactive session with memory assistant.")
    print("Type 'exit' to end the session.\n")
    
    while True:
        query = input("Your question: ")
        
        if query.lower() in ['exit', 'quit', 'bye']:
            print("Ending session. Goodbye!")
            break
            
        # Process the query
        print("\nThinking...")
        response = assistant.process_query(query)
        print(f"\nAssistant: {response}\n")

def main():
    # Create and initialize the assistant
    assistant = create_historical_assistant()
    
    # Run in interactive mode
    interactive_session(assistant)

if __name__ == "__main__":
    main() 
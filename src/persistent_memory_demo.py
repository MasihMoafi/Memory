import sys
import os
import json
from datetime import datetime
import time  # For clear demo steps

# Add parent directory to path for importing
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.memory_model import OllamaMemoryAssistant

def run_persistent_memory_demo():
    """Demonstrate how memory persists across sessions"""
    
    # Create a memory directory that will persist
    memory_dir = os.path.abspath("./persistent_memories")
    os.makedirs(memory_dir, exist_ok=True)
    
    print(f"Using persistent memory directory: {memory_dir}")
    
    # Create assistant with persistent memory
    print("\nCreating assistant with persistent memory...")
    assistant = OllamaMemoryAssistant(
        user_id="persistent_demo",
        model_name="gemma3:12b",  # Using a larger model for better responses
        memory_dir=memory_dir
    )
    
    # Check if memory file exists and has content
    memory_file = os.path.join(memory_dir, "persistent_demo_memory.json")
    first_run = not os.path.exists(memory_file) or os.path.getsize(memory_file) < 100
    
    if first_run:
        print("\nThis appears to be the first run - adding initial memories...")
        
        # Add some initial memories
        assistant.learn_fact("persistent_fact", {
            "name": "Memory Persistence Test",
            "created_at": str(os.path.getmtime(__file__)),
            "purpose": "To demonstrate that memories persist between sessions"
        })
        
        assistant.learn_procedure("memory_test_procedure", {
            "steps": [
                "1. Create a memory in the first session",
                "2. Exit the program",
                "3. Run the program again",
                "4. Verify that the memory still exists"
            ]
        })
        
        assistant.remember_interaction("first_session", {
            "event": "Initial program run",
            "action": "Created persistent memories",
            "note": "If you see this in a future session, persistence is working!"
        })
        
        print("\n✅ Initial memories created and saved to disk.")
        print("\n📝 Next steps:")
        print("  1. Run this script again")
        print("  2. The script will detect existing memories")
        print("  3. It will add a new memory and retrieve all memories")
    else:
        print("\n🎉 Previous memories detected! Memory persistence is working!")
        
        # Add a new memory in this session
        assistant.remember_interaction("subsequent_session", {
            "event": "Subsequent program run",
            "action": "Verified memory persistence",
            "note": f"Successfully loaded memories from {memory_dir}"
        })
        
        # Query all memories to verify they exist
        print("\nRetrieving all memories to verify persistence:")
        
        # Get facts
        semantic_results = assistant.memory.semantic.search_knowledge("")
        print("\n📚 FACTS:")
        for key, value in semantic_results:
            print(f"  - {key}: {value}")
        
        # Get procedures
        procedural_results = assistant.memory.procedural.search_procedures("")
        print("\n📋 PROCEDURES:")
        for key, value in procedural_results:
            print(f"  - {key}: {value}")
        
        # Get past interactions
        episodic_results = assistant.memory.episodic.search_interactions("")
        print("\n🕰️ PAST INTERACTIONS:")
        for i, memory in enumerate(episodic_results):
            print(f"  - Memory {i+1}: {memory}")
    
    print("\n✨ Persistent Memory Demo Complete ✨")
    if first_run:
        print("Run this script again to verify that memories persist!")

if __name__ == "__main__":
    run_persistent_memory_demo() 
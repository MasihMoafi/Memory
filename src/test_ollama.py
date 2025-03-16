import requests
import sys

def test_ollama_connection(base_url="http://localhost:11434"):
    """Test connection to Ollama server and list available models"""
    print(f"Testing connection to Ollama at {base_url}...")
    
    try:
        # Check if Ollama is running
        response = requests.get(f"{base_url}/api/tags", timeout=5)
        response.raise_for_status()
        
        # Get available models
        models = response.json().get("models", [])
        
        if models:
            print("\n✅ Ollama is running!")
            print("\nAvailable models:")
            for model in models:
                print(f"- {model['name']}")
            
            # Recommend a model for testing
            recommended = next((m for m in models if "llama" in m["name"].lower()), models[0])
            print(f"\nRecommended model for testing: {recommended['name']}")
            return recommended["name"]
        else:
            print("\n⚠️ Ollama is running but no models found. Please pull a model using:")
            print("ollama pull llama3:8b")
            return None
            
    except requests.exceptions.ConnectionError:
        print("\n❌ Could not connect to Ollama. Is it running?")
        print("To start Ollama, open a terminal and run: ollama serve")
        return None
    except requests.exceptions.Timeout:
        print("\n❌ Connection to Ollama timed out.")
        return None
    except requests.exceptions.HTTPError as e:
        print(f"\n❌ HTTP error: {e}")
        return None
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        return None

def test_ollama_inference(model_name, base_url="http://localhost:11434"):
    """Test if Ollama can generate a response with the selected model"""
    if not model_name:
        return False
    
    print(f"\nTesting inference with model '{model_name}'...")
    
    try:
        response = requests.post(
            f"{base_url}/api/generate",
            json={
                "model": model_name,
                "prompt": "Hello, how are you today?",
                "stream": False
            },
            timeout=30
        )
        
        response.raise_for_status()
        result = response.json()
        
        if "response" in result:
            print("\n✅ Model inference successful!")
            print(f"\nModel response: {result['response'][:100]}...")
            return True
        else:
            print("\n❌ Model returned an invalid response format.")
            return False
            
    except Exception as e:
        print(f"\n❌ Error during inference: {e}")
        return False

if __name__ == "__main__":
    model = test_ollama_connection()
    if model:
        test_ollama_inference(model) 
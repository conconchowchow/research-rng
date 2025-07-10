from abc import ABC, abstractmethod
from typing import List, Dict, Any
import os
from dotenv import load_dotenv
import google.generativeai as genai

load_dotenv()

class ModelProvider(ABC):
    """Abstract base class for LLM model providers."""
    
    @abstractmethod
    def generate_response(self, prompt: str, model_name: str = None) -> str:
        """Generate a response for the given prompt."""
        pass
    
    @abstractmethod
    def get_available_models(self) -> List[str]:
        """Get list of available model names."""
        pass

class GeminiProvider(ModelProvider):
    """Google Gemini model provider."""
    
    def __init__(self, api_key: str = None):
        """Initialize Gemini provider with API key."""
        if api_key is None:
            api_key = os.getenv('GOOGLE_API_KEY')
            if api_key is None:
                raise ValueError("Google API key not provided. Please set GOOGLE_API_KEY environment variable or pass api_key parameter.")
        
        genai.configure(api_key=api_key)
        self.available_models = [
            "gemini-2.5-flash",
            "gemini-2.5-flash-lite-preview-06-17",
            "gemini-2.0-flash",
        ]
    
    def generate_response(self, prompt: str, model_name: str = None) -> str:
        """Generate a response using Gemini model."""
        if model_name is None:
            model_name = self.available_models[0]  # Default to first model
        
        try:
            model = genai.GenerativeModel(model_name)
            response = model.generate_content(prompt)
            return response.text
        except Exception as e:
            print(f"Error generating response with {model_name}: {e}")
            return f"Error: Could not generate response with {model_name}"
    
    def get_available_models(self) -> List[str]:
        """Get list of available Gemini models."""
        return self.available_models.copy()

class MockProvider(ModelProvider):
    """Mock model provider for testing without API calls."""
    
    def __init__(self):
        """Initialize mock provider with predefined responses."""
        self.responses = {
            "architecture": [
                "Gothic architecture is characterized by pointed arches, ribbed vaults, and flying buttresses. These elements create soaring vertical spaces and allow for large windows with intricate tracery.",
                "Baroque architecture features dramatic use of light and shadow, ornate decoration, and curved forms. It emphasizes grandeur and emotional intensity through elaborate details.",
                "Modernist architecture embraces clean lines, minimal ornamentation, and functional design. It often uses industrial materials like steel, glass, and concrete.",
                "Art Deco architecture combines geometric patterns with luxurious materials. It features bold angular forms, stylized decorative elements, and rich colors."
            ],
            "constitution": [
                "The United States Constitution establishes the framework of federal government with three branches: legislative, executive, and judicial. It defines the separation of powers and system of checks and balances.",
                "The Bill of Rights comprises the first ten amendments to the Constitution, protecting fundamental freedoms like speech, religion, and due process of law.",
                "The Constitution can be amended through a specific process requiring broad consensus, demonstrating both its stability and ability to evolve with changing times."
            ]
        }
        self.available_models = ["mock-model-1", "mock-model-2", "mock-model-3"]
    
    def generate_response(self, prompt: str, model_name: str = None) -> str:
        """Generate a mock response based on prompt content."""
        if model_name is None:
            model_name = self.available_models[0]
        
        prompt_lower = prompt.lower()
        
        if "architecture" in prompt_lower:
            import random
            return random.choice(self.responses["architecture"])
        elif "constitution" in prompt_lower:
            import random
            return random.choice(self.responses["constitution"])
        else:
            return f"This is a generic response from {model_name} for the prompt: {prompt[:50]}..."
    
    def get_available_models(self) -> List[str]:
        """Get list of available mock models."""
        return self.available_models.copy() 